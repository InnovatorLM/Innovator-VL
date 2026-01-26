"""
Adapter to integrate judge model rewards into existing RL training pipeline.
Maintains the same interface as the original RewardSystem for compatibility.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Union

from .async_judge_reward import AsyncJudgeRewardPool, RewardRequest, RewardResult
from .judge_model_reward import BaseJudgeReward, HttpJudgeReward, JudgeRequest, JudgeResponse
from .reward_system import RewardSystem as RuleBasedRewardSystem

logger = logging.getLogger("JudgeRewardAdapter")


class JudgeRewardSystem:
    """
    Drop-in replacement for RuleBasedRewardSystem that uses external judge model.

    Features:
    - Async execution with synchronous interface (for ray compatibility)
    - Automatic fallback to rule-based rewards on failures
    - Configurable judge model settings via YAML config
    - Thread-safe operations
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_judge: bool = True,
        use_async: bool = True
    ):
        """
        Args:
            config: Judge model configuration dict
            use_judge: Whether to use judge model (False = fallback to rules)
            use_async: Whether to use async implementation (recommended)
        """
        self.use_judge = use_judge
        self.config = config or self._get_default_config()

        if use_judge:
            if use_async:
                # Use async implementation with thread pool
                self._executor = ThreadPoolExecutor(max_workers=4)
                self._event_loop = None
                self._loop_thread = None
                self._init_async_loop()
                self._reward_pool = AsyncJudgeRewardPool(self.config)
                self._initialized = False
            else:
                # Use sync implementation
                self._judge_reward = HttpJudgeReward(self.config)

        # Fallback reward system
        self._rule_reward = RuleBasedRewardSystem()

        logger.info(f"JudgeRewardSystem initialized: use_judge={use_judge}, use_async={use_async}")

    def _get_default_config(self) -> Dict:
        """Get default configuration from environment or return defaults."""
        return {
            'judge_url': os.environ.get('JUDGE_MODEL_URL', 'http://localhost:8888'),
            'api_key': os.environ.get('JUDGE_MODEL_API_KEY'),
            'use_cache': True,
            'cache_size': 5000,
            'cache_ttl': 3600,
            'fallback_to_rule': True,
            'max_batch_size': os.environ.get('JUDGE_MAX_BATCH_SIZE', 16),
            'batch_timeout': float(os.environ.get('JUDGE_BATCH_TIMEOUT', 0.05)),
            'max_concurrent_requests': os.environ.get('JUDGE_MAX_CONCURRENT', 32),
            'timeout': float(os.environ.get('JUDGE_TIMEOUT', 30.0)),
        }

    def _init_async_loop(self):
        """Initialize background thread with event loop for async operations."""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready
        while self._event_loop is None:
            time.sleep(0.01)

    def start(self):
        """Start the reward system (required for async version)."""
        if hasattr(self, '_reward_pool') and not self._initialized:
            future = asyncio.run_coroutine_threadsafe(
                self._reward_pool.start(),
                self._event_loop
            )
            try:
                future.result(timeout=10)
                self._initialized = True
                logger.info("Async judge reward pool started")
            except Exception as e:
                logger.error(f"Failed to start async reward pool: {e}")
                self.use_judge = False  # Fall back to rules

    def reward(
        self,
        prompt: str,
        completions: str,
        answer: Union[str, list[str]],
        answer_type=None,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute rewards maintaining the same interface as RuleBasedRewardSystem.

        Returns:
            Dict containing:
            - reward: Main reward score
            - acc_reward: Accuracy reward (backward compatibility)
            - bbox_reward: BBox reward if applicable
            - explanation: Optional explanation from judge
            - source: 'judge' or 'rule'
        """
        if not self.use_judge:
            # Direct fallback to rules
            return self._rule_reward.reward(
                prompt, completions, answer, answer_type, *args, **kwargs
            )

        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]

        try:
            if hasattr(self, '_reward_pool'):
                # Async implementation
                return self._compute_reward_async(
                    prompt, completions, answer, answer_type, **kwargs
                )
            else:
                # Sync implementation
                return self._compute_reward_sync(
                    prompt, completions, answer, answer_type, **kwargs
                )
        except Exception as e:
            logger.error(f"Judge model failed: {e}, falling back to rules")

            if self.config.get('fallback_to_rule', True):
                rule_result = self._rule_reward.reward(
                    prompt, completions, answer, answer_type, *args, **kwargs
                )
                rule_result['source'] = 'rule_fallback'
                rule_result['explanation'] = f'Judge failed: {str(e)}'
                return rule_result
            else:
                raise e

    def _compute_reward_sync(
        self,
        prompt: str,
        completions: str,
        answer: str,
        answer_type=None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute reward using sync judge model."""
        request = JudgeRequest(
            prompt=prompt,
            completions=completions,
            answer=answer,
            answer_type=answer_type,
            metadata=kwargs
        )

        response = self._judge_reward.compute_reward(request)

        # Handle bbox reward for spatial reasoning
        bbox_reward = 0.0
        if kwargs.get('problem_type') and str(kwargs['problem_type']) == 'ProblemType.SPATIAL_REASONING':
            from .format import must_have_bbox_reward_fn
            bbox_reward = must_have_bbox_reward_fn(completions)

        return {
            'reward': response.score,
            'acc_reward': response.score,  # backward compatibility
            'bbox_reward': bbox_reward * 0.5 if bbox_reward > 0 else 0.0,
            'explanation': response.explanation,
            'source': 'judge',
            'latency': response.latency,
        }

    def _compute_reward_async(
        self,
        prompt: str,
        completions: str,
        answer: str,
        answer_type=None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute reward using async judge model."""
        # Create async request
        request_id = f"{threading.current_thread().ident}_{time.time()}"

        future = asyncio.run_coroutine_threadsafe(
            self._reward_pool.compute_reward(
                prompt=prompt,
                completion=completions,
                answer=answer,
                answer_type=answer_type,
                request_id=request_id
            ),
            self._event_loop
        )

        # Wait for result with timeout
        try:
            result: RewardResult = future.result(timeout=self.config.get('timeout', 30.0))
        except Exception as e:
            logger.error(f"Async judge request failed: {e}")
            raise e

        # Handle bbox reward
        bbox_reward = 0.0
        if kwargs.get('problem_type') and str(kwargs['problem_type']) == 'ProblemType.SPATIAL_REASONING':
            from .format import must_have_bbox_reward_fn
            bbox_reward = must_have_bbox_reward_fn(completions)

        return {
            'reward': result.score,
            'acc_reward': result.score,
            'bbox_reward': bbox_reward * 0.5 if bbox_reward > 0 else 0.0,
            'explanation': result.explanation,
            'source': 'judge',
            'cached': result.cached,
            'latency': result.latency,
        }

    def get_metrics(self) -> Dict[str, any]:
        """Get performance metrics."""
        metrics = {
            'use_judge': self.use_judge,
            'initialized': hasattr(self, '_initialized') and self._initialized,
        }

        if hasattr(self, '_reward_pool'):
            try:
                pool_metrics = asyncio.run_coroutine_threadsafe(
                    self._reward_pool.get_metrics(),
                    self._event_loop
                ).result(timeout=1.0)
                metrics.update(pool_metrics)
            except:
                pass

        return metrics

    def close(self):
        """Cleanup resources."""
        if hasattr(self, '_reward_pool') and self._initialized:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._reward_pool.stop(),
                    self._event_loop
                )
                future.result(timeout=2.0)
            except:
                pass

        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True, timeout=5)


def create_judge_reward(config: Dict, use_judge: bool = True) -> JudgeRewardSystem:
    """Factory function to create judge reward system."""
    return JudgeRewardSystem(config, use_judge=use_judge)


# Backward compatibility alias
RewardSystem = JudgeRewardSystem


# Configuration examples for YAML
YAML_CONFIG_EXAMPLE = """
# In your training config YAML file:

reward:
  type: judge_model  # or 'rule_based' for fallback
  config:
    judge_url: http://localhost:8888
    api_key: ${JUDGE_MODEL_API_KEY}  # Optional API key
    use_cache: true
    cache_size: 5000
    cache_ttl: 3600
    fallback_to_rule: true  # Fallback to rules if judge fails

    # Performance tuning
    max_batch_size: 16  # Batch requests for efficiency
    batch_timeout: 0.05  # 50ms to form a batch
    max_concurrent_requests: 32  # Max inflight requests
    timeout: 30.0  # Request timeout

    # Circuit breaker
    circuit_breaker_threshold: 10
    circuit_breaker_timeout: 60
""


if __name__ == "__main__":
    # Test the adapter
    import time

    # Create with mock config
    config = {
        'judge_url': 'http://localhost:8888',
        'use_cache': True,
        'fallback_to_rule': True,
        'max_batch_size': 8,
        'batch_timeout': 0.05
    }

    reward_system = JudgeRewardSystem(config, use_judge=True)

    # Start async components
    reward_system.start()

    # Test requests
    test_cases = [
        ("What is 2+2?", "The answer is **boxed{4}**", "4"),
        ("What is the capital of France?", "The capital is **boxed{Paris}**", "Paris"),
    ]

    for prompt, completion, answer in test_cases:
        result = reward_system.reward(prompt, completion, answer)
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Reward: {result['reward']}")
        print(f"Source: {result.get('source', 'unknown')}")
        print()

    # Print metrics
    print("Metrics:", reward_system.get_metrics())

    # Cleanup
    reward_system.close()
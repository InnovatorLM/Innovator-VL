"""
适配器：将增强版Judge模型集成到现有训练系统
保持与原有RewardSystem完全相同的接口
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Union

from .enhanced_judge import EnhancedJudgeModel, create_enhanced_judge, ScoringConfig
from .reward_system import RewardSystem as BaseRewardSystem


class EnhancedJudgeRewardSystem(BaseRewardSystem):
    """
    Drop-in replacement for RewardSystem with enhanced scoring
    -
    Features:
    - Multi-layer validation (format, thinking, answer correctness)
    - Configurable weights based on prompt type
    - Async execution with sync interface
    - Backward compatibility with existing code
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_enhanced: bool = True,
        async_pool_size: int = 4
    ):
        """
        Args:
            config: Enhanced judge model config
            use_enhanced: Whether to use enhanced scoring (False = fallback to rules)
            async_pool_size: Size of worker pool for async execution
        """
        super().__init__()

        self.use_enhanced = use_enhanced
        self.judge_config = config or {}
        self.async_pool_size = async_pool_size

        if use_enhanced:
            # Initialize async components
            self._executor = ThreadPoolExecutor(max_workers=async_pool_size)
            self._loop_thread = None
            self._event_loop = None
            self._judge_model = None
            self._initialized = False

            # Start async event loop in background thread
            self._init_async_loop()
        else:
            # Fallback to rule-based system
            pass

    def _init_async_loop(self):
        """Initialize background event loop for async operations"""
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop
            loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be ready
        while self._event_loop is None:
            pass

    async def _get_judge_model(self) -> EnhancedJudgeModel:
        """Get or create judge model instance"""
        if not self._judge_model:
            self._judge_model = await create_enhanced_judge(self.judge_config)
        return self._judge_model

    def start(self):
        """Start the judge system (call before training)"""
        if self.use_enhanced and not self._initialized:
            try:
                # Ensure judge model is initialized
                future = asyncio.run_coroutine_threadsafe(
                    self._get_judge_model(),
                    self._event_loop
                )
                self._judge_model = future.result(timeout=30)
                self._initialized = True
                print("Enhanced judge system started successfully")
            except Exception as e:
                print(f"Failed to start enhanced judge system: {e}")
                print("Falling back to rule-based system")
                self.use_enhanced = False

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
        评估模型回复的奖励
        保持与原有RewardSystem完全相同的接口
        """
        if not self.use_enhanced:
            # Fallback to rule-based system
            rule_result = super().reward(prompt, completions, answer, answer_type, *args, **kwargs)
            rule_result["source"] = "rule_based"
            return rule_result

        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]

        try:
            # 确定prompt类型（需要传递这个信息）
            prompt_type = kwargs.get("prompt_type", "normal")

            # 异步执行判断
            future = asyncio.run_coroutine_threadsafe(
                self._async_judge_single(
                    prompt=prompt,
                    completion=completions,
                    answer=answer,
                    answer_type=answer_type,
                    prompt_type=prompt_type,
                    **kwargs
                ),
                self._event_loop
            )
            result = future.result(timeout=30.0)
            return self._convert_to_standard_format(result)

        except Exception as e:
            print(f"Enhanced judge failed: {e}, falling back to rules")
            rule_result = super().reward(prompt, completions, answer, answer_type, *args, **kwargs)
            rule_result["source"] = "rule_fallback"
            rule_result["error"] = str(e)
            return rule_result

    async def _async_judge_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        answer_type: str,
        prompt_type: str,
        **kwargs
    ) -> Dict[str, float]:
        """异步执行单个评判任务"""
        judge_model = await self._get_judge_model()

        return await judge_model.judge_single(
            prompt=prompt,
            completion=completion,
            answer=answer,
            prompt_type=prompt_type,
            answer_type=answer_type,
            **kwargs
        )

    def _convert_to_standard_format(self, enhanced_result: Dict) -> Dict[str, float]:
        """将增强版结果转换为标准格式以兼容现有代码"""
        return {
            "acc_reward": enhanced_result["answer_score"],     # 答案正确性（原有字段）
            "thinking_reward": enhanced_result["thinking_score"], # 思考质量（新增字段）
            "reward": enhanced_result["score"],                # 总分
            "format_score": enhanced_result["format_score"],   # 格式分
            "bbox_reward": enhanced_result.get("bbox_reward", 0.0), # 空间推理任务用
            "explanation": enhanced_result["explanation"],     # 解释信息
            "source": "enhanced_judge",                         # 来源标识
            "validation_method": enhanced_result.get("validation_method", "unknown"),
            "prompt_type": enhanced_result.get("prompt_type", "unknown"),
            "confidence": enhanced_result.get("confidence", 0.0)
        }

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        if not self.use_enhanced:
            return {"mode": "rule_based"}

        return {
            "mode": "enhanced",
            "initialized": self._initialized,
            "pool_size": self.async_pool_size,
        }

    def close(self):
        """清理资源"""
        if self.use_enhanced and self._initialized:
            try:
                if self._judge_model:
                    # 可以在这里添加清理逻辑
                    pass
            except:
                pass

            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True, timeout=5)


# 工厂函数
def create_enhanced_judge_reward(config: Optional[Dict] = None, use_enhanced: bool = True) -> EnhancedJudgeRewardSystem:
    """创建增强版奖励系统"""
    return EnhancedJudgeRewardSystem(config, use_enhanced)


# 与配置集成的示例
ENHANCED_JUDGE_CONFIG = {
    "scoring_weights": {
        "thinking_prompt": {
            "format": 0.10,
            "thinking": 0.30,
            "answer": 0.60
        },
        "normal_prompt": {
            "format": 0.05,
            "thinking": 0.15,
            "answer": 0.80
        }
    },
    "async_pool_size": 4,
    "timeout": 30.0,
    "validation_layers": ["exact_match", "math_verify", "semantic"],
    "min_thinking_length": 10
}


# 集成示例
if __name__ == "__main__":
    import asyncio
    import time

    # 创建增强版奖励系统
    reward_system = create_enhanced_judge_reward(ENHANCED_JUDGE_CONFIG)
    reward_system.start()

    # 测试用例
    test_cases = [
        {
            "prompt": "计算 2+2",
            "completion": """
<think>
这是一个简单的加法问题。
我有两个数字：2 和 2。
将它们相加：2 + 2 = 4。
</think>
<answer>4</answer>
            """,
            "answer": "4",
            "prompt_type": "thinking",
            "answer_type": "NUMBER"
        },
        {
            "prompt": "What is 3*3?",
            "completion": "<answer>9</answer>",
            "answer": "9",
            "prompt_type": "normal",
            "answer_type": "NUMBER"
        }
    ]

    print("Testing enhanced judge system...")
    for i, test_case in enumerate(test_cases):
        result = reward_system.reward(**test_case)
        print(f"\nTest {i+1} ({test_case['prompt_type']}):")
        print(f"Total reward: {result['reward']:.3f}")
        print(f"Answer reward: {result['acc_reward']:.3f}")
        print(f"Thinking reward: {result.get('thinking_reward', 0.0):.3f}")
        print(f"Source: {result['source']}")

    print(f"\nMetrics: {reward_system.get_metrics()}")
    reward_system.close()
"""
Judge model based reward system for external reward model.
Supports both sync and async calling with batching and caching.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("JudgeModelReward")


@dataclass
class JudgeRequest:
    """Request sent to judge model."""
    prompt: str
    completion: str
    answer: Optional[str] = None
    answer_type: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class JudgeResponse:
    """Response from judge model."""
    score: float  # Main reward score [0, 1]
    explanation: Optional[str] = None  # Optional reasoning
    metadata: Optional[Dict] = None
    latency: float = 0.0  # Response latency


class RewardCache:
    """Simple in-memory cache for rewards to avoid duplicate calls."""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
        self.ttl = ttl
        logger.info(f"RewardCache initialized with max_size={max_size}, ttl={ttl}s")

    def _get_key(self, request: JudgeRequest) -> str:
        """Generate cache key from request."""
        content = f"{request.prompt}||{request.completion}||{request.answer}||{request.answer_type}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, request: JudgeRequest) -> Optional[JudgeResponse]:
        """Get cached reward if exists and not expired."""
        key = self._get_key(request)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return response
            else:
                del self.cache[key]
        return None

    def set(self, request: JudgeRequest, response: JudgeResponse):
        """Cache reward response."""
        key = self._get_key(request)
        self.cache[key] = (response, time.time())

        # Simple LRU eviction
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]


class BaseJudgeReward(ABC):
    """Abstract base class for judge-based reward computation."""

    def __init__(self, config: Dict):
        self.config = config
        self.cache_enabled = config.get('use_cache', True)
        self.cache = RewardCache(
            max_size=config.get('cache_size', 10000),
            ttl=config.get('cache_ttl', 3600)
        ) if self.cache_enabled else None
        self.fallback_to_rule = config.get('fallback_to_rule', True)

    @abstractmethod
    def _request_judge(self, request: JudgeRequest) -> JudgeResponse:
        """Send single request to judge model."""
        pass

    @abstractmethod
    async def _request_judge_async(self, request: JudgeRequest) -> JudgeResponse:
        """Send single async request to judge model."""
        pass

    def compute_reward(self, request: JudgeRequest) -> JudgeResponse:
        """Compute reward with caching."""
        # Check cache first
        if self.cache_enabled:
            cached = self.cache.get(request)
            if cached is not None:
                return cached

        try:
            # Try judge model
            response = self._request_judge(request)

            # Cache result
            if self.cache_enabled:
                self.cache.set(request, response)

            return response

        except Exception as e:
            logger.error(f"Judge model failed: {e}")

            if self.fallback_to_rule:
                logger.info("Falling back to rule-based reward")
                from .reward_system import RewardSystem
                rule_reward = RewardSystem()
                rule_result = rule_reward.reward(
                    prompt=request.prompt,
                    completions=request.completion,
                    answer=request.answer,
                    answer_type=request.answer_type
                )
                return JudgeResponse(
                    score=rule_result.get("reward", 0.0),
                    explanation=f"Fallback to rule-based reward: {e}",
                    metadata={"source": "rule_based", "error": str(e)}
                )
            else:
                raise e


class HttpJudgeReward(BaseJudgeReward):
    """HTTP-based judge reward system."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config['judge_url']
        self.api_key = config.get('api_key', None)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)

        # For sync calls
        self.session = requests.Session()
        if self.api_key:
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _request_judge(self, request: JudgeRequest) -> JudgeResponse:
        """Send HTTP request to judge model."""
        start_time = time.time()

        url = urljoin(self.base_url, '/judge')
        payload = {
            'prompt': request.prompt,
            'completion': request.completion,
            'answer': request.answer,
            'answer_type': request.answer_type,
            'metadata': request.metadata
        }

        response = self.session.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        latency = time.time() - start_time

        return JudgeResponse(
            score=result.get('score', 0.0),
            explanation=result.get('explanation'),
            metadata=result.get('metadata'),
            latency=latency
        )

    async def _request_judge_async(self, request: JudgeRequest) -> JudgeResponse:
        """Send async HTTP request to judge model."""
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        url = urljoin(self.base_url, '/judge')
        payload = {
            'prompt': request.prompt,
            'completion': request.completion,
            'answer': request.answer,
            'answer_type': request.answer_type,
            'metadata': request.metadata
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()

        latency = time.time() - start_time

        return JudgeResponse(
            score=result.get('score', 0.0),
            explanation=result.get('explanation'),
            metadata=result.get('metadata'),
            latency=latency
        )


class BatchJudgeReward:
    """Batch processing wrapper for judge reward."""

    def __init__(self, judge_reward: BaseJudgeReward, batch_size: int = 32):
        self.judge = judge_reward
        self.batch_size = batch_size
        self.queue: List[JudgeRequest] = []
        self.results: Dict[int, JudgeResponse] = {}

    async def add_request(self, request: JudgeRequest) -> JudgeResponse:
        """Add request and wait for batch processing."""
        request_id = len(self.queue)
        self.queue.append(request)

        # Process batch when full
        if len(self.queue) >= self.batch_size:
            await self._process_batch()

        # Wait for result
        while request_id not in self.results:
            await asyncio.sleep(0.01)

        return self.results.pop(request_id)

    async def _process_batch(self):
        """Process current batch of requests."""
        if not self.queue:
            return

        batch = self.queue.copy()
        self.queue.clear()

        # Process batch concurrently
        tasks = [
            self.judge._request_judge_async(req)
            for req in batch
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Store results
        for i, (req, resp) in enumerate(zip(batch, responses)):
            if isinstance(resp, Exception):
                logger.error(f"Batch request failed: {resp}")
                resp = await self.judge.compute_reward_async(req)
            self.results[len(batch) - len(batch) + i] = resp

    async def compute_reward_async(self, request: JudgeRequest) -> JudgeResponse:
        """Async compute reward."""
        if hasattr(self.judge, 'compute_reward_async'):
            return await self.judge.compute_reward_async(request)
        else:
            # Fallback to sync in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.judge.compute_reward, request
            )


def create_judge_reward(config: Dict) -> HttpJudgeReward:
    """Factory function to create judge reward system."""
    return HttpJudgeReward(config)


# Example usage
if __name__ == "__main__":
    config = {
        'judge_url': 'http://localhost:8888',
        'api_key': 'your-api-key',
        'use_cache': True,
        'cache_size': 5000,
        'cache_ttl': 3600,
        'fallback_to_rule': True,
        'timeout': 30,
        'max_retries': 3
    }

    judge = create_judge_reward(config)
    request = JudgeRequest(
        prompt="What is 2+2?",
        completion="The answer is **boxed{4}**",
        answer="4",
        answer_type="NUMBER"
    )

    response = judge.compute_reward(request)
    print(f"Score: {response.score}, Explanation: {response.explanation}")
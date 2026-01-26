"""
Async judge model reward system optimized for RL training.
Supports concurrent requests with automatic batching and retry logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import aiohttp
import json

logger = logging.getLogger("AsyncJudgeReward")


@dataclass(frozen=True)
class RewardRequest:
    """Immutable reward request with unique ID."""
    prompt: str
    completion: str
    answer: Optional[str] = None
    answer_type: Optional[str] = None
    request_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RewardResult:
    """Reward computation result."""
    score: float
    request_id: str
    latency: float
    cached: bool = False
    explanation: Optional[str] = None
    json_response: Optional[Dict] = field(default_factory=dict)




class AsyncJudgeRewardPool:
    """
    High-performance async judge model reward system.

    Features:
    - Concurrent request processing
    - Dynamic batching
    - Circuit breaker pattern
    - Request deduplication
    - Connection pooling
    - Adaptive timeout
    """

    def __init__(self, config: Dict):
        self.config = config

        # Judge model config
        self.judge_url = config['judge_url']
        self.api_key = config.get('api_key', None)
        self.timeout = config.get('timeout', 30.0)
        self.max_batch_size = config.get('max_batch_size', 32)
        self.batch_timeout = config.get('batch_timeout', 0.1)  # 100ms to form a batch
        self.max_concurrent_requests = config.get('max_concurrent_requests', 64)

        # Connection pool
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.session: Optional[aiohttp.ClientSession] = None

        # Request deduplication (avoid duplicate requests in flight)
        self.inflight_requests: Dict[str, asyncio.Future] = {}

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'batch_requests': 0,
            'avg_latency': 0.0,
            'errors': 0,
        }

        # Background tasks
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 10),
            recovery_timeout=config.get('circuit_breaker_timeout', 60),
            expected_exception=aiohttp.ClientError
        )

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self):
        """Start the async reward pool."""
        if self._running:
            return

        self._running = True
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            keepalive_timeout=30,
        )

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        self.session = aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )

        # Start batch processor
        self._batch_task = asyncio.create_task(self._batch_processor())
        logger.info("AsyncJudgeRewardPool started")

    async def stop(self):
        """Stop the async reward pool."""
        if not self._running:
            return

        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        if self.session:
            await self.session.close()

        logger.info("AsyncJudgeRewardPool stopped")

    async def compute_reward(
        self,
        prompt: str,
        completion: str,
        answer: Optional[str] = None,
        answer_type: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> RewardResult:
        """
        Compute reward for a single prompt-completion pair.

        Returns immediately with a future that will contain the result.
        """
        if request_id is None:
            request_id = f"req_{self.metrics['total_requests']}_{time.time()}"

        request = RewardRequest(
            prompt=prompt,
            completion=completion,
            answer=answer,
            answer_type=answer_type,
            request_id=request_id
        )

        # Check for in-flight deduplication
        key = self._request_key(request)
        if key in self.inflight_requests:
            logger.debug(f"Deduplicating in-flight request: {request_id}")
            self.metrics['cache_hits'] += 1
            return await self.inflight_requests[key]

        # Create future for this request
        future = asyncio.Future()
        self.inflight_requests[key] = future

        # Queue the request
        try:
            self._request_queue.put_nowait(request)
        except asyncio.QueueFull:
            future.set_exception(asyncio.QueueFull("Request queue is full"))
            self.inflight_requests.pop(key, None)
            raise

        self.metrics['total_requests'] += 1

        # Wait for result
        try:
            result = await future
            return result
        finally:
            self.inflight_requests.pop(key, None)

    async def compute_rewards_batch(
        self,
        requests: List[Tuple[str, str, Optional[str], Optional[str]]]
    ) -> List[RewardResult]:
        """
        Compute rewards for multiple requests concurrently.

        Args:
            requests: List of (prompt, completion, answer, answer_type) tuples

        Returns:
            List of RewardResult objects in same order as input
        """
        # Create tasks for all requests
        tasks = [
            asyncio.create_task(self.compute_reward(prompt, completion, answer, answer_type))
            for prompt, completion, answer, answer_type in requests
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                # Return zero score for failed requests
                final_results.append(RewardResult(
                    score=0.0,
                    request_id=f"failed_{i}",
                    latency=0.0,
                    explanation=f"Request failed: {str(result)}"
                ))
            else:
                final_results.append(result)

        return final_results

    def _request_key(self, request: RewardRequest) -> str:
        """Generate deduplication key for a request."""
        return f"{request.prompt}||{request.completion}||{request.answer}||{request.answer_type}"

    async def _batch_processor(self):
        """
        Background task that processes requests in batches.
        """
        logger.info("Batch processor started")

        while self._running:
            batch = []
            futures = []

            # Collect batch of requests
            try:
                # Wait for first request
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0  # Check every second if no requests
                )
                batch.append(request)
                futures.append(self.inflight_requests.get(self._request_key(request), None))

                # Collect more requests within batch_timeout or until max_batch_size
                end_time = asyncio.get_event_loop().time() + self.batch_timeout
                while len(batch) < self.max_batch_size and asyncio.get_event_loop().time() < end_time:
                    try:
                        request = await asyncio.wait_for(
                            self._request_queue.get(),
                            timeout=0.01  # Short timeout to keep collecting
                        )
                        key = self._request_key(request)
                        future = self.inflight_requests.get(key)
                        if future:
                            batch.append(request)
                            futures.append(future)
                    except asyncio.TimeoutError:
                        break

            except asyncio.TimeoutError:
                continue  # No requests in queue
            except Exception as e:
                logger.error(f"Error collecting batch: {e}")
                continue

            if not batch:
                continue

            # Process the batch
            try:
                self.metrics['batch_requests'] += 1
                logger.debug(f"Processing batch of {len(batch)} requests")

                # Send batch to judge model
                responses = await self._send_batch_to_judge(batch)

                # Set results
                for future, response in zip(futures, responses):
                    if future and not future.done():
                        future.set_result(response)

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.metrics['errors'] += 1

                # Set exceptions for all pending futures
                for future in futures:
                    if future and not future.done():
                        future.set_exception(e)

    async def _send_batch_to_judge(
        self,
        batch: List[RewardResult],
    ) -> List[RewardResponse]:
        """Send a batch of requests to the judge model."""

        async with self.semaphore:
            # Use circuit breaker
            return await self._circuit_breaker.call_async(
                self._judge_api_call,
                batch
            )

    async def _judge_api_call(
        self,
        batch: List[RewardRequest]
    ) -> List[JudgeResponse]:
        """Actual HTTP call to judge model API."""

        batch_payload = {
            'requests': [
                {
                    'prompt': req.prompt,
                    'completion': req.completion,
                    'answer': req.answer,
                    'answer_type': req.answer_type,
                    'request_id': req.request_id
                }
                for req in batch
            ]
        }

        start_time = time.time()

        async with self.session.post(
            f'{self.judge_url}/batch_judge',
            json=batch_payload
        ) as response:
            response.raise_for_status()
            result = await response.json()

            total_latency = time.time() - start_time

            # Convert to our format
            responses = []
            for item in result['responses']:
                responses.append(RewardResult(
                    score=item['score'],
                    request_id=item['request_id'],
                    latency=total_latency / len(batch),  # Approximate per-item latency
                    explanation=item.get('explanation'),
                    cached=False,
                    json_response=item
                ))

        return responses

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        total = self.metrics['total_requests']
        hit_rate = self.metrics['cache_hits'] / max(total, 1)
        batch_rate = self.metrics['batch_requests'] / max(total, 1)

        return {
            'total_requests': total,
            'cache_hit_rate': hit_rate,
            'batch_processing_rate': batch_rate,
            'current_queue_size': self._request_queue.qsize(),
            'in_flight_requests': len(self.inflight_requests),
            'errors': self.metrics['errors'],
            'circuit_breaker': self._circuit_breaker.get_state(),
        }


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception=aiohttp.ClientError):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()

    async def call_async(self, func, *args, **kwargs):
        """Call function with circuit breaker protection."""
        async with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = await func(*args, **kwargs)

                if self._state == "HALF_OPEN":
                    self._state = "CLOSED"
                    self._failure_count = 0

                return result

            except self.expected_exception as e:
                self._failure_count += 1
                self._last_failure_time = time.time()

                if self._failure_count >= self.failure_threshold:
                    self._state = "OPEN"
                    logger.error(f"Circuit breaker opened after {self._failure_count} failures")

                raise e

    def get_state(self):
        """Get current circuit breaker state."""
        return {
            'state': self._state,
            'failure_count': self._failure_count,
            'last_failure_time': self._last_failure_time,
        }


# Example usage and testing
async def example_usage():
    """Example usage of async judge reward pool."""
    import random

    config = {
        'judge_url': 'http://localhost:8888',
        'api_key': None,
        'max_batch_size': 16,
        'batch_timeout': 0.05,
        'max_concurrent_requests': 32,
        'timeout': 10.0
    }

    async with AsyncJudgeRewardPool(config) as pool:
        # Generate test requests
        requests = [
            (f"What is {i}+{i}?", f"The answer is **boxed{2*i}**", str(2*i), "NUMBER")
            for i in range(20)
        ]

        # Compute batch rewards
        results = await pool.compute_rewards_batch(requests)

        # Print results
        total_score = 0
        for result in results:
            total_score += result.score
            print(f"Score: {result.score:.3f}, Latency: {result.latency:.3f}s")

        print(f"\nAverage score: {total_score/len(results):.3f}")
        print(f"Metrics: {pool.get_metrics()}")


if __name__ == "__main__":
    asyncio.run(example_usage())
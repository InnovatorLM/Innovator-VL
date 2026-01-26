"""
Example judge model server for external reward computation.

This is a reference implementation showing how to build a judge model server
that works with our reward adapter. You can deploy this as a separate service
and modify according to your specific judge model.

Components:
1. FastAPI server for handling HTTP requests
2. Judge model wrapper (replace with your actual model)
3. Batch processing for efficiency
4. Caching and rate limiting
"""

import asyncio
import time
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
from logging import config

# Configure logging
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
})

logger = logging.getLogger(__name__)


# Request/Response Models
class JudgeRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt")
    completion: str = Field(..., description="Model's completion")
    answer: Optional[str] = Field(None, description="Ground truth answer")
    answer_type: Optional[str] = Field(None, description="Type of answer")
    request_id: Optional[str] = Field(None, description="Unique request ID")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class JudgeResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Reward score [0, 1]")
    request_id: Optional[str] = Field(None, description="Request ID")
    explanation: Optional[str] = Field(None, description="Explanation for the score")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class BatchJudgeRequest(BaseModel):
    requests: List[JudgeRequest] = Field(..., min_items=1, max_items=128)


class BatchJudgeResponse(BaseModel):
    responses: List[JudgeResponse]


# Placeholder Judge Model - Replace with your actual model
class DummyJudgeModel:
    """Example judge model that returns scores based on simple rules."""

    def __init__(self):
        logger.info("Initializing dummy judge model")
        # Add any model loading or initialization here

    def score(self, prompt: str, completion: str, answer: Optional[str] = None) -> Dict:
        """
        Compute score for a single example.
        Replace this with your actual model inference.
        """
        score = 0.5  # Default neutral score
        explanation = "Neutral score"

        if answer and answer in completion:
            score = 1.0
            explanation = f"Correct answer '{answer}' found in completion"
        elif answer:
            # Check for variations
            answer_lower = answer.lower().strip()
            completion_lower = completion.lower()

            if answer_lower in completion_lower:
                score = 0.8
                explanation = f"Answer variant found (subtle difference)"
            else:
                # Check for boxed content
                if "**boxed{" in completion:
                    # Try to extract boxed content
                    import re
                    boxed_matches = re.findall(r'\*\*boxed\{([^}]+)\}', completion)
                    if boxed_matches:
                        if answer_lower in [match.lower().strip() for match in boxed_matches]:
                            score = 0.9
                            explanation = f"Answer found in boxed format"
                        else:
                            score = 0.0
                            explanation = f"Boxed content doesn't match: {boxed_matches[0]} vs {answer}"
                else:
                    score = 0.0
                    explanation = f"Answer '{answer}' not found in completion"

        # Add some complexity based on completion quality
        if len(completion) < len(prompt):
            score *= 0.9  # Penalize very short completions

        return {
            'score': score,
            'explanation': explanation,
        }


# Global model instance
judge_model: Optional[DummyJudgeModel] = None


# FastAPI lifespan for model management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle."""
    global judge_model

    # Startup
    logger.info("Starting judge model server")
    judge_model = DummyJudgeModel()
    logger.info("Judge model loaded successfully")

    # Pre-warm the model (optional)
    _ = judge_model.score("test", "test completion", "test")

    yield

    # Shutdown
    logger.info("Shutting down judge model server")
    judge_model = None


# Create FastAPI app
app = FastAPI(
    title="Judge Model API",
    description="External judge model for RL reward computation",
    version="1.0.0",
    lifespan=lifespan
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": judge_model is not None
    }


# Single judge endpoint
@app.post("/judge", response_model=JudgeResponse)
async def judge_single(prompt: JudgeRequest):
    """Judge a single prompt-completion pair."""
    if not judge_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        result = judge_model.score(
            prompt.prompt,
            prompt.completion,
            prompt.answer
        )
        latency = time.time() - start_time

        logger.info(f"Judged request {prompt.request_id or 'unknown'} "
                   f"in {latency:.3f}s, score: {result['score']:.3f}")

        return JudgeResponse(
            score=result['score'],
            request_id=prompt.request_id,
            explanation=result['explanation'],
            metadata={
                'latency': latency,
                'model_version': 'dummy-v1'
            }
        )

    except Exception as e:
        logger.error(f"Error in judge_single: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch judge endpoint
@app.post("/batch_judge", response_model=BatchJudgeResponse)
async def judge_batch(request: BatchJudgeRequest):
    """Judge a batch of prompt-completion pairs."""
    if not judge_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()
        responses = []

        # Process batch with optional concurrency limit
        for prompt in request.requests:
            result = judge_model.score(
                prompt.prompt,
                prompt.completion,
                prompt.answer
            )

            responses.append(JudgeResponse(
                score=result['score'],
                request_id=prompt.request_id,
                explanation=result['explanation'],
                metadata={}
            ))

        total_latency = time.time() - start_time
        logger.info(f"Processed batch of {len(request.requests)} "
                   f"requests in {total_latency:.3f}s "
                   f"(avg: {total_latency/len(request.requests):.3f}s per request)")

        return BatchJudgeResponse(responses=responses)

    except Exception as e:
        logger.error(f"Error in judge_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    return {
        "model_loaded": judge_model is not None,
        "timestamp": time.time(),
        "uptime": time.time() - (getattr(app.state, 'start_time', time.time()) or time.time())
    }


# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


# If using a real model, you'd want caching and rate limiting
# Here's an example with simple in-memory cache:

from functools import lru_cache
import hashlib

class CachedJudgeModel(DummyJudgeModel):
    """Judge model with simple caching."""

    def _cache_key(self, prompt: str, completion: str, answer: str) -> str:
        """Generate cache key."""
        content = f"{prompt}||{completion}||{answer}"
        return hashlib.md5(content.encode()).hexdigest()

    @lru_cache(maxsize=10000)
    def _cached_score(self, cache_key: str, completion: str, answer: str) -> Dict:
        """Cached scoring (cache_key included to enable invalidation)."""
        return super().score("", completion, answer)

    def score(self, prompt: str, completion: str, answer: Optional[str] = None) -> Dict:
        """Score with caching."""
        cache_key = self._cache_key(prompt, completion, answer or "")
        # Reuse parent's scoring but with cache
        result = self._cached_score(cache_key, completion, answer or "")
        return result


# Example of how to extend for specific models
class CustomJudgeModel:
    """Template for integrating a custom judge model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        """Load your actual model here."""
        # Example:
        # self.model = load_model(model_path)
        # self.tokenizer = load_tokenizer(model_path)
        # self.device = device
        # self.model.to(device)
        pass

    def score(self, prompt: str, completion: str, answer: Optional[str] = None) -> Dict:
        """
        Implement your scoring logic here.

        Example workflow:
        1. Combine prompt and completion
        2. Tokenize
        3. Run model inference
        4. Post-process outputs
        5. Return score with explanation
        """
        # Example:
        # text = f"Question: {prompt}\nAnswer: {completion}\n\\nRate this answer from 0 to 1:"
        # inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(**inputs, max_length=512)
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        # # Parse the model output to extract score
        # score = parse_score_from_text(response)
        #
        # return {
        #     'score': score,
        #     'explanation': f"Model feedback: {response}"
        # }

        # For now, raise not implemented
        raise NotImplementedError("Please implement your custom scoring logic")


# Example usage
if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "judge_server_example:app",
        host="0.0.0.0",
        port=8888,
        workers=1,  # Keep 1 for model state
        loop="uvloop",  # Linux optimized event loop
        log_level="info"
    )

    # Test the server while running:
    # curl -X POST http://localhost:8888/judge \
    #   -H "Content-Type: application/json" \
    #   -d '{
    #     "prompt": "What is 2+2?",
    #     "completion": "The answer is **boxed{4}**",
    #     "answer": "4",
    #     "answer_type": "NUMBER",
    #     "request_id": "test_123"
    #   }'
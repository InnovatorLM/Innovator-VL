# Judge Model Integration Guide

This document explains how to integrate external judge model based rewards into the RL training pipeline, replacing the current rule-based reward system.

## Architecture Overview

The new system consists of three main components:

1. **Judge Model Server** (`judge_server_example.py`): External service hosting your judge model
2. **Reward Adapters** (`judge_reward_adapter.py`): Drop-in replacement for existing reward system
3. **Async Batch Processor** (`async_judge_reward.py`): High-performance async client

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Training      │──────│ JudgeReward      │──────│ Judge Model     │
│   Pipeline      │      │ Adapter          │      │ Server          │
│                 │      │                  │      │ (External)      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
          │                      │                         │
          │ 1. Send requests     │ 2. Batch requests      │ 3. Inference
          │    (sync interface)  │    (async behind)     │    on GPU
```

## Quick Start

### 1. Deploy Judge Model Server

First, deploy your judge model as a separate service. Use the example server as template:

```bash
# Start the judge model server
python judge_server_example.py
```

Customize the `CustomJudgeModel` class in `judge_server_example.py` to load and use your actual judge model.

### 2. Update Training Configuration

Add judge model configuration to your training YAML:

```yaml
# Add to your training config (llavaov15-8b_stage2_grpo.yaml)
reward:
  type: judge_model  # Change from None to judge_model
  config:
    judge_url: ${JUDGE_MODEL_URL}  # http://localhost:8888
    api_key: ${JUDGE_MODEL_API_KEY}  # Optional
    use_cache: true
    cache_size: 5000

    # Performance
g    max_batch_size: 16  # Match your GPU memory
    batch_timeout: 0.05  # 50ms to form batches
    max_concurrent_requests: 32
    timeout: 30.0

    # Reliability
    fallback_to_rule: true  # Fallback to rules on failure
    max_retries: 3
```

Environment variables:
```bash
export JUDGE_MODEL_URL="http://your-judge-server:8888"
export JUDGE_MAX_BATCH_SIZE=16
export JUDGE_MAX_CONCURRENT=32
```

### 3. Integrate in Training Script

The integration is backwards compatible. Just replace the import:

```python
# In trains/grpo.py or workflow/vision_rlvr.py
# old: from reward.reward_system import RewardSystem
# new:
from reward.judge_reward_adapter import JudgeRewardSystem as RewardSystem

# Everything else stays the same!
reward_system = RewardSystem(config.get('reward', {}).get('config', {}))
```

## Judge Model Implementation Guide

### Example Judge Model

Your judge model should:
1. Take (prompt, completion, answer) as input
2. Return a score [0, 1] with optional explanation
3. Handle batch processing efficiently

```python
class YourJudgeModel:
    def __init__(self, model_path: str):
        # Load your model
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(model_path)

    def score(self, prompt: str, completion: str, answer: str) -> Dict:
        # Combine prompt and completion
        text = f"Question: {prompt}\nGenerated Answer: {completion}"

        # Check if answer exists (simplistic example)
        if answer and answer.lower() in completion.lower():
            score = 0.9
        else:
            # Run your model to get detailed scoring
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            score = outputs.score.item()

        return {
            'score': min(max(score, 0.0), 1.0),
            'explanation': f'Model confidence: {score:.3f}'
        }
```

### Advanced Judge Model Features

1. **Multi-aspect scoring**: Score different aspects separately
```python
return {
    'score': overall_score,
    'explanation': f'Correctness: {correctness}, Clarity: {clarity}',
    'detailed_scores': {
        'correctness': correctness,
        'clarity': clarity,
        'relevance': relevance
    }
}
```

2. **Contextual scoring**: Use answer type for better scoring
```python
def score(self, prompt: str, completion: str, answer: str, answer_type: str):
    if answer_type == "MATH_EXPRESSIONS":
        # symbolic math checking
    elif answer_type == "MULTIPLE_CHOICE":
        # letter matching
    # ... other types
```

3. **Confidence calibration**: Adjust scores based on uncertainty
```python
confidence = compute_uncertainty(outputs)
if confidence < 0.5:
    # Unclear case, be more conservative
    score = score * 0.8
```

## Configuration Options

### Performance Tuning

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `max_batch_size` | Max requests per batch | 8-32 (based on GPU) |
| `batch_timeout` | Time to wait for batch formation | 0.01-0.1s |
| `max_concurrent_requests` | Simultaneous HTTP requests | 16-64 |
| `cache_size` | Cached reward entries | 1000-10000 |
| `timeout` | HTTP request timeout | 10-30s |

### Reliability Options

| Parameter | Description |
|-----------|-------------|
| `fallback_to_rule` | Fallback to rules on failure |
| `circuit_breaker_threshold` | Failures before opening circuit |
| `circuit_breaker_timeout` | Recovery time after circuit opens |
| `max_retries` | Retry failed HTTP requests |

## Expected Judge Model Server API

### Single Request
```bash
POST /judge
{
    "prompt": "What is 2+2?",
    "completion": "The answer is **boxed{4}**",
    "answer": "4",
    "answer_type": "NUMBER",
    "request_id": "unique_id"
}

Response:
{
    "score": 0.95,
    "request_id": "unique_id",
    "explanation": "Correct answer found with proper formatting",
    "metadata": {"latency": 0.05}
}
```

### Batch Request
```bash
POST /batch_judge
{
    "requests": [
        {p"prompt": "...", "completion": "..."},
        {p"prompt": "...", "completion": "..."}
    ]
}

Response:
{
    "responses": [
        {"score": 0.8, "request_id": "1"},
        {"score": 0.6, "request_id": "2"}
    ]
}
```

## Monitoring and Debugging

### 1. Check judge model health
```bash
curl http://your-judge-server:8888/health
```

### 2. Monitor adapter metrics
```python
# In your training loop
if global_step % 100 == 0:
    metrics = reward_system.get_metrics()
    print(f"Judge metrics: {metrics}")
    # Track: cache_hit_rate, latency, error_rate
```

### 3. Analyze reward distribution
```python
reward_scores = []
for batch in batch_iterations:
    for item in batch:
        result = reward_system.reward(item.prompt, item.completion, item.answer)
        reward_scores.append(result['reward'])

# Plot histogram to check reward distribution
```

## Performance Considerations

### 1. Batch Processing
- Collects requests for 50-100ms before sending
- Reduces HTTP overhead significantly
- Aim for >80% of requests batched

### 2. Caching
- Caches identical (prompt, completion, answer) tuples
- Useful for repeated generations during training
- Monitor cache hit rate (should be 20-50%)

### 3. Async Overhead
- Async processing adds ~1ms per request
- Amortized by batching and concurrent processing
- Overall throughput typically 10-100x better

### 4. Network Latency
- Co-locate judge server with training cluster
- Use same data center / availability zone
- Expected latency: 5-50ms with 10-100 requests

## Fallback Strategy

The system automatically falls back to rule-based rewards when:
1. Judge model server is unavailable
2. HTTP requests fail after retries
3. Circuit breaker is open
4. Individual requests timeout

Fallback behavior:
- Logs warning with error details
- Returns rule-based scores
- Tag rewards with `source: rule_fallback`
- Continues training without interruption

## Advanced Usage

### Custom Judge Logic
Create a custom judge class:
```python
from reward.judge_reward_adapter import BaseJudgeReward

class MyCustomJudge(BaseJudgeReward):
    def _request_judge(self, request):
        # Your custom logic here
        return super()._request_judge(request)
```

### Multi-critique Judge
Chain multiple judges:
```python
class MultiCritiqueJudge:
    def __init__(self, judges: List[str]):
        self.judges = [create_judge_reward({'judge_url': url})
                      for url in judges]

    def score(self, prompt: str, completion: str, answer: str):
        scores = []
        for judge in self.judges:
            scores.append(judge.compute_reward(prompt, completion, answer))

        # Average scores or use voting
        return {
            'score': np.mean([s.score for s in scores]),
            'explanation': f'Voting平均得分: {np.mean(scores):.2f}'
        }
```

## Troubleshooting

### Common Issues

1. **"Circuit breaker open"**
   - Judge server likely down
   - Check server health and logs

2. **"Request queue full"**
   - Training asks for rewards faster than processing
   - Increase `max_concurrent_requests` or reduce `batch_size`

3. **"Timeout waiting for results"**
   - Judge model too slow
   - Optimize model or increase timeout

4. **Poor reward quality**
   - Judge model needs better training
   - Add more training data for your task
   - Fine-tune on domain-specific examples

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger("JudgeReward").setLevel(logging.DEBUG)
```

Check metrics at each step:
```python
print(f"Inflight: {metrics['current_queue_size']}")
print(f"Hit rate': {metrics['cache_hit_rate']:.1%}")
print(f"Circuit: {metrics['circuit_breaker']['state']}")
```

## Next Steps

1. **Implement your judge model** based on the example server
2. **Start small**: Begin with single GPU judge, scale later
3. **Monitor performance** and tune batching parameters
4. **Compare reward quality** between rules and judge model
5. **Iterate on judge model** based on training results

The new system provides much more flexibility while maintaining backward compatibility and reliability through fallback mechanisms.
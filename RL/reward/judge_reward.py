"""
Judge Model based reward function (simple mode)
Uses custom_reward.py's compute_score function
"""

import os
from dataset.const import AnswerType

# Import from custom_reward
from .custom_reward import compute_score

def judge_reward_fn(completions: str, answer: str, **kwargs):
    """
    Reward function that uses Judge Model to evaluate answers.
    
    Args:
        completions: Model's completion string
        answer: Ground truth answer
        **kwargs: Additional info including prompt, problem_type, etc.
    
    Returns:
        float: Reward score (0.0 to 1.0)
    """
    # Only use judge model if explicitly enabled
    if os.getenv("USE_LLM_JUDGE", "False") != "True":
        # Fallback to string matching if judge is disabled
        from .string_matching import string_matching_reward_fn
        return string_matching_reward_fn(completions, answer)
    
    # Prepare extra_info for judge
    extra_info = {}
    if "prompt" in kwargs:
        extra_info["question"] = kwargs["prompt"]
    if "problem_type" in kwargs:
        extra_info["problem_type"] = kwargs["problem_type"]
    if "prompt_type" in kwargs:
        extra_info["prompt_type"] = kwargs["prompt_type"]
    if "data_source" in kwargs:
        extra_info["data_source"] = kwargs["data_source"]
    
    # Get data source from kwargs or use default
    data_source = kwargs.get("data_source", "general")
    
    # Call compute_score from custom_reward
    result = compute_score(
        data_source=data_source,
        solution_str=completions,
        ground_truth=answer,
        extra_info=extra_info if extra_info else None
    )
    
    # Return the final score
    return result["score"]

import re
import os

from math_verify import parse, verify
from openai import OpenAI

# vLLM Judge Model配置
VLLM_JUDGE_URL = os.getenv("VLLM_JUDGE_URL", "http://localhost:8000/v1")
VLLM_JUDGE_MODEL = os.getenv("VLLM_JUDGE_MODEL", "judge-model")
VLLM_JUDGE_API_KEY = os.getenv("VLLM_JUDGE_API_KEY", "dummy-key")
USE_LLM_JUDGE = os.getenv("USE_LLM_JUDGE", "True")  # 默认启用vLLM judge
JUDGE_MODE = os.getenv("JUDGE_MODE", "enhanced")  # simple|enhanced 两种评分模式

# Judge 调用配置
JUDGE_TIMEOUT_SECONDS = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "60.0"))  # 超时时间（秒）
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "3"))  # 最大重试次数
JUDGE_RETRY_DELAY = float(os.getenv("JUDGE_RETRY_DELAY", "0.5"))  # 重试延迟（秒）

# Don't initialize client at module level - it will be created per-process
# to avoid multiprocessing serialization issues
_client = None

def get_client():
    """Get or create OpenAI client. Creates a new client per process."""
    global _client
    if _client is None:
        # Re-read env vars in case they were set after module import
        url = os.getenv("VLLM_JUDGE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("VLLM_JUDGE_API_KEY", "dummy-key")
        # 在客户端级别设置超时
        _client = OpenAI(
            api_key=api_key, 
            base_url=url,
            timeout=JUDGE_TIMEOUT_SECONDS
        )
    return _client

# 添加日志记录
import logging
import time
logger = logging.getLogger("CustomReward")
logger.setLevel(logging.INFO)
# 避免重复创建handler
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

JUDGE_PROMPT = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.

# Input
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0"""

JUDGE_PROMPT_WITH_ANSWER = """
You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case. You will receive the question, the ground truth answer, and the model prediction.

# Input
Question:
```
{question}
```

Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{prediction}
```

# Evaluation Rules
- For multiple-choice questions: Score 1 if the predicted answer matches the ground truth answer, it can be directly in option letters or the content of the options.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically, it can be in different format.
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
1 or 0
"""


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \boxed{} format.

    Args:
        predict_str (str): The prediction string containing the boxed answer.

    Returns:
        str: The extracted answer from \boxed{}, or an empty string if not found.
    """
    # Find all occurrences of \boxed{
    boxed_start = "\\boxed{"
    start_indices = []
    
    # Find all positions where \boxed{ starts
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1
    
    if not start_indices:
        return ""
    
    # For each \boxed{ occurrence, find the matching closing brace
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1  # Position at the opening brace of \boxed{
        
        while pos < len(predict_str):
            char = predict_str[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1
    
    # Return the last (rightmost) match if multiple found
    return results[-1] if results else ""


def extract_anwser_tag(predict_str) -> str:
    """Extract the answer tag from the prediction string.
    
    This function now handles both <answer> tags and \boxed{} format.

    Args:
        predict_str: The prediction string containing the answer tag. Can be str, list, or None.

    Returns:
        str: The extracted answer tag, or an empty string if not found.
    """
    # Handle None or empty input
    if predict_str is None:
        return ""
    
    # Handle list input - convert to string
    if isinstance(predict_str, list):
        if len(predict_str) == 0:
            return ""
        predict_str = str(predict_str[0]) if len(predict_str) == 1 else ", ".join(str(item) for item in predict_str)
    
    # Ensure it's a string
    predict_str = str(predict_str)
    
    # First try to extract from <answer> tags
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)
    
    # If no <answer> tag found, try to extract from \boxed{} format
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer
    
    # If neither format found, try to extract the last number or expression
    # This is a fallback for cases where the answer is just stated without formatting
    lines = predict_str.strip().split('\n')
    for line in reversed(lines):
        # Look for patterns like "The answer is 204" or just "204"
        if line.strip():
            # Try to find numbers at the end of the line
            number_match = re.search(r'\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)', line)
            if number_match:
                return number_match.group(1)
    
    return ""


def format_reward(predict_str: str) -> float:
    """Check if the prediction string follows the expected format.
    
    Now handles both <think><answer> format and \boxed{} format.
    """
    # Check for <think>.*</think>.*<answer>.*</answer> pattern
    think_answer_pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    if re.fullmatch(think_answer_pattern, predict_str):
        return 1.0
    
    # Check for \boxed{} format (common in mathematical solutions)
    if extract_boxed_answer(predict_str):
        return 1.0
    
    # Check for basic answer format (contains some mathematical content and ends with a number)
    if len(predict_str.strip()) > 50:  # Reasonable solution length
        # Look for mathematical expressions or reasoning
        has_math = bool(re.search(r'[=\+\-\*/\(\)\[\]\\]', predict_str))
        # Look for final answer
        has_answer = bool(extract_anwser_tag(predict_str))
        
        if has_math and has_answer:
            return 0.8  # Partial credit for reasonable format
    
    return 0.0


def simple_parse(predict_str) -> str:
    """Parse the prediction string to extract the answer.

    Args:
        predict_str: The prediction string to be parsed. Can be str, list, or None.

    Returns:
        str: The parsed answer from the prediction string.
    """
    # Handle None or empty input
    if predict_str is None:
        return ""
    
    # Handle list input - convert to string
    if isinstance(predict_str, list):
        if len(predict_str) == 0:
            return ""
        # If list has one element, use it; otherwise join with comma
        if len(predict_str) == 1:
            predict_str = str(predict_str[0])
        else:
            predict_str = ", ".join(str(item) for item in predict_str)
    
    # Ensure it's a string
    predict_str = str(predict_str)
    
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]

    return predict_str.strip()


def parse_mcq(predict_str: str) -> str:
    """
    Parse multiple choice answers from various formats.
    Handles formats like: "A", "A.", "A)", "(A)", "The answer is A", "A: xxx", etc.
    """
    if not predict_str or predict_str.strip() == "":
        return ""
    
    # Clean up the response
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    
    # Add spaces to avoid partial matches
    response = " " + response + " "
    
    # All possible choice letters (extend if needed)
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    candidates = []
    
    # Pattern 1: Look for choices with parentheses e.g., (A), (B), (C), (D)
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))
    
    # Pattern 2: Look for choices with periods e.g., A., B., C., D.
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))
    
    # Pattern 3: Look for choices with colons e.g., A:, B:, C:, D:
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))
    
    # Pattern 4: Look for choices with right parentheses e.g., A), B), C), D)
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))
    
    # Pattern 5: Look for choices with spaces after e.g., A B C D
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))
    
    # Pattern 6: Look for choices with dashes e.g., A- B- C- D-
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))
    
    # Pattern 7: Look for choices with underscores e.g., A_ B_ C_ D_
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))
    
    # Pattern 8: Look for choices with equal signs e.g., A= B= C= D=
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))
    
    # Pattern 9: Look for common answer phrases followed by choices
    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "the answer", "answer", "correct answer", "the correct answer",
        "the best answer is", "best answer is", "the best answer", "best answer",
        "the option is", "option is", "the correct option is", "correct option is",
        "the choice is", "choice is", "the correct choice is", "correct choice is",
        "i choose", "i select", "i pick", "my answer is", "my choice is"
    ]
    
    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            # Look for choices after the phrase
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))
    
    # Pattern 10: Look for choices at the very beginning of the response
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))
    
    # Pattern 11: Look for choices at the very end of the response
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))
    
    # Pattern 12: Look for choices with numbers (e.g., "1. A", "2. B")
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))
    
    # If no candidates found, try to extract from the entire response
    if not candidates:
        # Look for any choice letter in the response
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))
    
    # Return the best candidate
    if candidates:
        # Sort by position (later in text) and priority of format
        format_priority = {
            "start": 10, "end": 9, "numbered": 8, "phrase": 7, 
            "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3,
            "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0
        }
        
        # Sort by format priority first, then by position
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]
    
    return ""


def relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """Check if the prediction string matches the ground truth exactly.

    Args:
        predict_str (str): The prediction string to be checked.
        ground_truth (str): The ground truth string for comparison.
        relax_portion (float): The minimum portion of length required for partial matches.

    Returns:
        float: 1.0 if the prediction matches the ground truth, otherwise 0.0.
    """
    # If the question is an mcq
    if ground_truth in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = parse_mcq(predict_str)
        if predict_str == ground_truth:
            return 1.0
        return 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0



def llm_as_judge_sync(predict_str, ground_truth, extra_info):
    """原简单版LLM judge，保持兼容性，带超时和重试机制"""
    # Handle list inputs
    if isinstance(ground_truth, list):
        ground_truth = ", ".join(str(item) for item in ground_truth) if ground_truth else ""
    if isinstance(predict_str, list):
        predict_str = ", ".join(str(item) for item in predict_str) if predict_str else ""
    
    # Ensure string type
    ground_truth = str(ground_truth) if ground_truth is not None else ""
    predict_str = str(predict_str) if predict_str is not None else ""
    
    if extra_info is not None and "question" in extra_info:
        prompt = JUDGE_PROMPT_WITH_ANSWER.format(question=extra_info["question"], answer=ground_truth, prediction=predict_str)
    else:
        prompt = JUDGE_PROMPT.format(answer=ground_truth, prediction=predict_str)
    
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        "max_tokens": 5,
        "model": VLLM_JUDGE_MODEL,
    }
    
    # 重试机制
    last_exception = None
    for attempt in range(JUDGE_MAX_RETRIES + 1):
        try:
            response = get_client().chat.completions.create(**payload)
            try:
                score = int(response.choices[0].message.content)
                return score
            except (ValueError, KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Failed to parse judge response (attempt {attempt + 1}/{JUDGE_MAX_RETRIES + 1}): {e}")
                if attempt < JUDGE_MAX_RETRIES:
                    time.sleep(JUDGE_RETRY_DELAY * (attempt + 1))  # 指数退避
                    continue
                return 0
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            # 判断是否应该重试
            is_retryable = any(keyword in error_msg for keyword in [
                "timeout", "connection", "network", "unavailable", 
                "rate limit", "503", "502", "500", "429"
            ])
            
            if is_retryable and attempt < JUDGE_MAX_RETRIES:
                logger.warning(
                    f"Judge model call failed (attempt {attempt + 1}/{JUDGE_MAX_RETRIES + 1}): {e}. "
                    f"Retrying in {JUDGE_RETRY_DELAY * (attempt + 1):.1f}s..."
                )
                time.sleep(JUDGE_RETRY_DELAY * (attempt + 1))  # 指数退避
                continue
            else:
                logger.error(f"Judge model call failed after {attempt + 1} attempts: {e}")
                if not is_retryable:
                    # 非重试性错误，直接返回0
                    return 0
    
    # 所有重试都失败了
    logger.error(f"Judge model call failed after {JUDGE_MAX_RETRIES + 1} attempts. Last error: {last_exception}")
    return 0


def llm_as_judge_enhanced(predict_str, ground_truth, extra_info=None):
    """增强版vLLM judge - 支持两部分评分"""

    # 确定问题类型
    question_type = "general"
    if extra_info:
        if "math" in str(extra_info.get("data_source", "")).lower():
            question_type = "math"
        elif any(word in str(extra_info.get("question", "")).lower() for word in ["reason", "explain", "why", "how"]):
            question_type = "reasoning"
        elif any(word in str(extra_info.get("question", "")).lower() for word in ["code", "program", "function"]):
            question_type = "coding"

    # 构建增强版系统prompt
    system_prompt = f"""You are an expert evaluator specializing in {question_type} problems.

Your task is to evaluate both:
1. Reasoning quality (if thinking process is present)
2. Answer correctness

You MUST output a JSON object with this exact structure:
{{
  "total_score": <float 0.0-1.0>,
  "reasoning_score": <float 0.0-1.0>,  // quality of thinking process
  "answer_score": <float 0.0-1.0>,    // correctness of final answer
  "format_score": <float 0.0-1.0>,    // adherence to expected format
  "explanation": "brief explanation of evaluation",
  "confidence": <float 0.0-1.0>
}}

Evaluation Guidelines:
=== REASONING QUALITY ===
- Look for \u003cthink\u003e...\u003c/think\u003e sections
- Check logic, coherence, and relevance to question
- Score 1.0: Clear, logical, step-by-step reasoning
- Score 0.5: Some reasoning but unclear or incomplete
- Score 0.0: No reasoning or completely illogical

=== ANSWER CORRECTNESS ===
- Compare final answer with ground truth
- Consider different but equivalent forms (e.g., 1/2 = 0.5)
- Score 1.0: Exactly correct answer
- Score 0.5: Partially correct or minor errors
- Score 0.0: Completely wrong answer

=== FORMAT ADHERENCE ===
- Check for \u003canswer\u003e...\u003c/answer\u003e tags or \boxed{{}} format
- Verify thinking section if expected
- Score based on format compliance"""

    # 构建用户prompt
    user_content = f"""
Question: {extra_info.get('question', '') if extra_info else 'See prediction below'}
Ground Truth Answer: {ground_truth}
Model Completion/Prediction:
{predict_str}

Evaluate this completion based on the criteria above.
Pay special attention to:
1. Any thinking/reasoning process present
2. Final answer correctness
3. Format compliance (tags, boxes, etc.)
""".strip()

    # 调用vLLM judge model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    payload = {
        "messages": messages,
        "temperature": 0.1,  # 低温度确保一致性
        "max_tokens": 300,   # 足够的空间输出JSON
        "model": VLLM_JUDGE_MODEL,
        "response_format": {"type": "json_object"},  # 强制JSON格式（如果兼容）
    }

    # 重试机制
    last_exception = None
    content = None
    
    for attempt in range(JUDGE_MAX_RETRIES + 1):
        try:
            response = get_client().chat.completions.create(**payload)
            content = response.choices[0].message.content

            # 解析JSON响应
            import json
            try:
                result = json.loads(content)
                return {
                    "score": result.get("total_score", 0.0),
                    "reasoning_score": result.get("reasoning_score", 0.0),
                    "answer_score": result.get("answer_score", 0.0),
                    "format_score": result.get("format_score", 0.0),
                    "explanation": result.get("explanation", ""),
                    "confidence": result.get("confidence", 0.8)
                }
            except json.JSONDecodeError:
                # Fallback: 提取第一个小数作为score
                if content:
                    try:
                        import re
                        scores = re.findall(r'\d+\.\d+', content)
                        if len(scores) >= 1:
                            logger.warning(f"JSON parse failed, extracted score from text: {scores[0]}")
                            return {
                                "score": float(scores[0]),
                                "reasoning_score": 0.0,
                                "answer_score": float(scores[0]),
                                "format_score": 0.0,
                                "explanation": f"Parse error, extracted score: {scores[0]}",
                                "confidence": 0.6
                            }
                    except Exception as parse_e:
                        logger.warning(f"Failed to extract score from response: {parse_e}")
                
                # JSON解析失败，如果是最后一次尝试，返回错误
                if attempt < JUDGE_MAX_RETRIES:
                    logger.warning(f"JSON decode failed (attempt {attempt + 1}/{JUDGE_MAX_RETRIES + 1}), retrying...")
                    time.sleep(JUDGE_RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return {
                        "score": 0.0,
                        "reasoning_score": 0.0,
                        "answer_score": 0.0,
                        "format_score": 0.0,
                        "explanation": "Failed to parse judge response",
                        "confidence": 0.0
                    }

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            # 判断是否应该重试
            is_retryable = any(keyword in error_msg for keyword in [
                "timeout", "connection", "network", "unavailable", 
                "rate limit", "503", "502", "500", "429"
            ])
            
            if is_retryable and attempt < JUDGE_MAX_RETRIES:
                logger.warning(
                    f"Judge model call failed (attempt {attempt + 1}/{JUDGE_MAX_RETRIES + 1}): {e}. "
                    f"Retrying in {JUDGE_RETRY_DELAY * (attempt + 1):.1f}s..."
                )
                time.sleep(JUDGE_RETRY_DELAY * (attempt + 1))  # 指数退避
                continue
            else:
                logger.error(f"Judge model call failed after {attempt + 1} attempts: {e}")
                if not is_retryable:
                    # 非重试性错误，直接返回失败结果
                    return {
                        "score": 0.0,
                        "reasoning_score": 0.0,
                        "answer_score": 0.0,
                        "format_score": 0.0,
                        "explanation": f"Judge error: {str(e)}",
                        "confidence": 0.0
                    }
    
    # 所有重试都失败了
    logger.error(f"Judge model call failed after {JUDGE_MAX_RETRIES + 1} attempts. Last error: {last_exception}")
    return {
        "score": 0.0,
        "reasoning_score": 0.0,
        "answer_score": 0.0,
        "format_score": 0.0,
        "explanation": f"Judge error after retries: {str(last_exception) if last_exception else 'Unknown error'}",
        "confidence": 0.0
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """Compute the score for a given solution based on the data source.

    增强版：支持vLLM judge model的两部分评分（思考+答案）

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        sandbox_fusion_url: Not used in this implementation.
        concurrent_semaphore: Not used in this implementation.

    Returns:
        dict: A dictionary containing the computed score and other metrics.
    """

    # Step 1: 格式评估（基于原有逻辑）
    format_score = 0.1  # 基础格式权重
    format_reward_score = format_reward(solution_str)

    # Step 2: 使用vLLM Judge模型评估（新增增强版）
    if USE_LLM_JUDGE == "True":
        if JUDGE_MODE == "enhanced":
            # 增强版：两部分评分
            judge_result = llm_as_judge_enhanced(solution_str, ground_truth, extra_info)

            # 综合评分：权重可以根据 prompt_type 调整
            prompt_type = extra_info.get("prompt_type", "normal") if extra_info else "normal"
            if prompt_type == "thinking":
                # thinking prompt：更重视推理过程
                final_score = (
                    0.10 * judge_result["format_score"] +
                    0.35 * judge_result["reasoning_score"] +
                    0.55 * judge_result["answer_score"]
                )
            else:
                # normal prompt：更重视答案正确性
                final_score = (
                    0.05 * judge_result["format_score"] +
                    0.15 * judge_result["reasoning_score"] +
                    0.80 * judge_result["answer_score"]
                )

            # 返回增强的结果格式
            return {
                "score": final_score,
                "acc_score": judge_result["answer_score"],  # 答案分的一致性
                "format_reward_score": format_reward_score,
                "reasoning_score": judge_result["reasoning_score"],
                "format_score": judge_result["format_score"],
                "thinking_score": judge_result["reasoning_score"],
                "predict_str": extract_anwser_tag(solution_str).strip(),
                "ground_truth": simple_parse(ground_truth),
                "judge_explanation": judge_result["explanation"],
                "judge_confidence": judge_result["confidence"],
                "validation_method": "vllm_enhanced",
                "source": "vllm_judge"
            }

        # 简单版：保持原有的二进制评分
        else:
            return _compute_score_simple(data_source, solution_str, ground_truth, extra_info,
                                       format_score, format_reward_score)

    # 不使用LLM judge：回退到原有逻辑
    else:
        return _compute_score_simple(data_source, solution_str, ground_truth, extra_info,
                                   format_score, format_reward_score)


def _compute_score_simple(data_source, solution_str, ground_truth, extra_info,
                        format_score, format_reward_score):
    """原有的评分逻辑（简化版）"""
    extracted_answer = extract_anwser_tag(solution_str).strip()
    predict_str = simple_parse(extracted_answer)
    gt = simple_parse(ground_truth)

    acc_score = relax_exact_match(predict_str, gt)
    if acc_score == 0.0:
        try:
            gold = parse(gt)
            pred = parse(predict_str)
            acc_score = int(verify(gold, pred))
        except Exception:
            acc_score = 0.0

    if acc_score == 0.0 and USE_LLM_JUDGE == "True":
        acc_score = llm_as_judge_sync(predict_str, ground_truth, extra_info)

    # 特殊情况：直接评判整个solution
    if acc_score == 0.0 and USE_LLM_JUDGE == "True" and format_reward_score == 0.0 and len(solution_str) < 500:
        acc_score = llm_as_judge_sync(solution_str, ground_truth, extra_info)

    final_score = (1.0 - format_score) * acc_score + format_score * format_reward_score

    return {
        "score": final_score,
        "acc_score": acc_score,
        "format_reward_score": format_reward_score,
        "predict_str": predict_str,
        "ground_truth": gt,
        "validation_method": "rule_based",
        "source": "rules"
    }
from .utils import extract_boxed_content


def string_matching_reward_fn(completions: str, answer):
    """String matching reward function.
    
    Args:
        completions: Model's completion string
        answer: Ground truth answer (can be str, list, or None)
    
    Returns:
        float: 1.0 if match, 0.0 otherwise
    """
    answers = extract_boxed_content(completions)

    if answers:
        answer_str = answers[-1]
    else:
        answer_str = ""

    if answer_str == "":
        return 0
    if answer is None:
        return 0
    
    # Handle list input
    if isinstance(answer, list):
        if len(answer) == 0:
            return 0
        elif len(answer) == 1:
            answer = str(answer[0])
        else:
            answer = ", ".join(str(item) for item in answer)
    
    # Ensure string type
    answer = str(answer)

    if answer_str.strip().lower() == answer.strip().lower():
        return 1.0
    else:
        return 0.0
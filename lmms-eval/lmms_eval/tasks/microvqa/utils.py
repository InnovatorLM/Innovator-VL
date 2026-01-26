import re
from typing import Dict, Any, List
from PIL import Image

PROMPT_TEMPLATE = """\
The following is a multiple choice question (with answers). 
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{question}

Options:
{choices}
"""

PROMPT_TEMPLATE_NO_IMAGE = """\
The following is a multiple choice question (with answers).
If an image is mentioned ignore this information and try your best to answer the question.
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{question}

Options:
{choices}
"""

PROMPT_TEMPLATE_NO_IMAGE_V2 = """\
We have a multiple choice question (with answers) and images.
However I will not give you the question text or the images, I will only give you the choices, so please try your best to answer the question.
Think step by step and then output the answer in the format of "The answer is (X)" at the end, where X is the correct letter choice.

{choices}
"""


# REGEX_PATTERN = r"answer is \*?\*?\(?([0-9])\)?\*?\*?"
REGEX_PATTERN = r"[Tt]he answer is[:：\s]*\*?\*?[\(\（]?([A-Za-z0-9])[\)\）]?\*?\*?"




def _format_choices(choices: List[str]) -> str:
    s = ""
    for j, ch in enumerate(choices):
        letter = chr(65 + j)
        s += f"  ({letter}): {ch}\n"
    return s

def _get_last_sentence_regex(text: str) -> str:
    sentence_pattern = r'[^.!?]+[.!?]+'
    sentences = re.findall(sentence_pattern, text)
    return sentences[-1].strip() if sentences else text.strip()


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    mode = lmms_eval_specific_kwargs.get("mode", "default")

    if mode == "stage1":
        question = doc["question_1"]
        choices = doc["choices_1"]
    else:
        question = doc["question"]
        choices = doc["choices"]

    choices_str = _format_choices(choices)

    if mode == "noimage_v2":
        return PROMPT_TEMPLATE_NO_IMAGE_V2.format(
            choices=choices_str
        )

    if mode == "noimage_v3":
        question = _get_last_sentence_regex(question)
        return PROMPT_TEMPLATE_NO_IMAGE.format(
            question=question,
            choices=choices_str
        )

    if mode == "noimage":
        return PROMPT_TEMPLATE_NO_IMAGE.format(
            question=question,
            choices=choices_str
        )

    # default
    return PROMPT_TEMPLATE.format(
        question=question,
        choices=choices_str
    )


def doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    mode = lmms_eval_specific_kwargs.get("mode", "default")

    if mode.startswith("noimage"):
        return None

    return [img.convert("RGB") for img in doc["images_list"]]


def doc_to_target(doc: Dict[str, Any], model_specific_target_kwargs=None) -> int:
    model_specific_target_kwargs = model_specific_target_kwargs or {}
    mode = model_specific_target_kwargs.get("mode", "default")

    if mode == "stage1":
        return doc["correct_index_1"]

    return doc["correct_index"]

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    response = results[0]
    gt = doc_to_target(doc)

    match = re.search(REGEX_PATTERN, response)
    
    if match is not None:
        token = match.group(1).upper()
    else:
        res_clean = response.strip()
        if len(res_clean) == 1:
            token = res_clean.upper()
        else:
            token = None

    if token is not None:
        if token.isdigit():
            pred = int(token) - 1
            filtered_answer = chr(65 + pred) if 0 <= pred < 26 else token
        elif 'A' <= token <= 'Z':
            pred = ord(token) - ord('A')
            filtered_answer = token
        else:
            pred = -1
            filtered_answer = token
    else:
        pred = -1
        filtered_answer = response.strip()

    return {
        "accuracy": float(pred == gt),
        "filtered_answer": filtered_answer,
        "gt_index": gt
    }

def aggregation(results: List[float]) -> float:
    if not results:
        return 0.0
    return sum(results) / len(results)
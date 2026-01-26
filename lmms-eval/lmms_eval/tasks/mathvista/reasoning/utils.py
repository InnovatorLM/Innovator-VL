import json
import os
from pathlib import Path

from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

THINKING_PROMPT = (
    "Think and solve the following question step by step. "
    "Please put your thinking and analysis procedure within <think></think>. "
    "Put ONLY your final answer within <answer></answer>."
)

def mathvista_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]

def _generate_original_mathvista_query(doc):
    question = doc["question"]
    unit = doc.get("unit", "")
    choices = doc.get("choices", [])
    question_type = doc["question_type"]
    answer_type = doc["answer_type"]
    precision = doc.get("precision", 0)

    hint_text = ""
    if question_type == "multi_choice":
        hint_text = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
    else:
        if answer_type == "integer":
            hint_text = "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
        elif answer_type == "float":
            if precision == 1:
                hint_text = "Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."
            elif precision == 2:
                hint_text = "Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."
        elif answer_type == "list":
            hint_text = "Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."

    question_text = f"Question: {question}"
    if unit:
        question_text += f" (Unit: {unit})"

    choices_text = ""
    if choices:
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A')+i)}) {choice}")
        choices_text = "\n".join(texts)
    
    elements = [question_text, choices_text, hint_text]

    query = "\n".join([e for e in elements if e != ""])
    
    return query

def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    original_query = _generate_original_mathvista_query(doc)

    return f"{THINKING_PROMPT}\n<image 1>\n{original_query}"

def mathvista_process_results(doc, results):
    acc_score = 0
    format_score = 0
    
    question_text_for_log = _generate_original_mathvista_query(doc)
    extra_info = {"question": question_text_for_log}
    
    raw_response = results[0] if results else ""

    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    choices = doc.get("choices", [])
    ground_truth = doc["answer"]
    
    if choices and doc["question_type"] == "multi_choice":
        try:
            if doc["answer"] in choices:
                choice_index = choices.index(doc["answer"])
                ground_truth = options[choice_index]
        except (ValueError, IndexError):
            pass

    for pred in results:
        score_dict = compute_score(
            data_source="mathvista", 
            solution_str=pred.strip(), 
            ground_truth=ground_truth, 
            extra_info=extra_info
        )
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {
        "acc_score": acc_score / len(results) if results else 0.0, 
        "format_score": format_score / len(results) if results else 0.0,
        "raw_output": raw_response,
        "question": question_text_for_log
    }
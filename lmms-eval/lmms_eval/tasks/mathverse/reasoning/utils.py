import json
import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

THINKING_PROMPT = (
    "Think and solve the following question step by step. "
    "Please put your thinking and analysis procedure within <think></think>. "
    "Put ONLY your final answer within <answer></answer>."
)

def mathverse_doc_to_visual(doc):
    if doc.get("image") is None or str(doc["image"]).strip() == "":
        return []
    return [doc["image"].convert("RGB")]

def _get_base_question_text(doc):
    return doc["question"]

def mathverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question_content = _get_base_question_text(doc)
    
    return f"{THINKING_PROMPT}\n<image 1>\n{question_content}"

def mathverse_process_results(doc, results):
    acc_score = 0
    format_score = 0
    
    question = _get_base_question_text(doc)
    extra_info = {"question": question}
    
    raw_response = results[0] if results else ""

    for pred in results:
        score_dict = compute_score(
            data_source="mathvista",
            solution_str=pred.strip(), 
            ground_truth=doc["answer"], 
            extra_info=extra_info
        )
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {
        "acc_score": acc_score / len(results) if results else 0.0, 
        "format_score": format_score / len(results) if results else 0.0,
        "raw_output": raw_response,
        "question": question
    }
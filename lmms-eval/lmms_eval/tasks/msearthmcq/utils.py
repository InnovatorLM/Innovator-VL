import re
import json
from typing import Dict, Any, List
from PIL import Image

PROMPT_TEMPLATE = """
You are tasked with answering a multiple-choice question about the given input image.

{question}

Based on the image, select the correct option (e.g., 'A', 'B', 'C') or directly state the correct option content.

The output must be written in **JSON format** using the structure below:
```json
{{
    "answer": "Correct option or short answer",
    "Explanation": "Reasoning explaining how to derive the correct answer." 
}}
```
"""

def doc_to_text(doc: Dict[str, Any], model_specific_prompt_kwargs: Dict[str, Any] = None) -> str:
    query = doc.get("query", "")
    question = query.replace("<image>", "").strip()
    
    return PROMPT_TEMPLATE.format(question=question)

def doc_to_visual(doc: Dict[str, Any], model_specific_prompt_kwargs: Dict[str, Any] = None):
    if "image" in doc:
        image = doc["image"]
        
        if isinstance(image, Image.Image):
            return [image.convert("RGB")]
            
    return []

def doc_to_target(doc: Dict[str, Any], model_specific_prompt_kwargs: Dict[str, Any] = None) -> str:
    return doc.get("response", "")

def clean_json_string(json_string: str) -> str:
    return re.sub(r'\\', '', json_string)

def remove_trailing_commas(json_str: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', json_str)

def extract_json_from_text(text: str):
    json_block_pattern = r'(?<=```json\n)([\s\S]*?)(?=\n```)'
    json_object_pattern = r'{[\s\S]*?}'

    match_block = re.search(json_block_pattern, text)
    if match_block:
        json_str = match_block.group(1)
    else:
        match_object = re.search(json_object_pattern, text)
        if match_object:
            json_str = match_object.group(0)
        else:
            return False, text

    try:
        json_str = remove_trailing_commas(json_str)
        json_result = json.loads(json_str)
        return True, json_result
    except Exception:
        return False, text

def parse_option_and_content(answer_text: str):
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
        
    match = re.match(r'^\s*([abcdABCD])(?!\w)\s*[):.,\s]?\s*(.*)$', answer_text)
    if match:
        option = match.group(1).strip().upper() 
        content = match.group(2).strip()
        return option, (content if content else None)
    else:
        return None, answer_text.strip()

def is_correct_answer(gpt_answer: str, response: str) -> bool:
    gpt_option, gpt_content = parse_option_and_content(gpt_answer)
    real_option, real_content = parse_option_and_content(response)
    
    if gpt_option and real_option:
        return gpt_option == real_option
    elif gpt_content and real_content:
        return gpt_content.strip(". ") == real_content.strip(". ")
    else:
        return str(gpt_answer).strip().lower() == str(response).strip().lower()

def extract_option_loose(text: str):
    patterns = [
        r'boxed\{([A-D])\}',
        r'Final Answer\s*[:：]\s*([A-D])',
        r'Answer\s*[:：]\s*([A-D])',
        r'\b([A-D])\b'
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    raw_output = results[0]
    
    cleaned_output = clean_json_string(raw_output)
    
    is_json, processed_answer = extract_json_from_text(cleaned_output)
    
    if is_json and isinstance(processed_answer, dict) and "answer" in processed_answer:
        model_answer = processed_answer["answer"]
    else:
        loose_opt = extract_option_loose(cleaned_output)
        model_answer = loose_opt if loose_opt else cleaned_output

    gt_response = doc.get("response", "")

    correct = is_correct_answer(model_answer, gt_response)

    question_text = doc_to_text(doc)

    return {
        "accuracy": float(correct),
        "question": question_text,      
        "raw_output": raw_output,     
        "ground_truth": gt_response
    }

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total_correct = sum(r["accuracy"] for r in results)
    total_count = len(results)
    return {
        "acc": total_correct / total_count if total_count > 0 else 0
    }
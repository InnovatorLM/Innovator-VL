import re
from PIL import Image

def doc_to_visual(doc):
    image = doc.get("image")
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return []

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    content = doc.get("problem", "")
    return f"{pre_prompt}{content}{post_prompt}"

def doc_to_target(doc, lmms_eval_specific_kwargs=None):
    full_answer = doc.get("answer", "")
    match = re.search(r"([A-D])", str(full_answer))
    return match.group(1) if match else full_answer

def extract_characters_regex(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    matches = re.search(r"\b([A-D])\b|(?<=\()([A-D])(?=\))", s.upper())
    if matches:
        return matches.group(1) if matches.group(1) else matches.group(2)
    for char in ["A", "B", "C", "D"]:
        if f" {char} " in f" {s.upper()} " or s.upper().startswith(char):
            return char
    return ""

def process_results(doc, results):
    prediction = results[0] if isinstance(results, list) else results
    pred_ans = extract_characters_regex(prediction)
    target_ans = doc_to_target(doc)
    question = doc_to_text(doc)
    is_correct = (pred_ans == str(target_ans)) if target_ans is not None else False
    return {
        "exact_match_accuracy": float(is_correct),
        "question": question,            
        "raw_output": prediction,        
        "ground_truth": target_ans
    }

def aggregation(results):
    return sum(results) / len(results) if results else 0.0
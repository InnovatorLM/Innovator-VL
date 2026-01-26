import re

def doc_to_visual(doc):
    return []

def doc_to_text(doc):
    instruction = (
        "Answer the multiple-choice question based solely on the provided context. \n"
        "If you are still unsure about the answer, output option 7.\n"
        "Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: \"The correct option is Option X.\"\n"
    )
    
    question = doc["question"]
    options_str = ""
    for option in doc["options"]:
        options_str += option + "\n"
    
    full_prompt = f"{instruction}\n Question: \n{question}\n Options: \n{options_str}\nThe correct option is:"
    return full_prompt

def doc_to_target(doc):
    answer_str = str(doc.get("answer", ""))
    match = re.search(r'\d+', answer_str)
    return match.group() if match else ""

def process_results(doc, results):
    prediction_text = str(results[0]).strip() if results else ""
    
    pred_num = "None"
    
    try:
        pred_match = re.search(r'\d+', prediction_text)
        if pred_match:
            pred_num = pred_match.group()
    except Exception:
        pred_num = "None"

    target = doc_to_target(doc)
    
    score = 1.0 if str(pred_num) == str(target) else 0.0
    
    return {
        "protein_lm_accuracy": score,
        "extracted_answer": pred_num,
        "ground_truth": target
    }
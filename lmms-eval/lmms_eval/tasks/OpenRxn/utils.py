import os
from typing import Dict, Any, List
from PIL import Image
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "default-model")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

def api_judge_answer(question: str, ground_truth: str, model_prediction: str) -> bool:
    """
    Use judge model API to judge if model prediction is correct
    """
    system_prompt = """You are a professional evaluation assistant. Please carefully compare whether the model's predicted answer matches the standard answer.

Evaluation criteria:
1. For chemical formulas/E-SMILES: Consider correct if structures are identical
2. For numerical answers: Consider correct if values are the same (allow minor differences in decimal places)
3. For text answers: Consider correct if semantics are the same
4. For Yes/No questions: Consider correct if the answer direction is consistent

Please only answer "correct" or "incorrect", do not explain the reasons."""

    user_prompt = f"""Question: {question}

Standard Answer: {ground_truth}
Model Prediction: {model_prediction}

Please judge whether the model prediction is correct? Only answer "correct" or "incorrect":"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        judgment = completion.choices[0].message.content.strip().lower()
        
        if judgment == "correct":
            return True
        elif judgment == "incorrect":
            return False
        else:
            print(f"Warning: Model returned unexpected judgment: '{judgment}'")
            return False
        
    except Exception as e:
        print(f"API judgment error: {e}")
        return False

def doc_to_visual(doc):
    image = doc.get("image")
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return []

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") if lmms_eval_specific_kwargs else ""
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") if lmms_eval_specific_kwargs else ""
    content = doc.get("question", "")
    return f"{pre_prompt}{content}{post_prompt}"

def doc_to_target(doc):
    return doc.get("answer", "")

def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    prediction = results[0] if isinstance(results, list) else results
    target = doc_to_target(doc)
    question = doc_to_text(doc)
    api_judge_correct = False
    try:
        api_judge_correct = api_judge_answer(question, target, prediction)
    except Exception as e:
        print(f"API judgment failed during process_results, using basic matching: {e}")
    return {
        "api_judge_accuracy": float(api_judge_correct),
        "question": question,
        "raw_output": prediction,
        "ground_truth": target
    }

def aggregation(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0
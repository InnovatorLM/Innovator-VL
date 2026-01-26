import re
import json
import numpy as np
import math
from loguru import logger

from lmms_eval.tasks.SmolInstruct_v2.metrics import (
    calculate_smiles_metrics, 
    calculate_formula_metrics, 
    calculate_text_metrics, 
    calculate_number_metrics, 
    calculate_boolean_metrics
)
from lmms_eval.tasks.SmolInstruct_v2.prediction_extraction import extract_pred



TASK_PROMPTS = {
    "property_prediction-hiv": "Analyze the SMILES string and predict its HIV activity. Answer strictly with 'Yes' or 'No'.",
    "property_prediction-bbbp": "Does this molecule penetrate the Blood-Brain Barrier (BBBP)? Answer strictly with 'Yes' or 'No'.",
    "property_prediction-clintox": "Predict the clinical toxicity of this molecule. Answer strictly with 'Yes' or 'No'.",
    "property_prediction-sider": "Does this drug cause the specific side effect? Answer strictly with 'Yes' or 'No'.",
    
    "property_prediction-esol": (
        "As a specialized chemist, calculate the logSol (aqueous solubility) for the molecule below. "
        "Consider the molecular weight, number of rotatable bonds, and aromatic proportion. "
        "Provide the final logSol value as a floating point number. Output: [value]"
    ),
    "property_prediction-lipo": (
        "Analyze the lipophilicity (logP) of this molecular structure. "
        "Focus on the hydrophobic and hydrophilic balance. "
        "Provide the numerical logP value only. Output: [value]"
    ),
    
    "forward_synthesis": "Predict the major product(s) for these reactants. Output the product(s) in SMILES format.",
    "retrosynthesis": "Suggest the starting materials (reactants) for the given product. If multiple reactants are needed, separate them with a period (.). Output only the SMILES strings.",
    
    "name_conversion-s2i": (
        "Convert this SMILES structure to its IUPAC name. "
        "Output only the IUPAC name, without any explanation or extra text."
    ),

    "name_conversion-i2s": (
        "Convert this IUPAC name to its SMILES structure. "
        "Output only the SMILES string, without any explanation or extra text."
    ),

    "name_conversion-s2f": (
        "Determine the molecular formula of this SMILES string. "
        "Output only the molecular formula, without any explanation or extra text."
    ),

    "name_conversion-i2f": (
        "Determine the molecular formula of this IUPAC name. "
        "Output only the molecular formula, without any explanation or extra text."
    ),

    "molecule_captioning": "Describe the following molecule's structure and chemical classification in detail:",
    "molecule_generation": "Generate the SMILES string for a molecule that fits this description:"
}

def extract_first_number(text):
    """从文本中提取第一个浮点数"""
    # 匹配包括负号和小数点的数字
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0
    return 0

def smol_doc_to_visual(doc):
    return []

def smol_doc_to_text(doc):
    """
    将 task-specific prompt + 原始输入 拼接成最终模型输入
    """
    task = doc["task"]
    user_input = doc.get("input", doc.get("question", "")).strip()

    prompt = TASK_PROMPTS.get(task, "").strip()

    if prompt:
        return f"{prompt}\n\n{user_input}"
    else:
        return user_input


def smol_doc_to_target(doc):
    return doc.get("answer", doc.get("output", ""))
def smol_process_results(doc, results):

    raw_output = str(results[0]).strip() if results[0] is not None else ""
    task = doc["task"]
    MODEL_TYPE = "llama" 
    

    def extract_between_tags_robust(text, tag_pattern):

        match_start = re.search(tag_pattern, text, re.IGNORECASE)
        if match_start:
            content_after = text[match_start.end():]
  
            close_idx = content_after.find('</')
            if close_idx != -1:
                return content_after[:close_idx].strip()
            else:
                return content_after.strip()
        return None


    temp_sample = {"output": [raw_output]}
    try:
        preds = extract_pred(temp_sample, MODEL_TYPE, task)
 
        if preds and preds[0] is not None:
            pred = preds[0]
        else:
            pred = ""
    except Exception as e:
        logger.error(f"Extraction failed for task {task}: {e}")
        pred = raw_output 


    SMILES_TASKS = [
        'forward_synthesis', 'retrosynthesis', 
        'molecule_generation', 'name_conversion-i2s'
    ]
    FORMULA_TASKS = [
        'name_conversion-s2f', 'name_conversion-i2f'
    ]


    found_tag = False 


    if task in SMILES_TASKS:
        extracted = extract_between_tags_robust(raw_output, r'<SMILES>|<SMILE>')
        if extracted:
            pred = extracted
            found_tag = True
            
    elif task in FORMULA_TASKS:

        extracted = extract_between_tags_robust(raw_output, r'<(?:MOL)?FORMULA>')
        if extracted:
            pred = extracted
            found_tag = True

  
    if not found_tag and (task in SMILES_TASKS or task in FORMULA_TASKS):
        if " " in pred:
            candidate = pred.split()[-1]
            
 
            if candidate.endswith('.') and not re.search(r'\.[a-zA-Z0-9]', candidate):
                 candidate = candidate.rstrip('.')
            

            if len(candidate) > 1:
                pred = candidate


    if task in ("property_prediction-esol", "property_prediction-lipo"):
        pred_num = extract_first_number(pred)
        pred = str(pred_num)
    

    if task in SMILES_TASKS and pred:

        pred = re.sub(r'<[^>]+>', '', pred).strip()
        pred = pred.replace(';', '.')


    if task == 'molecule_captioning':
        if pred is None:
            pred = ""
        else:
            pred = str(pred)

    return {
        "smol_metrics": {
            "pred": pred,
            "gold": doc["answer"],
            "task": task
        }
    }



def is_valid_smiles(s):
    if not isinstance(s, str) or len(s.strip()) == 0:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(s) is not None
    except:
        return False


def smol_aggregate_results(results):
    """
    收集所有样本的 pred 和 gold，统一调用官方提供的计算函数
    """
    task_groups = {}
    for res in results:

        data = res
        t = data["task"]
        if t not in task_groups:
            task_groups[t] = {"preds": [], "golds": []}
        

        task_groups[t]["preds"].append([data["pred"]])
        task_groups[t]["golds"].append([data["gold"]])

    final_scores = {}
    score = 0.0 
    for task, data in task_groups.items():
        preds = data["preds"]
        golds = data["golds"]
        
        try:

            if task in ('forward_synthesis', 'molecule_generation', 'name_conversion-i2s'):
                r = calculate_smiles_metrics(preds, golds)
                score = r.get('t1_morgan_fps', 0) 
            elif task == 'retrosynthesis':
                valid_preds = [
                    p[0] for p in preds
                    if isinstance(p, list)
                    and len(p) > 0
                    and is_valid_smiles(p[0])
                ]

                if len(valid_preds) == 0:
                    score = 0.0
                else:
                    r = calculate_smiles_metrics(
                        preds, golds,
                        metrics=('exact_match', 'fingerprint', 'multiple_match')
                    )
                    score = r.get('t1_morgan_fps', 0)

            
            elif task == 'molecule_captioning':
                r = calculate_text_metrics(preds, golds)
                score = r.get('rouge_1', 0)
            
            elif task in ('name_conversion-i2f', 'name_conversion-s2f'):
                r = calculate_formula_metrics(preds, golds, metrics=('element_match',))
                score = r.get('num_t1_ele_match', 0) / len(preds) if len(preds) > 0 else 0
            
            elif task == 'name_conversion-s2i':
                r = calculate_formula_metrics(preds, golds, metrics=('split_match',))
                score = r.get('num_t1_split_match', 0) / len(preds) if len(preds) > 0 else 0
            
            elif task in ('property_prediction-esol', 'property_prediction-lipo'):
                r = calculate_number_metrics(preds, golds)
                rmse_val = r.get('RMSE', None)
                if rmse_val is None or not isinstance(rmse_val, (int, float)) or math.isnan(rmse_val):
                    score = 0.0
                else:
                    score = math.exp(-rmse_val)
            
            elif task in ('property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider'):
                r = calculate_boolean_metrics(preds, golds)
                score = r.get('num_correct', 0) / len(preds) if len(preds) > 0 else 0
            
            else:
                score = 0.0

            final_scores[f"{task}_score"] = score
        except Exception as e:
            logger.error(f"Error calculating metric for {task}: {e}")
            raise   
            final_scores[f"{task}_score"] = 0.0
    return float(score)




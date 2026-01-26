import pandas as pd

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.wemath.wemath_utils import (
    calculate_metrics,
    compute_final_scores,
    process_steps_data,
    update_main_results_df,
)

THINKING_PROMPT = (
    "Think and solve the following question step by step. "
    "Please put your thinking and analysis procedure within <think></think>. "
    "Put ONLY your final answer within <answer></answer>."
)


def _get_base_question_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\n" + doc["option"]


def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question_content = _get_base_question_text(doc, lmms_eval_specific_kwargs)
    return f"{THINKING_PROMPT}\n<image 1>\n{question_content}"


def wemath_doc_to_visual(doc):
    return [doc["image_path"].convert("RGB")]


def wemath_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    
    question = _get_base_question_text(doc, None)
    extra_info = {"question": question}
    
    raw_response = results[0] if results else ""

    for pred in results:
        score_dict = compute_score(
            data_source="wemath", 
            solution_str=pred.strip(), 
            ground_truth=doc["answer"], 
            extra_info=extra_info
        )
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    data_dict = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "acc_score": acc_score,
    }

    return {
        # "wemath_loose": data_dict, 
        # "wemath_strict": data_dict, 
        "acc_score": acc_score / len(results) if results else 0.0, 
        "format_score": format_score / len(results) if results else 0.0,
        "raw_output": raw_response,
        "question": question
    }



def wemath_aggregate_results(results, metric_name):
    data = pd.DataFrame(results)
    data["joker"] = data["acc_score"] == 1.0
    data_2steps = data[data["key"].str.contains("2steps")]
    data_3steps = data[data["key"].str.contains("3steps")]
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    if metric_name == "wemath_loose":
        return score_dict["Score (Loose)"]
    elif metric_name == "wemath_strict":
        return score_dict["Score (Strict)"]
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")


def wemath_aggregate_results_loose(results):
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict(results):
    return wemath_aggregate_results(results, "wemath_strict")

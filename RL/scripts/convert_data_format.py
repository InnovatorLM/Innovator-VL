import pandas as pd
import ast
import argparse
import os
from tqdm import tqdm
import numpy as np

def process_row(row):
    new_row = row.copy()
    
    # 1. Extract answer from reward_model
    reward_model = row.get('reward_model')
    ground_truth = None
    
    if isinstance(reward_model, str):
        try:
            reward_model = ast.literal_eval(reward_model)
        except:
            pass
    
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get('ground_truth')
    
    if ground_truth:
        # Format as list containing string, consistent with old data
        new_row['answer'] = [str(ground_truth)]
    elif 'answer' not in new_row or new_row['answer'] is None:
        # Fallback if no reward_model or ground_truth found
        new_row['answer'] = []

    # 2. Extract problem from prompt list (user role content)
    prompt_data = row.get('prompt')
    if isinstance(prompt_data, str):
        try:
            prompt_data = ast.literal_eval(prompt_data)
        except:
            pass
            
    user_content = ""
    if isinstance(prompt_data, (list, np.ndarray)):
        for item in prompt_data:
            if isinstance(item, dict) and item.get('role') == 'user':
                user_content = item.get('content', "")
                break
    
    if user_content:
        # Add newline after <image> if present
        if "<image>" in user_content and not user_content.startswith("<image>\n"):
            user_content = user_content.replace("<image>", "<image>\n", 1)
        new_row['problem'] = user_content
    elif 'problem' not in new_row:
         new_row['problem'] = ""

    # 3. Set fixed fields
    new_row['answer_type'] = "judge"
    new_row['prompt_type'] = "long"
    
    # Ensure problem_type exists
    if 'problem_type' not in new_row:
        new_row['problem_type'] = "general"

    # Rename original 'prompt' key to 'vanilla_prompt' to avoid conflict with reward function argument
    if 'prompt' in new_row:
        new_row['vanilla_prompt'] = new_row.pop('prompt')

    return new_row

def convert_parquet(input_path, output_path):
    print(f"Reading {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return

    print(f"Loaded {len(df)} rows. Processing...")
    
    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        new_rows.append(process_row(row.to_dict()))
        
    new_df = pd.DataFrame(new_rows)
    
    # Columns to keep/ensure order
    # We keep original columns plus new/modified ones, but maybe prioritize standard ones
    # key_columns = ['id', 'images', 'problem', 'answer', 'problem_type', 'answer_type', 'source', 'prompt_type']
    # If you want to drop the raw columns like 'reward_model' or 'prompt' (the list one), do it here.
    # For now, keeping everything is safer unless size is an issue.
    
    print(f"Saving to {output_path}...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    new_df.to_parquet(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert new dataset format to standard training format.")
    parser.add_argument("input", help="Input parquet file path")
    parser.add_argument("output", help="Output parquet file path")
    args = parser.parse_args()
    
    convert_parquet(args.input, args.output)


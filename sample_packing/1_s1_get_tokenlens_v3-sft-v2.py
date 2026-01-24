#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from PIL import Image
import glob
import time
from multiprocessing import Process, Manager
from transformers import AutoProcessor
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from tool import cfg
import tempfile

# --- Worker-level functions ---

def process_file_batch(file_pairs, task_id, progress_queue, tmp_dir, model_checkpoint, max_token_len):
    """
    Worker function. It processes its assigned file pairs, calculates token lengths,
    and writes the results to a temporary file.
    """
    # Each worker initializes its own processor and template
    processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)
    
    # Using a simple string template for SFT as the original script's Jinja was complex
    # and this captures the essential logic for tokenization.
    sft_template_part1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    sft_template_part2 = "<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Create a unique temporary file for this worker's results
    worker_tmp_file = os.path.join(tmp_dir, f"worker_{task_id}_results.txt")
    
    with open(worker_tmp_file, 'w', encoding='utf-8') as f_out:
        processed_count = 0
        for json_path, img_path in file_pairs:
            try:
                with open(json_path, 'r', encoding='utf-8') as f_json:
                    json_data = json.load(f_json)
                
                # Reconstruct the text input from messages
                full_text = sft_template_part1
                for msg in json_data.get("messages", []):
                    content = msg.get("content", "").replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                    full_text += sft_template_part2.format(role=msg.get("role", ""), content=content)
                full_text += "<|im_start|>assistant\n"

                # Load image
                image = Image.open(img_path)

                # Process and get token length
                inputs = processor(text=[full_text], images=[image], return_tensors="pt")
                token_len = inputs["input_ids"].shape[1]

                if token_len <= max_token_len:
                    base_name = os.path.splitext(os.path.basename(json_path))[0]
                    f_out.write(f"{base_name}:{token_len}\n")

                processed_count += 1
                if processed_count % 100 == 0:
                    progress_queue.put({'task_id': task_id, 'advance': 100})
            
            except Exception:
                # Silently skip files that cause errors
                continue
    
    # Report remaining progress and signal completion
    progress_queue.put({'task_id': task_id, 'advance': processed_count % 100, 'completed': True})
    return worker_tmp_file

# --- Main orchestrator ---

def main(workers):
    # Get config values
    DATA_DIR = cfg['data']['directory']
    MODEL_CHECKPOINT = cfg['model']['checkpoint']
    MAX_TOKEN_LEN = cfg['sample']['max_len']
    FINAL_OUTPUT_FILE = os.path.join(DATA_DIR, 'part_00', 'token_info.txt') # Assuming part_00 for output
    TMP_BASE_DIR = "/mnt/innovator/data/wenzichen/tmp_processing" # Use a fast, large disk

    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    os.makedirs(TMP_BASE_DIR, exist_ok=True)
    
    print("Finding all paired json/jpg files...")
    # Find all json files, assume a matching jpg exists
    all_json_files = glob.glob(os.path.join(DATA_DIR, 'part_00', '*.json'), recursive=False)
    
    file_pairs = []
    for json_path in all_json_files:
        img_path = json_path.replace('.json', '.jpg')
        if os.path.exists(img_path):
            file_pairs.append((json_path, img_path))
            
    if not file_pairs:
        print("No paired files found. Exiting.")
        return

    print(f"Found {len(file_pairs)} paired files.")
    
    file_chunks = np.array_split(file_pairs, workers)
    file_chunks = [chunk for chunk in file_chunks if len(chunk) > 0]
    
    with tempfile.TemporaryDirectory(prefix="token_len_", dir=TMP_BASE_DIR) as tmp_dir, \
         Manager() as manager, \
         Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
         ) as progress:
        
        print(f"Using temporary directory for worker outputs: {tmp_dir}")
        progress_queue = manager.Queue()
        processes = []
        worker_output_files = []

        print(f"Starting {len(file_chunks)} worker processes to calculate token lengths...")
        for i, chunk in enumerate(file_chunks):
            task_id = progress.add_task(f"Worker {i+1}", total=len(chunk))
            p = Process(
                target=process_file_batch,
                args=(chunk, task_id, progress_queue, tmp_dir, MODEL_CHECKPOINT, MAX_TOKEN_LEN)
            )
            processes.append(p)
            p.start()

        completed_workers = 0
        while completed_workers < len(processes):
            update = progress_queue.get() # This will block until a message is available
            progress.update(update['task_id'], advance=update['advance'])
            if update.get('completed', False):
                completed_workers += 1
        
        for p in processes:
            p.join()
        
        print("\nAll workers finished. Merging results...")
        
        # Collect all worker output files
        worker_output_files = glob.glob(os.path.join(tmp_dir, "worker_*_results.txt"))

        # Merge all temporary files into the final output file
        # This is a simple but potentially large merge.
        # For extreme scale, a more sophisticated merge might be needed.
        all_results = []
        for temp_file in worker_output_files:
            with open(temp_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        name, length_str = line.strip().split(':')
                        all_results.append((int(length_str), name))
                    except ValueError:
                        continue
        
        # Sort by token length
        all_results.sort(key=lambda x: x[0])
        
        with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f_final:
            for length, name in all_results:
                f_final.write(f"{name}:{length}\n")

        print(f"Processing complete. Final token info file created at: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    # Recommended to use a number of workers close to the number of CPU cores
    main(32)


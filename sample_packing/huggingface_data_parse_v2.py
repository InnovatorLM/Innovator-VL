from datasets import load_dataset
from tool import cfg,get_ip_info,get_init_file
import os
import json
import re
import numpy as np
from PIL import Image
import io
import glob
import random
import shutil
import tempfile
from multiprocessing import Process, Manager
from functools import partial
# This new implementation requires the 'rich' library for multi-progress bar display.
# If not installed, please run: pip install rich
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

# --- Worker-level functions ---

def check_caption(content: str) -> bool:
    if content.lower().startswith(("i'm sorry", "i am sorry", "i cannot", "i can't")):
        return False
    words = re.findall(r'\b\w+\b', content.lower())
    if len(words) >= 8:
        for i in range(len(words) - 7):
            if len(set(words[i:i+8])) == 1:
                return False
    if len(content) > 3500 or len(content) < 50:
        return False
    return True

def check_image_in_memory(img: Image.Image) -> bool:
    """Checks a PIL Image object in memory to avoid redundant file I/O."""
    try:
        if img is None:
            return False
        img_array = np.array(img)
        if np.all(img_array == 0):
            return False
        return True
    except Exception:
        return False

def process_file_chunk(file_paths, task_id, progress_queue, dst_dir, do_filter_caption, do_filter_image):
    """
    Worker function. It processes its assigned files and reports progress
    back to the main process via a shared queue.
    """
    local_processed_count = 0
    try:
        dataset = load_dataset("parquet", data_files=list(file_paths), split="train", streaming=True)
    except Exception as e:
        print(f"Worker {task_id} failed to load dataset with error: {e}")
        return

    for item in dataset:
        try:
            if do_filter_caption and not check_caption(str(item['caption'])):
                continue
            
            pil_image = item['image']
            if do_filter_image and not check_image_in_memory(pil_image):
                continue
            
            name = str(item['id']).replace('/', '_')
            name = os.path.splitext(name)[0]

            image_path = os.path.join(dst_dir, name + '.jpg')
            pil_image.save(image_path)
            
            json_data = {
                "messages": [
                    {"content": "<image>", "role": "user"},
                    {"content": str(item['caption']), "role": "assistant"}
                ],
                "images": [image_path]
            }
            json_path = os.path.join(dst_dir, name + '.json')
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=None)
            
            local_processed_count += 1
            # Report progress back to the main process every 100 items to avoid flooding the queue.
            if local_processed_count % 100 == 0:
                progress_queue.put({'task_id': task_id, 'advance': 100})
        except Exception:
            pass
    
    # Report any remaining progress
    if local_processed_count % 100 != 0:
        progress_queue.put({'task_id': task_id, 'advance': local_processed_count % 100})

# --- Main orchestrator ---

def main(workers):
    data_path = cfg['hf_data']
    _, _, _, _, FINAL_DESTINATION_DIR = get_init_file()

    # --- CHANGE: Ensure the final destination directory exists from the start ---
    os.makedirs(FINAL_DESTINATION_DIR, exist_ok=True)
    print(f"All workers will write directly to final destination: {FINAL_DESTINATION_DIR}")

    print("Finding all parquet files for sampling...")
    all_files = glob.glob(os.path.join(data_path, '**/*.parquet'), recursive=True)
    all_files.sort()
    print(f"Found {len(all_files)} files.")

    total_samples = 85_000_000
    target_samples = cfg['data']['max_samples']
    fraction_to_sample = target_samples / total_samples
    
    num_files_to_sample = int(len(all_files) * fraction_to_sample)
    if num_files_to_sample == 0 and len(all_files) > 0:
        num_files_to_sample = 1

    print(f"Randomly sampling {num_files_to_sample} files out of {len(all_files)}...")
    random.seed(42)
    sampled_files = random.sample(all_files, num_files_to_sample)
        
    file_chunks = np.array_split(sampled_files, workers)
    file_chunks = [chunk for chunk in file_chunks if len(chunk) > 0]
    
    estimated_total_per_worker = target_samples / len(file_chunks) if file_chunks else 0
    
    # --- CHANGE: Removed the tempfile.TemporaryDirectory context manager ---
    with Manager() as manager, \
         Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TransferSpeedColumn(),
         ) as progress:
        
        progress_queue = manager.Queue()
        processes = []

        print(f"Starting {len(file_chunks)} worker processes...")
        for i, chunk in enumerate(file_chunks):
            task_id = progress.add_task(f"Worker {i+1}", total=estimated_total_per_worker)
            p = Process(
                target=process_file_chunk,
                args=(
                    chunk, task_id, progress_queue, FINAL_DESTINATION_DIR, # <-- Pass the final directory
                    cfg['data']['filter_with_caption'],
                    cfg['data']['filter_with_image']
                )
            )
            processes.append(p)
            p.start()

        # Main process becomes the progress monitor
        active_workers = len(processes)
        while active_workers > 0:
            while not progress_queue.empty():
                update = progress_queue.get()
                progress.update(update['task_id'], advance=update['advance'])
            
            active_workers = sum(1 for p in processes if p.is_alive())
            time.sleep(0.1)
        progress.stop()

        for p in processes:
            p.join()

        # --- CHANGE: Removed the final copytree and related print statements ---
        print("Processing complete. All files have been written directly to the final destination.")

if __name__=="__main__":
    import time
    main(32)

from datasets import load_dataset
from multiprocessing import Pool
from tool import cfg,get_ip_info,get_init_file
import os
from functools import partial
from tqdm import tqdm
import json
import re
import numpy as np
from PIL import Image
import io
import glob
import random
import shutil
import tempfile
from itertools import islice

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
        # Check if the image is all black
        if np.all(img_array == 0):
            return False
        return True
    except Exception:
        return False

def process_chunk(chunk, ip_indx, ip_num, dst_dir, do_filter_caption, do_filter_image):
    """Processes a chunk of data items, designed to be called by a worker process."""
    processed_count = 0
    for index, item in chunk:
        try:
            if index % ip_num != ip_indx:
                continue

            # --- Optimization: In-memory validation before any I/O ---
            if do_filter_caption and not check_caption(item['caption']):
                continue
            
            pil_image = item['image']
            if do_filter_image and not check_image_in_memory(pil_image):
                continue
            
            # Defensive casting to prevent potential type issues from the source data
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
            
            processed_count += 1
        except Exception:
            # Silently skip items that cause exceptions to keep the pipeline running.
            pass
    return processed_count

def chunked(iterable, size):
    """Helper function to break an iterator into chunks of a specific size."""
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            return
        yield chunk

def main(workers):
    data_path = cfg['hf_data']
    # Renamed DEFAULT_DIRECTORY for clarity
    _, _, _, _, FINAL_DESTINATION_DIR = get_init_file()

    print("Finding all parquet files for sampling...")
    # Use a recursive glob pattern ('**') to ensure all .parquet files are found,
    # regardless of their subdirectory depth. This is more robust.
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
    
    print("Loading the sampled dataset in streaming mode...")
    dataset = load_dataset('parquet', data_files=sampled_files, split="train", streaming=True)

    # Use a robust iterator that explicitly casts types to prevent pyarrow issues
    data_iter = ((i, {'id': item['id'], 'caption': item['caption'], 'image': item['image']}) 
                 for i, item in enumerate(dataset))
    
    ip_indx, ip_num, _ = get_ip_info()
    
    # --- Major Optimization: Write to local temp dir and process in chunks ---
    with tempfile.TemporaryDirectory(prefix="huggingface_parse_") as tmp_dir:
        print(f"Using temporary directory for writes: {tmp_dir}")
        
        CHUNK_SIZE = 500  # Process 500 samples per chunk to reduce IPC overhead
        
        # Pre-set the constant arguments for the worker function
        worker_func = partial(process_chunk, 
                              ip_indx=ip_indx, 
                              ip_num=ip_num, 
                              dst_dir=tmp_dir,
                              do_filter_caption=cfg['data']['filter_with_caption'],
                              do_filter_image=cfg['data']['filter_with_image'])
        
        with Pool(processes=workers) as pool, tqdm(total=target_samples, desc="Processing samples") as bar:
            # map the worker function to the chunked data iterator
            for count in pool.imap_unordered(worker_func, chunked(data_iter, CHUNK_SIZE)):
                bar.update(count)

        print(f"Processing complete. Copying files from {tmp_dir} to {FINAL_DESTINATION_DIR}...")
        shutil.copytree(tmp_dir, FINAL_DESTINATION_DIR, dirs_exist_ok=True)
        print("File copy complete.")

if __name__=="__main__":
    main(128)
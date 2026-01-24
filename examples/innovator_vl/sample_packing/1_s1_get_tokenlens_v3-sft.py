#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import threading
import logging
import psutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import merge
from jinja2 import Template
from transformers import AutoProcessor
from qwen_vl_utils import fetch_image
from queue import Empty
import multiprocessing
from multiprocessing import Pool, Manager, Value
from tqdm import tqdm
from tool import get_ip_info,cfg

# Declares a global cross-process counter (defined in the main module for child processes to inherit)
global_total_counter = None
MAX_TOKEN_LEN = cfg['sample']['max_len']
task_type = cfg['sample']['task_type']
DEFAULT_DIRECTORY = Path(cfg['data']['directory'])
try:
    ip_index,_,_=get_ip_info(cfg['hosts'])
    print('success init ip ,>>>>>>>>>>>>>>>>>>>>>>>>')
except:
    print(f"getting ip_index error, default to 0")
    ip_index=0
DEFAULT_DIRECTORY=Path(os.path.join(DEFAULT_DIRECTORY,f'part_{ip_index:02d}'))
save_files_dir=os.path.join(DEFAULT_DIRECTORY,"save_files")
if os.path.exists(save_files_dir) is False:
    os.makedirs(save_files_dir)

OUTPUT_FILE = Path(cfg['data']['output_base'])
TOKEN_INFO_FILE = Path(cfg['data']['output_token'])
OUTPUT_FILE=os.path.join(save_files_dir,OUTPUT_FILE)
TOKEN_INFO_FILE=os.path.join(save_files_dir,TOKEN_INFO_FILE)


CKPT_DIR = cfg['model']['checkpoint']
MIN_PIXELS = cfg['image']['min_pixels']
MAX_PIXELS = cfg['image']['max_pixels']
TIME_OUT = cfg['processing']['time_out']
# Merging parameters (Two levels only: stage0 → stage1)
STAGE1_CHUNK = cfg['processing']['stage1_merge_chunk']
chunk_size = cfg['processing']['chunk_size']
n_workers = cfg['processing']['n_workers']
MIN_WORKERS = cfg['processing']['min_workers']
MAX_WORKERS = cfg['processing']['max_workers']
use_shm = cfg['logging']['use_shm']
log_level = cfg['logging']['level']
log_name = cfg['logging']['file']
log_file=os.path.join(save_files_dir,'logs')
if os.path.exists(log_file) is False:
    os.makedirs(log_file)
log_file=os.path.join(log_file,log_name) 

file_handler = logging.FileHandler(
    log_file,
    delay=True,
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

EXTENSIONS = (".json", ".jpg")


temp_dir = '/dev/shm' if use_shm else None  # None indicates using the system default temporary directory

def count_lines(file_path):
    """Counts the number of valid lines in a file (non-empty and containing the delimiter)"""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip() and ':' in line.strip())
    except Exception as e:
        logger.error(f"❌ Failed to count lines in file {file_path}: {str(e)}")
        return 0

def find_paired_files(directory):
    json_set=set()
    img_set=set()
    with os.scandir(directory) as entries:
        for entry in tqdm(entries,total=10000000):
            if entry.name.endswith('.json'):
                json_set.add(entry.name[:-5])
            if entry.name.endswith('.jpg'):
                img_set.add(entry.name[:-4])
    paired = json_set & img_set
    logger.info(f"Found {len(paired)} pairs of matched files")

    return paired


def write_base_names_to_file(base_names, output_file):
    """Writes paired filenames to a file"""
    try:
        content = "\n".join(sorted(base_names)) + "\n"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"ℹ️ Written {len(base_names)} paired filenames to {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to write to {output_file}: {str(e)}")
        raise


def read_lines_in_chunks(file_path, chunk_size):
    """Reads file content by chunks"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = [line.strip() for _, line in zip(range(chunk_size), f) if line.strip()]
            if not chunk:
                break
            logger.info(f"ℹ️ Read a data chunk containing {len(chunk)} samples")
            yield chunk


if task_type=="pretrain":
    CAP_TEMPLATE = Template("<|vision_start|><|image_pad|><|vision_end|>{{ captions[0].content }}<|im_end|>")
elif task_type=="sft":
    chat_template  = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] | replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>') }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""
    CAP_TEMPLATE = Template(chat_template)
    pass

def process_sample(json_path, img_path, processor):
    """Process a single sample and return (token_len, file name)"""
    try:
        if not Path(json_path).exists():
            raise FileNotFoundError(f"❌ The JSON file does not exist: {json_path}")
        if not Path(img_path).exists():
            raise FileNotFoundError(f"❌ The image file does not exist: {img_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if task_type=="pretrain":
            txt_input = CAP_TEMPLATE.render(captions=json_data['captions'])
        elif task_type=="sft":
            txt_input = CAP_TEMPLATE.render(json_data)
        img_input = fetch_image({
            'type': 'image',
            'image': img_path,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        })
        base_name = Path(img_path).stem
        inputs = processor(
            text=[txt_input],
            images=img_input,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return (inputs["input_ids"].shape[1], base_name)

    except Exception as e:
        return (None, f"❌ Failed processing [{Path(img_path).stem}]: {str(e)}")


def get_adaptive_workers(min_workers=20, max_workers=96):
    """Adjust the number of threads according to the system load"""
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        if cpu_usage > 80 or mem_usage > 85:
            adjusted = max(min_workers, max_workers // 2)
            logger.info(f"The system load is too high. Adjust the number of threads to {adjusted} (CPU: {cpu_usage}%, Memory: {mem_usage}%)")
            return adjusted
        return max_workers
    except Exception as e:
        logger.warning(f"Failed to obtain the system load. Use the default number of threads {max_workers}: {str(e)}")
        return max_workers

gt_maxlen=0
def merge_files_by_token(input_files, output_file, max_token=MAX_TOKEN_LEN):
    """Merge multiple sorted files, filter out the data greater than max_token in ascending order of token_len, and return (output path, number of data entries)"""
    if not input_files:
        logger.warning("⚠️ There are no files to merge")
        return (None, 0)

    valid_files = []
    total_lines = 0
    for f in input_files:
        line_count = count_lines(f)
        if line_count > 0:
            valid_files.append(f)
            total_lines += line_count
            logger.debug(f"ℹ️ file to be merged {os.path.basename(f)} contains {line_count} data ")
        else:
            logger.warning(f"⚠️ file {os.path.basename(f)} is empty or invalid, skip ")

    if not valid_files:
        return (None, 0)

    def sort_key(line):
        # _, token_str = line.strip().split(':', 1)
        token_str = line.strip().split(':')[-1]
        return int(token_str)

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Create an iterator for all files
            iterators = []
            file_handles = []
            for fpath in valid_files:
                try:
                    fh = open(fpath, 'r', encoding='utf-8')
                    file_handles.append(fh)
                    iterators.append(((sort_key(line), line) for line in fh))
                except Exception as e:
                    logger.error(f"❌ failed to open the file {os.path.basename(fpath)} : {str(e)}")

            # Merge sort and write, filtering out rows > max_token (other conditions can be added later)
            filtered_max_len = 0
            for _, line in merge(*iterators, key=lambda x: x[0]):
                _, token_str = line.strip().split(':', 1)
                if int(token_str) <= max_token:   
                    out_f.write(line)
                else:
                    logger.warning(f"⚠️ token length: {token_str} > {max_token}: eliminated!!")
                    filtered_max_len+=1
                    gt_maxlen

            # Close all file handles
            for fh in file_handles:
                try:
                    fh.close()
                except Exception as e:
                    logger.warning(f"⚠️ Close the file {fh.name} failure: {str(e)}")

        output_lines = count_lines(output_file)+filtered_max_len
        if output_lines != total_lines: 
            logger.error(f"❌ The merged data is lost! Enter {total_lines} ,output {output_lines} ,the erroneous file has been deleted")
            if os.path.exists(output_file):
                os.remove(output_file)
            return (None, 0)
        else:
            logger.info(f"✅ 📊 The merge was successful. Enter {total_lines} ,output {output_lines-filtered_max_len} (token ≤ {max_token}) datas")

        return (output_file, output_lines-filtered_max_len)
    except Exception as e:
        logger.error(f"❌ File merge failed: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed file deletion {output_file}: {str(e)}")
        return (None, 0)


def stage1_merger(input_queue, chunk_size, stage1_files, stop_event):
    """
    Fix stage1 merge threads
    - Ensure that all stage0 files are merged, including those with less than 10 files at the end
    - Solve the problems of thread timeout and data loss
    """
    buffer = []
    batch_counter = 0
    logger.info(f"💡 The stage1 merge thread starts and merges every {chunk_size} stage0 file")

    try:
        # Loop condition: There is a file in the queue or in the buffer or no stop signal has been received
        while (not input_queue.empty()) or buffer or (not stop_event.is_set()):
            # Retrieve files from the queue (with timeout to prevent permanent blocking)
            if not input_queue.empty():
                try:
                    file_path = input_queue.get(timeout=1) 
                    buffer.append(file_path)
                    input_queue.task_done()
                    logger.debug(f"ℹ️ stage1 receives file {os.path.basename(file_path)}, current buffer: {len(buffer)}/{chunk_size}")
                    # The merge will be executed when the number of merges is reached
                    if len(buffer) >= chunk_size:
                        batch_counter += 1
                        merged_file = tempfile.NamedTemporaryFile(
                            mode='w', delete=False,
                            prefix=f"stage1_batch{batch_counter:03d}_",
                            encoding='utf-8',
                            dir=temp_dir
                        ).name
                        
                        # Carry out the merger
                        merged_path, line_count = merge_files_by_token(buffer, merged_file)
                        if merged_path and line_count > 0:
                            stage1_files.append(merged_path)
                            logger.info(f"📊 stage1 batch {batch_counter} completed: {os.path.basename(merged_path)},containing {line_count} pieces of data(combined with{len(buffer)} files)")
                        else:
                            logger.warning(f"⚠️ stage1 batch {batch_counter} merge failed, skip this batch ")
                        # Clear the buffer
                        buffer = []
                except Empty:
                    continue 
                except Exception as e:
                    logger.error(f"❌ Error in stage1 file processing: {str(e)}", exc_info=True)
            else:
                # When the queue is empty, check whether it is necessary to forcibly merge the remaining files
                if buffer and stop_event.is_set():
                    # When a stop signal is received and there are files in the buffer, force a merge
                    batch_counter += 1
                    merged_file = tempfile.NamedTemporaryFile(
                        mode='w', delete=False,
                        prefix=f"stage1_remaining_batch{batch_counter:03d}_",
                        encoding='utf-8',
                        dir=temp_dir
                    ).name
                    
                    merged_path, line_count = merge_files_by_token(buffer, merged_file)
                    if merged_path and line_count > 0:
                        stage1_files.append(merged_path)
                        logger.info(f"📊 stage1 remaining files merged completed: {os.path.basename(merged_path)}, containing {line_count} pieces of data (merged {len(buffer)} files) ")
                    else:
                        logger.warning(f"❌ stage1 remaining file merge failed, data may be lost ")
                    buffer = []
                else:
                    # Short hibernation to reduce CPU usage
                    threading.Event().wait(0.5)

        if buffer:
            logger.error(f"❌ stage1 there are still {len(buffer)} files in the buffer that have not been processed when the thread exits! Data loss")

    except Exception as e:
        logger.error(f"❌ stage1 thread exits abnormally: {str(e)}", exc_info=True)
    finally:
        logger.info(f"📊 stage1 thread exits, generating a total of {len(stage1_files)} files ")

def process_chunk(args):
    """
    The processing logic of a single process: Handle a large chunk, and internally use multi-threading in parallel
    Args:
        args: A tuple containing parameters such as chunk data, processor configuration, and queue
    """
    # Obtain the counter from the global variable, not the parameter
    global global_total_counter
    
    chunk_idx, chunk, ckpt_dir, min_pixels, max_pixels, stage0_queue = args
    processor = None
    processed_count = 0 
    
    
    try:
        # Each process initializes the processor independently (Processor instances cannot be shared among processes)
        processor = AutoProcessor.from_pretrained(
            ckpt_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
            use_fast=False
        )
        
        full_paths = []
        for fn in chunk:
            full_paths.append(str(DEFAULT_DIRECTORY / f"{fn}.json"))
            full_paths.append(str(DEFAULT_DIRECTORY / f"{fn}.jpg"))
        
        n_samples = len(chunk)
        logger.info(f"👉 process {multiprocessing.current_process().name} begins processing block {chunk_idx} containing {n_samples} samples ")
        
        # Create a thread pool within the process (reuse threads)
        n_workers = get_adaptive_workers(min_workers=MIN_WORKERS, max_workers=MAX_WORKERS) 
        chunk_results = []
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix=f"proc-{multiprocessing.current_process().pid}-thread"
        ) as executor:
            tasks = [
                executor.submit(
                    process_sample,
                    full_paths[idx*2],
                    full_paths[idx*2+1],
                    processor
                ) for idx in range(n_samples)
            ]
            
            # Collect the results of thread tasks
            for future in as_completed(tasks):
                try:
                    token_len, name = future.result()
                    if token_len is not None:
                        chunk_results.append((token_len, name))
                        processed_count += 1 
                    else:
                        logger.warning(name)
                except Exception as e:
                    logger.error(f"❌ in-process task error: {str(e)}")
        
        # Write to the stage0 file and place it in the cross-process queue
        if chunk_results:
            chunk_results_sorted = sorted(chunk_results, key=lambda x: x[0])
            with tempfile.NamedTemporaryFile(
                mode='w+', delete=False,
                prefix=f"stage0_chunk{chunk_idx:03d}_",
                encoding='utf-8',
                dir=temp_dir  
            ) as f:
                stage0_file = f.name
                for token_len, name in chunk_results_sorted:
                    f.write(f"{name}:{token_len}\n")
            
            line_count = count_lines(stage0_file)
            stage0_queue.put(stage0_file)  
            proc_status = "🟢" if processed_count==n_samples else "🟡"
            logger.info(f"{proc_status} process {multiprocessing.current_process().name} completion block {chunk_idx} Valid samples {processed_count}/{n_samples}")
            
            with global_total_counter.get_lock():
                global_total_counter.value += processed_count
                
            return stage0_file 
        
    except Exception as e:
        logger.error(f"❌ process {multiprocessing.current_process().name} failed: {str(e)}")
    finally:
        if processor:
            del processor
    return None

def main():
    global global_total_counter  
    processor = None  
    stage0_files = []  
    stage1_files = []

    try:

        logger.info(f"💡 --------------Start the data processing flow--------------")
        # Search for the paired file and write it to a temporary file (a sample with the same json and jpg file names)
        base_names = find_paired_files(DEFAULT_DIRECTORY)    # DEFAULT_DIRECTORY is the location where the original data is stored (jpg and json).
        total_original = len(base_names) 
        logger.info(f"👉 finds {total_original} for the original sample file ")
        if total_original == 0:
            logger.warning("⚠️ no original sample, exit program ")
            return
        # Write the paired file names to the file for subsequent block reading
        write_base_names_to_file(base_names, OUTPUT_FILE)
        
        # Initialize the cross-process queue (used to pass the stage0 file path to the merge thread)
        manager = Manager()  
        stage0_queue = manager.Queue()
        stop_event = manager.Event() 

        # Cross-process counter, used to count the total number of processed samples (initial value 0)
        global_total_counter = Value('i', 0)  

        # Start the stage1 merge thread (daemon thread)
        stage1_thread = threading.Thread(
            target=stage1_merger,
            args=(stage0_queue, STAGE1_CHUNK, stage1_files, stop_event),
            daemon=True
        )
        stage1_thread.start()
        logger.info("💡 stage1 merge thread started ")

        # Read the file in chunks and start multiple processes to handle each chunk
        all_chunks = list(read_lines_in_chunks(OUTPUT_FILE, chunk_size))
        total_chunks = len(all_chunks)
        n_processes = min(multiprocessing.cpu_count(), total_chunks)
        logger.info(f"👉 is divided into {total_chunks} blocks and starts {n_processes} processes to handle ")

        # Prepare the arguments for each process
        process_args = [
            (
                idx + 1,  
                chunk,    
                CKPT_DIR, 
                MIN_PIXELS,
                MAX_PIXELS,
                stage0_queue, 
            ) for idx, chunk in enumerate(all_chunks)
        ]
        # Start the process pool (it is recommended to set the number of processes to 1 to 2 times the number of CPU cores)
        with Pool(processes=n_processes) as process_pool:
            result = process_pool.map_async(process_chunk, process_args)
            try:
                stage0_files = result.get(timeout=TIME_OUT) 
            except multiprocessing.TimeoutError:
                logger.error("❌ some processes timeout, forced termination ")
                process_pool.terminate()
        
        stage0_files = [f for f in stage0_files if f is not None]
        logger.info(f"✅ all processes completed, generating {len(stage0_files)} stage0 files ")  
        total_processed = global_total_counter.value 
        logger.info(f"👉 number of original samples: {total_original} number of processed samples: {total_processed}")

        # 验证数据完整性
        if total_processed != total_original:
            logger.warning(f"❌ data incomplete! Original {total_original}, valid processed {total_processed}, difference {total_original - total_processed}")
        else:
            logger.info("✅ data integrity verification passed, all samples processed effectively ")

        logger.info("🔄 wait for stage0 queue processing to complete...")
        stage0_queue.join() 
        logger.info("💡 stage0 queue all files processed ")
        logger.info("💡 notifies the stage1 thread to stop and process the remaining files..." )
        stop_event.set()

        timeout_counter = 0
        while stage1_thread.is_alive() and timeout_counter < 60:
            logger.debug(f"🔄 wait for the stage1 thread to complete ({timeout_counter}/60 seconds) ")
            threading.Event().wait(1) 
            timeout_counter += 1
        
        if stage1_thread.is_alive():
            logger.warning("⚠️ the stage1 thread has timed out and has not exited. There may be an exception (but an attempt has been made to force merge the remaining files) ")
        else:
            logger.info("💡 stage1 thread has exited normally ")

        # Verify whether the number of stage1 files matches (merge 1 for every 10 stage0 files, and even if there are less than 10, it still counts as 1)
        expected_stage1_count = (len(stage0_files) + STAGE1_CHUNK - 1) // STAGE1_CHUNK
        if len(stage1_files) != expected_stage1_count:
            logger.warning(f"⚠️ ℹ️ abnormal number of stage1 files! Expected {expected_stage1_count}, actual {len(stage1_files)}")
        else:
            logger.info(f"✅ stage1 file count verified: {len(stage1_files)} ")

        if not stage1_files:
            logger.warning("⚠️ did not generate stage1 file, check if there is an error in intermediate processing ")
            return

        stage1_total = sum(count_lines(f) for f in stage1_files)
        logger.info(f"ℹ️ starts the final merge: {len(stage1_files)} stage1 files, total data volume: {stage1_total} pieces ")

        # Merge into the final file
        final_path, final_lines = merge_files_by_token(stage1_files, TOKEN_INFO_FILE)

        if final_path and final_lines > 0:
            logger.info(f"✅ final result file generation completed: {TOKEN_INFO_FILE}, containing {final_lines} data ")
            if final_lines != total_processed:
                logger.error(f"❌ inconsistent data volume! Total processed data {total_processed}, final file {final_lines}")
            else:
                logger.info("✅💡 data volume verification passed, all data correctly written to the final file ")
        else:
            logger.error("❌ final file merge failed ")

        if os.path.exists(TOKEN_INFO_FILE):
            final_count = count_lines(TOKEN_INFO_FILE)
            logger.info(f"ℹ️ final result file contains {final_count} pieces of data ")
            if final_count != total_processed:
                logger.error(f"❌ the final file data is incomplete! Process the {total_processed} entry and the final file {final_count} entry")
            else:
                logger.info("✅ final file data integrity verification passed ")

    except Exception as e:
        logger.error(f"❌ main process error: {str(e)}", exc_info=True)
    finally:
        if processor:
            del processor

        stop_event.set()

        if stage1_thread and stage1_thread.is_alive():
            stage1_thread.join(timeout=2)        
        
        threading.Event().wait(2)

        all_temp_files = stage0_files + stage1_files
        for fpath in all_temp_files:
            if fpath != str(TOKEN_INFO_FILE) and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.debug(f" cleared temporary files: {os.path.basename(fpath)}")
                except Exception as e:
                    logger.warning(f"failed to clear the temporary files {os.path.basename(fpath)}: {str(e)}")

        logger.info(" Program Execution completed ")

if __name__ == "__main__":
    main()


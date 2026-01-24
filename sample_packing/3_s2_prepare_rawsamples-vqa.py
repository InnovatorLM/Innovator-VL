import bisect
import os
import json
import sys
from typing import Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tool import get_init_file

input_token_file ,MAX_TOKEN_LEN,save_files_dir,big_dir,DEFAULT_DIRECTORY= get_init_file()
SRC_DIR_IMGS = DEFAULT_DIRECTORY   # The storage location of image data
SRC_DIR_JSONS = DEFAULT_DIRECTORY   # The storage location of json data
SRC_DST_EXTENSIONS = ("jpg", "json")
f_toklens_originalsample = input_token_file
PACKED_LENGTH = MAX_TOKEN_LEN
dst_dir_json = os.path.join(save_files_dir,'row_packing_jsons')
dst_dir_image = os.path.join(save_files_dir,'row_packing_images')
if os.path.exists(dst_dir_json) is False:
    os.makedirs(dst_dir_json)
if os.path.exists(dst_dir_image) is False:
    os.makedirs(dst_dir_image)
MAX_WORKERS = 96 

task_type = "sft" 

f_TEST=False     
n_packed_samples=100  

PROMPTS = [
    "What about this picture?",
    "Please provide a vivid description of the image.",
    "Please Depict the image in words."
    "Could you please transcribe thr image into a descriptive paragraph?"
    "What is the content of this figure?",
    "What do you see here?",
    "Tell me about this image.",
    "What's going on in this artwork?",
    "What is depicted in this painting?",
    "What is the subject matter here?",
    "What can you make out in this picture?",
    "What's the main thing shown in this image?",
    "What's the gist of this artwork?",
    "What's the essence of this figure?",
    "What's the general idea here?",
    "What does this image show?",
    "What's the core element in this painting?",
    "What's the overview of this scene?",
    "What's the primary focus of this artwork?",
    "What's the fundamental subject matter?",
    "What's the general view presented?",
    "What's the main impression given by this picture?",
    "What's the central theme shown?",
    "What's the overall presentation here?",
    "What's the key element you notice?",
    "What's the fundamental concept in this image?",
    "What's the overall content?",
    "What's the main thing you get from this?",
    "What's the general subject?",
    "What's the core idea conveyed?",
    "What's the basic representation?",
    "What's the main point of this figure?"
]

import random

def get_random_prompts(prompts, n):
    if n > len(prompts):
        return random.choices(prompts, k=n)
    else:
        return random.sample(prompts, n)

BASE_NAMES = [] 

def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)

def greedy_knapsack(numbers: List[int], capacity: int) -> Tuple[List[List[int]], List[List[int]]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem.
    Parameter
    ----
    numbers : List[int]
        The list of item sizes can be in any order (here it is entered in ascending order)
    capacity : int
        Backpack capacity

    Return
    ----
    Tuple[List[List[int]], List[List[int]]]
        The first list: The size of the items in each backpack
        The second list: The original subscripts corresponding to the items in each backpack
    
    """
    # Save the original index to correspond one-to-one with the input numbers
    indexed_numbers = [(val, idx) for idx, val in enumerate(numbers)]
    # Since the input has been sorted, it can be used directly (maintaining the processing method consistent with the original logic)
    knapsacks = []
    index_knapsacks = []
    iii = int(0)
    while indexed_numbers:
        current_knapsack = []
        current_indices = []
        remaining_capacity = capacity

        while True:
            current_values = [val for val, idx in indexed_numbers]
            index = search_for_fit(current_values, remaining_capacity)
            if index == -1:
                break  

            # Retrieve the found item and its original index
            val, idx = indexed_numbers.pop(index)
            remaining_capacity -= val
            current_knapsack.append(val)
            current_indices.append(idx)

        if iii%1000==0:
            print(f"---------the {iii} th pack----------")
            print(f"{current_knapsack}--->{sum(current_knapsack)}")
            print(current_indices)
            print(f"\n")
        iii+=1
        knapsacks.append(tuple(current_knapsack))
        index_knapsacks.append(tuple(current_indices))

    return tuple(knapsacks), tuple(index_knapsacks)   

def extract_content(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if task_type=="sft":
            try:
                user_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "assistant")
                return user_content
            except Exception as e:
                pass
        elif task_type=="pretrain":
            if data.get('captions') and len(data['captions']) > 0:
                return data['captions'][0].get('content', "")
            else:
                assert 0, "No valid caption content found"
            
    except FileNotFoundError:
        return f" Error: File {json_file} does not exist"
    except json.JSONDecodeError:
        return f" Error: File {json_file} is not in valid JSON format"
    except Exception as e:
        return f"An error occurred during the extraction process: {str(e)}"

def extract_prompt(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assistant_content = next(msg["content"] for msg in data["messages"] if msg["role"] == "user")
        return assistant_content
            
    except FileNotFoundError:
        return f" Error: File {json_file} does not exist"
    except json.JSONDecodeError:
        return f" Error: File {json_file} is not in valid JSON format"
    except Exception as e:
        return f"An error occurred during the extraction process: {str(e)}"


def prepare_dirs(target_dir, new_dir):
    os.chdir(target_dir)
    print(f"--------change to directory {target_dir}--------")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"Directory '{new_dir}' created.")
    else:
        print(f"Directory '{new_dir}' already exists.")


def dataset_tokinfo_generator(f_name):
    """
    Dataset token information generator, reading and parsing file content line by line
    
    Parameter:
        f_name (str): The file path containing token information
        
    Generated:
        tuple: (base_name, token_len) - The basic file name and token length after parsing
    """
    try:
        with open(f_name, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                    
                parts = stripped_line.split(':')
                if len(parts) == 2:
                    base_name = parts[0].strip()
                    token_len_str = parts[1].strip()
                    
                    try:
                        token_len = int(token_len_str)
                        yield (base_name, token_len)
                    except ValueError:
                        print(
                            f"Warning: '{token_len_str}' cannot be converted to an integer. This line has been skipped",
                            file=sys.stderr
                        )
                        continue
                        
    except FileNotFoundError:
        print(f" error: file '{f_name}' does not exist ", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error occurred while processing file: {str(e)}", file=sys.stderr)
        return


class TokenInfoReader:
    """
    Token information reader
    
    It supports batch reading, full reading and breakpoint resumption functions, and is suitable for processing text files containing token information.
    File format requirements: One record per line, in the format of "base_name: token_len"
    """
    
    def __init__(self, f_name):
        """
        Initialize the reader
        
        Parameter
            f_name (str): The file path containing token information
        """
        self.f_name = f_name
        self.generator = dataset_tokinfo_generator(f_name)
        self._current_position = 0 

    def read(self, count=None):
        """
        Read the record

        Parameter:
            count (int, optional): The number of records to be read, default to None (read all remaining records)

        Return:
            tuple: (base_names list, token_lens list, actual read quantity)
        """
        base_names = []
        token_lens = []
        read_count = 0
        
        while True:
            if count is not None and read_count >= count:
                break
                
            try:
                base_name, token_len = next(self.generator)
                base_names.append(base_name)
                token_lens.append(token_len)
                read_count += 1
                self._current_position += 1
                
            except StopIteration:
                break
        
        return base_names, token_lens, read_count
    
    def get_current_position(self):
        return self._current_position


def process_knapsack(s1, idx_knapsack, dst_dir_image,dst_dir_json):
    """
    Process individual packing data

    Parameter:
        s1: Index of the current processing group
        idx_knapsack: The list of indexes contained in the backpack
        dst_dir: Target directory path
    """
    
    packed_imgs, packed_caps = [], []  
    
    packed_b_names = (idx["name"] for idx in idx_knapsack)
    
    if task_type == "pretrain":
        packed_info = (
            (os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
             extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")))
            for b_name in packed_b_names
        )
    elif task_type == "sft":
        packed_info = (
            (os.path.join(SRC_DIR_IMGS, f"{b_name}.{SRC_DST_EXTENSIONS[0]}"),
             extract_content(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")),
             extract_prompt(os.path.join(SRC_DIR_JSONS, f"{b_name}.{SRC_DST_EXTENSIONS[1]}")))
            for b_name in packed_b_names
        ) 
        
    json_dst = os.path.join(dst_dir_json, f"ps_{s1:08d}.{SRC_DST_EXTENSIONS[1]}")
    
    if task_type=="pretrain":
        for s2, (img_src, cap_src) in enumerate(packed_info):
            packed_imgs.append(img_src)
            packed_caps.append(cap_src)
        selected_prompts = get_random_prompts(PROMPTS, len(packed_imgs))
    elif task_type=="sft":
        selected_prompts = []
        
        for s2, (img_src, cap_src, prompt_src) in enumerate(packed_info):
            packed_imgs.append(img_src)
            packed_caps.append(cap_src)
            selected_prompts.append(prompt_src)
        pass
    json_data = {
        "images": packed_imgs,
        "captions": packed_caps,
        "prompts": selected_prompts
    }
    try:
        with open(json_dst, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f" thread {threading.current_thread().name} failed to generate JSON file {json_dst} : {str(e)}")
    return s1


if __name__ == "__main__":

    print("Step1-----------------Read the tokenlen information of the original ds-----------------Start")
    info_reader = TokenInfoReader(f_toklens_originalsample)
    base_names, token_lens, n_count = info_reader.read()
    
    BASE_NAMES=tuple(base_names)
    print(f" read {n_count} datas ")
    print("Step1-----------------Read the tokenlen information of the original ds-----------------Stop\n\n")
    
    print("Step2-----------------packing grouping-----------------Start")
    
    import pickle
    def load_bin_boxes(file_path: str):
        with open(file_path, 'rb') as f:
            bin_boxes = pickle.load(f)
        print(f"The packing result has been loaded: {file_path}")
        return bin_boxes

    bin_boxs =os.path.join(save_files_dir,'bins_boxs.pkl')
    bin_boxs=load_bin_boxes(bin_boxs)
    total_knapsacks = len(bin_boxs)
    
    print(f"raw data number----{n_count}----,after packing number----{total_knapsacks}----")
    print("Step2-----------------packing grouping-----------------Stop\n\n")

    print("Step3----------------- Start building the new dataset -----------------Start")
    print(f" starts processing the {total_knapsacks} group of data using {MAX_WORKERS} threads ")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="PackThread") as executor:
        # Submit all tasks
        if f_TEST:
            futures = {
                executor.submit(process_knapsack, s1, idx_knapsack, dst_dir_image,dst_dir_json): s1
                for s1, idx_knapsack in enumerate(bin_boxs[0:n_packed_samples])
            }
        else:
            futures = {
                executor.submit(process_knapsack, s1, idx_knapsack, dst_dir_image,dst_dir_json): s1
                for s1, idx_knapsack in enumerate(bin_boxs)
            }

        from tqdm import tqdm
        tty = open(os.devnull, 'w') if os.name == 'nt' else open('/dev/tty', 'w')
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Packing progress",
                           unit="pack",
                           file=tty
                          ):
            try:
                future.result()
            except Exception as e:
                s1 = futures[future]
                print(f"an error occurred when processing the {s1} th group of data: {e}")
                
    print("----------------- The new dataset was successfully constructed -----------------Stop")

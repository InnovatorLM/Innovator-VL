import io
import math
import re
import os
import glob
from typing import Any, Dict, Optional, Union

from areal.utils import logging
from datasets import concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from PIL.Image import Image as ImageObject

# Disable DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

from .preprocess import get_preprocess_func
from .prompt import PROBLEM_TYPE_SPECIAL_PROMPT

logger = logging.getLogger("Dataset")

def convert_image(
    image: Union[Dict[str, Any], ImageObject, str],
    max_pixels: Optional[int],
) -> ImageObject:
    if isinstance(image, dict):
        if "image" in image:
            image = image["image"]
        elif "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))
        else:
            raise ValueError(f"Unknown image format: {image.keys()}")
    elif isinstance(image, str):
        image_format = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
        image_format.extend([fmt.upper() for fmt in image_format])
        if image.endswith(tuple(image_format)):
            with Image.open(image) as img:
                image = img.copy()
        else:
            image = Image.open(io.BytesIO(image))
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    # 如果 image 已经是 PIL Image 对象，跳过上面的所有分支，直接继续处理

    if image.mode in ("CMYK", "YCbCr", "LAB", "HSV", "P"):
        image = image.convert("RGB")

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))
    return image


def process_dataset(
    path: str,
    split: str,
    processor,
    max_length: Optional[int] = None,
    ignore_prompt_type: bool = False,
):

    # detect pattern like "[10%]" in the path and remove it, storing the percentage
    percent_match = re.search(r"\[(\d{1,3})%\]", path)
    percent = None
    if percent_match:
        percent = int(percent_match.group(1))
        path = re.sub(r"\[\d{1,3}%\]", "", path)

    if path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=path)['train']
    elif path.endswith(".json"):
        dataset = load_dataset("json", data_files=path)['train']
    else:
        dataset = load_dataset(path=path, split=split)

    def general_process(sample):
        # Handle cases where images is None or empty list (text-only samples)
        raw_images = sample.get("images", None)
        if raw_images is None or len(raw_images) == 0:
            processed_images = []
        else:
            try:
                processed_images = [
                    convert_image(image, 512 * 512) for image in raw_images
                ]
            except Image.DecompressionBombError:
                logger.warning(f"Skipping sample due to large image size (DecompressionBombError).")
                return {"__valid__": False}
            except Exception as e:
                logger.warning(f"Skipping sample due to image error: {e}")
                return {"__valid__": False}

        if "qwen" in processor.image_processor.image_processor_type.lower():
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        system_prompt = {
            "role": "system",
            "content": "You are a helpful AI assistant.",
        }

        thinking_prompt = (
            "Think and solve the following question step by step. "
            "Please put your thinking and analysis procedure within <think></think>. "
            "Put ONLY your final answer within <answer></answer>."
        )
        normal_prompt = "Put ONLY your final answer within <answer></answer>."

        if ignore_prompt_type:
            prompt = thinking_prompt
        else:
            if "prompt_type" in sample and sample["prompt_type"] == "normal":
                prompt = normal_prompt
            else:
                prompt = thinking_prompt

        prompt = (
            prompt
            + (f" {sample['format_guidance']}\n" if "format_guidance" in sample else "")
        )

        if str(sample["problem_type"]) in PROBLEM_TYPE_SPECIAL_PROMPT:
            prompt += PROBLEM_TYPE_SPECIAL_PROMPT[str(sample["problem_type"])]
        else:
            prompt += "\n"

        messages = [
            {
                "role": "user",
                "content": prompt + sample["problem"]
                .replace("<image>", image_token)
            }
        ]
        messages.insert(0, system_prompt)
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sample.update({"messages": messages, "images": processed_images, "__valid__": True})
        return sample

    dataset = dataset.map(
        lambda sample: general_process(get_preprocess_func(path)(sample)),
        num_proc=16,
        load_from_cache_file=True,
    )
    
    # Filter out invalid samples
    dataset = dataset.filter(lambda x: x.get("__valid__", True))
    # Remove the temporary validity column
    if "__valid__" in dataset.column_names:
        dataset = dataset.remove_columns(["__valid__"])

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # Process the sample to get the total token count including image tokens
            # Handle None or empty images (text-only samples)
            sample_images = sample.get("images", None)
            has_images = sample_images is not None and len(sample_images) > 0
            processed_input = processor(
                text=[sample["messages"]],
                images=sample_images if has_images else None,
                padding=False,
                return_tensors="pt",
                return_length=True,
                return_attention_mask=False,
            )
            total_tokens = len(processed_input["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    if percent is not None:
        try:
            total_rows = len(dataset)
        except Exception:
            total_rows = getattr(dataset, "num_rows", None)

        if total_rows is None:
            return dataset

        take_n = max(1, math.floor(total_rows * percent / 100))
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(take_n))

    return dataset


def get_dataset(
    path: str,
    split: str,
    processor,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
    ignore_prompt_type: bool = False,
):
    if ignore_prompt_type:
        logger.info("Ignoring prompt type for thinking dataset processing.")
    
    # Expand globs and split commas
    raw_paths = path.split(",")
    expanded_paths = []
    for p in raw_paths:
        p = p.strip()
        # Handle glob patterns like *.parquet
        if "*" in p:
            glob_files = sorted(glob.glob(p))
            if glob_files:
                expanded_paths.extend(glob_files)
            else:
                expanded_paths.append(p)
        elif os.path.isdir(p):
            # If directory, recursively look for parquet files
            parquet_files = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
            if parquet_files:
                expanded_paths.extend(parquet_files)
            else:
                expanded_paths.append(p)
        else:
            expanded_paths.append(p)
            
    dataset_list = [
        process_dataset(p, split, processor, max_length, ignore_prompt_type) for p in expanded_paths
    ]
    
    # Align columns before concatenation
    all_features = set()
    for ds in dataset_list:
        all_features.update(ds.column_names)
    
    def add_missing_columns(example, current_columns):
        for feature in all_features:
            if feature not in current_columns:
                example[feature] = None
        return example

    dataset_list = [
        ds.map(lambda x: add_missing_columns(x, ds.column_names), desc="Aligning columns") 
        if set(ds.column_names) != all_features else ds 
        for ds in dataset_list
    ]

    # Unify features to avoid type mismatches (e.g. null vs string)
    if dataset_list:
        # Use the features from the first dataset as a base, but relax strictness if needed
        # Actually, simpler approach: cast all datasets to the features of the first one (or a superset)
        # But specifically for 'images', we know it causes issues.
        # Let's find a dataset with 'string' path in images if possible to use as reference.
        target_features = dataset_list[0].features
        for ds in dataset_list:
            # If we find a dataset where images struct has a string path, use its features for images
            if "images" in ds.features:
                img_feature = ds.features["images"]
                # Check if it's a sequence/list
                if hasattr(img_feature, "feature"): 
                    inner = img_feature.feature
                    # Check if inner is a dict-like object (Feature) before using 'in' operator
                    if hasattr(inner, "__contains__") and not isinstance(inner, ImageObject):
                        if "path" in inner and hasattr(inner["path"], "dtype") and inner["path"].dtype == "string":
                            target_features = ds.features
                            break
        
        # Now cast all datasets to this target feature set where possible
        new_list = []
        for ds in dataset_list:
            try:
                # Try to cast to target features (handling null <-> string mismatch)
                ds = ds.cast(target_features)
            except Exception:
                # If exact cast fails, try to cast specifically the images column if it exists
                # Or just proceed and hope for the best if cast is impossible
                pass
            new_list.append(ds)
        dataset_list = new_list
    
    dataset = concatenate_datasets(dataset_list)
    dataset = dataset.shuffle(seed=42)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset

import logging
import re
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

import math
from datetime import timedelta
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs

def split_model(model_name, num_layers=None):
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        num_layers = {
            "InternVL2-1B": 24,
            "InternVL2-2B": 24,
            "InternVL2-4B": 32,
            "InternVL2-8B": 32,
            "InternVL2-26B": 48,
            "InternVL2-40B": 60,
            "InternVL2-Llama3-76B": 80,
            "InternVL2_5-1B": 24,
            "InternVL2_5-2B": 24,
            "InternVL2_5-4B": 36,
            "InternVL2_5-8B": 32,
            "InternVL2_5-26B": 48,
            "InternVL2_5-38B": 64,
            "InternVL2_5-78B": 80,
            "InternVL3_5-8B": 32,
            "Intern-S1-mini": 36, 
        }.get(model_name.split("/")[-1], 32)
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map

@register_model("Intern_S1")
class Intern_S1(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/Intern-S1-mini",
        modality: str = "image",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        num_layers=None,
        interleave_visuals: Optional[bool] = False,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ):
        super().__init__()
        self.path = pretrained
        self.num_frame = num_frame
        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for Intern-S1, but got {batch_size}."
        self.batch_size_per_gpu = batch_size
        
        self.interleave_visuals = interleave_visuals
        self.system_prompt = system_prompt

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            device_map = split_model(pretrained.split("/")[-1], num_layers=num_layers)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self._processor = AutoProcessor.from_pretrained(
            self.path,
            trust_remote_code=True,
        )
        self._tokenizer = self._processor.tokenizer

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self._tokenizer.padding_side = 'left' 

        self._model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.device_map,
        ).eval()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._rank = 0
            self._world_size = 1

        self.modality = modality

    @property
    def processor(self): 
        return self._processor

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input_list):
        new_list = []
        for i in input_list:
            if isinstance(i, list):
                new_list.extend(self.flatten(i))
            else:
                new_list.append(i)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, v in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if isinstance(visuals, list):
                visuals = self.flatten(visuals)
            else:
                visuals = [visuals]
            
            final_visuals = [v for v in visuals if isinstance(v, Image.Image)]

            if self.modality == "image":
                if "<image>" in contexts:
                    contexts = contexts.replace("<image>", "")

                messages = []
                if self.system_prompt:
                    messages.append({
                        "role": "system", 
                        "content": [{"type": "text", "text": self.system_prompt}]
                    })

                content_parts = []
                
                if self.interleave_visuals is False:
                    for img in final_visuals:
                        content_parts.append({"type": "image", "image": img})
                    
                    if not contexts.strip():
                        contexts = "Describe the image."
                    content_parts.append({"type": "text", "text": contexts})
                
                else:
                    image_placeholders = re.findall(r"<image \d+>", contexts)
                    text_parts = re.split(r"<image \d+>", contexts)
                    
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        
                        image_idx = min(img_idx, len(final_visuals) - 1) if final_visuals else 0
                        
                        if final_visuals and image_idx < len(final_visuals):
                            content_parts.append({"type": "image", "image": final_visuals[image_idx]})
                        
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                messages.append({"role": "user", "content": content_parts})

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False
                ).to(self.device)

                generation_kwargs = gen_kwargs.copy()

                if "pad_token_id" not in generation_kwargs:
                    generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

                with torch.inference_mode():
                    generate_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )

                response = self.processor.decode(
                    generate_ids[0, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()


            elif self.modality == "video":
                # Video logic remains mostly similar but we need to respect the loop structure
                if not visuals:
                    raise ValueError("Video modality requires at least one visual input.")
                video_path = visuals[0] 

                pixel_values, num_patches_list = load_video(video_path, num_segments=self.num_frame, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

                total_video_patches = sum(num_patches_list)
                total_video_tokens = ["<image>"] * total_video_patches
                total_video_tokens_str = " ".join(total_video_tokens) + "\n"
                full_context = f"{total_video_tokens_str}{contexts}"

                video_inputs = self.tokenizer(
                    full_context, 
                    return_tensors="pt", 
                    padding=True 
                ).to(self.device)
                
                input_ids = video_inputs.input_ids
                attention_mask = video_inputs.attention_mask

                generation_kwargs = gen_kwargs.copy()
                
                if "pad_token_id" not in generation_kwargs:
                    generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        **generation_kwargs
                    )

                generated_ids = outputs[:, input_ids.shape[1]:]
                response = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()

            res.append(response)
            pbar.update(1)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Intern-S1")
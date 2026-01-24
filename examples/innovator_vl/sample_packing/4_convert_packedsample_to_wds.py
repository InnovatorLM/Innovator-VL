#!/usr/bin/env python3

import argparse
import uuid
import json
import os
import yaml
import webdataset as wds
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('/vlm/yinxie/code/AIAK-Megatron')
sys.path.append('/vlm/yinxie/code/AIAK-Megatron/megatron')

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.prepare import WebdatasetPreparator
from megatron.energon.flavors.webdataset.structs import ShardInfo, WebdatasetInfo, WebdatasetSplits
from tool import get_init_file

def sample_loader_template(media: str=None):
    """Returns a template for a sample_loader.py file."""
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    messages=[]",
        "    for message in sample['json']:",
        "        assert message['role'] in ['system','user','assistant']",
        "        messages.append(dict(",
        "            role=message['role'],",
        "            content=message['content']",
        "        ))",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        "        video=sample.get('mp4'),",
        "        image=sample.get('jpg')," if media == 'mix' else "",
        "        messages=messages,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])
    
### ZXW   

def sample_loader_template_caption(media=None):
    """A loader that adapts to the captioning of the entire multi-image"""
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    data = sample['json']",
        "    images = [sample.get(f'img{i}.jpg') for i in range(len(data['images']))]",
        "    captions = data['captions'] ",
        "    prompts = data['prompts']",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        "        captions=captions,",
        "        prompts=prompts,",
        "        images=images,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])
def stream_samples_caption(src_dir: str):
    for json_path in Path(src_dir).glob("*.json"):
        sample_id = json_path.stem            
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        yield {
            "id": sample_id,
            "images": raw["images"],           
            "prompts": raw.get("prompts", []), 
            "captions": raw["captions"]       
        }

def construct_sample_caption(args, entry):
    """Pack the entire sample"""
    sample = {"__key__": entry["id"]}
    for idx, img_path in enumerate(entry["images"]):
        with open(img_path, "rb") as f:
            sample[f"img{idx}.jpg"] = f.read()

    payload = {
        "prompts": entry["prompts"],
        "captions": entry["captions"],
        "images": entry["images"]
    }
    sample["json"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return sample


def construct_sample(args, vision, path, entry):
    """ construct webdataset sample """
    assert vision == 'image' or vision == 'video'
    directory = args.image_dir if vision == 'image' else args.video_dir

    with open(os.path.join(directory, path), "rb") as vision_file:
        vision_data = vision_file.read()
    sample = {
        "__key__": entry.get('id', path).replace('.', '_'),
        "jpg" if vision == 'image' else 'mp4': vision_data,
        "json": json.dumps(entry[args.columns_messages]).encode("utf-8"),
    }
    return sample


def convert_to_wds(args):
    """ Convert dataset to wds format """
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    tar = os.path.join(args.output_dir, 'pretrain-%06d.tar')
    if args.mode == "caption_pack":
        with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as sink:
            for entry in tqdm(stream_samples_caption(args.json_file)):
                sample=construct_sample_caption(args, entry)
                sink.write(sample)
                
        write_config(EPath(args.output_dir).absolute(), args.media,
                     template_func=sample_loader_template_caption,
                     class_name="PackedCaptioningSample")   
    print(f"Dataset successfully converted to wds")



def write_config(path: EPath, media=None, template_func=None, class_name=None):
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    if class_name is None:
        class_name = "MultiMixQASample" if media == 'mix' else "MultiVidQASample"
    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": class_name,
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader"
    }

    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    tpl = (template_func or sample_loader_template)(media)
    with (path / MAIN_FOLDER_NAME / "sample_loader.py").open("w") as f:
        f.write(tpl)

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=32,
    )


def _add_arguments(parser: argparse.ArgumentParser):
    
    input_token_file ,MAX_TOKEN_LEN,save_files_dir,big_dir,DEFAULT_DIRECTORY= get_init_file()
    output_dir=DEFAULT_DIRECTORY+'_wds'
    last_save_dir_json=os.path.join(save_files_dir,"row_packing_jsons")
    last_save_dir_image=os.path.join(save_files_dir,"row_packing_images")
    
    """Add arguments"""
    group = parser.add_argument_group(title='wds')
    group.add_argument('--output_dir', type=str, default=output_dir, help='Output directory')
    group.add_argument('--json_file', type=str, default=last_save_dir_json,
                       help='Directory (multi-image captioning) or single file (old format)')
    group.add_argument('--image_dir', type=str, default=last_save_dir_image, help='Image directory')
    group.add_argument('--video_dir', type=str, required=False, help='Video directory')
    group.add_argument('--maxcount', type=int, default=10000, help='Number of samples per shard')
    group.add_argument('--maxsize', type=int, default=3000000000, help='Maximum size of each shard')
    group.add_argument('--media', type=str, choices=["mix", "image", "video"], default="image", help='Media type')
    group.add_argument('--columns_messages', type=str, default="messages", help='Column name for messages')
    # 新增模式选择
    group.add_argument('--mode', type=str,
                       choices=["chat", "caption_pack"],
                       default="caption_pack",
                       help="chat= Old format (single-image dialogue); caption_pack= New format (Full multi-image caption)")
    return parser


def parse_args():
    """arguments"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    """main function"""
    args = parse_args()
    convert_to_wds(args)


if __name__ == '__main__':
    main()



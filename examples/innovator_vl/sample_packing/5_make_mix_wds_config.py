from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
import yaml
import os
from tool import cfg,get_ip_info
from pathlib import Path
from tqdm import tqdm
import time

def find_and_merge_yaml_files(base_dir):
    base_path = Path(base_dir)
    wds_dirs = list(base_path.glob("*_wds"))
    
    merged_split_data = {
        "exclude": [],
        "split_parts": {
            "test": [],
            "train": [],
            "val": []
        }
    }
    merged_info_data = {"shard_counts": {}}
    
    for wds_dir in tqdm(wds_dirs):
        nv_meta_dir = wds_dir / ".nv-meta"
        split_file = nv_meta_dir / "split.yaml"
        info_file = nv_meta_dir / ".info.yaml"
        
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = yaml.safe_load(f)
            if split_data and 'split_parts' in split_data:
                for split_type in ['train', 'test', 'val']:
                    if split_type in split_data['split_parts']:
                        prefixed_items = [f"{wds_dir.name}/{item}" for item in split_data['split_parts'][split_type]]
                        merged_split_data['split_parts'][split_type].extend(prefixed_items)
        
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                info_data = yaml.safe_load(f)
            if info_data and 'shard_counts' in info_data:
                for tar_file, count in info_data['shard_counts'].items():
                    prefixed_tar_file = f"{wds_dir.name}/{tar_file}"
                    merged_info_data['shard_counts'][prefixed_tar_file] = count
    
    with open(base_path / MAIN_FOLDER_NAME / "split.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(merged_split_data, f, default_flow_style=False, allow_unicode=True)
    
    with open(base_path /MAIN_FOLDER_NAME/ ".info.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(merged_info_data, f, default_flow_style=False, allow_unicode=True)


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
def sample_loader_template_caption(media=None):
    """A loader that ADAPTS to the captioning of the entire multi-image"""
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

def write_config(path: Path, media=None, template_func=None, class_name=None,mode='merge'):
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    if mode!='merge':
        all_tars = list(path.glob("*_wds/*.tar")) + list(path.glob("*_wds/*.tgz"))
        all_tars=[]
        for name in os.listdir(str(path)) :
            if name.endswith('_wds'):
                tmp_list=list(path.glob(f"{name}/*.tar"))
                all_tars.extend(tmp_list)
        # all_tars = list(path.glob("**_wds/*.tar"))
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
    if mode!='merge':
        BaseWebdatasetFactory.prepare_dataset(
            path,
            all_tars,
            split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
            tar_index_only=False,
            workers=80,
        )

    
def main(mode='merge'):
    ip_indx,_,_=get_ip_info(cfg['hosts'])
    if ip_indx==0:
        path=cfg['data']['directory']
        num_wds_dir=0
        wds_path=[]
        tmp=[]
        for name in os.listdir(path) :
            if name.endswith('_wds'):
                num_wds_dir+=1
                tmp.append(os.path.join(path,name))
                wds_path.append(os.path.join(path,name,MAIN_FOLDER_NAME,'.info.yaml'))
        if num_wds_dir>1:
            all_done=False
            while not all_done:
                all_done=True
                for i in wds_path:
                    all_done=all_done and os.path.exists(i)
                if not all_done:
                    print('sleep 60 seconds')
                    time.sleep(60)
            if all_done:
                print('all done')
                write_config(Path(path).absolute(), 'image',
                            template_func=sample_loader_template_caption,
                            class_name="PackedCaptioningSample",mode=mode) 
                if mode=='merge':
                    find_and_merge_yaml_files(path)
            print(f'config generation completed,your train file path is: {path}')
        else:
            print(f'config generation completed,your train file path is: {tmp[0]}')

if __name__=="__main__":
    main('generate')
#!/usr/bin/env python3
"""
convert_pairs.py
把 {name}.json 转成目标格式，字段顺序：content, role

# 全量转换
python convert_pairs.py /data_1/test_samples /data_1/converted_samples

# 抽查前 100 条
python convert_pairs.py /path/to/src /path/to/dst --samples 100
python convert_pairs.py /data_1/aiak_caption_emova+llava_recap /data_3/aiak_pretrain_vqa_5500k --samples 10   # 可以选择不拷贝图片（line 70）

# 指定进程数
python convert_pairs.py /path/to/src /path/to/dst --workers 8

"""
import argparse
import json
import os
import shutil
from glob import iglob
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from tqdm import tqdm

logger = None          # 若需要写到文件可自行初始化


def build_pairs(src_dir: str) -> List[Tuple[str, str]]:
    """收集成对的 json / jpg 绝对路径"""
    pairs = []
    for json_path in iglob(os.path.join(src_dir, "*.json")):
        base = os.path.splitext(os.path.basename(json_path))[0]
        jpg_path = os.path.join(src_dir, f"{base}.jpg")
        if os.path.isfile(jpg_path):
            pairs.append((json_path, jpg_path))
    return pairs



def convert_one(pair: Tuple[str, str], dst_dir: str) -> str:
    """单条转换逻辑（含拷贝图片）"""
    json_src, jpg_src = pair
    base = os.path.splitext(os.path.basename(json_src))[0]
    dst_json = os.path.join(dst_dir, f"{base}.json")
    dst_jpg  = os.path.join(dst_dir, f"{base}.jpg")

    # 读原 json
    with open(json_src, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 修正键名
    messages_raw = data.get("massages:", [])
    # 2. 调整字段顺序
    messages_new = [
        {"content": m["content"], "role": m["role"]}
        for m in messages_raw
    ]
    new_obj = {
        "messages": messages_new,
        "images": [os.path.basename(jpg_src)]
    }

    # 写目标 json
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(new_obj, f, ensure_ascii=False, indent=None)

    # 3. 拷贝图片
    shutil.copy2(jpg_src, dst_jpg)

    return base


def process_chunk(chunk: List[Tuple[str, str]], dst_dir: str, pos: int):
    """供多进程 map 使用的 chunk 处理"""
    ret = []
    for p in chunk:
        ret.append(convert_one(p, dst_dir))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="含 json/jpg 的源目录")
    parser.add_argument("dst", help="输出目录")
    parser.add_argument("--workers", type=int, default=cpu_count(),
                        help="进程数，默认=CPU核心数")
    parser.add_argument("--samples", type=int, default=None,
                        help="仅处理前 N 条")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    pairs = build_pairs(args.src)
    if not pairs:
        print("未找到任何 json/jpg 成对文件")
        return

    if args.samples:
        pairs = pairs[: args.samples]

    # 分块：一条一个任务粒度太大，按进程数切块
    chunk_size = max(1, len(pairs) // args.workers)
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    total = 0

    with Pool(processes=args.workers) as pool:
        with tqdm(total=len(pairs), desc="convert") as bar:
            for chunk in chunks:
                future = pool.apply_async(process_chunk, (chunk, args.dst, total))
                done = future.get()
                total += len(done)
                if total % 2000 == 0:
                    print(f"[INFO] 已处理 {total} 条")
                bar.update(len(done))

    print(f"[DONE] 总计转换 {total} 条数据")
    # 输出日志位置提示（如需要）
    # print("详细日志见 output.log")


if __name__ == "__main__":
    main()

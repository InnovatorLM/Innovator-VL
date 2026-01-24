#!/usr/bin/env python3
"""
分析重复样本的模式，帮助找出问题原因
"""
import argparse
from pathlib import Path
from collections import defaultdict
import webdataset as wds
from tqdm import tqdm

def extract_subdir_name(shard_name):
    """从shard文件名提取子目录名"""
    parts = shard_name.rsplit('_', 1)
    if len(parts) >= 2 and parts[1].replace('.tar', '').isdigit():
        return parts[0]
    return shard_name.replace('.tar', '')

def analyze_duplicates(output_dir, max_shards=50):
    """分析重复样本的模式"""
    output_path = Path(output_dir)
    all_shards = sorted(output_path.glob("*.tar"))
    
    if max_shards:
        all_shards = all_shards[:max_shards]
    
    print(f"分析前 {len(all_shards)} 个shard文件...\n")
    
    # 按子目录分组
    subdir_samples = defaultdict(lambda: defaultdict(list))  # subdir -> sample_id -> [shards]
    
    for shard_path in tqdm(all_shards, desc="读取shard"):
        shard_name = shard_path.name
        subdir_name = extract_subdir_name(shard_name)
        
        try:
            for sample in wds.WebDataset(str(shard_path), shardshuffle=False):
                sample_id = sample.get('__key__', '')
                if sample_id:
                    subdir_samples[subdir_name][sample_id].append(shard_name)
        except Exception as e:
            print(f"读取 {shard_name} 失败: {e}")
            continue
    
    # 找出同一子目录内的重复
    print("\n" + "="*60)
    print("重复分析结果:")
    print("="*60)
    
    total_duplicates = 0
    subdir_stats = []
    
    for subdir_name, samples_dict in subdir_samples.items():
        duplicates_in_subdir = {sid: shards for sid, shards in samples_dict.items() if len(shards) > 1}
        if duplicates_in_subdir:
            total_duplicates += len(duplicates_in_subdir)
            subdir_stats.append((subdir_name, len(duplicates_in_subdir), len(samples_dict)))
    
    # 按重复数量排序
    subdir_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n发现有重复的子目录数: {len(subdir_stats)}")
    print(f"总重复样本数: {total_duplicates}")
    
    if subdir_stats:
        print("\n重复最多的前10个子目录:")
        for subdir_name, dup_count, total_samples in subdir_stats[:10]:
            print(f"  {subdir_name}: {dup_count} 重复 / {total_samples} 总样本 ({dup_count/total_samples*100:.1f}%重复率)")
        
        # 分析重复模式
        print("\n分析重复模式（查看前5个重复样本）:")
        count = 0
        for subdir_name, samples_dict in subdir_samples.items():
            duplicates = {sid: shards for sid, shards in samples_dict.items() if len(shards) > 1}
            if duplicates and count < 5:
                sample_id, shards = list(duplicates.items())[0]
                print(f"\n子目录: {subdir_name}")
                print(f"  样本ID: {sample_id}")
                print(f"  出现在shard数: {len(shards)}")
                print(f"  Shard列表: {', '.join(shards[:5])}" + (f" ... (共{len(shards)}个)" if len(shards) > 5 else ""))
                count += 1
    
    print("\n" + "="*60)
    print("\n可能的原因:")
    print("1. 之前运行过但没有清理，新运行时没有检查已存在的shard（旧代码bug）")
    print("2. 同一个parquet文件被处理了多次（如rglob找到了重复路径）")
    print("3. 多进程并发处理导致的问题")
    print("4. 数据源本身就有重复的样本ID")
    
    return subdir_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析重复样本的模式')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--max-shards', type=int, default=50, help='最多分析多少个shard')
    
    args = parser.parse_args()
    analyze_duplicates(args.output_dir, args.max_shards)


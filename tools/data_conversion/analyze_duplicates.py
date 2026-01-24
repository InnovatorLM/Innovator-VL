#!/usr/bin/env python3
"""
分析输出目录中是否存在重复的样本ID，用于检查是否有数据重复

使用方法:
    python analyze_duplicates.py --output_dir /path/to/output [--num-workers 8] [--sample 1000]
"""

import argparse
from pathlib import Path
from collections import defaultdict
import webdataset as wds
from tqdm import tqdm
from multiprocessing import Pool
import random


def process_single_shard(shard_path_str: str) -> tuple:
    """
    处理单个shard文件，提取样本ID
    
    Returns:
        (shard_name, sample_ids, sample_count, error): 
        - shard_name: shard文件名
        - sample_ids: 样本ID列表
        - sample_count: 样本数量
        - error: 错误信息（如果有）
    """
    shard_path = Path(shard_path_str)
    sample_ids = []
    sample_count = 0
    error = None
    
    try:
        for sample in wds.WebDataset(str(shard_path), shardshuffle=False):
            sample_id = sample.get('__key__', '')
            if sample_id:
                sample_ids.append(sample_id)
            sample_count += 1
    except Exception as e:
        error = str(e)
    
    return shard_path.name, sample_ids, sample_count, error


def find_duplicate_samples(output_dir: str, max_shards_to_check: int = None, 
                          num_workers: int = 4, sample_shards: bool = False) -> dict:
    """
    检查所有shard文件中是否有重复的样本ID
    
    Args:
        output_dir: 输出目录
        max_shards_to_check: 最多检查多少个shard（None表示全部）
        num_workers: 并行进程数
        sample_shards: 是否采样检查（如果是True，随机选择shard）
    
    Returns:
        统计信息字典
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return {}
    
    # 查找所有tar文件
    all_shards = sorted([str(p) for p in output_path.glob("*.tar")])
    total_shard_count = len(all_shards)
    
    if len(all_shards) == 0:
        print(f"未找到任何shard文件")
        return {}
    
    # 采样或限制检查数量
    if sample_shards and max_shards_to_check:
        # 随机采样
        if max_shards_to_check < len(all_shards):
            all_shards = random.sample(all_shards, max_shards_to_check)
            print(f"随机采样检查 {max_shards_to_check} 个shard文件（共 {total_shard_count} 个）...")
        else:
            print(f"检查所有 {len(all_shards)} 个shard文件...")
    elif max_shards_to_check:
        all_shards = all_shards[:max_shards_to_check]
        print(f"检查前 {max_shards_to_check} 个shard文件（共 {total_shard_count} 个）...")
    else:
        print(f"检查所有 {len(all_shards)} 个shard文件...")
    
    print(f"使用 {num_workers} 个进程并行处理...")
    
    # 使用多进程并行处理
    sample_to_shards = defaultdict(list)
    total_samples = 0
    failed_shards = []
    
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_shard, all_shards),
                total=len(all_shards),
                desc="扫描shard文件"
            ))
    else:
        results = [process_single_shard(shard) for shard in tqdm(all_shards, desc="扫描shard文件")]
    
    # 收集结果，同时按子目录分组
    subdir_to_samples = defaultdict(lambda: defaultdict(list))  # subdir -> sample_id -> [shards]
    
    for shard_name, sample_ids, sample_count, error in results:
        total_samples += sample_count
        
        if error:
            failed_shards.append((shard_name, f"读取失败: {error}"))
        elif sample_count == 0:
            failed_shards.append((shard_name, "空文件"))
        else:
            # 从shard文件名提取子目录名（格式: subdir_name_XXXXXX.tar）
            # 如果包含下划线和数字，取最后的下划线之前的部分作为子目录名
            parts = shard_name.rsplit('_', 1)
            if len(parts) >= 2 and parts[1].replace('.tar', '').isdigit():
                subdir_name = parts[0]  # 子目录名
            else:
                # 如果格式不符合预期，使用整个文件名（去掉.tar）作为子目录名
                subdir_name = shard_name.replace('.tar', '')
            
            # 记录每个样本ID出现的shard（全局记录）
            for sample_id in sample_ids:
                sample_to_shards[sample_id].append(shard_name)
            
            # 按子目录记录样本ID（用于检测同一子目录内的重复）
            for sample_id in sample_ids:
                subdir_to_samples[subdir_name][sample_id].append(shard_name)
    
    # 找出所有重复的样本ID（跨所有shard）
    all_duplicates = {sid: shards for sid, shards in sample_to_shards.items() if len(shards) > 1}
    
    # 找出同一子目录内的重复（这才是真正的问题）
    same_subdir_duplicates = {}
    for subdir_name, samples_dict in subdir_to_samples.items():
        for sample_id, shards in samples_dict.items():
            if len(shards) > 1:
                # 同一子目录内，同一个样本ID出现在多个shard中
                key = f"{subdir_name}:{sample_id}"
                same_subdir_duplicates[key] = shards
    
    # 统计信息
    stats = {
        'total_shards_checked': len(all_shards),
        'total_shards_in_dir': total_shard_count,
        'total_samples': total_samples,
        'unique_sample_ids': len(sample_to_shards),
        'all_duplicate_samples': len(all_duplicates),  # 所有重复（包括跨子目录）
        'same_subdir_duplicates': len(same_subdir_duplicates),  # 同一子目录内的重复（真正的问题）
        'failed_shards': len(failed_shards),
        'all_duplicates_detail': all_duplicates,
        'same_subdir_duplicates_detail': same_subdir_duplicates,
        'failed_shards_detail': failed_shards
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='检查shard文件中是否有重复的样本ID')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--max-shards',
        type=int,
        default=None,
        help='最多检查多少个shard文件（默认检查全部）'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=32,
        help='并行进程数（默认: 8）'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='随机采样检查（需要配合--max-shards使用）'
    )
    parser.add_argument(
        '--show-duplicates',
        action='store_true',
        help='显示重复的样本ID详情'
    )
    parser.add_argument(
        '--show-failed',
        action='store_true',
        help='显示读取失败的shard文件'
    )
    
    args = parser.parse_args()
    
    stats = find_duplicate_samples(
        args.output_dir, 
        args.max_shards, 
        num_workers=args.num_workers,
        sample_shards=args.sample
    )
    
    if not stats:
        return
    
    print("\n" + "="*60)
    print("重复检查结果:")
    print("="*60)
    if 'total_shards_in_dir' in stats:
        print(f"目录中总shard数: {stats['total_shards_in_dir']:,}")
    print(f"检查的shard文件数: {stats['total_shards_checked']:,}")
    print(f"总样本数: {stats['total_samples']:,}")
    print(f"唯一样本ID数: {stats['unique_sample_ids']:,}")
    print(f"\n重复分析:")
    print(f"  跨所有shard的重复样本ID数: {stats['all_duplicate_samples']:,} (这可能是正常的，因为不同数据源可能有相同样本)")
    print(f"  同一子目录内的重复样本ID数: {stats['same_subdir_duplicates']:,} ⚠️ (这才是真正的问题！)")
    print(f"读取失败的shard数: {stats['failed_shards']}")
    
    if stats['same_subdir_duplicates'] > 0:
        print("\n⚠️  警告: 发现同一子目录内的重复样本ID！")
        print(f"   有 {stats['same_subdir_duplicates']} 个样本ID在同一子目录内的多个shard中重复出现")
        print(f"   这说明转换逻辑可能有问题，需要检查！")
        
        if args.show_duplicates:
            print("\n同一子目录内重复样本详情（显示前20个）:")
            count = 0
            for key, shards in list(stats['same_subdir_duplicates_detail'].items())[:20]:
                subdir_name, sample_id = key.split(':', 1)
                print(f"  子目录: {subdir_name}")
                print(f"  样本ID: {sample_id}")
                print(f"    出现在shard: {', '.join(shards)}")
                count += 1
                if count >= 20:
                    break
            if len(stats['same_subdir_duplicates_detail']) > 20:
                print(f"  ... 还有 {len(stats['same_subdir_duplicates_detail']) - 20} 个重复样本")
    else:
        print("\n✓  未发现同一子目录内的重复样本ID")
        if stats['all_duplicate_samples'] > 0:
            print(f"   注意: 有 {stats['all_duplicate_samples']:,} 个样本ID跨不同子目录重复，这是正常的（不同数据源可能包含相同样本）")
    
    if stats['failed_shards'] > 0:
        print(f"\n读取失败的shard文件 ({stats['failed_shards']} 个):")
        if args.show_failed:
            for shard_name, reason in stats['failed_shards_detail']:
                print(f"  - {shard_name}: {reason}")
        else:
            print("  （使用 --show-failed 查看详情）")
    
    # 计算重复率
    if stats['total_samples'] > 0:
        all_duplicate_rate = (stats['all_duplicate_samples'] / stats['unique_sample_ids']) * 100 if stats['unique_sample_ids'] > 0 else 0
        same_subdir_rate = (stats['same_subdir_duplicates'] / stats['unique_sample_ids']) * 100 if stats['unique_sample_ids'] > 0 else 0
        print(f"\n重复率统计:")
        print(f"  跨所有shard的重复率: {all_duplicate_rate:.2f}% (可能正常)")
        if stats['same_subdir_duplicates'] > 0:
            print(f"  同一子目录内重复率: {same_subdir_rate:.2f}% ⚠️ (需要关注)")
    
    print("="*60)
    
    # 建议
    if stats['same_subdir_duplicates'] > 0:
        print("\n建议:")
        print("1. 同一子目录内的重复是不正常的，说明转换逻辑可能有问题")
        print("2. 检查resume逻辑是否正确收集了所有已处理的样本ID")
        print("3. 检查是否有多个进程同时处理同一个子目录")
        print("4. 如果确认有问题，可能需要重新处理这些子目录")
    elif stats['all_duplicate_samples'] > 0:
        print("\n说明:")
        print("跨子目录的样本ID重复是正常的，因为:")
        print("  - 不同数据源可能包含相同的样本（如相同的图像出现在多个数据集中）")
        print("  - 这是训练数据的正常情况，不需要处理")


if __name__ == '__main__':
    main()


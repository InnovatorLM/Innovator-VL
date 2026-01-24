#!/usr/bin/env python3
"""
检查输出目录中的shard文件，检测不完整的文件
用于诊断多次resume后可能存在的问题

使用方法:
    python check_shards.py --output_dir /path/to/output
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import webdataset as wds
from collections import defaultdict


def check_shard_integrity(shard_path: Path) -> Tuple[bool, int, int]:
    """
    检查shard文件的完整性
    
    Returns:
        (is_readable, sample_count, error_flag):
        - is_readable: 文件是否可读
        - sample_count: 成功读取的样本数量
        - error_flag: 0=正常, 1=读取异常, 2=文件为空或很小
    """
    try:
        sample_count = 0
        has_data = False
        
        # 检查文件大小（如果文件太小，可能不完整）
        file_size = shard_path.stat().st_size
        if file_size < 1024:  # 小于1KB，可能是空文件或不完整
            return False, 0, 2
        
        try:
            for sample in wds.WebDataset(str(shard_path)):
                sample_id = sample.get('__key__', '')
                if sample_id:
                    has_data = True
                sample_count += 1
                
                # 如果样本数超过预期（比如10001），说明shard可能有问题
                if sample_count > 10001:
                    break
            
            # 如果能完整读取，认为是完整的
            return True, sample_count, 0
        except Exception as e:
            # 读取过程中出错，可能是不完整
            return has_data, sample_count, 1  # 返回已读取的部分
    except Exception as e:
        # 文件不存在或无法访问
        return False, 0, 2


def check_all_shards(output_dir: str) -> Dict:
    """
    检查输出目录中的所有shard文件
    
    Returns:
        统计信息字典
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        return {}
    
    # 查找所有tar文件
    all_shards = sorted(output_path.glob("*.tar"))
    
    if len(all_shards) == 0:
        print(f"未找到任何shard文件")
        return {}
    
    print(f"找到 {len(all_shards)} 个shard文件，开始检查...\n")
    
    stats = {
        'total': len(all_shards),
        'complete': 0,
        'incomplete': 0,
        'readable_but_empty': 0,
        'unreadable': 0,
        'by_subdir': defaultdict(lambda: {'total': 0, 'complete': 0, 'incomplete': 0}),
        'incomplete_files': []
    }
    
    for shard_path in all_shards:
        # 解析子目录名称（从文件名中提取前缀）
        # 格式: subdir_name_XXXXXX.tar
        parts = shard_path.stem.split('_')
        if len(parts) >= 2:
            subdir_name = '_'.join(parts[:-1])  # 除最后一部分（编号）外的所有部分
        else:
            subdir_name = 'unknown'
        
        stats['by_subdir'][subdir_name]['total'] += 1
        
        is_readable, sample_count, error_flag = check_shard_integrity(shard_path)
        
        if error_flag == 2:
            # 文件无法读取或太小
            stats['unreadable'] += 1
            stats['by_subdir'][subdir_name]['incomplete'] += 1
            stats['incomplete_files'].append({
                'path': str(shard_path),
                'reason': '文件无法读取或文件大小异常',
                'size': shard_path.stat().st_size if shard_path.exists() else 0,
                'samples': sample_count
            })
            print(f"❌ 无法读取: {shard_path.name} (大小: {shard_path.stat().st_size if shard_path.exists() else 0} bytes)")
        elif error_flag == 1:
            # 读取过程中出错（不完整）
            stats['incomplete'] += 1
            stats['by_subdir'][subdir_name]['incomplete'] += 1
            stats['incomplete_files'].append({
                'path': str(shard_path),
                'reason': '读取异常（可能不完整）',
                'size': shard_path.stat().st_size,
                'samples': sample_count
            })
            print(f"⚠️  不完整: {shard_path.name} (已读取 {sample_count} 个样本)")
        elif sample_count == 0:
            # 可读但为空
            stats['readable_but_empty'] += 1
            stats['by_subdir'][subdir_name]['incomplete'] += 1
            stats['incomplete_files'].append({
                'path': str(shard_path),
                'reason': '文件可读但为空',
                'size': shard_path.stat().st_size,
                'samples': 0
            })
            print(f"⚠️  空文件: {shard_path.name}")
        else:
            # 完整的文件
            stats['complete'] += 1
            stats['by_subdir'][subdir_name]['complete'] += 1
            if len(all_shards) <= 20:  # 如果文件不多，显示详细信息
                print(f"✓  完整: {shard_path.name} ({sample_count} 个样本)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='检查shard文件完整性')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--list-incomplete',
        action='store_true',
        help='列出所有不完整的文件'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='自动修复：备份并删除不完整的shard文件（危险操作，请谨慎）'
    )
    
    args = parser.parse_args()
    
    stats = check_all_shards(args.output_dir)
    
    if not stats:
        return
    
    print("\n" + "="*60)
    print("检查结果统计:")
    print("="*60)
    print(f"总shard文件数: {stats['total']}")
    print(f"  完整文件: {stats['complete']} ✓")
    print(f"  不完整文件: {stats['incomplete']} ⚠️")
    print(f"  无法读取: {stats['unreadable']} ❌")
    print(f"  空文件: {stats['readable_but_empty']} ⚠️")
    
    # 按子目录统计
    if len(stats['by_subdir']) > 1:
        print("\n按子目录统计:")
        for subdir, subdir_stats in sorted(stats['by_subdir'].items()):
            status = "✓" if subdir_stats['incomplete'] == 0 else "⚠️"
            print(f"  {status} {subdir}: {subdir_stats['complete']}完整 / {subdir_stats['incomplete']}不完整 / {subdir_stats['total']}总计")
    
    # 列出不完整的文件
    if args.list_incomplete and stats['incomplete_files']:
        print("\n不完整的shard文件列表:")
        for item in stats['incomplete_files']:
            print(f"  - {Path(item['path']).name}")
            print(f"    原因: {item['reason']}, 大小: {item['size']} bytes, 样本数: {item['samples']}")
    
    # 修复建议
    if stats['incomplete'] > 0 or stats['unreadable'] > 0:
        print("\n" + "="*60)
        print("修复建议:")
        print("="*60)
        print("1. 使用新版本的代码（已包含自动修复功能）运行 resume，")
        print("   新代码会自动检测并处理不完整的shard文件")
        print("2. 或者手动删除不完整的shard文件（使用 --list-incomplete 查看列表）")
        print("\n注意：删除不完整的shard后，重新运行 resume 时会重新处理这些数据")
        print("     已处理的样本ID会被记录，不会重复处理")
        
        if args.fix:
            print("\n开始修复...")
            backup_dir = Path(args.output_dir) / ".backup_incomplete"
            backup_dir.mkdir(exist_ok=True)
            
            for item in stats['incomplete_files']:
                shard_path = Path(item['path'])
                if shard_path.exists():
                    backup_path = backup_dir / shard_path.name
                    print(f"  备份并删除: {shard_path.name}")
                    try:
                        shard_path.rename(backup_path)
                    except Exception as e:
                        print(f"    错误: {e}")
            
            print(f"\n修复完成！不完整的文件已备份到: {backup_dir}")
    
    print("="*60)


if __name__ == '__main__':
    main()


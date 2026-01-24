#!/usr/bin/env python3
"""
统一检查脚本：扫描目录中所有 parquet 文件，找出所有有问题的字段
"""

import sys
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Features, Image as DatasetImage
from tqdm import tqdm

def check_field_type(feature, field_name=None):
    """
    检查字段类型是否有问题
    返回: (is_problematic, issue_type, details)
    """
    # 检查是否是普通 dict（不是 Features 对象）
    if isinstance(feature, dict) and not isinstance(feature, Features):
        return True, "普通 dict", f"键: {list(feature.keys()) if hasattr(feature, 'keys') else 'N/A'}"
    
    # 检查是否是 List/Sequence 包含普通 dict
    if hasattr(feature, "feature"):
        inner = feature.feature
        if isinstance(inner, dict) and not isinstance(inner, Features):
            # 跳过 images 字段的特殊结构（有 bytes 和 path 键）
            if field_name == "images" and hasattr(inner, "keys") and "bytes" in inner and "path" in inner:
                # images 字段需要转换为 Image 类型，不算作"问题字段"（会被单独处理）
                return False, None, None
            return True, "List/Sequence 包含普通 dict", f"内部键: {list(inner.keys()) if hasattr(inner, 'keys') else 'N/A'}"
    
    return False, None, None

def scan_file(parquet_path):
    """扫描单个文件，返回有问题的字段列表"""
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_path))['train']
        problematic_fields = []
        
        for col in dataset.column_names:
            if col not in dataset.features:
                continue
            
            feature = dataset.features[col]
            is_problematic, issue_type, details = check_field_type(feature, field_name=col)
            
            if is_problematic:
                problematic_fields.append({
                    "field": col,
                    "issue_type": issue_type,
                    "details": details,
                    "feature_type": type(feature).__name__
                })
        
        return problematic_fields, len(dataset)
    except Exception as e:
        print(f"  ❌ 扫描失败: {type(e).__name__}: {e}")
        return None, 0

def main():
    if len(sys.argv) < 2:
        print("用法: python3 check_all_problematic_fields.py <parquet_file_or_directory>")
        print("\n示例:")
        print("  python3 check_all_problematic_fields.py /path/to/file.parquet")
        print("  python3 check_all_problematic_fields.py /path/to/directory -r")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    recursive = "-r" in sys.argv or "--recursive" in sys.argv
    
    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        sys.exit(1)
    
    # 收集所有 parquet 文件
    if input_path.is_file() and input_path.suffix == ".parquet":
        parquet_files = [input_path]
    elif input_path.is_dir():
        if recursive:
            parquet_files = list(input_path.rglob("*.parquet"))
        else:
            parquet_files = list(input_path.glob("*.parquet"))
    else:
        print(f"❌ 无效的路径: {input_path}")
        sys.exit(1)
    
    if not parquet_files:
        print(f"❌ 未找到 parquet 文件")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"扫描 {len(parquet_files)} 个 parquet 文件")
    print(f"{'='*80}\n")
    
    # 统计所有有问题的字段
    all_problematic_fields = defaultdict(lambda: {
        "issue_types": set(),
        "files": [],
        "total_samples": 0
    })
    
    file_results = []
    
    for parquet_file in tqdm(parquet_files, desc="扫描文件"):
        problematic_fields, num_samples = scan_file(parquet_file)
        
        if problematic_fields is None:
            continue
        
        file_results.append({
            "file": parquet_file,
            "problematic_fields": problematic_fields,
            "num_samples": num_samples
        })
        
        # 统计每个问题字段
        for field_info in problematic_fields:
            field_name = field_info["field"]
            all_problematic_fields[field_name]["issue_types"].add(field_info["issue_type"])
            all_problematic_fields[field_name]["files"].append(str(parquet_file))
            all_problematic_fields[field_name]["total_samples"] += num_samples
    
    # 打印详细报告
    print(f"\n{'='*80}")
    print("详细报告")
    print(f"{'='*80}\n")
    
    for file_result in file_results:
        print(f"文件: {file_result['file']}")
        print(f"  样本数: {file_result['num_samples']}")
        if file_result['problematic_fields']:
            print(f"  问题字段:")
            for field_info in file_result['problematic_fields']:
                print(f"    - {field_info['field']}: {field_info['issue_type']}")
                print(f"      类型: {field_info['feature_type']}, {field_info['details']}")
        else:
            print(f"  ✅ 无问题字段")
        print()
    
    # 打印汇总统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}\n")
    
    if all_problematic_fields:
        print(f"发现 {len(all_problematic_fields)} 个有问题的字段:\n")
        for field_name, info in sorted(all_problematic_fields.items()):
            print(f"  {field_name}:")
            print(f"    问题类型: {', '.join(info['issue_types'])}")
            print(f"    出现在 {len(info['files'])} 个文件中")
            print(f"    总样本数: {info['total_samples']}")
            print(f"    文件列表:")
            for file_path in info['files'][:5]:  # 只显示前5个
                print(f"      - {file_path}")
            if len(info['files']) > 5:
                print(f"      ... 还有 {len(info['files']) - 5} 个文件")
            print()
        
        # 生成建议的 FIELDS_TO_REMOVE 列表
        print(f"\n{'='*80}")
        print("建议的 FIELDS_TO_REMOVE 列表")
        print(f"{'='*80}\n")
        print("# 需要移除的字段列表（这些字段在训练时用不到，且有特征类型问题）")
        print("FIELDS_TO_REMOVE = [")
        for field_name in sorted(all_problematic_fields.keys()):
            issue_types = all_problematic_fields[field_name]["issue_types"]
            issue_desc = ", ".join(issue_types)
            print(f'    "{field_name}",      # {issue_desc}')
        print("]")
    else:
        print("✅ 所有文件都没有发现明显的问题字段！")
        print("   注意：images 字段如果有问题会被单独处理，不在此列表中")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()


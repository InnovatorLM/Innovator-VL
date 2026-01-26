#!/usr/bin/env python3
"""
扫描 parquet 文件，找出可能有特征类型问题的字段
"""

import sys
from pathlib import Path
from datasets import load_dataset, Features

def scan_fields(parquet_path):
    """扫描字段，找出可能有问题的字段"""
    print(f"\n{'='*80}")
    print(f"扫描文件: {parquet_path}")
    print(f"{'='*80}\n")
    
    try:
        dataset = load_dataset("parquet", data_files=str(parquet_path))['train']
        print(f"数据集大小: {len(dataset)}")
        print(f"所有字段: {dataset.column_names}\n")
        
        problematic_fields = []
        
        print("检查每个字段的特征类型:\n")
        for col in dataset.column_names:
            if col not in dataset.features:
                print(f"  ⚠️  {col}: 不在 features 中")
                continue
                
            feature = dataset.features[col]
            feature_type = type(feature).__name__
            
            # 检查是否是普通 dict（不是 Features 对象）
            is_plain_dict = isinstance(feature, dict) and not isinstance(feature, Features)
            
            # 检查是否是 List/Sequence 包含普通 dict
            has_nested_dict = False
            if hasattr(feature, "feature"):
                inner = feature.feature
                if isinstance(inner, dict) and not isinstance(inner, Features):
                    has_nested_dict = True
            
            if is_plain_dict:
                print(f"  ❌ {col}: {feature_type}")
                print(f"     问题: 普通 dict（不是 Features 对象）")
                print(f"     键: {list(feature.keys()) if hasattr(feature, 'keys') else 'N/A'}")
                problematic_fields.append((col, "普通 dict"))
            elif has_nested_dict:
                print(f"  ❌ {col}: {feature_type}")
                print(f"     问题: List/Sequence 包含普通 dict")
                inner = feature.feature
                print(f"     内部键: {list(inner.keys()) if hasattr(inner, 'keys') else 'N/A'}")
                problematic_fields.append((col, "List/Sequence 包含普通 dict"))
            else:
                print(f"  ✓  {col}: {feature_type}")
        
        print(f"\n{'='*80}")
        if problematic_fields:
            print(f"发现 {len(problematic_fields)} 个可能有问题的字段:")
            for field, issue_type in problematic_fields:
                print(f"  - {field}: {issue_type}")
        else:
            print("✅ 未发现明显的问题字段")
        print(f"{'='*80}\n")
        
        return problematic_fields
        
    except Exception as e:
        print(f"❌ 扫描失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 scan_problematic_fields.py <parquet_file_path>")
        print("\n示例:")
        print("  python3 scan_problematic_fields.py /path/to/file.parquet")
        sys.exit(1)
    
    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print(f"❌ 文件不存在: {parquet_path}")
        sys.exit(1)
    
    problematic_fields = scan_fields(parquet_path)
    sys.exit(0 if not problematic_fields else 1)


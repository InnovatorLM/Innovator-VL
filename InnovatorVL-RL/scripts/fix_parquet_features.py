#!/usr/bin/env python3
"""
预处理脚本：修复 parquet 文件中的特征类型问题
- 移除有问题的字段（训练时用不到）
- 统一 images 字段类型（转换为 Sequence(Image(...))），必须统一才能 concatenate_datasets
"""

import sys
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, Features, Sequence, Image as DatasetImage
from tqdm import tqdm

# 需要移除的字段列表（这些字段在训练时用不到，且有特征类型问题）
# 注意：如果发现其他字段也有类似问题（普通 dict 或 List 包含普通 dict），
# 可以添加到这个列表中
FIELDS_TO_REMOVE = [
    "reward_model",      # 普通 dict，训练时不需要
    "vanilla_prompt",   # List 包含普通 dict，训练时不需要
    # 可以根据需要添加其他字段，例如：
    # "other_problematic_field",
]

# Images 字段统一目标类型
# 选项：
# - "image": 统一为 Sequence(Image(decode=True)) - 推荐，datasets 库标准格式
# - "dict": 统一为 List({'bytes': Value('binary'), 'path': Value('string')}) - 原始格式
IMAGES_TARGET_TYPE = "image"  # 或 "dict"

def fix_dataset_features(dataset):
    """修复数据集的特征类型：移除问题字段，统一 images 字段类型"""
    
    # 1. 移除有问题的字段
    columns_to_keep = [col for col in dataset.column_names if col not in FIELDS_TO_REMOVE]
    removed_fields = [col for col in dataset.column_names if col in FIELDS_TO_REMOVE]
    
    if removed_fields:
        print(f"  移除字段: {', '.join(removed_fields)}")
        # 只保留需要的列
        dataset = dataset.select_columns(columns_to_keep)
    else:
        print(f"  无需移除字段")
    
    # 2. 统一 images 字段类型（必须统一，否则无法 concatenate_datasets）
    if "images" in dataset.column_names:
        img_feature = dataset.features["images"]
        needs_conversion = False
        target_type = None
        
        # 检查当前类型和目标类型
        if hasattr(img_feature, "feature"):
            inner = img_feature.feature
            if IMAGES_TARGET_TYPE == "image":
                # 目标：Sequence(Image(...))
                if isinstance(inner, dict) and not isinstance(inner, Features):
                    needs_conversion = True
                    target_type = "image"
                elif not isinstance(inner, DatasetImage):
                    needs_conversion = True
                    target_type = "image"
            else:
                # 目标：List({'bytes': ..., 'path': ...})
                if isinstance(inner, DatasetImage):
                    needs_conversion = True
                    target_type = "dict"
                elif not (isinstance(inner, dict) and not isinstance(inner, Features)):
                    needs_conversion = True
                    target_type = "dict"
        else:
            needs_conversion = True
            target_type = IMAGES_TARGET_TYPE
        
        if needs_conversion:
            if target_type == "image":
                print(f"  统一 images 字段类型为 Sequence(Image(...))...")
                new_features_dict = dict(dataset.features)
                new_features_dict["images"] = Sequence(DatasetImage(decode=True))
                new_features = Features(new_features_dict)
                
                try:
                    dataset = dataset.cast(new_features)
                    print(f"  ✅ images 字段统一成功")
                except Exception as e:
                    print(f"  ⚠️  cast 失败: {e}")
                    print(f"  使用 from_dict 重新创建...")
                    # 重新创建数据集
                    data_dict = {col: [dataset[j][col] for j in range(len(dataset))] 
                                for col in dataset.column_names}
                    dataset = Dataset.from_dict(data_dict, features=new_features)
                    print(f"  ✅ 通过 from_dict 成功统一 images 字段")
            else:
                # 统一为 dict 格式
                print(f"  统一 images 字段类型为 List({{'bytes': ..., 'path': ...}})...")
                from datasets import Value
                from datasets.features.features import List as FeaturesList
                
                new_features_dict = dict(dataset.features)
                new_features_dict["images"] = FeaturesList({
                    "bytes": Value("binary"),
                    "path": Value("string")
                })
                new_features = Features(new_features_dict)
                
                try:
                    # 如果当前是 Image 类型，需要转换为 dict
                    # 注意：这需要重新编码图像，可能比较慢
                    print(f"  ⚠️  从 Image 类型转换为 dict 格式，这可能需要重新编码图像...")
                    data_dict = {col: [dataset[j][col] for j in range(len(dataset))] 
                                for col in dataset.column_names}
                    
                    # 转换 images 数据
                    import io
                    from PIL import Image
                    converted_images = []
                    for img_list in data_dict["images"]:
                        converted_img_list = []
                        for img in img_list:
                            if isinstance(img, Image.Image):
                                # PIL Image -> dict with bytes
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format='PNG')
                                converted_img_list.append({
                                    "bytes": img_bytes.getvalue(),
                                    "path": None
                                })
                            elif isinstance(img, dict):
                                # 已经是 dict，保持原样
                                converted_img_list.append(img)
                            else:
                                # 其他类型，尝试转换
                                converted_img_list.append({
                                    "bytes": None,
                                    "path": str(img) if img else None
                                })
                        converted_images.append(converted_img_list)
                    data_dict["images"] = converted_images
                    
                    dataset = Dataset.from_dict(data_dict, features=new_features)
                    print(f"  ✅ 通过 from_dict 成功统一 images 字段为 dict 格式")
                except Exception as e:
                    print(f"  ❌ 转换失败: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
        else:
            print(f"  ✓ images 字段类型已正确")
    
    return dataset


def process_parquet_file(input_path, output_path=None):
    """处理单个 parquet 文件"""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return False
    
    if output_path is None:
        # 默认输出到同目录，添加 _fixed 后缀
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"\n{'='*80}")
    print(f"处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"{'='*80}\n")
    
    try:
        # 加载数据集
        print("加载数据集...")
        dataset = load_dataset("parquet", data_files=str(input_path))['train']
        print(f"数据集大小: {len(dataset)}")
        print(f"列名: {dataset.column_names}\n")
        
        # 检查原始特征
        print("原始列名:")
        print(f"  {', '.join(dataset.column_names)}")
        
        # 修复特征
        print("\n修复特征...")
        fixed_dataset = fix_dataset_features(dataset)
        
        # 验证修复结果
        print("\n修复后的列名:")
        print(f"  {', '.join(fixed_dataset.column_names)}")
        
        # 检查 images 字段
        if "images" in fixed_dataset.features:
            img_feature = fixed_dataset.features["images"]
            if hasattr(img_feature, "feature") and isinstance(img_feature.feature, DatasetImage):
                print(f"  ✅ images 字段类型正确: Sequence(Image(...))")
            else:
                print(f"  ⚠️  images 字段类型: {type(img_feature)}")
                if hasattr(img_feature, "feature"):
                    print(f"     内部类型: {type(img_feature.feature)}")
        
        # 保存修复后的数据集
        print(f"\n保存到: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_dataset.to_parquet(str(output_path))
        print(f"✅ 成功保存修复后的数据集")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="修复 parquet 文件中的特征类型问题")
    parser.add_argument("input", help="输入的 parquet 文件路径或目录")
    parser.add_argument("-o", "--output", help="输出文件路径或目录（可选，默认添加 _fixed 后缀）")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理目录中的所有 parquet 文件")
    parser.add_argument("--overwrite", action="store_true", help="覆盖原文件（默认是创建新文件）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == ".parquet":
        # 处理单个文件
        if args.overwrite:
            # 覆盖原文件
            process_parquet_file(input_path, input_path)
        else:
            process_parquet_file(input_path, args.output)
    elif input_path.is_dir():
        # 处理目录
        if args.recursive:
            parquet_files = list(input_path.rglob("*.parquet"))
        else:
            parquet_files = list(input_path.glob("*.parquet"))
        
        if not parquet_files:
            print(f"❌ 在 {input_path} 中未找到 parquet 文件")
            return
        
        print(f"找到 {len(parquet_files)} 个 parquet 文件\n")
        
        success_count = 0
        for parquet_file in tqdm(parquet_files, desc="处理文件"):
            if args.overwrite:
                # 覆盖原文件
                if process_parquet_file(parquet_file, parquet_file):
                    success_count += 1
            else:
                # 创建新文件
                if args.output:
                    # 如果指定了输出目录，保持相对路径结构
                    output_dir = Path(args.output)
                    relative_path = parquet_file.relative_to(input_path)
                    output_file = output_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    if process_parquet_file(parquet_file, output_file):
                        success_count += 1
                else:
                    # 默认在同目录创建 _fixed 文件
                    if process_parquet_file(parquet_file):
                        success_count += 1
        
        print(f"\n{'='*80}")
        print(f"处理完成: {success_count}/{len(parquet_files)} 个文件成功")
        print(f"{'='*80}")
    else:
        print(f"❌ 无效的输入路径: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
完整预处理脚本：不仅统一 features 类型，还统一数据内容
- 统一 images 字段类型
- 确保每个样本的 <image> token 数量与图像数量匹配
- 验证数据一致性
"""

import sys
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, Features, Sequence, Image as DatasetImage, Value
from datasets.features.features import List as FeaturesList
from tqdm import tqdm
import io
from PIL import Image

# 需要移除的字段列表
FIELDS_TO_REMOVE = [
    "reward_model",
    "vanilla_prompt",
]

def ensure_token_image_match(sample):
    """确保 <image> token 数量与图像数量匹配"""
    problem_text = sample.get("problem", "")
    images = sample.get("images", [])
    
    if images is None:
        images = []
    
    num_image_tokens = problem_text.count("<image>")
    num_images = len(images) if images else 0
    
    if num_image_tokens != num_images:
        if num_image_tokens > num_images:
            # 移除多余的 <image> token
            parts = problem_text.split("<image>")
            if len(parts) > num_images + 1:
                problem_text = "<image>".join(parts[:num_images + 1])
        else:
            # 截断多余的图像
            images = images[:num_image_tokens]
        
        sample["problem"] = problem_text
        sample["images"] = images
    
    return sample

def preprocess_dataset(input_path, output_path=None):
    """完整预处理数据集"""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"\n{'='*80}")
    print(f"预处理文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"{'='*80}\n")
    
    try:
        # 1. 加载数据集
        print("1. 加载数据集...")
        dataset = load_dataset("parquet", data_files=str(input_path))['train']
        print(f"   数据集大小: {len(dataset)}")
        print(f"   列名: {dataset.column_names}\n")
        
        # 2. 移除问题字段
        print("2. 移除问题字段...")
        columns_to_keep = [col for col in dataset.column_names if col not in FIELDS_TO_REMOVE]
        removed_fields = [col for col in dataset.column_names if col in FIELDS_TO_REMOVE]
        if removed_fields:
            print(f"   移除: {', '.join(removed_fields)}")
            dataset = dataset.select_columns(columns_to_keep)
        else:
            print(f"   无需移除字段")
        
        # 3. 统一 images 字段类型
        print("\n3. 统一 images 字段类型...")
        if "images" in dataset.column_names:
            img_feature = dataset.features["images"]
            needs_conversion = False
            
            if hasattr(img_feature, "feature"):
                inner = img_feature.feature
                if isinstance(inner, dict) and not isinstance(inner, Features):
                    needs_conversion = True
                elif not isinstance(inner, DatasetImage):
                    needs_conversion = True
            else:
                needs_conversion = True
            
            if needs_conversion:
                new_features_dict = dict(dataset.features)
                new_features_dict["images"] = Sequence(DatasetImage(decode=True))
                new_features = Features(new_features_dict)
                
                try:
                    dataset = dataset.cast(new_features)
                    print(f"   ✅ 转换成功")
                except Exception as e:
                    print(f"   ⚠️  cast 失败: {e}")
                    print(f"   使用 from_dict 重新创建...")
                    data_dict = {col: [dataset[j][col] for j in range(len(dataset))] 
                                for col in dataset.column_names}
                    dataset = Dataset.from_dict(data_dict, features=new_features)
                    print(f"   ✅ 通过 from_dict 成功转换")
            else:
                print(f"   ✓ 已经是正确的类型")
        
        # 4. 确保 token 和图像数量匹配
        print("\n4. 确保 <image> token 数量与图像数量匹配...")
        fixed_count = 0
        total_count = len(dataset)
        
        def fix_sample(sample):
            nonlocal fixed_count
            problem_text = sample.get("problem", "")
            images = sample.get("images", [])
            
            if images is None:
                images = []
            
            num_image_tokens = problem_text.count("<image>")
            num_images = len(images) if images else 0
            
            if num_image_tokens != num_images:
                fixed_count += 1
                if num_image_tokens > num_images:
                    parts = problem_text.split("<image>")
                    if len(parts) > num_images + 1:
                        sample["problem"] = "<image>".join(parts[:num_images + 1])
                else:
                    sample["images"] = images[:num_image_tokens]
            
            return sample
        
        dataset = dataset.map(fix_sample, desc="修复 token 和图像数量匹配")
        print(f"   ✅ 修复了 {fixed_count}/{total_count} 个样本")
        
        # 5. 验证结果
        print("\n5. 验证预处理结果...")
        # 检查 images 字段类型
        if "images" in dataset.features:
            img_feature = dataset.features["images"]
            if hasattr(img_feature, "feature") and isinstance(img_feature.feature, DatasetImage):
                print(f"   ✅ images 字段类型正确: Sequence(Image(...))")
            else:
                print(f"   ⚠️  images 字段类型: {type(img_feature)}")
        
        # 随机检查几个样本
        print(f"\n   随机检查 5 个样本的 token 和图像数量匹配:")
        import random
        indices = random.sample(range(min(5, len(dataset))), min(5, len(dataset)))
        for idx in indices:
            sample = dataset[idx]
            problem_text = sample.get("problem", "")
            images = sample.get("images", [])
            num_tokens = problem_text.count("<image>")
            num_images = len(images) if images else 0
            status = "✅" if num_tokens == num_images else "❌"
            print(f"     样本 {idx}: {status} tokens={num_tokens}, images={num_images}")
        
        # 6. 保存
        print(f"\n6. 保存预处理后的数据集...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(output_path))
        print(f"   ✅ 保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="完整预处理数据集：统一格式并验证一致性")
    parser.add_argument("input", help="输入的 parquet 文件路径或目录")
    parser.add_argument("-o", "--output", help="输出文件路径或目录（可选）")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理目录中的所有 parquet 文件")
    parser.add_argument("--overwrite", action="store_true", help="覆盖原文件")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == ".parquet":
        if args.overwrite:
            preprocess_dataset(input_path, input_path)
        else:
            preprocess_dataset(input_path, args.output)
    elif input_path.is_dir():
        if args.recursive:
            parquet_files = list(input_path.rglob("*.parquet"))
        else:
            parquet_files = list(input_path.glob("*.parquet"))
        
        if not parquet_files:
            print(f"❌ 未找到 parquet 文件")
            return
        
        print(f"找到 {len(parquet_files)} 个 parquet 文件\n")
        
        success_count = 0
        for parquet_file in tqdm(parquet_files, desc="处理文件"):
            if args.overwrite:
                if preprocess_dataset(parquet_file, parquet_file):
                    success_count += 1
            else:
                if args.output:
                    output_dir = Path(args.output)
                    relative_path = parquet_file.relative_to(input_path)
                    output_file = output_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    if preprocess_dataset(parquet_file, output_file):
                        success_count += 1
                else:
                    if preprocess_dataset(parquet_file):
                        success_count += 1
        
        print(f"\n{'='*80}")
        print(f"处理完成: {success_count}/{len(parquet_files)} 个文件成功")
        print(f"{'='*80}")
    else:
        print(f"❌ 无效的输入路径: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()


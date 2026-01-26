import os
import glob
from datasets import load_dataset
from tqdm import tqdm

def inspect_datasets(root_path):
    # 1. 递归查找所有 parquet 文件
    pattern = os.path.join(root_path, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        print(f"❌ 在 {root_path} 下未找到任何 parquet 文件。")
        return

    print(f"🔍 找到 {len(files)} 个文件，开始检查特征结构...\n")

    reference_features = None
    reference_file = None
    issues_found = 0

    for f in tqdm(files):
        try:
            # 仅加载元数据/第一条样本以节省内存和时间
            ds = load_dataset("parquet", data_files=f, split="train")
            current_features = ds.features
            
            if reference_features is None:
                reference_features = current_features
                reference_file = f
                print(f"✅ 基准文件设定为: {os.path.basename(f)}")
                continue
            
            # 对比特征
            if current_features != reference_features:
                issues_found += 1
                print(f"\n❌ [发现不一致] 文件: {f}")
                print(f"   - 基准列 ({os.path.basename(reference_file)}): {list(reference_features.keys())}")
                print(f"   - 当前列: {list(current_features.keys())}")
                
                # 特别检查报错的 images 字段
                if "images" in current_features and "images" in reference_features:
                    if current_features["images"] != reference_features["images"]:
                        print(f"   ⚠️ 警告: 'images' 字段类型冲突!")
                        print(f"      基准类型: {reference_features['images']}")
                        print(f"      当前类型: {current_features['images']}")
                
                # 检查是否存在某一列在某些文件中是 Dict, 在某些是 Value(null)
                for col in set(current_features.keys()) & set(reference_features.keys()):
                    if type(current_features[col]) != type(reference_features[col]):
                        print(f"   ⚠️ 警告: 字段 '{col}' 类型不匹配 (可能会触发 AttributeError)")

        except Exception as e:
            print(f"❌ 无法读取文件 {f}: {e}")

    if issues_found == 0:
        print("\n✨ 未发现结构性冲突，所有文件的 Schema 均一致。")
    else:
        print(f"\n💡 排查完成，共发现 {issues_found} 个文件存在潜在冲突。")

# 使用方法：
inspect_datasets("/root/innovator_data_wenzichen/0102_innovator_vl_RL_Data_Merged_debug")
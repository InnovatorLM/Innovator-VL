import os
import glob
from datasets import load_dataset
from collections import Counter
import pandas as pd

def analyze_shards(root_path, sample_size=5):
    pattern = os.path.join(root_path, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    
    summary = []
    all_column_stats = Counter()

    print(f"🔍 正在分析 {len(files)} 个分片...")

    for f in files:
        try:
            # 仅加载 schema 和前几行
            ds = load_dataset("parquet", data_files=f, split="train")
            num_rows = len(ds)
            columns = ds.column_names
            all_column_stats.update(columns)
            
            # 获取 images 字段的详细类型
            img_type = ds.features.get("images", "Missing")
            
            summary.append({
                "file": os.path.basename(f),
                "path": f,
                "rows": num_rows,
                "columns": ",".join(columns),
                "images_type": str(img_type)
            })
        except Exception as e:
            print(f"❌ 无法读取 {f}: {e}")

    # 转换成 DataFrame 方便观察
    df = pd.DataFrame(summary)
    
    print("\n" + "="*50)
    print("📊 字段出现频率汇总 (用于确定 standard_features):")
    for col, count in all_column_stats.items():
        print(f"- {col}: 存在于 {count}/{len(files)} 个文件中")
    
    print("\n" + "="*50)
    print("🖼️ Images 字段类型分布:")
    print(df["images_type"].value_counts())
    
    # 找出那些字段缺失的文件
    expected_cols = set(all_column_stats.keys())
    print("\n" + "="*50)
    print("⚠️ 结构异常检查:")
    for _, row in df.iterrows():
        missing = expected_cols - set(row["columns"].split(","))
        if missing:
            print(f"文件 {row['file']} 缺少字段: {missing}")

    return df

# 使用
df_results = analyze_shards("/mnt/innovator/data/wenzichen/0102_innovator_vl_RL_Data_Merged")
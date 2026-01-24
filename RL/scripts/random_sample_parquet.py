#!/usr/bin/env python3
"""
随机抽取parquet文件并复制到新目录
"""
import os
import random
import shutil
from pathlib import Path
from datetime import datetime

# 源目录和目标目录
source_dir = "/mnt/innovator/data/chenshuang/RL_DATA/mm_science_vqa_rl_no_img/1229"
base_target_dir = "/root/innovator_data_wenzichen"

# 创建带时间戳的新目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
target_dir = os.path.join(base_target_dir, f"mm_science_vqa_rl_no_img_{timestamp}")

# 获取所有parquet文件
print(f"正在扫描源目录: {source_dir}")
parquet_files = list(Path(source_dir).glob("*.parquet"))
print(f"找到 {len(parquet_files)} 个parquet文件")

if len(parquet_files) < 10:
    print(f"警告: 只有 {len(parquet_files)} 个文件，少于10个")
    selected_files = parquet_files
else:
    # 随机选择10个文件
    selected_files = random.sample(parquet_files, 10)

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)
print(f"创建目标目录: {target_dir}")

# 复制选中的文件
print(f"\n开始复制 {len(selected_files)} 个文件...")
for i, file_path in enumerate(selected_files, 1):
    dest_path = os.path.join(target_dir, file_path.name)
    shutil.copy2(file_path, dest_path)
    print(f"[{i}/{len(selected_files)}] 已复制: {file_path.name}")

print(f"\n完成! 文件已保存到: {target_dir}")
print(f"\n选中的文件列表:")
for file_path in selected_files:
    print(f"  - {file_path.name}")


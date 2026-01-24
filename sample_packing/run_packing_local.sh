#!/bin/bash

# --- Set a custom cache directory on a larger disk to avoid "No space left on device" errors ---
# We create a directory on the innovator mount, which likely has more space than the root partition.
export HF_DATASETS_CACHE="/mnt/innovator/data/wenzichen/.cache/huggingface_datasets"
mkdir -p $HF_DATASETS_CACHE

# 确保脚本在遇到错误时立即退出
set -e
set -u

# 一个简单的函数，用于打印日志并执行Python脚本
run_python_script() {
    local script_name=$1
    echo ">>>>>>>>>>> [START] Executing $script_name >>>>>>>>>>>>>>"
    # 使用 python3，或者您环境中的python可执行文件名
    python3 "$script_name"
    echo ">>>>>>>>>>> [DONE] Finished $script_name >>>>>>>>>>>>>>"
}

# --- 工作流开始 ---

# 0. 解析HuggingFace原始数据
run_python_script "huggingface_data_parse_v2.py"

# 1. 计算每个样本的Token长度
run_python_script "1_s1_get_tokenlens_v3-sft.py"

# 2. 执行Hash-Bucket打包算法
run_python_script "2_do_hashbacket.py"

# 3. 根据打包结果生成新的样本文件
run_python_script "3_s2_prepare_rawsamples-vqa.py"

# 4. 将打包好的样本转换为WebDataset格式
# 注意: 您当前打开的文件是 4_convert_packedsample_to_wds_multiprocess.py
# 如果您想使用它，请在这里替换文件名。否则使用默认的单进程版本。
run_python_script "4_convert_packedsample_to_wds_multiprocess.py" 

# 5. 生成最终的WDS配置文件
# 在单机环境下，第4步已经生成了配置文件，但这一步会确保最终配置正确。
run_python_script "5_make_mix_wds_config.py"

# echo "───────────────── All processing workflows have been successfully completed. ───────────────────"
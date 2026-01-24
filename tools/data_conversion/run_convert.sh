#!/bin/bash
# 使用innovator_vl_stable conda环境运行转换脚本

if [ -f /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/public/wenzichen/miniconda3/etc/profile.d/conda.sh
  conda activate innovator_vl_stable || true
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活conda环境并运行脚本
conda run -n innovator_vl_stable python "${SCRIPT_DIR}/convert_to_webdataset.py" "$@"


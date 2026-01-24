#!/bin/bash
# 修复异常文件的脚本
# 1. 删除异常格式的tar文件
# 2. 删除对应的.progress标记（强制重新转换）
# 3. 使用resume模式重新转换这些子目录

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data-webdataset"
INPUT_DIR="/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data"
FILES_TO_DELETE="$OUTPUT_DIR/files_to_delete.txt"
SUBDIRS_TO_REDO="$OUTPUT_DIR/subdirs_to_redo.txt"

echo "============================================================"
echo "修复异常文件脚本"
echo "============================================================"
echo ""

# 检查文件是否存在
if [ ! -f "$FILES_TO_DELETE" ]; then
    echo "错误: 找不到异常文件列表: $FILES_TO_DELETE"
    exit 1
fi

if [ ! -f "$SUBDIRS_TO_REDO" ]; then
    echo "错误: 找不到子目录列表: $SUBDIRS_TO_REDO"
    exit 1
fi

# 统计要删除的文件
ABNORMAL_COUNT=$(tail -n +2 "$FILES_TO_DELETE" | wc -l)
echo "发现 $ABNORMAL_COUNT 个异常文件需要删除"
echo ""

# 读取要删除的文件列表（跳过第一行标题）
ABNORMAL_FILES=$(tail -n +2 "$FILES_TO_DELETE" | awk '{print $1}')

echo "开始删除异常文件..."
DELETED_COUNT=0
for file in $ABNORMAL_FILES; do
    file_path="$OUTPUT_DIR/$file"
    if [ -f "$file_path" ]; then
        rm -f "$file_path"
        # 同时删除对应的索引文件（如果存在）
        rm -f "${file_path}.idx"
        DELETED_COUNT=$((DELETED_COUNT + 1))
        if [ $((DELETED_COUNT % 50)) -eq 0 ]; then
            echo "  已删除 $DELETED_COUNT 个文件..."
        fi
    fi
done
echo "已删除 $DELETED_COUNT 个异常文件"
echo ""

# 删除对应的.progress标记文件（强制重新转换）
echo "删除对应的.progress标记文件..."
SUBDIRS_COUNT=$(tail -n +1 "$SUBDIRS_TO_REDO" | wc -l)
PROGRESS_DELETED=0
while IFS= read -r subdir; do
    if [ -n "$subdir" ]; then
        progress_file="$OUTPUT_DIR/.progress/${subdir}.done"
        if [ -f "$progress_file" ]; then
            rm -f "$progress_file"
            PROGRESS_DELETED=$((PROGRESS_DELETED + 1))
        fi
    fi
done < "$SUBDIRS_TO_REDO"
echo "已删除 $PROGRESS_DELETED 个.progress标记文件"
echo ""

# 同时处理ureader_chart（如果它不在列表中）
if ! grep -q "^ureader_chart$" "$SUBDIRS_TO_REDO"; then
    echo "检测到ureader_chart未完成，也将一并处理"
    # 删除不完整的最后一个文件
    if [ -f "$OUTPUT_DIR/ureader_chart_450038.tar" ]; then
        rm -f "$OUTPUT_DIR/ureader_chart_450038.tar"
        rm -f "$OUTPUT_DIR/ureader_chart_450038.tar.idx"
        echo "  已删除不完整的ureader_chart_450038.tar"
    fi
    # 不删除.done文件（如果不存在的话），让resume模式继续处理
fi

echo ""
echo "============================================================"
echo "准备重新转换"
echo "============================================================"
echo ""
echo "将被重新转换的子目录数: $SUBDIRS_COUNT"
echo ""
echo "接下来将运行转换脚本（使用resume模式）"
echo "resume模式会："
echo "  - 跳过已完成的子目录（不会被重新转换）"
echo "  - 处理需要修复的子目录"
echo "  - 继续处理未完成的ureader_chart"
echo ""
echo "注意：如果您当前有正在运行的转换进程，请先停止它"
echo ""
echo "按 Ctrl+C 取消，或按 Enter 继续..."
read

# 运行转换脚本
cd "$SCRIPT_DIR"
echo ""
echo "开始重新转换..."
conda run -n innovator_vl_stable python convert_to_webdataset.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 8 \
    --resume \
    --batch-size 1000

echo ""
echo "============================================================"
echo "转换完成！"
echo "============================================================"


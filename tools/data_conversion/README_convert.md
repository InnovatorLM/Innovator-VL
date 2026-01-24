# LLaVA训练数据转换为WebDataset格式

这个脚本用于将 Innovator-VL-Insturct-Data 数据集的 parquet 格式转换为 WebDataset 格式。

## 功能特性

1. **自动遍历所有子目录**：自动处理数据目录下的所有子目录
2. **数据校验**：
   - 检查样本是否有图像文件
   - 检查 prompt 中包含 `<image>` 标记时是否有对应的图像
   - 验证图像格式的有效性
3. **自动跳过无效样本**：默认跳过无效样本，避免训练时出错
4. **分片存储**：将数据分成多个 tar 文件（shards），方便并行处理和传输

## 使用方法

### 方法1: 使用便捷脚本（推荐）

```bash
# 进入脚本目录
cd /root/innovator_code_wenzichen/Innovator-VL/tools/data_conversion

# 使用默认参数
bash run_convert.sh

# 指定输入和输出目录
bash run_convert.sh \
    --input_dir /path/to/Innovator-VL-Insturct-Data \
    --output_dir /path/to/output

# 自定义每个shard的样本数和并发进程数
bash run_convert.sh \
    --input_dir /path/to/data \
    --output_dir /path/to/output \
    --max_samples_per_shard 5000 \
    --num_workers 8  # 使用8个进程并发处理，大幅提升速度

# 如果转换中断，可以使用resume继续
bash run_convert.sh \
    --input_dir /path/to/data \
    --output_dir /path/to/output \
    --resume  # 跳过已完成的子目录，继续处理未完成的
```

### 方法2: 直接使用conda环境

```bash
conda run -n innovator_vl_stable python /root/innovator_code_wenzichen/Innovator-VL/tools/data_conversion/convert_to_webdataset.py \
    --input_dir /root/innovator_data_wenzichen/Innovator-VL-Insturct-Data \
    --output_dir /root/innovator_data_wenzichen/webdataset_output
```

## 参数说明

- `--input_dir`: 输入数据目录（默认: `/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data`）
- `--output_dir`: 输出WebDataset目录（默认: `/root/innovator_data_wenzichen/webdataset_output`）
- `--max_samples_per_shard`: 每个shard的最大样本数（默认: 10000）
- `--skip_invalid`: 跳过无效样本（默认: True）
- `--no-skip-invalid`: 不跳过无效样本
- `--log_level`: 日志级别，可选 DEBUG, INFO, WARNING, ERROR（默认: INFO）
- `--num_workers`: 并发进程数（默认: 4）。建议设置为 CPU 核心数，可以大幅提升处理速度
- `--resume`: 启用断点续传模式。如果转换中断，使用此参数可以跳过已完成的子目录，从中断处继续

## 输出格式

每个 shard 是一个 tar 文件，包含多个样本。每个样本包含：

- `{sample_id}.json`: 包含样本的元数据（id, conversations, data_source）
- `{sample_id}.jpg`: 图像数据（JPEG格式），如果没有图像则为空

## 数据格式说明

### 输入格式（Parquet）
- `id`: 样本ID（字符串）
- `image`: 图像字典 `{'bytes': b'...', 'path': '...'}`
- `conversations`: 对话列表，每个元素是 `{'role': 'user/assistant', 'content': '...'}`
- `data_source`: 数据来源（字符串）

### 输出格式（WebDataset）
- JSON字段包含完整的样本信息
- JPG字段包含图像的JPEG字节数据

## 校验规则

脚本会检查以下问题并跳过相应样本：

1. **缺少图像但prompt中有`<image>`标记**：prompt中提到了图像但实际没有图像数据
2. **图像格式无效**：图像无法解析或尺寸为0
3. **conversations为空**：对话数据为空或格式错误

## 性能优化

- **并发处理**：使用 `--num_workers` 参数设置并发进程数，可以大幅提升处理速度
  - 建议设置为 CPU 核心数（如 8、16、32）
  - 注意内存使用：每个进程会加载parquet文件到内存，确保有足够内存
  - 如果内存不足，可以减少 `num_workers` 数量

- **性能建议**：
  - 4核CPU：`--num_workers 4`
  - 8核CPU：`--num_workers 8`
  - 16核CPU：`--num_workers 16`
  - 32核CPU：`--num_workers 16-24`（避免过多进程导致I/O竞争）

## 断点续传（Resume）功能

如果转换过程被中断（例如系统重启、手动停止等），可以使用 `--resume` 参数继续处理：

```bash
# 继续之前的转换
bash run_convert.sh --resume
```

**工作原理：**
- 脚本会在输出目录的 `.progress` 子目录中记录已完成的子目录
- 使用 `--resume` 时，会跳过这些已完成的子目录
- 只处理未完成或中断的子目录
- 如果某个子目录处理失败，不会标记为完成，resume时会重新处理

**注意：**
- Resume功能按子目录级别工作（不是按单个样本）
- 如果某个子目录处理了一半就被中断，resume时会重新处理整个子目录（不会从中间恢复）
- 这是为了确保数据完整性，因为每个子目录可能产生多个shard文件

## 注意事项

- 脚本会使用 `innovator_vl_stable` conda环境运行
- 转换过程可能需要较长时间，取决于数据量大小和并发数
- 建议在后台运行，并定期检查日志
- 多进程处理会占用更多内存，请确保系统有足够内存空间
- 如果转换中断，可以使用 `--resume` 参数继续处理，避免重复工作

## 示例输出

```
2024-01-01 10:00:00 - INFO - 找到 150 个子目录
2024-01-01 10:00:01 - INFO - 处理目录: llava_instruct
2024-01-01 10:05:30 - INFO - 创建shard: llava_instruct_000010.tar
...
==================================================
转换完成！统计信息:
  总样本数: 1000000
  有效样本数: 985000
  无效样本数: 15000
  跳过原因:
    - prompt中包含<image>但缺少图像文件: 12000
    - 图像解析失败: 3000
  输出目录: /root/innovator_data_wenzichen/webdataset_output
  总shard数: 150
==================================================
```


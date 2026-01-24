#!/usr/bin/env python3
"""
将 Innovator-VL-Insturct-Data 转换为 WebDataset 格式

该脚本会：
1. 遍历所有数据子目录
2. 读取 parquet 文件
3. 校验样本（检查图像、prompt一致性等）
4. 转换为 webdataset 格式（支持多进程并发）

使用方法:
    cd /root/innovator_code_wenzichen/Innovator-VL/tools/data_conversion
    bash run_convert.sh --input_dir /path/to/data --output_dir /path/to/output
    
或者:
    conda run -n innovator_vl_stable python convert_to_webdataset.py --input_dir /path/to/data --output_dir /path/to/output --num_workers 8
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
from multiprocessing import Pool
import time
import hashlib
try:
    import yaml
except ImportError:
    yaml = None

try:
    import pyarrow.parquet as pq
    import pandas as pd
    import webdataset as wds
    from PIL import Image
    import io
    import numpy as np
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装: pip install pyarrow pandas webdataset pillow numpy")
    raise

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_non_empty_value(d: Dict[str, Any], *keys: str, default: str = '') -> str:
    """
    从字典中按优先级顺序获取非空值
    
    示例：get_non_empty_value(conv, 'role', 'from') 
    - 如果 'role' 存在且非空（非None、非空字符串），返回 'role' 的值
    - 否则如果 'from' 存在且非空，返回 'from' 的值
    - 否则返回 default
    
    特别处理：
    - 如果字段值为 None，视为空值，继续查找下一个key
    - 如果字段值为空字符串或只包含空白，视为空值，继续查找下一个key
    - 其他非空值（包括非字符串类型如数字），转换为字符串返回
    
    Args:
        d: 字典
        keys: 按优先级顺序的key列表
        default: 默认值
    
    Returns:
        第一个找到的非空值的字符串形式，或default
    """
    for key in keys:
        if key in d:
            value = d[key]
            # 检查值是否非空
            if value is None:
                continue  # None视为空，继续查找下一个key
            
            if isinstance(value, str):
                # 对于字符串，检查去除空白后是否为空
                if value.strip():
                    return value.strip()
            else:
                # 对于非字符串类型，直接使用（如数字、布尔值等）
                return str(value)
    
    return default


def get_sample_id(sample: Dict[str, Any], default_id: str = '', generate_stable_id: bool = True) -> str:
    """
    获取样本ID，兼容多种可能的key名称
    
    兼容的key（按优先级顺序）：
    - 'id'（最常用）
    - '__key__'
    - 'sample_id'
    - 'key'
    
    如果多个key都存在且值不同，优先使用'id'，并记录警告
    
    如果没有找到ID且generate_stable_id=True，会基于样本内容生成一个稳定的ID
    
    Args:
        sample: 样本字典
        default_id: 如果找不到ID时的默认值（如果不为空，直接返回）
        generate_stable_id: 如果为True且没有找到ID，基于内容生成稳定ID
    
    Returns:
        样本ID字符串
    """
    # 按优先级顺序尝试多种可能的key
    candidate_keys = ['id', '__key__', 'sample_id', 'key']
    found_ids = {}
    
    for key in candidate_keys:
        if key in sample and sample[key] is not None:
            value = str(sample[key]).strip()
            if value:  # 忽略空字符串
                found_ids[key] = value
    
    if not found_ids:
        # 如果没有找到ID
        if default_id:
            return default_id
        
        # 如果需要生成稳定ID，基于样本内容生成
        if generate_stable_id:
            # 基于关键字段生成稳定的ID（使用hash）
            # 使用conversations的第一条内容和图像信息的组合
            content_parts = []
            
            # 尝试获取conversations的第一条内容
            conversations = sample.get('conversations', [])
            if conversations:
                if isinstance(conversations, list) and len(conversations) > 0:
                    first_conv = conversations[0]
                    if isinstance(first_conv, dict):
                        content = get_non_empty_value(first_conv, 'content', 'value', default='')
                        if content:
                            # 只取前200个字符，避免太长
                            content_parts.append(content[:200])
            
            # 尝试获取图像信息（如果存在）
            image = sample.get('image', None)
            if image and isinstance(image, dict):
                image_bytes = image.get('bytes')
                if image_bytes and isinstance(image_bytes, bytes):
                    # 使用图像bytes的前100字节的hash（足够区分不同图像）
                    image_hash = hashlib.md5(image_bytes[:100]).hexdigest()[:16]
                    content_parts.append(f"img_{image_hash}")
            
            # 如果都没有，使用样本的所有key的hash
            if not content_parts:
                # 基于所有非空字段生成hash
                sample_str = json.dumps(sample, sort_keys=True, default=str)[:500]
                content_parts.append(sample_str)
            
            # 生成稳定的ID
            combined = "_".join(content_parts)
            stable_id = hashlib.md5(combined.encode('utf-8')).hexdigest()[:16]
            return f"auto_{stable_id}"
        
        return ""
    
    # 清理ID值，确保不包含路径分隔符和文件扩展名（webdataset会将其解析为文件路径或扩展名）
    def clean_id(id_value: str) -> str:
        """清理ID值，将路径分隔符替换为下划线，移除常见图像文件扩展名"""
        if not isinstance(id_value, str):
            id_value = str(id_value)
        # 将路径分隔符替换为下划线
        cleaned = id_value.replace('/', '_').replace('\\', '_')
        # 移除开头的路径组件（如果还有）
        while cleaned.startswith('_'):
            cleaned = cleaned[1:]
        # 移除常见的图像文件扩展名（避免webdataset将其解析为文件扩展名，导致生成png.json等错误的key）
        # 只移除扩展名，保留其他点号（如1911.10601_0中的点号应该保留）
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif']
        cleaned_lower = cleaned.lower()
        for ext in image_extensions:
            if cleaned_lower.endswith(ext):
                # 移除扩展名
                cleaned = cleaned[:-len(ext)]
                break
        return cleaned
    
    # 如果只有一个key，直接返回（但需要清理）
    if len(found_ids) == 1:
        return clean_id(list(found_ids.values())[0])
    
    # 如果多个key都存在，优先使用'id'
    if 'id' in found_ids:
        # 如果其他key的值与'id'不同，记录警告
        id_value = found_ids['id']
        for key, value in found_ids.items():
            if key != 'id' and value != id_value:
                logger.warning(f"样本存在多个ID字段且值不同: id={id_value}, {key}={value}，优先使用id")
        return clean_id(id_value)
    
    # 如果没有'id'，使用第一个找到的（按优先级）
    return clean_id(list(found_ids.values())[0])


def check_image_consistency(sample: Dict[str, Any]) -> Tuple[bool, str]:
    """
    检查样本的图像和prompt的一致性
    
    规则：
    1. 如果prompt中有<image>标记，必须有有效的图像数据
    2. 如果图像存在，尝试验证其有效性（但不强制要求prompt中有<image>标记）
    3. 纯文本对话（无图像无<image>标记）是允许的
    
    Args:
        sample: 样本字典，包含 'image' 和 'conversations' 字段
    
    Returns:
        (is_valid, error_message): 样本是否有效及错误信息
    """
    image = sample.get('image', None)
    conversations = sample.get('conversations', [])
    
    # 检查conversations格式
    if conversations is None:
        return False, "conversations为None"
    
    # 如果是numpy数组，转换为列表
    if isinstance(conversations, np.ndarray):
        conversations = conversations.tolist()
    
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False, "conversations为空或格式错误"
    
    # 检查所有对话中是否包含<image>标记
    has_image_tag = False
    for conv in conversations:
        # conversations格式兼容多种：'content'/'value' 表示内容
        if isinstance(conv, dict):
            # 兼容 'content' 或 'value'，优先使用非空值
            content = get_non_empty_value(conv, 'content', 'value', default='')
        else:
            content = str(conv)
        if isinstance(content, str) and '<image>' in content:
            has_image_tag = True
            break
    
    # 关键校验：如果有<image>标记，必须有有效的图像数据
    if has_image_tag:
        if image is None:
            return False, "prompt中包含<image>但缺少图像文件"
        # 检查image字典结构
        if isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes is None:
                return False, "prompt中包含<image>但图像bytes为空"
            
            # 验证图像有效性
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()
                # 重新打开图像（verify后需要重新打开）
                img = Image.open(io.BytesIO(image_bytes))
                if img.size[0] == 0 or img.size[1] == 0:
                    return False, "图像尺寸无效"
            except Exception as e:
                return False, f"图像解析失败: {str(e)}"
    
    # 如果有图像但prompt中没有<image>标记，这是允许的（可能是隐式图像）
    # 但仍需验证图像有效性
    if image is not None and isinstance(image, dict):
        image_bytes = image.get('bytes')
        if image_bytes is not None and not has_image_tag:
            sample_id = get_sample_id(sample, 'unknown')
            logger.debug(f"样本ID {sample_id}: 有图像但prompt中没有<image>标记")
            # 验证图像有效性（但不强制失败）
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.verify()
                img = Image.open(io.BytesIO(image_bytes))
                if img.size[0] == 0 or img.size[1] == 0:
                    return False, "图像尺寸无效"
            except Exception as e:
                sample_id = get_sample_id(sample, 'unknown')
                logger.warning(f"样本ID {sample_id}: 图像验证失败但继续处理: {str(e)}")
    
    # 纯文本对话（无图像无<image>标记）是完全允许的
    return True, ""


def process_image(image: Any) -> Optional[bytes]:
    """
    将图像转换为字节数据
    
    Args:
        image: 图像数据，可能是字典 {'bytes': b'...', 'path': '...'} 或字节数据或PIL Image
        
    Returns:
        图像的字节数据（JPEG格式），如果无效则返回None
    """
    if image is None:
        return None
    
    try:
        # 如果image是字典，提取bytes字段
        if isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes is None:
                return None
            # 如果是字节数据，验证并转换为JPEG
            if isinstance(image_bytes, bytes):
                img = Image.open(io.BytesIO(image_bytes))
                buffer = io.BytesIO()
                # 确保图像是RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(buffer, format='JPEG', quality=95)
                result = buffer.getvalue()
                # 及时释放图像对象，帮助GC
                img.close()
                del img, buffer
                return result
        
        # 如果已经是字节数据
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
            buffer = io.BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buffer, format='JPEG', quality=95)
            result = buffer.getvalue()
            # 及时释放图像对象，帮助GC
            img.close()
            del img, buffer
            return result
        
        # 如果是PIL Image对象
        if hasattr(image, 'save'):
            buffer = io.BytesIO()
            img_mode = image.mode
            if img_mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=95)
            result = buffer.getvalue()
            # 及时释放缓冲区
            del buffer
            return result
        
        logger.warning(f"未知的图像类型: {type(image)}")
        return None
    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        return None


def convert_conversations_format(conversations: Any) -> List[Dict[str, str]]:
    """
    转换conversations格式为标准的列表格式
    
    兼容多种字段名：
    - 角色：'role' 或 'from'
    - 内容：'content' 或 'value'
    
    Args:
        conversations: 可能是numpy数组或列表，包含对话字典
    
    Returns:
        标准化的conversations列表，统一为 {'role': '...', 'content': '...'} 格式
    """
    if conversations is None:
        return []
    
    # 如果是numpy数组，转换为列表
    if isinstance(conversations, np.ndarray):
        conversations = conversations.tolist()
    
    if not isinstance(conversations, list):
        return []
    
    # 标准化每个对话项
    normalized = []
    for conv in conversations:
        if isinstance(conv, dict):
            # 兼容角色字段：优先使用 'role'（如果非空），否则使用 'from'
            role = get_non_empty_value(conv, 'role', 'from', default='')
            # 兼容内容字段：优先使用 'content'（如果非空），否则使用 'value'
            content = get_non_empty_value(conv, 'content', 'value', default='')
            
            # 如果角色和内容都为空，跳过这个对话项（无效数据）
            if not role and not content:
                continue  # 跳过空的对话项
            
            # 标准化role值（映射到标准格式）
            role_mapping = {
                'human': 'user',
                'gpt': 'assistant',
                'assistant': 'assistant',  # 保持assistant不变
                'user': 'user',  # 保持user不变
                'system': 'system'  # 保持system不变
            }
            role = role_mapping.get(role.lower(), role)  # 转换为小写后映射，如果不存在则保持原值
            
            # 统一转换为标准格式
            normalized.append({
                'role': role,
                'content': content
            })
        else:
            logger.warning(f"未知的conversation格式: {type(conv)}")
    
    return normalized


def check_subdir_completed(subdir_name: str, output_path: Path) -> bool:
    """
    检查子目录是否已经处理完成
    
    Args:
        subdir_name: 子目录名称
        output_path: 输出目录路径
    
    Returns:
        bool: 如果已完成返回True，否则返回False
    """
    progress_file = output_path / ".progress" / f"{subdir_name}.done"
    if progress_file.exists():
        # 验证文件完整性：至少应该有1行内容（避免旧版本写入中断导致的不完整文件）
        try:
            content = progress_file.read_text()
            if len(content.strip()) > 0 and 'completed at' in content:
                return True
            else:
                # 文件存在但内容不完整，可能是写入中断，不认为是已完成
                logger.warning(f"检测到不完整的.done文件: {progress_file}，将重新处理")
                return False
        except Exception:
            # 如果读取失败，不认为是已完成（保守策略）
            return False
    return False


def process_single_subdir(args_tuple):
    """
    处理单个子目录的所有parquet文件（用于多进程）
    
    Args:
        args_tuple: (subdir_path, output_dir, max_samples_per_shard, skip_invalid, resume, batch_size)
    
    Returns:
        dict: 处理统计信息
    """
    subdir_path, output_dir, max_samples_per_shard, skip_invalid, resume, batch_size = args_tuple
    
    subdir = Path(subdir_path)
    output_path = Path(output_dir)
    
    stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'skipped_reasons': {},
        'subdir_name': subdir.name,
        'skipped': False
    }
    
    # 如果启用resume且该子目录已完成，则跳过
    if resume and check_subdir_completed(subdir.name, output_path):
        print(f"[进程 {os.getpid()}] 跳过已处理的子目录: {subdir.name}", flush=True)
        stats['skipped'] = True
        return stats
    
    # 递归查找所有parquet文件（支持子目录和子子目录）
    parquet_files = sorted(subdir.rglob('*.parquet'))
    
    if len(parquet_files) == 0:
        print(f"[进程 {os.getpid()}] 警告: 子目录 {subdir.name} 中没有找到parquet文件", flush=True)
        return stats
    
    print(f"[进程 {os.getpid()}] 子目录 {subdir.name}: 找到 {len(parquet_files)} 个parquet文件", flush=True)
    
    # 创建进度标记目录
    progress_dir = output_path / ".progress"
    progress_dir.mkdir(exist_ok=True)
    
    # 无论是否启用resume，都要检查已存在的shard并收集已处理的样本ID（避免重复处理）
    # 这是为了防止：1) resume=False时重复写入 2) 多次运行导致的重复
    processed_sample_ids = set()
    incomplete_shards = []  # 记录不完整的shard文件路径
    
    # 查找该子目录的所有已存在的shard文件
    existing_shards = sorted(output_path.glob(f"{subdir.name}_*.tar"))
    if len(existing_shards) > 0:
        mode_str = "resume模式" if resume else "正常模式"
        print(f"[进程 {os.getpid()}] ({mode_str}) 检测到 {len(existing_shards)} 个已存在的shard文件，快速检查完整性...", flush=True)
        
        # 性能优化：对于大文件（>1GB），使用快速检查模式
        # 快速模式只验证文件可读性，不完整读取所有样本（避免I/O瓶颈）
        for i, shard_path in enumerate(existing_shards):
            file_size_gb = shard_path.stat().st_size / (1024**3)
            
            # 对于大文件（>1GB），使用快速检查（只读前几个样本验证完整性）
            # 对于小文件，完整读取以收集所有sample_id进行精确去重
            use_fast_mode = file_size_gb > 1.0
            
            if use_fast_mode:
                # 快速模式：只验证文件完整性，不收集所有sample_id
                is_complete, sample_count, sample_ids = check_shard_completeness(shard_path, fast_mode=True)
                if not is_complete:
                    incomplete_shards.append(shard_path)
                    print(f"[进程 {os.getpid()}] 检测到不完整的shard文件: {shard_path.name}", flush=True)
            else:
                # 完整模式：读取所有样本以收集sample_id（用于精确去重）
                is_complete, sample_count, sample_ids = check_shard_completeness(shard_path, fast_mode=False)
                if is_complete:
                    processed_sample_ids.update(sample_ids)
                else:
                    processed_sample_ids.update(sample_ids)
                    incomplete_shards.append(shard_path)
                    print(f"[进程 {os.getpid()}] 检测到不完整的shard文件: {shard_path.name} (已读取 {len(sample_ids)} 个样本)", flush=True)
            
            # 显示进度（每5个文件显示一次，或每10个文件显示一次大文件）
            if (i + 1) % (10 if use_fast_mode else 5) == 0 or (i + 1) == len(existing_shards):
                print(f"[进程 {os.getpid()}] 已检查 {i+1}/{len(existing_shards)} 个shard文件...", flush=True)
        
        print(f"[进程 {os.getpid()}] 检查完成，已收集 {len(processed_sample_ids)} 个已处理的样本ID，发现 {len(incomplete_shards)} 个不完整的shard文件", flush=True)
        if len(existing_shards) > 0:
            large_files = sum(1 for s in existing_shards if s.stat().st_size / (1024**3) > 1.0)
            if large_files > 0:
                print(f"[进程 {os.getpid()}] 注意：{large_files} 个大文件使用快速检查模式，不预先加载所有样本ID（避免I/O瓶颈）", flush=True)
    
    # 使用子目录名称的哈希值来生成唯一的shard起始编号，避免进程间同步问题
    # 这样可以避免使用Manager().Value()的get_lock()问题
    # 使用子目录名称哈希的前8位，然后映射到0-999999范围
    subdir_hash = int(hashlib.md5(subdir.name.encode()).hexdigest()[:8], 16)
    # 每个子目录分配10000个编号空间，确保不同子目录的编号不会重叠
    # 使用哈希值取模来分配起始编号，范围是 [0, 999990]，步长为10000
    shard_base_idx = (subdir_hash % 100) * 10000
    current_shard_idx = 0  # 从0开始计数该子目录内的shard
    
    current_shard_samples = 0
    writer = None
    
    try:
        for parquet_file in parquet_files:
            try:
                # 使用pyarrow的迭代器分批读取，避免一次性加载整个文件到内存
                # 这样可以处理非常大的parquet文件，而不会耗尽内存
                parquet_file_obj = pq.ParquetFile(parquet_file)
                # batch_size在函数参数中传入，可根据内存情况调整（默认1000）
                
                for batch in parquet_file_obj.iter_batches(batch_size=batch_size):
                    # 将batch转换为pandas DataFrame（每次只处理一小批）
                    df_batch = batch.to_pandas()
                    
                    # 使用itertuples代替iterrows，效率更高且内存占用更少
                    for row_tuple in df_batch.itertuples(index=False):
                        stats['total_samples'] += 1
                        
                        # 将namedtuple转换为字典（更高效）
                        try:
                            # 尝试使用_asdict方法（Python 3.8+）
                            sample = row_tuple._asdict()
                        except AttributeError:
                            # 如果没有_asdict，手动构建字典
                            sample = {col: getattr(row_tuple, col) for col in df_batch.columns}
                        
                        # 准备样本数据（先获取ID，用于重复检查）
                        # 兼容多种可能的ID字段名
                        # 注意：不要使用total_samples作为default_id，因为它会导致同一无ID样本在不同位置得到不同ID
                        # 使用generate_stable_id=True，让没有ID的样本也能基于内容生成稳定的ID
                        sample_id = get_sample_id(sample, default_id='', generate_stable_id=True)
                        
                        # 确保sample_id不为空（如果仍然为空，使用fallback）
                        if not sample_id:
                            # 最终fallback：基于parquet文件路径的hash（确保同一文件总是相同）+ 样本内容的简单hash
                            # 这样可以确保同一parquet文件中的同一样本总是得到相同的ID
                            file_hash = hashlib.md5(str(parquet_file).encode()).hexdigest()[:8]
                            sample_hash = hashlib.md5(json.dumps(sample, sort_keys=True, default=str).encode()).hexdigest()[:8]
                            sample_id = f"fallback_{file_hash}_{sample_hash}"
                        
                        # 如果该样本已处理过（无论是否resume模式），跳过（避免重复，在图像处理前就跳过）
                        if sample_id in processed_sample_ids:
                            stats['valid_samples'] += 1  # 计入valid但跳过写入
                            continue
                        
                        # 校验样本
                        is_valid, error_msg = check_image_consistency(sample)
                        
                        if not is_valid:
                            stats['invalid_samples'] += 1
                            stats['skipped_reasons'][error_msg] = stats['skipped_reasons'].get(error_msg, 0) + 1
                            if skip_invalid:
                                continue
                        
                        # 如果shard已满或writer未创建，创建新的writer
                        if writer is None or current_shard_samples >= max_samples_per_shard:
                            if writer is not None:
                                writer.close()
                            
                            # 创建新的shard文件（使用子目录名前缀）
                            # 如果文件已存在：
                            # - 如果是resume模式且是不完整的shard，删除它并重新创建
                            # - 如果是完整的shard，使用下一个编号（避免覆盖）
                            while True:
                                shard_filename = f"{subdir.name}_{shard_base_idx + current_shard_idx:06d}.tar"
                                shard_path = output_path / shard_filename
                                if not shard_path.exists():
                                    break
                                
                                # 如果这个shard是不完整的，删除它并重新创建（无论是否resume模式）
                                if shard_path in incomplete_shards:
                                    print(f"[进程 {os.getpid()}] 删除不完整的shard文件并重新创建: {shard_filename}", flush=True)
                                    try:
                                        shard_path.unlink()
                                        break  # 删除成功，可以创建新文件
                                    except Exception as e:
                                        print(f"[进程 {os.getpid()}] 删除shard文件失败: {shard_filename}, 错误: {e}，跳过此编号", flush=True)
                                        current_shard_idx += 1
                                        continue
                                else:
                                    # 文件存在且是完整的，或不在不完整列表中，跳过此编号
                                    print(f"[进程 {os.getpid()}] shard文件已存在，跳过: {shard_filename}", flush=True)
                                    current_shard_idx += 1
                            
                            writer = wds.TarWriter(str(shard_path))
                            current_shard_samples = 0
                            current_shard_idx += 1
                        
                        # 处理图像
                        image_bytes = process_image(sample.get('image'))
                        
                        # 处理conversations（转换为texts格式）
                        conversations = convert_conversations_format(sample.get('conversations'))
                        
                        # 检查conversations是否为空（所有对话项都无效）
                        if not conversations or len(conversations) == 0:
                            stats['invalid_samples'] += 1
                            stats['skipped_reasons']['conversations为空（角色或内容字段都为空）'] = stats['skipped_reasons'].get('conversations为空（角色或内容字段都为空）', 0) + 1
                            if skip_invalid:
                                continue
                        
                        # 确定media类型
                        media = "image" if image_bytes is not None else "text"
                        
                        # 生成图像文件名（如果图像存在）
                        # 格式应该类似 "0_000000406340.jpg"（与目标格式保持一致）
                        # 如果原始数据中有文件名信息，优先使用；否则基于sample_id生成
                        image_filename = None
                        if image_bytes is not None:
                            # 优先检查原始数据中是否有图像文件名信息
                            image_name = None
                            if isinstance(sample.get('image'), dict):
                                image_name = sample['image'].get('path') or sample['image'].get('name')
                            
                            # 如果原始数据中没有，尝试从其他字段获取
                            if not image_name:
                                for key in ['image_name', 'image_path', 'filename', 'file_name']:
                                    if key in sample and sample[key]:
                                        image_name = sample[key]
                                        break
                            
                            # 如果找到了文件名，使用它；否则基于sample_id和shard索引生成
                            if image_name and isinstance(image_name, str):
                                # 提取文件名（去除路径）
                                image_filename = os.path.basename(image_name)
                                # 清理文件名：确保不包含路径分隔符（webdataset可能会将其解析为路径）
                                image_filename = image_filename.replace('/', '_').replace('\\', '_')
                                # 移除开头的下划线（如果路径以/开头）
                                while image_filename.startswith('_'):
                                    image_filename = image_filename[1:]
                                # 统一转换为小写，确保JSON中的name和实际tar中的key一致（避免大小写不匹配）
                                # 注意：这里统一使用小写，因为文件系统的key通常是大小写不敏感的，但Python的dict key是大小写敏感的
                                image_filename = image_filename.lower()
                                # 确保有扩展名
                                if not '.' in image_filename:
                                    image_filename += '.jpg'
                            else:
                                # 生成类似 "0_000000406340.jpg" 的格式
                                # 使用shard编号前缀 + sample_id，确保唯一性
                                shard_prefix = (shard_base_idx + current_shard_idx) % 1000  # 使用shard编号前缀
                                # 将sample_id转换为数字ID或使用其哈希值
                                try:
                                    # 尝试将sample_id转换为整数
                                    sample_id_num = int(sample_id.split('_')[-1]) if '_' in sample_id else int(sample_id)
                                except:
                                    # 如果无法转换，使用哈希值
                                    sample_id_num = abs(hash(sample_id)) % 1000000000
                                image_filename = f"{shard_prefix}_{sample_id_num:012d}.jpg"
                        
                        # 准备样本数据（匹配LLaVA-NeXT格式）
                        sample_data = {
                            'texts': conversations,  # 使用texts而不是conversations
                            'media': media,
                            'name': [image_filename] if image_filename else []  # 图像文件名列表
                        }
                        
                        # 准备写入的样本（图像以文件名作为key）
                        sample_dict = {
                            "__key__": sample_id,
                            "json": json.dumps(sample_data, ensure_ascii=False).encode('utf-8'),
                        }
                        
                        # 如果有图像，以文件名为key保存
                        if image_bytes is not None and image_filename:
                            sample_dict[image_filename] = image_bytes
                        
                        # 写入样本
                        writer.write(sample_dict)
                        
                        # 记录已处理的样本ID（用于后续去重，无论是否resume模式都需要记录）
                        processed_sample_ids.add(sample_id)
                        
                        stats['valid_samples'] += 1
                        current_shard_samples += 1
                        
                        # 释放sample_dict和sample_data的引用，帮助GC
                        del sample_dict, sample_data
                    
                    # 处理完一个batch后，清理df_batch释放内存
                    del df_batch
                    
            except Exception as e:
                # 在多进程环境中，直接打印可能比logger更安全
                print(f"[进程 {os.getpid()}] 处理文件 {parquet_file.name} 时出错: {str(e)}", flush=True)
                continue
        
        # 关闭最后一个writer
        if writer is not None:
            writer.close()
            writer = None  # 确保writer已关闭
        
        # 标记该子目录已完成（使用原子性写入，避免中断导致的不一致）
        progress_file = progress_dir / f"{subdir.name}.done"
        progress_file_tmp = progress_dir / f"{subdir.name}.done.tmp"
        try:
            # 先写入临时文件
            progress_file_tmp.write_text(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                         f"valid_samples: {stats['valid_samples']}\n"
                                         f"total_samples: {stats['total_samples']}")
            # 原子性重命名（POSIX系统保证原子性）
            progress_file_tmp.replace(progress_file)
        except Exception as e:
            # 如果写入失败，清理临时文件
            if progress_file_tmp.exists():
                progress_file_tmp.unlink()
            raise
            
    except Exception as e:
        print(f"[进程 {os.getpid()}] 处理目录 {subdir.name} 时出错: {str(e)}", flush=True)
        if writer is not None:
            try:
                writer.close()
            except:
                pass  # 关闭writer时出错，忽略
        # 如果出错，不标记为完成，以便resume时重新处理
        # 返回当前统计信息（可能部分处理）
        return stats
    
    print(f"[进程 {os.getpid()}] 完成处理 {subdir.name}: {stats['valid_samples']} 有效样本", flush=True)
    
    return stats


def check_shard_completeness(shard_path: Path, fast_mode: bool = False) -> Tuple[bool, int, List[str]]:
    """
    检查shard文件的完整性和可读性，并返回已处理的样本ID列表
    
    性能优化：
    - fast_mode=True: 只读取前几个样本验证完整性，避免完整读取大文件（几GB级别）
    - fast_mode=False: 完整读取所有样本（用于精确去重，但大文件会很慢）
    
    Args:
        shard_path: shard文件路径
        fast_mode: 是否使用快速模式（只检查文件完整性，不完全读取）
    
    Returns:
        (is_complete, sample_count, sample_ids): 
        - is_complete: shard是否完整可读
        - sample_count: 成功读取的样本数量（fast_mode下是估算值）
        - sample_ids: 成功读取的样本ID列表
    """
    import webdataset as wds
    import tarfile
    
    sample_ids = []
    count = 0
    
    # 快速模式：只读取前几个样本来验证文件完整性
    if fast_mode:
        try:
            # 先快速检查：尝试打开tar文件，验证格式
            with tarfile.open(shard_path, 'r') as tar:
                members = tar.getmembers()
                if len(members) == 0:
                    return False, 0, []
            
            # 尝试读取前10个样本验证可读性
            dataset = wds.WebDataset(str(shard_path), shardshuffle=False)
            sample_iter = iter(dataset)
            
            # 读取前10个样本验证文件开头正常
            for i, sample in enumerate(sample_iter):
                if i >= 10:
                    break
                sample_id = sample.get('__key__', '')
                if sample_id:
                    sample_ids.append(sample_id)
                count += 1
            
            # 如果前10个样本都能正常读取，认为文件基本完整
            # 对于去重，不收集所有ID（节省时间和内存）
            return True, 0, sample_ids  # sample_count=0表示使用了快速模式
            
        except Exception as e:
            # 如果读取失败，文件可能不完整
            return False, count, sample_ids
    
    # 完整模式：读取所有样本（用于精确去重，但会很慢）
    try:
        # 尝试读取shard中的所有样本（禁用shuffle以确保顺序读取）
        for sample in wds.WebDataset(str(shard_path), shardshuffle=False):
            sample_id = sample.get('__key__', '')
            if sample_id:
                sample_ids.append(sample_id)
            count += 1
        # 如果能够完整读取（没有异常），认为shard是完整的
        return True, count, sample_ids
    except Exception as e:
        # 如果读取失败，shard可能不完整，但返回已成功读取的部分
        return False, len(sample_ids), sample_ids


def count_samples_in_shard(shard_path: Path, default_count: int = 10000) -> int:
    """
    统计shard文件中的样本数量
    
    Args:
        shard_path: shard文件路径
        default_count: 如果统计失败返回的默认值
    
    Returns:
        样本数量
    """
    try:
        import webdataset as wds
        count = 0
        for _ in wds.WebDataset(str(shard_path), shardshuffle=False):
            count += 1
        return count
    except Exception:
        # 如果无法读取，返回默认值
        return default_count


def generate_nv_meta(output_path: Path, results: List[Dict], max_samples_per_shard: int):
    """
    生成.nv-meta目录和必要的元数据文件
    
    该函数会从输出目录中扫描所有已生成的tar文件，自动生成.nv-meta配置
    
    Args:
        output_path: 输出目录路径
        results: 处理结果列表（可选，主要用于日志）
        max_samples_per_shard: 每个shard的最大样本数
    """
    nv_meta_dir = output_path / ".nv-meta"
    nv_meta_dir.mkdir(exist_ok=True)
    
    # 从实际生成的tar文件中收集所有shard信息（不依赖results）
    all_shard_files = sorted([f.name for f in output_path.glob("*.tar")])
    
    if len(all_shard_files) == 0:
        logger.warning("输出目录中没有找到任何tar文件，无法生成.nv-meta")
        return
    
    logger.info(f"找到 {len(all_shard_files)} 个shard文件，开始生成.nv-meta...")
    
    shard_counts = {}
    
    logger.info(f"统计shard文件样本数（共{len(all_shard_files)}个shard）...")
    # 只统计前100个shard的样本数，避免太慢
    sample_count_limit = min(100, len(all_shard_files))
    for shard_file in tqdm(all_shard_files[:sample_count_limit], desc="统计shard样本数", leave=False):
        shard_path = output_path / shard_file
        try:
            count = count_samples_in_shard(shard_path, max_samples_per_shard)
            shard_counts[shard_file] = count
        except Exception as e:
            # 如果统计失败，使用默认值
            logger.debug(f"统计 {shard_file} 失败: {e}")
            shard_counts[shard_file] = max_samples_per_shard
    
    # 对于未统计的shard，使用默认值
    for shard_file in all_shard_files[sample_count_limit:]:
        shard_counts[shard_file] = max_samples_per_shard
    
    # 生成sample_loader.py（完全匹配LLaVA-NeXT格式）
    sample_loader_content = """def sample_loader(sample: dict) -> dict:
    # 解析JSON（兼容多种可能的json key格式，如json、png.json等）
    import json
    # 优先使用'json'，如果没有则尝试其他包含'json'的key（如'png.json'）
    json_key = 'json' if 'json' in sample else next((k for k in sample.keys() if 'json' in k.lower() and not k.startswith('__')), None)
    if json_key is None:
        raise KeyError("找不到json字段，样本keys: " + str(list(sample.keys())))
    json_data = json.loads(sample[json_key]) if isinstance(sample[json_key], bytes) else sample[json_key]
    
    messages = []
    system = None
    for message in json_data['texts']:
        assert message['role'] in ['system', 'user', 'assistant']
        if message['role'] == 'system':
            system = message['content']
            continue
        messages.append(dict(
            role=message['role'],
            content=message['content']
        ))
    video = []
    image = []
    if json_data['media'] == 'video':
        for name in json_data['name']:
            # 兼容多种可能的图像key格式（如name可能不匹配实际key）
            img_data = sample.get(name)
            if img_data is None:
                # 尝试查找包含name的其他key（大小写不敏感，可能带有前缀或后缀）
                name_lower = name.lower()
                matching_key = next((k for k in sample.keys() if not k.startswith('__') and (name_lower in k.lower() or k.lower() in name_lower or name_lower.split('.')[0] in k.lower())), None)
                if matching_key:
                    img_data = sample.get(matching_key)
            video.append(img_data)
    elif json_data['media'] == 'image':
        for name in json_data['name']:
            # 兼容多种可能的图像key格式（如name可能不匹配实际key）
            img_data = sample.get(name)
            if img_data is None:
                # 尝试查找包含name的其他key（大小写不敏感，可能带有前缀或后缀）
                name_lower = name.lower()
                matching_key = next((k for k in sample.keys() if not k.startswith('__') and (name_lower in k.lower() or k.lower() in name_lower or name_lower.split('.')[0] in k.lower())), None)
                if matching_key:
                    img_data = sample.get(matching_key)
            image.append(img_data)
    return dict(
        __key__=sample['__key__'],
        __restore_key__=sample.get('__restore_key__', sample['__key__']),
        video=video if len(video) > 0 else None,
        image=image if len(image) > 0 else None,
        system=system,
        messages=messages,
    )

def part_filter(part: str) -> bool:
    return True
"""
    
    sample_loader_path = nv_meta_dir / "sample_loader.py"
    sample_loader_path.write_text(sample_loader_content)
    logger.info(f"已生成: {sample_loader_path}")
    
    # 生成dataset.yaml
    dataset_yaml_content = {
        'sample_type': {
            '__module__': 'aiak_training_llm.data.multimodal',
            '__class__': 'MultiMixQASample'
        },
        'part_filter': 'sample_loader.py:part_filter',
        'sample_loader': 'sample_loader.py:sample_loader'
    }
    
    dataset_yaml_path = nv_meta_dir / "dataset.yaml"
    if yaml:
        try:
            with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_yaml_content, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"已生成: {dataset_yaml_path}")
        except Exception as e:
            logger.warning(f"使用yaml库写入失败，使用简单格式: {e}")
            yaml_content = """sample_type:
  __module__: aiak_training_llm.data.multimodal
  __class__: MultiMixQASample
part_filter: sample_loader.py:part_filter
sample_loader: sample_loader.py:sample_loader
"""
            dataset_yaml_path.write_text(yaml_content)
            logger.info(f"已生成: {dataset_yaml_path} (使用简单格式)")
    else:
        # 如果没有yaml库，使用简单格式
        yaml_content = """sample_type:
  __module__: aiak_training_llm.data.multimodal
  __class__: MultiMixQASample
part_filter: sample_loader.py:part_filter
sample_loader: sample_loader.py:sample_loader
"""
        dataset_yaml_path.write_text(yaml_content)
        logger.info(f"已生成: {dataset_yaml_path} (使用简单格式)")
    
    # 生成split.yaml（将所有shard分配到train）
    split_yaml_content = {
        'exclude': [],
        'split_parts': {
            'test': [],
            'train': sorted(all_shard_files),
            'val': []
        }
    }
    
    split_yaml_path = nv_meta_dir / "split.yaml"
    if yaml:
        try:
            with open(split_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(split_yaml_content, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"已生成: {split_yaml_path}")
        except Exception as e:
            logger.warning(f"使用yaml库写入失败，使用简单格式: {e}")
            split_content = "exclude: []\nsplit_parts:\n  test: []\n  train:\n"
            for shard in sorted(all_shard_files):
                split_content += f"  - {shard}\n"
            split_content += "  val: []\n"
            split_yaml_path.write_text(split_content)
            logger.info(f"已生成: {split_yaml_path} (使用简单格式)")
    else:
        # 如果没有yaml库，使用简单格式
        split_content = "exclude: []\nsplit_parts:\n  test: []\n  train:\n"
        for shard in sorted(all_shard_files):
            split_content += f"  - {shard}\n"
        split_content += "  val: []\n"
        split_yaml_path.write_text(split_content)
        logger.info(f"已生成: {split_yaml_path} (使用简单格式)")
    
    # 生成.info.yaml（shard统计信息）
    info_yaml_content = {
        'shard_counts': shard_counts
    }
    
    info_yaml_path = nv_meta_dir / ".info.yaml"
    if yaml:
        try:
            with open(info_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(info_yaml_content, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"已生成: {info_yaml_path}")
        except Exception as e:
            logger.warning(f"使用yaml库写入失败，使用简单格式: {e}")
            info_content = "shard_counts:\n"
            for shard, count in sorted(shard_counts.items()):
                info_content += f"  {shard}: {count}\n"
            info_yaml_path.write_text(info_content)
            logger.info(f"已生成: {info_yaml_path} (使用简单格式)")
    else:
        # 如果没有yaml库，使用简单格式
        info_content = "shard_counts:\n"
        for shard, count in sorted(shard_counts.items()):
            info_content += f"  {shard}: {count}\n"
        info_yaml_path.write_text(info_content)
        logger.info(f"已生成: {info_yaml_path} (使用简单格式)")
    
    logger.info(f"已生成.nv-meta目录和所有必需文件")


def generate_tar_index(tar_path: Path, idx_path: Path = None) -> Tuple[bool, Optional[str]]:
    """
    为tar文件生成索引文件(.idx)
    
    WebDataset标准索引格式：二进制格式，每个条目16字节
    - 8字节offset（小端序，unsigned long long）
    - 8字节size（小端序，unsigned long long）
    
    不包含文件名，文件名顺序与tar文件中的顺序一致
    
    参考：WebDataset标准格式
    https://github.com/webdataset/webdataset
    
    Args:
        tar_path: tar文件路径
        idx_path: 索引文件路径（如果不提供，使用tar_path + '.idx'）
    
    Returns:
        (success, error_message): 是否成功生成索引及错误信息
    """
    if idx_path is None:
        idx_path = tar_path.with_suffix(tar_path.suffix + '.idx')
    
    try:
        import tarfile
        import struct
        
        # 先检查文件是否存在且可读
        if not tar_path.exists():
            return False, f"文件不存在"
        
        # 检查文件大小，如果为0则可能是损坏的
        if tar_path.stat().st_size == 0:
            return False, f"文件大小为0（可能损坏）"
        
        index_entries = []
        try:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar:
                    # tarfile的member已经有offset属性（字节偏移）
                    if hasattr(member, 'offset') and member.offset is not None:
                        # 记录每个文件的位置信息
                        index_entries.append({
                            'offset': member.offset,
                            'size': member.size
                        })
                    else:
                        # 如果没有offset属性，旧版本tarfile需要计算
                        # 但现代版本都应该有，如果真没有则报错
                        return False, f"member {member.name} 没有offset属性，无法生成索引"
        except tarfile.ReadError as e:
            return False, f"tar文件读取错误（可能损坏）: {str(e)}"
        except EOFError as e:
            return False, f"unexpected end of data（文件不完整或损坏）"
        except Exception as e:
            # 捕获其他可能的异常
            error_str = str(e).lower()
            if 'unexpected end' in error_str or 'end of data' in error_str:
                return False, f"unexpected end of data（文件不完整或损坏）"
            return False, f"读取tar文件时出错: {str(e)}"
        
        # 如果没有找到任何条目，可能是空文件或损坏
        if len(index_entries) == 0:
            return False, f"tar文件中没有找到任何条目（可能为空或损坏）"
        
        # 写入索引文件（WebDataset标准格式）
        # 格式：每个条目16字节（8字节offset + 8字节size，小端序）
        try:
            with open(idx_path, 'wb') as f:
                for entry in index_entries:
                    # 写入offset（8字节，unsigned long long，小端序）
                    f.write(struct.pack('<Q', entry['offset']))
                    # 写入size（8字节，unsigned long long，小端序）
                    f.write(struct.pack('<Q', entry['size']))
        except Exception as e:
            return False, f"写入索引文件时出错: {str(e)}"
        
        return True, None
    except Exception as e:
        error_str = str(e).lower()
        if 'unexpected end' in error_str or 'end of data' in error_str:
            return False, f"unexpected end of data（文件不完整或损坏）"
        return False, f"生成索引时发生未知错误: {str(e)}"


def _generate_index_task(tar_path: Path) -> Tuple[str, bool, Optional[str]]:
    """
    为单个tar文件生成索引的任务函数（用于多进程）
    
    Args:
        tar_path: tar文件路径
    
    Returns:
        (result_message, success, error_message): 处理结果信息
    """
    idx_path = tar_path.with_suffix(tar_path.suffix + '.idx')
    # 如果索引文件已存在且较新，跳过
    if idx_path.exists() and idx_path.stat().st_mtime >= tar_path.stat().st_mtime:
        return f"{tar_path.name}: 索引已存在，跳过", True, None
    
    success, error_msg = generate_tar_index(tar_path, idx_path)
    if success:
        return f"{tar_path.name}: 成功生成索引", True, None
    else:
        error_str = error_msg or "未知错误"
        return f"{tar_path.name}: 生成索引失败 - {error_str}", False, error_str


def generate_indices_for_all_shards(output_path: Path, num_workers: int = 4):
    """
    为所有tar文件生成索引文件
    
    Args:
        output_path: 输出目录路径
        num_workers: 并发进程数
    
    Returns:
        failed_files: 失败的文件列表，每个元素为 (tar_path, error_message)
    """
    tar_files = sorted(list(output_path.glob("*.tar")))
    
    if len(tar_files) == 0:
        logger.warning("没有找到tar文件，无法生成索引")
        return []
    
    logger.info(f"开始为 {len(tar_files)} 个tar文件生成索引...")
    
    # 统计已存在的索引文件
    existing_indices = sum(1 for tf in tar_files if (tf.with_suffix(tf.suffix + '.idx')).exists())
    logger.info(f"已有 {existing_indices} 个索引文件")
    
    if num_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_generate_index_task, tar_files),
                total=len(tar_files),
                desc="生成索引文件"
            ))
    else:
        results = [_generate_index_task(tf) for tf in tqdm(tar_files, desc="生成索引文件")]
    
    # 统计结果和收集失败文件
    success_count = 0
    failed_files = []
    skipped_count = 0
    
    for result, tar_file in zip(results, tar_files):
        if isinstance(result, tuple) and len(result) == 3:
            result_msg, success, error_msg = result
        else:
            # 兼容旧格式（如果返回的是字符串）
            result_msg = result
            success = '成功' in result_msg or '已存在' in result_msg
            error_msg = None if success else "未知错误"
        
        if '已存在' in result_msg:
            skipped_count += 1
            success_count += 1
        elif success:
            success_count += 1
        else:
            failed_files.append((tar_file, error_msg))
            logger.error(f"ERROR - 生成索引文件失败 {tar_file.name}: {error_msg}")
    
    logger.info(f"索引生成完成: {success_count}/{len(tar_files)} 个文件")
    
    if len(failed_files) > 0:
        logger.warning(f"\n失败的索引生成文件（共 {len(failed_files)} 个）:")
        for tar_file, error_msg in failed_files:
            logger.warning(f"  - {tar_file.name}: {error_msg}")
        logger.warning("\n这些tar文件可能已损坏或不完整，建议检查文件完整性或重新生成这些shard文件")
    
    return failed_files


def convert_parquet_to_webdataset(
    input_dir: str,
    output_dir: str,
    max_samples_per_shard: int = 10000,
    skip_invalid: bool = True,
    num_workers: int = 4,
    resume: bool = False,
    batch_size: int = 1000,
    generate_index: bool = False
):
    """
    将parquet文件转换为webdataset格式（支持多进程并发和断点续传）
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出webdataset目录
        max_samples_per_shard: 每个shard的最大样本数
        skip_invalid: 是否跳过无效样本
        num_workers: 并发进程数（默认: 4）
        resume: 是否启用断点续传（默认: False）。如果启用，会跳过已完成的子目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有子目录
    subdirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    # 如果启用resume，检查已完成的子目录
    skipped_count = 0
    if resume:
        completed_subdirs = set()
        progress_dir = output_path / ".progress"
        if progress_dir.exists():
            for done_file in progress_dir.glob("*.done"):
                completed_subdirs.add(done_file.stem)
        
        skipped_count = len(completed_subdirs)
        if skipped_count > 0:
            logger.info(f"断点续传模式：跳过 {skipped_count} 个已完成的子目录")
            # 过滤掉已完成的子目录
            subdirs = [d for d in subdirs if d.name not in completed_subdirs]
    
    logger.info(f"找到 {len(subdirs)} 个子目录需要处理（已跳过 {skipped_count} 个）")
    logger.info(f"使用 {num_workers} 个进程并发处理")
    
    if len(subdirs) == 0:
        logger.info("所有子目录已完成处理，无需继续")
        # 即使没有新处理的子目录，如果之前没有生成.nv-meta，也应该生成
        nv_meta_dir = output_path / ".nv-meta"
        if not nv_meta_dir.exists() or not (nv_meta_dir / "dataset.yaml").exists():
            logger.info("检测到缺少.nv-meta文件，开始生成...")
            generate_nv_meta(output_path, [], max_samples_per_shard)
        else:
            logger.info(".nv-meta文件已存在，无需重新生成")
        return
    
    # 准备参数列表（不再需要全局计数器，每个子目录根据名称生成唯一shard编号）
    args_list = [
        (str(subdir), str(output_path), max_samples_per_shard, skip_invalid, resume, batch_size)
        for subdir in subdirs
    ]
    
    start_time = time.time()
    
    # 使用进程池并发处理
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            # 使用imap以便显示进度
            results = list(tqdm(
                pool.imap(process_single_subdir, args_list),
                total=len(args_list),
                desc="处理子目录"
            ))
    else:
        # 单进程模式（用于调试）
        results = [process_single_subdir(args) for args in tqdm(args_list, desc="处理子目录")]
    
    elapsed_time = time.time() - start_time
    
    # 合并统计信息
    merged_stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'skipped_reasons': {},
        'subdir_stats': {}
    }
    
    total_shards = 0
    skipped_subdirs_count = 0
    for result in results:
        # 如果是跳过的子目录（resume模式），不计入统计
        if result.get('skipped', False):
            skipped_subdirs_count += 1
            continue
            
        merged_stats['total_samples'] += result['total_samples']
        merged_stats['valid_samples'] += result['valid_samples']
        merged_stats['invalid_samples'] += result['invalid_samples']
        
        # 合并跳过原因
        for reason, count in result['skipped_reasons'].items():
            merged_stats['skipped_reasons'][reason] = merged_stats['skipped_reasons'].get(reason, 0) + count
        
        # 保存每个子目录的统计
        merged_stats['subdir_stats'][result['subdir_name']] = {
            'total': result['total_samples'],
            'valid': result['valid_samples'],
            'invalid': result['invalid_samples']
        }
        
        # 估算shard数（基于valid_samples）
        estimated_shards = (result['valid_samples'] + max_samples_per_shard - 1) // max_samples_per_shard
        total_shards += estimated_shards
    
    # 生成.nv-meta目录和文件
    generate_nv_meta(output_path, results, max_samples_per_shard)
    
    # 生成索引文件（如果启用）
    if generate_index:
        logger.info("开始生成tar文件索引...")
        generate_indices_for_all_shards(output_path, num_workers)
    
    # 打印统计信息
    logger.info("\n" + "="*60)
    logger.info("转换完成！统计信息:")
    if skipped_subdirs_count > 0:
        logger.info(f"  跳过的子目录数（已处理）: {skipped_subdirs_count}")
    logger.info(f"  总样本数: {merged_stats['total_samples']:,}")
    logger.info(f"  有效样本数: {merged_stats['valid_samples']:,}")
    logger.info(f"  无效样本数: {merged_stats['invalid_samples']:,}")
    logger.info(f"  跳过原因:")
    for reason, count in sorted(merged_stats['skipped_reasons'].items(), key=lambda x: -x[1]):
        logger.info(f"    - {reason}: {count:,}")
    logger.info(f"  输出目录: {output_path}")
    logger.info(f"  估算总shard数: {total_shards}")
    logger.info(f"  处理时间: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    logger.info(f"  处理速度: {merged_stats['valid_samples']/elapsed_time:.0f} 样本/秒")
    logger.info("="*60)
    
    # 可选：显示前10个处理最多的子目录
    if len(merged_stats['subdir_stats']) > 0:
        top_subdirs = sorted(
            merged_stats['subdir_stats'].items(),
            key=lambda x: x[1]['valid'],
            reverse=True
        )[:10]
        logger.info("\n处理样本最多的前10个子目录:")
        for subdir_name, subdir_stat in top_subdirs:
            logger.info(f"  {subdir_name}: {subdir_stat['valid']:,} 有效样本")


def main():
    parser = argparse.ArgumentParser(description='将LLaVA训练数据转换为WebDataset格式')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data',
        help='输入数据目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data-webdataset',
        help='输出WebDataset目录'
    )
    parser.add_argument(
        '--max_samples_per_shard',
        type=int,
        default=10000,
        help='每个shard的最大样本数（默认: 10000）'
    )
    parser.add_argument(
        '--skip_invalid',
        action='store_true',
        default=True,
        help='跳过无效样本（默认: True）'
    )
    parser.add_argument(
        '--no-skip-invalid',
        dest='skip_invalid',
        action='store_false',
        help='不跳过无效样本'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别（默认: INFO）'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='并发进程数（默认: 4，设为1则为单进程模式）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='启用断点续传模式：跳过已完成的子目录，从中断处继续（默认: False）'
    )
    parser.add_argument(
        '--only-generate-nv-meta',
        action='store_true',
        default=False,
        help='仅生成.nv-meta文件，不进行转换（适用于已完成的转换）'
    )
    parser.add_argument(
        '--generate-index',
        action='store_true',
        default=False,
        help='转换完成后生成索引文件(.idx)，用于加速随机访问（默认: False）'
    )
    parser.add_argument(
        '--only-generate-index',
        action='store_true',
        default=False,
        help='仅生成索引文件，不进行转换（适用于已完成的转换）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Parquet文件分批读取的批次大小（默认: 1000）。如果内存不足，可减小此值（如500或200）'
    )
    
    args = parser.parse_args()
    
    # 如果只生成.nv-meta，直接调用生成函数后退出
    if args.only_generate_nv_meta:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            logger.error(f"输出目录不存在: {output_path}")
            return
        logger.info("仅生成.nv-meta文件模式")
        generate_nv_meta(output_path, [], args.max_samples_per_shard)
        logger.info("完成！")
        return
    
    # 如果只生成索引文件，直接调用生成函数后退出
    if args.only_generate_index:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            logger.error(f"输出目录不存在: {output_path}")
            return
        logger.info("仅生成索引文件模式")
        generate_indices_for_all_shards(output_path, args.num_workers)
        logger.info("完成！")
        return
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 执行转换
    convert_parquet_to_webdataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_samples_per_shard=args.max_samples_per_shard,
        skip_invalid=args.skip_invalid,
        num_workers=args.num_workers,
        resume=args.resume,
        batch_size=args.batch_size,
        generate_index=args.generate_index
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
验证转换后的WebDataset格式是否正确

使用方法:
    conda run -n innovator_vl_stable python verify_webdataset_format.py --dataset_dir /path/to/webdataset
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
try:
    import webdataset as wds
except ImportError:
    print("请安装webdataset: pip install webdataset")
    sys.exit(1)


def verify_sample_format(sample: Dict[str, Any], sample_idx: int = 0) -> tuple[bool, List[str]]:
    """
    验证单个样本的格式
    
    Returns:
        (is_valid, errors): 是否有效及错误列表
    """
    errors = []
    
    # 1. 检查必需的key
    required_keys = ['__key__', 'json']
    for key in required_keys:
        if key not in sample:
            errors.append(f"缺少必需的key: {key}")
    
    # 2. 检查JSON格式
    if 'json' in sample:
        try:
            json_data = json.loads(sample['json']) if isinstance(sample['json'], bytes) else sample['json']
        except Exception as e:
            errors.append(f"JSON解析失败: {e}")
            return False, errors
        
        # 检查JSON结构
        required_json_keys = ['texts', 'media', 'name']
        for key in required_json_keys:
            if key not in json_data:
                errors.append(f"JSON中缺少必需的key: {key}")
        
        # 验证texts格式
        if 'texts' in json_data:
            if not isinstance(json_data['texts'], list):
                errors.append(f"texts必须是列表，当前类型: {type(json_data['texts'])}")
            else:
                for i, msg in enumerate(json_data['texts']):
                    if not isinstance(msg, dict):
                        errors.append(f"texts[{i}]必须是字典")
                    else:
                        if 'role' not in msg:
                            errors.append(f"texts[{i}]缺少'role'字段")
                        if 'content' not in msg:
                            errors.append(f"texts[{i}]缺少'content'字段")
                        if 'role' in msg and msg['role'] not in ['system', 'user', 'assistant']:
                            errors.append(f"texts[{i}]的role值无效: {msg['role']}")
        
        # 验证media格式
        if 'media' in json_data:
            if json_data['media'] not in ['image', 'text', 'video']:
                errors.append(f"media值无效: {json_data['media']}")
        
        # 验证name格式
        if 'name' in json_data:
            if not isinstance(json_data['name'], list):
                errors.append(f"name必须是列表，当前类型: {type(json_data['name'])}")
            else:
                # 如果media是image，name应该不为空
                if json_data.get('media') == 'image' and len(json_data['name']) == 0:
                    errors.append("media为image时，name列表不应为空")
                
                # 检查name中的文件名是否存在于sample中
                for name in json_data['name']:
                    if name not in sample:
                        errors.append(f"name中指定的文件名'{name}'在sample中不存在")
    
    # 3. 检查图像（如果有）
    if 'name' in json_data and json_data.get('media') == 'image':
        for name in json_data.get('name', []):
            if name in sample:
                image_data = sample[name]
                if not isinstance(image_data, bytes):
                    errors.append(f"图像数据'{name}'不是bytes类型")
                elif len(image_data) == 0:
                    errors.append(f"图像数据'{name}'为空")
    
    return len(errors) == 0, errors


def verify_sample_loader(sample: Dict[str, Any], nv_meta_dir: Path) -> tuple[bool, str]:
    """
    验证sample_loader能否正确解析样本
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # 导入sample_loader
        import sys
        sys.path.insert(0, str(nv_meta_dir))
        from sample_loader import sample_loader
        
        # 解析样本
        result = sample_loader(sample)
        
        # 检查结果格式
        required_keys = ['__key__', 'messages']
        for key in required_keys:
            if key not in result:
                return False, f"sample_loader返回结果缺少key: {key}"
        
        if not isinstance(result['messages'], list):
            return False, "sample_loader返回的messages不是列表"
        
        # 检查图像
        if result.get('image'):
            if not isinstance(result['image'], list):
                return False, "sample_loader返回的image不是列表"
        
        return True, ""
    except Exception as e:
        return False, f"sample_loader解析失败: {str(e)}"


def verify_webdataset_format(dataset_dir: str, check_sample_loader: bool = True, num_samples: int = 10):
    """
    验证WebDataset格式
    
    Args:
        dataset_dir: WebDataset目录路径
        check_sample_loader: 是否检查sample_loader
        num_samples: 检查的样本数量
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 目录不存在: {dataset_path}")
        return False
    
    # 查找tar文件
    tar_files = sorted(dataset_path.glob("*.tar"))
    if len(tar_files) == 0:
        print(f"❌ 目录中没有找到tar文件: {dataset_path}")
        return False
    
    print(f"找到 {len(tar_files)} 个tar文件")
    print(f"检查前 {num_samples} 个样本...")
    print("="*60)
    
    all_valid = True
    total_checked = 0
    
    # 检查.nv-meta
    nv_meta_dir = dataset_path / ".nv-meta"
    if check_sample_loader and not nv_meta_dir.exists():
        print("⚠️  警告: .nv-meta目录不存在，跳过sample_loader验证")
        check_sample_loader = False
    
    # 读取并验证样本
    for tar_file in tar_files[:1]:  # 只检查第一个tar文件的前几个样本
        try:
            dataset = wds.WebDataset(str(tar_file))
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                
                total_checked += 1
                print(f"\n样本 {total_checked} (来自 {tar_file.name}):")
                print(f"  __key__: {sample.get('__key__', 'N/A')}")
                
                # 验证格式
                is_valid, errors = verify_sample_format(sample, i)
                if is_valid:
                    print("  ✓ 格式验证通过")
                else:
                    print(f"  ✗ 格式验证失败:")
                    for error in errors:
                        print(f"    - {error}")
                    all_valid = False
                
                # 验证sample_loader
                if check_sample_loader and is_valid:
                    loader_valid, loader_error = verify_sample_loader(sample, nv_meta_dir)
                    if loader_valid:
                        print("  ✓ sample_loader验证通过")
                    else:
                        print(f"  ✗ sample_loader验证失败: {loader_error}")
                        all_valid = False
                
        except Exception as e:
            print(f"  ✗ 读取tar文件失败: {e}")
            all_valid = False
    
    print("\n" + "="*60)
    if all_valid:
        print(f"✅ 所有检查通过！共检查 {total_checked} 个样本")
    else:
        print(f"❌ 检查发现问题！共检查 {total_checked} 个样本")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='验证WebDataset格式')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/root/innovator_data_wenzichen/Innovator-VL-Insturct-Data-webdataset',
        help='WebDataset目录路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='检查的样本数量（默认: 10）'
    )
    parser.add_argument(
        '--no-sample-loader',
        action='store_true',
        help='跳过sample_loader验证'
    )
    
    args = parser.parse_args()
    
    verify_webdataset_format(
        args.dataset_dir,
        check_sample_loader=not args.no_sample_loader,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()


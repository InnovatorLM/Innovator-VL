#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import sys
import json
import torch
import argparse
from os.path import dirname
from copy import deepcopy
from einops import rearrange
from safetensors.torch import load_file, save_file

SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.custom.innovator_vl_30b_a3b.util import (
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_ep,
    save_megatron_checkpoint_tp_ep
)

def parse_args(title=None):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Merger Arguments', allow_abbrev=False)
    group = parser.add_argument_group(title='checkpoint')
    group.add_argument('--language_model_path', type=str, help="Path to language model."),
    group.add_argument('--vision_model_path', type=str, help="Path to vision model."),
    group.add_argument('--vision_patch', type=str, help="Path to vision patch."),
    group.add_argument('--adapter_path', type=str, help="Path to adapter."),
    group.add_argument("--save_ckpt_path", type=str, help="Path to save checkpoint.")
    group.add_argument("--megatron_path", type=str, help="Base directory of Megatron repository")
    group.add_argument("--tensor_model_parallel_size", type=int, default=1, help="Tensor parallel size.")
    group.add_argument("--pipeline_model_parallel_size", type=int, default=1, help="Pipeline parallel size.")

    return parser.parse_args()


def merge_dict(source, destination):
    """ merge two dictionaries recursively """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value


args = parse_args()
if args.megatron_path is not None:
    sys.path.insert(0, args.megatron_path)

print("===== merge megatron checkpoints ======")


# 仅 LLM 使用 2D 装载，其他模块使用原来的 1D（仅 TP）装载
language_model = load_megatron_checkpoint_tp_ep(args.language_model_path)
vision_model = load_megatron_checkpoint(args.vision_model_path)
adapter = load_megatron_checkpoint(args.adapter_path)
patch = load_megatron_checkpoint(args.vision_patch)

# 解析 LLM 的 2D 维度：state_dict[tp][ep]
tp_size = len(language_model)
assert tp_size > 0, "language_model(tp) 维度为空"
ep_size = len(language_model[0])
assert ep_size > 0, "language_model(ep) 维度为空"

def merge_dict(src_dict, dst_dict):
    """将src_dict的内容合并到dst_dict中"""
    for key, value in src_dict.items():
        if key in dst_dict:
            if isinstance(value, dict) and isinstance(dst_dict[key], dict):
                merge_dict(value, dst_dict[key])
            # 如果key已存在且不是字典，跳过以避免覆盖
        else:
            dst_dict[key] = value

# 合并：把每个 TP 分片的模块权重，广播到该 TP 分片的所有 EP 分片
for module_name, module in [("vision", vision_model), ("adapter", adapter), ("patch", patch)]:
    assert isinstance(module, list), f"{module_name} 模块应为 1D TP 分片列表"
    assert len(module) == tp_size, (
        f"{module_name} 的 TP 分片数({len(module)}) 与 LLM 的 TP 分片数({tp_size}) 不一致"
    )
    for t in range(tp_size):
        src = module[t]
        assert 'model' in src, f"{module_name}[{t}] 缺少 'model' 键"
        for e in range(ep_size):
            dst = language_model[t][e]
            assert 'model' in dst, f"LLM[tp={t}][ep={e}] 缺少 'model' 键"
            merge_dict(src['model'], dst['model'])

# 保存为 2D Megatron 布局
save_megatron_checkpoint_tp_ep(language_model, args.save_ckpt_path)
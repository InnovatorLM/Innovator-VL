#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import os
import torch
import json
import resource

import concurrent.futures
from convert_checkpoint.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.common_checkpoint import CommonCheckpoint

from convert_checkpoint.utils import (
    transpose_shape0,
    touch_file,
    check_all_done,
    get_done_keys,
    make_hf_sub_checkpoints,
)

from convert_checkpoint import utils
from convert_checkpoint.common_checkpoint import (
    WORD_EMBEDDINGS,
    WORD_POSITION_EMBEDDINGS,
    WORD_BLOCK_POSITION_EMBEDDINGS,
    TRANSFORMER,
    LAYER_PREFIX,
    INPUT_LAYERNORM,
    ROTARY_EMB_INV_FREQ,
    ATTENTION_ROTARY_EMB_INV_FREQ,
    ATTENTION_QUERY_KEY_VALUE,
    ATTENTION_QKV_MAP,
    ATTENTION_DENSE,
    POST_ATTENTION_LAYERNORM,
    MOE_GATE,
    MOE_GATE_BIAS,
    MOE_MLP,
    MOE_EXPERT,
    MOE_SHARED_EXPERT,
    MLP_DENSE_H_TO_4H,
    MLP_DENSE_4H_TO_H,
    POST_MLP_LAYERNORM,
    FINAL_LAYERNORM,
    WORD_EMBEDDINGS_FOR_HEAD,
    MTP_WORD_EMBEDDING,
    MTP_ENORM,
    MTP_HNORM,
    MTP_EH_PROJ,
    MTP_SHARED_HEAD_NORM,
    MTP_SHARED_HEAD_HEAD,
)


def merge_transformers_sharded_states(path, num_checkpoints, load_safe=False):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    if load_safe:
        from safetensors.torch import load_file
    args = parse_args()
    state_dict = {}
    current_chunks = [None] * (num_checkpoints + 1)
    def load_files(i):
        if load_safe:
            if args.checkpoint_format is not None:
                checkpoint_path = os.path.join(path, args.checkpoint_format.format(i=i, num_checkpoints=num_checkpoints))
            else:
                checkpoint_path = os.path.join(path, f"model-{i:05d}-of-{num_checkpoints:05d}.safetensors")
            current_chunks[i] = load_file(checkpoint_path, device="cpu")
        else:
            if args.checkpoint_format is not None:
                checkpoint_path = os.path.join(path, args.checkpoint_format.format(i=i, num_checkpoints=num_checkpoints))
            else:
                checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
            current_chunks[i] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"Loaded huggingface checkpoint: {checkpoint_path}")
    if args.max_workers is None:
        for i in range(1, num_checkpoints + 1):
            load_files(i)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i in range(1, num_checkpoints + 1):
                executor.submit(load_files, i)
        executor.shutdown(wait=True)
    for i in range(1, num_checkpoints + 1):
        state_dict.update(current_chunks[i])
    return state_dict


class HuggingFaceCheckpoint(AbstractCheckpoint):
    """
       HuggingFaceCheckpoint
    """

    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.optim_dict = None
        self.sched_dict = None

    def update_tensor(self, key, value):
        if value is not None:
            self.state_dict[key] = value

    @staticmethod
    def convert_from_common(c_ckpt, c_config):
        """
        Convert HuggingFace checkpoint to common checkpoint.

            Args:
                c_cpkt: CommonCheckpoint
                c_config: CommonConfig
        """

        print("==================== Common -> HuggingFace ====================")

        name_map = c_config.get("name_map")["huggingface"]
        cargs = c_config.get_args("common")
        hargs = c_config.get_args("huggingface")
        args = parse_args()
        cache_path = args.cache_path
        if args.load_platform == "megatron":
            margs = c_config.get_args("megatron")
        else:
            margs = c_config.get_args("mcore")
        dtype = c_config.get_dtype()
        num_nextn_predict_layers = hargs.get("num_nextn_predict_layers", 0)
        ori_num_layers = cargs["num_layers"]
        num_layers = ori_num_layers + num_nextn_predict_layers
        first_k_dense_replace = margs.get("first_k_dense_replace", None)
        sub_num_layers_for_save = args.sub_num_layers_for_save
        if utils.LOADED_LAYERS is None and sub_num_layers_for_save is None and \
                args.max_workers is not None and args.max_workers > 1:
            args.sub_num_layers_for_save = 5
            sub_num_layers_for_save = args.sub_num_layers_for_save
        separate_dtype = c_config.get("separate_dtype")

        if args.num_experts is not None:
            if MOE_MLP in name_map:
                moe_mlp = name_map[MOE_MLP]
            moe_expert = name_map[MOE_EXPERT]
            if MOE_SHARED_EXPERT in name_map:
                moe_shared_expert = name_map[MOE_SHARED_EXPERT]

        h_ckpt = HuggingFaceCheckpoint(num_layers)
        h_ckpt.set_dtype(dtype)

        # 1.1 word_embeddings
        if WORD_EMBEDDINGS in name_map:
            name = name_map[WORD_EMBEDDINGS] + ".weight"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(name, c_ckpt.get_word_embedding())
            c_ckpt.set_word_embedding(None)

        # 1.2 add_position_embedding:
        if margs.get("add_position_embedding", False):
            name = name_map[WORD_POSITION_EMBEDDINGS] + ".weight"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(name, c_ckpt.get_word_position_embedding())
            c_ckpt.set_word_position_embedding(None)

        if margs.get("add_block_position_embedding", False):
            name = name_map[WORD_BLOCK_POSITION_EMBEDDINGS] + ".weight"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(name, c_ckpt.get_word_block_position_embedding())
            c_ckpt.set_word_block_position_embedding(None)

        # 2. Transformer
        hidden_size = cargs["hidden_size"]
        heads = cargs["num_attention_heads"]
        use_rotary_position_embeddings = margs.get("use_rotary_position_embeddings", False)
        hidden_size_per_head = hidden_size // heads

        transformer = name_map[TRANSFORMER]
        layer_prefix = name_map[LAYER_PREFIX]

        if utils.LOADED_LAYERS is not None or sub_num_layers_for_save is not None:
            save_option = dict()
            save_option["save_optim"] = False
            save_option["save_safe"] = args.safetensors

        if utils.LOADED_LAYERS is not None:
            cur_sub_num_layer = min(utils.LOADED_LAYERS)
            sub_h_ckpts = {}
            sub_h_ckpts[cur_sub_num_layer] = HuggingFaceCheckpoint(num_layers)
        elif sub_num_layers_for_save is not None:
            cur_sub_num_layer = 0
            sub_h_ckpt_count = (num_layers + sub_num_layers_for_save - 1) // sub_num_layers_for_save
            sub_h_ckpts = {index : HuggingFaceCheckpoint(num_layers) for index in range(sub_h_ckpt_count)}

        if utils.LOADED_STATE_DICT is not None:
            del utils.LOADED_STATE_DICT
        for layer_id in range(num_layers):
            if utils.LOADED_LAYERS is not None and layer_id not in utils.LOADED_LAYERS:
                continue
            if utils.LOADED_LAYERS is None and layer_id > 0 and sub_num_layers_for_save is not None \
                    and layer_id % sub_num_layers_for_save == 0:
                sub_h_ckpts[cur_sub_num_layer].state_dict = h_ckpt.state_dict
                h_ckpt.state_dict = {}
                cur_sub_num_layer += 1

            if cache_path is None:
                one_layer_weights = None
            else:
                one_layer_weights = torch.load(f"{cache_path}/checkpoint_{layer_id}.pt",
                                               map_location="cpu", weights_only=False)
            # 2.1 input_layernorm
            if INPUT_LAYERNORM in name_map:
                name = name_map[INPUT_LAYERNORM]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                h_ckpt.update_tensor(name + ".weight",  c_ckpt.get_layer_input_layernorm_weight(
                    layer_id, one_layer_weights=one_layer_weights))
                h_ckpt.update_tensor(name + ".bias", c_ckpt.get_layer_input_layernorm_bias(
                    layer_id, one_layer_weights=one_layer_weights))
                c_ckpt.clear_layer_input_layernorm(layer_id, one_layer_weights=one_layer_weights)

            # 2.2 rotary_emb.inv_freq
            if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
                assert use_rotary_position_embeddings == True, \
                f"args.use_rotary_position_embeddings is required to be set to True since we capture the rotary_emb op"
                name = name_map[ATTENTION_ROTARY_EMB_INV_FREQ]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                h_ckpt.update_tensor(name, c_ckpt.get_layer_attention_rotary_emb_inv_freq(
                    layer_id, one_layer_weights=one_layer_weights))
                c_ckpt.clear_layer_attention_rotary_emb_inv_freq(layer_id, one_layer_weights=one_layer_weights)
            if ROTARY_EMB_INV_FREQ in name_map:
                assert use_rotary_position_embeddings == True, \
                f"args.use_rotary_position_embeddings is required to be set to True since we capture the rotary_emb op"
                name = name_map[ROTARY_EMB_INV_FREQ]
                print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                h_ckpt.update_tensor(name, c_ckpt.get_layer_attention_rotary_emb_inv_freq(
                    layer_id, one_layer_weights=one_layer_weights))
                c_ckpt.clear_layer_attention_rotary_emb_inv_freq(layer_id, one_layer_weights=one_layer_weights)

            # 2.3 self attention query_key_value
            if ATTENTION_QUERY_KEY_VALUE in name_map:
                name = name_map[ATTENTION_QUERY_KEY_VALUE]
                names = name if isinstance(name, list) else [name]
                weight = c_ckpt.get_layer_attention_query_key_value_weight(
                    layer_id, one_layer_weights=one_layer_weights)
                bias = c_ckpt.get_layer_attention_query_key_value_bias(layer_id, one_layer_weights=one_layer_weights)

                # transpose weight in shape[0] for llama
                transpose_query_key_value = margs.get("transpose_query_key_value", False)
                num_key_value_heads = cargs.get("num_key_value_heads", heads)
                assert heads % num_key_value_heads == 0
                num_repeats = heads // num_key_value_heads
                num_splits = num_repeats + 2 # repeats*Q + K + V
                if not transpose_query_key_value:
                    assert len(names) == 1
                    weight = [weight]
                    bias = [bias]
                else:
                    weight = transpose_shape0(weight, num_key_value_heads, num_splits)
                    weight = list(torch.chunk(weight, num_splits, dim=0))
                    q, k, v = torch.cat(weight[:-2]), weight[-2], weight[-1]
                    q = transpose_shape0(q, num_repeats, num_key_value_heads)
                    if len(names) == 1:
                        weight = [torch.cat((q, k, v), dim=0)]
                    else:
                        assert len(names) == 3
                        weight = [q, k, v]


                    if bias is not None:
                        bias = transpose_shape0(bias, num_key_value_heads, num_splits)
                        bias = list(torch.chunk(bias, num_splits, dim=0))
                        q, k, v = torch.cat(bias[:-2]), bias[-2], bias[-1]
                        q = transpose_shape0(q, num_repeats, num_key_value_heads)
                        if len(names) == 1:
                            bias = [torch.cat((q, k, v), dim=0)]
                        else:
                            assert len(names) == 3
                            bias = [q, k, v]

                for i in range(len(names)):
                    item = f"{transformer}.{layer_prefix}.{layer_id}.{names[i]}"
                    print(f"> update_tensor: {item}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    h_ckpt.update_tensor(item + ".weight", weight[i])
                    if bias is not None:
                        h_ckpt.update_tensor(item + ".bias", bias[i])
                c_ckpt.clear_layer_attention_query_key_value(layer_id, one_layer_weights=one_layer_weights)
            if ATTENTION_QKV_MAP in name_map:
                name = name_map[ATTENTION_QKV_MAP]
                names = name if isinstance(name, list) else [name]
                for sub_key, name_hf in name_map[ATTENTION_QKV_MAP].items():
                    item = f"{transformer}.{layer_prefix}.{layer_id}.{name_hf}"
                    weight, weight_scale_inv = c_ckpt.get_layer_attention_weight_by_name(
                        sub_key, layer_id, one_layer_weights=one_layer_weights)
                    if weight is None:
                        continue
                    bias = c_ckpt.get_layer_attention_bias_by_name(
                        sub_key, layer_id, one_layer_weights=one_layer_weights)
                    print(f"> update_tensor: {item}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    h_ckpt.update_tensor(item + ".weight", weight)
                    if bias is not None:
                        h_ckpt.update_tensor(item + ".bias", bias)
                    if weight_scale_inv is not None:
                        h_ckpt.update_tensor(item + ".weight_scale_inv", weight_scale_inv)
                    c_ckpt.clear_layer_attention_by_name(sub_key, layer_id, one_layer_weights=one_layer_weights)

            # 2.4 self attention dense
            name = name_map[ATTENTION_DENSE]
            name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            weight, weight_scale_inv = c_ckpt.get_layer_attention_dense_weight(
                layer_id, one_layer_weights=one_layer_weights)
            h_ckpt.update_tensor(name + ".weight",  weight)
            h_ckpt.update_tensor(name + ".bias", c_ckpt.get_layer_attention_dense_bias(
                layer_id, one_layer_weights=one_layer_weights))
            if weight_scale_inv is not None:
                h_ckpt.update_tensor(name + ".weight_scale_inv", weight_scale_inv)
            c_ckpt.clear_layer_attention_dense(layer_id, one_layer_weights=one_layer_weights)

            # 2.5 post attention layernorm
            name = name_map[POST_ATTENTION_LAYERNORM]
            name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(name + ".weight",  c_ckpt.get_layer_post_attention_layernorm_weight(
                layer_id, one_layer_weights=one_layer_weights))
            h_ckpt.update_tensor(name + ".bias", c_ckpt.get_layer_post_attention_layernorm_bias(
                layer_id, one_layer_weights=one_layer_weights))
            c_ckpt.clear_layer_post_attention_layernorm(layer_id, one_layer_weights=one_layer_weights)

            # 2.6 mlp dense h_to_4h
            name = name_map[MLP_DENSE_H_TO_4H]
            names = name if isinstance(name, list) else [name]
            def update_ffn_tensor_node(item_prefix, weight, bias, weight_scale_inv=None):
                if weight is None:
                    return
                weight = torch.chunk(weight, len(names), dim=0)
                if bias is not None:
                    bias = torch.chunk(bias, len(names), dim=0)
                if weight_scale_inv is not None:
                    weight_scale_inv = torch.chunk(weight_scale_inv, len(names), dim=0)

                for i in range(len(names)):
                    item = f"{item_prefix}.{names[i]}"
                    print(f"> update_tensor: {item}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    h_ckpt.update_tensor(item + ".weight", weight[i])
                    if bias is not None:
                        h_ckpt.update_tensor(item + ".bias", bias[i])
                    if weight_scale_inv is not None:
                        h_ckpt.update_tensor(item + ".weight_scale_inv", weight_scale_inv[i])
            if args.num_experts is None:
                weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_h_to_4h_weight(
                    layer_id, one_layer_weights=one_layer_weights)
                bias = c_ckpt.get_layer_mlp_dense_h_to_4h_bias(layer_id, one_layer_weights=one_layer_weights)
                update_ffn_tensor_node(f"{transformer}.{layer_prefix}.{layer_id}",
                                       weight, bias, weight_scale_inv=weight_scale_inv)
                c_ckpt.clear_layer_mlp_dense_h_to_4h(layer_id, one_layer_weights=one_layer_weights)
            else:
                if first_k_dense_replace is not None and layer_id < first_k_dense_replace:
                    weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_h_to_4h_weight(
                        layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                    bias = c_ckpt.get_layer_mlp_dense_h_to_4h_bias(
                        layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                    update_ffn_tensor_node(f"{transformer}.{layer_prefix}.{layer_id}.{moe_mlp}",
                                           weight, bias, weight_scale_inv=weight_scale_inv)
                    c_ckpt.clear_layer_mlp_dense_h_to_4h(layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                else:
                    name = name_map[MOE_GATE]
                    hf_name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                    print(f"> update_tensor: {hf_name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    weight = c_ckpt.get_layer_moe_gate_weight(layer_id, one_layer_weights=one_layer_weights)
                    h_ckpt.update_tensor(hf_name + ".weight", weight)
                    if MOE_GATE_BIAS in name_map:
                        bias_name = name_map[MOE_GATE_BIAS]
                        bias_name = f"{transformer}.{layer_prefix}.{layer_id}.{bias_name}"
                    else:
                        bias_name = hf_name + ".bias"
                    bias = c_ckpt.get_layer_moe_gate_bias(layer_id, one_layer_weights=one_layer_weights)
                    h_ckpt.update_tensor(bias_name, bias)
                    c_ckpt.clear_layer_moe_gate(layer_id, one_layer_weights=one_layer_weights)

                    if MOE_SHARED_EXPERT in name_map:
                        weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_h_to_4h_weight(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                        bias = c_ckpt.get_layer_mlp_dense_h_to_4h_bias(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                        update_ffn_tensor_node(f"{transformer}.{layer_prefix}.{layer_id}.{moe_shared_expert}",
                                               weight, bias, weight_scale_inv=weight_scale_inv)
                        c_ckpt.clear_layer_mlp_dense_h_to_4h(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                    for expert_id in range(0, args.num_experts):
                        weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_h_to_4h_weight(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)
                        bias = c_ckpt.get_layer_mlp_dense_h_to_4h_bias(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)
                        update_ffn_tensor_node(f"{transformer}.{layer_prefix}.{layer_id}.{moe_expert}.{expert_id}",
                                               weight, bias, weight_scale_inv=weight_scale_inv)
                        c_ckpt.clear_layer_mlp_dense_h_to_4h(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)

            # 2.7 mlp dense 4h_to_h
            if args.num_experts is None:
                name = name_map[MLP_DENSE_4H_TO_H]
                hf_name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                print(f"> update_tensor: {hf_name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_4h_to_h_weight(
                    layer_id, one_layer_weights=one_layer_weights)
                h_ckpt.update_tensor(hf_name + ".weight", weight)
                h_ckpt.update_tensor(hf_name + ".bias", c_ckpt.get_layer_mlp_dense_4h_to_h_bias(
                    layer_id, one_layer_weights=one_layer_weights))
                h_ckpt.update_tensor(hf_name + ".weight_scale_inv", weight_scale_inv)
                c_ckpt.clear_layer_mlp_dense_4h_to_h(layer_id, one_layer_weights=one_layer_weights)
            else:
                name = name_map[MLP_DENSE_4H_TO_H]
                if first_k_dense_replace is not None and layer_id < first_k_dense_replace:
                    hf_name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_mlp}.{name}"
                    print(f"> update_tensor: {hf_name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                    weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_4h_to_h_weight(
                        layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                    h_ckpt.update_tensor(hf_name + ".weight",  weight)
                    h_ckpt.update_tensor(hf_name + ".bias", c_ckpt.get_layer_mlp_dense_4h_to_h_bias(
                        layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights))
                    h_ckpt.update_tensor(hf_name + ".weight_scale_inv", weight_scale_inv)
                    c_ckpt.clear_layer_mlp_dense_4h_to_h(layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                else:
                    if MOE_SHARED_EXPERT in name_map:
                        hf_name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_shared_expert}.{name}"
                        print(f"> update_tensor: {hf_name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                        weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_4h_to_h_weight(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                        h_ckpt.update_tensor(hf_name + ".weight", weight)
                        h_ckpt.update_tensor(hf_name + ".bias", c_ckpt.get_layer_mlp_dense_4h_to_h_bias(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights))
                        h_ckpt.update_tensor(hf_name + ".weight_scale_inv", weight_scale_inv)
                        c_ckpt.clear_layer_mlp_dense_4h_to_h(
                            layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                    for expert_id in range(0, args.num_experts):
                        hf_name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_expert}.{expert_id}.{name}"
                        print(f"> update_tensor: {hf_name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                        weight, weight_scale_inv = c_ckpt.get_layer_mlp_dense_4h_to_h_weight(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)
                        h_ckpt.update_tensor(hf_name + ".weight",  weight)
                        h_ckpt.update_tensor(hf_name + ".bias", c_ckpt.get_layer_mlp_dense_4h_to_h_bias(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights))
                        h_ckpt.update_tensor(hf_name + ".weight_scale_inv", weight_scale_inv)
                        c_ckpt.clear_layer_mlp_dense_4h_to_h(
                            layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)

            # 2.8 post mlp layernorm
            if POST_MLP_LAYERNORM in name_map:
                name = name_map[POST_MLP_LAYERNORM]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                h_ckpt.update_tensor(name + ".weight",  c_ckpt.get_layer_post_mlp_layernorm_weight(
                    layer_id, one_layer_weights=one_layer_weights))
                h_ckpt.update_tensor(name + ".bias", c_ckpt.get_layer_post_mlp_layernorm_bias(
                    layer_id, one_layer_weights=one_layer_weights))
                c_ckpt.clear_layer_post_mlp_layernorm(layer_id, one_layer_weights=one_layer_weights)

            if MTP_WORD_EMBEDDING in name_map and layer_id >= ori_num_layers:
                (mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj), \
                    (mtp_shared_head_norm, mtp_shared_head_head) = c_ckpt.get_layer_mtp_weight(
                        layer_id, one_layer_weights=one_layer_weights)
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_WORD_EMBEDDING]}.weight",
                                     mtp_word_embedding)
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_ENORM]}.weight", mtp_enorm)
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_HNORM]}.weight", mtp_hnorm)
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_EH_PROJ]}.weight",
                                     mtp_eh_proj)
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_SHARED_HEAD_NORM]}.weight",
                                     mtp_shared_head_norm.clone())
                h_ckpt.update_tensor(f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_SHARED_HEAD_HEAD]}.weight",
                                     mtp_shared_head_head.clone())
                print(f"> update_tensor mtp, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}. "\
                      f"mtp_word_embedding: {mtp_word_embedding.shape}, "\
                      f"mtp_enorm: {mtp_enorm.shape}, mtp_hnorm: {mtp_hnorm.shape}, "\
                      f"mtp_eh_proj: {mtp_eh_proj.shape}, mtp_shared_head_norm: {mtp_shared_head_norm.shape}, "\
                      f"mtp_shared_head_head: {mtp_shared_head_head.shape}")
                c_ckpt.clear_layer_mtp_weight(layer_id, one_layer_weights=one_layer_weights)

        # 2.9 final_layernorm
        if FINAL_LAYERNORM in name_map:
            name = name_map[FINAL_LAYERNORM]
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(f"{name}.weight", c_ckpt.get_final_layernorm_weight())
            h_ckpt.update_tensor(f"{name}.bias", c_ckpt.get_final_layernorm_bias())
            c_ckpt.clear_final_layernorm()

        # 3 word embedding for head
        if WORD_EMBEDDINGS_FOR_HEAD in name_map:
            name = name_map[WORD_EMBEDDINGS_FOR_HEAD] + ".weight"
            print(f"> update_tensor: {name}, max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            h_ckpt.update_tensor(name, c_ckpt.get_word_embeddings_for_head_weight())
            c_ckpt.clear_word_embeddings_for_head()

        # 4 optimizer
        if c_ckpt.has_optimizer():
            h_ckpt.optim_dict = c_ckpt.state_dict["optimizer"]["optimizer"]
            h_ckpt.sched_dict = c_ckpt.state_dict["optimizer"]["opt_param_scheduler"] # todo

        if utils.LOADED_LAYERS is not None:
            sub_h_ckpts[cur_sub_num_layer].state_dict = h_ckpt.state_dict
            h_ckpt.state_dict = {}
        elif sub_num_layers_for_save is not None:
            sub_h_ckpts[cur_sub_num_layer].state_dict = h_ckpt.state_dict
            h_ckpt.state_dict = {}
        def save_sub_ckpt(sub_num_layer):
            if utils.LOADED_MIN_E is None:
                sub_dir_name= sub_num_layer
            else:
                sub_dir_name = sub_num_layer * 1000 + utils.LOADED_MIN_E
            sub_h_ckpts[sub_num_layer].save(
                f"{args.save_ckpt_path}/sub_checkpoint/{sub_dir_name}", None, **save_option)
        if args.max_workers is not None and args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for sub_num_layer in sorted(sub_h_ckpts.keys()):
                    futures.append(executor.submit(save_sub_ckpt, sub_num_layer))

            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e
        elif utils.LOADED_LAYERS is not None or sub_num_layers_for_save is not None:
            for sub_num_layer in sorted(sub_h_ckpts.keys()):
                if utils.LOADED_MIN_E is None:
                    sub_dir_name= sub_num_layer
                else:
                    sub_dir_name = sub_num_layer * 1000 + utils.LOADED_MIN_E
                sub_h_ckpts[sub_num_layer].save(
                    f"{args.save_ckpt_path}/sub_checkpoint/{sub_dir_name}", None, **save_option)
        if utils.LOADED_LAYERS is None and sub_num_layers_for_save is not None:
            make_hf_sub_checkpoints(args.save_ckpt_path)

        del c_ckpt.state_dict
        return h_ckpt

    def convert_to_common(self, config):
        """
        Convert HuggingFace checkpoint to common checkpoint.

            Args:
                config: CommonConfig
        """

        print("==================== HuggingFace -> Common ====================")

        name_map = config.get_name_map("huggingface")
        cargs = config.get_args("common")
        args = parse_args()
        hargs = config.get_args("huggingface")
        if args.save_platform == "megatron":
            margs = config.get_args("megatron")
        else:
            margs = config.get_args("mcore")
        cache_path = args.cache_path
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
        num_nextn_predict_layers = hargs.get("num_nextn_predict_layers", 0)
        ori_num_layers = cargs["num_layers"]
        num_layers = ori_num_layers + num_nextn_predict_layers

        c_ckpt = CommonCheckpoint(num_layers)
        c_ckpt.set_dtype(self.dtype)

        # 1.1 word_embeddings
        if WORD_EMBEDDINGS in name_map:
            name = name_map[WORD_EMBEDDINGS] + ".weight"
            word_embeddings_weight = self.state_dict[name]
            c_ckpt.set_word_embedding(word_embeddings_weight)
            print(f"> word_embeddings, shape: {word_embeddings_weight.shape}")

        # 1.2 add_position_embedding:
        if margs.get("add_position_embedding", False):
            name = name_map[WORD_POSITION_EMBEDDINGS] + ".weight"
            weight = self.state_dict[name]
            c_ckpt.set_word_position_embedding(weight)
            print(f"> add_position_embedding, shape: {weight.shape}")

        if margs.get("add_block_position_embedding", False):
            name = name_map[WORD_BLOCK_POSITION_EMBEDDINGS] + ".weight"
            weight = self.state_dict[name]
            c_ckpt.set_word_block_position_embedding(weight)
            print(f"> add_block_position_embedding, shape: {weight.shape}")

        # 2. Transformer
        hidden_size = cargs["hidden_size"]
        heads = cargs["num_attention_heads"]
        use_rotary_position_embeddings = margs.get("use_rotary_position_embeddings", False)
        hidden_size_per_head = hidden_size // heads

        transformer = name_map[TRANSFORMER]
        layer_prefix = name_map[LAYER_PREFIX]

        def set_one_layer_weight(layer_id):
            if cache_path is None:
                one_layer_weights = None
            else:
                if os.path.exists(f"{cache_path}/checkpoint_{layer_id}.pt") and \
                    os.path.exists(f"{cache_path}/checkpoint_{layer_id}.done"):
                    return
                if os.path.exists(f"{cache_path}/checkpoint_{layer_id}.done"):
                    os.remove(f"{cache_path}/checkpoint_{layer_id}.done")
                if os.path.exists(f"{cache_path}/checkpoint_{layer_id}.pt"):
                    os.remove(f"{cache_path}/checkpoint_{layer_id}.pt")
                one_layer_weights = {}
            # 2.1 input_layernorm
            if INPUT_LAYERNORM in name_map:
                name = name_map[INPUT_LAYERNORM]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                weight = self.state_dict[name + ".weight"]
                bias = self.state_dict.get(name + ".bias")
                c_ckpt.set_layer_input_layernorm(layer_id, weight, bias, one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} {name} weight: {weight.shape}")

            # 2.2 rotary_emb.inv_freq
            if ROTARY_EMB_INV_FREQ in name_map:
                assert use_rotary_position_embeddings == True, \
                f"args.use_rotary_position_embeddings is required to be set to True since we capture the rotary_emb op"
                name = name_map[ROTARY_EMB_INV_FREQ]
                param = self.state_dict[name]
                c_ckpt.set_layer_attention_rotary_emb_inv_freq(layer_id, param, one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} {name}: param shape: {param.shape}")
            if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
                assert use_rotary_position_embeddings == True, \
                f"args.use_rotary_position_embeddings is required to be set to True since we capture the rotary_emb op"
                name = name_map[ATTENTION_ROTARY_EMB_INV_FREQ]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                if name in self.state_dict:
                    param = self.state_dict[name]
                else:
                    dim = hidden_size // heads
                    param = 1.0 / (hargs.get("rotary_base", 10000) ** (torch.arange(0, dim, 2).float() / dim))
                c_ckpt.set_layer_attention_rotary_emb_inv_freq(layer_id, param, one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} {name}: param shape: {param.shape}")

            # 2.3 self attention query_key_value
            if ATTENTION_QUERY_KEY_VALUE in name_map:
                transpose_query_key_value = margs.get("transpose_query_key_value", False)
                name = name_map[ATTENTION_QUERY_KEY_VALUE]
                names = name if isinstance(name, list) else [name]
                weight = []
                bias = []
                for item  in names:
                    item = f"{transformer}.{layer_prefix}.{layer_id}.{item}"
                    weight.append(self.state_dict[item + ".weight"])
                    bias.append(self.state_dict.get(item + ".bias"))

                if not transpose_query_key_value:
                    assert len(names) == 1
                    weight = torch.cat(weight, dim=0)
                    bias = torch.cat(bias, dim=0) if None not in bias else None
                else:
                    num_key_value_heads = cargs.get("num_key_value_heads", heads)
                    assert heads % num_key_value_heads == 0
                    num_repeats = heads // num_key_value_heads
                    num_splits = num_repeats + 2

                    if len(names) == 1:
                        weight = list(torch.chunk(weight[0], num_splits, dim=0))
                        q, k, v = torch.cat(weight[:-2]), weight[-2], weight[-1]
                    else:
                        assert len(names) == 3
                        q, k, v = weight[0], weight[1], weight[2]
                    q = transpose_shape0(q, num_key_value_heads, num_repeats)
                    weight = torch.cat([q, k, v], dim=0)
                    weight = transpose_shape0(weight, num_splits, num_key_value_heads)

                    if None in bias:
                        assert all(x is None for x in bias)
                        bias = None
                    else:
                        if len(names) == 1:
                            bias = list(torch.chunk(bias[0], num_splits, dim=0))
                            q, k, v = torch.cat(bias[:-2]), bias[-2], bias[-1]
                        else:
                            assert len(names) == 3
                            q, k, v = bias[0], bias[1], bias[2]
                        q = transpose_shape0(q, num_key_value_heads, num_repeats)
                        bias = torch.cat([q, k, v], dim=0)
                        bias = transpose_shape0(bias, num_splits, num_key_value_heads)

                c_ckpt.set_layer_attention_query_key_value(
                    layer_id, weight, bias, one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} attention query key value weight: {weight.shape}")
            if ATTENTION_QKV_MAP in name_map:
                for key, name_hf in name_map[ATTENTION_QKV_MAP].items():
                    item = f"{transformer}.{layer_prefix}.{layer_id}.{name_hf}"
                    weight = self.state_dict[item + ".weight"]
                    bias = self.state_dict.get(item + ".bias")
                    scale_key = item + ".weight_scale_inv"
                    weight_scale_inv = self.state_dict[scale_key] if scale_key in self.state_dict else None
                    c_ckpt.set_layer_attention_by_name(
                        key, layer_id, weight, bias, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)

            # 2.4 self attention dense
            name = name_map[ATTENTION_DENSE]
            name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
            weight = self.state_dict[name + ".weight"]
            bias = self.state_dict.get(name + ".bias")
            weight_scale_inv = self.state_dict[name + ".weight_scale_inv"] \
                if name + ".weight_scale_inv" in self.state_dict else None
            c_ckpt.set_layer_attention_dense(layer_id, weight, bias, one_layer_weights=one_layer_weights,
                                             weight_scale_inv=weight_scale_inv)
            print(f"> layer {layer_id} attention dense weight: {weight.shape}")

            # 2.5 post attention layernorm
            name = name_map[POST_ATTENTION_LAYERNORM]
            name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
            weight = self.state_dict[name + ".weight"]
            bias = self.state_dict.get(name + ".bias")
            c_ckpt.set_layer_post_attention_layernorm(
                layer_id, weight, bias, one_layer_weights=one_layer_weights)
            print(f"> layer {layer_id} post attention layernorm weight: {weight.shape}")

            if args.num_experts is not None:
                if MOE_MLP in name_map:
                    moe_mlp = name_map[MOE_MLP]
                moe_expert = name_map[MOE_EXPERT]
                if MOE_SHARED_EXPERT in name_map:
                    moe_shared_expert = name_map[MOE_SHARED_EXPERT]

            def set_mlp_weight(is_moe_mlp = False, expert_id = None, is_shared = False):
                # 2.6 mlp dense h_to_4h
                name = name_map[MLP_DENSE_H_TO_4H]
                names = name if isinstance(name, list) else [name]
                weight = []
                bias = []
                weight_scale_inv = None
                for item in names:
                    if is_moe_mlp:
                        item = f"{transformer}.{layer_prefix}.{layer_id}.{moe_mlp}.{item}"
                    elif is_shared:
                        item = f"{transformer}.{layer_prefix}.{layer_id}.{moe_shared_expert}.{item}"
                    elif expert_id is None:
                        item = f"{transformer}.{layer_prefix}.{layer_id}.{item}"
                    else:
                        item = f"{transformer}.{layer_prefix}.{layer_id}.{moe_expert}.{expert_id}.{item}"
                    weight.append(self.state_dict[item + ".weight"])
                    bias.append(self.state_dict.get(item + ".bias"))
                    if item + ".weight_scale_inv" in self.state_dict:
                        if weight_scale_inv is None:
                            weight_scale_inv = []
                        weight_scale_inv.append(self.state_dict[item + ".weight_scale_inv"])

                weight = torch.cat(weight, dim=0)
                bias = torch.cat(bias, dim=0) if None not in bias else None
                weight_scale_inv = torch.cat(weight_scale_inv, dim=0) if weight_scale_inv is not None else None
                if is_moe_mlp:
                    c_ckpt.set_layer_mlp_dense_h_to_4h(
                        layer_id, weight, bias, is_moe_mlp=True, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id}, mlp dense h_to_4h weight: {weight.shape}")
                elif is_shared:
                    c_ckpt.set_layer_mlp_dense_h_to_4h(
                        layer_id, weight, bias, is_shared=True, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id}, shared_expert mlp dense h_to_4h weight: {weight.shape}")
                elif expert_id is None:
                    c_ckpt.set_layer_mlp_dense_h_to_4h(
                        layer_id, weight, bias, one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id} mlp dense h_to_4h weight: {weight.shape}")
                else:
                    c_ckpt.set_layer_mlp_dense_h_to_4h(
                        layer_id, weight, bias, expert_id=expert_id, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id}, expert_id {expert_id} mlp dense h_to_4h weight: {weight.shape}")

                # 2.7 mlp dense 4h_to_h
                name = name_map[MLP_DENSE_4H_TO_H]
                if is_moe_mlp:
                    name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_mlp}.{name}"
                elif is_shared:
                    name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_shared_expert}.{name}"
                elif expert_id is None:
                    name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                else:
                    name = f"{transformer}.{layer_prefix}.{layer_id}.{moe_expert}.{expert_id}.{name}"
                weight = self.state_dict[name + ".weight"]
                bias = self.state_dict.get(name + ".bias")
                weight_scale_inv = self.state_dict[name + ".weight_scale_inv"] \
                    if name + ".weight_scale_inv" in self.state_dict else None
                if is_moe_mlp:
                    c_ckpt.set_layer_mlp_dense_4h_to_h(
                        layer_id, weight, bias, is_moe_mlp=True, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id} moe mlp dense 4h_to_h weight: {weight.shape}")
                elif is_shared:
                    c_ckpt.set_layer_mlp_dense_4h_to_h(
                        layer_id, weight, bias, is_shared=True, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id} shared_expert mlp dense 4h_to_h weight: {weight.shape}")
                elif expert_id is None:
                    c_ckpt.set_layer_mlp_dense_4h_to_h(
                        layer_id, weight, bias, one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id} mlp dense 4h_to_h weight: {weight.shape}")
                else:
                    c_ckpt.set_layer_mlp_dense_4h_to_h(
                        layer_id, weight, bias, expert_id=expert_id, one_layer_weights=one_layer_weights,
                        weight_scale_inv=weight_scale_inv)
                    print(f"> layer {layer_id}, expert_id {expert_id} mlp dense 4h_to_h weight: {weight.shape}")

                # 2.8 post mlp layernorm
                if POST_MLP_LAYERNORM in name_map:
                    name = name_map[POST_MLP_LAYERNORM]
                    name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                    weight = self.state_dict[name + ".weight"]
                    bias = self.state_dict.get(name + ".bias")
                    c_ckpt.set_layer_post_mlp_layernorm(
                        layer_id, weight, bias, one_layer_weights=one_layer_weights)
                    print(f"> layer {layer_id} post mlp layernorm weight: {weight.shape}")

            first_k_dense_replace = hargs.get("first_k_dense_replace", None)
            if args.num_experts is None:
                set_mlp_weight()
            elif first_k_dense_replace is not None and layer_id < first_k_dense_replace:
                set_mlp_weight(is_moe_mlp=True)
            else:
                # moe gate
                name = name_map[MOE_GATE]
                name = f"{transformer}.{layer_prefix}.{layer_id}.{name}"
                if MOE_GATE_BIAS in name_map:
                    bias_name = name_map[MOE_GATE_BIAS]
                    bias_name = f"{transformer}.{layer_prefix}.{layer_id}.{bias_name}"
                else:
                    bias_name = name + ".bias"
                weight = self.state_dict[name + ".weight"]
                bias = self.state_dict.get(bias_name)
                c_ckpt.set_layer_moe_gate(layer_id, weight, bias, one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} moe gate weight: {weight.shape}")
                if bias is not None:
                    print(f"> layer {layer_id} moe gate bias: {bias.shape}")

                for expert_id in range(0, args.num_experts):
                    set_mlp_weight(expert_id=expert_id)

                if MOE_SHARED_EXPERT in name_map:
                    set_mlp_weight(expert_id=None, is_shared=True)

            if MTP_WORD_EMBEDDING in name_map and layer_id >= ori_num_layers:
                mtp_word_embedding = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_WORD_EMBEDDING]}.weight"]
                mtp_enorm = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_ENORM]}.weight"]
                mtp_hnorm = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_HNORM]}.weight"]
                mtp_eh_proj = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_EH_PROJ]}.weight"]
                mtp_shared_head_norm = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_SHARED_HEAD_NORM]}.weight"]
                mtp_shared_head_head = self.state_dict[
                    f"{transformer}.{layer_prefix}.{layer_id}.{name_map[MTP_SHARED_HEAD_HEAD]}.weight"]
                c_ckpt.set_layer_mtp_weight(layer_id, mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj, \
                                            mtp_shared_head_norm, mtp_shared_head_head, \
                                            one_layer_weights=one_layer_weights)
                print(f"> layer {layer_id} mtp. mtp_word_embedding: {mtp_word_embedding.shape}, "\
                        f"mtp_enorm: {mtp_enorm.shape}, mtp_hnorm: {mtp_hnorm.shape}, "\
                        f"mtp_eh_proj: {mtp_eh_proj.shape}, mtp_shared_head_norm: {mtp_shared_head_norm.shape}, "\
                        f"mtp_shared_head_head: {mtp_shared_head_head.shape}")

            if cache_path is not None:
                torch.save(one_layer_weights, f"{cache_path}/checkpoint_{layer_id}.pt")
                done_file_name = f"{cache_path}/checkpoint_{layer_id}.done"
                with open(done_file_name, 'w'):
                    os.utime(done_file_name, None)

        if args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for one_layer_id in range(num_layers):
                    futures.append(executor.submit(set_one_layer_weight, one_layer_id))

            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e
        else:
            for one_layer_id in range(num_layers):
                set_one_layer_weight(one_layer_id)

        # 2.9 final_layernorm
        if FINAL_LAYERNORM in name_map:
            name = name_map[FINAL_LAYERNORM]
            weight = self.state_dict[f"{name}.weight"]
            bias = self.state_dict.get(f"{name}.bias")
            c_ckpt.set_final_layernorm(weight, bias)
            print(f"> {name} final_layernorm weight: {weight.shape}")

        # 3 word embedding for head
        if WORD_EMBEDDINGS_FOR_HEAD in name_map:
            name = name_map[WORD_EMBEDDINGS_FOR_HEAD] + ".weight"
            weight = self.state_dict.get(name, word_embeddings_weight)
            c_ckpt.set_word_embeddings_for_head(weight)
            print(f"> {name} word embedding for head weight: {weight.shape}")

        # 4 optimizer
        if self.has_optimizer():
            opt_param_scheduler = self.sched_dict # todo
            c_ckpt.state_dict["optimizer"] = {
                "optimizer": self.optim_dict,
                "opt_param_scheduler": opt_param_scheduler
            }
        return c_ckpt

    def has_optimizer(self):
        """ has optimizer """
        return self.optim_dict is not None and self.sched_dict is not None

    def load(self, load_path, load_safe=False):
        """ load ckpt """
        if load_safe:
            from safetensors.torch import load_file
            sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
            if len(sub_dirs) == 1:
                checkpoint_name = "model.safetensors"
                self.state_dict = load_file(os.path.join(load_path, checkpoint_name), device="cpu")
            else:
                num_checkpoints = len(sub_dirs)
                self.state_dict = merge_transformers_sharded_states(load_path, num_checkpoints, True)
        else:
            sub_dirs = [x for x in os.listdir(load_path) if x.startswith("pytorch_model")]
            if len(sub_dirs) == 1:
                checkpoint_name = "pytorch_model.bin"
                self.state_dict = torch.load(os.path.join(load_path, checkpoint_name),
                                             map_location="cpu", weights_only=False)
            else:
                num_checkpoints = len(sub_dirs) - 1
                self.state_dict = merge_transformers_sharded_states(load_path, num_checkpoints)


    def print_memory_usage(self, desc):
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2  # 转为 MB
        print(f"{desc}内存占用: {mem:.2f} MB")

    def save(self, save_path, h_config=None, save_optim=True, save_safe=False):
        """ save ckpt """
        from huggingface_hub import split_torch_state_dict_into_shards
        from transformers.modeling_utils import SAFE_WEIGHTS_INDEX_NAME
        from safetensors.torch import save_file
        if not save_safe:
            print("Warning: --safetensors is required!!!")
        os.makedirs(save_path, exist_ok=True)
        state_dict_split = split_torch_state_dict_into_shards(self.state_dict)
        self.print_memory_usage(f"before save {save_path}")
        has_safetensor_file = False
        for shard_file, tensors in state_dict_split.filename_to_tensors.items():
            shard = {}
            for tensor in tensors:
                shard[tensor] = self.state_dict[tensor].contiguous()
                del self.state_dict[tensor]
            shard_path = os.path.join(save_path, shard_file)
            save_file(shard, shard_path, metadata={"format": "pt"})
            has_safetensor_file = True
            print(f"Saving HuggingFace shard to: {shard_path}")
        self.print_memory_usage(f"after save {save_path}")

        args = parse_args()
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        elif has_safetensor_file and (args.save_sub_checkpoint_by_pp or args.sub_num_layers_for_save is not None):
            for key in state_dict_split.tensor_to_filename.keys():
                if state_dict_split.tensor_to_filename[key] == "model.safetensors":
                    state_dict_split.tensor_to_filename[key] = "model-00001-of-00001.safetensors"
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_path, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            os.rename(os.path.join(save_path, 'model.safetensors'), \
                      os.path.join(save_path, 'model-00001-of-00001.safetensors'))

        if h_config is not None:
            h_config.save(save_path)
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""

Authors: lihaibing(lihaibing@baidu.com)
Date:    2025/07/22 15:10:30
"""
import argparse
import json
import os
import torch

ITER = "iter"
MBS = "mbs"
LAYER_NUMBER = "layer_number"
NAME_SPLIT = "."

def get_argument():
    """
    Get arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace-file-path-a', type=str,
        default="", help="file path")
    parser.add_argument(
        '--trace-file-path-b', type=str,
        default="", help="file path ")
    parser.add_argument(
        '--output-folder', type=str,
        default="./output", help="file path ")

    return parser.parse_args()

def get_common_files(folder_a, folder_b):
    """get common files"""
    # file name format like:
    #   decoder.layers.9.self_attention.linear_q_proj.bwd_dy.step1.rank000.pt
    a_files = {}
    b_files = {}
    for root, dirs, files in os.walk(folder_a):
        for sub_file in files:
            if sub_file.endswith(".pt"):
                a_files[sub_file] = True

    for root, dirs, files in os.walk(folder_b):
        for sub_file in files:
            if sub_file.endswith(".pt"):
                b_files[sub_file] = True

    comm_files = {}
    for key in a_files:
        if key in b_files:
            comm_files[key] = True

    return comm_files

def diff_comm_files(folder_a, folder_b, comm_files):
    """compare comm files for two folders"""
    results = {}
    for file_name in comm_files:
        file_a = folder_a + "/" + file_name
        file_b = folder_b + "/" + file_name
        tensor_a = torch.load(file_a)
        tensor_b = torch.load(file_b)
        delta = torch.sum(torch.abs(tensor_a - tensor_b)).item()
        sum_a = torch.sum(torch.abs(tensor_a)).item()
        ratio = 0
        if sum_a != 0:
            ratio = round(delta / sum_a, 3)
        results[file_name] = str(ratio) + "," + str(delta) + "," + str(sum_a)
    return results

def parse_file_name(file_name):
    """parse file name """
    # file name format like:
    #   decoder.layers.9.self_attention.linear_q_proj.bwd_dy.step1.rank000.pt
    # return: rank, iter, mbs, layer, keyword
    keys = file_name.split(NAME_SPLIT)
    assert len(keys) == 9, f"invalid name format : {file_name}"
    num_keys = len(keys)

    layer_index = -1
    index = 0
    rank = -1
    iter_num = -1
    for i in keys:
        if i.startswith("rank"):
            rank = int(i[4:])
        elif i.startswith("step"):
            iter_num = int(i[4:])
        elif i == "layers":
            layer_index = index + 1
        index += 1

    keyword = "."
    index = 0
    for i in keys:
        if i.startswith("rank") or i.startswith("step") or i == "layers" or index == layer_index:
            continue
        keyword += i
        keyword += "."
        index += 1

    layer_number = -1
    if layer_index != -1 and layer_index < num_keys:
        layer_number = int(keys[layer_index])

    # return: rank, iter, mbs, layer, keyword
    return rank, iter_num, 0, layer_number, keyword

def diff_2_folders(folder_a, folder_b):
    """compare two folders"""
    comm_files = get_common_files(folder_a, folder_b)
    diff_results = diff_comm_files(folder_a, folder_b, comm_files)
    # for all iters: [[iter0],[iter1],...]
    # for each iter: [{mbs0},{mbs1}...]
    # for each mbs: {'-1': {}, '0': {}, '1':{}, ...}
    # for each layer: {'key1':value1, 'key2':value2, ...}
    # for each value: ratio,delta,sum_a

    rank_dict = {}

    for key in diff_results:
        rank, iter_num, mbs, layer, keyword = parse_file_name(key)
        if rank not in rank_dict:
            rank_dict[rank] = []
        iters = rank_dict[rank]
        while iter_num >= len(iters):
            iters.append([])

        mbs_list = iters[iter_num]
        while mbs >= len(mbs_list):
            mbs_list.append({})

        mbs_dict = mbs_list[mbs]
        if layer not in mbs_dict:
            mbs_dict[layer] = {}

        keyword_dict = mbs_dict[layer]
        if keyword not in keyword_dict:
            keyword_dict[keyword] = diff_results[key]
        else:
            old_str = keyword_dict[keyword].split(",")
            add_str = diff_results[key].split(",")
            new_delta = float(old_str[1]) + float(add_str[1])
            new_sum = float(old_str[2]) + float(add_str[2])
            new_ratio = round(new_delta / new_sum, 3)
            keyword_dict[keyword] = str(new_ratio) + "," + str(new_delta) + "," + str(new_sum)

    return rank_dict

def compare(folder_a, folder_b):
    """compare two folders"""
    rank_dict = diff_2_folders(folder_a, folder_b)
    # for all iters: [[iter0],[iter1],...]
    # for each iter: [{mbs0},{mbs1}...]
    # for each mbs: {'-1': {}, '0': {}, '1':{}, ...}
    # for each layer: {'key1':value1, 'key2':value2, ...}
    # for each value: ratio,delta,sum_a
    return rank_dict


def save_for_rank(rank, iters, args):
    """save info for a specific rank"""
    file_name = args.output_folder + "/" + "starts_" + str(rank)
    save_trace_file = open(file_name, 'w', encoding='utf-8')
    for i in range(len(iters)):
        mbs_list = iters[i]
        for j in range(len(mbs_list)):
            layer_dict = mbs_list[j]
            for tmp_key in layer_dict:
                keyword_dict = layer_dict[tmp_key]
                keyword_dict[LAYER_NUMBER] = tmp_key
                keyword_dict[MBS] = j
                keyword_dict[ITER] = i
                save_trace_file.write(json.dumps(keyword_dict) + "\n")

    if save_trace_file is not None:
        save_trace_file.close()

if __name__ == "__main__":
    args = get_argument()
    rank_dict = compare(args.trace_file_path_a, args.trace_file_path_b)
    for rank in rank_dict:
        save_for_rank(rank, rank_dict[rank], args)

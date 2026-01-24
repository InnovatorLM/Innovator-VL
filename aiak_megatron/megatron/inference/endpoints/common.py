# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""common functions"""

import torch
import threading

GENERATE_NUM = 0
BEAM_NUM = 1
LOCK = threading.Lock()


def send_do_generate():
    """send do generate"""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(choice, 0)


def send_do_beam_search():
    """send do beam search"""
    choice = torch.tensor([BEAM_NUM], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(choice, 0)

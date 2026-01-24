# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Enums for model types"""

from enum import Enum

ModelType = Enum(
    'ModelType', ["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma"]
)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""abstract engine"""
from abc import ABC, abstractmethod
from typing import List


class AbstractEngine(ABC):
    """abstract class."""
    @staticmethod
    @abstractmethod
    def generate(self) -> dict:
        """The abstract backend's generate function.

        To define a new backend, implement this and return the outputs as a dictionary.

        Returns:
            dict: The output dictionary containing keys for `input_prompt`, `generated_text`, `generated_tokens`.
        """
        pass

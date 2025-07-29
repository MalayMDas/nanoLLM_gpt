"""
GPT Language Model Package

A clean, modular implementation of GPT with training, inference, and serving capabilities.
"""

__version__ = "1.0.0"
__author__ = "GPT Project Team"

from .model import GPT, GPTConfig
from .config import ModelConfig, TrainingConfig, GenerationConfig, APIConfig

__all__ = [
    "GPT",
    "GPTConfig",
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "APIConfig",
]

"""
Teacher model module for Knowledge Distillation.

This module provides functionality for:
- Loading the DeepSeek-R1-Distill-Llama-8B teacher model
- Capturing layer activations during forward pass
- Saving activations for knowledge distillation training
"""

from .teacher_model import load_teacher_model, get_layer_modules
from .hooks import ActivationStore, attach_hooks
from .dataset import TextDataset, create_dataloader
from . import config

__all__ = [
    'load_teacher_model',
    'get_layer_modules',
    'ActivationStore',
    'attach_hooks',
    'TextDataset',
    'create_dataloader',
    'config'
]

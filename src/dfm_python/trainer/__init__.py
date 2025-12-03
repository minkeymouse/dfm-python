"""Trainer classes for Dynamic Factor Models.

This package provides specialized PyTorch Lightning Trainer classes
for DFM and DDFM models with model-specific defaults and configurations.
"""

from .dfm import DFMTrainer
from .ddfm import DDFMTrainer

__all__ = [
    'DFMTrainer',
    'DDFMTrainer',
]


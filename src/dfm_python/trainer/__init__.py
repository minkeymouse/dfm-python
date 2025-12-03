"""Trainer classes for Dynamic Factor Models.

This package provides specialized PyTorch Lightning Trainer classes
for DFM and DDFM models with model-specific defaults and configurations.
"""

try:
    from .dfm import DFMTrainer
    from .ddfm import DDFMTrainer
    _has_trainers = True
except ImportError:
    _has_trainers = False
    DFMTrainer = None  # type: ignore
    DDFMTrainer = None  # type: ignore

__all__ = [
    'DFMTrainer',
    'DDFMTrainer',
]


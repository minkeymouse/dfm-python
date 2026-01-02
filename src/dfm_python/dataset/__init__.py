"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM, KDFM) that use gradient descent training.

This module provides:
- Base dataset class: DeepFactorModelDataset
- Dataset classes: DDFMDataset, DDFMMCDataset, KDFMDataset
- DataLoader factories: create_ddfm_dataloader, create_kdfm_dataloader
"""

from .base import DeepFactorModelDataset
from .ddfm_dataset import DDFMDataset, DDFMMCDataset
from .kdfm_dataset import KDFMDataset
from .dataloader import create_ddfm_dataloader, create_kdfm_dataloader
from .process import (
    TimeIndex,
    _get_scaler,
)

__all__ = [
    # Base class
    'DeepFactorModelDataset',
    # Datasets
    'DDFMDataset',
    'DDFMMCDataset',
    'KDFMDataset',
    # Dataloaders
    'create_ddfm_dataloader',
    'create_kdfm_dataloader',
    # Preprocessing
    'TimeIndex',
    '_get_scaler',
]

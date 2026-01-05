"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM, KDFM) that use gradient descent training.

This module provides:
- Base dataset class: BaseFactorModelDataset
- Dataset classes: DDFMDataset, KDFMDataset, DFMDataset
- DataLoader factories: create_ddfm_dataloader, create_kdfm_dataloader
"""

from .base import BaseFactorModelDataset
from .ddfm_dataset import DDFMDataset
from .kdfm_dataset import KDFMDataset
from .dfm_dataset import DFMDataset
from .dataloader import create_ddfm_dataloader, create_kdfm_dataloader
from .process import (
    TimeIndex,
    _get_scaler,
)

__all__ = [
    # Base class
    'BaseFactorModelDataset',
    # Datasets
    'DDFMDataset',
    'KDFMDataset',
    'DFMDataset',
    # Dataloaders
    'create_ddfm_dataloader',
    'create_kdfm_dataloader',
    # Preprocessing
    'TimeIndex',
    '_get_scaler',
]

"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM, KDFM) that use gradient descent training.

This module provides:
- Dataset classes: DDFMDataset, KDFMDataset, DFMDataset
- DataLoader factories: create_ddfm_dataloader, create_kdfm_dataloader
"""

from .ddfm_dataset import DDFMDataset
from .kdfm_dataset import KDFMDataset
from .dfm_dataset import DFMDataset
from .time import TimeIndex

__all__ = [
    # Datasets
    'DDFMDataset',
    'KDFMDataset',
    'DFMDataset',
    # Time utilities
    'TimeIndex',
]

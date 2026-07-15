"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM) that use gradient descent training.

This module provides:
- Dataset classes: DDFMDataset, DFMDataset
- DataLoader factories: create_ddfm_dataloader
"""

from .ddfm_dataset import DDFMDataset
from .dfm_dataset import DFMDataset
from .ivdfm_dataset import iVDFMDataset
from .afm_dataset import AFMDataset
from .time import TimeIndex

__all__ = [
    # Datasets
    'DDFMDataset',
    'DFMDataset',
    'iVDFMDataset',
    'AFMDataset',
    # Time utilities
    'TimeIndex',
]

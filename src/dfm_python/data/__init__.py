"""Data loading and dataset utilities for DFM.

This module provides:
- Dataset classes: DFMDataset (linear DFM), DDFMDataset (deep DDFM)
- DataLoader factories: create_dfm_dataloader, create_ddfm_dataloader
- Data reading: read_data, load_data
- Nowcast data classes moved to src.nowcasting in main project

Note: Transformation functions are application-specific and should be provided
by users in their preprocessing pipelines. The package does not include
transformation utilities to remain generic.
"""

from .dataset import DFMDataset, DDFMDataset
from .dataloader import create_dfm_dataloader, create_ddfm_dataloader
from .utils import read_data, load_data
# Nowcast data classes moved to src.nowcasting in main project

__all__ = [
    # Datasets
    'DFMDataset',
    'DDFMDataset',
    # Dataloaders
    'create_dfm_dataloader',
    'create_ddfm_dataloader',
    # Data reading
    'read_data',
    'load_data',
]


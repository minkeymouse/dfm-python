"""Data loading, transformation, and dataset utilities for DFM.

This module provides:
- Dataset classes: DFMDataset (linear DFM), DDFMDataset (deep DDFM)
- DataLoader factories: create_dfm_dataloader, create_ddfm_dataloader
- Data reading: read_data, load_data
- Transformation functions: pch_transform, pc1_transform, etc.
"""

from .dataset import DFMDataset, DDFMDataset
from .dataloader import create_dfm_dataloader, create_ddfm_dataloader
from .utils import read_data, load_data
from .transformation import (
    pch_transform,
    pc1_transform,
    pca_transform,
    cch_transform,
    cca_transform,
    log_transform,
    identity_transform,
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
)

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
    # Transformations
    'pch_transform',
    'pc1_transform',
    'pca_transform',
    'cch_transform',
    'cca_transform',
    'log_transform',
    'identity_transform',
    'make_pch_transformer',
    'make_pc1_transformer',
    'make_pca_transformer',
    'make_cch_transformer',
    'make_cca_transformer',
]


"""PyTorch DataLoader implementations for PyTorch-based Dynamic Factor Models.

This module provides DataLoader factories for PyTorch-based DFM models:
- Windowed sequences: Standard DataLoader with batching for neural network training
- Full sequences: DataLoader for models that process entire time series (Lightning-compatible)
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Union
from .dataset import DDFMDataset, KDFMDataset
from ..logger import get_logger

_logger = get_logger(__name__)


def _create_dataloader(
    dataset: Union[DDFMDataset, KDFMDataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    auto_pin_memory: bool = False
) -> DataLoader:
    """Create DataLoader (shared helper).
    
    Parameters
    ----------
    dataset : DDFMDataset or KDFMDataset
        Dataset instance
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data
    num_workers : int
        Number of worker processes
    pin_memory : bool
        Whether to pin memory for faster GPU transfer
    auto_pin_memory : bool, default False
        If True, automatically disable pin_memory if CUDA is not available
        
    Returns
    -------
    DataLoader
        PyTorch DataLoader
    """
    if auto_pin_memory and not torch.cuda.is_available():
        pin_memory = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


def create_ddfm_dataloader(
    dataset: DDFMDataset,
    batch_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader for windowed sequence datasets.
    
    This factory creates a standard PyTorch DataLoader with batching for models
    that train on windowed sequences (e.g., neural network-based models).
    Shuffling is typically enabled for training.
    
    Parameters
    ----------
    dataset : DDFMDataset
        Dataset instance with windowed sequences
    batch_size : int, default 100
        Batch size for training
    shuffle : bool, default True
        Whether to shuffle samples (typically True for training)
    num_workers : int, default 0
        Number of worker processes for DataLoader
    pin_memory : bool, default True
        Whether to pin memory for faster GPU transfer (typically True for GPU training)
        
    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader for windowed sequence training
    """
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        auto_pin_memory=True
    )


def create_kdfm_dataloader(
    dataset: KDFMDataset,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """Create DataLoader for full sequence datasets.
    
    This factory creates a PyTorch DataLoader for models that process the entire
    time series at once. The full sequence is typically used as a single batch.
    The DataLoader is needed for PyTorch Lightning compatibility, allowing
    models to work with standard Lightning training patterns.
    
    Parameters
    ----------
    dataset : KDFMDataset
        Dataset instance with full sequence data
    batch_size : int, optional
        Batch size. If None, uses full sequence (dataset length = 1).
        For full sequence datasets, this is typically 1.
    num_workers : int, default 0
        Number of worker processes for DataLoader
    pin_memory : bool, default False
        Whether to pin memory for faster GPU transfer
        
    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader for full sequence training
    """
    if batch_size is None:
        batch_size = len(dataset)
    
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        auto_pin_memory=False
    )


"""DataLoader implementations for DFM and DDFM.

This module provides specialized DataLoaders for linear DFM and deep DDFM.
DFM uses a simple DataLoader that returns full sequences.
DDFM uses a standard DataLoader with batching for neural network training.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Union
from .dataset import DFMDataset, DDFMDataset
from ..logger import get_logger

_logger = get_logger(__name__)


def _create_dataloader(
    dataset: Union[DFMDataset, DDFMDataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    auto_pin_memory: bool = False
) -> DataLoader:
    """Create DataLoader (shared helper).
    
    Parameters
    ----------
    dataset : DFMDataset or DDFMDataset
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


def create_dfm_dataloader(
    dataset: DFMDataset,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """Create DataLoader for linear DFM.
    
    For linear DFM, we typically use the full sequence as a single batch.
    The DataLoader is mainly for compatibility with Lightning, but since
    DFM doesn't require autograd/backprop, batching is not critical.
    
    Parameters
    ----------
    dataset : DFMDataset
        DFM dataset instance
    batch_size : int, optional
        Batch size. If None, uses full sequence (dataset length = 1).
        For DFM, this is typically 1 (full sequence).
    num_workers : int, default 0
        Number of worker processes for DataLoader
    pin_memory : bool, default False
        Whether to pin memory for faster GPU transfer
        
    Returns
    -------
    DataLoader
        Configured DataLoader for DFM
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


def create_ddfm_dataloader(
    dataset: DDFMDataset,
    batch_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader for Deep DFM (DDFM).
    
    For DDFM, we use standard batching with windowed sequences for
    neural network training. Shuffling is typically enabled for training.
    
    Parameters
    ----------
    dataset : DDFMDataset
        DDFM dataset instance with windowed sequences
    batch_size : int, default 100
        Batch size for training (matches original DDFM)
    shuffle : bool, default True
        Whether to shuffle samples (typically True for training)
    num_workers : int, default 0
        Number of worker processes for DataLoader
    pin_memory : bool, default True
        Whether to pin memory for faster GPU transfer (typically True for GPU training)
        
    Returns
    -------
    DataLoader
        Configured DataLoader for DDFM
    """
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        auto_pin_memory=True
    )


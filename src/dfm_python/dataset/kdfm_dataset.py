"""PyTorch Dataset class for Kernel Dynamic Factor Model (KDFM).

This module provides dataset implementation for KDFM training with full sequences.
"""

import torch
import numpy as np
from typing import Union
from ..logger import get_logger
from .base import DeepFactorModelDataset

_logger = get_logger(__name__)


class KDFMDataset(DeepFactorModelDataset):
    """PyTorch Dataset for full sequence time series data.
    
    This dataset is designed for PyTorch-based models that process the entire
    time series at once (e.g., state-space models trained via gradient descent).
    It returns the full sequence for each sample, suitable for models that require
    the complete time series for training.
    
    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data tensor/array (T x N) where T is time periods and N is number of series
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray]
    ):
        """Initialize KDFM dataset with full sequence.
        
        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Data tensor/array (T x N) where T is time periods and N is number of series
        """
        super().__init__(data)
        
        # For full sequence datasets, we use the full sequence as a single sample
        self.n_samples = 1
    
    def __len__(self) -> int:
        """Return number of samples (always 1 for full sequence)."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the full data sequence.
        
        Parameters
        ----------
        idx : int
            Sample index (ignored, always returns full sequence)
            
        Returns
        -------
        torch.Tensor
            Full data sequence (T x N)
        """
        if idx != 0:
            raise IndexError(f"KDFMDataset only has 1 sample (full sequence), got idx={idx}")
        return self.data


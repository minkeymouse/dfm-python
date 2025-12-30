"""PyTorch Dataset classes for PyTorch-based Dynamic Factor Models.

This module provides Dataset implementations for PyTorch-based DFM models that use
gradient descent training. Two dataset types are provided:
- Windowed sequences: For models that train on overlapping windows (e.g., neural network-based models)
- Full sequences: For models that process the entire time series at once (e.g., state-space models)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Union, Any
from ..logger import get_logger
from ..config.constants import DEFAULT_TORCH_DTYPE

_logger = get_logger(__name__)


class DDFMDataset(Dataset):
    """PyTorch Dataset for windowed time series data.
    
    This dataset handles windowed sequences for PyTorch-based models that require
    batch training on overlapping windows (e.g., neural network-based DFM models).
    It creates overlapping windows from the time series for batch training.
    
    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data tensor/array (T x N) where T is time periods and N is number of series
    window_size : int
        Window size for creating sequences
    stride : int, default 1
        Stride for windowing. Default 1 means overlapping windows.
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        window_size: int,
        stride: int = 1
    ):
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=DEFAULT_TORCH_DTYPE)
        else:
            self.data = data.float() if data.dtype != DEFAULT_TORCH_DTYPE else data
        
        self.T, self.N = self.data.shape
        self.window_size = window_size
        self.stride = stride
        
        if self.window_size > self.T:
            _logger.warning(
                f"window_size ({self.window_size}) > sequence length ({self.T}). "
                f"Using full sequence as single window."
            )
            self.window_size = self.T
        
        # Compute number of samples
        if self.window_size >= self.T:
            self.n_samples = 1
        else:
            self.n_samples = (self.T - self.window_size) // stride + 1
    
    def __len__(self) -> int:
        """Return number of windowed samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a windowed data sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        x : torch.Tensor
            Input data window (window_size x N)
        target : torch.Tensor
            Target data (same as x for autoencoder/reconstruction)
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for {self.n_samples} samples")
        
        if self.window_size >= self.T:
            # Return full sequence
            x = self.data
        else:
            # Return window
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size
            x = self.data[start_idx:end_idx, :]
        
        # For autoencoder/reconstruction tasks, target is same as input
        target = x.clone()
        
        return x, target


class KDFMDataset(Dataset):
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
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=DEFAULT_TORCH_DTYPE)
        else:
            self.data = data.float() if data.dtype != DEFAULT_TORCH_DTYPE else data
        
        self.T, self.N = self.data.shape
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
    
    def get_data(self) -> torch.Tensor:
        """Get the full data tensor.
        
        Returns
        -------
        torch.Tensor
            Full data sequence (T x N)
        """
        return self.data


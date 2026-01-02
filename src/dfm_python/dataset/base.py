"""Base dataset class for Deep Dynamic Factor Models.

This module provides the base dataset class that all factor model datasets inherit from.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Union
from ..logger import get_logger
from ..config.constants import DEFAULT_TORCH_DTYPE

_logger = get_logger(__name__)


class DeepFactorModelDataset(Dataset):
    """Base dataset class for Deep Dynamic Factor Models.
    
    This class provides common functionality for all factor model datasets,
    including data validation and tensor conversion.
    
    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data tensor/array (T x N) where T is time periods and N is number of series
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray]
    ):
        """Initialize base dataset with data validation and conversion.
        
        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Data tensor/array (T x N) where T is time periods and N is number of series
        """
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=DEFAULT_TORCH_DTYPE)
        else:
            self.data = data.float() if data.dtype != DEFAULT_TORCH_DTYPE else data
        
        if self.data.ndim != 2:
            raise ValueError(
                f"Data must be 2D (T x N), got shape {self.data.shape}. "
                f"Expected (time_steps, num_series)."
            )
        
        self.T, self.N = self.data.shape
        
        if self.T == 0 or self.N == 0:
            raise ValueError(
                f"Data must have at least one time step and one series, got shape {self.data.shape}"
            )
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx: int):
        """Get a sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def get_data(self) -> torch.Tensor:
        """Get the full data tensor.
        
        Returns
        -------
        torch.Tensor
            Full data sequence (T x N)
        """
        return self.data


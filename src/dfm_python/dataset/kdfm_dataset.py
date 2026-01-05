"""PyTorch Dataset class for Kernel Dynamic Factor Model (KDFM).

This module provides dataset implementation for KDFM training with full sequences.
Handles data loading, preprocessing, and train/val splitting.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Union, Optional, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..config import DFMConfig

from ..logger import get_logger
from .base import BaseFactorModelDataset
from ..config.constants import DEFAULT_TORCH_DTYPE
from ..config import DFMConfig
from ..dataset.process import TimeIndex
from ..utils.errors import ConfigurationError, DataValidationError
from ..utils.misc import get_config_attr
from ..utils.common import ensure_numpy

_logger = get_logger(__name__)


class KDFMDataset(BaseFactorModelDataset, Dataset):
    """PyTorch Dataset for full sequence time series data.
    
    This dataset is designed for PyTorch-based models that process the entire
    time series at once (e.g., state-space models trained via gradient descent).
    It returns the full sequence for each sample, suitable for models that require
    the complete time series for training.
    
    Can be initialized from raw data (config, data_path, data) or from preprocessed data.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Model configuration object (required if loading from raw data)
    config_path : str or Path, optional
        Path to configuration file
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed before passing.
    target_series : str or List[str], optional
        Target series column names. Can be a single string or list of strings.
    time_index : str, List[str], or TimeIndex, optional
        Time index for the data. Can be TimeIndex object, column name(s), or None.
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). If provided, creates train/val split.
    data_processed : torch.Tensor, optional
        Preprocessed data tensor (T x N). If provided, uses preprocessed data mode.
    """
    
    def __init__(
        self,
        # Raw data mode parameters
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target_series: Optional[Union[str, List[str]]] = None,
        time_index: Optional[Union[str, List[str], TimeIndex]] = None,
        val_split: Optional[float] = None,
        # Preprocessed data mode parameter
        data_processed: Optional[torch.Tensor] = None,
    ):
        # Initialize base class with common attributes
        super().__init__(config=config, config_path=config_path, target_series=target_series)
        
        # Determine initialization mode
        if data_processed is not None:
            # Preprocessed data mode (existing API)
            self.n_samples = 1
            self.data = data_processed
            self.data_processed = data_processed
            self.val_split = None
            self.train_dataset = None
            self.val_dataset = None
        else:
            # Raw data mode
            self.val_split = val_split
            
            # Load and preprocess data (using DFMDataset logic via composition)
            from ..dataset.dfm_dataset import DFMDataset
            self._dfm_dataset = DFMDataset(
                config=config,
                config_path=config_path,
                data_path=data_path,
                data=data,
                target_series=target_series,
                time_index=time_index,
            )
            
            # Get processed data
            data_processed_np = self._dfm_dataset.get_processed_data()
            
            # Convert to torch tensor if needed
            if not isinstance(data_processed_np, torch.Tensor):
                data_processed = torch.tensor(data_processed_np, dtype=DEFAULT_TORCH_DTYPE)
            else:
                data_processed = data_processed_np
            
            # Handle train/val split
            if self.val_split is not None and 0 < self.val_split < 1:
                T = data_processed.shape[0]
                split_idx = int(T * (1 - self.val_split))
                
                train_data = data_processed[:split_idx, :]
                val_data = data_processed[split_idx:, :]
                
                # Create separate datasets for train/val
                self.train_dataset = KDFMDataset(data_processed=train_data, config=config, target_series=target_series)
                self.val_dataset = KDFMDataset(data_processed=val_data, config=config, target_series=target_series)
                
                # Use train dataset as primary
                self.data = train_data
                self.data_processed = train_data
            else:
                # Use all data for training
                self.data = data_processed
                self.data_processed = data_processed
                self.train_dataset = None
                self.val_dataset = None
            
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
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor."""
        return self.data_processed if hasattr(self, 'data_processed') else self.data
    
    def get_initialization_params(self) -> dict:
        """Get initialization parameters for DFM model (delegates to internal DFMDataset).
        
        Returns
        -------
        dict
            Dictionary containing initialization parameters (X, R_mat, q, etc.)
        """
        if hasattr(self, '_dfm_dataset'):
            return self._dfm_dataset.get_initialization_params()
        else:
            # If initialized from preprocessed data, return minimal params
            return {
                'X': ensure_numpy(self.data_processed) if hasattr(self, 'data_processed') else ensure_numpy(self.data),
                'target_scaler': self.target_scaler if hasattr(self, 'target_scaler') else None,
            }

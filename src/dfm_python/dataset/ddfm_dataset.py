"""PyTorch Dataset for Deep Dynamic Factor Model (DDFM).

This module provides dataset implementation for DDFM training.
Handles data loading and scaling.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Any

from ..config.constants import DEFAULT_TORCH_DTYPE
from ..config import DFMConfig
from ..dataset.base import BaseFactorModelDataset
from ..logger import get_logger

_logger = get_logger(__name__)


class DDFMDataset(BaseFactorModelDataset, Dataset):
    """PyTorch Dataset for DDFM training.
    
    Handles data loading and scaling. Data should be preprocessed (imputation, etc.) before passing.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Model configuration object
    data : pd.DataFrame
        Preprocessed data (imputation done, but scaling handled here)
    target_series : str or List[str], optional
        Target series column names
    scaler : sklearn scaler, optional
        sklearn scaler instance (e.g., StandardScaler, RobustScaler). Defaults to StandardScaler().
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[DFMConfig] = None,
        target_series: Optional[Union[str, List[str]]] = None,
        scaler: Optional[Any] = None,
    ):
        # Initialize base class
        super().__init__(config=config, target_series=target_series)
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # Handle scaling
        if scaler is not None:
            self.scaler = scaler
        else:
            try:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            except ImportError:
                raise ImportError(
                    "sklearn is required for data scaling. Install with: pip install scikit-learn"
                )
        
        # Fit and transform using scaler
        self.scaler.fit(data.values)
        data_scaled = self.scaler.transform(data.values)
        self.data_processed = torch.tensor(data_scaled, dtype=DEFAULT_TORCH_DTYPE)
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor."""
        return self.data_processed


class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training with corrupted inputs and clean targets.
    
    This dataset is used during the sequential MC loop in DDFM training.
    It provides efficient batching for autoencoder.fit() calls.
    
    Parameters
    ----------
    x_corrupted : torch.Tensor
        Corrupted input data (T, N_input) - corrupted/noisy input
    y_clean : torch.Tensor
        Clean target data (T, N) - clean target for reconstruction
    mask : torch.Tensor
        Missing data mask (T, N), True where data is missing
    """
    
    def __init__(
        self,
        x_corrupted: torch.Tensor,
        y_clean: torch.Tensor,
        mask: torch.Tensor
    ):
        self.x_corrupted = x_corrupted
        self.y_clean = y_clean
        self.mask = mask
        
        # Verify shapes
        T_x, N_input = x_corrupted.shape
        T_y, N = y_clean.shape
        T_m, N_m = mask.shape
        
        if T_x != T_y or T_x != T_m:
            raise ValueError(
                f"Time dimension mismatch: x_corrupted.shape={x_corrupted.shape}, "
                f"y_clean.shape={y_clean.shape}, mask.shape={mask.shape}"
            )
        if N != N_m:
            raise ValueError(
                f"Feature dimension mismatch: y_clean.shape={y_clean.shape}, mask.shape={mask.shape}"
            )
    
    def __len__(self) -> int:
        """Return number of time steps."""
        return self.x_corrupted.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get data for a single time step.
        
        Parameters
        ----------
        idx : int
            Time step index
            
        Returns
        -------
        x : torch.Tensor
            Corrupted input (N_input,)
        y : torch.Tensor
            Clean target (N,)
        mask : torch.Tensor
            Missing data mask (N,)
        """
        return self.x_corrupted[idx], self.y_clean[idx], self.mask[idx]

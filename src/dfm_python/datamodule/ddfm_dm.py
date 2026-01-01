"""PyTorch Lightning DataModule for DDFM training.

This module provides DDFMDataModule for Deep Dynamic Factor Models.
Uses DDFMDataset with windowed sequences for neural network training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Any, List
from pathlib import Path
import pytorch_lightning as lightning_pl

from .base import BaseDataModule
from ..config.constants import DEFAULT_WINDOW_SIZE, DEFAULT_DDFM_BATCH_SIZE, DEFAULT_TORCH_DTYPE
from ..config import DFMConfig
from ..dataset.dataset import DDFMDataset
from ..dataset.dataloader import create_ddfm_dataloader
from ..dataset.process import TimeIndex, _get_scaler
from ..logger import get_logger
from ..utils.errors import DataValidationError, ConfigurationError

_logger = get_logger(__name__)


class DDFMDataModule(BaseDataModule, lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DDFM training.
    
    This DataModule handles data loading for Deep Dynamic Factor Models.
    Uses DDFMDataset with windowed sequences for neural network training.
    
    **Important**: 
    - Data must be **preprocessed** before passing to this DataModule (imputation, scaling, etc.)
    - DDFM can handle missing data (NaN values) implicitly through state-space model and MCMC
    
    **Target Series Handling**:
    - Target series are passed through as raw data (no preprocessing by this module)
    - Target series scaling: If you want to scale target series, provide a **fitted** sklearn scaler 
      (e.g., StandardScaler, RobustScaler) in config.target_scaler. The scaler must be fitted on 
      target data before passing to the config. Use scaler.inverse_transform() for unstandardization.
    - If no target_scaler provided, target series are assumed to be already in the desired scale
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed (imputation, scaling, etc.)
        before passing to this DataModule. Can contain NaN values - DDFM will handle them.
        If None, data_path must be provided.
    target_series : str or List[str], optional
        Target series column names. Can be a single string or list of strings.
    time_index : str, List[str], or TimeIndex, optional
        Time index for the data. Can be TimeIndex object, column name(s), or None.
    window_size : int, default 100
        Window size for DDFMDataset (number of time steps per window)
    stride : int, default 1
        Stride for windowing in DDFMDataset (1 = overlapping windows)
    batch_size : int, default DEFAULT_DDFM_BATCH_SIZE (100)
        Batch size for DataLoader (matches original DDFM)
    num_workers : int, default 0
        Number of worker processes for DataLoader
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). If None, no validation split.
    
    Examples
    --------
    **Basic usage with preprocessed data**:
    
    >>> from dfm_python import DDFMDataModule
    >>> from sktime.transformations.compose import TransformerPipeline
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sklearn.preprocessing import StandardScaler
    >>> 
    >>> # Preprocess data first
    >>> pipeline = TransformerPipeline([
    ...     ('impute', Imputer(method="ffill")),
    ...     ('scaler', StandardScaler())
    ... ])
    >>> df_preprocessed = pipeline.fit_transform(df_raw)
    >>> 
    >>> # Create DataModule with preprocessed data
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_preprocessed,  # Already preprocessed
    ...     target_series=['market_forward_excess_returns']
    ... )
    >>> dm.setup()
    
    **Using target scaler from config (must be fitted first)**:
    
    >>> from sklearn.preprocessing import RobustScaler
    >>> import pandas as pd
    >>> 
    >>> # Fit scaler on target data first
    >>> target_data = df_preprocessed[['returns']]
    >>> scaler = RobustScaler()
    >>> scaler.fit(target_data)  # Must fit before passing to config
    >>> 
    >>> # Set fitted scaler in config
    >>> config.target_scaler = scaler
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_preprocessed,  # Already preprocessed
    ...     target_series=['returns']
    ... )
    >>> dm.setup()
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target_series: Optional[Union[str, List[str]]] = None,
        time_index: Optional[Union[str, List[str], TimeIndex]] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = 1,
        batch_size: int = DEFAULT_DDFM_BATCH_SIZE,
        num_workers: int = 0,
        val_split: Optional[float] = None,
        **kwargs
    ):
        # Initialize LightningDataModule first (no arguments)
        lightning_pl.LightningDataModule.__init__(self)
        # Initialize BaseDataModule (handles target_series, time_index; target_scaler comes from config)
        BaseDataModule.__init__(
            self,
            config=config,
            config_path=config_path,
            data_path=data_path,
            data=data,
            target_series=target_series,
            time_index=time_index,
            **kwargs
        )
        
        # DDFM-specific parameters
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset: Optional[DDFMDataset] = None
        self.val_dataset: Optional[DDFMDataset] = None
        self.data_processed: Optional[torch.Tensor] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data.
        
        This method handles:
        - Loading data from file or using provided data (data must be preprocessed)
        - Separating target and feature columns
        - Target scaler is stored in self.target_scaler (from base class) for inverse transformation
        """
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise DataValidationError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame.",
                    details="Both data and data_path are None. One must be provided."
                )
            
            # Load data from file using base class method
            X, Time, Z = self.load_data(self.data_path)
            self.data = X
            self.time_index = Time
        
        # Convert to pandas DataFrame if needed
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pd.DataFrame(self.data, columns=pd.Index(series_ids))
        elif isinstance(self.data, pd.DataFrame):
            X_df = self.data.copy()
        else:
            raise DataValidationError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or pandas.DataFrame.",
                details=f"Received type: {type(self.data).__name__}. Expected: numpy.ndarray or pandas.DataFrame."
            )
        
        # Extract time index from column if specified
        if self.time_index is None and self.time_index_column is not None:
            if not isinstance(X_df, pd.DataFrame):
                raise DataValidationError(
                    "time_index_column can only be used with DataFrame input. "
                    "Please provide data as pandas.DataFrame.",
                    details=f"time_index_column is set but data is {type(X_df).__name__}, not DataFrame."
                )
            
            self.time_index = self._extract_time_index_from_dataframe(X_df)
            time_cols = [self.time_index_column] if isinstance(self.time_index_column, str) else self.time_index_column
            X_df = X_df.drop(columns=time_cols)
            _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        
        # Separate target and feature columns
        all_columns = list(X_df.columns)
        target_cols = [col for col in self.target_series if col in all_columns]
        
        # Data is already preprocessed - use as-is
        X_transformed = X_df.copy()
        
        # Target scaler is stored in self.target_scaler (from base class)
        # No need to extract Mx/Wx - use scaler.inverse_transform() directly
        
        # Convert to torch tensor - filter numeric columns
        X_transformed = self._filter_numeric_columns(X_transformed)
        
        X_processed_np = X_transformed.to_numpy()
        self.data_processed = torch.tensor(X_processed_np, dtype=DEFAULT_TORCH_DTYPE)
        
        # Create train/val splits if requested
        if self.val_split is not None and 0 < self.val_split < 1:
            T = self.data_processed.shape[0]
            split_idx = int(T * (1 - self.val_split))
            
            train_data = self.data_processed[:split_idx, :]
            val_data = self.data_processed[split_idx:, :]
            
            # Use DDFMDataset with windowing
            self.train_dataset = DDFMDataset(train_data, window_size=self.window_size, stride=self.stride)
            self.val_dataset = DDFMDataset(val_data, window_size=self.window_size, stride=self.stride)
        else:
            # Use all data for training
            self.train_dataset = DDFMDataset(self.data_processed, window_size=self.window_size, stride=self.stride)
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.train_dataset is None:
            raise ConfigurationError(
                "DataModule train_dataloader failed: setup() must be called before train_dataloader(). "
                "Please call dm.setup() first to load and preprocess data.",
                details="train_dataloader() requires train_dataset attribute which is set by setup()."
            )
        
        return create_ddfm_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create DataLoader for validation."""
        if self.val_dataset is None:
            return None
        
        return create_ddfm_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor."""
        self._check_setup('get_processed_data')
        if self.data_processed is None:
            raise ConfigurationError(
                "DataModule get_processed_data failed: setup() must be called before get_processed_data(). "
                "Please call dm.setup() first to load and preprocess data.",
                details="get_processed_data() requires data_processed attribute which is set by setup()."
            )
        return self.data_processed


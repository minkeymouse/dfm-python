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

from ..config import DFMConfig
from ..data.utils import load_data as _load_data
from ..data.dataset import DDFMDataset
from ..data.dataloader import create_ddfm_dataloader
from ..utils.time import TimeIndex
from ..logger import get_logger
from .utils import (
    _get_scaler,
    _get_mean,
    _get_scale,
)

_logger = get_logger(__name__)


class DDFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DDFM training.
    
    This DataModule handles data loading for Deep Dynamic Factor Models.
    Uses DDFMDataset with windowed sequences for neural network training.
    
    **Important**: 
    - Data must be **preprocessed** before passing to this DataModule (imputation, scaling, etc.)
    - DDFM can handle missing data (NaN values) implicitly through state-space model and MCMC
    - Only target series Mx, Wx statistics are computed (for inverse transformation in prediction)
    - Feature Mx, Wx are not needed since predictions only return target series
    
    **Target Series Handling**:
    - Target series are passed through as raw data (no preprocessing by this module)
    - Optional `target_scaler` can be used to scale targets separately if needed
    - Target Mx, Wx are computed for inverse transformation during prediction
    
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
        Target series column names. These will be used for computing Mx, Wx statistics
        for inverse transformation during prediction.
        
        Example:
            target_series=['market_forward_excess_returns']  # Single target
            target_series=['target1', 'target2']  # Multiple targets
    target_scaler : str or Any, optional
        Optional scaler for target series. Can be:
        - `None` (default): Targets are not scaled (Mx=0, Wx=1)
        - `'standard'`: StandardScaler for targets
        - `'robust'`: RobustScaler for targets (more robust to outliers)
        - Scaler instance: Custom scaler object (must be fitted)
        
        **Note**: Target scaler is used to compute Mx, Wx statistics for inverse transformation.
        Targets themselves remain in raw form in the data.
    time_index : TimeIndex, optional
        Time index for the data. If None and time_index_column is provided,
        time index will be extracted from the data.
    time_index_column : str or list of str, optional
        Column name(s) in DataFrame to use as time index. If provided:
        - The column(s) will be extracted from the DataFrame
        - TimeIndex will be created from the column(s)
        - The column(s) will be excluded from the data (not used as features)
        - If multiple columns are provided, they will be combined
    window_size : int, default 100
        Window size for DDFMDataset (number of time steps per window)
    stride : int, default 1
        Stride for windowing in DDFMDataset (1 = overlapping windows)
    batch_size : int, default 100
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
    
    **Using target scaler**:
    
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_preprocessed,  # Already preprocessed
    ...     target_series=['returns'],
    ...     target_scaler='robust'  # RobustScaler for targets
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
        target_scaler: Optional[Union[str, Any]] = None,
        time_index: Optional[TimeIndex] = None,
        time: Optional[TimeIndex] = None,  # Legacy parameter name (alias for time_index)
        time_index_column: Optional[Union[str, List[str]]] = None,
        window_size: int = 100,
        stride: int = 1,
        batch_size: int = 100,
        num_workers: int = 0,
        val_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        
        # Load config if config_path provided
        if config is None and config_path is not None:
            from ..config import YamlSource
            source = YamlSource(config_path)
            config = source.load()
        
        if config is None:
            raise ValueError(
                "DataModule initialization failed: either config or config_path must be provided. "
                "Please provide a DFMConfig object or a path to a configuration file."
            )
        
        self.config = config
        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data
        # Support both time_index and time (legacy) parameter names
        self.time_index = time_index if time_index is not None else time
        self.time_index_column = time_index_column
        
        # Target series handling
        if target_series is None:
            self.target_series = []
        elif isinstance(target_series, str):
            self.target_series = [target_series]
        else:
            self.target_series = list(target_series)
        
        # Handle target_scaler: can be string ('standard', 'robust') or scaler instance
        if target_scaler is None:
            self.target_scaler = None
            self.target_scaler_type = None
        elif isinstance(target_scaler, str):
            # String: 'standard' or 'robust'
            if target_scaler.lower() not in ['standard', 'robust']:
                raise ValueError(
                    f"target_scaler must be 'standard', 'robust', or a scaler instance. "
                    f"Got: {target_scaler}"
                )
            self.target_scaler_type = target_scaler.lower()
            self.target_scaler = None  # Will be created in setup()
        else:
            # Scaler instance
            self.target_scaler = target_scaler
            self.target_scaler_type = None
        
        # DDFM-specific parameters
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset: Optional[DDFMDataset] = None
        self.val_dataset: Optional[DDFMDataset] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        self.data_raw: Optional[pd.DataFrame] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data.
        
        This method handles:
        - Loading data from file or using provided data (data must be preprocessed)
        - Separating target and feature columns
        - Computing target Mx, Wx statistics for inverse transformation
        - Keeping targets in raw form (to avoid inverse transform issues)
        """
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise ValueError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame."
                )
            
            # Load data from file
            X, Time, Z = _load_data(
                self.data_path,
                self.config,
            )
            self.data = X
            self.time_index = Time
        
        # Convert to pandas DataFrame if needed
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pd.DataFrame(self.data, columns=pd.Index(series_ids))
        elif isinstance(self.data, pd.DataFrame):
            X_df = self.data.copy()
        else:
            raise TypeError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or pandas.DataFrame."
            )
        
        # Extract time index from column if specified
        if self.time_index is None and self.time_index_column is not None:
            if not isinstance(X_df, pd.DataFrame):
                raise ValueError(
                    "time_index_column can only be used with DataFrame input. "
                    "Please provide data as pandas.DataFrame."
                )
            
            # Handle single string or list of strings
            time_cols = [self.time_index_column] if isinstance(self.time_index_column, str) else self.time_index_column
            
            # Check if columns exist
            missing_cols = [col for col in time_cols if col not in X_df.columns]
            if missing_cols:
                raise ValueError(
                    f"time_index_column(s) {missing_cols} not found in DataFrame. "
                    f"Available columns: {list(X_df.columns)}"
                )
            
            # Extract time index column(s)
            time_data = X_df[time_cols]
            
            # Create TimeIndex from the column(s)
            from ..utils.time import parse_timestamp
            if len(time_cols) == 1:
                # Single column: convert to list of timestamps
                time_list = [parse_timestamp(str(val)) for val in time_data.iloc[:, 0]]
            else:
                # Multiple columns: combine them (e.g., year, month, day)
                # For now, convert to string and parse
                time_list = [parse_timestamp(' '.join(str(val) for val in row)) for row in time_data.values]
            
            self.time_index = TimeIndex(time_list)
            
            # Remove time index column(s) from data
            X_df = X_df.drop(columns=time_cols)
            _logger.info(f"Extracted time index from column(s): {time_cols}, removed from data")
        
        # Store raw data
        self.data_raw = X_df.copy()
        
        # Separate target and feature columns
        all_columns = list(X_df.columns)
        target_cols = [col for col in self.target_series if col in all_columns]
        
        # Data is already preprocessed - use as-is
        X_transformed = X_df.copy()
        
        # Compute target Mx, Wx only (for inverse transformation in prediction)
        # Feature Mx, Wx are not needed since we only return target predictions
        if target_cols:
            if self.target_scaler_type is not None:
                # Create scaler from string type
                from sklearn.preprocessing import StandardScaler, RobustScaler
                if self.target_scaler_type == 'standard':
                    target_scaler_instance = StandardScaler()
                else:  # 'robust'
                    target_scaler_instance = RobustScaler()
                # Fit on raw target data
                target_scaler_instance.fit(X_df[target_cols])
                target_values = np.asarray(X_df[target_cols].values)
                self.Mx = _get_mean(target_scaler_instance, target_values)
                self.Wx = _get_scale(target_scaler_instance, target_values)
                self.target_scaler = target_scaler_instance  # Store for later use
            elif self.target_scaler is not None:
                # Scaler instance provided - should already be fitted
                # Just extract statistics (no fit() call needed)
                target_values = np.asarray(X_df[target_cols].values)
                self.Mx = _get_mean(self.target_scaler, target_values)
                self.Wx = _get_scale(self.target_scaler, target_values)
            else:
                # No target scaler: targets not preprocessed (Mx=0, Wx=1)
                # Targets remain in raw form
                self.Mx = np.zeros(len(target_cols))
                self.Wx = np.ones(len(target_cols))
        else:
            self.Mx = np.array([])
            self.Wx = np.array([])
        
        # Convert to torch tensor
        # Select only numeric columns (exclude datetime and other object types)
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X_transformed.columns):
            excluded_cols = [col for col in X_transformed.columns if col not in numeric_cols]
            _logger.warning(
                f"Excluding non-numeric columns from data: {excluded_cols}. "
                f"These columns will not be used in model training."
            )
            X_transformed = X_transformed[numeric_cols]
        
        X_processed_np = X_transformed.to_numpy()
        self.data_processed = torch.tensor(X_processed_np, dtype=torch.float32)
        
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
            raise RuntimeError(
                "DataModule train_dataloader failed: setup() must be called before train_dataloader(). "
                "Please call dm.setup() first to load and preprocess data."
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
    
    def get_std_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get standardization parameters (Mx, Wx) if available."""
        if self.data_processed is None:
            raise RuntimeError(
                "DataModule get_std_params failed: setup() must be called before get_std_params(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.Mx, self.Wx
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor."""
        if self.data_processed is None:
            raise RuntimeError(
                "DataModule get_processed_data failed: setup() must be called before get_processed_data(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.data_processed
    
    def get_raw_data(self) -> pd.DataFrame:
        """Get raw data DataFrame (before preprocessing)."""
        if self.data_raw is None:
            raise RuntimeError(
                "DataModule get_raw_data failed: setup() must be called before get_raw_data(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.data_raw
    
    def get_target_indices(self) -> List[int]:
        """Get indices of target series columns."""
        if self.data_raw is None:
            return []
        
        all_columns = list(self.data_raw.columns)
        return [all_columns.index(col) for col in self.target_series if col in all_columns]


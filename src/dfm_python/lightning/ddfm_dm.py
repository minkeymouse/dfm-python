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
    _check_sktime,
    _get_scaler,
    _get_mean,
    _get_scale,
    _compute_mx_wx,
    create_passthrough_transformer,
    _is_pipeline_fitted,
    _is_scaler_fitted,
)

_logger = get_logger(__name__)


class DDFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DDFM training.
    
    This DataModule handles data loading for Deep Dynamic Factor Models.
    Uses DDFMDataset with windowed sequences for neural network training.
    
    **Important**: DDFM can handle missing data (NaN values) implicitly:
    - **DDFM**: Uses state-space model and MCMC procedure to handle missing data through
      idiosyncratic component estimation
    
    **Target Series Handling**:
    - Target series are **NOT preprocessed** by the feature pipeline to avoid inverse transform issues
    - Target series are passed through as raw data (no imputation, transformation, or scaling)
    - This is critical for dynamic factor models where targets are endogenous variables
    - Only feature columns (non-target) are preprocessed by the pipeline
    - Optional `target_scaler` can be used to scale targets separately if needed
    
    **Usage Pattern**:
    - Data can contain NaN values - models will handle them implicitly
    - Supports target series that are not preprocessed (to avoid inverse transform issues)
    - If `pipeline=None`, a passthrough transformer is used by default (no-op)
    - Users can optionally provide their preprocessing pipeline to extract statistics (Mx/Wx)
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    pipeline : Any, optional
        sktime-compatible preprocessing pipeline (e.g., TransformerPipeline) used to extract statistics.
        
        **If `preprocessed=True`**: Pipeline is assumed to be already fitted (from preprocessing step).
        Only used for statistics extraction (no fit/transform calls).
        
        **If `preprocessed=False`**: Pipeline is used to preprocess feature columns only.
        Target columns are excluded from preprocessing.
        
        **Note**: Target series are never preprocessed by this pipeline.
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Data array or DataFrame. Can contain NaN values - DDFM will handle them.
        If None, data_path must be provided.
    target_series : str or List[str], optional
        Target series column names. These will **NOT be preprocessed** by the feature pipeline.
        
        **Important**: Target series are passed through as raw data to avoid inverse transform issues.
        This is critical for dynamic factor models where targets are endogenous variables.
        Only feature columns (non-target) will be preprocessed by the pipeline.
        
        Example:
            target_series=['market_forward_excess_returns']  # Single target
            target_series=['target1', 'target2']  # Multiple targets
    target_scaler : str or Any, optional
        Optional scaler for target series. Can be:
        - `None` (default): Targets are not scaled (raw data passed through)
        - `'standard'`: StandardScaler for targets
        - `'robust'`: RobustScaler for targets (more robust to outliers)
        - Scaler instance: Custom scaler object (must be fitted if `preprocessed=True`)
        
        **Note**: Even if `target_scaler` is provided, targets are still excluded from
        the feature preprocessing pipeline. This allows separate scaling of targets
        while keeping them in original scale for inverse transforms.
        
        **Typical usage**: `None` (targets should not be preprocessed to avoid inverse issues)
    preprocessed : bool, default False
        Whether data is already preprocessed.
        
        **If `True`**:
        - Data is assumed to be already preprocessed (features are scaled/transformed)
        - Pipeline is assumed to be already fitted (from preprocessing step)
        - Pipeline is only used for statistics extraction (no fit/transform calls)
        - Target series remain in raw form (not preprocessed)
        
        **If `False`**:
        - Pipeline will be used to preprocess feature columns (fit_transform)
        - Target columns are excluded from preprocessing
        - Target series remain in raw form
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
    **Basic usage with target series**:
    
    >>> from dfm_python import DDFMDataModule
    >>> 
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_raw,
    ...     pipeline=preprocess_pipeline,
    ...     target_series=['market_forward_excess_returns'],
    ...     preprocessed=False
    ... )
    >>> dm.setup()
    >>> # Features are preprocessed, targets remain raw
    
    **Using preprocessed data with target series**:
    
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_preprocessed,  # Already preprocessed
    ...     pipeline=preprocess_pipeline,  # Already fitted
    ...     target_series=['target1', 'target2'],
    ...     preprocessed=True  # Data already preprocessed
    ... )
    >>> dm.setup()
    >>> # Features use preprocessed data, targets remain raw
    
    **Using target scaler**:
    
    >>> dm = DDFMDataModule(
    ...     config=config,
    ...     data=df_raw,
    ...     pipeline=preprocess_pipeline,
    ...     target_series=['returns'],
    ...     target_scaler='robust',  # RobustScaler for targets
    ...     preprocessed=False
    ... )
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        pipeline: Optional[Any] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        target_series: Optional[Union[str, List[str]]] = None,
        target_scaler: Optional[Union[str, Any]] = None,
        preprocessed: bool = False,
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
        _check_sktime()
        
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
        self.pipeline = pipeline
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
        
        self.preprocessed = preprocessed
        
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
        self.target_Mx: Optional[np.ndarray] = None
        self.target_Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        self.data_raw: Optional[pd.DataFrame] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data.
        
        This method handles:
        - Loading data from file or using provided data
        - Separating target and feature columns
        - Preprocessing features (if not preprocessed) or extracting statistics (if preprocessed)
        - Keeping targets unprocessed (to avoid inverse transform issues)
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
        feature_cols = [col for col in all_columns if col not in target_cols]
        
        if self.preprocessed:
            # Already preprocessed: use data as-is, extract statistics only
            # Pipeline is assumed to be already fitted (passed from preprocessing step)
            X_transformed = X_df.copy()
            
            # Pipeline is for statistics extraction only (already fitted, no fit/transform)
            if self.pipeline is not None and feature_cols:
                # Pipeline should already be fitted - just extract statistics
                # No fit() call needed - pipeline was fitted during preprocessing
                scaler = _get_scaler(self.pipeline)
                if scaler is not None:
                    feature_values = np.asarray(X_df[feature_cols].values)
                    self.Mx = _get_mean(scaler, feature_values)
                    self.Wx = _get_scale(scaler, feature_values)
                else:
                    # Fallback: compute from data
                    feature_values = np.asarray(X_df[feature_cols].values)
                    self.Mx, self.Wx = _compute_mx_wx(feature_values)
            elif feature_cols:
                # No pipeline: compute from data
                feature_values = np.asarray(X_df[feature_cols].values)
                self.Mx, self.Wx = _compute_mx_wx(feature_values)
            else:
                # No features: set to empty
                self.Mx = np.array([])
                self.Wx = np.array([])
            
            # Target scaler (optional, typically None)
            # Targets are NOT preprocessed by feature pipeline - they remain raw
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
                    self.target_Mx = _get_mean(target_scaler_instance, target_values)
                    self.target_Wx = _get_scale(target_scaler_instance, target_values)
                    self.target_scaler = target_scaler_instance  # Store for later use
                elif self.target_scaler is not None:
                    # Scaler instance provided - should already be fitted
                    # Just extract statistics (no fit() call needed)
                    target_values = np.asarray(X_df[target_cols].values)
                    self.target_Mx = _get_mean(self.target_scaler, target_values)
                    self.target_Wx = _get_scale(self.target_scaler, target_values)
                else:
                    # No target scaler: targets not preprocessed (Mx=0, Wx=1)
                    # Targets remain in raw form
                    self.target_Mx = np.zeros(len(target_cols))
                    self.target_Wx = np.ones(len(target_cols))
            else:
                self.target_Mx = np.array([])
                self.target_Wx = np.array([])
        else:
            # Not preprocessed: preprocess features, keep targets as-is
            if self.pipeline is not None and feature_cols:
                # Preprocess features only
                X_features_transformed = self.pipeline.fit_transform(X_df[feature_cols])
                
                # Extract feature Mx, Wx
                scaler = _get_scaler(self.pipeline)
                if scaler is not None:
                    feature_values = np.asarray(X_features_transformed.values)
                    self.Mx = _get_mean(scaler, feature_values)
                    self.Wx = _get_scale(scaler, feature_values)
                else:
                    feature_values = np.asarray(X_features_transformed.values)
                    self.Mx, self.Wx = _compute_mx_wx(feature_values)
            elif feature_cols:
                # No pipeline: compute from data
                feature_values = np.asarray(X_df[feature_cols].values)
                self.Mx, self.Wx = _compute_mx_wx(feature_values)
                X_features_transformed = X_df[feature_cols]
            else:
                # No features
                X_features_transformed = pd.DataFrame()
                self.Mx = np.array([])
                self.Wx = np.array([])
            
            # Targets: keep as-is (no preprocessing by feature pipeline)
            # Targets are NOT preprocessed - they remain in raw form
            if target_cols:
                X_target = X_df[target_cols].copy()  # Raw target data
            else:
                X_target = pd.DataFrame()
            
            # Target scaler (optional, typically None)
            # Even if target_scaler is provided, targets are still excluded from feature pipeline
            if target_cols:
                if self.target_scaler_type is not None:
                    # Create scaler from string type
                    from sklearn.preprocessing import StandardScaler, RobustScaler
                    if self.target_scaler_type == 'standard':
                        target_scaler_instance = StandardScaler()
                    else:  # 'robust'
                        target_scaler_instance = RobustScaler()
                    # Fit on raw target data
                    target_scaler_instance.fit(X_target)
                    target_values = np.asarray(X_target.values)
                    self.target_Mx = _get_mean(target_scaler_instance, target_values)
                    self.target_Wx = _get_scale(target_scaler_instance, target_values)
                    self.target_scaler = target_scaler_instance  # Store for later use
                elif self.target_scaler is not None:
                    # Scaler instance provided - fit on raw target data
                    self.target_scaler.fit(X_target)
                    target_values = np.asarray(X_target.values)
                    self.target_Mx = _get_mean(self.target_scaler, target_values)
                    self.target_Wx = _get_scale(self.target_scaler, target_values)
                else:
                    # No target scaler: targets not preprocessed (Mx=0, Wx=1)
                    # Targets remain in raw form
                    self.target_Mx = np.zeros(len(target_cols))
                    self.target_Wx = np.ones(len(target_cols))
            else:
                self.target_Mx = np.array([])
                self.target_Wx = np.array([])
            
            # Combine features and targets
            if target_cols:
                X_transformed = pd.concat([X_features_transformed, X_target], axis=1)
                # Preserve original column order
                X_transformed = X_transformed[all_columns]
            else:
                X_transformed = X_features_transformed
        
        # Combine Mx, Wx in column order (feature + target)
        if target_cols and self.Mx is not None and self.Wx is not None and self.target_Mx is not None and self.target_Wx is not None:
            full_Mx = np.zeros(len(all_columns))
            full_Wx = np.ones(len(all_columns))
            
            for i, col in enumerate(all_columns):
                if col in feature_cols:
                    feat_idx = feature_cols.index(col)
                    full_Mx[i] = self.Mx[feat_idx]
                    full_Wx[i] = self.Wx[feat_idx]
                elif col in target_cols:
                    tgt_idx = target_cols.index(col)
                    full_Mx[i] = self.target_Mx[tgt_idx]
                    full_Wx[i] = self.target_Wx[tgt_idx]
            
            self.Mx = full_Mx
            self.Wx = full_Wx
        
        # Convert to torch tensor
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
    
    def get_pipeline(self) -> Any:
        """Get the preprocessing pipeline used for statistics extraction."""
        return self.pipeline
    
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
    
    def is_data_preprocessed(self) -> bool:
        """Check if data is already preprocessed."""
        return self.preprocessed
    
    def get_target_indices(self) -> List[int]:
        """Get indices of target series columns."""
        if self.data_raw is None:
            return []
        
        all_columns = list(self.data_raw.columns)
        return [all_columns.index(col) for col in self.target_series if col in all_columns]


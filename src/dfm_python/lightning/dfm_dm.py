"""PyTorch Lightning DataModule for DFM training.

This module provides DFMDataModule for linear DFM models.
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
from ..data.dataset import DFMDataset
from ..data.dataloader import create_dfm_dataloader
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
)

_logger = get_logger(__name__)


class DFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DFM training.
    
    This DataModule handles data loading for linear DFM models.
    Uses DFMDataset which returns full sequences (no windowing).
    
    **Important**: DFM can handle missing data (NaN values) implicitly:
    - **DFM**: Uses Kalman filter's `handle_missing_data()` method to skip NaN observations
    
    **Usage Pattern**:
    - Data can contain NaN values - models will handle them implicitly
    - If `pipeline=None`, a passthrough transformer is used by default (no-op)
    - Users can optionally provide their preprocessing pipeline to extract statistics (Mx/Wx)
    - For better performance, users can preprocess data (imputation, scaling) before passing,
      but it's not required - models will handle missing data automatically
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    pipeline : Any, optional
        sktime-compatible preprocessing pipeline (e.g., TransformerPipeline) used to extract statistics.
        
        **Purpose**: The pipeline is used to extract statistics (e.g., Mx/Wx from StandardScaler)
        needed for forecasting and nowcasting operations. It is NOT used to preprocess data - data
        must be preprocessed by the user before passing to this DataModule.
        
        **If None**: Uses passthrough transformer (no statistics extracted). Mx/Wx will be computed
        from the data as fallback. This is the default.
        
        **If provided**: The pipeline will be fitted on the data to extract statistics (e.g., 
        standardization parameters from StandardScaler). These statistics are used for transforming
        predictions back to original scale during forecasting/nowcasting.
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Data array or DataFrame. Can contain NaN values - DFM will handle them:
        - DFM: Uses Kalman filter to implicitly handle missing data
        - Standardized/scaled data (mean=0, std=1) is recommended for better performance
        - Feature-engineered if needed
        If None, data_path must be provided.
    preprocessed : bool, default False
        Whether data is already preprocessed.
        
        **If `True`**:
        - Data is assumed to be already preprocessed (scaled/transformed)
        - Pipeline is assumed to be already fitted (from preprocessing step)
        - Pipeline is only used for statistics extraction (no fit/transform calls)
        
        **If `False`**:
        - Pipeline will be used to preprocess data (fit_transform)
    time_index : TimeIndex, optional
        Time index for the data. If None and time_index_column is provided,
        time index will be extracted from the data.
    time_index_column : str or list of str, optional
        Column name(s) in DataFrame to use as time index. If provided:
        - The column(s) will be extracted from the DataFrame
        - TimeIndex will be created from the column(s)
        - The column(s) will be excluded from the data (not used as features)
        - If multiple columns are provided, they will be combined
    batch_size : int, optional
        Batch size for DataLoader. For DFM, typically 1 (full sequence).
    num_workers : int, default 0
        Number of worker processes for DataLoader
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). If None, no validation split.
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        pipeline: Optional[Any] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        preprocessed: bool = False,
        time_index: Optional[TimeIndex] = None,
        time: Optional[TimeIndex] = None,  # Legacy parameter name (alias for time_index)
        time_index_column: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = None,
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
        self.preprocessed = preprocessed
        # Support both time_index and time (legacy) parameter names
        self.time_index = time_index if time_index is not None else time
        self.time_index_column = time_index_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset: Optional[DFMDataset] = None
        self.val_dataset: Optional[DFMDataset] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data.
        
        This method is called by Lightning to set up the data module.
        It loads preprocessed data and extracts statistics from the pipeline for forecasting/nowcasting.
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
        
        # Extract time index from column if specified (must be done before preprocessing)
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
        
        # Determine pipeline to use
        if self.pipeline is None:
            pipeline_to_use = create_passthrough_transformer()
        else:
            pipeline_to_use = self.pipeline
        
        # Set pandas output for sktime pipelines
        try:
            if hasattr(pipeline_to_use, 'set_output'):
                pipeline_to_use.set_output(transform="pandas")
        except (AttributeError, ValueError) as e:
            _logger.debug(f"Could not set pandas output on pipeline: {e}")
        
        # Apply pipeline based on preprocessed flag
        if self.preprocessed:
            # Already preprocessed: use data as-is, extract statistics only
            # Pipeline should already be fitted - just extract statistics
            X_transformed = X_df.copy()
            
            # Pipeline is for statistics extraction only (already fitted, no fit/transform)
            if pipeline_to_use is not None:
                # Try to extract statistics from fitted pipeline
                try:
                    scaler = _get_scaler(pipeline_to_use)
                    if scaler is not None:
                        X_processed_np = X_transformed.to_numpy()
                        self.Mx = _get_mean(scaler, X_processed_np)
                        self.Wx = _get_scale(scaler, X_processed_np)
                except (AttributeError, ImportError, Exception) as e:
                    _logger.debug(f"Could not extract scaler from pipeline: {e}")
        else:
            # Not preprocessed: preprocess data using pipeline
            try:
                X_transformed = pipeline_to_use.fit_transform(X_df)
            except Exception as e:
                raise ValueError(
                    f"DataModule setup failed: pipeline fit_transform error: {e}. "
                    f"Ensure pipeline is sktime-compatible (e.g., TransformerPipeline with StandardScaler) "
                    f"and supports pandas DataFrames."
                ) from e
        
        # Ensure output is pandas DataFrame
        if not isinstance(X_transformed, pd.DataFrame):
            if isinstance(X_transformed, np.ndarray):
                n_cols = X_transformed.shape[1] if len(X_transformed.shape) > 1 else 1
                if n_cols == len(X_df.columns):
                    X_transformed = pd.DataFrame(X_transformed, columns=pd.Index(X_df.columns))
                else:
                    X_transformed = pd.DataFrame(X_transformed, 
                        columns=pd.Index([f'feature_{i}' for i in range(n_cols)]))
                # Preserve index
                if len(X_transformed) == len(X_df):
                    X_transformed.index = X_df.index
            else:
                raise TypeError(
                    f"DataModule setup failed: pipeline returned unsupported type {type(X_transformed)}. "
                    f"Expected pandas.DataFrame or numpy.ndarray."
                )
        
        # Convert transformed data to numpy
        # Ensure all columns are numeric (exclude any remaining non-numeric columns like date)
        numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X_transformed.columns):
            non_numeric = [col for col in X_transformed.columns if col not in numeric_cols]
            _logger.warning(f"Excluding non-numeric columns from data: {non_numeric}")
            X_transformed = X_transformed[numeric_cols]
        
        X_processed_np = X_transformed.to_numpy()
        
        # Extract standardization parameters if not already extracted
        if not self.preprocessed:
            # Try to extract standardization parameters if pipeline includes a scaler
            try:
                scaler = _get_scaler(pipeline_to_use)
                if scaler is not None:
                    self.Mx = _get_mean(scaler, X_processed_np)
                    self.Wx = _get_scale(scaler, X_processed_np)
            except (AttributeError, ImportError, Exception) as e:
                _logger.debug(f"Could not extract scaler from pipeline: {e}")
                pass
        
        # Convert to torch tensor
        self.data_processed = torch.tensor(X_processed_np, dtype=torch.float32)
        
        # If Mx and Wx are still None, compute from processed data as fallback
        if self.Mx is None or self.Wx is None:
            mx_fallback, wx_fallback = _compute_mx_wx(X_processed_np)
            if self.Mx is None:
                self.Mx = mx_fallback
            if self.Wx is None:
                self.Wx = wx_fallback
        
        # Create train/val splits if requested
        if self.val_split is not None and 0 < self.val_split < 1:
            T = self.data_processed.shape[0]
            split_idx = int(T * (1 - self.val_split))
            
            train_data = self.data_processed[:split_idx, :]
            val_data = self.data_processed[split_idx:, :]
            
            # For linear DFM, use full sequences (no windowing)
            self.train_dataset = DFMDataset(train_data)
            self.val_dataset = DFMDataset(val_data)
        else:
            # Use all data for training
            self.train_dataset = DFMDataset(self.data_processed)
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule train_dataloader failed: setup() must be called before train_dataloader(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        
        return create_dfm_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create DataLoader for validation."""
        if self.val_dataset is None:
            return None
        
        return create_dfm_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
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


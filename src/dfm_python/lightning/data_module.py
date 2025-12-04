"""PyTorch Lightning DataModule for DFM training.

This module provides LightningDataModule implementations for loading and
preprocessing data for Dynamic Factor Model training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import polars as pl
from typing import Optional, Union, Tuple, Any
from pathlib import Path
import pytorch_lightning as lightning_pl

from ..config import DFMConfig
from ..data.utils import load_data as _load_data
from ..data.dataset import DFMDataset, DDFMDataset
from ..data.dataloader import create_dfm_dataloader, create_ddfm_dataloader
from ..utils.time import TimeIndex
from ..logger import get_logger

_logger = get_logger(__name__)


def _check_sktime_available():
    """Check if sktime is available and raise ImportError if not."""
    try:
        import sktime
        return True
    except ImportError:
        raise ImportError(
            "DataModule initialization failed: sktime is required for sktime transformers. "
            "Install it with: pip install sktime"
        )


def _extract_scaler_from_transformer(transformer: Any) -> Optional[Any]:
    """Extract StandardScaler from transformer, handling wrappers.
    
    This function recursively searches through transformer wrappers and pipelines
    to find a StandardScaler instance.
    
    Parameters
    ----------
    transformer : Any
        Transformer to search (can be StandardScaler, TabularToSeriesAdaptor, Pipeline, etc.)
        
    Returns
    -------
    Optional[StandardScaler]
        StandardScaler instance if found, None otherwise
    """
    from sklearn.preprocessing import StandardScaler
    
    # Check if transformer is TabularToSeriesAdaptor (sktime wrapper)
    if hasattr(transformer, 'transformer'):
        # TabularToSeriesAdaptor wraps the underlying transformer
        return _extract_scaler_from_transformer(transformer.transformer)
    # Check if transformer is a pipeline with StandardScaler
    elif hasattr(transformer, 'steps'):
        for name, step in transformer.steps:
            scaler = _extract_scaler_from_transformer(step)
            if scaler is not None:
                return scaler
    # Check if transformer is ColumnTransformer
    elif hasattr(transformer, 'transformers'):
        for name, trans, cols in transformer.transformers:
            scaler = _extract_scaler_from_transformer(trans)
            if scaler is not None:
                return scaler
    # Check if transformer is StandardScaler itself
    elif isinstance(transformer, StandardScaler):
        return transformer
    return None


def _extract_mx_from_scaler(scaler: Any, data: np.ndarray) -> Optional[np.ndarray]:
    """Extract mean (Mx) from StandardScaler with fallbacks.
    
    Parameters
    ----------
    scaler : StandardScaler
        StandardScaler instance
    data : np.ndarray
        Processed data array (T x N) for fallback computation
        
    Returns
    -------
    Optional[np.ndarray]
        Mean values (N,) if extracted, None if fallback needed
    """
    # Check if StandardScaler has with_mean attribute
    if hasattr(scaler, 'with_mean') and hasattr(scaler, 'with_std'):
        if scaler.with_mean:
            # If with_mean=True, try to get mean_ from scaler
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                mean_val = scaler.mean_
                if not isinstance(mean_val, np.ndarray):
                    mean_val = np.asarray(mean_val)
                return mean_val
            else:
                # If mean_ not available, return None to trigger fallback
                return None
        else:
            # If with_mean=False, mean is zero
            return np.zeros(data.shape[1])
    elif hasattr(scaler, 'mean_'):
        # Fallback: Try to get mean_ from StandardScaler
        # (for older sklearn versions or custom scalers)
        try:
            mean_val = scaler.mean_
            if mean_val is not None:
                if not isinstance(mean_val, np.ndarray):
                    mean_val = np.asarray(mean_val)
                return mean_val
        except (AttributeError, TypeError):
            pass
    return None


def _extract_wx_from_scaler(scaler: Any, data: np.ndarray) -> Optional[np.ndarray]:
    """Extract scale (Wx) from StandardScaler with fallbacks.
    
    Parameters
    ----------
    scaler : StandardScaler
        StandardScaler instance
    data : np.ndarray
        Processed data array (T x N) for fallback computation
        
    Returns
    -------
    Optional[np.ndarray]
        Scale values (N,) if extracted, None if fallback needed
    """
    # Check if StandardScaler has with_std attribute
    if hasattr(scaler, 'with_mean') and hasattr(scaler, 'with_std'):
        if scaler.with_std:
            # If with_std=True, try to get scale_ from scaler
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                scale_val = scaler.scale_
                if not isinstance(scale_val, np.ndarray):
                    scale_val = np.asarray(scale_val)
                return _normalize_wx(scale_val)
            else:
                # If scale_ not available, return None to trigger fallback
                return None
        else:
            # If with_std=False, scale is one
            return np.ones(data.shape[1])
    elif hasattr(scaler, 'scale_'):
        # Fallback: Try to get scale_ from StandardScaler
        # (for older sklearn versions or custom scalers)
        try:
            scale_val = scaler.scale_
            if scale_val is not None:
                if not isinstance(scale_val, np.ndarray):
                    scale_val = np.asarray(scale_val)
                return _normalize_wx(scale_val)
        except (AttributeError, TypeError):
            pass
    return None


def _normalize_wx(wx: np.ndarray) -> np.ndarray:
    """Normalize Wx to avoid division by zero.
    
    Parameters
    ----------
    wx : np.ndarray
        Scale values (N,)
        
    Returns
    -------
    np.ndarray
        Normalized scale values with zeros replaced by 1.0
    """
    return np.where(wx == 0, 1.0, wx)


def _compute_mx_wx_from_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Mx and Wx from data as fallback.
    
    Parameters
    ----------
    data : np.ndarray
        Processed data array (T x N)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Mx, Wx) tuple where Mx is mean and Wx is normalized std
    """
    mx = np.mean(data, axis=0)
    wx = np.std(data, axis=0)
    wx = _normalize_wx(wx)
    return mx, wx


class DFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DFM training.
    
    This DataModule handles data loading and preprocessing for DFM/DDFM models.
    It supports two usage patterns:
    
    1. **Preprocessed Data**: Provide already-preprocessed data (standardized, no missing values).
       Use a passthrough transformer to avoid double standardization.
    
    2. **Raw Data with TransformerPipeline**: Provide raw data and a sktime TransformerPipeline
       for preprocessing. The pipeline will be applied in setup().
    
    The preprocessing pipeline (if provided) should handle:
    - Missing value imputation (forward fill, backward fill, etc.)
    - Outlier treatment (clipping, winsorization, etc.)
    - Standardization/scaling (mean=0, std=1)
    - Any other transformations needed
    
    For linear DFM, this uses DFMDataset which returns full sequences.
    For DDFM, use DDFMDataModule which uses DDFMDataset with windowing.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    transformer : Any, optional
        sktime-compatible transformer or TransformerPipeline for preprocessing.
        
        **If data is already preprocessed**: Use a passthrough transformer (e.g., from
        `create_passthrough_transformer()`) to avoid double standardization.
        
        **If data is raw**: Provide a TransformerPipeline (from sktime.transformations.compose)
        that handles imputation, scaling, and other preprocessing steps. The pipeline
        will be applied in setup() using fit_transform().
        
        **If None**: Uses default StandardScaler (assumes data is already cleaned but needs scaling).
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pl.DataFrame, optional
        Data array or DataFrame. Can be:
        - Preprocessed data (standardized, no missing values) - use passthrough transformer
        - Raw data (may have missing values, not standardized) - use TransformerPipeline
        If None, data_path must be provided.
    time_index : TimeIndex, optional
        Time index for the data
    batch_size : int, optional
        Batch size for DataLoader. For DFM, typically 1 (full sequence).
    num_workers : int, default 0
        Number of worker processes for DataLoader
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). If None, no validation split.
    
    Examples
    --------
    **Using preprocessed data with passthrough transformer**:
    
    >>> from dfm_python import DFMDataModule
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> 
    >>> # Create passthrough transformer
    >>> passthrough = TabularToSeriesAdaptor(
    ...     FunctionTransformer(func=None, inverse_func=None, validate=False)
    ... )
    >>> 
    >>> # Data is already preprocessed
    >>> dm = DFMDataModule(
    ...     config=model.config,
    ...     transformer=passthrough,  # Avoid double standardization
    ...     data=df_preprocessed  # Already standardized, no missing values
    ... )
    >>> dm.setup()
    
    **Using raw data with TransformerPipeline**:
    
    >>> from dfm_python import DFMDataModule
    >>> from sktime.transformations.compose import TransformerPipeline
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import StandardScaler
    >>> 
    >>> # Create preprocessing pipeline
    >>> preprocessing_pipeline = TransformerPipeline([
    ...     ('impute_ffill', Imputer(method="ffill")),
    ...     ('impute_bfill', Imputer(method="bfill")),
    ...     ('scaler', TabularToSeriesAdaptor(StandardScaler()))
    ... ])
    >>> 
    >>> # Data is raw (may have missing values, not standardized)
    >>> dm = DFMDataModule(
    ...     config=model.config,
    ...     transformer=preprocessing_pipeline,  # Preprocessing happens in setup()
    ...     data=df_raw  # Raw data with missing values
    ... )
    >>> dm.setup()  # Pipeline is applied here
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        transformer: Optional[Any] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pl.DataFrame]] = None,
        time_index: Optional[TimeIndex] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        val_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        _check_sktime_available()
        
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
        # Store transformer (can be None, passthrough, or TransformerPipeline)
        # If None, will create default StandardScaler in setup() if needed
        self.transformer = transformer
        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data
        self.time_index = time_index
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
        """Load and preprocess data.
        
        This method is called by Lightning to set up the data module.
        It loads data, applies user-provided transformer (or TransformerPipeline), 
        and creates train/val datasets.
        
        If a TransformerPipeline is provided, it will be applied to raw data in this method.
        If data is already preprocessed, use a passthrough transformer to avoid double standardization.
        """
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise ValueError(
                    "DataModule setup failed: either data_path or data must be provided. "
                    "Please provide a path to a data file or a data array/DataFrame."
                )
            
            # Load data from file
            # Note: load_data returns (X, Time, Z) where X and Z are both raw data
            X, Time, Z = _load_data(
                self.data_path,
                self.config,
            )
            # Use raw data (X or Z, they're the same)
            self.data = X
            self.time_index = Time
        
        # Convert to Polars DataFrame if needed
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pl.DataFrame(self.data, schema=series_ids)
        elif isinstance(self.data, pl.DataFrame):
            X_df = self.data
        else:
            raise TypeError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or polars.DataFrame."
            )
        
        # Ensure data is Polars DataFrame
        if isinstance(X_df, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pl.DataFrame(X_df, schema=series_ids)
        elif not isinstance(X_df, pl.DataFrame):
            raise TypeError(
                f"DataModule setup failed: unsupported data type {type(X_df)}. "
                f"Please provide data as numpy.ndarray or polars.DataFrame."
            )
        
        # Determine transformer to use
        # If None, create default StandardScaler (assumes data is cleaned but needs scaling)
        if self.transformer is None:
            from sktime.transformations.series.adapt import TabularToSeriesAdaptor
            from sklearn.preprocessing import StandardScaler
            transformer_to_use = TabularToSeriesAdaptor(StandardScaler())
        else:
            transformer_to_use = self.transformer
        
        # Set Polars output for sktime transformers (sktime 1.4+ supports Polars)
        try:
            if hasattr(transformer_to_use, 'set_output'):
                transformer_to_use.set_output(transform="polars")  # Use Polars directly
        except (AttributeError, ValueError) as e:
            # Transformer doesn't support set_output or Polars output
            # This is OK - sktime will handle it
            _logger.debug(f"Could not set Polars output on transformer: {e}")
        
        # Apply transformer (or TransformerPipeline) to data
        # This handles both:
        # 1. Preprocessed data with passthrough transformer (no-op)
        # 2. Raw data with TransformerPipeline (full preprocessing)
        # sktime transformers now support Polars DataFrames directly
        try:
            X_transformed = transformer_to_use.fit_transform(X_df)
        except Exception as e:
            raise ValueError(
                f"DataModule setup failed: transformer fit_transform error: {e}. "
                f"Ensure transformer is sktime-compatible (e.g., TransformerPipeline, TabularToSeriesAdaptor) "
                f"and supports Polars DataFrames. If using preprocessed data, use a passthrough transformer "
                f"to avoid double standardization."
            ) from e
        
        # Ensure output is Polars DataFrame
        if not isinstance(X_transformed, pl.DataFrame):
            # If transformer returned numpy array or pandas DataFrame, convert to Polars
            if hasattr(X_transformed, 'to_polars'):
                X_transformed = X_transformed.to_polars()
            elif hasattr(X_transformed, 'values'):
                # Pandas DataFrame - convert to Polars
                X_transformed = pl.DataFrame(X_transformed.values, schema=X_df.columns)
            elif isinstance(X_transformed, np.ndarray):
                # Numpy array - convert to Polars with original column names
                X_transformed = pl.DataFrame(X_transformed, schema=X_df.columns)
            else:
                raise TypeError(
                    f"DataModule setup failed: transformer returned unsupported type {type(X_transformed)}. "
                    f"Expected polars.DataFrame, pandas.DataFrame, or numpy.ndarray."
                )
        
        # Convert transformed data to numpy (needed for both tensor conversion and Mx/Wx computation)
        X_processed_np = X_transformed.to_numpy()
        
        # Try to extract standardization parameters if transformer includes StandardScaler
        # This is optional - some transformers may not have standardization
        # Mx and Wx are already initialized in __init__
        try:
            # Extract StandardScaler from transformer (works with TransformerPipeline too)
            scaler = _extract_scaler_from_transformer(transformer_to_use)
            
            if scaler is not None:
                # Try to extract Mx and Wx from scaler
                self.Mx = _extract_mx_from_scaler(scaler, X_processed_np)
                self.Wx = _extract_wx_from_scaler(scaler, X_processed_np)
        except (AttributeError, ImportError, Exception) as e:
            # StandardScaler not found or extraction failed
            # Will compute Mx and Wx from data below
            pass
        
        # Convert to torch tensor
        self.data_processed = torch.tensor(X_processed_np, dtype=torch.float32)
        
        # If Mx and Wx are still None, compute from processed data as fallback
        if self.Mx is None or self.Wx is None:
            mx_fallback, wx_fallback = _compute_mx_wx_from_data(X_processed_np)
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
    
    def get_standardization_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get standardization parameters (Mx, Wx) if available.
        
        Returns
        -------
        Mx : np.ndarray or None
            Mean values (N,) if transformer includes StandardScaler, None otherwise
        Wx : np.ndarray or None
            Standard deviation values (N,) if transformer includes StandardScaler, None otherwise
        """
        if self.data_processed is None:
            raise RuntimeError(
                "DataModule get_standardization_params failed: setup() must be called before get_standardization_params(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.Mx, self.Wx
    
    def get_transformer(self) -> Any:
        """Get the transformer used for preprocessing.
        
        Returns
        -------
        transformer : Any
            The sktime transformer or TransformerPipeline provided by the user.
            Returns None if default StandardScaler was used.
        """
        return self.transformer
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor.
        
        Returns
        -------
        data : torch.Tensor
            Processed data (T x N)
        """
        if self.data_processed is None:
            raise RuntimeError(
                "DataModule get_processed_data failed: setup() must be called before get_processed_data(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.data_processed


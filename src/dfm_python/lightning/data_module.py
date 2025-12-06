"""PyTorch Lightning DataModule for DFM training.

This module provides LightningDataModule implementations for loading and
preprocessing data for Dynamic Factor Model training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Any
from pathlib import Path
import pytorch_lightning as lightning_pl

from ..config import DFMConfig
from ..data.utils import load_data as _load_data
from ..data.dataset import DFMDataset
from ..data.dataloader import create_dfm_dataloader
from ..utils.time import TimeIndex
from ..logger import get_logger

_logger = get_logger(__name__)


def _check_sktime():
    """Check if sktime is available and raise ImportError if not."""
    try:
        import sktime
        return True
    except ImportError:
        raise ImportError(
            "DataModule initialization failed: sktime is required for sktime transformers. "
            "Install it with: pip install sktime"
        )


def _get_scaler(transformer: Any) -> Optional[Any]:
    """Extract scaler from transformer, handling wrappers and pipelines.
    
    Recursively searches through transformer wrappers and pipelines
    to find any scaler instance (StandardScaler, MinMaxScaler, RobustScaler, etc.)
    that has mean/center and scale attributes for unstandardization.
    
    This function depends on sktime's pipeline structure to traverse
    TransformerPipeline and sklearn transformers (StandardScaler, etc.).
    
    Parameters
    ----------
    transformer : Any
        Transformer to search (StandardScaler, TransformerPipeline, sklearn Pipeline, etc.)
        
    Returns
    -------
    Optional[Any]
        Scaler instance if found (any scaler with mean/center and scale attributes), 
        None otherwise
    """
    if transformer is None:
        return None
    
    # Check if transformer is wrapped (TabularToSeriesAdaptor or similar)
    if hasattr(transformer, 'transformer'):
        # Recursively search the wrapped transformer
        wrapped = transformer.transformer
        if hasattr(transformer, 'pipeline'):
            wrapped = transformer.pipeline
        return _get_scaler(wrapped)
    
    # Check if transformer is a pipeline (sktime TransformerPipeline or sklearn Pipeline)
    # Both have 'steps' attribute: list of (name, transformer) tuples
    if hasattr(transformer, 'steps'):
        for name, step in transformer.steps:
            scaler = _get_scaler(step)
            if scaler is not None:
                return scaler
    
    # Check if transformer is ColumnEnsembleTransformer (or ColumnTransformer)
    if hasattr(transformer, 'transformers'):
        for name, trans, cols in transformer.transformers:
            scaler = _get_scaler(trans)
            if scaler is not None:
                return scaler
    
    # Check if transformer is a scaler (has mean/center and scale attributes)
    # Support common sklearn scalers: StandardScaler, MinMaxScaler, RobustScaler, etc.
    # A scaler should have either 'mean_' or 'center_' (for mean) and 'scale_' (for scale)
    has_mean_attr = hasattr(transformer, 'mean_') or hasattr(transformer, 'center_')
    has_scale_attr = hasattr(transformer, 'scale_')
    
    if has_mean_attr and has_scale_attr:
        # This looks like a scaler - return it
        return transformer
    
    return None


def _get_scaler_attr(scaler: Any, attr_name: str, data: np.ndarray, default_value: Optional[float] = None, normalize: bool = False) -> Optional[np.ndarray]:
    """Extract attribute from any scaler with fallbacks.
    
    Supports multiple scaler types (StandardScaler, MinMaxScaler, RobustScaler, etc.)
    by checking for common attribute names and enable flags.
    
    Parameters
    ----------
    scaler : Any
        Scaler instance (StandardScaler, MinMaxScaler, RobustScaler, or any scaler
        with mean/center and scale attributes)
    attr_name : str
        Attribute name to extract ('mean_', 'center_', or 'scale_')
    data : np.ndarray
        Processed data array (T x N) for fallback computation
    default_value : float, optional
        Default value if attribute is disabled (0.0 for mean, 1.0 for scale)
    normalize : bool, default False
        Whether to normalize the result (for scale, replaces zeros with 1.0)
        
    Returns
    -------
    Optional[np.ndarray]
        Attribute values (N,) if extracted, None if fallback needed
    """
    # Map attribute names to their enable flags (for StandardScaler)
    # Other scalers may not have these flags, so we'll try direct access
    enable_flag_map = {
        'mean_': 'with_mean',
        'center_': 'with_mean',  # Some scalers use 'center_' instead
        'scale_': 'with_std'
    }
    enable_flag = enable_flag_map.get(attr_name)
    
    # Try to get attribute directly first (works for most scalers)
    # Check for both 'mean_' and 'center_' for mean extraction
    attr_names_to_try = [attr_name]
    if attr_name == 'mean_':
        attr_names_to_try = ['mean_', 'center_']  # Try both
    
    for try_attr_name in attr_names_to_try:
        if hasattr(scaler, try_attr_name):
            try:
                attr_val = getattr(scaler, try_attr_name)
                if attr_val is not None:
                    if not isinstance(attr_val, np.ndarray):
                        attr_val = np.asarray(attr_val)
                    if normalize:
                        attr_val = _normalize_wx(attr_val)
                    return attr_val
            except (AttributeError, TypeError):
                continue
    
    # If direct access failed, check enable flags (for StandardScaler)
    if enable_flag and hasattr(scaler, enable_flag):
        enabled = getattr(scaler, enable_flag)
        if not enabled:
            # If disabled, return default value
            if default_value is not None:
                return np.full(data.shape[1], default_value, dtype=float)
            return None
    
    # No attribute found
    return None


def _get_mean(scaler: Any, data: np.ndarray) -> Optional[np.ndarray]:
    """Extract mean (Mx) from any scaler with fallbacks.
    
    Supports StandardScaler (mean_), MinMaxScaler (center_), RobustScaler (center_),
    and other scalers with mean or center attributes.
    """
    # Try 'mean_' first (StandardScaler), then 'center_' (MinMaxScaler, RobustScaler, etc.)
    result = _get_scaler_attr(scaler, 'mean_', data, default_value=0.0)
    if result is not None:
        return result
    # Fallback to 'center_' for scalers that use that attribute name
    return _get_scaler_attr(scaler, 'center_', data, default_value=0.0)


def _get_scale(scaler: Any, data: np.ndarray) -> Optional[np.ndarray]:
    """Extract scale (Wx) from StandardScaler with fallbacks."""
    return _get_scaler_attr(scaler, 'scale_', data, default_value=1.0, normalize=True)


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


def _compute_mx_wx(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def create_passthrough_transformer() -> Any:
    """Create a passthrough transformer for preprocessed data.
    
    This is the default transformer used when `pipeline=None` in DFMDataModule.
    It performs no transformation on the data (passthrough).
    
    **Purpose**: When data is already preprocessed by the user, this transformer
    is used to avoid any additional transformations. It does not extract statistics
    (Mx/Wx will be computed from data as fallback).
    
    Returns
    -------
    Any
        Passthrough transformer that does nothing to the data
        
    Examples
    --------
    >>> from dfm_python.lightning.data_module import create_passthrough_transformer
    >>> from dfm_python import DFMDataModule
    >>> 
    >>> # Data is already preprocessed by user
    >>> passthrough = create_passthrough_transformer()
    >>> dm = DFMDataModule(
    ...     config=config,
    ...     pipeline=passthrough,  # No transformation (same as pipeline=None)
    ...     data=df_preprocessed  # Already preprocessed by user
    ... )
    """
    _check_sktime()
    
    from sklearn.preprocessing import FunctionTransformer
    
    # Return FunctionTransformer directly (no TabularToSeriesAdaptor needed)
    # Per sktime docs: sklearn transformers work directly in TransformerPipeline
    return FunctionTransformer(func=None, inverse_func=None, validate=False)


class DFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DFM training.
    
    This DataModule handles data loading for DFM/DDFM models.
    
    **Important**: DFM and DDFM can handle missing data (NaN values) implicitly:
    - **DFM**: Uses Kalman filter's `handle_missing_data()` method to skip NaN observations
    - **DDFM**: Uses state-space model and MCMC procedure to handle missing data through
      idiosyncratic component estimation
    
    **Usage Pattern**:
    - Data can contain NaN values - models will handle them implicitly
    - If `pipeline=None`, a passthrough transformer is used by default (no-op)
    - Users can optionally provide their preprocessing pipeline to extract statistics (Mx/Wx)
    - For better performance, users can preprocess data (imputation, scaling) before passing,
      but it's not required - models will handle missing data automatically
    
    For linear DFM, this uses DFMDataset which returns full sequences.
    For DDFM, use DDFMDataModule which uses DDFMDataset with windowing.
    
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
        
        **Example**: Users can pass a full preprocessing pipeline:
        ```python
        from sklearn.preprocessing import StandardScaler
        
        preprocessing_pipeline = TransformerPipeline([
            ("impute_ffill", Imputer(method="ffill")),
            ("impute_bfill", Imputer(method="bfill")),
            ("window_features", WindowSummarizer(...)),
            ("standardize", StandardScaler())  # Unified scaling - no wrapper needed!
        ])
        dm = DFMDataModule(config=config, pipeline=preprocessing_pipeline, data=df_preprocessed)
        ```
        
        **Note**: Users must preprocess their data (imputation, scaling, feature engineering) before
        passing it to this DataModule. The pipeline parameter is for extracting statistics from
        the preprocessing pipeline, not for performing preprocessing.
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided.
    data : np.ndarray or pd.DataFrame, optional
        Data array or DataFrame. Can contain NaN values - DFM/DDFM will handle them:
        - DFM: Uses Kalman filter to implicitly handle missing data
        - DDFM: Uses state-space model and MCMC to handle missing data
        - Standardized/scaled data (mean=0, std=1) is recommended for better performance
        - Feature-engineered if needed
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
    **Using preprocessed data (recommended - default behavior)**:
    
    >>> from dfm_python import DFMDataModule
    >>> import pandas as pd
    >>> 
    >>> # Data is already preprocessed (standardized, no missing values)
    >>> # Users should preprocess their data using sktime or other tools before this step
    >>> dm = DFMDataModule(
    ...     config=config,
    ...     data=df_preprocessed  # Already preprocessed by user
    ... )
    >>> dm.setup()  # Uses passthrough transformer by default
    
    **Using preprocessed data with pipeline to extract statistics**:
    
    >>> from dfm_python import DFMDataModule
    >>> from sktime.transformations.compose import TransformerPipeline
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.summarize import WindowSummarizer
    >>> from sklearn.preprocessing import StandardScaler
    >>> 
    >>> # User created preprocessing pipeline (data already preprocessed)
    >>> # Pass pipeline to extract Mx/Wx for forecasting/nowcasting
    >>> # Per sktime docs: sklearn transformers work directly (unified scaling)
    >>> preprocessing_pipeline = TransformerPipeline([
    ...     ("impute_ffill", Imputer(method="ffill")),
    ...     ("impute_bfill", Imputer(method="bfill")),
    ...     ("window_features", WindowSummarizer(...)),
    ...     ("standardize", StandardScaler())  # Unified scaling - no wrapper needed!
    ... ])
    >>> dm = DFMDataModule(
    ...     config=config,
    ...     pipeline=preprocessing_pipeline,  # Extract Mx/Wx from StandardScaler in pipeline
    ...     data=df_preprocessed  # Already preprocessed by user
    ... )
    >>> dm.setup()  # Fits pipeline to extract statistics, Mx/Wx available for predictions
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        pipeline: Optional[Any] = None,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        time_index: Optional[TimeIndex] = None,
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
        # Store pipeline (can be None, passthrough, or user-provided preprocessing pipeline)
        # If None, will use passthrough transformer in setup() (assumes preprocessed data)
        self.pipeline = pipeline
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
        """Load and prepare data.
        
        This method is called by Lightning to set up the data module.
        It loads preprocessed data and extracts statistics from the pipeline for forecasting/nowcasting.
        
        **Important**: DFM and DDFM can handle missing data (NaN values) implicitly:
        - **DFM**: Uses Kalman filter's `handle_missing_data()` method
        - **DDFM**: Uses state-space model and MCMC procedure
        
        **Pipeline Purpose**:
        - The pipeline is used to extract statistics (e.g., Mx/Wx from StandardScaler) needed for
          forecasting and nowcasting operations, not to preprocess the data
        - The pipeline is fitted on the **entire dataset** to extract statistics
        - If `val_split` is provided, it's used for validation during training (e.g., DDFM neural network)
        
        **Usage**:
        - Data can contain NaN values - models will handle them automatically
        - Optionally provide pipeline (e.g., TransformerPipeline with StandardScaler) to extract Mx/Wx for predictions
        - If pipeline=None, uses passthrough (no statistics extracted, Mx/Wx computed from data)
        - For better performance, users can preprocess data (imputation, scaling) before passing,
          but it's not required
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
        
        # Convert to pandas DataFrame if needed
        # Note: DFM/DDFM can handle NaN values implicitly via Kalman filter
        # (DFM uses handle_missing_data() in Kalman filter, DDFM uses state-space model)
        # So we don't validate NaN here - let the models handle it
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pd.DataFrame(self.data, columns=series_ids)
        elif isinstance(self.data, pd.DataFrame):
            X_df = self.data
        else:
            raise TypeError(
                f"DataModule setup failed: unsupported data type {type(self.data)}. "
                f"Please provide data as numpy.ndarray or pandas.DataFrame."
            )
        
        # Determine pipeline to use
        # If None, use passthrough transformer (assumes data is already fully preprocessed)
        if self.pipeline is None:
            pipeline_to_use = create_passthrough_transformer()
        else:
            pipeline_to_use = self.pipeline
        
        # Set pandas output for sktime pipelines (sktime supports pandas)
        try:
            if hasattr(pipeline_to_use, 'set_output'):
                pipeline_to_use.set_output(transform="pandas")  # Use pandas directly
        except (AttributeError, ValueError) as e:
            # Pipeline doesn't support set_output or pandas output
            # This is OK - sktime will handle it
            _logger.debug(f"Could not set pandas output on pipeline: {e}")
        
        # Check if pipeline is already fitted (warn user if so)
        if hasattr(pipeline_to_use, 'fit') and hasattr(pipeline_to_use, 'transform'):
            try:
                # Try to check if pipeline is fitted by accessing a fitted attribute
                if hasattr(pipeline_to_use, 'get_params'):
                    # This is a basic check - some pipelines may not have this
                    pass
            except Exception:
                pass
        
        # Apply pipeline to extract statistics (Mx/Wx) for forecasting/nowcasting
        # Note: Data is already preprocessed by the user. The pipeline is used to extract
        # standardization parameters (e.g., from StandardScaler) needed for predictions.
        # For DFM/DDFM, we fit on the entire dataset since these models learn the full time series dynamics.
        # If val_split is used, it's for validation during training (e.g., DDFM neural network training).
        # sktime pipelines support pandas DataFrames directly
        try:
            X_transformed = pipeline_to_use.fit_transform(X_df)
        except Exception as e:
            raise ValueError(
                f"DataModule setup failed: pipeline fit_transform error: {e}. "
                f"Ensure pipeline is sktime-compatible (e.g., TransformerPipeline with StandardScaler) "
                f"and supports pandas DataFrames. Data must be preprocessed before passing to this DataModule. "
                f"If pipeline creates new columns (e.g., WindowSummarizer), column names will be "
                f"automatically extracted from the pipeline."
            ) from e
        
        # Note: DFM/DDFM can handle NaN values implicitly via Kalman filter
        # So we don't validate NaN here - let the models handle it
        
        # Ensure output is pandas DataFrame with compatible index
        if not isinstance(X_transformed, pd.DataFrame):
            # If transformer returned numpy array, convert to pandas
            if isinstance(X_transformed, np.ndarray):
                # Try to get feature names from pipeline (handles pipelines that create new columns)
                if hasattr(pipeline_to_use, 'get_feature_names_out'):
                    try:
                        new_cols = pipeline_to_use.get_feature_names_out(X_df.columns)
                        X_transformed = pd.DataFrame(X_transformed, columns=new_cols)
                    except Exception:
                        # Fallback: use input columns if shape matches
                        n_cols = X_transformed.shape[1] if len(X_transformed.shape) > 1 else 1
                        if n_cols == len(X_df.columns):
                            X_transformed = pd.DataFrame(X_transformed, columns=X_df.columns)
                        else:
                            # Create generic column names for transformers that create new columns
                            X_transformed = pd.DataFrame(X_transformed, 
                                columns=[f'feature_{i}' for i in range(n_cols)])
                else:
                    # No feature name method - use input columns if shape matches
                    n_cols = X_transformed.shape[1] if len(X_transformed.shape) > 1 else 1
                    if n_cols == len(X_df.columns):
                        X_transformed = pd.DataFrame(X_transformed, columns=X_df.columns)
                    else:
                        # Create generic column names
                        X_transformed = pd.DataFrame(X_transformed, 
                            columns=[f'feature_{i}' for i in range(n_cols)])
                
                # Ensure index is compatible (DatetimeIndex, PeriodIndex, or RangeIndex)
                # Use input DataFrame index if length matches
                if len(X_transformed) == len(X_df):
                    if isinstance(X_df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
                        X_transformed.index = X_df.index
                    else:
                        # Try to convert to DatetimeIndex
                        try:
                            X_transformed.index = pd.to_datetime(X_df.index)
                        except (ValueError, TypeError):
                            # Fallback to RangeIndex
                            X_transformed.index = pd.RangeIndex(start=0, stop=len(X_transformed))
                else:
                    # Length doesn't match - use RangeIndex
                    X_transformed.index = pd.RangeIndex(start=0, stop=len(X_transformed))
        
        # Ensure DataFrame index is compatible (DatetimeIndex, PeriodIndex, or RangeIndex)
        # This handles cases where pipeline output has incompatible index
        if isinstance(X_transformed, pd.DataFrame):
            if not isinstance(X_transformed.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
                # Try to use input DataFrame index if length matches
                if len(X_transformed) == len(X_df):
                    if isinstance(X_df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)):
                        X_transformed.index = X_df.index
                    else:
                        # Try to convert to DatetimeIndex
                        try:
                            X_transformed.index = pd.to_datetime(X_df.index)
                        except (ValueError, TypeError):
                            # Fallback to RangeIndex
                            X_transformed.index = pd.RangeIndex(start=0, stop=len(X_transformed))
                else:
                    # Try to convert existing index
                    try:
                        X_transformed.index = pd.to_datetime(X_transformed.index)
                    except (ValueError, TypeError):
                        # Fallback to RangeIndex
                        X_transformed.index = pd.RangeIndex(start=0, stop=len(X_transformed))
            elif hasattr(X_transformed, 'to_numpy'):
                # Has to_numpy method (pandas DataFrame) - convert to numpy then back to DataFrame
                # Try to preserve column names if available
                if hasattr(X_transformed, 'columns'):
                    X_transformed = pd.DataFrame(X_transformed.to_numpy(), columns=X_transformed.columns)
                else:
                    # Fallback to input columns if shape matches
                    arr = np.asarray(X_transformed.to_numpy())
                    n_cols = arr.shape[1] if len(arr.shape) > 1 else 1
                    if n_cols == len(X_df.columns):
                        X_transformed = pd.DataFrame(arr, columns=X_df.columns)
                    else:
                        X_transformed = pd.DataFrame(arr, 
                            columns=[f'feature_{i}' for i in range(n_cols)])
            elif hasattr(X_transformed, 'values'):
                # Has values attribute - try to convert (fallback for other types)
                if hasattr(X_transformed, 'columns'):
                    X_transformed = pd.DataFrame(X_transformed.values, columns=X_transformed.columns)
                else:
                    # Fallback to input columns if shape matches
                    arr = np.asarray(X_transformed.values)
                    n_cols = arr.shape[1] if len(arr.shape) > 1 else 1
                    if n_cols == len(X_df.columns):
                        X_transformed = pd.DataFrame(arr, columns=X_df.columns)
                    else:
                        X_transformed = pd.DataFrame(arr, 
                            columns=[f'feature_{i}' for i in range(n_cols)])
            else:
                raise TypeError(
                    f"DataModule setup failed: pipeline returned unsupported type {type(X_transformed)}. "
                    f"Expected pandas.DataFrame or numpy.ndarray."
                )
        
        # Convert transformed data to numpy (needed for both tensor conversion and Mx/Wx computation)
        X_processed_np = X_transformed.to_numpy()
        
        # Try to extract standardization parameters if pipeline includes a scaler
        # This is optional - some pipelines may not have standardization
        # Mx and Wx are already initialized in __init__
        try:
            # Extract scaler from pipeline (supports TransformerPipeline, StandardScaler, etc.)
            scaler = _get_scaler(pipeline_to_use)
            
            if scaler is not None:
                # Try to extract Mx and Wx from scaler (supports multiple scaler types)
                self.Mx = _get_mean(scaler, X_processed_np)
                self.Wx = _get_scale(scaler, X_processed_np)
        except (AttributeError, ImportError, Exception) as e:
            # Scaler not found or extraction failed
            # Will compute Mx and Wx from data below
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
                "DataModule get_std_params failed: setup() must be called before get_std_params(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        return self.Mx, self.Wx
    
    def get_pipeline(self) -> Any:
        """Get the preprocessing pipeline used for statistics extraction.
        
        Returns
        -------
        pipeline : Any
            The sktime preprocessing pipeline (e.g., TransformerPipeline) provided by the user.
            Returns None if passthrough transformer was used (default).
        """
        return self.pipeline
    
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


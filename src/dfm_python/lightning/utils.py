"""Utility functions for DFM/DDFM DataModules.

This module contains shared utility functions for data preprocessing,
scaling, and statistics extraction used by both DFMDataModule and DDFMDataModule.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Any, List, Dict, Protocol
from pathlib import Path

from ..config import DFMConfig, SeriesConfig
from ..logger import get_logger

_logger = get_logger(__name__)


# ============================================================================
# sktime/sklearn checks
# ============================================================================

def _check_sktime():
    """Check if sktime is available and raise ImportError if not."""
    try:
        import sktime
        return True
    except ImportError:
        raise ImportError(
            "DataModule utilities require sktime. Install with: pip install sktime"
        )


def _check_sklearn():
    """Check if sklearn is available and raise ImportError if not."""
    try:
        import sklearn
        return True
    except ImportError:
        raise ImportError(
            "Scaling utilities require scikit-learn. Install with: pip install scikit-learn"
        )


# ============================================================================
# Scaler extraction utilities
# ============================================================================

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
    
    This function replaces zero or NaN values in Wx with 1.0 to prevent
    division by zero during standardization/unstandardization.
    
    Parameters
    ----------
    wx : np.ndarray
        Scale values (N,), may contain zeros or NaN
        
    Returns
    -------
    np.ndarray
        Normalized scale values with zeros and NaN replaced by 1.0
    """
    # Replace both zero and NaN with 1.0
    return np.where((wx == 0) | np.isnan(wx), 1.0, wx)


def _compute_mx_wx(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Mx and Wx from data as fallback.
    
    This function computes mean (Mx) and standard deviation (Wx) from data,
    handling missing values (NaN) by using nan-aware functions. If NaN values
    are detected, a warning is issued recommending imputation.
    
    Parameters
    ----------
    data : np.ndarray
        Processed data array (T x N), may contain NaN values
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Mx, Wx) tuple where Mx is mean and Wx is normalized std.
        NaN values are handled using nanmean and nanstd.
        
    Notes
    -----
    If data contains NaN values, this function will:
    1. Use np.nanmean() and np.nanstd() to compute statistics ignoring NaN
    2. Issue a warning recommending data imputation for better results
    3. Continue processing with available data
    """
    # Check for NaN values
    has_nan = np.any(np.isnan(data))
    if has_nan:
        nan_count = np.sum(np.isnan(data))
        nan_pct = 100.0 * nan_count / data.size
        _logger.warning(
            f"Data contains {nan_count} NaN values ({nan_pct:.1f}%). "
            f"Using nanmean/nanstd to compute standardization parameters. "
            f"For better results, consider imputing missing values before creating DataModule. "
            f"Suggested approaches: forward-fill, backward-fill, or interpolation."
        )
    
    # Use nan-aware functions to handle missing values
    mx = np.nanmean(data, axis=0)
    wx = np.nanstd(data, axis=0)
    
    # Check if any series have all NaN values (would result in NaN std)
    nan_std_mask = np.isnan(wx)
    if np.any(nan_std_mask):
        n_nan_std = np.sum(nan_std_mask)
        _logger.warning(
            f"{n_nan_std} series have all NaN values, resulting in NaN standard deviation. "
            f"These will be normalized to 1.0 to avoid division by zero. "
            f"Consider imputing or removing these series."
        )
        # Replace NaN std with 1.0 (will be normalized by _normalize_wx anyway)
        wx = np.where(nan_std_mask, 1.0, wx)
    
    wx = _normalize_wx(wx)
    return mx, wx


# ============================================================================
# Pipeline utilities
# ============================================================================

def create_passthrough_transformer() -> Any:
    """Create a passthrough transformer for preprocessed data.
    
    This is the default transformer used when `pipeline=None` in DataModules.
    It performs no transformation on the data (passthrough).
    
    **Purpose**: When data is already preprocessed by the user, this transformer
    is used to avoid any additional transformations. It does not extract statistics
    (Mx/Wx will be computed from data as fallback).
    
    Returns
    -------
    Any
        Passthrough transformer that does nothing to the data
    """
    _check_sktime()
    
    from sklearn.preprocessing import FunctionTransformer
    
    # Return FunctionTransformer directly (no TabularToSeriesAdaptor needed)
    # Per sktime docs: sklearn transformers work directly in TransformerPipeline
    return FunctionTransformer(func=None, inverse_func=None, validate=False)


def _is_pipeline_fitted(pipeline: Any) -> bool:
    """Check if pipeline is already fitted.
    
    Parameters
    ----------
    pipeline : Any
        Pipeline to check
        
    Returns
    -------
    bool
        True if pipeline appears to be fitted, False otherwise
    """
    if pipeline is None:
        return False
    
    # Check if pipeline has steps
    if hasattr(pipeline, 'steps'):
        # Check if any step has fitted attributes
        for name, step in pipeline.steps:
            if hasattr(step, 'mean_') or hasattr(step, 'center_') or hasattr(step, 'scale_'):
                return True
            if hasattr(step, 'transformers'):
                # ColumnEnsembleTransformer
                return True
    
    # Check if it's a scaler directly
    if hasattr(pipeline, 'mean_') or hasattr(pipeline, 'center_'):
        return True
    
    return False


def _is_scaler_fitted(scaler: Any) -> bool:
    """Check if scaler is already fitted.
    
    Parameters
    ----------
    scaler : Any
        Scaler to check
        
    Returns
    -------
    bool
        True if scaler appears to be fitted, False otherwise
    """
    if scaler is None:
        return False
    
    return hasattr(scaler, 'mean_') or hasattr(scaler, 'center_') or hasattr(scaler, 'scale_')


# ============================================================================
# Scaling utilities (from scaling.py)
# ============================================================================

class ScalingStrategy(Protocol):
    """Protocol for custom scaling strategies (optional - unified scaling recommended).
    
    **Note**: Unified scaling (StandardScaler for all series) is recommended for
    factor models. This protocol is provided for advanced use cases only.
    """
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Determine scaler type for a series."""
        ...


class DefaultScalingStrategy:
    """Default scaling strategy: unified scaling (StandardScaler for all series)."""
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Return 'standard' for unified scaling (all series use StandardScaler)."""
        return 'standard'


class NoScalingStrategy:
    """Strategy that applies no scaling to any series."""
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Return None (no scaling) for all series."""
        return None


def _create_scaler_transformer(
    scaler_type: Optional[str],
    use_robust_as_default: bool = False
) -> Any:
    """Create a raw sklearn scaler for unified scaling."""
    _check_sklearn()
    
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, 
        MaxAbsScaler, QuantileTransformer
    )
    
    if scaler_type is None or scaler_type == 'none':
        if use_robust_as_default:
            scaler_type_lower = 'robust'
        else:
            # Passthrough: no scaling
            return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    else:
        scaler_type_lower = scaler_type.lower()
    
    if scaler_type_lower == 'standard':
        return StandardScaler()
    elif scaler_type_lower == 'robust':
        return RobustScaler()
    elif scaler_type_lower == 'minmax':
        return MinMaxScaler()
    elif scaler_type_lower == 'maxabs':
        return MaxAbsScaler()
    elif scaler_type_lower == 'quantile':
        return QuantileTransformer(output_distribution='normal')
    else:
        _logger.warning(
            f"Unknown scaler type '{scaler_type}', using StandardScaler as fallback"
        )
        return StandardScaler()


def create_scaling_transformer_from_config(
    config: DFMConfig,
    strategy: Optional[ScalingStrategy] = None,
    column_names: Optional[List[str]] = None
) -> Any:
    """Create a unified scaling transformer from model-level scaler configuration."""
    _check_sklearn()
    
    # Get scaler type from model-level config (config.scaler)
    if hasattr(config, 'scaler') and config.scaler is not None:
        scaler_type = config.scaler
    elif strategy is not None:
        # Fallback to strategy if config.scaler not available
        if len(config.series) > 0:
            first_series = config.series[0]
            first_column_name = config.get_series_ids()[0] if config.series else "series_0"
            scaler_type = strategy.get_scaler(first_series, 0, first_column_name)
        else:
            scaler_type = 'standard'
    else:
        # Default to 'standard' for unified scaling
        scaler_type = 'standard'
    
    # Return raw sklearn scaler (works directly in TransformerPipeline per sktime docs)
    if scaler_type is None or scaler_type == 'null':
        # No scaling - return passthrough
        from sktime.transformations.series.func_transform import FunctionTransformer
        return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    else:
        # Unified scaling - return raw sklearn scaler
        return _create_scaler_transformer(scaler_type)


def create_uniform_scaling_transformer(
    scaler_type: str = 'standard'
) -> Any:
    """Create a unified scaling transformer (StandardScaler for all series)."""
    _check_sklearn()
    
    return _create_scaler_transformer(scaler_type)


def create_preprocessing_pipeline_with_scaling(
    config: DFMConfig,
    imputation_steps: Optional[List[Any]] = None,
    feature_engineering: Optional[Any] = None,
    scaling_strategy: Optional[ScalingStrategy] = None,
    column_names: Optional[List[str]] = None
) -> Any:
    """Create a complete preprocessing pipeline with automatic per-series scaling."""
    _check_sktime()
    
    from sktime.transformations.compose import TransformerPipeline
    
    steps = []
    
    # Add imputation steps
    if imputation_steps:
        for i, step in enumerate(imputation_steps):
            steps.append((f"impute_{i}", step))
    
    # Add feature engineering
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    
    # Add unified scaling (StandardScaler for all series)
    scaling_transformer = create_scaling_transformer_from_config(
        config, strategy=scaling_strategy, column_names=column_names
    )
    steps.append(("scaling", scaling_transformer))
    
    return TransformerPipeline(steps)


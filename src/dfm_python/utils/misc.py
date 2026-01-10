"""Miscellaneous utilities for DFM operations.

This module combines:
- Helper functions (parameter resolution, config access)
- Validation utilities
- Exception classes
- Parameter resolution utilities
"""

from typing import Optional, Any, List, Union, Tuple, Dict, TYPE_CHECKING, NoReturn

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from ..config.schema import DFMConfig, DFMResult
else:
    torch = None

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
    if not TYPE_CHECKING:
        torch = None

from ..logger import get_logger
from .errors import NumericalError, ConfigValidationError

_logger = get_logger(__name__)


def resolve_param(
    override: Optional[Any] = None,
    default: Any = None,
    *,
    name: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Any:
    """Resolve parameter value from multiple sources with priority."""
    # Named pattern: extract by name from multiple sources
    if name is not None:
        if kwargs is not None and name in kwargs:
            return kwargs.pop(name) if isinstance(kwargs, dict) else kwargs.get(name)
        if config is not None and hasattr(config, name):
            return getattr(config, name)
        if defaults is not None and name in defaults:
            return defaults[name]
        return None
    
    # Simple pattern: override > default
    if override is not None:
        return override
    return default


def get_config_attr(
    config: Optional[Any],
    attr_name: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """Get configuration attribute with fallback and validation."""
    if config is None:
        if required:
            raise ConfigValidationError(f"Config is None, cannot access required attribute '{attr_name}'")
        return default
    
    if hasattr(config, attr_name):
        value = getattr(config, attr_name)
        if value is not None:
            return value
    
    if required:
        raise ConfigValidationError(f"Config missing required attribute '{attr_name}'")
    
    return default


def get_clock_frequency(config: Optional["DFMConfig"], default: Optional[str] = None) -> str:
    """Get clock frequency from config.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    default : str, optional
        Default clock frequency if config is None
        
    Returns
    -------
    str
        Clock frequency string
    """
    from ..config.constants import DEFAULT_CLOCK_FREQUENCY
    return get_config_attr(config, 'clock', default or DEFAULT_CLOCK_FREQUENCY)


def compute_default_horizon(
    config: Optional["DFMConfig"] = None,
    default: Optional[int] = None
) -> int:
    """Compute default forecast horizon from clock frequency.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object to extract clock frequency from
    default : int, optional
        Default value to use if clock frequency cannot be determined.
        If None, uses DEFAULT_FORECAST_HORIZON constant.
        
    Returns
    -------
    int
        Default horizon in periods (typically 1 year worth of periods)
    """
    from ..config.constants import DEFAULT_FORECAST_HORIZON
    from ..config import get_periods_per_year
    
    if default is None:
        default = DEFAULT_FORECAST_HORIZON
    
    try:
        if config is not None:
            clock = get_clock_frequency(config)
            return get_periods_per_year(clock)
    except (AttributeError, ImportError, ValueError):
        _logger.debug(f"Could not determine horizon from clock frequency, using default={default}")
    
    return default


def resolve_target_series(
    dataset: Optional[Any],
    series_ids: Optional[List[str]] = None,
    result: Optional[Any] = None,
    model_name: str = "model"
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Resolve target series from Dataset.
    
    This utility function resolves target series from the Dataset's target_series attribute
    and maps them to indices in the series_ids list.
    
    Parameters
    ----------
    dataset : Any, optional
        Dataset instance with target_series attribute
    series_ids : List[str], optional
        Available series IDs from config or result. Used for validation.
    result : Any, optional
        Result object that may contain series_ids. Used as fallback.
    model_name : str, default="model"
        Model name for error messages
        
    Returns
    -------
    Tuple[Optional[List[str]], Optional[List[int]]]
        Tuple of (target_series_ids, target_indices) where:
        - target_series_ids: List of target series IDs (None if not resolved)
        - target_indices: List of indices into series_ids (None if not resolved)
        
    Raises
    ------
    DataError
        If target series are not found in available series
    """
    from ..utils.errors import DataError
    from ..config.constants import MAX_WARNING_ITEMS, MAX_ERROR_ITEMS
    
    # Get target series from Dataset
    target_series = None
    if dataset is not None:
        target_series = getattr(dataset, 'target_series', None)
        if target_series is not None and len(target_series) > 0:
            target_series = target_series if isinstance(target_series, list) else [target_series]
    
    # Resolve indices if we have both target_series and series_ids
    target_indices = None
    if target_series is not None and series_ids is not None:
        target_indices = []
        for tgt_id in target_series:
            if tgt_id in series_ids:
                target_indices.append(series_ids.index(tgt_id))
            else:
                _logger.warning(
                    f"{model_name} prediction: target series '{tgt_id}' not found in series_ids. "
                    f"Available: {series_ids[:MAX_WARNING_ITEMS]}{'...' if len(series_ids) > MAX_WARNING_ITEMS else ''}. "
                    f"Skipping this target series."
                )
        
        if len(target_indices) == 0:
            raise DataError(
                f"{model_name} prediction failed: none of the specified target series found",
                details=f"Target: {target_series}, Available: {series_ids[:MAX_ERROR_ITEMS]}{'...' if len(series_ids) > MAX_ERROR_ITEMS else ''}"
            )
    
    return target_series, target_indices


# Scaling utilities
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

def _check_sklearn():
    """Check if sklearn is available."""
    try:
        import sklearn
        return True
    except ImportError:
        return False

def select_columns_by_prefix(df: pd.DataFrame, prefixes: List[str], count_per_prefix: int = 2) -> List[str]:
    """Select columns from DataFrame that start with given prefixes."""
    selected = []
    for prefix in prefixes:
        matching = [col for col in df.columns if col.startswith(prefix)]
        selected.extend(matching[:count_per_prefix])
    return selected


def get_target_scaler(dataset: Optional[Any] = None, model: Optional[Any] = None) -> Optional[Any]:
    """Get target scaler from dataset or model.
    
    Parameters
    ----------
    dataset : Any, optional
        Dataset instance (DFMDataset, DDFMDataset)
    model : Any, optional
        Model instance (DFM, DDFM)
        
    Returns
    -------
    Any or None
        Target scaler from dataset or model, or None if not available
    """
    if dataset is not None:
        return getattr(dataset, 'target_scaler', None)
    if model is not None:
        return getattr(model, 'target_scaler', None)
    return None

# Metric functions (moved to metric.py)
from .metric import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
)

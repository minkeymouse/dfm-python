"""Helper functions for DFM operations.

This module provides utility functions for:
- Safe configuration access (safe_get_attr, safe_get_method)
- Parameter resolution (resolve_param)
- Data standardization (standardize_data, safe_mean_std)
- Config access helpers (get_clock_frequency, get_series_ids, get_frequencies_from_config, etc.)
- Validation helpers (_validate_config_loaded, _validate_data_loaded, _validate_result_loaded)
"""

import numpy as np
from typing import Optional, Any, List, Callable, Union
import logging

from ..config import DFMConfig

_logger = logging.getLogger(__name__)


def safe_get_attr(obj: Any, attr_name: str, default: Any = None) -> Any:
    """Safely get attribute from object with default value.
    
    Parameters
    ----------
    obj : Any
        Object to get attribute from (may be None)
    attr_name : str
        Name of attribute to get
    default : Any, optional
        Default value if attribute doesn't exist or obj is None
        
    Returns
    -------
    Any
        Attribute value or default
    """
    if obj is None:
        return default
    return getattr(obj, attr_name, default)


def safe_get_method(obj: Any, method_name: str, default: Any = None) -> Any:
    """Safely get and call a method from config object.
    
    Parameters
    ----------
    obj : Any
        Configuration object (may be None)
    method_name : str
        Name of the method to call
    default : Any, optional
        Default value to return if config is None or method doesn't exist
        
    Returns
    -------
    Any
        Method result or default
    """
    if obj is None:
        return default
    
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return default
    
    try:
        return method()
    except Exception as e:
        _logger.debug(f"Error calling {method_name}: {e}")
        return default


def resolve_param(override: Optional[Any], config_value: Optional[Any], default: Any = None) -> Any:
    """Resolve parameter value from override, config, or default.
    
    Priority: override > config_value > default
    
    Parameters
    ----------
    override : Any, optional
        Parameter override value (highest priority)
    config_value : Any, optional
        Configuration value (medium priority)
    default : Any, optional
        Default value (lowest priority)
        
    Returns
    -------
    Any
        Resolved parameter value
    """
    if override is not None:
        return override
    if config_value is not None:
        return config_value
    return default


def safe_mean_std(data: np.ndarray, axis: int = 0, ddof: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std safely, handling NaN values.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    axis : int, default 0
        Axis along which to compute
    ddof : int, default 0
        Delta degrees of freedom for std calculation
        
    Returns
    -------
    mean : np.ndarray
        Mean values
    std : np.ndarray
        Standard deviation values
    """
    mean = np.nanmean(data, axis=axis)
    std = np.nanstd(data, axis=axis, ddof=ddof)
    # Replace zero std with 1.0 to avoid division by zero
    std = np.where(std == 0, 1.0, std)
    return mean, std


def standardize_data(
    X: np.ndarray,
    clip_data: bool = True,
    clip_threshold: float = 100.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize data: (X - mean) / std.
    
    Parameters
    ----------
    X : np.ndarray
        Input data (T x N)
    clip_data : bool, default True
        Whether to clip extreme values before standardization
    clip_threshold : float, default 100.0
        Threshold for clipping (values beyond ±threshold are clipped)
        
    Returns
    -------
    X_standardized : np.ndarray
        Standardized data (T x N)
    Mx : np.ndarray
        Mean values (N,)
    Wx : np.ndarray
        Standard deviation values (N,)
    """
    T, N = X.shape
    
    # Clip extreme values if requested
    if clip_data:
        X = np.clip(X, -clip_threshold, clip_threshold)
    
    # Compute mean and std
    Mx, Wx = safe_mean_std(X, axis=0)
    
    # Standardize
    X_standardized = (X - Mx) / Wx
    
    return X_standardized, Mx, Wx


def get_clock_frequency(config: DFMConfig, default: str = 'm') -> str:
    """Get clock frequency from config.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
    default : str, default 'm'
        Default clock frequency if not found
        
    Returns
    -------
    str
        Clock frequency code
    """
    return safe_get_attr(config, 'clock', default)


def get_series_ids(config: DFMConfig) -> List[str]:
    """Get list of series IDs from config.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of series IDs
    """
    result = safe_get_method(config, 'get_series_ids')
    if result is None:
        # Fallback: extract from series configs
        if hasattr(config, 'series') and config.series:
            return [safe_get_attr(s, 'series_id', f'series_{i}') 
                   for i, s in enumerate(config.series)]
        return []
    return result if isinstance(result, list) else []


def get_series_names(config: DFMConfig) -> List[str]:
    """Get list of series names from config.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of series names
    """
    result = safe_get_method(config, 'get_series_names')
    if result is None:
        # Fallback: use series_ids
        return get_series_ids(config)
    return result if isinstance(result, list) else []


def get_frequencies_from_config(config: DFMConfig) -> List[str]:
    """Get list of frequencies from config.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of frequency codes for each series
    """
    if not hasattr(config, 'series') or not config.series:
        return []
    
    frequencies = []
    for series_config in config.series:
        freq = safe_get_attr(series_config, 'frequency', 'm')
        frequencies.append(freq)
    
    return frequencies


def get_series_id_by_index(config: DFMConfig, index: int) -> Optional[str]:
    """Get series ID by index.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
    index : int
        Series index
        
    Returns
    -------
    str, optional
        Series ID or None if index is out of range
    """
    series_ids = get_series_ids(config)
    if 0 <= index < len(series_ids):
        return series_ids[index]
    return None


def find_series_index(config: DFMConfig, series_id: str) -> Optional[int]:
    """Find index of series by ID.
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
    series_id : str
        Series ID to find
        
    Returns
    -------
    int, optional
        Series index or None if not found
    """
    series_ids = get_series_ids(config)
    try:
        return series_ids.index(series_id)
    except ValueError:
        return None


def _validate_config_loaded(config: Optional[DFMConfig]) -> None:
    """Validate that config is loaded.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
        
    Raises
    ------
    ValueError
        If config is None or invalid
    """
    if config is None:
        raise ValueError("Configuration not loaded. Call load_config() first.")
    if not isinstance(config, DFMConfig):
        raise ValueError(f"Invalid config type: {type(config)}. Expected DFMConfig.")


def _validate_data_loaded(data: Optional[np.ndarray]) -> None:
    """Validate that data is loaded.
    
    Parameters
    ----------
    data : np.ndarray, optional
        Data array
        
    Raises
    ------
    ValueError
        If data is None or invalid
    """
    if data is None:
        raise ValueError("Data not loaded. Call load_data() first.")
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Invalid data type: {type(data)}. Expected np.ndarray.")


def _validate_result_loaded(result: Optional[Any]) -> None:
    """Validate that result is available.
    
    Parameters
    ----------
    result : Any, optional
        Result object
        
    Raises
    ------
    ValueError
        If result is None
    """
    if result is None:
        raise ValueError("Model not trained. Call train() or fit() first.")


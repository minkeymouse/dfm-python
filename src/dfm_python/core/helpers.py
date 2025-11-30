"""Helper functions for DFM operations.

This module provides utility functions for:
- Safe configuration access (safe_get_attr, safe_get_method)
- Parameter resolution (resolve_param)
- Data standardization (standardize_data, safe_mean_std)
- Config access helpers (get_clock_frequency, get_series_ids, get_frequencies_from_config, etc.)
- Validation helpers (_validate_config_loaded, _validate_data_loaded, _validate_result_loaded)
"""

import numpy as np
import sys
import logging
from typing import Optional, Any, List, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime

from ..config import DFMConfig
from ..core.results import DFMParams
from ..core.time import TimeIndex

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


def safe_mean_std(data: np.ndarray, axis: int = 0, ddof: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    DFMConfigError
        If config is None or invalid
    """
    from .helpers import DFMConfigError
    
    if config is None:
        raise DFMConfigError("Configuration not loaded. Call load_config() first.")
    if not isinstance(config, DFMConfig):
        raise DFMConfigError(f"Invalid config type: {type(config)}. Expected DFMConfig.")


def _validate_data_loaded(data: Optional[np.ndarray]) -> None:
    """Validate that data is loaded.
    
    Parameters
    ----------
    data : np.ndarray, optional
        Data array
        
    Raises
    ------
    DFMDataError
        If data is None or invalid
    """
    from .helpers import DFMDataError
    
    if data is None:
        raise DFMDataError("Data not loaded. Call load_data() first.")
    if not isinstance(data, np.ndarray):
        raise DFMDataError(f"Invalid data type: {type(data)}. Expected np.ndarray.")


def _validate_result_loaded(result: Optional[Any]) -> None:
    """Validate that result is available.
    
    Parameters
    ----------
    result : Any, optional
        Result object
        
    Raises
    ------
    DFMEstimationError
        If result is None
    """
    from .helpers import DFMEstimationError
    
    if result is None:
        raise DFMEstimationError("Model not trained. Call train() or fit() first.")


# ============================================================================
# Exception classes (merged from exceptions.py)
# ============================================================================
"""Exception classes for DFM package.

This module provides specific exception types for better error handling
and clearer error messages throughout the package.
"""


class DFMError(Exception):
    """Base exception class for all DFM-related errors."""
    pass


class DFMConfigError(DFMError):
    """Exception raised for configuration-related errors."""
    pass


class DFMDataError(DFMError):
    """Exception raised for data-related errors."""
    pass


class DFMEstimationError(DFMError):
    """Exception raised during model estimation."""
    pass


class DFMValidationError(DFMError):
    """Exception raised for validation failures."""
    pass


class DFMImportError(DFMError, ImportError):
    """Exception raised when required dependencies are missing."""
    pass



# ============================================================================
# Logging utilities (merged from logging_utils.py)
# ============================================================================
"""Logging utilities for DFM package.

This module provides standardized logging configuration and utilities
for consistent logging across the package.
"""



def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    This is the standard way to get a logger in the DFM package.
    All modules should use: _logger = get_logger(__name__)
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Logger instance configured for the package
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Use package-level logger configuration
        package_logger = logging.getLogger('dfm_python')
        if not package_logger.handlers:
            # Configure root logger for dfm_python package
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            package_logger.addHandler(handler)
            package_logger.setLevel(logging.INFO)
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """Set up logging configuration for the package.
    
    This is an alias for configure_logging for backward compatibility.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
    configure_logging(level=level, format_string=format_string)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """Configure package-wide logging.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure package logger
    logger = logging.getLogger('dfm_python')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)



# ============================================================================
# Parameter resolution (merged from parameter_resolver.py)
# ============================================================================
"""Parameter resolution utilities.

This module provides a centralized ParameterResolver class to eliminate
duplicate parameter resolution logic across the codebase.
"""



class ParameterResolver:
    """Centralized parameter resolution for DFM estimation.
    
    This class provides a consistent interface for resolving parameters
    from multiple sources (overrides, config, defaults) with proper
    priority handling.
    
    Priority order: override > config_value > default
    """
    
    def __init__(self, config: DFMConfig, params: Optional[DFMParams] = None):
        """Initialize parameter resolver.
        
        Parameters
        ----------
        config : DFMConfig
            Configuration object
        params : DFMParams, optional
            Parameter overrides. If None, uses empty DFMParams().
        """
        self.config = config
        self.params = params if params is not None else DFMParams()
    
    def resolve(
        self,
        param_name: str,
        default: Any = None,
        config_attr: Optional[str] = None
    ) -> Any:
        """Resolve a single parameter.
        
        Parameters
        ----------
        param_name : str
            Name of parameter in params object (e.g., 'ar_lag', 'threshold')
        default : Any, optional
            Default value if not found in params or config
        config_attr : str, optional
            Name of attribute in config object. If None, uses param_name.
            
        Returns
        -------
        Any
            Resolved parameter value
        """
        if config_attr is None:
            config_attr = param_name
        
        # Get override value from params
        override = getattr(self.params, param_name, None)
        
        # Get config value
        config_value = getattr(self.config, config_attr, None)
        
        # Resolve using standard priority
        return resolve_param(override, config_value, default)
    
    def resolve_all(self, param_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve multiple parameters at once.
        
        Parameters
        ----------
        param_specs : dict
            Dictionary mapping parameter names to their specifications.
            Each specification is a dict with:
            - 'default': default value (required)
            - 'config_attr': config attribute name (optional, defaults to param_name)
            
        Returns
        -------
        dict
            Dictionary of resolved parameters
            
        Examples
        --------
        >>> resolver = ParameterResolver(config, params)
        >>> resolved = resolver.resolve_all({
        ...     'threshold': {'default': 1e-4},
        ...     'max_iter': {'default': 5000},
        ...     'ar_lag': {'default': 1, 'config_attr': 'ar_lag'},
        ... })
        """
        result = {}
        for param_name, spec in param_specs.items():
            default = spec.get('default')
            config_attr = spec.get('config_attr', param_name)
            result[param_name] = self.resolve(param_name, default, config_attr)
        return result
    
    def resolve_estimation_params(self) -> Dict[str, Any]:
        """Resolve all standard estimation parameters.
        
        Returns
        -------
        dict
            Dictionary containing all resolved estimation parameters
        """
        return self.resolve_all({
            'p': {'default': 1, 'config_attr': 'ar_lag'},
            'nan_method': {'default': 2, 'config_attr': 'nan_method'},
            'nan_k': {'default': 3, 'config_attr': 'nan_k'},
            'threshold': {'default': 1e-4, 'config_attr': 'threshold'},
            'max_iter': {'default': 5000, 'config_attr': 'max_iter'},
            'clock': {'default': 'm', 'config_attr': 'clock'},
            'clip_ar_coefficients': {'default': True, 'config_attr': 'clip_ar_coefficients'},
            'ar_clip_min': {'default': -0.99, 'config_attr': 'ar_clip_min'},
            'ar_clip_max': {'default': 0.99, 'config_attr': 'ar_clip_max'},
            'clip_data_values': {'default': False, 'config_attr': 'clip_data_values'},
            'data_clip_threshold': {'default': 100.0, 'config_attr': 'data_clip_threshold'},
            'use_regularization': {'default': False, 'config_attr': 'use_regularization'},
            'regularization_scale': {'default': 1e-6, 'config_attr': 'regularization_scale'},
            'min_eigenvalue': {'default': 1e-8, 'config_attr': 'min_eigenvalue'},
            'max_eigenvalue': {'default': 1e8, 'config_attr': 'max_eigenvalue'},
            'use_damped_updates': {'default': False, 'config_attr': 'use_damped_updates'},
            'damping_factor': {'default': 0.5, 'config_attr': 'damping_factor'},
        })



# Additional helper functions are in their respective modules
# (nowcasting helpers in nowcast_utils.py, loader helpers in loader.py)


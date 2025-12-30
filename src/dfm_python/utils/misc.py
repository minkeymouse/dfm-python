"""Miscellaneous utilities for DFM operations.

This module combines:
- Helper functions (parameter resolution, config access)
- Validation utilities
- Exception classes
- Parameter resolution utilities
"""

from typing import Optional, Any, List, Union, Tuple, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch
    from ..config.schema import DFMConfig, DFMResult, FitParams
    from ..datamodule import DFMDataModule
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

_logger = get_logger(__name__)


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


def get_clock_frequency(config: Optional["DFMConfig"], default: Optional[str] = None) -> str:
    """Get clock frequency from config.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    default : str, optional
        Default clock frequency if not found. If None, uses DEFAULT_CLOCK_FREQUENCY.
        
    Returns
    -------
    str
        Clock frequency code
    """
    from ..config.constants import DEFAULT_CLOCK_FREQUENCY
    return getattr(config, 'clock', default or DEFAULT_CLOCK_FREQUENCY) if config else (default or DEFAULT_CLOCK_FREQUENCY)


# Removed redundant wrapper functions:
# - get_series_ids() -> use config.get_series_ids() directly
# - get_frequencies() -> use config.get_frequencies() directly  
# - get_series_id() -> use config.get_series_id() directly
# These were just thin wrappers that added no value


def check_finite_array(
    arr: np.ndarray,
    name: str = "array",
    context: Optional[str] = None,
    fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    """Check if numpy array contains only finite values, with fallback.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str, default "array"
        Name for error messages
    context : str, optional
        Additional context for error messages
    fallback : np.ndarray, optional
        Fallback array to use if check fails
        
    Returns
    -------
    np.ndarray
        Original array if finite, fallback if provided and check fails
        
    Raises
    ------
    ValueError
        If array contains non-finite values and no fallback provided
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(~np.isfinite(arr))
        context_str = f" in {context}" if context else ""
        msg = f"{name}{context_str} contains {nan_count} non-finite values"
        
        if fallback is not None:
            _logger.warning(f"{msg}. Using fallback array")
            return fallback
        else:
            _logger.error(msg)
            raise ValueError(msg)
    
    return arr




# ============================================================================
# Exception classes (merged from exceptions.py)
# ============================================================================
"""Exception classes for DFM package.

This module provides specific exception types for better error handling
and clearer error messages throughout the package.
"""


# DFMError moved to utils.errors - import from there instead
# Removed exception aliases - use DFMError directly or proper exceptions from utils.errors
# Use:
# - DFMError (base exception) from utils.errors
# - ConfigurationError, DataError, NumericalError, etc. from utils.errors
from .errors import DFMError


# ParameterResolver removed - overengineered and unused
# Use resolve_param() function directly instead








# Preprocessing functions moved to dataset.process
# Re-export for backward compatibility (only commonly used functions)
from ..dataset.process import (
    _check_sklearn,
    _get_scaler,
    _get_scaler_attr,
    _normalize_wx,
    TimeIndex,
)
# Note: _get_mean and _get_scale are internal utilities, not re-exported
from ..config.constants import DEFAULT_DAMPING_FACTOR



# Time utilities moved to dataset.process
# Re-export parse_timestamp for backward compatibility
from ..dataset.process import parse_timestamp

# Metric functions moved to metric.py
# Re-export for backward compatibility
from .metric import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
)




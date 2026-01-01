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
from .errors import NumericalError

_logger = get_logger(__name__)


def resolve_param(
    override: Optional[Any] = None,
    config_value: Optional[Any] = None,
    default: Any = None,
    *,
    name: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Any:
    """Resolve parameter value from multiple sources with priority.
    
    Supports two calling patterns:
    1. Legacy: resolve_param(override, config_value, default)
    2. New: resolve_param(name=name, kwargs=kwargs, config=config, defaults=defaults)
    
    Priority (new pattern): kwargs[name] > config.name > defaults[name] > None
    Priority (legacy pattern): override > config_value > default
    
    Parameters
    ----------
    override : Any, optional
        Parameter override value (highest priority in legacy mode)
    config_value : Any, optional
        Configuration value (medium priority in legacy mode)
    default : Any, optional
        Default value (lowest priority in legacy mode)
    name : str, optional
        Parameter name (for new pattern)
    kwargs : Dict[str, Any], optional
        Keyword arguments dict (for new pattern)
    config : Any, optional
        Configuration object with attributes (for new pattern)
    defaults : Dict[str, Any], optional
        Default values dictionary (for new pattern)
        
    Returns
    -------
    Any
        Resolved parameter value
    """
    # New pattern: extract by name from multiple sources
    if name is not None:
        if kwargs is not None and name in kwargs:
            return kwargs.pop(name) if isinstance(kwargs, dict) else kwargs.get(name)
        if config is not None and hasattr(config, name):
            return getattr(config, name)
        if defaults is not None and name in defaults:
            return defaults[name]
        return None
    
    # Legacy pattern: override > config_value > default
    if override is not None:
        return override
    if config_value is not None:
        return config_value
    return default


def extract_param(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None,
    default: Any = None
) -> Any:
    """Extract parameter with fallback chain: kwargs > config > defaults > default.
    
    Parameters
    ----------
    name : str
        Parameter name
    kwargs : Dict[str, Any], optional
        Keyword arguments dict
    config : Any, optional
        Configuration object with attributes
    defaults : Dict[str, Any], optional
        Default values dictionary
    default : Any, optional
        Final fallback value
        
    Returns
    -------
    Any
        Extracted parameter value
    """
    value = resolve_param(name=name, kwargs=kwargs, config=config, defaults=defaults)
    return value if value is not None else default


def extract_bool_param(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None,
    default: bool = False
) -> bool:
    """Extract boolean parameter with fallback chain.
    
    Parameters
    ----------
    name : str
        Parameter name
    kwargs : Dict[str, Any], optional
        Keyword arguments dict
    config : Any, optional
        Configuration object with attributes
    defaults : Dict[str, Any], optional
        Default values dictionary
    default : bool, default False
        Final fallback value
        
    Returns
    -------
    bool
        Extracted boolean parameter value
    """
    value = resolve_param(name=name, kwargs=kwargs, config=config, defaults=defaults)
    return value if value is not None else default


def extract_opt_param(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Extract optional parameter with fallback chain.
    
    Parameters
    ----------
    name : str
        Parameter name
    kwargs : Dict[str, Any], optional
        Keyword arguments dict
    config : Any, optional
        Configuration object with attributes
    defaults : Dict[str, Any], optional
        Default values dictionary
        
    Returns
    -------
    Optional[Any]
        Extracted parameter value, or None if not found
    """
    return resolve_param(name=name, kwargs=kwargs, config=config, defaults=defaults)


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
    from ..utils.helper import get_config_attr
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
    datamodule: Optional[Any],
    series_ids: Optional[List[str]] = None,
    result: Optional[Any] = None,
    model_name: str = "model"
) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Resolve target series from DataModule.
    
    This utility function resolves target series from the DataModule's target_series attribute
    and maps them to indices in the series_ids list.
    
    Parameters
    ----------
    datamodule : Any, optional
        DataModule instance with target_series attribute
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
    
    # Get target series from DataModule
    target_series = None
    if datamodule is not None:
        target_series = getattr(datamodule, 'target_series', None)
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








# Preprocessing functions moved to dataset.process
# Re-export for backward compatibility (only commonly used functions)
from ..dataset.process import (
    _check_sklearn,
    _get_scaler,
    TimeIndex,
)



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

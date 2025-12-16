"""Helper functions for DFM operations.

This module provides utility functions for:
- Safe configuration access (safe_get_attr, safe_get_method)
- Parameter resolution (resolve_param)
- Config access helpers (get_clock_frequency, get_series_ids, get_frequencies, etc.)
- Validation helpers (_validate_config, _validate_data, _validate_result)
"""

import numpy as np
from typing import Optional, Any, List, Union, Tuple, Dict

from ..config.schema import DFMConfig
from ..config.results import FitParams
from ..logger import get_logger

_logger = get_logger(__name__)


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
    return getattr(config, 'clock', default) if config is not None else default


def get_series_ids(config: DFMConfig) -> List[str]:
    """Get list of series IDs from config.
    
    This is a convenience wrapper around config.get_series_ids().
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of series IDs
    """
    if config is None:
        return []
    return config.get_series_ids()


def get_series_names(config: DFMConfig) -> List[str]:
    """Get list of series names from config.
    
    This is a convenience wrapper around config.get_series_names().
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of series names
    """
    if config is None:
        return []
    return config.get_series_names()


def get_frequencies(config: DFMConfig) -> List[str]:
    """Get list of frequencies from config.
    
    This is a convenience wrapper around config.get_frequencies().
    
    Parameters
    ----------
    config : DFMConfig
        Configuration object
        
    Returns
    -------
    List[str]
        List of frequency codes for each series
    """
    if config is None:
        return []
    return config.get_frequencies()


def get_series_id(config: DFMConfig, index: int) -> Optional[str]:
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


def _validate_config(config: Optional[DFMConfig]) -> None:
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


def _validate_data(data_module: Optional[Any]) -> None:
    """Validate that DataModule is provided and set up.
    
    Parameters
    ----------
    data_module : DFMDataModule, optional
        DataModule instance
        
    Raises
    ------
    DFMDataError
        If data_module is None or not set up
    """
    from ..lightning import DFMDataModule
    
    if data_module is None:
        raise DFMDataError("DataModule not provided. Provide a DFMDataModule instance via train().")
    if not isinstance(data_module, DFMDataModule):
        raise DFMDataError(f"Invalid data_module type: {type(data_module)}. Expected DFMDataModule.")
    if data_module.data_processed is None:
        raise DFMDataError("DataModule not set up. Call data_module.setup() first.")


def _validate_result(result: Optional[Any]) -> None:
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
    # DFMEstimationError is defined in this module below
    
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
    
    def __init__(self, config: DFMConfig, params: Optional[FitParams] = None):
        """Initialize parameter resolver.
        
        Parameters
        ----------
        config : DFMConfig
            Configuration object
        params : FitParams, optional
            Parameter overrides. If None, uses empty FitParams().
        """
        self.config = config
        self.params = params if params is not None else FitParams()
    
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





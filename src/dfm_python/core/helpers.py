"""Helper functions for common patterns across the package.

This module provides utility functions for common code patterns to reduce
duplication and improve maintainability.
"""

from typing import Optional, Any
from ..config import DFMConfig


def safe_get_method(config: Optional[DFMConfig], method_name: str, default: Any = None) -> Any:
    """Safely get a method from config if it exists and is callable.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    method_name : str
        Name of the method to retrieve
    default : Any
        Default value if method doesn't exist or isn't callable
        
    Returns
    -------
    Any
        Method result if callable, else default value
        
    Notes
    -----
    - Use this for config methods that need to be called (e.g., `get_series_ids()`)
    - Returns default if method doesn't exist or isn't callable
    - Used throughout package for safe config method access
    - Prefer `safe_get_attr()` for simple attribute access (no method call needed)
    """
    if config is None:
        return default
    method = getattr(config, method_name, None)
    if method is not None and callable(method):
        return method()
    return default


def safe_get_attr(config: Optional[DFMConfig], attr_name: str, default: Any = None) -> Any:
    """Safely get an attribute from config if it exists.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    attr_name : str
        Name of the attribute to retrieve
    default : Any
        Default value if attribute doesn't exist
        
    Returns
    -------
    Any
        Attribute value or default
        
    Notes
    -----
    - Use this for simple attribute access (e.g., `min_eigenvalue`, `ar_clip_max`)
    - Returns default if attribute doesn't exist
    - Used throughout package for safe config attribute access
    - Prefer `safe_get_method()` for methods that need to be called
    """
    if config is None:
        return default
    return getattr(config, attr_name, default)


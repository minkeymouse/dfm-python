"""Common helper functions for error handling, config access, and validation.

This module provides reusable helpers to reduce code duplication and improve
consistency across the codebase.
"""

from typing import Any, Optional, Callable, Type, Union, NoReturn
import numpy as np

from ..logger import get_logger
from ..utils.errors import NumericalError, ConfigValidationError
from ..utils.misc import resolve_param

_logger = get_logger(__name__)


def handle_linear_algebra_error(
    operation: Callable,
    operation_name: str,
    fallback_value: Optional[Any] = None,
    fallback_func: Optional[Callable] = None,
    *args,
    **kwargs
) -> Any:
    """Handle linear algebra errors with fallback.
    
    This helper consolidates the common pattern of catching
    (np.linalg.LinAlgError, ValueError) and providing a fallback.
    
    Parameters
    ----------
    operation : Callable
        Function to execute that may raise LinAlgError or ValueError
    operation_name : str
        Name of operation (for logging)
    fallback_value : Any, optional
        Value to return if operation fails
    fallback_func : Callable, optional
        Function to call if operation fails (takes *args, **kwargs)
    *args
        Positional arguments to pass to operation
    **kwargs
        Keyword arguments to pass to operation
        
    Returns
    -------
    Any
        Result of operation if successful, otherwise fallback_value or
        result of fallback_func
        
    Examples
    --------
    >>> # With fallback value
    >>> A = handle_linear_algebra_error(
    ...     np.linalg.solve, "matrix solve",
    ...     fallback_value=np.eye(3),
    ...     X, y
    ... )
    
    >>> # With fallback function
    >>> A = handle_linear_algebra_error(
    ...     np.linalg.solve, "matrix solve",
    ...     fallback_func=lambda: create_scaled_identity(3, 0.5),
    ...     X, y
    ... )
    """
    try:
        return operation(*args, **kwargs)
    except (np.linalg.LinAlgError, ValueError) as e:
        _logger.warning(
            f"{operation_name} failed ({type(e).__name__}): {e}. Using fallback."
        )
        if fallback_func is not None:
            return fallback_func(*args, **kwargs)
        elif fallback_value is not None:
            return fallback_value
        else:
            raise


def get_config_attr(
    config: Optional[Any],
    attr_name: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """Get configuration attribute with fallback and validation.
    
    This helper standardizes config attribute access, replacing
    getattr(config, 'attr', default) patterns throughout the codebase.
    
    Parameters
    ----------
    config : Any, optional
        Configuration object
    attr_name : str
        Attribute name to access
    default : Any, optional
        Default value if attribute not found
    required : bool, default False
        If True, raise AttributeError if attribute not found
        
    Returns
    -------
    Any
        Attribute value, default value, or None
        
    Raises
    ------
    AttributeError
        If required=True and attribute not found
        
    Examples
    --------
    >>> # Basic usage
    >>> clip_enabled = get_config_attr(config, 'clip_ar_coefficients', True)
    
    >>> # Required attribute
    >>> clock = get_config_attr(config, 'clock', required=True)
    """
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


def validate_finite_array(
    arr: np.ndarray,
    name: str = "array",
    context: Optional[str] = None,
    error_class: Type[Exception] = NumericalError,
    fallback: Optional[np.ndarray] = None
) -> np.ndarray:
    """Validate that array contains only finite values, with optional fallback.
    
    This helper standardizes finite array checks, replacing manual
    np.any(~np.isfinite()) patterns throughout the codebase.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    name : str, default "array"
        Name for error messages
    context : str, optional
        Additional context for error messages
    error_class : Type[Exception], default NumericalError
        Exception class to raise if validation fails
    fallback : np.ndarray, optional
        Fallback array to return if validation fails (instead of raising)
        
    Returns
    -------
    np.ndarray
        Original array if finite, or fallback if provided and check fails
        
    Raises
    ------
    error_class
        If array contains non-finite values and no fallback provided
        
    Examples
    --------
    >>> # Basic usage (raises on failure)
    >>> validate_finite_array(Z_forecast, "factor forecast")
    
    >>> # With context
    >>> validate_finite_array(X_forecast, "forecast", context="DDFM prediction")
    
    >>> # With fallback (returns fallback on failure)
    >>> arr = validate_finite_array(data, "data", fallback=default_data)
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(~np.isfinite(arr))
        context_str = f" in {context}" if context else ""
        msg = f"{name}{context_str} contains {nan_count} non-finite values"
        
        if fallback is not None:
            from ..logger import get_logger
            _logger = get_logger(__name__)
            _logger.warning(f"{msg}. Using fallback array")
            return fallback
        else:
            raise error_class(
                msg,
                details="This indicates numerical instability. Please check model parameters and training convergence."
            )
    
    return arr


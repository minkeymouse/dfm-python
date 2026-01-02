"""Common helper functions for error handling, config access, and validation.

This module provides reusable helpers to reduce code duplication and improve
consistency across the codebase.
"""

from typing import Any, Optional, Callable, Type, Union, NoReturn
import numpy as np

from ..logger import get_logger
from ..utils.errors import NumericalError

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


# get_config_attr moved to utils.misc for consolidation with other config utilities

def validate_finite_array(
    arr: np.ndarray,
    name: str = "array",
    context: Optional[str] = None,
    error_class: Type[Exception] = NumericalError,
    fallback: Optional[np.ndarray] = None,
    sanitize: bool = False,
    nan_value: float = None,
    inf_value: float = None
) -> np.ndarray:
    """Validate that array contains only finite values, with optional fallback or sanitization.
    
    This helper standardizes finite array checks, replacing manual
    np.any(~np.isfinite()) patterns throughout the codebase.
    
    Can optionally sanitize the array (replace NaN/Inf) instead of raising,
    or use a fallback array.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    name : str, default "array"
        Name for error messages
    context : str, optional
        Additional context for error messages
    error_class : Type[Exception], default NumericalError
        Exception class to raise if validation fails (only if sanitize=False and fallback=None)
    fallback : np.ndarray, optional
        Fallback array to return if validation fails (instead of raising or sanitizing)
    sanitize : bool, default False
        If True, sanitize the array instead of raising (uses sanitize_array)
    nan_value : float, optional
        Value to replace NaN with if sanitize=True. If None, uses DEFAULT_ZERO_VALUE.
    inf_value : float, optional
        Value to replace Inf with if sanitize=True. If None, uses MAX_EIGENVALUE.
        
    Returns
    -------
    np.ndarray
        Original array if finite, sanitized array if sanitize=True, or fallback if provided
        
    Raises
    ------
    error_class
        If array contains non-finite values and no fallback/sanitization provided
        
    Examples
    --------
    >>> # Basic usage (raises on failure)
    >>> validate_finite_array(Z_forecast, "factor forecast")
    
    >>> # With context
    >>> validate_finite_array(X_forecast, "forecast", context="DDFM prediction")
    
    >>> # With fallback (returns fallback on failure)
    >>> arr = validate_finite_array(data, "data", fallback=default_data)
    
    >>> # With sanitization (replaces NaN/Inf instead of raising)
    >>> arr = validate_finite_array(data, "data", sanitize=True)
    """
    if np.all(np.isfinite(arr)):
        return arr
    
    nan_count = np.sum(~np.isfinite(arr))
    context_str = f" in {context}" if context else ""
    msg = f"{name}{context_str} contains {nan_count} non-finite values"
    
    if fallback is not None:
        _logger.warning(f"{msg}. Using fallback array")
        return fallback
    elif sanitize:
        from ..utils.common import sanitize_array
        from ..config.constants import DEFAULT_ZERO_VALUE, MAX_EIGENVALUE
        _logger.warning(f"{msg}. Sanitizing array")
        return sanitize_array(
            arr,
            nan_value=nan_value if nan_value is not None else DEFAULT_ZERO_VALUE,
            inf_value=inf_value if inf_value is not None else MAX_EIGENVALUE
        )
    else:
        raise error_class(
            msg,
            details="This indicates numerical instability. Please check model parameters and training convergence."
        )


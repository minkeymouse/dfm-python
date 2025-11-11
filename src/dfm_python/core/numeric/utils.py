"""General utility functions for numerical operations."""

import logging
import numpy as np

_logger = logging.getLogger(__name__)


def _check_finite(array: np.ndarray, name: str = "array", raise_on_invalid: bool = False) -> bool:
    """Check if array contains only finite values.
    
    Parameters
    ----------
    array : np.ndarray
        Array to check
    name : str
        Name of array for logging
    raise_on_invalid : bool
        If True, raise ValueError on invalid values. If False, only log warning.
        
    Returns
    -------
    bool
        True if all values are finite, False otherwise
        
    Raises
    ------
    ValueError
        If raise_on_invalid=True and array contains non-finite values
        
    Notes
    -----
    - Checks for both NaN and Inf values
    - Provides detailed error messages with counts of NaN/Inf values
    - Used throughout the package for input validation and debugging
    - When raise_on_invalid=False, logs warnings but allows execution to continue
    """
    has_nan = np.any(np.isnan(array))
    has_inf = np.any(np.isinf(array))
    
    if has_nan or has_inf:
        msg = f"{name} contains "
        issues = []
        if has_nan:
            issues.append(f"{np.sum(np.isnan(array))} NaN values")
        if has_inf:
            issues.append(f"{np.sum(np.isinf(array))} Inf values")
        msg += " and ".join(issues)
        
        if raise_on_invalid:
            raise ValueError(msg)
        else:
            _logger.warning(msg)
        return False
    return True


def _safe_divide(numerator: np.ndarray, denominator: float, default: float = 0.0) -> np.ndarray:
    """Safely divide numerator by denominator, handling zero and invalid values.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : float
        Denominator value
    default : float, default 0.0
        Default value to use if denominator is zero, NaN, or Inf
        
    Returns
    -------
    np.ndarray
        Result of division, with invalid results replaced by default value
        
    Notes
    -----
    - Replaces non-finite results (NaN, Inf) with default value
    - Useful for avoiding division by zero errors in numerical computations
    - Preserves array shape and dtype of numerator
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.full_like(numerator, default)
    result = numerator / denominator
    result = np.where(np.isfinite(result), result, default)
    return result

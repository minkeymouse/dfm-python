"""AR coefficient clipping functions."""

from typing import Optional, Tuple, Dict, Any
import logging
import numpy as np

_logger = logging.getLogger(__name__)


def _clip_ar_coefficients(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
                         warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds.
    
    Clips transition matrix (AR) coefficients to ensure stability of the
    factor dynamics. Coefficients outside [min_val, max_val] are clipped
    to the nearest bound.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix containing AR coefficients (any shape)
    min_val : float, default -0.99
        Minimum allowed AR coefficient (lower bound for clipping)
    max_val : float, default 0.99
        Maximum allowed AR coefficient (upper bound for clipping)
    warn : bool, default True
        If True, log warning when clipping is applied
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped transition matrix with all values in [min_val, max_val]
    stats : dict
        Statistics dictionary with:
        - 'n_clipped': Number of coefficients that were clipped
        - 'n_total': Total number of coefficients
        - 'clipped_indices': List of flattened indices that were clipped
        - 'min_violations': Number of values below min_val
        - 'max_violations': Number of values above max_val
        
    Notes
    -----
    - Default bounds [-0.99, 0.99] ensure factor dynamics remain stable
    - Used in EM step to prevent explosive or oscillatory factor behavior
    - Clipping preserves matrix structure (only values are modified)
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def _apply_ar_clipping(A: np.ndarray, config: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration.
    
    This is a convenience wrapper around `_clip_ar_coefficients()` that reads
    clipping parameters from a configuration object. If clipping is disabled
    in config, returns the matrix unchanged.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix to clip (any shape)
    config : object, optional
        Configuration object with clipping parameters. If None, uses defaults.
        Expected attributes:
        - clip_ar_coefficients: bool, whether clipping is enabled (default: True)
        - ar_clip_min: float, minimum AR coefficient (default: -0.99)
        - ar_clip_max: float, maximum AR coefficient (default: 0.99)
        - warn_on_ar_clip: bool, whether to log warnings (default: True)
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped transition matrix (unchanged if clipping disabled)
    stats : dict
        Statistics about clipping operation (same format as _clip_ar_coefficients)
        
    Notes
    -----
    - Wrapper function that delegates to `_clip_ar_coefficients()` after reading config
    - If config is None, uses default bounds [-0.99, 0.99]
    - If clipping is disabled in config, returns A unchanged with empty stats
    - Used in EM step to apply configurable AR coefficient constraints
    - Default bounds ensure factor dynamics remain stable (prevent explosive behavior)
    """
    if config is None:
        return _clip_ar_coefficients(A, -0.99, 0.99, True)
    
    from ..helpers import safe_get_attr
    
    clip_enabled = safe_get_attr(config, 'clip_ar_coefficients', True)
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = safe_get_attr(config, 'ar_clip_min', -0.99)
    max_val = safe_get_attr(config, 'ar_clip_max', 0.99)
    warn = safe_get_attr(config, 'warn_on_ar_clip', True)
    return _clip_ar_coefficients(A, min_val, max_val, warn)

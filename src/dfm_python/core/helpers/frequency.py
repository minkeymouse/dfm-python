"""Frequency-related helpers for tent weights and frequency inference."""

from typing import Optional, Dict, Any
import numpy as np
import logging
from ...utils.aggregation import (
    FREQUENCY_HIERARCHY,
    get_tent_weights_for_pair,
    generate_tent_weights,
)

_logger = logging.getLogger(__name__)


def get_tent_weights(
    freq: str,
    clock: str,
    tent_weights_dict: Optional[Dict[str, np.ndarray]],
    logger: Optional[Any] = None
) -> Optional[np.ndarray]:
    """Safely get tent weights for a frequency pair with fallback generation.
    
    Parameters
    ----------
    freq : str
        Target frequency ('q', 'sa', 'a', etc.)
    clock : str
        Clock frequency ('m', 'q', etc.)
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights
    logger : logging.Logger, optional
        Logger instance for warnings. If None, uses module logger.
        
    Returns
    -------
    tent_weights : np.ndarray, optional
        Tent weights array, or None if cannot be determined
        
    Notes
    -----
    - Checks dictionary first, then generates symmetric weights if needed
    - Uses FREQUENCY_HIERARCHY to determine number of periods
    - Raises ValueError if frequency pair cannot be handled
    """
    if logger is None:
        logger = _logger
    
    # Try dictionary first
    if tent_weights_dict and freq in tent_weights_dict:
        return tent_weights_dict[freq]
    
    # Try helper function
    tent_weights = get_tent_weights_for_pair(freq, clock)
    if tent_weights is not None:
        return tent_weights
    
    # Generate symmetric weights as fallback
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    freq_h = FREQUENCY_HIERARCHY.get(freq, 3)
    n_periods_est = freq_h - clock_h + 1
    
    if 0 < n_periods_est <= 12:
        tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
        logger.warning(f"get_tent_weights: generated symmetric tent weights for '{freq}'")
        return tent_weights
    else:
        raise ValueError(f"get_tent_weights: cannot determine tent weights for '{freq}'")


def infer_nQ(
    frequencies: Optional[np.ndarray],
    clock: str
) -> int:
    """Infer number of slower-frequency series from frequencies array.
    
    Parameters
    ----------
    frequencies : np.ndarray, optional
        Array of frequency strings (n,) for each series
    clock : str
        Clock frequency ('m', 'q', 'sa', 'a')
        
    Returns
    -------
    nQ : int
        Number of series with frequency slower than clock frequency
        
    Notes
    -----
    - Returns 0 if frequencies is None
    - Uses FREQUENCY_HIERARCHY to compare frequencies
    - Clock frequency is typically 'm' (monthly)
    """
    if frequencies is None:
        return 0
    
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    return sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_h)


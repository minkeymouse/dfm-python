"""EM convergence checking routines."""

from typing import Tuple
import logging
import numpy as np

_logger = logging.getLogger(__name__)

# Convergence and stability constants
MIN_LOG_LIKELIHOOD_DELTA = -1e-3  # Threshold for detecting likelihood decrease
DAMPING = 0.95  # Damping factor when numerical errors occur
MAX_LOADING_REPLACE = 0.99  # Replacement for Inf in loadings


def em_converged(
    loglik: float,
    previous_loglik: float,
    threshold: float = 1e-4,
    check_decreased: bool = True
) -> Tuple[bool, bool]:
    """Check whether EM algorithm has converged.
    
    Convergence is determined by relative change in log-likelihood:
    |loglik - previous_loglik| / avg(|loglik|, |previous_loglik|) < threshold
    
    Parameters
    ----------
    loglik : float
        Current iteration log-likelihood value.
    previous_loglik : float
        Previous iteration log-likelihood value.
    threshold : float, default 1e-4
        Convergence threshold for relative change in log-likelihood.
        Algorithm converges when relative change falls below this value.
    check_decreased : bool, default True
        If True, check for likelihood decreases and log warning.
        Useful for detecting numerical issues or convergence problems.
        
    Returns
    -------
    converged : bool
        True if algorithm has converged (relative change < threshold).
    decreased : bool
        True if likelihood decreased significantly (only if check_decreased=True).
        A decrease may indicate numerical issues or convergence problems.
    
    Notes
    -----
    - Convergence criterion matches MATLAB implementation (Nowcasting/functions/dfm.m).
    - Formula from Numerical Recipes in C (pg. 423).
    - Default threshold (1e-4) matches MATLAB default.
    - Relative change formula: delta / avg(|loglik|, |previous_loglik|).
    
    Examples
    --------
    >>> loglik_prev = -1000.0
    >>> loglik_curr = -1000.01
    >>> converged, decreased = em_converged(loglik_curr, loglik_prev)
    >>> assert converged == True  # Small relative change
    """
    converged = False
    decrease = False
    
    if check_decreased and (loglik - previous_loglik) < MIN_LOG_LIKELIHOOD_DELTA:
        _logger.warning(f"Likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}")
        decrease = True
    
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease

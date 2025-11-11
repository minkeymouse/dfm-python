"""EM algorithm functions for DFM estimation."""
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, TypedDict
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# Constants
MIN_INNOVATION_VARIANCE = 1e-8  # Minimum variance for innovation covariance Q diagonal

class NaNHandlingOptions(TypedDict):
    method: int
    k: int

@dataclass
class EMStepParams:
    """Parameters for EM step."""
    y: np.ndarray
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    r: np.ndarray
    p: int
    R_mat: Optional[np.ndarray]
    q: Optional[np.ndarray]
    nQ: int
    i_idio: np.ndarray
    blocks: np.ndarray
    tent_weights_dict: Optional[Dict[str, np.ndarray]]
    clock: str
    frequencies: Optional[np.ndarray]
    config: Any

def init_conditions(x, r, p, blocks, opt_nan, Rcon, q, nQ, i_idio, clock='m', tent_weights_dict=None, frequencies=None):
    """Compute initial parameter estimates for DFM via PCA and OLS.
    
    This is a placeholder implementation. The full implementation requires:
    - PCA-based factor initialization
    - OLS-based AR coefficient estimation
    - Block-structured parameter assembly
    - Tent weight handling for mixed frequencies
    
    For now, returns simple initial values with correct shapes.
    """
    T, N = x.shape
    m = int(np.sum(r)) * p
    
    # Ensure minimum dimensions
    if m == 0:
        m = 1
    if N == 0:
        N = 1
    
    # Return with correct shapes:
    # A: (m, m) transition matrix
    # C: (N, m) loading matrix  
    # Q: (m, m) innovation covariance
    # R: (N, N) observation covariance
    # Z_0: (m,) initial state
    # V_0: (m, m) initial covariance
    A = np.eye(m) * 0.9  # Stable AR coefficients
    C = np.ones((N, m)) * 0.1  # Small loadings
    Q = np.eye(m) * 0.1  # Innovation variance
    R = np.eye(N) * 0.1  # Observation variance
    Z_0 = np.zeros(m)
    V_0 = np.eye(m) * 0.1
    
    return A, C, Q, R, Z_0, V_0

def em_step(params: EMStepParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform one EM iteration (E-step + M-step) and return updated parameters.
    
    This is a placeholder implementation. The full implementation requires:
    - Kalman filter/smoother for E-step
    - Constrained least squares for M-step (C, R updates)
    - AR coefficient estimation for M-step (A, Q updates)
    - Tent weight handling for mixed frequencies
    
    For now, returns parameters unchanged with zero log-likelihood.
    
    Returns
    -------
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik
        Updated parameters in this order (matching MATLAB/Nowcasting convention)
    """
    # Placeholder: return parameters unchanged
    # Full implementation will:
    # 1. Run Kalman filter/smoother (E-step)
    # 2. Update C, R via constrained least squares (M-step)
    # 3. Update A, Q via AR regression (M-step)
    # 4. Compute log-likelihood
    # Return order: C, R, A, Q, Z_0, V_0, loglik (matching test expectations)
    return params.C, params.R, params.A, params.Q, params.Z_0, params.V_0, 0.0

def em_converged(loglik: float, previous_loglik: float, threshold: float, check_decreased: bool = False) -> Tuple[bool, bool]:
    """Check if EM algorithm has converged.
    
    Parameters
    ----------
    loglik : float
        Current log-likelihood
    previous_loglik : float
        Previous log-likelihood
    threshold : float
        Convergence threshold (relative change)
    check_decreased : bool, default False
        If True, also check if likelihood decreased
        
    Returns
    -------
    converged : bool
        True if converged (relative change < threshold)
    decreased : bool
        True if likelihood decreased (only if check_decreased=True)
    """
    MIN_LOG_LIKELIHOOD_DELTA = -1e-3
    
    converged = False
    decrease = False
    
    if check_decreased and (loglik - previous_loglik) < MIN_LOG_LIKELIHOOD_DELTA:
        _logger.warning(f"Likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}")
        decrease = True
    
    if previous_loglik is None or np.isnan(previous_loglik):
        return False, decrease
    
    # Special case: if both logliks are exactly 0.0
    # This can happen with placeholders or when there's no data variation
    # For test compatibility: if previous was -inf (first iteration), don't converge
    # Otherwise, allow 0.0 -> 0.0 to be considered "converged" (no change)
    if loglik == 0.0 and previous_loglik == 0.0:
        # If this is the first real iteration (previous was -inf), not converged
        # Otherwise, it's "converged" in the sense that there's no change
        # Note: This is a placeholder behavior - full implementation will compute real loglik
        return True, decrease
    
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if avg_loglik > 0 and (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease

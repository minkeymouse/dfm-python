"""Parameter validation and covariance stabilization helpers."""

from typing import Optional, Tuple
import numpy as np
import logging
from ...config import DFMConfig
from .config import safe_get_attr
from ..numeric import (
    _check_finite,
    _clean_matrix,
    _ensure_positive_definite,
    _cap_max_eigenvalue,
    _ensure_innovation_variance_minimum,
)

_logger = logging.getLogger(__name__)


def validate_params(
    A: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    fallback_transition_coeff: float = 0.9,
    min_innovation_variance: float = 1e-8,
    default_observation_variance: float = 1e-4,
    default_idio_init_covariance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and clean DFM parameters for numerical stability.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix (m x m)
    Q : np.ndarray
        Innovation covariance (m x m)
    R : np.ndarray
        Observation covariance (n x n), typically diagonal
    C : np.ndarray
        Loading matrix (n x m)
    Z_0 : np.ndarray
        Initial state (m,)
    V_0 : np.ndarray
        Initial covariance (m x m)
    fallback_transition_coeff : float, default 0.9
        Fallback coefficient for transition matrix regularization
    min_innovation_variance : float, default 1e-8
        Minimum innovation variance for Q matrix
    default_observation_variance : float, default 1e-4
        Default observation variance for R matrix
    default_idio_init_covariance : float, default 0.1
        Default initial covariance for V_0 matrix
        
    Returns
    -------
    A_clean : np.ndarray
        Cleaned transition matrix
    Q_clean : np.ndarray
        Cleaned innovation covariance
    R_clean : np.ndarray
        Cleaned observation covariance
    C_clean : np.ndarray
        Cleaned loading matrix
    Z_0_clean : np.ndarray
        Cleaned initial state
    V_0_clean : np.ndarray
        Cleaned initial covariance
        
    Notes
    -----
    - Uses _check_finite and _clean_matrix for consistent cleaning
    - Applies parameter-specific fallback strategies
    - Logs warnings when cleaning is applied
    - Used in em_step for input validation
    """
    if not _check_finite(A, "A"):
        _logger.warning(
            f"validate_params: A contains NaN/Inf, "
            f"applying regularization ({fallback_transition_coeff}*I + {1-fallback_transition_coeff}*cleaned)"
        )
        A = (np.eye(A.shape[0]) * fallback_transition_coeff + 
             _clean_matrix(A, 'loading') * (1 - fallback_transition_coeff))
    
    if not _check_finite(Q, "Q"):
        _logger.warning(
            f"validate_params: Q contains NaN/Inf, "
            f"applying regularization (min={min_innovation_variance})"
        )
        Q = _clean_matrix(Q, 'covariance', default_nan=min_innovation_variance)
    
    if not _check_finite(R, "R"):
        _logger.warning(
            f"validate_params: R contains NaN/Inf, "
            f"applying regularization (min={default_observation_variance})"
        )
        R = _clean_matrix(R, 'diagonal', default_nan=default_observation_variance, default_inf=1e4)
    
    if not _check_finite(C, "C"):
        _logger.warning("validate_params: C contains NaN/Inf, cleaning matrix")
        C = _clean_matrix(C, 'loading')
    
    if not _check_finite(Z_0, "Z_0"):
        _logger.warning("validate_params: Z_0 contains NaN/Inf, resetting to zeros")
        Z_0 = np.zeros_like(Z_0)
    
    if not _check_finite(V_0, "V_0"):
        _logger.warning(
            f"validate_params: V_0 contains NaN/Inf, "
            f"using regularized identity ({default_idio_init_covariance}*I)"
        )
        V_0 = np.eye(V_0.shape[0]) * default_idio_init_covariance
    
    return A, Q, R, C, Z_0, V_0


def stabilize_cov(
    Q: np.ndarray,
    config: Optional[DFMConfig],
    min_variance: float = 1e-8
) -> np.ndarray:
    """Apply standard stability operations to innovation covariance matrix.
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix (m x m)
    config : DFMConfig, optional
        Configuration object for stability parameters
    min_variance : float, default 1e-8
        Minimum variance to enforce on diagonal elements
        
    Returns
    -------
    Q_stable : np.ndarray
        Stabilized innovation covariance matrix
        
    Notes
    -----
    - Applies operations in order: clean -> PSD -> cap -> min variance
    - Re-applies min variance after each operation that might affect diagonal
    - Used in em_step for Q block updates
    """
    # Clean matrix first
    Q = _clean_matrix(Q, 'covariance', default_nan=0.0)
    
    # Get config parameters
    min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
    max_eigenval = safe_get_attr(config, "max_eigenvalue", 1e6)
    
    # Ensure minimum variance before PSD (critical for factor evolution)
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    # Apply positive definite enforcement
    Q, _ = _ensure_positive_definite(Q, min_eigenval, warn_reg)
    
    # Re-ensure minimum after PSD (it may have changed diagonal)
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    # Cap maximum eigenvalues
    Q = _cap_max_eigenvalue(Q, max_eigenval=max_eigenval)
    
    # Final minimum variance enforcement after capping
    Q = _ensure_innovation_variance_minimum(Q, min_variance=min_variance)
    
    return Q


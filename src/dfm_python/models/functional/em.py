"""EM algorithm implementation for DFM.

This module provides the Expectation-Maximization algorithm for DFM parameter estimation.
Uses pykalman for the E-step (Kalman filter/smoother) and implements the M-step
with block structure preservation.

Includes numerical stability utilities to ensure convergence safety.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

from ...ssm.kalman import DFMKalmanFilter
from ...logger import get_logger

_logger = get_logger(__name__)

# Numerical stability constants
_MIN_EIGENVAL = 1e-8
_MIN_DIAGONAL_VARIANCE = 1e-6
_INV_REGULARIZATION = 1e-6


def ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to symmetrize
        
    Returns
    -------
    np.ndarray
        Symmetric matrix
    """
    return 0.5 * (M + M.T)


def ensure_positive_definite(
    M: np.ndarray,
    min_eigenval: float = _MIN_EIGENVAL,
    warn: bool = True
) -> np.ndarray:
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue to enforce
    warn : bool
        Whether to log warnings
        
    Returns
    -------
    np.ndarray
        Positive semi-definite matrix
    """
    M = ensure_symmetric(M)
    
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    try:
        eigenvals = np.linalg.eigh(M)[0]
        min_eig = float(np.min(eigenvals))
        
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0], dtype=M.dtype) * reg_amount
            M = ensure_symmetric(M)
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal."
                )
    except (np.linalg.LinAlgError, ValueError) as e:
        M = M + np.eye(M.shape[0], dtype=M.dtype) * min_eigenval
        M = ensure_symmetric(M)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed: {e}). "
                f"Added {min_eigenval:.2e} to diagonal."
            )
    
    return M


def cap_max_eigenval(
    M: np.ndarray,
    max_eigenval: float = 0.99,
    warn: bool = True
) -> np.ndarray:
    """Cap maximum eigenvalue of matrix to prevent numerical explosion.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (square matrix)
    max_eigenval : float
        Maximum allowed eigenvalue
    warn : bool
        Whether to log warnings
        
    Returns
    -------
    np.ndarray
        Matrix with capped eigenvalues
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    try:
        eigenvals = np.linalg.eigvals(M)
        max_eig = float(np.max(np.abs(eigenvals)))
        
        if max_eig > max_eigenval:
            scale_factor = max_eigenval / max_eig
            M = M * scale_factor
            if warn:
                _logger.warning(
                    f"Matrix maximum eigenvalue capped: {max_eig:.2e} -> {max_eigenval:.2e} "
                    f"(scale_factor={scale_factor:.2e})"
                )
    except (np.linalg.LinAlgError, ValueError):
        pass
    
    return M


def ensure_covariance_stable(
    M: np.ndarray,
    min_eigenval: float = _MIN_EIGENVAL
) -> np.ndarray:
    """Ensure covariance matrix is symmetric and positive semi-definite.
    
    Parameters
    ----------
    M : np.ndarray
        Covariance matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue to enforce
        
    Returns
    -------
    np.ndarray
        Stable covariance matrix
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    # Ensure symmetric and positive semi-definite
    M = ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
    
    return M


def check_finite(arr: np.ndarray, name: str = "array") -> bool:
    """Check if array contains only finite values.
    
    Parameters
    ----------
    arr : np.ndarray
        Array to check
    name : str
        Name for error messages
        
    Returns
    -------
    bool
        True if array is finite, False otherwise
    """
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    
    if has_nan or has_inf:
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        msg = f"{name} contains "
        issues = []
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values")
        if inf_count > 0:
            issues.append(f"{inf_count} Inf values")
        msg += " and ".join(issues)
        _logger.warning(msg)
        return False
    return True


@dataclass
class EMConfig:
    """Configuration for EM algorithm parameters."""
    regularization: float = 1e-6
    min_norm: float = 1e-8
    max_eigenval: float = 0.99
    min_variance: float = 1e-4
    max_variance: float = 1e4
    min_iterations_for_convergence_check: int = 2
    convergence_log_interval: int = 10
    progress_log_interval: int = 5
    small_loglik_threshold: float = 1e-10
    convergence_threshold: float = 1e-10
    # Initialization constants (used by DFM initialization)
    default_transition_coef: float = 0.9
    default_process_noise: float = 0.1
    default_observation_noise: float = 1e-4
    matrix_regularization: float = 1e-6
    eigenval_floor: float = 1e-8
    quarterly_ar_coef: float = 0.1
    tent_kernel_size: int = 5
    quarterly_variance_denominator: float = 19.0
    extreme_forecast_threshold: float = 50.0


_DEFAULT_EM_CONFIG = EMConfig()


def _update_transition_matrix(EZ: np.ndarray, A: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update transition matrix A using OLS regression."""
    T, m = EZ.shape
    if T <= 1:
        return A
    
    try:
        Y = EZ[1:, :]  # (T-1, m)
        X = EZ[:-1, :]  # (T-1, m)
        XTX = X.T @ X + np.eye(m) * config.regularization
        XTY = X.T @ Y
        A_new = np.linalg.solve(XTX, XTY).T
        return cap_max_eigenval(A_new, max_eigenval=config.max_eigenval, warn=False)
    except (np.linalg.LinAlgError, ValueError):
        return A


def _update_observation_matrix(X: np.ndarray, EZ: np.ndarray, EZZ: np.ndarray, C: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation matrix C using OLS regression."""
    try:
        N = X.shape[1]
        m = EZ.shape[1]
        X_clean = np.ma.filled(np.ma.masked_invalid(X), 0.0)
        sum_yEZ = X_clean.T @ EZ  # (N, m)
        sum_EZZ = np.sum(EZZ, axis=0) + np.eye(m) * config.regularization
        C_new = np.linalg.solve(sum_EZZ.T, sum_yEZ.T).T
        # Normalize columns
        for j in range(m):
            norm = np.linalg.norm(C_new[:, j])
            if norm > config.min_norm:
                C_new[:, j] /= norm
        return C_new
    except (np.linalg.LinAlgError, ValueError):
        return C


def _update_process_noise(EZ: np.ndarray, A_new: np.ndarray, Q: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update process noise covariance Q from residuals."""
    T, m = EZ.shape
    if T <= 1:
        return Q
    
    residuals = EZ[1:, :] - EZ[:-1, :] @ A_new.T
    if m == 1:
        Q_new = np.array([[np.var(residuals, axis=0)]])
    else:
        Q_new = np.cov(residuals.T)
    Q_new = ensure_covariance_stable(Q_new, min_eigenval=config.min_variance)
    return np.maximum(Q_new, np.eye(m) * config.min_variance)


def _update_observation_noise(X: np.ndarray, EZ: np.ndarray, C_new: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation noise covariance R (diagonal) from residuals."""
    X_clean = np.ma.filled(np.ma.masked_invalid(X), 0.0)
    residuals = X_clean - EZ @ C_new.T
    diag_R = np.var(residuals, axis=0)
    diag_R = np.clip(diag_R, config.min_variance, config.max_variance)
    R_new = np.diag(diag_R)
    return ensure_covariance_stable(R_new, min_eigenval=config.min_variance)


def em_step(
    X: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    kalman_filter: Optional[DFMKalmanFilter] = None,
    config: Optional[EMConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Optional[DFMKalmanFilter]]:
    """Perform one EM step: pykalman E-step + custom M-step.
    
    E-step: Uses pykalman's Kalman filter/smoother (handles missing data).
    M-step: Custom OLS that preserves block structure and mixed-frequency constraints.
    Note: We can't use pykalman's built-in em() because it doesn't handle block structure.
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    A, C, Q, R, Z_0, V_0 : np.ndarray
        Current model parameters
    kalman_filter : DFMKalmanFilter, optional
        Existing Kalman filter instance. If None, creates a new one.
    config : EMConfig, optional
        EM configuration. If None, uses defaults.
        
    Returns
    -------
    A_new, C_new, Q_new, R_new, Z_0_new, V_0_new : np.ndarray
        Updated parameters
    loglik : float
        Log-likelihood value
    kalman_filter : DFMKalmanFilter
        Updated Kalman filter instance
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    # Create or update Kalman filter
    if kalman_filter is None:
        kalman_filter = DFMKalmanFilter(
            transition_matrices=A, observation_matrices=C,
            transition_covariance=Q, observation_covariance=R,
            initial_state_mean=Z_0, initial_state_covariance=V_0
        )
    else:
        kalman_filter.update_parameters(A, C, Q, R, Z_0, V_0)
    
    # E-step: pykalman handles missing data via masked arrays
    X_masked = np.ma.masked_invalid(X)
    EZ, V_smooth, _, loglik = kalman_filter.filter_and_smooth(X_masked)
    
    # Compute smoothed factor covariances
    EZZ = V_smooth + np.einsum('ti,tj->tij', EZ, EZ)  # (T, m, m)
    
    # M-step: Update parameters using helper functions
    A_new = _update_transition_matrix(EZ, A, config)
    C_new = _update_observation_matrix(X, EZ, EZZ, C, config)
    Q_new = _update_process_noise(EZ, A_new, Q, config)
    R_new = _update_observation_noise(X, EZ, C_new, config)
    
    # Update initial state
    Z_0_new = EZ[0, :] if EZ.shape[0] > 0 else Z_0
    V_0_new = ensure_covariance_stable(V_smooth[0] if len(V_smooth) > 0 else V_0, min_eigenval=config.min_variance)
    
    return A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, kalman_filter


def run_em_algorithm(
    X: np.ndarray,
    initial_params: Dict[str, np.ndarray],
    update_params_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None],
    get_params_fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    max_iter: int = 100,
    threshold: float = 1e-4,
    config: Optional[EMConfig] = None
) -> Dict[str, Any]:
    """Run full EM algorithm until convergence.
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    initial_params : dict
        Initial parameters {'A', 'C', 'Q', 'R', 'Z_0', 'V_0'}
    update_params_fn : callable
        Function to update model parameters: update_params_fn(A, C, Q, R, Z_0, V_0)
    get_params_fn : callable
        Function to get current parameters: get_params_fn() -> (A, C, Q, R, Z_0, V_0)
    max_iter : int
        Maximum iterations
    threshold : float
        Convergence threshold
    config : EMConfig, optional
        EM configuration
        
    Returns
    -------
    dict
        Final state with keys: 'A', 'C', 'Q', 'R', 'Z_0', 'V_0', 'loglik', 'num_iter', 'converged'
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    # Initialize parameters
    update_params_fn(
        initial_params['A'],
        initial_params['C'],
        initial_params['Q'],
        initial_params['R'],
        initial_params['Z_0'],
        initial_params['V_0']
    )
    A, C, Q, R, Z_0, V_0 = get_params_fn()
    
    # Initialize state
    previous_loglik = float('-inf')
    num_iter = 0
    converged = False
    loglik = float('-inf')
    change = 0.0
    kalman_filter = None
    
    # EM loop
    while num_iter < max_iter and not converged:
        # E-step + M-step using pykalman
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, kalman_filter = em_step(
            X, A, C, Q, R, Z_0, V_0, kalman_filter=kalman_filter, config=config
        )
        
        # Check for NaN/Inf (early stopping)
        if not all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                   for p in [A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik]):
            _logger.error(f"EM: NaN/Inf at iteration {num_iter + 1}, stopping")
            break
        
        # Update parameters
        update_params_fn(A_new, C_new, Q_new, R_new, Z_0_new, V_0_new)
        A, C, Q, R, Z_0, V_0 = A_new, C_new, Q_new, R_new, Z_0_new, V_0_new
        
        # Check convergence (relative change in log-likelihood)
        if num_iter >= config.min_iterations_for_convergence_check:
            if abs(previous_loglik) < config.small_loglik_threshold:
                change = abs(loglik - previous_loglik)
            else:
                change = abs((loglik - previous_loglik) / previous_loglik) if previous_loglik != 0.0 else abs(loglik - previous_loglik)
            converged = change < threshold
        else:
            change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else 0.0
        
        previous_loglik = loglik
        num_iter += 1
        
        # Log progress
        if num_iter % config.progress_log_interval == 0 or num_iter == 1:
            status = " ✓" if converged else ""
            _logger.info(f"EM iteration {num_iter}/{max_iter}: loglik={loglik:.4f}, change={change:.2e}{status}")
    
    # Final status
    if converged:
        print(f"\n✓ EM converged after {num_iter} iterations (loglik: {loglik:.6f})")
    else:
        print(f"\n⚠ EM stopped after {num_iter} iterations (loglik: {loglik:.6f}, change: {change:.2e})")
    
    _logger.info(f"EM training completed: converged={converged}, iterations={num_iter}, loglik={loglik:.6f}")
    
    return {
        'A': A,
        'C': C,
        'Q': Q,
        'R': R,
        'Z_0': Z_0,
        'V_0': V_0,
        'loglik': loglik,
        'num_iter': num_iter,
        'converged': converged
    }


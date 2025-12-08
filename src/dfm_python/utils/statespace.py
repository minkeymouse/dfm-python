"""State-space utility functions.

This module contains utility functions extracted from state_space.py
to keep the main file under 1000 lines.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import warnings
from ..logger import get_logger

_logger = get_logger(__name__)

# PyTorch imports (optional, for DDFM utilities)
if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = None
    nn = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _has_torch = True
except ImportError:
    _has_torch = False
    if not TYPE_CHECKING:
        torch = None
        nn = None
    F = None

# Numerical stability constants for NumPy operations
# Note: Similar constants exist in ssm/utils.py (DEFAULT_MIN_EIGENVAL, DEFAULT_MIN_DIAGONAL_VARIANCE)
# for PyTorch operations. They have the same values but are kept separate for context clarity.
MIN_EIGENVAL_CLEAN = 1e-8
MIN_DIAGONAL_VARIANCE = 1e-6
DEFAULT_VARIANCE_FALLBACK = 1.0
MIN_VARIANCE_COVARIANCE = 1e-10


def _ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose."""
    return 0.5 * (M + M.T)


def _clean_matrix(M: np.ndarray, matrix_type: str = 'general', 
                  default_nan: float = 0.0, default_inf: Optional[float] = None) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability."""
    if matrix_type == 'covariance':
        M = np.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = _ensure_symmetric(M)
        try:
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < MIN_EIGENVAL_CLEAN:
                M = M + np.eye(M.shape[0]) * (MIN_EIGENVAL_CLEAN - min_eigenval)
                M = _ensure_symmetric(M)
        except (np.linalg.LinAlgError, ValueError):
            M = M + np.eye(M.shape[0]) * MIN_EIGENVAL_CLEAN
            M = _ensure_symmetric(M)
    elif matrix_type == 'diagonal':
        diag = np.diag(M)
        diag = np.nan_to_num(diag, nan=default_nan, 
                            posinf=default_inf if default_inf is not None else 1e4,
                            neginf=default_nan)
        diag = np.maximum(diag, MIN_DIAGONAL_VARIANCE)
        M = np.diag(diag)
    elif matrix_type == 'loading':
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M

def _safe_determinant(M: np.ndarray, use_logdet: bool = True) -> float:
    """Compute determinant safely to avoid overflow warnings.
    
    Uses log-determinant computation for large matrices or matrices with high
    condition numbers to avoid numerical overflow. For positive semi-definite
    matrices, uses Cholesky decomposition which is more stable.
    
    Parameters
    ----------
    M : np.ndarray
        Square matrix for which to compute determinant
    use_logdet : bool
        Whether to use log-determinant computation (default: True)
        
    Returns
    -------
    det : float
        Determinant of M, or 0.0 if computation fails
    """
    if M.size == 0 or M.shape[0] == 0:
        return 0.0
    
    if M.shape[0] != M.shape[1]:
        _logger.debug("_safe_determinant: non-square matrix, returning 0.0")
        return 0.0
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(M)):
        _logger.debug("_safe_determinant: matrix contains NaN/Inf, returning 0.0")
        return 0.0
    
    # For small matrices (1x1 or 2x2), direct computation is safe
    if M.shape[0] <= 2:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                det = np.linalg.det(M)
                if np.isfinite(det):
                    return float(det)
        except (RuntimeWarning, OverflowError):
            pass
        # Fall through to log-determinant
    
    # Check condition number to decide on method
    try:
        eigenvals = np.linalg.eigvals(M)
        eigenvals = eigenvals[np.isfinite(eigenvals)]
        if len(eigenvals) > 0:
            max_eig = np.max(np.abs(eigenvals))
            min_eig = np.max(np.abs(eigenvals[eigenvals != 0])) if np.any(eigenvals != 0) else max_eig
            cond_num = max_eig / max(min_eig, 1e-12)
        else:
            cond_num = np.inf
    except (np.linalg.LinAlgError, ValueError):
        cond_num = np.inf
    
    # Use log-determinant for large condition numbers or if requested
    if use_logdet or cond_num > 1e10:
        try:
            # Try Cholesky decomposition first (more stable for PSD matrices)
            try:
                L = np.linalg.cholesky(M)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                # Check if log_det is too large to avoid overflow in exp
                if log_det > 700:  # exp(700) is near float64 max
                    _logger.debug("_safe_determinant: log_det too large, returning 0.0")
                    return 0.0
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    det = np.exp(log_det)
                if np.isfinite(det) and det > 0:
                    return float(det)
            except np.linalg.LinAlgError:
                # Not PSD: fall back to slogdet for general matrices
                try:
                    sign, log_det = np.linalg.slogdet(M)
                    # If determinant is non-positive or invalid, return 0.0
                    if not np.isfinite(log_det) or sign <= 0:
                        return 0.0
                    # Avoid overflow in exp
                    if log_det > 700:
                        _logger.debug("_safe_determinant: log_det too large, returning 0.0")
                        return 0.0
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        det = np.exp(log_det)
                    if np.isfinite(det):
                        return float(det)
                except (np.linalg.LinAlgError, ValueError, OverflowError):
                    pass
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            pass
    
    # Fallback: direct computation with exception handling
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            det = np.linalg.det(M)
            if np.isfinite(det):
                return float(det)
    except (np.linalg.LinAlgError, ValueError, OverflowError):
        pass
    
    _logger.debug("_safe_determinant: all methods failed, returning 0.0")
    return 0.0




def _compute_cov_safe(data: np.ndarray, rowvar: bool = True, 
                              pairwise_complete: bool = False,
                              min_eigenval: float = 1e-8,
                              fallback_to_identity: bool = True) -> np.ndarray:
    """Compute covariance matrix safely with robust error handling."""
    if data.size == 0:
        if fallback_to_identity:
            return np.eye(1) if data.ndim == 1 else np.eye(data.shape[1] if rowvar else data.shape[0])
        raise ValueError("Cannot compute covariance: data is empty")
    
    # Handle 1D case
    if data.ndim == 1:
        var_val = _compute_var_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE, 
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = _compute_var_safe(series_data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Compute covariance
    try:
        if pairwise_complete:
            # Pairwise complete covariance: compute covariance for each pair separately
            if rowvar:
                data_for_cov = data.T  # Transpose to (N, T) for np.cov
            else:
                data_for_cov = data
            
            # Compute pairwise complete covariance manually
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    var_i = data_for_cov[i, :]
                    var_j = data_for_cov[j, :]
                    complete_mask = np.isfinite(var_i) & np.isfinite(var_j)
                    if np.sum(complete_mask) < 2:
                        if i == j:
                            cov[i, j] = DEFAULT_VARIANCE_FALLBACK
                        else:
                            cov[i, j] = 0.0
                    else:
                        var_i_complete = var_i[complete_mask]
                        var_j_complete = var_j[complete_mask]
                        if i == j:
                            cov[i, j] = np.var(var_i_complete, ddof=0)
                        else:
                            mean_i = np.mean(var_i_complete)
                            mean_j = np.mean(var_j_complete)
                            cov[i, j] = np.mean((var_i_complete - mean_i) * (var_j_complete - mean_j))
                            cov[j, i] = cov[i, j]  # Symmetric
            
            # Ensure minimum variance
            np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE_COVARIANCE))
        else:
            # Standard covariance (listwise deletion)
            if rowvar:
                complete_rows = np.all(np.isfinite(data), axis=1)
                if np.sum(complete_rows) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[complete_rows, :]
                data_for_cov = data_clean.T  # (N, T)
                cov = np.cov(data_for_cov, rowvar=True)  # Returns (N, N)
            else:
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[:, complete_cols]
                data_for_cov = data_clean.T  # (T, N)
                cov = np.cov(data_for_cov, rowvar=False)  # Returns (N, N)
            
            # np.cov can sometimes return unexpected shapes, so verify
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                if len(cov) == n_vars:
                    cov = np.diag(cov)
                else:
                    raise ValueError(f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})")
        
        # Ensure correct shape
        if cov.shape != (n_vars, n_vars):
            raise ValueError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}"
            )
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise ValueError("Covariance contains non-finite values")
        
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            eye_matrix = np.eye(n_vars)
            cov = cov + eye_matrix * reg_amount
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return np.eye(n_vars)
        raise


def _compute_var_safe(data: np.ndarray, ddof: int = 0, 
                           min_variance: float = MIN_VARIANCE_COVARIANCE,
                           default_variance: float = DEFAULT_VARIANCE_FALLBACK) -> float:
    """Compute variance safely with robust error handling."""
    if data.size == 0:
        return default_variance
    
    # Flatten if 2D
    if data.ndim > 1:
        data = data.flatten()
    
    # Compute variance with NaN handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_val = np.nanvar(data, ddof=ddof)
    
    # Validate and enforce minimum
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    
    return float(var_val)


def _estimate_ar(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
                             vsmooth_sum: Optional[np.ndarray] = None,
                             T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR coefficients and innovation variances from expectations."""
    if np.isscalar(EZZ_FB):
        EZZ_FB = np.array([EZZ_FB])
        EZZ_BB = np.array([EZZ_BB])
    if EZZ_FB.ndim > 1:
        EZZ_FB_diag = np.diag(EZZ_FB).copy()
        EZZ_BB_diag = np.diag(EZZ_BB).copy()
    else:
        EZZ_FB_diag = EZZ_FB.copy()
        EZZ_BB_diag = EZZ_BB.copy()
    if vsmooth_sum is not None:
        if vsmooth_sum.ndim > 1:
            vsmooth_diag = np.diag(vsmooth_sum)
        else:
            vsmooth_diag = vsmooth_sum
        EZZ_BB_diag = EZZ_BB_diag + vsmooth_diag
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * MIN_DIAGONAL_VARIANCE, MIN_VARIANCE_COVARIANCE)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    # Use _clean_matrix for consistency
    if EZZ_FB_diag.ndim == 0:
        EZZ_FB_diag_clean = _clean_matrix(np.array([EZZ_FB_diag]), 'general', default_nan=0.0, default_inf=1e6)
        EZZ_FB_diag = EZZ_FB_diag_clean[0] if EZZ_FB_diag_clean.size > 0 else 0.0
    else:
        EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0, default_inf=1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    Q_diag = None
    return A_diag, Q_diag


def _clip_ar(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
                         warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds."""
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
    """Apply AR coefficient clipping based on configuration."""
    if config is None:
        return _clip_ar(A, -0.99, 0.99, True)
    
    
    clip_enabled = getattr(config, 'clip_ar_coefficients', True) if config is not None else True
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = getattr(config, 'ar_clip_min', -0.99) if config is not None else -0.99
    max_val = getattr(config, 'ar_clip_max', 0.99) if config is not None else 0.99
    warn = getattr(config, 'warn_on_ar_clip', True) if config is not None else True
    return _clip_ar(A, min_val, max_val, warn)


def _compute_reg_param(
    matrix: np.ndarray,
    scale_factor: float = 1e-5,
    warn: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """Compute regularization parameter for matrix inversion."""
    stats = {
        'regularized': False,
        'condition_number': None,
        'reg_amount': 0.0
    }
    
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0, stats
    
    try:
        eigenvals = np.linalg.eigvalsh(matrix)
        eigenvals = eigenvals[np.isfinite(eigenvals) & (eigenvals != 0)]
        
        if len(eigenvals) == 0:
            reg_param = scale_factor
            stats['regularized'] = True
            stats['reg_amount'] = reg_param
            if warn:
                _logger.warning(f"Matrix has no valid eigenvalues, using default regularization: {reg_param:.2e}")
            return reg_param, stats
        
        max_eig = np.max(np.abs(eigenvals))
        min_eig = np.min(np.abs(eigenvals[eigenvals != 0]))
        cond_num = max_eig / max(min_eig, 1e-12)
        stats['condition_number'] = float(cond_num)
        
        if cond_num > 1e8:
            reg_param = scale_factor * (cond_num / 1e8)
            stats['regularized'] = True
            stats['reg_amount'] = reg_param
            if warn:
                _logger.warning(f"Matrix is ill-conditioned (cond={cond_num:.2e}), applying regularization: {reg_param:.2e}")
        else:
            reg_param = scale_factor
            stats['reg_amount'] = reg_param
            
    except (np.linalg.LinAlgError, ValueError) as e:
        reg_param = scale_factor
        stats['regularized'] = True
        stats['reg_amount'] = reg_param
        if warn:
            _logger.warning(f"Regularization computation failed ({type(e).__name__}), using default: {reg_param:.2e}")
    
    return reg_param, stats

def _cap_eigenval(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
    """Cap maximum eigenvalue of matrix to prevent numerical explosion.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (square matrix)
    max_eigenval : float, default 1e6
        Maximum allowed eigenvalue
        
    Returns
    -------
    M_capped : np.ndarray
        Matrix with capped eigenvalues
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    try:
        eigenvals = np.linalg.eigvalsh(M)
        max_eig = float(np.max(np.abs(eigenvals)))
        
        if max_eig > max_eigenval:
            # Scale matrix to cap maximum eigenvalue
            # _ensure_symmetric is already imported at the top
            scale_factor = max_eigenval / max_eig
            M = M * scale_factor
            M = _ensure_symmetric(M)
    
    except (np.linalg.LinAlgError, ValueError):
        # If eigendecomposition fails, return matrix as-is
        pass
    
    return M


def _ensure_variance_min(Q: np.ndarray, min_variance: float = 1e-8) -> np.ndarray:
    """Ensure minimum variance on diagonal of innovation covariance matrix.
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix
    min_variance : float, default 1e-8
        Minimum variance to enforce
        
    Returns
    -------
    Q_stable : np.ndarray
        Matrix with minimum variance enforced
    """
    if Q.size == 0 or Q.shape[0] == 0:
        return Q
    
    Q_diag = np.diag(Q).copy()
    Q_diag = np.maximum(Q_diag, min_variance)
    Q = np.diag(Q_diag) + (Q - np.diag(np.diag(Q)))  # Preserve off-diagonal
    return Q


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling zero denominators.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    default : float, default 0.0
        Default value when denominator is zero
        
    Returns
    -------
    result : np.ndarray
        Division result
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator, out=np.full_like(numerator, default), where=denominator!=0)
    return result



def estimate_var1(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(1) dynamics for factors.
    
    Note: Maximum supported VAR order is VAR(2). Use estimate_var2() for VAR(2) estimation.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x m)
    Q : np.ndarray
        Innovation covariance (m x m)
    """
    T, m = factors.shape
    
    if T < 2:
        # Not enough data, use identity
        A = np.eye(m)
        Q = np.eye(m) * 0.1
        return A, Q
    
    # Prepare data for OLS: f_t = A @ f_{t-1}
    Y = factors[1:, :]  # T-1 x m (dependent)
    X = factors[:-1, :]  # T-1 x m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y
    try:
        A = np.linalg.solve(X.T @ X + np.eye(m) * 1e-6, X.T @ Y).T
    except np.linalg.LinAlgError:
        # Fallback to pinv
        A = np.linalg.pinv(X) @ Y
    
    # Ensure stability: clip eigenvalues
    eigenvals = np.linalg.eigvals(A)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= 0.99:
        A = A * (0.99 / max_eigenval)
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = np.cov(residuals.T)
    
    # Ensure Q is positive definite
    Q = (Q + Q.T) / 2  # Symmetrize
    eigenvals_Q = np.linalg.eigvals(Q)
    min_eigenval = np.min(eigenvals_Q)
    if min_eigenval < 1e-8:
        Q = Q + np.eye(m) * (1e-8 - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * 0.01)
    
    return A, Q


def estimate_var2(factors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate VAR(2) dynamics for factors.
    
    Note: VAR(2) is the maximum supported VAR order in this implementation.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (m x 2m) = [A1, A2]
    Q : np.ndarray
        Innovation covariance (m x m)
    """
    T, m = factors.shape
    
    if T < 3:
        # Not enough data, use VAR(1) fallback
        _logger.warning(
            f"Insufficient data (T={T}) for VAR(2). Falling back to VAR(1)."
        )
        A1, Q = estimate_var1(factors)
        # Pad A to VAR(2) format: [A1, A2] where A2 = 0
        A = np.hstack([A1, np.zeros((A1.shape[0], A1.shape[1]))])
        return A, Q
    
    # Prepare data for VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
    Y = factors[2:, :]  # T-2 x m (dependent)
    X = np.hstack((factors[1:-1, :], factors[:-2, :]))  # T-2 x 2m (independent)
    
    # OLS: A = (X'X)^{-1} X'Y, where A = [A1, A2]
    try:
        A = np.linalg.solve(X.T @ X + np.eye(2 * m) * 1e-6, X.T @ Y).T
    except np.linalg.LinAlgError:
        # Fallback to pinv
        A = np.linalg.pinv(X) @ Y
    
    # Split into A1 and A2
    A1 = A[:, :m]
    A2 = A[:, m:]
    
    # Ensure stability: check eigenvalues of companion form
    companion = np.block([
        [A1, A2],
        [np.eye(m), np.zeros((m, m))]
    ])
    eigenvals = np.linalg.eigvals(companion)
    max_eigenval = np.max(np.abs(eigenvals))
    if max_eigenval >= 0.99:
        scale = 0.99 / max_eigenval
        A1 = A1 * scale
        A2 = A2 * scale
        A = np.hstack((A1, A2))
    
    # Estimate innovation covariance
    residuals = Y - X @ A.T
    Q = np.cov(residuals.T)
    
    # Ensure Q is positive definite
    Q = (Q + Q.T) / 2  # Symmetrize
    eigenvals_Q = np.linalg.eigvals(Q)
    min_eigenval = np.min(eigenvals_Q)
    if min_eigenval < 1e-8:
        Q = Q + np.eye(m) * (1e-8 - min_eigenval)
    
    # Floor for Q
    Q = np.maximum(Q, np.eye(m) * 0.01)
    
    return A, Q


def estimate_idio_dynamics(
    residuals: np.ndarray,
    missing_mask: np.ndarray,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR(1) dynamics for idiosyncratic components.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from observation equation (T x N)
    missing_mask : np.ndarray
        Missing data mask (T x N), True where data is missing
    min_obs : int
        Minimum number of observations required for estimation
        
    Returns
    -------
    A_eps : np.ndarray
        AR(1) coefficients (N x N), diagonal matrix
    Q_eps : np.ndarray
        Innovation covariance (N x N), diagonal matrix
    """
    T, N = residuals.shape
    A_eps = np.zeros((N, N))
    Q_eps = np.zeros((N, N))
    
    for j in range(N):
        # Find valid consecutive pairs (both t-1 and t must be non-missing)
        valid = ~missing_mask[:, j]
        valid_pairs = valid[:-1] & valid[1:]
        
        if np.sum(valid_pairs) < min_obs:
            # Insufficient data: use zero AR(1) coefficient
            _logger.warning(
                f"Insufficient observations ({np.sum(valid_pairs)}) for idio AR(1) "
                f"estimation for series {j}. Using zero AR(1) coefficient."
            )
            A_eps[j, j] = 0.0
            # Use variance of available residuals
            if np.sum(valid) > 0:
                Q_eps[j, j] = np.var(residuals[valid, j])
            else:
                Q_eps[j, j] = 1e-8
        else:
            # Extract valid consecutive pairs
            eps_t = residuals[1:, j][valid_pairs]
            eps_t_1 = residuals[:-1, j][valid_pairs]
            
            # Estimate AR(1) coefficient using covariance
            var_eps_t_1 = np.var(eps_t_1)
            if var_eps_t_1 > 1e-10:
                cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
                A_eps[j, j] = cov_eps / var_eps_t_1
                
                # Ensure stability: clip AR(1) coefficient
                if abs(A_eps[j, j]) >= 0.99:
                    sign = np.sign(A_eps[j, j])
                    A_eps[j, j] = sign * 0.99
                    _logger.debug(
                        f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                    )
            else:
                A_eps[j, j] = 0.0
            
            # Estimate innovation covariance
            residuals_ar = eps_t - A_eps[j, j] * eps_t_1
            Q_eps[j, j] = np.var(residuals_ar)
            Q_eps[j, j] = max(Q_eps[j, j], 1e-8)  # Floor
    
    return A_eps, Q_eps


def build_observation_matrix(C: np.ndarray, factor_order: int, N: int) -> np.ndarray:
    """Build observation matrix H including idiosyncratic components.
    
    Constructs the observation matrix H = [C, I] for VAR(1) or
    H = [C, 0, I] for VAR(2), where C loads on factors and I on idio.
    
    Parameters
    ----------
    C : np.ndarray
        Loading matrix (N x m) from decoder
    factor_order : int
        VAR lag order (1 or 2)
    N : int
        Number of series
        
    Returns
    -------
    H : np.ndarray
        Observation matrix (N x state_dim)
    """
    N_series, m = C.shape
    
    if factor_order == 1:
        # H = [C, I] where C loads on f_t, I loads on eps_t
        H = np.hstack([C, np.eye(N_series)])
    elif factor_order == 2:
        # H = [C, 0, I] where C loads on f_t, 0 on f_{t-1}, I on eps_t
        H = np.hstack([C, np.zeros((N_series, m)), np.eye(N_series)])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
    return H


def build_state_space(
    factors: np.ndarray,
    A_f: np.ndarray,
    Q_f: np.ndarray,
    A_eps: np.ndarray,
    Q_eps: np.ndarray,
    factor_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space model with companion form.
    
    Constructs the complete state-space model including both factors
    and idiosyncratic components in the state vector.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
    A_f : np.ndarray
        Factor transition matrix (m x m) for VAR(1) or (m x 2m) for VAR(2)
    Q_f : np.ndarray
        Factor innovation covariance (m x m)
    A_eps : np.ndarray
        Idiosyncratic AR(1) coefficients (N x N), diagonal
    Q_eps : np.ndarray
        Idiosyncratic innovation covariance (N x N), diagonal
    factor_order : int
        VAR lag order (1 or 2)
        
    Returns
    -------
    A : np.ndarray
        Complete transition matrix (state_dim x state_dim)
    Q : np.ndarray
        Complete innovation covariance (state_dim x state_dim)
    Z_0 : np.ndarray
        Initial state vector
    V_0 : np.ndarray
        Initial state covariance
    """
    T, m = factors.shape
    N = A_eps.shape[0]
    
    if factor_order == 1:
        # State: [f_t, eps_t]
        # Transition: f_t = A_f @ f_{t-1} + v_f, eps_t = A_eps @ eps_{t-1} + v_eps
        # Block diagonal structure
        A = np.block([
            [A_f, np.zeros((m, N))],
            [np.zeros((N, m)), A_eps]
        ])
        
        Q = np.block([
            [Q_f, np.zeros((m, N))],
            [np.zeros((N, m)), Q_eps]
        ])
        
        # Initial state: [f_0, eps_0]
        Z_0 = np.concatenate([factors[0, :], np.zeros(N)])
        
        # Initial covariance: block diagonal
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))  # Use Q_eps as initial idio covariance
        V_0 = np.block([
            [V_f, np.zeros((m, N))],
            [np.zeros((N, m)), V_eps]
        ])
        
    elif factor_order == 2:
        # State: [f_t, f_{t-1}, eps_t]
        # Transition: f_t = A1 @ f_{t-1} + A2 @ f_{t-2} + v_f
        #            f_{t-1} = f_{t-1} (identity)
        #            eps_t = A_eps @ eps_{t-1} + v_eps
        A1 = A_f[:, :m]
        A2 = A_f[:, m:]
        
        A = np.block([
            [A1, A2, np.zeros((m, N))],
            [np.eye(m), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), A_eps]
        ])
        
        Q = np.block([
            [Q_f, np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), Q_eps]
        ])
        
        # Initial state: [f_0, f_{-1}, eps_0]
        # Use f_0 for both f_0 and f_{-1} (or use first two if available)
        if T >= 2:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        else:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        
        # Initial covariance: block diagonal
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))
        V_0 = np.block([
            [V_f, V_f, np.zeros((m, N))],
            [V_f, V_f, np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), V_eps]
        ])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
    return A, Q, Z_0, V_0


def estimate_state_space_params(
    f_t: np.ndarray,
    eps_t: np.ndarray,
    factor_order: int,
    bool_no_miss: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate state-space transition parameters from factors and residuals.
    
    Estimates the transition matrix A, innovation covariance W, initial mean mu_0,
    initial covariance Σ_0, and latent states x_t for the companion form state-space
    representation.
    
    Parameters
    ----------
    f_t : np.ndarray
        Common factors (T x m)
    eps_t : np.ndarray
        Idiosyncratic terms (T x N)
    factor_order : int
        Lag order for common factors. Only VAR(1) and VAR(2) are supported.
        Higher orders will raise NotImplementedError.
    bool_no_miss : np.ndarray, optional
        Boolean array (T x N) indicating non-missing values.
        If None, assumes no missing values.
        
    Returns
    -------
    A : np.ndarray
        Transition matrix (state_dim x state_dim) in companion form
    W : np.ndarray
        Innovation covariance matrix (state_dim x state_dim), diagonal
    mu_0 : np.ndarray
        Unconditional mean of initial state (state_dim,)
    Σ_0 : np.ndarray
        Unconditional covariance of initial state (state_dim x state_dim)
    x_t : np.ndarray
        Latent states (state_dim x T) in companion form
        
    Notes
    -----
    The companion form depends on factor_order:
    - VAR(1): x_t = [f_t, eps_t], A = [[A_f, 0], [0, A_eps]]
    - VAR(2): x_t = [f_t, f_{t-1}, eps_t], A = [[A_f, 0, 0], [I, 0, 0], [0, 0, A_eps]]
    
    The innovation covariance W is diagonal, and Σ_0 enforces zero correlation
    between factors and idiosyncratic components.
    """
    T, m = f_t.shape
    T_eps, N = eps_t.shape
    
    if T != T_eps:
        raise ValueError(f"Time dimension mismatch: f_t has {T} timesteps, eps_t has {T_eps}")
    
    # Estimate factor dynamics (VAR)
    if factor_order == 2:
        if T < 3:
            raise ValueError("Insufficient data for VAR(2). Need at least 3 timesteps.")
        f_past = np.hstack((f_t[1:-1, :], f_t[:-2, :]))  # (T-2) x 2m
        f_future = f_t[2:, :]  # (T-2) x m
        # OLS: A_f = (f_past' @ f_past)^{-1} @ f_past' @ f_future
        try:
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(2*m) * 1e-6, f_past.T @ f_future).T
        except np.linalg.LinAlgError:
            A_f = (np.linalg.pinv(f_past) @ f_future).T
        # Split into A1 and A2
        A1 = A_f[:, :m]  # m x m
        A2 = A_f[:, m:]  # m x m
    elif factor_order == 1:
        if T < 2:
            raise ValueError("Insufficient data for VAR(1). Need at least 2 timesteps.")
        f_past = f_t[:-1, :]  # (T-1) x m
        f_future = f_t[1:, :]  # (T-1) x m
        # OLS: A_f = (f_past' @ f_past)^{-1} @ f_past' @ f_future
        try:
            A_f = np.linalg.solve(f_past.T @ f_past + np.eye(m) * 1e-6, f_past.T @ f_future).T
        except np.linalg.LinAlgError:
            A_f = (np.linalg.pinv(f_past) @ f_future).T
        A1 = A_f
        A2 = None
    else:
        raise NotImplementedError(
            f"Only VAR(1) or VAR(2) for common factors are supported (maximum supported order is VAR(2)). "
            f"Got factor_order={factor_order}. Please use factor_order=1 (VAR(1)) or factor_order=2 (VAR(2))"
        )
    
    # Estimate idiosyncratic AR(1) dynamics
    A_eps, _, _ = estimate_idio_params(eps_t, bool_no_miss, min_obs=5)
    
    # Construct companion form state vector and transition matrix
    if factor_order == 2:
        # x_t = [f_t, f_{t-1}, eps_t]
        x_t = np.vstack([
            f_t[1:, :].T,  # m x (T-1)
            f_t[:-1, :].T,  # m x (T-1)
            eps_t[1:, :].T  # N x (T-1)
        ])  # (2m + N) x (T-1)
        
        # Transition matrix in companion form
        A = np.vstack([
            np.hstack([A1, A2, np.zeros((m, N))]),  # f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            np.hstack([np.eye(m), np.zeros((m, m)), np.zeros((m, N))]),  # f_{t-1} = f_{t-1}
            np.hstack([np.zeros((N, m)), np.zeros((N, m)), A_eps])  # eps_t = A_eps @ eps_{t-1}
        ])
    else:  # factor_order == 1
        # x_t = [f_t, eps_t]
        x_t = np.vstack([
            f_t.T,  # m x T
            eps_t.T  # N x T
        ])  # (m + N) x T
        
        # Transition matrix
        A = np.vstack([
            np.hstack([A1, np.zeros((m, N))]),  # f_t = A1 @ f_{t-1}
            np.hstack([np.zeros((N, m)), A_eps])  # eps_t = A_eps @ eps_{t-1}
        ])
    
    # Estimate innovation covariance (diagonal)
    # w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    W = np.diag(np.diag(np.cov(w_t)))
    # Ensure positive diagonal
    W = np.maximum(W, np.eye(W.shape[0]) * 1e-8)
    
    # Unconditional moments of initial state
    mu_0 = np.mean(x_t, axis=1)
    Σ_0 = np.cov(x_t)
    
    # Enforce zero correlation between factors and idiosyncratic components
    if factor_order == 2:
        factor_dim = 2 * m
    else:
        factor_dim = m
    
    Σ_0[:factor_dim, factor_dim:] = 0
    Σ_0[factor_dim:, :factor_dim] = 0
    # Ensure diagonal covariance for idiosyncratic components
    Σ_0[factor_dim:, factor_dim:] = np.diag(np.diag(Σ_0[factor_dim:, factor_dim:]))
    
    # Ensure positive semidefinite
    eigenvals = np.linalg.eigvals(Σ_0)
    if np.any(eigenvals < 0):
        Σ_0 = Σ_0 + np.eye(Σ_0.shape[0]) * (1e-8 - np.min(eigenvals))
    
    return A, W, mu_0, Σ_0, x_t


def estimate_idio_params(
    eps: np.ndarray,
    idx_no_missings: Optional[np.ndarray] = None,
    min_obs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate AR(1) parameters for idiosyncratic components.
    
    Falls back to zero-coefficient models when insufficient observations are
    available instead of raising errors, ensuring downstream pipelines remain
    robust.
    """
    T, N = eps.shape
    phi = np.zeros((N, N))
    mu_eps = np.zeros(N)
    std_eps = np.zeros(N)
    
    if idx_no_missings is None:
        idx_no_missings = np.ones((T, N), dtype=bool)
    
    insufficient_series = []
    
    for j in range(N):
        mask = idx_no_missings[:, j]
        observed = eps[mask, j]
        
        if observed.size == 0:
            mu_eps[j] = 0.0
            std_eps[j] = 1e-8
            insufficient_series.append((j, 0))
            continue
        
        mu_eps[j] = float(np.mean(observed))
        std_eps_j = float(np.std(observed))
        std_eps[j] = max(std_eps_j, 1e-8)
        
        valid_pairs = mask[:-1] & mask[1:]
        pair_count = int(np.sum(valid_pairs))
        
        if pair_count < max(min_obs, 1):
            insufficient_series.append((j, pair_count))
            continue
        
        eps_t = eps[1:, j][valid_pairs]
        eps_t_1 = eps[:-1, j][valid_pairs]
        var_prev = np.var(eps_t_1)
        
        if var_prev < 1e-10:
            insufficient_series.append((j, pair_count))
            continue
        
        cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
        coeff = cov_eps / var_prev
        phi[j, j] = float(np.clip(coeff, -0.99, 0.99))
    
    if insufficient_series:
        preview = ", ".join(f"{idx}:{cnt}" for idx, cnt in insufficient_series[:5])
        more = ""
        if len(insufficient_series) > 5:
            more = f", ... (+{len(insufficient_series) - 5} more)"
        _logger.warning(
            "Falling back to zero AR coefficients for %d series (insufficient observations). "
            "Series indices and available pairs: %s%s",
            len(insufficient_series),
            preview,
            more,
        )
    
    return phi, mu_eps, std_eps


def mse_missing(
    y_actual: Any,  # torch.Tensor or np.ndarray
    y_predicted: Any,  # torch.Tensor or np.ndarray
) -> Any:  # torch.Tensor or float
    """Mean Squared Error loss function that handles missing data.
    
    Computes MSE only on non-missing values. Missing values in y_actual
    (represented as NaN) are masked out from the loss computation.
    
    Supports both PyTorch tensors and NumPy arrays automatically.
    
    Parameters
    ----------
    y_actual : torch.Tensor or np.ndarray
        Actual values (batch_size x N or T x N) with NaN for missing values
    y_predicted : torch.Tensor or np.ndarray
        Predicted values (batch_size x N or T x N)
        
    Returns
    -------
    torch.Tensor or float
        Scalar MSE loss computed only on non-missing values
        Returns torch.Tensor for PyTorch inputs, float for NumPy inputs
    """
    # Auto-detect input type
    if _has_torch and isinstance(y_actual, torch.Tensor):
        # PyTorch path
        # Create mask: 1 for non-missing, 0 for missing
        mask = torch.where(
            torch.isnan(y_actual),
            torch.zeros_like(y_actual),
            torch.ones_like(y_actual)
        )
        
        # Replace NaN with 0 for computation
        y_actual_clean = torch.where(
            torch.isnan(y_actual),
            torch.zeros_like(y_actual),
            y_actual
        )
        
        # Apply mask to predictions
        y_predicted_masked = y_predicted * mask
        
        # Compute MSE (automatically ignores masked values)
        loss = F.mse_loss(y_actual_clean, y_predicted_masked, reduction='mean')
        return loss
    else:
        # NumPy path (fallback)
        return mse_missing_numpy(y_actual, y_predicted)


def convergence_checker(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
) -> Tuple[float, float]:
    """Check convergence of reconstruction error (matches original TensorFlow implementation).
    
    Returns only delta and loss_now (no converged flag), matching original code.
    
    Parameters
    ----------
    y_prev : np.ndarray
        Previous reconstruction (T x N)
    y_now : np.ndarray
        Current reconstruction (T x N)
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
        
    Returns
    -------
    delta : float
        Relative change in loss: |loss_now - loss_prev| / loss_prev
    loss_now : float
        Current MSE loss (on non-missing values)
    """
    # Mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    # Compute MSE on non-missing values (matches original implementation)
    y_prev_valid = y_prev[mask]
    y_now_valid = y_now[mask]
    y_actual_valid = y_actual[mask]
    
    loss_prev = np.mean((y_actual_valid - y_prev_valid) ** 2)
    loss_now = np.mean((y_actual_valid - y_now_valid) ** 2)
    
    # Relative change
    if loss_prev < 1e-10:
        delta = abs(loss_now - loss_prev)
    else:
        delta = abs(loss_now - loss_prev) / loss_prev
    
    return delta, loss_now


def check_convergence(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
    threshold: float = 1e-6,
) -> Tuple[float, float, bool]:
    """Check convergence of reconstruction error.
    
    Computes the relative change in MSE between two iterations and checks
    if convergence has been reached. This is a wrapper around convergence_checker()
    that adds a converged flag.
    
    Parameters
    ----------
    y_prev : np.ndarray
        Previous reconstruction (T x N)
    y_now : np.ndarray
        Current reconstruction (T x N)
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
    threshold : float
        Convergence threshold for relative change in loss
        
    Returns
    -------
    relative_change : float
        Relative change in loss: |loss_now - loss_prev| / loss_prev
    loss_now : float
        Current MSE loss (on non-missing values)
    converged : bool
        True if relative change is below threshold
    """
    # Use convergence_checker for the core computation
    delta, loss_now = convergence_checker(y_prev, y_now, y_actual)
    converged = delta < threshold
    return delta, loss_now, converged


def mse_missing_numpy(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
) -> float:
    """NumPy version of missing-aware MSE loss.
    
    Computes MSE only on non-missing values. Missing values in y_actual
    (represented as NaN) are masked out from the loss computation.
    
    Parameters
    ----------
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
    y_predicted : np.ndarray
        Predicted values (T x N)
        
    Returns
    -------
    float
        MSE loss computed only on non-missing values
        
    """
    # Create mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    if np.sum(mask) == 0:
        # All values are missing
        return 0.0
    
    # Compute MSE only on non-missing values
    y_actual_valid = y_actual[mask]
    y_predicted_valid = y_predicted[mask]
    
    mse = np.mean((y_actual_valid - y_predicted_valid) ** 2)
    
    return mse



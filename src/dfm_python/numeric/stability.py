"""Numerical stability functions for matrix operations.

This module provides functions to ensure numerical stability of matrices,
including symmetry enforcement, positive definiteness, eigenvalue capping,
matrix cleaning, safe determinant computation, missing data handling,
and analytical computations.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, Union
import torch
from torch import Tensor

from ..logger import get_logger
from ..utils.errors import DataValidationError, DataError, NumericalError
from ..utils.helper import handle_linear_algebra_error
from ..config.constants import (
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_FACTOR_VARIANCE,
    MAX_EIGENVALUE,
    MATRIX_TYPE_GENERAL,
    MATRIX_TYPE_COVARIANCE,
    MATRIX_TYPE_DIAGONAL,
    MATRIX_TYPE_LOADING,
    DEFAULT_REGULARIZATION_SCALE,
    MIN_CONDITION_NUMBER,
    MAX_CONDITION_NUMBER,
    DEFAULT_EIGENVALUE_MAX_MAGNITUDE,
    DEFAULT_MAX_VARIANCE,
    MAX_LOG_DETERMINANT,
    CHOLESKY_LOG_DET_FACTOR,
    SYMMETRY_AVERAGE_FACTOR,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_VARIANCE_FALLBACK,
    DEFAULT_CLEAN_NAN,
)

_logger = get_logger(__name__)

# Numerical stability constants
MIN_EIGENVAL_CLEAN = MIN_EIGENVALUE
MIN_VARIANCE_COVARIANCE = MIN_FACTOR_VARIANCE


def create_scaled_identity(n: int, scale: float = DEFAULT_IDENTITY_SCALE, dtype: type = np.float32) -> np.ndarray:
    """Create a scaled identity matrix: scale * I_n.
    
    This is a common pattern used throughout the codebase for initializing
    transition matrices, regularization terms, and default covariances.
    
    Parameters
    ----------
    n : int
        Matrix dimension
    scale : float, default DEFAULT_IDENTITY_SCALE
        Scaling factor (uses DEFAULT_IDENTITY_SCALE constant)
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    np.ndarray
        Scaled identity matrix (n x n)
    """
    return np.eye(n, dtype=dtype) * scale


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
    return SYMMETRY_AVERAGE_FACTOR * (M + M.T)


def clean_matrix(
    M: np.ndarray,
    matrix_type: Optional[str] = None,
    default_nan: float = DEFAULT_ZERO_VALUE,
    default_inf: Optional[float] = None
) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to clean
    matrix_type : str, optional
        Type of matrix: 'covariance', 'diagonal', 'loading', or 'general'
    default_nan : float, default DEFAULT_ZERO_VALUE
        Default value for NaN replacement (uses DEFAULT_ZERO_VALUE constant)
    default_inf : float, optional
        Default value for Inf replacement
        
    Returns
    -------
    np.ndarray
        Cleaned matrix
    """
    if matrix_type is None:
        matrix_type = MATRIX_TYPE_GENERAL
    
    if matrix_type == MATRIX_TYPE_COVARIANCE:
        M = np.nan_to_num(M, nan=default_nan, posinf=MAX_EIGENVALUE, neginf=-MAX_EIGENVALUE)
        M = ensure_symmetric(M)
        def _apply_eigenvalue_cleaning():
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < MIN_EIGENVAL_CLEAN:
                from .stability import create_scaled_identity
                M_cleaned = M + create_scaled_identity(M.shape[0], MIN_EIGENVAL_CLEAN - min_eigenval)
                return ensure_symmetric(M_cleaned)
            return M
        
        def _fallback_cleaning():
            from .stability import create_scaled_identity
            M_cleaned = M + create_scaled_identity(M.shape[0], MIN_EIGENVAL_CLEAN)
            return ensure_symmetric(M_cleaned)
        
        M = handle_linear_algebra_error(
            _apply_eigenvalue_cleaning, "eigenvalue cleaning",
            fallback_func=_fallback_cleaning
        )
    elif matrix_type == MATRIX_TYPE_DIAGONAL:
        diag = np.diag(M)
        default_inf_val = default_inf if default_inf is not None else DEFAULT_MAX_VARIANCE
        diag = np.nan_to_num(diag, nan=default_nan, posinf=default_inf_val, neginf=default_nan)
        diag = np.maximum(diag, MIN_DIAGONAL_VARIANCE)
        M = np.diag(diag)
    elif matrix_type == MATRIX_TYPE_LOADING:
        M = np.nan_to_num(M, nan=default_nan, posinf=DEFAULT_EIGENVALUE_MAX_MAGNITUDE, neginf=-DEFAULT_EIGENVALUE_MAX_MAGNITUDE)
    else:
        default_inf_val = default_inf if default_inf is not None else MAX_EIGENVALUE
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M


def cap_max_eigenval(
    M: np.ndarray,
    max_eigenval: float = MAX_EIGENVALUE,
    symmetric: bool = False,
    warn: bool = False
) -> np.ndarray:
    """Cap maximum eigenvalue of matrix to prevent numerical explosion.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (square matrix)
    max_eigenval : float, default MAX_EIGENVALUE
        Maximum allowed eigenvalue
    symmetric : bool, default False
        If True, assumes matrix is symmetric and uses eigvalsh (faster).
        If False, uses eigvals for general matrices (e.g., transition matrices).
    warn : bool, default False
        Whether to log warnings when capping occurs
        
    Returns
    -------
    np.ndarray
        Matrix with capped eigenvalues
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    def _cap_max_eigenvalue():
        if symmetric:
            eigenvals = np.linalg.eigvalsh(M)
        else:
            eigenvals = np.linalg.eigvals(M)
        max_eig = float(np.max(np.abs(eigenvals)))
        
        if max_eig > max_eigenval:
            scale_factor = max_eigenval / max_eig
            M_capped = M * scale_factor
            if symmetric:
                M_capped = ensure_symmetric(M_capped)
            if warn:
                _logger.warning(
                    f"Matrix maximum eigenvalue capped: {max_eig:.2e} -> {max_eigenval:.2e} "
                    f"(scale_factor={scale_factor:.2e})"
                )
            return M_capped
        return M
    
    M = handle_linear_algebra_error(
        _cap_max_eigenvalue, "maximum eigenvalue capping",
        fallback_value=M  # If eigendecomposition fails, return matrix as-is
    )
    
    return M


def ensure_positive_definite(
    M: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    warn: bool = False
) -> np.ndarray:
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to stabilize (assumed symmetric)
    min_eigenval : float, default MIN_EIGENVALUE
        Minimum eigenvalue to enforce
    warn : bool, default False
        Whether to log warnings
        
    Returns
    -------
    np.ndarray
        Positive semi-definite matrix
    """
    M = ensure_symmetric(M)
    
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    def _apply_regularization():
        eigenvals = np.linalg.eigh(M)[0]
        min_eig = float(np.min(eigenvals))
        
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M_reg = M + create_scaled_identity(M.shape[0], reg_amount, M.dtype)
            M_reg = ensure_symmetric(M_reg)
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal."
                )
            return M_reg
        return M
    
    def _fallback_regularization():
        M_reg = M + create_scaled_identity(M.shape[0], min_eigenval, M.dtype)
        M_reg = ensure_symmetric(M_reg)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed). "
                f"Added {min_eigenval:.2e} to diagonal."
            )
        return M_reg
    
    M = handle_linear_algebra_error(
        _apply_regularization, "matrix regularization",
        fallback_func=_fallback_regularization
    )
    
    return M


def ensure_covariance_stable(
    M: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE
) -> np.ndarray:
    """Ensure covariance matrix is symmetric and positive semi-definite.
    
    Parameters
    ----------
    M : np.ndarray
        Covariance matrix to stabilize
    min_eigenval : float, default MIN_EIGENVALUE
        Minimum eigenvalue to enforce
        
    Returns
    -------
    np.ndarray
        Stable covariance matrix
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    # Ensure symmetric and positive semi-definite
    return ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)


def ensure_process_noise_stable(
    Q: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    max_eigenval: float = MAX_EIGENVALUE,
    warn: bool = True,
    dtype: type = np.float32
) -> np.ndarray:
    """Ensure process noise covariance Q is stable with both minimum and maximum eigenvalue bounds.
    
    This function ensures Q (process noise) is:
    1. Positive definite (minimum eigenvalue >= min_eigenval)
    2. Bounded above (maximum eigenvalue <= max_eigenval)
    
    This prevents both singularity (from zero eigenvalues) and numerical explosion
    (from extremely large eigenvalues) in the Kalman filter.
    
    Parameters
    ----------
    Q : np.ndarray
        Process noise covariance matrix (m x m)
    min_eigenval : float, default MIN_EIGENVALUE
        Minimum eigenvalue to enforce (prevents singularity)
    max_eigenval : float, default MAX_EIGENVALUE
        Maximum eigenvalue to enforce (prevents explosion)
    warn : bool, default True
        Whether to log warnings when capping occurs
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    np.ndarray
        Stable process noise covariance matrix
    """
    if Q.size == 0 or Q.shape[0] == 0:
        return Q
    
    # Ensure minimum eigenvalue (positive definiteness)
    Q = ensure_covariance_stable(Q, min_eigenval=min_eigenval)
    
    # Cap maximum eigenvalue (prevent explosion)
    Q = cap_max_eigenval(Q, max_eigenval=max_eigenval, symmetric=True, warn=warn)
    
    return Q.astype(dtype)


def stabilize_innovation_covariance(
    Q: np.ndarray,
    min_eigenval: float = MIN_EIGENVALUE,
    min_floor: Optional[float] = None,
    max_eigenval: float = MAX_EIGENVALUE,
    dtype: type = np.float32
) -> np.ndarray:
    """Stabilize innovation covariance matrix Q with symmetrization, eigenvalue regularization, and floor.
    
    This is a common pattern used in VAR estimation to ensure Q is:
    1. Symmetric
    2. Positive semi-definite (with minimum eigenvalue)
    3. Bounded above (with maximum eigenvalue cap to prevent explosion)
    4. Floored to minimum values (typically MIN_Q_FLOOR)
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix (m x m)
    min_eigenval : float, default MIN_EIGENVALUE
        Minimum eigenvalue to enforce
    min_floor : float, optional
        Minimum floor value for all elements. If None, no floor is applied.
        Typically MIN_Q_FLOOR from constants.
    max_eigenval : float, default MAX_EIGENVALUE
        Maximum eigenvalue to enforce (prevents numerical explosion)
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    np.ndarray
        Stabilized covariance matrix
    """
    if Q.size == 0 or Q.shape[0] == 0:
        return Q
    
    # Ensure minimum and maximum eigenvalue bounds (generic process noise stabilization)
    Q = ensure_process_noise_stable(Q, min_eigenval=min_eigenval, max_eigenval=max_eigenval, warn=False, dtype=dtype)
    
    # Apply floor if specified
    if min_floor is not None:
        Q = np.maximum(Q, create_scaled_identity(Q.shape[0], min_floor, dtype))
    
    return Q.astype(dtype)


def compute_reg_param(
    matrix: np.ndarray,
    scale_factor: float = DEFAULT_REGULARIZATION_SCALE,
    warn: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """Compute regularization parameter for matrix inversion.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix for which to compute regularization
    scale_factor : float, default DEFAULT_REGULARIZATION_SCALE
        Base scale factor for regularization
    warn : bool, default True
        Whether to log warnings
        
    Returns
    -------
    reg_param : float
        Regularization parameter
    stats : dict
        Statistics about regularization computation
    """
    stats = {
        'regularized': False,
        'condition_number': None,
        'reg_amount': DEFAULT_ZERO_VALUE
    }
    
    if matrix.size == 0 or matrix.shape[0] == 0:
        return DEFAULT_ZERO_VALUE, stats
    
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
        cond_num = max_eig / max(min_eig, MIN_CONDITION_NUMBER)
        stats['condition_number'] = float(cond_num)
        
        if cond_num > MAX_CONDITION_NUMBER:
            reg_param = scale_factor * (cond_num / MAX_CONDITION_NUMBER)
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


def solve_regularized_ols(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = DEFAULT_REGULARIZATION_SCALE,
    use_XTX: bool = True,
    dtype: type = np.float32
) -> np.ndarray:
    """Solve regularized OLS: (X'X + reg*I)^(-1) X'y with fallback to pinv.
    
    This is a common pattern used throughout the codebase for solving
    regularized least squares problems with robust error handling.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (T x p) or covariance matrix (p x p) if use_XTX=False
    y : np.ndarray
        Target vector/matrix (T x n) or (p x n) if use_XTX=False
    regularization : float, default DEFAULT_REGULARIZATION_SCALE
        Regularization parameter
    use_XTX : bool, default True
        If True, X is design matrix and we compute X'X.
        If False, X is already X'X (covariance matrix).
    dtype : type, default np.float32
        Data type for computation
        
    Returns
    -------
    np.ndarray
        Solution coefficients (p x n) or (p,) if y is 1D
    """
    if use_XTX:
        # Standard OLS: (X'X + reg*I)^(-1) X'y
        try:
            XTX = X.T @ X
            XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype)
            # Handle both 1D and 2D y
            if y.ndim == 1:
                beta = np.linalg.solve(XTX_reg, X.T @ y)
            else:
                beta = np.linalg.solve(XTX_reg, X.T @ y).T
            return beta.astype(dtype)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to pinv
            if y.ndim == 1:
                beta = np.linalg.pinv(X) @ y
            else:
                beta = (np.linalg.pinv(X) @ y).T
            return beta.astype(dtype)
    else:
        # X is already X'X (covariance matrix)
        try:
            X_reg = X + create_scaled_identity(X.shape[0], regularization, dtype)
            if y.ndim == 1:
                beta = np.linalg.solve(X_reg, y)
            else:
                beta = np.linalg.solve(X_reg, y.T).T
            return beta.astype(dtype)
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to pinv
            if y.ndim == 1:
                beta = np.linalg.pinv(X) @ y
            else:
                beta = (np.linalg.pinv(X) @ y.T).T
            return beta.astype(dtype)


def safe_determinant(M: np.ndarray, use_logdet: bool = True) -> float:
    """Compute determinant safely to avoid overflow warnings.
    
    Uses log-determinant computation for large matrices or matrices with high
    condition numbers to avoid numerical overflow. For positive semi-definite
    matrices, uses Cholesky decomposition which is more stable.
    
    Parameters
    ----------
    M : np.ndarray
        Square matrix for which to compute determinant
    use_logdet : bool, default True
        Whether to use log-determinant computation (default: True)
        
    Returns
    -------
    float
        Determinant of M, or DEFAULT_ZERO_VALUE if computation fails
    """
    if M.size == 0 or M.shape[0] == 0:
        return DEFAULT_ZERO_VALUE
    
    if M.shape[0] != M.shape[1]:
        _logger.debug("safe_determinant: non-square matrix, returning 0.0")
        return DEFAULT_ZERO_VALUE
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(M)):
        _logger.debug("safe_determinant: matrix contains NaN/Inf, returning 0.0")
        return DEFAULT_ZERO_VALUE
    
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
            cond_num = max_eig / max(min_eig, MIN_CONDITION_NUMBER)
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
                log_det = CHOLESKY_LOG_DET_FACTOR * np.sum(np.log(np.diag(L)))
                # Check if log_det is too large to avoid overflow in exp
                if log_det > MAX_LOG_DETERMINANT:
                    _logger.debug("safe_determinant: log_det too large, returning 0.0")
                    return DEFAULT_ZERO_VALUE
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
                        return DEFAULT_ZERO_VALUE
                    # Avoid overflow in exp
                    if log_det > MAX_LOG_DETERMINANT:
                        _logger.debug("safe_determinant: log_det too large, returning 0.0")
                        return DEFAULT_ZERO_VALUE
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
    
    _logger.debug("safe_determinant: all methods failed, returning 0.0")
    return DEFAULT_ZERO_VALUE


# ============================================================================
# Analytical Computation Functions
# ============================================================================

def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = DEFAULT_ZERO_VALUE
) -> np.ndarray:
    """Safely divide arrays, handling zero denominators.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : np.ndarray
        Denominator array
    default : float, default DEFAULT_ZERO_VALUE
        Default value when denominator is zero (uses DEFAULT_ZERO_VALUE constant)
        
    Returns
    -------
    np.ndarray
        Division result
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, default),
            where=denominator != 0
        )
    return result


def compute_var_safe(
    data: np.ndarray,
    ddof: int = 0,
    min_variance: float = MIN_VARIANCE_COVARIANCE,
    default_variance: float = DEFAULT_VARIANCE_FALLBACK
) -> float:
    """Compute variance safely with robust error handling.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    ddof : int, default 0
        Delta degrees of freedom
    min_variance : float, default MIN_VARIANCE_COVARIANCE
        Minimum variance to enforce
    default_variance : float, default DEFAULT_VARIANCE_FALLBACK
        Default variance if computation fails
        
    Returns
    -------
    float
        Variance value
    """
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


def compute_cov_safe(
    data: np.ndarray,
    rowvar: bool = True,
    pairwise_complete: bool = False,
    min_eigenval: float = MIN_EIGENVALUE,
    fallback_to_identity: bool = True
) -> np.ndarray:
    """Compute covariance matrix safely with robust error handling.
    
    Parameters
    ----------
    data : np.ndarray
        Data array (T x N or N x T depending on rowvar)
    rowvar : bool, default True
        If True, each row represents a variable (N x T).
        If False, each column represents a variable (T x N).
    pairwise_complete : bool, default False
        If True, compute pairwise complete covariance
    min_eigenval : float, default MIN_EIGENVALUE
        Minimum eigenvalue to enforce for positive definiteness (uses MIN_EIGENVALUE constant)
    fallback_to_identity : bool, default True
        If True, fall back to identity matrix on failure
        
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N)
    """
    if data.size == 0:
        if fallback_to_identity:
            n = 1 if data.ndim == 1 else (data.shape[1] if rowvar else data.shape[0])
            return create_scaled_identity(n, DEFAULT_IDENTITY_SCALE)
        raise DataError(
            "Cannot compute covariance: data is empty",
            details="Input data has zero size. Provide non-empty data for covariance computation."
        )
    
    # Handle 1D case
    if data.ndim == 1:
        var_val = compute_var_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                   default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = compute_var_safe(series_data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
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
                            cov[i, j] = DEFAULT_ZERO_VALUE
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
                    raise DataError(
                        "Insufficient complete observations for covariance",
                        details=f"Only {np.sum(complete_rows)} complete rows available, need at least 2 for covariance computation"
                    )
                data_clean = data[complete_rows, :]
                data_for_cov = data_clean.T  # (N, T)
                cov = np.cov(data_for_cov, rowvar=True)  # Returns (N, N)
            else:
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise DataError(
                        "Insufficient complete observations for covariance",
                        details=f"Only {np.sum(complete_cols)} complete columns available, need at least 2 for covariance computation"
                    )
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
                    raise NumericalError(
                        f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})",
                        details="Covariance computation returned unexpected shape. This may indicate numerical issues with input data."
                    )
        
        # Ensure correct shape
        if cov.shape != (n_vars, n_vars):
            raise NumericalError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}",
                details="Covariance matrix has incorrect shape. This may indicate numerical issues or data preprocessing problems."
            )
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise NumericalError(
                "Covariance contains non-finite values",
                details="Covariance matrix contains NaN or Inf values. Check input data for missing values or numerical issues."
            )
        
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            cov = cov + create_scaled_identity(n_vars, reg_amount)
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return create_scaled_identity(n_vars, DEFAULT_IDENTITY_SCALE)
        raise


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
        return DEFAULT_ZERO_VALUE
    
    # Compute MSE only on non-missing values
    y_actual_valid = y_actual[mask]
    y_predicted_valid = y_predicted[mask]
    
    mse = float(np.mean((y_actual_valid - y_predicted_valid) ** 2))
    
    return mse


def convergence_checker(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
) -> Tuple[float, float]:
    """Check convergence of reconstruction error.
    
    Returns only delta and loss_now (no converged flag).
    
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
    # Match original: use boolean indexing like original's y_prev[~np.isnan(y_actual)]
    # Original: loss_minus = mse(y_prev[~np.isnan(y_actual)], y_actual[~np.isnan(y_actual)])
    # This flattens arrays and selects non-missing values
    mask = ~np.isnan(y_actual)
    
    # Flatten and select non-missing values (matching original's indexing)
    y_prev_flat = y_prev.flatten()
    y_now_flat = y_now.flatten()
    y_actual_flat = y_actual.flatten()
    mask_flat = mask.flatten()
    
    y_prev_valid = y_prev_flat[mask_flat]
    y_now_valid = y_now_flat[mask_flat]
    y_actual_valid = y_actual_flat[mask_flat]
    
    # Compute MSE (matching original sklearn.metrics.mean_squared_error)
    loss_prev = float(np.mean((y_actual_valid - y_prev_valid) ** 2))
    loss_now = float(np.mean((y_actual_valid - y_now_valid) ** 2))
    
    # Relative change (matching original: np.abs(loss - loss_minus) / loss_minus)
    # Edge case: When loss_prev is very small (< MIN_FACTOR_VARIANCE), use absolute difference
    # to avoid division by zero and numerical instability. This is appropriate because:
    # 1. Very small loss_prev indicates near-perfect fit, so absolute change is meaningful
    # 2. Relative change would be unstable (small denominator amplifies noise)
    # 3. This edge case is rare in practice (only when loss is extremely small)
    # Note: This does not contribute to fast convergence - fast convergence is due to actual
    # loss reduction, not edge case handling (verified: tolerance=0.0005 matches TensorFlow)
    if loss_prev < MIN_FACTOR_VARIANCE:
        # Avoid division by zero and numerical instability
        delta = float(abs(loss_now - loss_prev))
    else:
        delta = float(abs(loss_now - loss_prev) / loss_prev)
    
    return delta, loss_now


def check_convergence_with_tolerance(
    y_prev: np.ndarray,
    y_now: np.ndarray,
    y_actual: np.ndarray,
    tolerance: float,
    tolerance_multiplier: float = 10.0
) -> Tuple[float, bool]:
    """Check convergence with tolerance threshold.
    
    Wrapper around convergence_checker that adds tolerance-based convergence flag.
    
    Parameters
    ----------
    y_prev : np.ndarray
        Previous reconstruction (T x N)
    y_now : np.ndarray
        Current reconstruction (T x N)
    y_actual : np.ndarray
        Actual values (T x N) with NaN for missing values
    tolerance : float
        Convergence tolerance threshold
    tolerance_multiplier : float, default 10.0
        Multiplier for fallback delta when not finite
        
    Returns
    -------
    delta : float
        Relative change in loss: |loss_now - loss_prev| / loss_prev
    converged : bool
        Whether convergence criterion is met (delta < tolerance)
    """
    if y_prev is None:
        return float('inf'), False
    
    # Use numeric utility for convergence checking
    delta, _ = convergence_checker(
        y_prev=y_prev,
        y_now=y_now,
        y_actual=y_actual
    )
    
    # Ensure delta is finite
    if not np.isfinite(delta):
        _logger.warning(
            f"Convergence check: delta is not finite ({delta}). Using large default value"
        )
        delta = tolerance * tolerance_multiplier
    
    converged = delta < tolerance
    return delta, converged


def safe_matrix_power(
    matrix: Union[np.ndarray, Tensor],
    power: int,
    max_power: int = 1000,
    check_stability: bool = True
) -> Union[np.ndarray, Tensor]:
    """Safely compute matrix power with stability checks.
    
    This function computes matrix powers with numerical stability checks.
    Supports both NumPy arrays and PyTorch tensors.
    
    Parameters
    ----------
    matrix : np.ndarray or Tensor
        Matrix to raise to power (shape: (..., n, n))
    power : int
        Power to raise matrix to (must be >= 0)
    max_power : int, default=1000
        Maximum allowed power (safety check)
    check_stability : bool, default=True
        If True, checks for NaN/Inf in result
        
    Returns
    -------
    np.ndarray or Tensor
        Matrix raised to power (shape: (..., n, n))
        
    Raises
    ------
    ValueError
        If power < 0 or power > max_power
    NumericalError
        If result contains NaN/Inf and check_stability=True
    """
    if power < 0:
        raise DataValidationError(
            f"power must be >= 0, got {power}",
            details="Power parameter must be non-negative for matrix power computation"
        )
    if power > max_power:
        raise DataValidationError(
            f"power {power} exceeds maximum {max_power}. "
            f"This may indicate a configuration error.",
            details=f"Power parameter exceeds maximum allowed value {max_power}. Check configuration or reduce power value."
        )
    
    if power == 0:
        # Identity matrix
        n = matrix.shape[-1]
        if isinstance(matrix, Tensor):
            identity = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
            # Expand to match matrix batch dimensions
            while identity.dim() < matrix.dim():
                identity = identity.unsqueeze(0)
        else:
            identity = create_scaled_identity(n, DEFAULT_IDENTITY_SCALE, dtype=matrix.dtype)
            # Expand to match matrix batch dimensions
            while identity.ndim < matrix.ndim:
                identity = np.expand_dims(identity, axis=0)
        return identity
    
    # Compute matrix power
    if isinstance(matrix, Tensor):
        result = torch.matrix_power(matrix, power)
        # Check for numerical issues
        if check_stability:
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise NumericalError(
                    f"Matrix power computation resulted in NaN/Inf values. "
                    f"This may indicate numerical instability. "
                    f"Consider: (1) Regularization, (2) Lower power, (3) Checking matrix condition number."
                )
    else:
        result = np.linalg.matrix_power(matrix, power)
        # Check for numerical issues
        if check_stability:
            if np.any(~np.isfinite(result)):
                raise NumericalError(
                    f"Matrix power computation resulted in NaN/Inf values. "
                    f"This may indicate numerical instability. "
                    f"Consider: (1) Regularization, (2) Lower power, (3) Checking matrix condition number."
                )
    
    return result


def extract_matrix_block(
    matrix: Union[np.ndarray, Tensor],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int
) -> Union[np.ndarray, Tensor]:
    """Extract a block from a matrix.
    
    This utility is useful for extracting VAR coefficients from companion matrices.
    Supports both NumPy arrays and PyTorch tensors.
    
    Parameters
    ----------
    matrix : np.ndarray or Tensor
        Matrix to extract block from (shape: (..., m, n))
    row_start : int
        Start row index (inclusive)
    row_end : int
        End row index (exclusive)
    col_start : int
        Start column index (inclusive)
    col_end : int
        End column index (exclusive)
        
    Returns
    -------
    np.ndarray or Tensor
        Extracted block (shape: (..., row_end - row_start, col_end - col_start))
    """
    return matrix[..., row_start:row_end, col_start:col_end]


def compute_forecast_metrics(
    forecast: np.ndarray,
    actual: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute forecast evaluation metrics (RMSE, MAE, R²).
    
    Parameters
    ----------
    forecast : np.ndarray
        Forecast values of shape (horizon, n_vars) or (horizon * n_vars,)
    actual : np.ndarray
        Actual values of same shape as forecast
    mask : np.ndarray, optional
        Boolean mask to exclude certain values from computation
        
    Returns
    -------
    dict
        Dictionary with 'rmse', 'mae', 'r2' keys
    """
    # Flatten arrays
    forecast_flat = forecast.flatten()
    actual_flat = actual.flatten()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        forecast_flat = forecast_flat[mask_flat]
        actual_flat = actual_flat[mask_flat]
    else:
        # Remove NaN/Inf values
        valid_mask = ~(np.isnan(forecast_flat) | np.isnan(actual_flat) | 
                      np.isinf(forecast_flat) | np.isinf(actual_flat))
        forecast_flat = forecast_flat[valid_mask]
        actual_flat = actual_flat[valid_mask]
    
    if len(forecast_flat) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    
    # Compute metrics
    errors = forecast_flat - actual_flat
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    
    # Compute R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float(np.nan)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


__all__ = [
    # Matrix utilities
    'create_scaled_identity',
    # Matrix stability
    'ensure_symmetric',
    'clean_matrix',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_reg_param',
    'safe_determinant',
    # Analytical computations
    'safe_divide',
    'compute_var_safe',
    'compute_cov_safe',
    'mse_missing_numpy',
    'convergence_checker',
    'safe_matrix_power',
    'extract_matrix_block',
    'compute_forecast_metrics',
    'solve_regularized_ols',
]


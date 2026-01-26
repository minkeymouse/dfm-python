"""Numerical stability functions for matrix operations.

This module provides functions to ensure numerical stability of matrices,
including symmetry enforcement, positive definiteness, eigenvalue capping,
matrix cleaning, safe determinant computation, missing data handling,
and analytical computations.

**Function Selection Guide:**

- `ensure_positive_definite()`: O(m³) eigendecomposition approach. Use for critical 
  matrices that need exact positive semi-definite guarantee (e.g., initial covariance 
  matrices, critical model parameters). Expensive but guarantees exact PSD property.

- `ensure_covariance_stable()`: Wrapper around `ensure_positive_definite()`. Use for 
  general covariance matrices (observation noise R, state covariance V). Same O(m³) cost.

- `ensure_process_noise_stable()`: For process noise Q matrices requiring both min and 
  max eigenvalue bounds. Uses eigendecomposition when needed. Use for Q matrices that 
  must be bounded to prevent numerical explosion.

- Direct diagonal loading (in kalman.py): O(m²) approach used for high-frequency 
  operations (E-step in EM algorithm). Fast but less precise. Used in `update_parameters()` 
  and smoothing loops where speed is critical and matrices are already well-conditioned.

**Performance Considerations:**
- Eigendecomposition (O(m³)): Accurate but expensive, use sparingly
- Diagonal loading (O(m²)): Fast, suitable for high-frequency operations
- Choose based on frequency of operation and required precision
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any

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


# Removed clean_matrix - unused

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
    
    Uses O(m³) eigendecomposition to compute exact eigenvalues and apply precise 
    regularization. This is expensive but guarantees exact positive semi-definite 
    property. Use for critical matrices (initial covariances, model parameters) 
    where exact PSD property is required.
    
    For high-frequency operations (e.g., Kalman filter E-step), consider using 
    faster O(m²) diagonal loading instead (see kalman.py).
    
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
        
    Note
    ----
    Computational complexity: O(m³) due to eigendecomposition. For matrices that
    are already well-conditioned, consider faster diagonal loading approaches.
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
    
    Wrapper around `ensure_positive_definite()` for general covariance matrices.
    Uses O(m³) eigendecomposition. Use for observation noise R or state covariance V
    matrices where exact PSD property is important.
    
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
        
    Note
    ----
    Computational complexity: O(m³). For high-frequency operations, consider
    faster diagonal loading approaches used in kalman.py.
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


# Removed compute_reg_param - unused

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


# Removed safe_determinant - unused
# Removed safe_divide - unused


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


# Removed mse_missing_numpy - unused

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


__all__ = [
    # Matrix utilities
    'create_scaled_identity',
    # Matrix stability
    'ensure_symmetric',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_var_safe',
    'compute_cov_safe',
    'convergence_checker',
    'solve_regularized_ols',
]


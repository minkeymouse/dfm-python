"""Covariance and variance computation functions."""

import logging
import warnings
import numpy as np

_logger = logging.getLogger(__name__)

# Numerical stability constants
DEFAULT_VARIANCE_FALLBACK = 1.0  # Default variance when computation fails or result is invalid
MIN_VARIANCE_COVARIANCE = 1e-10  # Minimum variance threshold for covariance matrix diagonal


def _compute_covariance_safe(data: np.ndarray, rowvar: bool = True, 
                              pairwise_complete: bool = False,
                              min_eigenval: float = 1e-8,
                              fallback_to_identity: bool = True) -> np.ndarray:
    """Compute covariance matrix safely with robust error handling.
    
    This function computes covariance matrices with automatic handling of:
    - Missing data (NaN values)
    - Edge cases (empty data, single series, etc.)
    - Numerical instability (negative eigenvalues)
    - Fallback strategies for failed computations
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (T x N) where T is time periods and N is number of series.
        If rowvar=True, each row is a variable (standard). If rowvar=False, each column is a variable.
    rowvar : bool
        If True (default), each row represents a variable, each column an observation.
        If False, each column represents a variable, each row an observation.
    pairwise_complete : bool
        If True, use pairwise complete observations (more robust to missing data).
        If False, use listwise deletion (all variables must be observed simultaneously).
    min_eigenval : float
        Minimum eigenvalue threshold for positive semi-definiteness.
    fallback_to_identity : bool
        If True, return identity matrix if covariance computation fails.
        If False, raise an exception.
        
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N) where N is number of variables.
        Guaranteed to be positive semi-definite.
        
    Notes
    -----
    - For single series, returns variance as 1x1 matrix
    - For empty data, returns identity matrix if fallback_to_identity=True
    - Automatically regularizes if negative eigenvalues are found
    - Uses pairwise complete observations when pairwise_complete=True for robustness
    
    Examples
    --------
    >>> data = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 4.0]])
    >>> cov = _compute_covariance_safe(data, pairwise_complete=True)
    >>> assert cov.shape == (2, 2)
    >>> assert np.all(np.linalg.eigvalsh(cov) >= 0)  # PSD
    """
    if data.size == 0:
        if fallback_to_identity:
            return np.eye(1) if data.ndim == 1 else np.eye(data.shape[1] if rowvar else data.shape[0])
        raise ValueError("Cannot compute covariance: data is empty")
    
    # Handle 1D case
    if data.ndim == 1:
        var_val = _compute_variance_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE, 
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = _compute_variance_safe(series_data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Compute covariance
    try:
        if pairwise_complete:
            # Pairwise complete covariance: compute covariance for each pair separately
            # This is more robust when data has missing values
            if rowvar:
                # data is (T, N): each row is time, each column is series
                # We want (N, N) covariance matrix
                data_for_cov = data.T  # Transpose to (N, T) for np.cov
            else:
                # data is (N, T): each row is series, each column is time
                # We want (N, N) covariance matrix
                data_for_cov = data
            
            # Compute pairwise complete covariance manually
            # For each pair of variables, compute covariance using only complete observations
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    # Extract data for variables i and j
                    var_i = data_for_cov[i, :]
                    var_j = data_for_cov[j, :]
                    # Find complete observations (both non-NaN)
                    complete_mask = np.isfinite(var_i) & np.isfinite(var_j)
                    if np.sum(complete_mask) < 2:
                        # Not enough data, use default variance
                        if i == j:
                            cov[i, j] = DEFAULT_VARIANCE_FALLBACK
                        else:
                            cov[i, j] = 0.0
                    else:
                        # Compute covariance for this pair
                        var_i_complete = var_i[complete_mask]
                        var_j_complete = var_j[complete_mask]
                        if i == j:
                            # Variance
                            cov[i, j] = np.var(var_i_complete, ddof=0)
                        else:
                            # Covariance
                            mean_i = np.mean(var_i_complete)
                            mean_j = np.mean(var_j_complete)
                            cov[i, j] = np.mean((var_i_complete - mean_i) * (var_j_complete - mean_j))
                            cov[j, i] = cov[i, j]  # Symmetric
            
            # Ensure minimum variance
            np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE_COVARIANCE))
        else:
            # Standard covariance (listwise deletion)
            # Remove rows/columns with any NaN
            if rowvar:
                # data is (T, N): remove rows (time periods) with any NaN
                complete_rows = np.all(np.isfinite(data), axis=1)
                if np.sum(complete_rows) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[complete_rows, :]
                # np.cov with rowvar=True expects each row to be a variable
                # But our data is (T, N) where each row is time, each column is series
                # So we need to transpose: (N, T) where each row is series, each column is time
                data_for_cov = data_clean.T  # (N, T)
                cov = np.cov(data_for_cov, rowvar=True)  # Returns (N, N)
            else:
                # data is (N, T): remove columns (time periods) with any NaN
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[:, complete_cols]
                # np.cov with rowvar=False expects each column to be a variable
                # Our data is (N, T) where each row is series, each column is time
                # So we need: (N, T) where each column is series, each row is time
                # Actually, np.cov with rowvar=False treats each column as variable
                # So we need to transpose: (T, N) where each column is series
                data_for_cov = data_clean.T  # (T, N)
                cov = np.cov(data_for_cov, rowvar=False)  # Returns (N, N)
            
            # np.cov can sometimes return unexpected shapes, so verify
            if cov.ndim == 0:
                # Single value case
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                # 1D case - convert to 2D
                if len(cov) == n_vars:
                    cov = np.diag(cov)
                else:
                    raise ValueError(f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})")
        
        # Ensure correct shape - critical check to prevent broadcasting errors
        if cov.shape != (n_vars, n_vars):
            raise ValueError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}"
            )
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise ValueError("Covariance contains non-finite values")
        
        # Ensure positive semi-definite
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            # Regularize if needed
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            # Double-check shape before broadcasting to prevent errors
            if cov.shape != (n_vars, n_vars):
                raise ValueError(
                    f"Cannot regularize: cov shape {cov.shape} != expected ({n_vars}, {n_vars})"
                )
            eye_matrix = np.eye(n_vars)
            if eye_matrix.shape != cov.shape:
                raise ValueError(
                    f"Cannot regularize: eye shape {eye_matrix.shape} != cov shape {cov.shape}"
                )
            cov = cov + eye_matrix * reg_amount
            # Re-verify shape after regularization
            if cov.shape != (n_vars, n_vars):
                raise ValueError(
                    f"Shape changed after regularization: {cov.shape} != expected ({n_vars}, {n_vars})"
                )
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return np.eye(n_vars)
        raise


def _compute_variance_safe(data: np.ndarray, ddof: int = 0, 
                           min_variance: float = MIN_VARIANCE_COVARIANCE,
                           default_variance: float = DEFAULT_VARIANCE_FALLBACK) -> float:
    """Compute variance safely with robust error handling.
    
    This function computes variance with automatic handling of:
    - Missing data (NaN values)
    - Edge cases (empty data, insufficient samples)
    - Numerical instability (non-finite results)
    - Minimum variance threshold enforcement
    
    Parameters
    ----------
    data : np.ndarray
        Data array (1D or 2D). If 2D, variance is computed over all elements.
    ddof : int
        Delta degrees of freedom. Default is 0 (population variance).
    min_variance : float
        Minimum allowed variance threshold. Default is MIN_VARIANCE_COVARIANCE (1e-10).
        Values below this are replaced with default_variance.
    default_variance : float
        Default variance value to use when computation fails or result is invalid.
        Default is DEFAULT_VARIANCE_FALLBACK (1.0).
        
    Returns
    -------
    float
        Variance value, guaranteed to be finite and >= min_variance.
        
    Notes
    -----
    - Uses np.nanvar for automatic NaN handling
    - Returns default_variance if result is NaN, Inf, or < min_variance
    - Flattens 2D arrays before computation
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, np.nan, 4.0])
    >>> var = _compute_variance_safe(data)
    >>> assert var >= 1e-10  # Minimum threshold
    >>> assert np.isfinite(var)  # Always finite
    """
    if data.size == 0:
        return default_variance
    
    # Flatten if 2D
    if data.ndim > 1:
        data = data.flatten()
    
    # Compute variance with NaN handling
    # Suppress warning when ddof >= number of non-NaN values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_val = np.nanvar(data, ddof=ddof)
    
    # Validate and enforce minimum
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    
    return float(var_val)

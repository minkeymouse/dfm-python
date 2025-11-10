from typing import Optional, Tuple, Dict, Any
import logging
import numpy as np

try:
    from scipy.sparse.linalg import eigs
    from scipy.sparse import csc_matrix
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    eigs = None
    csc_matrix = None

_logger = logging.getLogger(__name__)


def _check_finite(array: np.ndarray, name: str = "array", raise_on_invalid: bool = False) -> bool:
    """Check if array contains only finite values.
    
    Parameters
    ----------
    array : np.ndarray
        Array to check
    name : str
        Name of array for logging
    raise_on_invalid : bool
        If True, raise ValueError on invalid values. If False, only log warning.
        
    Returns
    -------
    bool
        True if all values are finite, False otherwise
        
    Raises
    ------
    ValueError
        If raise_on_invalid=True and array contains non-finite values
    """
    has_nan = np.any(np.isnan(array))
    has_inf = np.any(np.isinf(array))
    
    if has_nan or has_inf:
        msg = f"{name} contains "
        issues = []
        if has_nan:
            issues.append(f"{np.sum(np.isnan(array))} NaN values")
        if has_inf:
            issues.append(f"{np.sum(np.isinf(array))} Inf values")
        msg += " and ".join(issues)
        
        if raise_on_invalid:
            raise ValueError(msg)
        else:
            _logger.warning(msg)
        return False
    return True


def _ensure_square_matrix(M: np.ndarray, method: str = 'diag') -> np.ndarray:
    """Ensure matrix is square by extracting diagonal if needed.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to ensure is square
    method : str
        Method to use if matrix is not square:
        - 'diag': Extract diagonal (default)
        - 'eye': Return identity matrix of appropriate size
        
    Returns
    -------
    np.ndarray
        Square matrix (n x n)
    """
    if M.size == 0:
        return M
    if M.shape[0] != M.shape[1]:
        if method == 'diag':
            return np.diag(np.diag(M))
        elif method == 'eye':
            size = max(M.shape[0], M.shape[1])
            return np.eye(size)
    return M


def _ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to symmetrize
        
    Returns
    -------
    np.ndarray
        Symmetric matrix (M + M.T) / 2
    """
    return 0.5 * (M + M.T)


def _ensure_real(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real by extracting real part if complex.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix that may be complex due to numerical errors
        
    Returns
    -------
    np.ndarray
        Real matrix (extracts real part if complex)
    """
    if np.iscomplexobj(M):
        return np.real(M)
    return M


def _ensure_real_and_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real and symmetric.
    
    This is a common operation in Kalman filtering where covariance matrices
    should be real and symmetric, but numerical errors can introduce complex
    values or asymmetry.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to process (may be complex or asymmetric due to numerical errors)
        
    Returns
    -------
    np.ndarray
        Real, symmetric matrix
    """
    M = _ensure_real(M)
    M = _ensure_symmetric(M)
    return M


def _ensure_covariance_stable(M: np.ndarray, min_eigenval: float = 1e-8,
                               ensure_real: bool = True) -> np.ndarray:
    """Ensure covariance matrix is real, symmetric, and positive semi-definite.
    
    This is a comprehensive function for stabilizing covariance matrices in
    numerical computations, commonly used in Kalman filtering.
    
    Parameters
    ----------
    M : np.ndarray
        Covariance matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue threshold for positive semi-definiteness
    ensure_real : bool
        If True, extract real part if matrix is complex
        
    Returns
    -------
    np.ndarray
        Stabilized covariance matrix (real, symmetric, PSD)
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    # Step 1: Ensure real (if needed)
    if ensure_real:
        M = _ensure_real(M)
    
    # Step 2: Ensure symmetric
    M = _ensure_symmetric(M)
    
    # Step 3: Ensure positive semi-definite
    try:
        eigenvals = np.linalg.eigvalsh(M)  # Use eigvalsh for symmetric matrices
        min_eig = np.min(eigenvals)
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0]) * reg_amount
            M = _ensure_symmetric(M)
    except (np.linalg.LinAlgError, ValueError):
        # Eigendecomposition failed - add regularization
        M = M + np.eye(M.shape[0]) * min_eigenval
        M = _ensure_symmetric(M)
    
    return M


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
        var_val = np.nanvar(data, ddof=0)
        var_val = 1.0 if (np.isnan(var_val) or np.isinf(var_val) or var_val < 1e-10) else var_val
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = np.nanvar(series_data, ddof=0)
        var_val = 1.0 if (np.isnan(var_val) or np.isinf(var_val) or var_val < 1e-10) else var_val
        return np.array([[var_val]])
    
    # Compute covariance
    try:
        if pairwise_complete:
            # Use pairwise complete observations (more robust)
            cov = np.cov(data.T, rowvar=False) if rowvar else np.cov(data, rowvar=True)
        else:
            # Standard covariance (listwise deletion)
            cov = np.cov(data, rowvar=rowvar)
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise ValueError("Covariance contains non-finite values")
        
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            # Regularize if needed
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            cov = cov + np.eye(n_vars) * reg_amount
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return np.eye(n_vars)
        raise


def _ensure_innovation_variance_minimum(Q: np.ndarray, min_variance: float = 1e-8) -> np.ndarray:
    """Ensure innovation covariance matrix Q has minimum diagonal values.
    
    This is critical for factor evolution: if Q[i,i] = 0, factor i cannot evolve
    (innovation variance is zero). This function enforces a minimum variance
    threshold on the diagonal elements while preserving off-diagonal structure.
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix (m x m) where m is the state dimension.
        Can be any square matrix representing innovation variances.
    min_variance : float
        Minimum allowed variance for each diagonal element. Default is 1e-8.
        This ensures factors can evolve even with very small innovations.
        
    Returns
    -------
    np.ndarray
        Q matrix with guaranteed minimum diagonal values. Off-diagonal elements
        are preserved unchanged.
        
    Notes
    -----
    - Only modifies diagonal elements, preserving correlation structure
    - If Q is empty or non-square, returns Q unchanged
    - This is a common operation in DFM initialization and EM steps
    
    Examples
    --------
    >>> Q = np.array([[0.0, 0.1], [0.1, 0.0]])  # Zero diagonal
    >>> Q_safe = _ensure_innovation_variance_minimum(Q, min_variance=1e-8)
    >>> assert np.all(np.diag(Q_safe) >= 1e-8)  # All diagonals >= 1e-8
    """
    if Q.size == 0 or Q.shape[0] == 0 or Q.shape[0] != Q.shape[1]:
        return Q
    
    Q_diag = np.diag(Q)
    Q_diag = np.maximum(Q_diag, min_variance)
    # Preserve off-diagonal elements: Q_new = diag(Q_diag) + (Q - diag(Q))
    Q = np.diag(Q_diag) + Q - np.diag(np.diag(Q))
    return Q


def _compute_principal_components(cov_matrix: np.ndarray, n_components: int,
                                   block_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition with fallbacks."""
    if cov_matrix.size == 1:
        eigenvector = np.array([[1.0]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else 1.0
        return np.array([eigenvalue]), eigenvector
    
    n_series = cov_matrix.shape[0]
    
    # Strategy 1: Sparse eigs when feasible
    if n_components < n_series - 1 and SCIPY_SPARSE_AVAILABLE:
        try:
            cov_sparse = csc_matrix(cov_matrix)
            eigenvalues, eigenvectors = eigs(cov_sparse, k=n_components, which='LM')
            eigenvectors = eigenvectors.real
            if np.any(~np.isfinite(eigenvalues)) or np.any(~np.isfinite(eigenvectors)):
                raise ValueError("Invalid eigenvalue results")
            return eigenvalues.real, eigenvectors
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if block_idx is not None:
                _logger.warning(
                    f"init_conditions: Sparse eigendecomposition failed for block {block_idx+1}, "
                    f"falling back to np.linalg.eig. Error: {type(e).__name__}"
                )
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
            return eigenvalues[sort_idx].real, eigenvectors[:, sort_idx].real
    
    # Strategy 2: Full eig
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        valid_mask = np.isfinite(eigenvalues)
        if np.sum(valid_mask) < n_components:
            raise ValueError("Not enough valid eigenvalues")
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask]
        sort_idx = np.argsort(np.abs(valid_eigenvalues))[::-1][:n_components]
        return valid_eigenvalues[sort_idx].real, valid_eigenvectors[:, sort_idx].real
    except (IndexError, ValueError, np.linalg.LinAlgError) as e:
        if block_idx is not None:
            _logger.warning(
                f"init_conditions: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = np.eye(n_series)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def _clean_matrix(M: np.ndarray, matrix_type: str = 'general', 
                  default_nan: float = 0.0, default_inf: Optional[float] = None) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability."""
    if matrix_type == 'covariance':
        M = np.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = _ensure_symmetric(M)
        try:
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < 1e-8:
                M = M + np.eye(M.shape[0]) * (1e-8 - min_eigenval)
                M = _ensure_symmetric(M)
        except (np.linalg.LinAlgError, ValueError):
            M = M + np.eye(M.shape[0]) * 1e-8
            M = _ensure_symmetric(M)
    elif matrix_type == 'diagonal':
        diag = np.diag(M)
        diag = np.nan_to_num(diag, nan=default_nan, 
                            posinf=default_inf if default_inf is not None else 1e4,
                            neginf=default_nan)
        diag = np.maximum(diag, 1e-6)
        M = np.diag(diag)
    elif matrix_type == 'loading':
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M


def _ensure_positive_definite(M: np.ndarray, min_eigenval: float = 1e-8, 
                              warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ensure matrix is positive semi-definite by adding regularization if needed."""
    M = _ensure_symmetric(M)
    stats = {
        'regularized': False,
        'min_eigenval_before': None,
        'reg_amount': 0.0,
        'min_eigenval_after': None
    }
    if M.size == 0 or M.shape[0] == 0:
        return M, stats
    try:
        eigenvals = np.linalg.eigvalsh(M)
        min_eig = float(np.min(eigenvals))
        stats['min_eigenval_before'] = float(min_eig)
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0]) * reg_amount
            M = _ensure_symmetric(M)
            stats['regularized'] = True
            stats['reg_amount'] = float(reg_amount)
            eigenvals_after = np.linalg.eigvalsh(M)
            stats['min_eigenval_after'] = float(np.min(eigenvals_after))
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal. This biases the covariance matrix."
                )
        else:
            stats['min_eigenval_after'] = float(min_eig)
    except (np.linalg.LinAlgError, ValueError) as e:
        M = M + np.eye(M.shape[0]) * min_eigenval
        M = _ensure_symmetric(M)
        stats['regularized'] = True
        stats['reg_amount'] = float(min_eigenval)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed: {e}). "
                f"Added {min_eigenval:.2e} to diagonal. This biases the covariance matrix."
            )
    return M, stats


def _compute_regularization_param(matrix: np.ndarray, scale_factor: float = 1e-5, 
                                  warn: bool = True) -> Tuple[float, Dict[str, Any]]:
    """Compute regularization parameter based on matrix scale."""
    trace = np.trace(matrix)
    reg_param = max(trace * scale_factor, 1e-8)
    stats = {'trace': float(trace), 'scale_factor': float(scale_factor), 'reg_param': float(reg_param)}
    if warn and reg_param > 1e-8:
        _logger.info(
            f"Regularization parameter computed: {reg_param:.2e} "
            f"(trace={trace:.2e}, scale={scale_factor:.2e})."
        )
    return reg_param, stats


def _clip_ar_coefficients(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
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
    """Apply AR coefficient clipping based on configuration.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix to clip
    config : object, optional
        Configuration object with clipping parameters. If None, uses defaults.
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped transition matrix
    stats : dict
        Statistics about clipping operation
    """
    if config is None:
        return _clip_ar_coefficients(A, -0.99, 0.99, True)
    
    from ..core.em import _get_config_param
    
    clip_enabled = _get_config_param(config, 'clip_ar_coefficients', True)
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = _get_config_param(config, 'ar_clip_min', -0.99)
    max_val = _get_config_param(config, 'ar_clip_max', 0.99)
    warn = _get_config_param(config, 'warn_on_ar_clip', True)
    return _clip_ar_coefficients(A, min_val, max_val, warn)


def _cap_max_eigenvalue(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
    """Cap maximum eigenvalue of a matrix to prevent numerical explosion."""
    try:
        eigenvals = np.linalg.eigvals(M)
        max_eig = np.max(eigenvals)
        if max_eig > max_eigenval:
            scale = max_eigenval / max_eig
            return M * scale
    except (np.linalg.LinAlgError, ValueError):
        M_diag = np.diag(M)
        M_diag = np.maximum(M_diag, 1e-8)
        M_diag = np.minimum(M_diag, max_eigenval)
        M_capped = np.diag(M_diag)
        return _ensure_symmetric(M_capped)
    return M


def _estimate_ar_coefficient(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
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
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * 1e-6, 1e-10)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0) if EZZ_FB_diag.ndim > 0 else np.nan_to_num(EZZ_FB_diag, nan=0.0, posinf=1e6, neginf=-1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    if T is not None:
        Q_diag = None
    else:
        Q_diag = None
    return A_diag, Q_diag


def _compute_variance_safe(data: np.ndarray, ddof: int = 0, 
                           min_variance: float = 1e-10,
                           default_variance: float = 1.0) -> float:
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
        Minimum allowed variance threshold. Default is 1e-10.
        Values below this are replaced with default_variance.
    default_variance : float
        Default variance value to use when computation fails or result is invalid.
        Default is 1.0.
        
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
    var_val = np.nanvar(data, ddof=ddof)
    
    # Validate and enforce minimum
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    
    return float(var_val)


def _safe_divide(numerator: np.ndarray, denominator: float, default: float = 0.0) -> np.ndarray:
    """Safely divide numerator by denominator, handling zero and invalid values."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.full_like(numerator, default)
    result = numerator / denominator
    result = np.where(np.isfinite(result), result, default)
    return result



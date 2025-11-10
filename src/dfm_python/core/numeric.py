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
    """Ensure matrix is symmetric by averaging with its transpose."""
    return 0.5 * (M + M.T)


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


def _safe_divide(numerator: np.ndarray, denominator: float, default: float = 0.0) -> np.ndarray:
    """Safely divide numerator by denominator, handling zero and invalid values."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.full_like(numerator, default)
    result = numerator / denominator
    result = np.where(np.isfinite(result), result, default)
    return result



"""State-space utility functions.

This module contains utility functions extracted from state_space.py
to keep the main file under 1000 lines.
"""

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


# Re-export for backward compatibility
# These functions are used by other modules
def _compute_principal_components(cov_matrix: np.ndarray, n_components: int,
                                   block_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition with fallbacks."""
    try:
        from scipy.sparse.linalg import eigs
        from scipy.sparse import csc_matrix
        SCIPY_SPARSE_AVAILABLE = True
    except ImportError:
        SCIPY_SPARSE_AVAILABLE = False
        eigs = None
        csc_matrix = None
    
    if cov_matrix.size == 1:
        eigenvector = np.array([[1.0]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else DEFAULT_VARIANCE_FALLBACK
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


def _compute_covariance_safe(data: np.ndarray, rowvar: bool = True, 
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


def _compute_variance_safe(data: np.ndarray, ddof: int = 0, 
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
    """Apply AR coefficient clipping based on configuration."""
    if config is None:
        return _clip_ar_coefficients(A, -0.99, 0.99, True)
    
    from .helpers import safe_get_attr
    
    clip_enabled = safe_get_attr(config, 'clip_ar_coefficients', True)
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = safe_get_attr(config, 'ar_clip_min', -0.99)
    max_val = safe_get_attr(config, 'ar_clip_max', 0.99)
    warn = safe_get_attr(config, 'warn_on_ar_clip', True)
    return _clip_ar_coefficients(A, min_val, max_val, warn)


def _compute_regularization_param(
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

def _cap_max_eigenvalue(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
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
            scale_factor = max_eigenval / max_eig
            M = M * scale_factor
            M = _ensure_symmetric(M)
    
    except (np.linalg.LinAlgError, ValueError):
        # If eigendecomposition fails, return matrix as-is
        pass
    
    return M


def _ensure_innovation_variance_minimum(Q: np.ndarray, min_variance: float = 1e-8) -> np.ndarray:
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

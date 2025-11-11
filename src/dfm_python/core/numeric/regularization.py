"""Regularization and positive semi-definite enforcement functions."""

from typing import Optional, Tuple, Dict, Any
import logging
import numpy as np

_logger = logging.getLogger(__name__)

# Import from matrix module for symmetrization
from .matrix import _ensure_symmetric, MIN_EIGENVAL_CLEAN


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


def _ensure_positive_definite(M: np.ndarray, min_eigenval: float = 1e-8, 
                              warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    This function ensures a matrix is positive semi-definite (PSD) by adding
    a diagonal regularization term if the minimum eigenvalue is below the threshold.
    The matrix is first symmetrized, then regularized if necessary.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to regularize (any shape, will be symmetrized first)
    min_eigenval : float, default 1e-8
        Minimum eigenvalue threshold. If minimum eigenvalue < min_eigenval,
        regularization is applied by adding (min_eigenval - min_eig) * I to diagonal.
    warn : bool, default True
        If True, log warning when regularization is applied
        
    Returns
    -------
    M_regularized : np.ndarray
        Positive semi-definite matrix (guaranteed min eigenvalue >= min_eigenval)
    stats : dict
        Statistics dictionary with:
        - 'regularized': bool, whether regularization was applied
        - 'min_eigenval_before': float, minimum eigenvalue before regularization
        - 'reg_amount': float, amount of regularization added to diagonal
        - 'min_eigenval_after': float, minimum eigenvalue after regularization
        
    Notes
    -----
    - Matrix is first symmetrized via _ensure_symmetric()
    - Regularization adds a diagonal term: M + reg_amount * I
    - Used extensively in covariance matrix stabilization (Kalman filter, EM algorithm)
    - Regularization biases the matrix but ensures numerical stability
    - If eigendecomposition fails, applies conservative regularization (min_eigenval * I)
    """
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
    """Compute regularization parameter based on matrix scale.
    
    Computes a regularization parameter as a fraction of the matrix trace.
    This is used to stabilize matrix inversions by adding a small diagonal term.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix for which to compute regularization parameter
    scale_factor : float, default 1e-5
        Scaling factor applied to matrix trace (typically 1e-5 to 1e-3)
    warn : bool, default True
        If True, log info message when regularization is applied
        
    Returns
    -------
    reg_param : float
        Regularization parameter (max(trace * scale_factor, 1e-8))
        Minimum value of 1e-8 ensures numerical stability
    stats : dict
        Dictionary with 'trace', 'scale_factor', and 'reg_param' values
        
    Notes
    -----
    - Used in EM step for loading matrix (C) updates to prevent singular matrices
    - Minimum value of 1e-8 ensures regularization even for very small traces
    - Formula: reg_param = max(trace(matrix) * scale_factor, 1e-8)
    """
    trace = np.trace(matrix)
    reg_param = max(trace * scale_factor, 1e-8)
    stats = {'trace': float(trace), 'scale_factor': float(scale_factor), 'reg_param': float(reg_param)}
    if warn and reg_param > 1e-8:
        _logger.info(
            f"Regularization parameter computed: {reg_param:.2e} "
            f"(trace={trace:.2e}, scale={scale_factor:.2e})."
        )
    return reg_param, stats


def _cap_max_eigenvalue(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
    """Cap maximum eigenvalue of a matrix to prevent numerical explosion.
    
    If the maximum eigenvalue exceeds max_eigenval, the entire matrix is scaled
    down proportionally to bring the maximum eigenvalue to max_eigenval.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (any shape)
    max_eigenval : float, default 1e6
        Maximum allowed eigenvalue
        
    Returns
    -------
    np.ndarray
        Matrix with capped eigenvalues (scaled if necessary)
        
    Notes
    -----
    - If eigendecomposition fails, falls back to diagonal capping
    - Preserves matrix structure (scales entire matrix, not just eigenvalues)
    """
    try:
        eigenvals = np.linalg.eigvals(M)
        max_eig = np.max(eigenvals)
        if max_eig > max_eigenval:
            scale = max_eigenval / max_eig
            return M * scale
    except (np.linalg.LinAlgError, ValueError):
        M_diag = np.diag(M)
        M_diag = np.maximum(M_diag, MIN_EIGENVAL_CLEAN)
        M_diag = np.minimum(M_diag, max_eigenval)
        M_capped = np.diag(M_diag)
        return _ensure_symmetric(M_capped)
    return M


def _estimate_ar_coefficient(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
                             vsmooth_sum: Optional[np.ndarray] = None,
                             T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR coefficients and innovation variances from expectations.
    
    Computes AR(1) coefficients from smoothed factor expectations:
    A = E[Z_t Z_{t-1}'] / E[Z_{t-1} Z_{t-1}']
    
    Parameters
    ----------
    EZZ_FB : np.ndarray
        Expected cross-covariance E[Z_t Z_{t-1}'] (diagonal extracted)
    EZZ_BB : np.ndarray
        Expected lag covariance E[Z_{t-1} Z_{t-1}'] (diagonal extracted)
    vsmooth_sum : np.ndarray, optional
        Additional variance terms from smoothing (added to EZZ_BB)
    T : int, optional
        Number of time periods. Currently unused but reserved for future
        computation of innovation variances Q_diag.
        
    Returns
    -------
    A_diag : np.ndarray
        AR coefficients (diagonal elements)
    Q_diag : np.ndarray
        Innovation variances. Currently always None, but reserved for future
        implementation when T parameter is utilized.
        
    Notes
    -----
    - Handles scalar, 1D, and 2D input arrays
    - Enforces minimum denominator threshold to prevent division by zero
    - Cleans non-finite values in numerator before division
    - Future: Q_diag computation will use T parameter for proper scaling
    """
    from .matrix import _clean_matrix, MIN_DIAGONAL_VARIANCE
    from .covariance import MIN_VARIANCE_COVARIANCE
    
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
    # Use _clean_matrix for consistency (handles both scalar and array cases)
    if EZZ_FB_diag.ndim == 0:
        # Scalar case: convert to array, clean, then extract scalar
        EZZ_FB_diag_clean = _clean_matrix(np.array([EZZ_FB_diag]), 'general', default_nan=0.0, default_inf=1e6)
        EZZ_FB_diag = EZZ_FB_diag_clean[0] if EZZ_FB_diag_clean.size > 0 else 0.0
    else:
        EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0, default_inf=1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    # Q_diag computation reserved for future implementation when T parameter is utilized
    Q_diag = None
    return A_diag, Q_diag

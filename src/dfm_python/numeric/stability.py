"""Numerical stability functions for matrix operations.

This module provides functions to ensure numerical stability of matrices,
including symmetry enforcement, positive definiteness, eigenvalue capping,
matrix cleaning, and safe determinant computation.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any

from ..logger import get_logger
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
    DEFAULT_MAX_VARIANCE,
    MAX_LOG_DETERMINANT,
)

_logger = get_logger(__name__)

# Numerical stability constants
MIN_EIGENVAL_CLEAN = MIN_EIGENVALUE
MIN_VARIANCE_COVARIANCE = MIN_FACTOR_VARIANCE


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


def clean_matrix(
    M: np.ndarray,
    matrix_type: Optional[str] = None,
    default_nan: float = 0.0,
    default_inf: Optional[float] = None
) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to clean
    matrix_type : str, optional
        Type of matrix: 'covariance', 'diagonal', 'loading', or 'general'
    default_nan : float, default 0.0
        Default value for NaN replacement
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
        try:
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < MIN_EIGENVAL_CLEAN:
                M = M + np.eye(M.shape[0]) * (MIN_EIGENVAL_CLEAN - min_eigenval)
                M = ensure_symmetric(M)
        except (np.linalg.LinAlgError, ValueError):
            M = M + np.eye(M.shape[0]) * MIN_EIGENVAL_CLEAN
            M = ensure_symmetric(M)
    elif matrix_type == MATRIX_TYPE_DIAGONAL:
        diag = np.diag(M)
        default_inf_val = default_inf if default_inf is not None else DEFAULT_MAX_VARIANCE
        diag = np.nan_to_num(
            diag,
            nan=default_nan,
            posinf=default_inf_val,
            neginf=default_nan
        )
        diag = np.maximum(diag, MIN_DIAGONAL_VARIANCE)
        M = np.diag(diag)
    elif matrix_type == MATRIX_TYPE_LOADING:
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
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
    
    try:
        if symmetric:
            eigenvals = np.linalg.eigvalsh(M)
        else:
            eigenvals = np.linalg.eigvals(M)
        max_eig = float(np.max(np.abs(eigenvals)))
        
        if max_eig > max_eigenval:
            scale_factor = max_eigenval / max_eig
            M = M * scale_factor
            if symmetric:
                M = ensure_symmetric(M)
            if warn:
                _logger.warning(
                    f"Matrix maximum eigenvalue capped: {max_eig:.2e} -> {max_eigenval:.2e} "
                    f"(scale_factor={scale_factor:.2e})"
                )
    except (np.linalg.LinAlgError, ValueError):
        # If eigendecomposition fails, return matrix as-is
        pass
    
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
        cond_num = max_eig / max(min_eig, MIN_CONDITION_NUMBER)
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
        Determinant of M, or 0.0 if computation fails
    """
    if M.size == 0 or M.shape[0] == 0:
        return 0.0
    
    if M.shape[0] != M.shape[1]:
        _logger.debug("safe_determinant: non-square matrix, returning 0.0")
        return 0.0
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(M)):
        _logger.debug("safe_determinant: matrix contains NaN/Inf, returning 0.0")
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
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                # Check if log_det is too large to avoid overflow in exp
                if log_det > MAX_LOG_DETERMINANT:
                    _logger.debug("safe_determinant: log_det too large, returning 0.0")
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
                    if log_det > MAX_LOG_DETERMINANT:
                        _logger.debug("safe_determinant: log_det too large, returning 0.0")
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
    
    _logger.debug("safe_determinant: all methods failed, returning 0.0")
    return 0.0


__all__ = [
    'ensure_symmetric',
    'clean_matrix',
    'cap_max_eigenval',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'compute_reg_param',
    'safe_determinant',
]


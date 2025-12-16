"""State-space model (SSM) utility functions.

This module provides numerical stability utilities for PyTorch-based SSM operations:
- Matrix validation and cleaning
- Covariance matrix stabilization
- Safe matrix operations (inverse, determinant)
- Numerical stability for Kalman filtering and EM algorithm

These utilities are critical for GPU numerical stability, as PyTorch can throw
RuntimeError for near-singular matrices (e.g., "cholesky_cpu: U(0,0) is zero" or
"inverse_cuda: singular matrix").
"""

from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import torch

import torch
from ..logger import get_logger

_logger = get_logger(__name__)

# Default numerical stability constants for PyTorch operations
# Note: Similar constants exist in utils/statespace.py (MIN_EIGENVAL_CLEAN, MIN_DIAGONAL_VARIANCE)
# for NumPy operations. They have the same values but are kept separate for context clarity.
DEFAULT_MIN_EIGENVAL = 1e-8
DEFAULT_MIN_DIAGONAL_VARIANCE = 1e-6
DEFAULT_INV_REGULARIZATION = 1e-6


def check_finite(tensor: "torch.Tensor", name: str = "tensor") -> bool:
    """Check if tensor contains only finite values.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    name : str
        Name for error messages
        
    Returns
    -------
    bool
        True if tensor is finite, False otherwise
    """
    has_nan = torch.any(torch.isnan(tensor))
    has_inf = torch.any(torch.isinf(tensor))
    
    if has_nan or has_inf:
        nan_count = torch.sum(torch.isnan(tensor)).item()
        inf_count = torch.sum(torch.isinf(tensor)).item()
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


def ensure_real(tensor: "torch.Tensor") -> "torch.Tensor":
    """Ensure tensor is real by extracting real part if complex.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to ensure is real
        
    Returns
    -------
    torch.Tensor
        Real tensor
    """
    if torch.is_complex(tensor):
        return torch.real(tensor)
    return tensor


def ensure_symmetric(tensor: "torch.Tensor") -> "torch.Tensor":
    """Ensure matrix is symmetric by averaging with its transpose.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Matrix to symmetrize
        
    Returns
    -------
    torch.Tensor
        Symmetric matrix
    """
    return 0.5 * (tensor + tensor.T)


def cap_max_eigenval(
    M: "torch.Tensor", 
    max_eigenval: float = 1e6,
    warn: bool = True
) -> "torch.Tensor":
    """Cap maximum eigenvalue of matrix to prevent numerical explosion.
    
    Parameters
    ----------
    M : torch.Tensor
        Matrix to cap (square matrix)
    max_eigenval : float, default 1e6
        Maximum allowed eigenvalue
    warn : bool, default True
        Whether to log warnings
        
    Returns
    -------
    torch.Tensor
        Matrix with capped eigenvalues
    """
    if M.numel() == 0 or M.shape[0] == 0:
        return M
    
    try:
        eigenvals = torch.linalg.eigvalsh(M)
        max_eig = torch.max(eigenvals)
        
        if max_eig > max_eigenval:
            # Scale matrix to cap maximum eigenvalue
            scale_factor = max_eigenval / max_eig
            M = M * scale_factor
            M = ensure_symmetric(M)
            if warn:
                _logger.warning(
                    f"Matrix maximum eigenvalue capped: {max_eig:.2e} -> {max_eigenval:.2e} "
                    f"(scale_factor={scale_factor:.2e})"
                )
    except (RuntimeError, ValueError):
        # If eigendecomposition fails, return matrix as-is
        pass
    
    return M


def ensure_real_and_symmetric(tensor: "torch.Tensor") -> "torch.Tensor":
    """Ensure matrix is real and symmetric.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Matrix to process
        
    Returns
    -------
    torch.Tensor
        Real and symmetric matrix
    """
    tensor = ensure_real(tensor)
    tensor = ensure_symmetric(tensor)
    return tensor


def ensure_positive_definite(
    M: "torch.Tensor", 
    min_eigenval: float = DEFAULT_MIN_EIGENVAL, 
    warn: bool = True
) -> "torch.Tensor":
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    Parameters
    ----------
    M : torch.Tensor
        Matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue to enforce
    warn : bool
        Whether to log warnings
        
    Returns
    -------
    torch.Tensor
        Positive semi-definite matrix
    """
    M = ensure_symmetric(M)
    
    if M.numel() == 0 or M.shape[0] == 0:
        return M
    
    try:
        eigenvals = torch.linalg.eigvalsh(M)
        min_eig = float(torch.min(eigenvals).detach())
        
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * reg_amount
            M = ensure_symmetric(M)
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal. This biases the covariance matrix."
                )
    except (RuntimeError, ValueError) as e:
        M = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * min_eigenval
        M = ensure_symmetric(M)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed: {e}). "
                f"Added {min_eigenval:.2e} to diagonal. This biases the covariance matrix."
            )
    
    return M


def ensure_covariance_stable(
    M: "torch.Tensor", 
    min_eigenval: float = DEFAULT_MIN_EIGENVAL,
    ensure_real: bool = True
) -> "torch.Tensor":
    """Ensure covariance matrix is real, symmetric, and positive semi-definite.
    
    Parameters
    ----------
    M : torch.Tensor
        Covariance matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue to enforce
    ensure_real : bool
        Whether to ensure matrix is real
        
    Returns
    -------
    torch.Tensor
        Stable covariance matrix
    """
    if M.numel() == 0 or M.shape[0] == 0:
        return M
    
    # Step 1: Ensure real (if needed)
    # Use function directly (defined in same module, avoid name collision with parameter)
    if ensure_real:
        M = ensure_real(M) if not isinstance(ensure_real, bool) else (torch.real(M) if torch.is_complex(M) else M)
    
    # Step 2: Ensure symmetric and positive semi-definite
    M = ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
    
    return M


def clean_matrix(
    M: "torch.Tensor", 
    matrix_type: str = 'general', 
    default_nan: float = 0.0, 
    default_inf: Optional[float] = None,
    min_eigenval: float = DEFAULT_MIN_EIGENVAL,
    min_diagonal_variance: float = DEFAULT_MIN_DIAGONAL_VARIANCE
) -> "torch.Tensor":
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability.
    
    Parameters
    ----------
    M : torch.Tensor
        Matrix to clean
    matrix_type : str
        Type of matrix: 'covariance', 'diagonal', 'loading', or 'general'
    default_nan : float
        Default value for NaN replacement
    default_inf : float, optional
        Default value for Inf replacement
    min_eigenval : float
        Minimum eigenvalue for covariance matrices
    min_diagonal_variance : float
        Minimum diagonal variance for diagonal matrices
        
    Returns
    -------
    torch.Tensor
        Cleaned matrix
    """
    if matrix_type == 'covariance':
        M = torch.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = ensure_symmetric(M)
        try:
            eigenvals = torch.linalg.eigvalsh(M)
            min_eigenval_val = torch.min(eigenvals)
            if min_eigenval_val < min_eigenval:
                M = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * (min_eigenval - min_eigenval_val)
                M = ensure_symmetric(M)
        except (RuntimeError, ValueError):
            M = M + torch.eye(M.shape[0], device=M.device, dtype=M.dtype) * min_eigenval
            M = ensure_symmetric(M)
    elif matrix_type == 'diagonal':
        diag = torch.diag(M)
        default_inf_val = default_inf if default_inf is not None else 1e4
        diag = torch.nan_to_num(diag, nan=default_nan, posinf=default_inf_val, neginf=default_nan)
        diag = torch.clamp(diag, min=min_diagonal_variance)
        M = torch.diag(diag)
    elif matrix_type == 'loading':
        M = torch.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = torch.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M


def safe_inverse(
    M: "torch.Tensor",
    regularization: float = DEFAULT_INV_REGULARIZATION,
    use_pinv_fallback: bool = True
) -> "torch.Tensor":
    """Safely compute matrix inverse with robust error handling.
    
    This function implements a progressive fallback strategy for matrix inversion:
    1. Try standard torch.linalg.inv()
    2. If that fails, try regularized inversion
    3. If that fails, use pseudo-inverse (if enabled)
    
    This is critical for GPU numerical stability, as PyTorch can throw RuntimeError
    for singular or near-singular matrices (e.g., "inverse_cuda: singular matrix").
    
    Parameters
    ----------
    M : torch.Tensor
        Matrix to invert (must be square)
    regularization : float
        Regularization amount to add to diagonal before inversion
    use_pinv_fallback : bool
        Whether to use pseudo-inverse as final fallback
        
    Returns
    -------
    torch.Tensor
        Inverse of M (or pseudo-inverse if standard inversion fails)
    """
    device = M.device
    dtype = M.dtype
    
    try:
        # First try: standard inversion (fastest)
        return torch.linalg.inv(M)
    except (RuntimeError, ValueError) as e:
        # Second try: regularized inversion
        try:
            M_reg = M + torch.eye(M.shape[0], device=device, dtype=dtype) * regularization
            return torch.linalg.inv(M_reg)
        except (RuntimeError, ValueError):
            # Third try: pseudo-inverse (most robust)
            if use_pinv_fallback:
                M_reg = M + torch.eye(M.shape[0], device=device, dtype=dtype) * regularization
                return torch.linalg.pinv(M_reg)
            else:
                raise RuntimeError(f"Matrix inversion failed and pinv fallback disabled: {e}")


def safe_determinant(M: "torch.Tensor", use_logdet: bool = True) -> float:
    """Compute determinant safely to avoid overflow warnings.
    
    Parameters
    ----------
    M : torch.Tensor
        Matrix for which to compute determinant
    use_logdet : bool
        Whether to use log-determinant computation (default: True)
        
    Returns
    -------
    float
        Determinant of M, or 0.0 if computation fails
    """
    if M.numel() == 0 or M.shape[0] == 0:
        return 0.0
    
    if M.shape[0] != M.shape[1]:
        _logger.debug("safe_determinant: non-square matrix, returning 0.0")
        return 0.0
    
    # Check for NaN/Inf
    if torch.any(~torch.isfinite(M)):
        _logger.debug("safe_determinant: matrix contains NaN/Inf, returning 0.0")
        return 0.0
    
    # For small matrices, direct computation is safe
    if M.shape[0] <= 2:
        try:
            det = torch.det(M)
            if torch.isfinite(det):
                return float(det.detach() if hasattr(det, 'detach') else det)
        except (RuntimeError, ValueError):
            pass
    
    # Use log-determinant for stability
    if use_logdet:
        try:
            # Try Cholesky decomposition first (more stable for PSD matrices)
            try:
                L = torch.linalg.cholesky(M)
                log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
                if log_det > 700:  # exp(700) is near float64 max
                    _logger.debug("safe_determinant: log_det too large, returning 0.0")
                    return 0.0
                det = torch.exp(log_det)
                if torch.isfinite(det) and det > 0:
                    return float(det)
            except RuntimeError:
                # Not PSD: fall back to slogdet for general matrices
                try:
                    sign, log_det = torch.linalg.slogdet(M)
                    if not torch.isfinite(log_det) or sign <= 0:
                        return 0.0
                    if log_det > 700:
                        _logger.debug("safe_determinant: log_det too large, returning 0.0")
                        return 0.0
                    det = torch.exp(log_det)
                    if torch.isfinite(det):
                        return float(det)
                except (RuntimeError, ValueError):
                    pass
        except (RuntimeError, ValueError):
            pass
    
    # Fallback: direct computation
    try:
        det = torch.det(M)
        if torch.isfinite(det):
            return float(det)
    except (RuntimeError, ValueError):
        pass
    
    _logger.debug("safe_determinant: all methods failed, returning 0.0")
    return 0.0


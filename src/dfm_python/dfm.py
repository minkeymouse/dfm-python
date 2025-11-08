"""Dynamic Factor Model (DFM) estimation using Expectation-Maximization algorithm.

This module implements the core DFM estimation framework, including:
- Initial parameter estimation via PCA and OLS
- EM algorithm for iterative parameter refinement
- Kalman filtering and smoothing for factor extraction
- Clock-based mixed-frequency handling with tent kernels
- Robust numerical stability and error handling

The implementation uses a clock-based approach, where all latent factors
evolve at a common clock frequency, with lower-frequency observations
mapped to higher-frequency latent states via deterministic tent kernels.
"""

import numpy as np
from scipy.linalg import inv, pinv, block_diag
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict, Union
import warnings
import logging

from .kalman import run_kf
from .config import DFMConfig

from .utils.data_utils import rem_nans_spline
from .utils.aggregation import (
    get_aggregation_structure,
    FREQUENCY_HIERARCHY,
    generate_R_mat,
    get_tent_weights_for_pair,
    generate_tent_weights,
)

def _rem_nans_spline(X: np.ndarray, method: int = 2, k: int = 3):
    """Treat NaNs in dataset for DFM estimation."""
    X_clean, _ = rem_nans_spline(X, method=method, k=k)
    T_orig, _ = X.shape
    return X_clean, np.ones(T_orig, dtype=bool)

_logger = logging.getLogger(__name__)

SLOWER_FREQ_VARIANCE_DIVISOR: float = 19.0
SCALING_FACTOR: float = 1e6

def _ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose."""
    return 0.5 * (M + M.T)


def _compute_principal_components(cov_matrix: np.ndarray, n_components: int, 
                                   block_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute principal components via eigendecomposition.
    
    This function extracts the top n_components eigenvectors and eigenvalues
    from a covariance matrix using various fallback strategies for numerical stability.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix for eigendecomposition
    n_components : int
        Number of principal components to extract
    block_idx : int, optional
        Block index for logging purposes
        
    Returns
    -------
    eigenvalues : np.ndarray
        Top n_components eigenvalues (sorted by magnitude, descending)
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns)
    """
    if cov_matrix.size == 1:
        # Single series case
        eigenvector = np.array([[1.0]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else 1.0
        return np.array([eigenvalue]), eigenvector
    
    n_series = cov_matrix.shape[0]
    
    # Strategy 1: Use sparse eigs if k < n_series - 1
    if n_components < n_series - 1:
        try:
            cov_sparse = csc_matrix(cov_matrix)
            eigenvalues, eigenvectors = eigs(cov_sparse, k=n_components, which='LM')
            eigenvectors = eigenvectors.real
            # Validate results
            if np.any(~np.isfinite(eigenvalues)) or np.any(~np.isfinite(eigenvectors)):
                raise ValueError("Invalid eigenvalue results")
            return eigenvalues.real, eigenvectors
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            # Fallback to regular eig
            if block_idx is not None:
                _logger.warning(
                    f"init_conditions: Sparse eigendecomposition failed for block {block_idx+1}, "
                    f"falling back to np.linalg.eig. Error: {type(e).__name__}"
                )
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            # Sort by magnitude and take top n_components
            sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
            return eigenvalues[sort_idx].real, eigenvectors[:, sort_idx].real
    
    # Strategy 2: Use regular eig for full eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Filter valid eigenvalues
        valid_mask = np.isfinite(eigenvalues)
        if np.sum(valid_mask) < n_components:
            raise ValueError("Not enough valid eigenvalues")
        # Sort by magnitude and take top n_components
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask]
        sort_idx = np.argsort(np.abs(valid_eigenvalues))[::-1][:n_components]
        return valid_eigenvalues[sort_idx].real, valid_eigenvectors[:, sort_idx].real
    except (IndexError, ValueError, np.linalg.LinAlgError) as e:
        # Ultimate fallback: use identity matrix
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
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to clean
    matrix_type : str
        Type of matrix: 'general', 'covariance', 'diagonal', 'loading'
    default_nan : float
        Default value for NaN (default: 0.0)
    default_inf : float, optional
        Default value for Inf. If None, uses matrix-specific defaults.
        
    Returns
    -------
    np.ndarray
        Cleaned matrix
    """
    if matrix_type == 'covariance':
        # Covariance matrices: ensure symmetry and positive semi-definite
        M = np.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = _ensure_symmetric(M)
        # Ensure positive semi-definite
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
        # Diagonal matrices: clean diagonal only
        diag = np.diag(M)
        diag = np.nan_to_num(diag, nan=default_nan, 
                            posinf=default_inf if default_inf is not None else 1e4,
                            neginf=default_nan)
        diag = np.maximum(diag, 1e-6)  # Ensure positive
        M = np.diag(diag)
    elif matrix_type == 'loading':
        # Loading matrices: clip extreme values
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        # General matrices: just clean NaN/Inf
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    
    return M


def _ensure_positive_definite(M: np.ndarray, min_eigenval: float = 1e-8, 
                              warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    WARNING: Adding regularization biases the covariance matrix. This should only
    be necessary for numerical stability. If frequently required, investigate
    the source of non-positive-definite matrices.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to regularize (should be symmetric)
    min_eigenval : float
        Minimum eigenvalue threshold (default: 1e-8)
    warn : bool
        Whether to log warnings when regularization is applied
        
    Returns
    -------
    M_reg : np.ndarray
        Positive semi-definite matrix
    stats : dict
        Statistics: {'regularized': bool, 'min_eigenval_before': float, 
                     'reg_amount': float, 'min_eigenval_after': float}
    """
    M = _ensure_symmetric(M)
    stats = {
        'regularized': False,
        'min_eigenval_before': None,
        'reg_amount': 0.0,
        'min_eigenval_after': None
    }
    
    try:
        eigenvals = np.linalg.eigvals(M)
        min_eig = np.min(eigenvals)
        stats['min_eigenval_before'] = float(min_eig)
        
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0]) * reg_amount
            M = _ensure_symmetric(M)
            stats['regularized'] = True
            stats['reg_amount'] = float(reg_amount)
            
            # Verify after regularization
            eigenvals_after = np.linalg.eigvals(M)
            stats['min_eigenval_after'] = float(np.min(eigenvals_after))
            
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal. This biases the covariance matrix. "
                    f"Consider investigating the source of non-positive-definite matrices."
                )
        else:
            stats['min_eigenval_after'] = float(min_eig)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Eigendecomposition failed - add small regularization
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


def _compute_regularization_param(matrix: np.ndarray, scale_factor: float = 1e-6, 
                                  warn: bool = True) -> Tuple[float, Dict[str, Any]]:
    """Compute regularization parameter based on matrix scale.
    
    WARNING: Regularization biases estimates toward zero. Use only when necessary
    for numerical stability. Consider investigating ill-conditioned matrices if
    regularization is frequently required.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to compute regularization for
    scale_factor : float
        Scaling factor relative to trace (default: 1e-6)
    warn : bool
        Whether to log warnings when regularization is computed
        
    Returns
    -------
    reg_param : float
        Regularization parameter
    stats : dict
        Statistics: {'trace': float, 'scale_factor': float, 'reg_param': float}
    """
    trace = np.trace(matrix)
    reg_param = max(trace * scale_factor, 1e-8)
    
    stats = {
        'trace': float(trace),
        'scale_factor': float(scale_factor),
        'reg_param': float(reg_param)
    }
    
    if warn and reg_param > 1e-8:
        _logger.info(
            f"Regularization parameter computed: {reg_param:.2e} "
            f"(trace={trace:.2e}, scale={scale_factor:.2e}). "
            f"This adds bias to estimates - consider investigating matrix conditioning."
        )
    
    return reg_param, stats


def _clip_ar_coefficients(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
                         warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds.
    
    AR(1) processes require |A| < 1 for stationarity. This function clips
    coefficients to a safe range (default: [-0.99, 0.99]) to ensure numerical
    stability while maintaining theoretical validity.
    
    WARNING: Clipping can bias estimates if true coefficients are near unit root.
    Consider investigating near-unit root behavior if clipping occurs frequently.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients (can be scalar, 1D array, or 2D matrix)
    min_val : float
        Minimum allowed value (default: -0.99)
    max_val : float
        Maximum allowed value (default: 0.99)
    warn : bool
        Whether to log warnings when clipping occurs
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients with same shape as input
    stats : dict
        Statistics about clipping: {'n_clipped': int, 'n_total': int, 'clipped_indices': list}
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    
    # Find values that need clipping
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    
    # Clip the values
    A_clipped = np.clip(A, min_val, max_val)
    
    # Prepare statistics
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    
    # Warn if clipping occurred
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]. "
            f"This may indicate near-unit root behavior. "
            f"Consider investigating stationarity if clipping occurs frequently."
        )
    
    return A_clipped, stats


def _apply_ar_clipping(A: np.ndarray, config: Optional[DFMConfig] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients to clip
    config : DFMConfig, optional
        Configuration object. If None, uses defaults (clipping enabled)
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients
    stats : dict
        Statistics about clipping
    """
    if config is None or config.clip_ar_coefficients:
        min_val = config.ar_clip_min if config else -0.99
        max_val = config.ar_clip_max if config else 0.99
        warn = config.warn_on_ar_clip if config else True
        return _clip_ar_coefficients(A, min_val, max_val, warn)
    else:
        # Clipping disabled - return as-is with empty stats
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}


def _apply_regularization(M: np.ndarray, matrix_type: str = 'covariance',
                         config: Optional[DFMConfig] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply regularization based on configuration.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to regularize
    matrix_type : str
        Type of matrix ('covariance', 'general')
    config : DFMConfig, optional
        Configuration object. If None, uses defaults (regularization enabled)
        
    Returns
    -------
    M_reg : np.ndarray
        Regularized matrix
    stats : dict
        Statistics about regularization
    """
    if config is None or config.use_regularization:
        min_eigenval = config.min_eigenvalue if config else 1e-8
        warn = config.warn_on_regularization if config else True
        return _ensure_positive_definite(M, min_eigenval, warn)
    else:
        # Regularization disabled - just ensure symmetric
        return _ensure_symmetric(M), {'regularized': False}


def _cap_max_eigenvalue(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
    """Cap maximum eigenvalue of a matrix to prevent numerical explosion.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (should be symmetric)
    max_eigenval : float
        Maximum allowed eigenvalue (default: 1e6)
        
    Returns
    -------
    np.ndarray
        Matrix with capped eigenvalues
    """
    try:
        eigenvals = np.linalg.eigvals(M)
        max_eig = np.max(eigenvals)
        if max_eig > max_eigenval:
            scale = max_eigenval / max_eig
            return M * scale
    except (np.linalg.LinAlgError, ValueError):
        # Eigendecomposition failed - cap diagonal values as fallback
        M_diag = np.diag(M)
        M_diag = np.maximum(M_diag, 1e-8)
        M_diag = np.minimum(M_diag, max_eigenval)
        M_capped = np.diag(M_diag)
        return _ensure_symmetric(M_capped)
    return M


def _estimate_ar_coefficient(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
                             vsmooth_sum: Optional[np.ndarray] = None,
                             T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR coefficients and innovation variances from expectations.
    
    This is the core EM update formula for AR(1) processes:
    A = E[z_t * z_{t-1}'] / E[z_{t-1} * z_{t-1}']
    Q = (E[z_t * z_t'] - A * E[z_t * z_{t-1}']) / T
    
    Parameters
    ----------
    EZZ_FB : np.ndarray
        E[z_t * z_{t-1}'] - can be diagonal (1D) or full matrix
    EZZ_BB : np.ndarray
        E[z_{t-1} * z_{t-1}'] - can be diagonal (1D) or full matrix
    vsmooth_sum : np.ndarray, optional
        Sum of smoothed covariances to add to expectations
    T : int, optional
        Number of time periods for variance computation
        
    Returns
    -------
    A_diag : np.ndarray
        AR coefficients (diagonal elements)
    Q_diag : np.ndarray
        Innovation variances (diagonal elements)
    """
    # Handle both scalar and array inputs
    if np.isscalar(EZZ_FB):
        EZZ_FB = np.array([EZZ_FB])
        EZZ_BB = np.array([EZZ_BB])
    
    # Extract diagonal if full matrices
    if EZZ_FB.ndim > 1:
        EZZ_FB_diag = np.diag(EZZ_FB).copy()
        EZZ_BB_diag = np.diag(EZZ_BB).copy()
    else:
        EZZ_FB_diag = EZZ_FB.copy()
        EZZ_BB_diag = EZZ_BB.copy()
    
    # Add smoothed covariance if provided
    if vsmooth_sum is not None:
        if vsmooth_sum.ndim > 1:
            vsmooth_diag = np.diag(vsmooth_sum)
        else:
            vsmooth_diag = vsmooth_sum
        EZZ_BB_diag = EZZ_BB_diag + vsmooth_diag
    
    # Regularize denominators to prevent division by zero
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * 1e-6, 1e-10)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    
    # Clean numerators
    EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0) if EZZ_FB_diag.ndim > 0 else np.nan_to_num(EZZ_FB_diag, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute AR coefficients
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    
    # Compute innovation variances if T is provided
    if T is not None:
        # Need E[z_t^2] - extract from EZZ_BB if available, or compute separately
        # For now, assume EZZ_idio is provided separately if needed
        # This will be handled by the caller
        Q_diag = None
    else:
        Q_diag = None
    
    return A_diag, Q_diag


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    RMSE is calculated as: sqrt(mean((actual - predicted)^2))
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    rmse_overall : float
        Overall RMSE averaged across all series and time periods
    rmse_per_series : np.ndarray
        RMSE for each series (N,) or scalar if 1D input
    """
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(f"actual and predicted must have same shape, got {actual.shape} and {predicted.shape}")
    
    # Create mask for valid values
    if mask is None:
        # Use non-NaN values in both actual and predicted
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        # Combine user mask with finite check
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Calculate squared errors
    errors_sq = (actual - predicted) ** 2
    
    # Handle 1D case (single series)
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        rmse_series = np.sqrt(np.mean(errors_sq[mask]))
        return rmse_series, np.array([rmse_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    
    # Calculate RMSE per series
    rmse_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            rmse_per_series[i] = np.sqrt(np.mean(errors_sq[series_mask, i]))
        else:
            rmse_per_series[i] = np.nan
    
    # Calculate overall RMSE (average across all valid observations)
    if np.any(mask):
        rmse_overall = np.sqrt(np.mean(errors_sq[mask]))
    else:
        rmse_overall = np.nan
    
    return rmse_overall, rmse_per_series


def _safe_divide(numerator: np.ndarray, denominator: float, default: float = 0.0) -> np.ndarray:
    """Safely divide numerator by denominator, handling zero and invalid values.
    
    This utility function prevents division by zero and handles NaN/Inf values
    that can arise in numerical computations. It's used throughout the codebase
    to ensure robust numerical operations.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array of any shape
    denominator : float
        Denominator value (scalar)
    default : float, optional
        Default value to use if denominator is zero, NaN, or Inf, by default 0.0.
        Also used to replace any NaN/Inf values in the result.
        
    Returns
    -------
    np.ndarray
        Result of division with same shape as numerator.
        Elements where denominator is invalid or result is non-finite are set to default.
        
    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> _safe_divide(x, 2.0)
    array([0.5, 1.0, 1.5])
    >>> _safe_divide(x, 0.0, default=1.0)
    array([1.0, 1.0, 1.0])
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.full_like(numerator, default)
    result = numerator / denominator
    # Replace any NaN/Inf in result with default
    result = np.where(np.isfinite(result), result, default)
    return result


# Validation functions removed - validation is done inline where needed for performance


@dataclass
class DFMResult:
    """DFM estimation results structure.
    
    This dataclass contains all outputs from the DFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Attributes
    ----------
    x_sm : np.ndarray
        Standardized smoothed data matrix (T x N), where T is time periods
        and N is number of series. Data is standardized (zero mean, unit variance).
    X_sm : np.ndarray
        Unstandardized smoothed data matrix (T x N). This is the original-scale
        version of x_sm, computed as X_sm = x_sm * Wx + Mx.
    Z : np.ndarray
        Smoothed factor estimates (T x m), where m is the state dimension.
        Columns represent different factors (common factors and idiosyncratic components).
    C : np.ndarray
        Observation/loading matrix (N x m). Each row corresponds to a series,
        each column to a factor. C[i, j] gives the loading of series i on factor j.
    R : np.ndarray
        Covariance matrix for observation equation residuals (N x N).
        Typically diagonal, representing idiosyncratic variances.
    A : np.ndarray
        Transition matrix (m x m) for the state equation. Describes how factors
        evolve over time: Z_t = A @ Z_{t-1} + error.
    Q : np.ndarray
        Covariance matrix for transition equation residuals (m x m).
        Describes the covariance of factor innovations.
    Mx : np.ndarray
        Series means (N,). Used for standardization: x = (X - Mx) / Wx.
    Wx : np.ndarray
        Series standard deviations (N,). Used for standardization.
    Z_0 : np.ndarray
        Initial state vector (m,). Starting values for factors at t=0.
    V_0 : np.ndarray
        Initial covariance matrix (m x m) for factors. Uncertainty about Z_0.
    r : np.ndarray
        Number of factors per block (n_blocks,). Each element specifies
        how many factors are in each block structure.
    p : int
        Number of lags in the autoregressive structure of factors. Typically p=1.
    rmse : float, optional
        Overall RMSE on original scale (averaged across all series).
    rmse_per_series : np.ndarray, optional
        RMSE per series on original scale (N,).
    rmse_std : float, optional
        Overall RMSE on standardized scale (averaged across all series).
    rmse_std_per_series : np.ndarray, optional
        RMSE per series on standardized scale (N,).
    converged : bool, optional
        Whether EM algorithm converged.
    num_iter : int, optional
        Number of EM iterations performed.
    loglik : float, optional
        Final log-likelihood value.
    
    Examples
    --------
    >>> Res = dfm(X, config, threshold=1e-4)
    >>> # Access smoothed factors
    >>> common_factor = Res.Z[:, 0]
    >>> # Access factor loadings for first series
    >>> loadings = Res.C[0, :]
    >>> # Reconstruct smoothed series from factors
    >>> reconstructed = Res.Z @ Res.C.T
    """
    x_sm: np.ndarray      # Standardized smoothed data (T x N)
    X_sm: np.ndarray      # Unstandardized smoothed data (T x N)
    Z: np.ndarray         # Smoothed factors (T x m)
    C: np.ndarray         # Observation matrix (N x m)
    R: np.ndarray         # Covariance for observation residuals (N x N)
    A: np.ndarray         # Transition matrix (m x m)
    Q: np.ndarray         # Covariance for transition residuals (m x m)
    Mx: np.ndarray        # Series means (N,)
    Wx: np.ndarray        # Series standard deviations (N,)
    Z_0: np.ndarray       # Initial state (m,)
    V_0: np.ndarray       # Initial covariance (m x m)
    r: np.ndarray         # Number of factors per block
    p: int                # Number of lags
    converged: bool = False  # Whether EM algorithm converged
    num_iter: int = 0     # Number of iterations completed
    loglik: float = -np.inf  # Final log-likelihood
    rmse: Optional[float] = None  # Overall RMSE (original scale)
    rmse_per_series: Optional[np.ndarray] = None  # RMSE per series (original scale)
    rmse_std: Optional[float] = None  # Overall RMSE (standardized scale)
    rmse_std_per_series: Optional[np.ndarray] = None  # RMSE per series (standardized scale)


def em_converged(loglik: float, previous_loglik: float, threshold: float = 1e-4,
                 check_decreased: bool = True) -> Tuple[bool, bool]:
    """Check whether EM algorithm has converged.
    
    Convergence is determined by the relative change in log-likelihood being
    below the threshold: |loglik - previous_loglik| / avg(|loglik|, |previous_loglik|) < threshold.
    
    Parameters
    ----------
    loglik : float
        Log-likelihood from current EM iteration. Should be finite and typically
        negative (log-likelihoods are usually negative).
    previous_loglik : float
        Log-likelihood from previous EM iteration. Used to compute change.
    threshold : float, optional
        Convergence threshold (default 1e-4). Smaller values require more iterations
        but provide more precise convergence. Typical range: 1e-5 to 1e-3.
    check_decreased : bool, optional
        Whether to check if log-likelihood decreased (default True). If True and
        loglik < previous_loglik - 1e-3, decrease flag is set and a warning printed.
        
    Returns
    -------
    converged : bool
        True if convergence criteria satisfied, False otherwise.
    decrease : bool
        True if log-likelihood decreased (only meaningful if check_decreased=True).
        A decrease may indicate numerical issues but is allowed to happen.
    
    Examples
    --------
    >>> loglik_current = -1234.56
    >>> loglik_previous = -1235.12
    >>> converged, decreased = em_converged(loglik_current, loglik_previous, threshold=1e-4)
    >>> if converged:
    ...     print("EM algorithm converged")
    """
    converged = False
    decrease = False
    
    if check_decreased:
        if loglik - previous_loglik < -1e-3:  # Allow for a little imprecision
            print(f"******likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}!")
            decrease = True
    
    # Check convergence criteria
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    
    if (delta_loglik / avg_loglik) < threshold:
        converged = True
    
    return converged, decrease


def group_series_by_frequency(
    idx_i: np.ndarray,
    frequencies: np.ndarray,
    clock: str
) -> Dict[str, np.ndarray]:
    """Group series indices by their actual frequency.
    
    This function groups series by their actual frequency values, allowing
    each frequency to be processed independently. Faster frequencies than clock
    are validated and rejected.
    
    Parameters
    ----------
    idx_i : np.ndarray
        Array of series indices to group (e.g., indices of series in a block)
    frequencies : np.ndarray
        Array of frequency strings for all series (length N)
    clock : str
        Clock frequency ('d', 'w', 'm', 'q', 'sa', 'a')
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with frequency strings as keys and arrays of indices as values.
        Also includes 'clock' key for series matching the clock frequency.
        
    Examples
    --------
    >>> frequencies = np.array(['m', 'm', 'q', 'm', 'q'])
    >>> idx_i = np.array([0, 1, 2, 3, 4])
    >>> groups = group_series_by_frequency(idx_i, frequencies, 'm')
    >>> groups['m']  # Monthly series
    array([0, 1, 3])
    >>> groups['q']  # Quarterly series
    array([2, 4])
    """
    if frequencies is None or len(frequencies) == 0:
        # Fallback: assume all are same as clock if frequencies not provided
        return {clock: idx_i.copy()}
    
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)  # Default to monthly
    
    freq_groups: Dict[str, List[int]] = {}
    faster_indices = []
    
    for idx in idx_i:
        if idx >= len(frequencies):
            # Index out of bounds - skip
            continue
        
        freq = frequencies[idx]
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)  # Default to monthly
        
        if freq_hierarchy < clock_hierarchy:
            # Faster frequency (lower hierarchy number) - NOT SUPPORTED
            faster_indices.append(idx)
        else:
            # Group by actual frequency
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(idx)
    
    # Validate: faster frequencies are not supported
    if len(faster_indices) > 0:
        faster_series_names = [f"series_{idx}" for idx in faster_indices]
        raise ValueError(
            f"Higher frequencies (daily, weekly) are not supported. "
            f"Found {len(faster_indices)} series with frequency faster than clock '{clock}'. "
            f"Please use monthly, quarterly, semi-annual, or annual frequencies only."
        )
    
    # Convert lists to numpy arrays
    return {freq: np.array(indices, dtype=int) for freq, indices in freq_groups.items()}


def init_conditions(x: np.ndarray, r: np.ndarray, p: int, blocks: np.ndarray,
                   opt_nan: dict, Rcon: Optional[np.ndarray], q: Optional[np.ndarray],
                   nQ: Optional[int], i_idio: np.ndarray,
                   clock: str = 'm', tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
                   frequencies: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray]:
    """Calculate initial conditions for parameter estimation.
    
    This function computes initial parameter estimates using principal component
    analysis (PCA) for factors and ordinary least squares (OLS) for AR coefficients.
    These initial values are then refined by the EM algorithm in `dfm()`.
    
    The function handles:
    - Factor extraction via eigendecomposition of residual covariance
    - AR coefficient estimation via OLS regression
    - Quarterly aggregation constraints for quarterly series
    - Idiosyncratic component initialization
    
    Parameters
    ----------
    x : np.ndarray
        Standardized data matrix (T x N), where T is time periods and N is series.
        Data should already be standardized (zero mean, unit variance).
        Missing values (NaN) are handled via spline interpolation.
    r : np.ndarray
        Number of common factors for each block (n_blocks,). Typically all ones
        indicating one factor per block. Example: np.array([1, 1, 1, 1]) for 4 blocks.
    p : int
        Number of lags in the transition equation. Typically p=1 (AR(1) process).
        Higher values create more complex dynamics but require more parameters.
    blocks : np.ndarray
        Block loading structure (N x n_blocks). Each row corresponds to a series,
        each column to a block. Value 1 indicates the series loads on that block.
        All series must load on the first block (global factor).
    opt_nan : dict
        Options for missing value handling. Should contain:
        - 'method': int, method for NaN handling (typically 2 or 3)
        - 'k': int, spline parameter (typically 3)
        See `rem_nans_spline` for details.
    Rcon : np.ndarray
        Aggregation constraints matrix for slower-frequency series (e.g., 4 x 5 for quarterly when clock is monthly). Implements the "tent structure"
        that aggregates monthly values to quarterly. Standard structure is:
        [[2, -1, 0, 0, 0],
         [3, 0, -1, 0, 0],
         [2, 0, 0, -1, 0],
         [1, 0, 0, 0, -1]]
    q : np.ndarray
        Constraints vector (4,). Typically all zeros: np.zeros(4).
        Used with Rcon to enforce slower-frequency aggregation.
    nQ : int, optional
        Number of slower-frequency variables (e.g., quarterly when clock is monthly).
        If None, computed from frequencies array. Must satisfy 0 <= nQ < N.
    i_idio : np.ndarray
        Logical array (N,) indicating idiosyncratic components: 1 for same-frequency series,
        0 for slower-frequency series. Typically: np.concatenate([np.ones(N-nQ), np.zeros(nQ)]).
        
    Returns
    -------
    A : np.ndarray
        Initial transition matrix (m x m), where m is the state dimension.
        Block diagonal structure: factors, monthly idiosyncratic, quarterly idiosyncratic.
    C : np.ndarray
        Initial observation/loading matrix (N x m). Loadings are estimated via PCA
        (eigendecomposition of residual covariance) for monthly series and constrained
        least squares for quarterly series.
    Q : np.ndarray
        Initial covariance matrix for transition residuals (m x m). Block diagonal.
        Diagonal elements represent innovation variances.
    R : np.ndarray
        Initial covariance matrix for observation residuals (N x N). Diagonal matrix.
        Diagonal elements are residual variances (idiosyncratic variances).
    Z_0 : np.ndarray
        Initial state vector (m,). Typically zeros: np.zeros(m).
    V_0 : np.ndarray
        Initial covariance matrix for state (m x m). Computed from steady-state
        of the AR process: V_0 = inv(I - kron(A, A)) @ vec(Q).
        
    Raises
    ------
    ValueError
        If inputs are invalid or dimensions don't match. Also raised if numerical
        issues prevent computation (though fallback strategies will be attempted).
        
    Notes
    -----
    - The function uses fallback strategies when numerical issues occur (e.g., singular
      matrices, eigendecomposition failures). These are logged at WARNING level.
    - For quarterly series, loadings must satisfy aggregation constraints that ensure
      quarterly aggregates match monthly weighted sums.
    - The state dimension m is computed as: sum(r) * max(p, 5) + monthly_idio + quarterly_idio * 5
      
    Examples
    --------
    >>> from dfm_python import load_config, load_data, dfm
    >>> # Load configuration from YAML or create DFMConfig directly
    >>> config = load_config('config.yaml')
    >>> # Load data from file
    >>> X, Time, Z = load_data('data.csv', config)
    >>> # Standardize data
    >>> x = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    >>> # Set up parameters
    >>> r = np.ones(config.Blocks.shape[1])
    >>> p = 1
    >>> nQ = np.sum(np.array(config.Frequency) == 'q')
    >>> i_idio = np.concatenate([np.ones(len(config.SeriesID) - nQ), np.zeros(nQ)])
    >>> Rcon = np.array([[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], 
    ...                  [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]])
    >>> q = np.zeros(4)
    >>> opt_nan = {'method': 2, 'k': 3}
    >>> # Compute initial conditions
    >>> A, C, Q, R, Z_0, V_0 = init_conditions(
    ...     x, r, p, config.Blocks, opt_nan, Rcon, q, nQ, i_idio
    ... )
    """
    # Input validation is done inline where needed
    
    # Handle missing_data method (no aggregation constraints)
    if Rcon is None or q is None:
        pC = 1  # No tent structure
    else:
        pC = Rcon.shape[1]  # Tent structure size (quarterly to monthly)
    ppC = int(max(p, pC))  # Ensure integer
    n_blocks = blocks.shape[1]  # Number of blocks
    
    # Spline without NaNs
    xBal, _ = _rem_nans_spline(x, method=opt_nan['method'], k=opt_nan['k'])
    
    T, N = xBal.shape
    
    # Determine pC from tent_weights_dict if available, otherwise use Rcon
    # pC is the maximum tent kernel size needed for slower frequencies
    pC = 1
    if tent_weights_dict is not None and len(tent_weights_dict) > 0:
        # Find maximum tent kernel size
        for tent_weights in tent_weights_dict.values():
            if tent_weights is not None and len(tent_weights) > pC:
                pC = len(tent_weights)
    elif Rcon is not None:
        pC = Rcon.shape[1]
    
    # Fallback: if nQ is provided but frequencies not, compute from nQ
    if nQ is None and frequencies is not None:
        # Count series slower than clock
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        nQ = sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_hierarchy)
    elif nQ is None:
        nQ = 0  # Default to 0 if neither provided
    
    # Create 2D boolean array for NaN indicators
    indNaN = np.isnan(xBal)
    
    # Create views/copies as needed (xNaN needs to be modifiable)
    xNaN = xBal.copy()  # Need copy for NaN assignment
    xNaN[indNaN] = np.nan
    data_residuals = xBal  # Residual data after removing factors (can use view, not modified)
    resNaN = xNaN.copy()  # Need copy for modifications
    
    # Initialize model coefficient output
    C = None
    A = None
    Q = None
    V_0 = None
    
    # Set first observations as NaNs for quarterly-monthly aggregation scheme
    # Only if using aggregation (pC > 1)
    if pC > 1:
        indNaN[:pC-1, :] = True
    
    # Loop for each block
    for i in range(n_blocks):
        r_i = int(r[i])  # r_i = 1 when block is loaded (ensure integer)
        
        # Reset block-specific variables at start of each iteration
        # Prevents dimension mismatch from previous block values
        factor_projection_lagged = None
        ar_coeffs = None
        
        # Observation equation: estimate factor loadings for this block
        block_loadings = np.zeros((N, int(r_i * ppC)))
        idx_i = np.where(blocks[:, i] == 1)[0]  # Series loading block i
        
        # Group series by their actual frequency
        freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
        idx_clock_freq = freq_groups.get(clock, np.array([], dtype=int))  # Same frequency as clock
        
        if len(idx_clock_freq) > 0:
            # Returns eigenvector v with largest eigenvalue d
            # Calculate covariance, handling edge cases
            try:
                # Extract data for this block, handling NaN values
                res_block = data_residuals[:, idx_clock_freq].copy()
                # Remove rows with any NaN (np.cov doesn't handle NaN)
                finite_rows = np.all(np.isfinite(res_block), axis=1)
                n_finite = np.sum(finite_rows)
                n_total = len(finite_rows)
                completeness_pct = (n_finite / n_total * 100) if n_total > 0 else 0.0
                
                # Check minimum data requirements
                min_obs_required = max(2, len(idx_clock_freq) + 1)  # Need at least n+1 observations for n series
                if n_finite < min_obs_required:
                    # Not enough finite data - use identity
                    _logger.warning(
                        f"init_conditions: Block {i+1} has insufficient data for covariance calculation. "
                        f"Series in block: {len(idx_clock_freq)}, Finite observations: {n_finite}/{n_total} "
                        f"({completeness_pct:.1f}%), Required: {min_obs_required}. "
                        f"Using identity matrix as fallback."
                    )
                    raise ValueError(
                        f"Insufficient finite data for block {i+1}: "
                        f"found {n_finite} finite observations, but need at least {min_obs_required} "
                        f"(number of series in block: {len(idx_clock_freq)}). "
                        f"This may be due to excessive missing data or transformation issues. "
                        f"Consider checking data quality or adjusting nan_method."
                    )
                
                res_block_clean = res_block[finite_rows, :]
                
                # Ensure res_block_clean is properly shaped before covariance calculation
                # Handle edge cases where indexing might produce unexpected shapes
                if res_block_clean.size == 0:
                    # No data - use identity
                    cov_res = np.eye(len(idx_clock_freq))
                elif res_block_clean.ndim == 0:
                    # 0D array - shouldn't happen, but handle it
                    cov_res = np.eye(len(idx_clock_freq))
                elif res_block_clean.ndim == 1:
                    # 1D array - reshape based on number of series
                    if len(idx_clock_freq) == 1:
                        # Single series: variance is just the variance
                        var_val = np.var(res_block_clean, ddof=0)
                        if np.isnan(var_val) or np.isinf(var_val) or var_val < 1e-10:
                            var_val = 1.0
                        cov_res = np.array([[var_val]])
                    else:
                        # Multiple series but 1D - reshape to (1, n_series)
                        res_block_clean = res_block_clean.reshape(1, -1)
                        if res_block_clean.shape[0] < 2:
                            cov_res = np.eye(len(idx_clock_freq))
                        else:
                            cov_res = np.cov(res_block_clean.T)
                elif res_block_clean.ndim == 2:
                    # 2D array - normal case
                    if len(idx_clock_freq) == 1:
                        # Single series: variance is just the variance
                        series_data = res_block_clean.flatten()
                        var_val = np.var(series_data, ddof=0)
                        if np.isnan(var_val) or np.isinf(var_val) or var_val < 1e-10:
                            var_val = 1.0
                        cov_res = np.array([[var_val]])
                    elif res_block_clean.shape[1] == 1:
                        # Single series but multiple observations
                        series_data = res_block_clean.flatten()
                        var_val = np.var(series_data, ddof=0)
                        if np.isnan(var_val) or np.isinf(var_val) or var_val < 1e-10:
                            var_val = 1.0
                        cov_res = np.array([[var_val]])
                    else:
                        # Multiple series: compute covariance matrix
                        if res_block_clean.shape[0] < 2:
                            # Not enough observations for covariance - use identity
                            cov_res = np.eye(len(idx_clock_freq))
                        else:
                            cov_res = np.cov(res_block_clean.T)
                else:
                    # 3D+ array - shouldn't happen, but use identity
                    cov_res = np.eye(len(idx_clock_freq))
                
                # Check for NaN/Inf in covariance
                if np.any(np.isnan(cov_res)) or np.any(np.isinf(cov_res)):
                    # Replace with identity if covariance invalid
                    _logger.warning(
                        f"init_conditions: Block {i+1} covariance contains NaN/Inf values. "
                        f"Series in block: {len(idx_clock_freq)}, Finite observations: {n_finite}/{n_total}. "
                        f"Using identity matrix as fallback."
                    )
                    cov_res = np.eye(len(idx_clock_freq))
                # Check for constant series (zero variance) and regularize
                var_diag = np.diag(cov_res)
                if np.any(var_diag < 1e-10):
                    cov_res = cov_res + np.eye(len(idx_clock_freq)) * 1e-8
            except (ValueError, np.linalg.LinAlgError) as e:
                # Covariance calculation failed - use identity as fallback
                # This can happen with insufficient data or numerical issues
                _logger.warning(
                    f"init_conditions: Covariance calculation failed for block {i+1}. "
                    f"Series in block: {len(idx_clock_freq)}, Series indices: {idx_clock_freq.tolist()}. "
                    f"Error: {type(e).__name__}: {str(e)}. Using identity matrix as fallback."
                )
                cov_res = np.eye(len(idx_clock_freq))
            
            # Compute principal components via eigendecomposition
            d, v = _compute_principal_components(cov_res, int(r_i), block_idx=i)
            
            # Flip sign for cleaner output
            if np.sum(v) < 0:
                v = -v
            
            # Scale eigenvectors by square root of eigenvalues for proper PCA loadings
            # This ensures loadings are on a reasonable scale relative to the data
            # Eigenvalues represent variance explained, so sqrt(eigenvalue) gives proper scale
            d_positive = np.maximum(d, 1e-8)  # Ensure positive
            # Handle scalar/0D case for np.diag
            sqrt_d = np.sqrt(d_positive)
            if np.isscalar(sqrt_d) or (isinstance(sqrt_d, np.ndarray) and sqrt_d.ndim == 0):
                # Scalar case: just multiply
                v_scaled = v * float(sqrt_d)
            else:
                # Array case: use diag
                v_scaled = v @ np.diag(sqrt_d)  # Scale by sqrt(eigenvalues)
            
            # Normalize to prevent extreme values while preserving relative magnitudes
            # Use L2 normalization per column to keep loadings bounded
            if v_scaled.ndim == 2 and v_scaled.shape[1] > 0:
                for col_idx in range(v_scaled.shape[1]):
                    col_norm = np.linalg.norm(v_scaled[:, col_idx])
                    if col_norm > 0:
                        # Handle scalar d_positive case
                        if np.isscalar(d_positive) or (isinstance(d_positive, np.ndarray) and d_positive.ndim == 0):
                            d_val = float(d_positive)
                        else:
                            d_val = d_positive[col_idx] if col_idx < len(d_positive) else 1.0
                        v_scaled[:, col_idx] = v_scaled[:, col_idx] / col_norm * np.sqrt(d_val)
            
            # For monthly series with loaded blocks, use scaled eigenvectors
            block_loadings[idx_clock_freq, :int(r_i)] = v_scaled
            f = data_residuals[:, idx_clock_freq] @ v_scaled  # Data projection for scaled eigenvector direction
            
            # Lag matrix using loading
            F = None
            max_lag = max(p + 1, pC)
            for kk in range(max_lag):
                lag_data = f[pC - kk:T - kk, :]
                if F is None:
                    F = lag_data
                else:
                    F = np.hstack([F, lag_data])
            
            # Projected data with lag structure (for slower frequency series)
            factor_projection_lagged = F[:, :int(r_i * pC)]
            
            # Process series with frequencies different from clock (need tent kernel)
            # Process each frequency group separately
            for freq, idx_iFreq in freq_groups.items():
                if freq == clock:
                    continue  # Already processed above
                
                # Get tent weights for this frequency
                if tent_weights_dict is not None and freq in tent_weights_dict:
                    tent_weights = tent_weights_dict[freq]
                else:
                    # Try to get from lookup
                    tent_weights = get_tent_weights_for_pair(freq, clock)
                    if tent_weights is None:
                        # Fallback: generate symmetric tent weights based on frequency gap
                        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                        n_periods_est = freq_hierarchy - clock_hierarchy + 1
                        if n_periods_est > 0 and n_periods_est <= 12:
                            tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                            _logger.warning(
                                f"init_conditions: No tent weights found for frequency '{freq}' "
                                f"with clock '{clock}', generating symmetric tent weights with {n_periods_est} periods."
                            )
                        else:
                            raise ValueError(
                                f"init_conditions: Cannot determine tent weights for frequency '{freq}' "
                                f"with clock '{clock}'. Please provide tent_weights_dict or ensure "
                                f"the frequency pair is supported."
                            )
                    
                    # Generate R_mat for this frequency
                    R_mat_freq, q_freq = generate_R_mat(tent_weights)
                    pC_freq = len(tent_weights)
                    
                    # Ensure factor projection has enough columns for this frequency
                    if factor_projection_lagged.shape[1] < r_i * pC_freq:
                        # Pad if needed
                        factor_projection_freq = np.hstack([
                            factor_projection_lagged, 
                            np.zeros((factor_projection_lagged.shape[0], r_i * pC_freq - factor_projection_lagged.shape[1]))
                        ])
                    else:
                        factor_projection_freq = factor_projection_lagged[:, :int(r_i * pC_freq)]
                    
                    # Create constraint matrix for this frequency
                    Rcon_i = np.kron(R_mat_freq, np.eye(int(r_i)))
                    q_i = np.kron(q_freq, np.zeros(int(r_i)))
                    
                    # Loop for series of this frequency
                    for j in idx_iFreq:
                        # For series j, values are dropped to accommodate lag structure
                        series_data = resNaN[pC_freq:, j]
                        
                        if len(series_data) < factor_projection_freq.shape[0]:
                            # Align dimensions: pad or trim as needed
                            if len(series_data) > 0:
                                series_data_padded = np.full(factor_projection_freq.shape[0], np.nan)
                                series_data_padded[:len(series_data)] = series_data
                                series_data = series_data_padded
                        
                        if np.sum(~np.isnan(series_data)) < factor_projection_freq.shape[1] + 2:
                            # Replace with spline if too many NaNs
                            series_data = data_residuals[pC_freq:, j]
                        
                        # Extract finite values for regression
                        finite_mask = ~np.isnan(series_data)
                        factor_projection_clean = factor_projection_freq[finite_mask, :]
                        series_data_clean = series_data[finite_mask]
                        
                        if len(series_data_clean) > 0 and factor_projection_clean.shape[0] > 0:
                            try:
                                # Least squares: C = inv(X'X) * X'y
                                gram_matrix = factor_projection_clean.T @ factor_projection_clean
                                gram_inv = inv(gram_matrix)
                                loadings = gram_inv @ factor_projection_clean.T @ series_data_clean
                                
                                # Apply constraints (tent kernel aggregation)
                                if Rcon_i is not None and q_i is not None and Rcon_i.size > 0:
                                    constraint_term = gram_inv @ Rcon_i.T @ inv(Rcon_i @ gram_inv @ Rcon_i.T) @ (Rcon_i @ loadings - q_i)
                                    loadings = loadings - constraint_term
                                
                                # Place in block_loadings matrix (use appropriate columns)
                                block_loadings[j, :int(pC_freq * r_i)] = loadings[:int(pC_freq * r_i)]
                            except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                                # Matrix inversion or constraint application failed - set to zero
                                _logger.debug(
                                    f"init_conditions: C matrix update failed for series {j} in block {i+1}, "
                                    f"setting to zero. Error: {type(e).__name__}"
                                )
                                block_loadings[j, :int(pC_freq * r_i)] = 0.0
        
        # Pad factor projection with zeros to match data dimensions (T x N)
        # This must be inside the block loop to handle different r_i per block
        if factor_projection_lagged is not None:
            # Validate dimensions match before stacking (safety check)
            expected_width = int(pC * r_i)
            if factor_projection_lagged.shape[1] != expected_width:
                _logger.warning(
                    f"init_conditions: Block {i+1} factor projection dimension mismatch. "
                    f"Expected width {expected_width}, got {factor_projection_lagged.shape[1]}. Resetting."
                )
                factor_projection_lagged = None
        
        if factor_projection_lagged is not None:
            # factor_projection_lagged has shape (T - pC + 1, r_i * pC) from lag_data
            # Pad to match data_residuals shape (T, r_i * pC)
            factor_projection_padded = np.vstack([
                np.zeros((pC - 1, int(pC * r_i))), 
                factor_projection_lagged
            ])
            # Ensure it matches T exactly
            if factor_projection_padded.shape[0] < T:
                factor_projection_padded = np.vstack([
                    factor_projection_padded, 
                    np.zeros((T - factor_projection_padded.shape[0], int(pC * r_i)))
                ])
        else:
            # If no clock-frequency series, create dummy projection
            factor_projection_padded = np.zeros((T, int(pC * r_i)))
        
        # Update residuals by removing factor projections (only for this block's series)
        # block_loadings has shape (N, r_i * ppC), but we only use the first r_i * pC columns for residuals
        # factor_projection_padded has shape (T, r_i * pC)
        block_loadings_residual = block_loadings[:, :int(pC * r_i)]  # Extract relevant columns
        if factor_projection_padded.shape[0] == data_residuals.shape[0]:
            # Dimensions match - subtract factor projection
            data_residuals[:, idx_i] = data_residuals[:, idx_i] - factor_projection_padded @ block_loadings_residual[idx_i, :].T
        else:
            # Trim or pad to match dimensions
            if factor_projection_padded.shape[0] > data_residuals.shape[0]:
                factor_projection_padded = factor_projection_padded[:data_residuals.shape[0], :]
            else:
                factor_projection_padded = np.vstack([
                    factor_projection_padded, 
                    np.zeros((data_residuals.shape[0] - factor_projection_padded.shape[0], factor_projection_padded.shape[1]))
                ])
            data_residuals[:, idx_i] = data_residuals[:, idx_i] - factor_projection_padded @ block_loadings_residual[idx_i, :].T
        
        resNaN = data_residuals.copy()
        resNaN[indNaN] = np.nan
        
        # Combine loadings from all blocks
        if C is None:
            C = block_loadings
        else:
            C = np.hstack([C, block_loadings])
        
        # Transition equation
        if len(idx_clock_freq) > 0:
            z = F[:, :int(r_i)]  # Projected data (no lag)
            Z_lag = F[:, int(r_i):int(r_i * (p + 1))]  # Data with lag 1
            
            # Transition equation: estimate AR coefficients for this block
            block_transition = np.zeros((int(r_i * ppC), int(r_i * ppC)))
            
            if Z_lag.shape[0] > 0 and Z_lag.shape[1] > 0:
                try:
                    # OLS regression: estimate AR coefficients from lagged factors
                    ar_coeffs = inv(Z_lag.T @ Z_lag) @ Z_lag.T @ z
                    block_transition[:int(r_i), :int(r_i * p)] = ar_coeffs.T
                except (np.linalg.LinAlgError, ValueError) as e:
                    # OLS regression failed - set AR coefficients to zero
                    # This can happen if Z_lag is singular or has insufficient rank
                    _logger.debug(
                        f"init_conditions: OLS regression failed for block {i+1}, "
                        f"setting AR coefficients to zero. Error: {type(e).__name__}"
                    )
                    block_transition[:int(r_i), :int(r_i * p)] = 0.0
            
            # Identity matrix for lag structure (shift operator)
            if r_i * (ppC - 1) > 0:
                block_transition[int(r_i):, :int(r_i * (ppC - 1))] = np.eye(int(r_i * (ppC - 1)))
            
            # Innovation covariance for this block
            block_innovation_cov = np.zeros((int(ppC * r_i), int(ppC * r_i)))
            if len(z) > 0:
                innovation_residuals = z - Z_lag @ ar_coeffs if ar_coeffs is not None else z
                # Clean innovation residuals before covariance calculation
                innovation_residuals = np.nan_to_num(innovation_residuals, nan=0.0, posinf=0.0, neginf=0.0)
                
                if innovation_residuals.shape[1] > 1:
                    # Use np.cov for multiple series
                    try:
                        Q_block = np.cov(innovation_residuals.T)
                        # Validate result
                        if np.any(~np.isfinite(Q_block)):
                            Q_block = np.eye(int(r_i)) * 0.1
                    except (ValueError, np.linalg.LinAlgError):
                        Q_block = np.eye(int(r_i)) * 0.1
                else:
                    # Single series: use variance
                    variance = np.var(innovation_residuals)
                    if not np.isfinite(variance) or variance <= 0:
                        variance = 0.1
                    Q_block = np.array([[variance]]) if innovation_residuals.ndim > 1 else np.eye(int(r_i)) * variance
                
                block_innovation_cov[:int(r_i), :int(r_i)] = Q_block
            
            # Clean block matrices before kron operation
            block_transition_clean = _clean_matrix(block_transition, 'loading')
            block_innovation_cov_clean = _clean_matrix(block_innovation_cov, 'covariance', default_nan=0.0)
            
            # Initial covariance
            try:
                # Check block_transition_clean for invalid values before kron
                if np.any(~np.isfinite(block_transition_clean)):
                    raise ValueError("Invalid values in block_transition")
                
                kron_transition = np.kron(block_transition_clean, block_transition_clean)
                # Check kron result
                if np.any(~np.isfinite(kron_transition)):
                    raise ValueError("Invalid values in kron(block_transition, block_transition)")
                
                identity_kron = np.eye(int((r_i * ppC)**2)) - kron_transition
                # Check before inversion
                if np.any(~np.isfinite(identity_kron)):
                    raise ValueError("Invalid values in (I - kron(A, A))")
                
                innovation_cov_flat = block_innovation_cov_clean.flatten()
                # Check innovation_cov_flat
                if np.any(~np.isfinite(innovation_cov_flat)):
                    raise ValueError("Invalid values in block_innovation_cov")
                
                init_cov_block = np.reshape(
                    inv(identity_kron) @ innovation_cov_flat,
                    (int(r_i * ppC), int(r_i * ppC))
                )
                # Verify result is valid
                if np.any(~np.isfinite(init_cov_block)):
                    raise ValueError("NaN/Inf in init_cov_block")
            except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                # Matrix inversion failed or result invalid - use diagonal fallback
                _logger.warning(
                    f"init_conditions: Initial covariance calculation failed for block {i+1}, "
                    f"using diagonal fallback (0.1 * I). Error: {type(e).__name__}"
                )
                init_cov_block = np.eye(int(r_i * ppC)) * 0.1
            
            # Combine transition and innovation matrices for all blocks
            if A is None:
                A = block_transition
                Q = block_innovation_cov
                V_0 = init_cov_block
            else:
                # Ensure all matrices are square before block_diag
                if (block_transition.shape[0] == block_transition.shape[1] and 
                    block_innovation_cov.shape[0] == block_innovation_cov.shape[1] and 
                    init_cov_block.shape[0] == init_cov_block.shape[1]):
                    A = block_diag(A, block_transition)
                    Q = block_diag(Q, block_innovation_cov)
                    V_0 = block_diag(V_0, init_cov_block)
                else:
                    # Fallback: concatenate if dimensions don't match
                    _logger.debug(
                        f"init_conditions: block_diag dimensions mismatch for block {i+1}, "
                        f"using np.block concatenation fallback"
                    )
                    A = np.block([[A, np.zeros((A.shape[0], block_transition.shape[1]))], 
                                  [np.zeros((block_transition.shape[0], A.shape[1])), block_transition]])
                    Q = np.block([[Q, np.zeros((Q.shape[0], block_innovation_cov.shape[1]))], 
                                  [np.zeros((block_innovation_cov.shape[0], Q.shape[1])), block_innovation_cov]])
                    V_0 = np.block([[V_0, np.zeros((V_0.shape[0], init_cov_block.shape[1]))], 
                                    [np.zeros((init_cov_block.shape[0], V_0.shape[1])), init_cov_block]])
        else:
            # Dummy if no clock-frequency series
            block_transition = np.eye(int(r_i * ppC)) * 0.9
            block_innovation_cov = np.eye(int(r_i * ppC)) * 0.1
            init_cov_block = np.eye(int(r_i * ppC)) * 0.1
            if A is None:
                A = block_transition
                Q = block_innovation_cov
                V_0 = init_cov_block
            else:
                # Ensure all matrices are square
                if (block_transition.shape[0] == block_transition.shape[1] and 
                    block_innovation_cov.shape[0] == block_innovation_cov.shape[1] and 
                    init_cov_block.shape[0] == init_cov_block.shape[1]):
                    A = block_diag(A, block_transition)
                    Q = block_diag(Q, block_innovation_cov)
                    V_0 = block_diag(V_0, init_cov_block)
                else:
                    # Fallback: use np.block
                    _logger.debug(
                        f"init_conditions: block_diag dimensions mismatch for block {i+1} (V_0), "
                        f"using np.block concatenation fallback"
                    )
                    A = np.block([[A, np.zeros((A.shape[0], block_transition.shape[1]))], 
                                  [np.zeros((block_transition.shape[0], A.shape[1])), block_transition]])
                    Q = np.block([[Q, np.zeros((Q.shape[0], block_innovation_cov.shape[1]))], 
                                  [np.zeros((block_innovation_cov.shape[0], Q.shape[1])), block_innovation_cov]])
                    V_0 = np.block([[V_0, np.zeros((V_0.shape[0], init_cov_block.shape[1]))], 
                                    [np.zeros((init_cov_block.shape[0], V_0.shape[1])), init_cov_block]])
    
    # Add idiosyncratic components
    # i_idio indicates which series have idiosyncratic components
    # For clock frequency: 1 (have idiosyncratic)
    # For slower frequencies: 0 (no direct idiosyncratic, use tent kernel structure)
    eyeN = np.eye(N)
    eyeN = eyeN[:, i_idio.astype(bool)]  # Keep only series with idiosyncratic components
    
    if C is None:
        C = eyeN
    else:
        C = np.hstack([C, eyeN])
    
    # Add idiosyncratic components for slower frequency series
    # Each slower frequency series needs tent kernel structure
    if nQ > 0 and frequencies is not None:
        # Count series by slower frequency
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        slower_series_count = {}  # freq -> count
        slower_series_indices = {}  # freq -> list of indices
        
        for j in range(N):
            if j < len(frequencies):
                freq = frequencies[j]
                freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                if freq_hierarchy > clock_hierarchy:
                    if freq not in slower_series_count:
                        slower_series_count[freq] = 0
                        slower_series_indices[freq] = []
                    slower_series_count[freq] += 1
                    slower_series_indices[freq].append(j)
        
        # Build idiosyncratic structure for each slower frequency
        n_same_freq = N - nQ  # Series with same frequency as clock
        slower_idio_blocks = []
        
        for freq, count in slower_series_count.items():
            # Get tent weights for this frequency
            if tent_weights_dict is not None and freq in tent_weights_dict:
                tent_weights = tent_weights_dict[freq]
            else:
                tent_weights = get_tent_weights_for_pair(freq, clock)
                if tent_weights is None:
                    # Fallback: generate symmetric tent weights based on frequency gap
                    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                    freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                    n_periods_est = freq_hierarchy - clock_hierarchy + 1
                    if n_periods_est > 0 and n_periods_est <= 12:
                        tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                    else:
                        raise ValueError(
                            f"init_conditions: Cannot determine tent weights for frequency '{freq}' "
                            f"with clock '{clock}'. Please provide tent_weights_dict or ensure "
                            f"the frequency pair is supported."
                        )
            
            n_periods = len(tent_weights)
            
            # Create block: zeros for same freq series, tent structure for slower freq series
            idio_block = np.zeros((N, n_periods * count))
            
            # Place tent structure for this frequency's series
            for idx, j in enumerate(slower_series_indices[freq]):
                col_start = idx * n_periods
                col_end = (idx + 1) * n_periods
                idio_block[j, col_start:col_end] = tent_weights
            
            slower_idio_blocks.append(idio_block)
        
        # Combine all slower frequency idiosyncratic blocks
        if slower_idio_blocks:
            slower_idio_full = np.hstack(slower_idio_blocks)
            C = np.hstack([C, slower_idio_full])
    
    # Initialize covariance matrix
    # Calculate variance, handling NaN/Inf cases
    var_values = np.nanvar(resNaN, axis=0)
    # Replace NaN/Inf with small default value
    var_values = np.where((np.isnan(var_values) | np.isinf(var_values)), 1e-4, var_values)
    # Ensure minimum variance to avoid numerical issues
    var_values = np.maximum(var_values, 1e-4)
    R = np.diag(var_values)
    
    # Monthly idiosyncratic AR parameters
    ii_idio = np.where(i_idio)[0]
    n_idio = len(ii_idio)
    BM = np.zeros((n_idio, n_idio))
    SM = np.zeros((n_idio, n_idio))
    
    for idx, i in enumerate(ii_idio):
        R[i, i] = 1e-4
        
        res_i = resNaN[:, i]
        res_i_full = data_residuals[:, i]
        
        # Count leading/ending NaNs
        leadZero = 0
        for t in range(T):
            if np.isnan(res_i[t]):
                leadZero += 1
            else:
                break
        
        endZero = 0
        for t in range(T - 1, -1, -1):
            if np.isnan(res_i[t]):
                endZero += 1
            else:
                break
        
        # Truncate
        if endZero > 0:
            res_i_trunc = res_i_full[leadZero:T - endZero]
        else:
            res_i_trunc = res_i_full[leadZero:]
        
        if len(res_i_trunc) > 1:
            # AR(1) process: res_i[1:] = BM * res_i[:-1] + error
            # MATLAB: inv(res_i(1:end-1)'*res_i(1:end-1))*res_i(1:end-1)'*res_i(2:end)
            # Copy to avoid read-only array issues
            res_i_copy = res_i_trunc.copy()
            X_ar = res_i_copy[:-1].reshape(-1, 1)  # Column vector (T-1 x 1)
            y_ar = res_i_copy[1:].reshape(-1, 1)   # Column vector (T-1 x 1)
            # X_ar.T @ X_ar is (1 x T-1) @ (T-1 x 1) = (1 x 1) scalar
            XTX = X_ar.T @ X_ar
            if XTX.size == 1 and XTX[0, 0] != 0 and not (np.isnan(XTX[0, 0]) or np.isinf(XTX[0, 0])):
                try:
                    BM_val = (1.0 / XTX[0, 0]) * (X_ar.T @ y_ar)
                    BM_val_clean = BM_val[0, 0] if BM_val.size > 0 else 0.1
                    # Ensure finite value
                    if np.isnan(BM_val_clean) or np.isinf(BM_val_clean):
                        BM_val_clean = 0.1
                    BM[idx, idx] = BM_val_clean
                    # Calculate variance with bounds checking
                    residuals = res_i_trunc[1:] - res_i_trunc[:-1] * BM[idx, idx]
                    var_val = np.var(residuals)
                    SM[idx, idx] = var_val if not (np.isnan(var_val) or np.isinf(var_val)) else 0.1
                except (ValueError, ZeroDivisionError, IndexError) as e:
                    # AR coefficient calculation or variance computation failed - use defaults
                    # This can happen with insufficient data or division by zero
                    BM[idx, idx] = 0.1
                    SM[idx, idx] = 0.1
            else:
                BM[idx, idx] = 0.1
                SM[idx, idx] = 0.1
        else:
            BM[idx, idx] = 0.1
            SM[idx, idx] = 0.1
    
    # Slower frequency idiosyncratic
    Rdiag = np.diag(R).copy()  # Copy to avoid read-only array
    # Handle division by zero and NaN/Inf
    # Validate Rdiag values before division
    # Get indices of slower frequency series
    if frequencies is not None:
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        slower_indices = [j for j in range(N) if j < len(frequencies) and 
                         FREQUENCY_HIERARCHY.get(frequencies[j], 3) > clock_hierarchy]
    else:
        # Fallback: assume last nQ series are slower
        slower_indices = list(range(N - nQ, N)) if nQ > 0 else []
    
    Rdiag_slower = Rdiag[slower_indices] if slower_indices else np.array([])
    # Process slower frequency idiosyncratic components
    # Group by frequency and create AR structure for each
    BQ_blocks = []
    SQ_blocks = []
    initViQ_blocks = []
    
    if len(slower_indices) > 0 and frequencies is not None:
        # Group slower series by frequency
        slower_freq_groups = {}
        for j in slower_indices:
            if j < len(frequencies):
                freq = frequencies[j]
                if freq not in slower_freq_groups:
                    slower_freq_groups[freq] = []
                slower_freq_groups[freq].append(j)
        
        for freq, idx_list in slower_freq_groups.items():
            n_freq = len(idx_list)
            
            # Get tent weights for this frequency
            if tent_weights_dict is not None and freq in tent_weights_dict:
                tent_weights = tent_weights_dict[freq]
            else:
                tent_weights = get_tent_weights_for_pair(freq, clock)
                if tent_weights is None:
                    # Fallback: generate symmetric tent weights based on frequency gap
                    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                    freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                    n_periods_est = freq_hierarchy - clock_hierarchy + 1
                    if n_periods_est > 0 and n_periods_est <= 12:
                        tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                    else:
                        raise ValueError(
                            f"init_conditions: Cannot determine tent weights for frequency '{freq}' "
                            f"with clock '{clock}'. Please provide tent_weights_dict or ensure "
                            f"the frequency pair is supported."
                        )
            
            n_periods = len(tent_weights)
            
            # Get variance for this frequency's series
            Rdiag_freq = Rdiag[idx_list]
            if len(Rdiag_freq) > 0:
                Rdiag_freq = np.where(
                    (np.isnan(Rdiag_freq) | np.isinf(Rdiag_freq) | (Rdiag_freq < 0)),
                    1e-4, Rdiag_freq
                )
                # Use appropriate variance divisor based on tent kernel size
                # For quarterly (5 periods), divisor is 19.0
                # For other frequencies, use a similar scaling
                variance_divisor = np.sum(tent_weights**2)  # Sum of squares of tent weights
                sig_e = _safe_divide(Rdiag_freq, variance_divisor, default=1e-4)
                sig_e = np.where((np.isnan(sig_e) | np.isinf(sig_e) | (sig_e < 1e-6)), 1e-4, sig_e)
            else:
                sig_e = np.ones(n_freq) * 1e-4
            
            # Set Rdiag for these series
            Rdiag[idx_list] = 1e-4
            
            # AR structure for this frequency
            rho0 = 0.1
            temp = np.zeros((n_periods, n_periods))
            temp[0, 0] = 1
            
            SQ_freq = np.kron(np.diag((1 - rho0**2) * sig_e), temp)
            
            # BQ template for this frequency
            BQ_template = np.vstack([
                np.hstack([rho0, np.zeros(n_periods - 1)]),
                np.hstack([np.eye(n_periods - 1), np.zeros((n_periods - 1, 1))])
            ])
            BQ_freq = np.kron(np.eye(int(n_freq)), BQ_template)
            
            BQ_blocks.append(BQ_freq)
            SQ_blocks.append(SQ_freq)
            
            # Initial covariance for this frequency
            try:
                I_kron = np.eye((n_periods * n_freq)**2)
                kron_mat = np.kron(BQ_freq, BQ_freq)
                inv_mat = I_kron - kron_mat
                if np.any(np.isnan(inv_mat)) or np.any(np.isinf(inv_mat)):
                    raise ValueError("NaN/Inf in inversion matrix")
                initViQ_freq = np.reshape(
                    inv(inv_mat) @ SQ_freq.flatten(),
                    (n_periods * n_freq, n_periods * n_freq)
                )
                if np.any(np.isnan(initViQ_freq)) or np.any(np.isinf(initViQ_freq)):
                    raise ValueError("NaN/Inf in initViQ result")
                initViQ_blocks.append(initViQ_freq)
            except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                _logger.warning(
                    f"init_conditions: {freq} idiosyncratic initial covariance calculation failed, "
                    f"using diagonal fallback. Error: {type(e).__name__}"
                )
                initViQ_blocks.append(np.eye(n_periods * n_freq) * 0.1)
    
    R = np.diag(Rdiag)
    
    # Combine all slower frequency blocks
    if BQ_blocks:
        BQ = block_diag(*BQ_blocks) if len(BQ_blocks) > 1 else BQ_blocks[0]
        SQ = block_diag(*SQ_blocks) if len(SQ_blocks) > 1 else SQ_blocks[0]
        initViQ = block_diag(*initViQ_blocks) if len(initViQ_blocks) > 1 else initViQ_blocks[0]
    else:
        # No slower frequency series
        BQ = np.array([]).reshape(0, 0)
        SQ = np.array([]).reshape(0, 0)
        initViQ = np.array([]).reshape(0, 0)
    
    
    try:
        # Calculate denominator, handling division by zero
        eye_diag = np.diag(np.eye(BM.shape[0]))
        BM_diag_sq = np.diag(BM)**2
        denom_val = eye_diag - BM_diag_sq
        # Avoid division by zero
        denom_val = np.where(np.abs(denom_val) < 1e-10, 1.0, denom_val)
        denom_viM = np.diag(1.0 / denom_val)
        # Check for NaN/Inf
        if np.any(np.isnan(denom_viM)) or np.any(np.isinf(denom_viM)):
            raise ValueError("NaN/Inf in denom_viM")
        initViM = denom_viM @ SM
        # Check result
        if np.any(np.isnan(initViM)) or np.any(np.isinf(initViM)):
            raise ValueError("NaN/Inf in initViM result")
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as e:
        # Matrix inversion failed, division by zero, or result invalid - use diagonal from SM
        # This can happen if denom_viM calculation fails or SM has issues
        _logger.warning(
            f"init_conditions: Monthly idiosyncratic initial covariance calculation failed, "
            f"using diagonal from SM as fallback. Error: {type(e).__name__}"
        )
        initViM = np.diag(np.diag(SM)).copy()
        # Ensure finite
        initViM = np.where(np.isnan(initViM) | np.isinf(initViM), 0.1, initViM)
    
    # Final block diagonal - ensure all matrices are square
    # Verify dimensions before block_diag
    if BM.shape[0] != BM.shape[1]:
        BM = np.diag(np.diag(BM))  # Force square
    if SM.shape[0] != SM.shape[1]:
        SM = np.diag(np.diag(SM))  # Force square
    if BQ.shape[0] != BQ.shape[1]:
        # BQ should already be square, but if not, force it
        min_dim = min(BQ.shape)
        BQ = np.diag(np.diag(BQ[:min_dim, :min_dim]))
    if SQ.shape[0] != SQ.shape[1]:
        SQ = np.diag(np.diag(SQ))
    
    A = block_diag(A, BM, BQ)
    Q = block_diag(Q, SM, SQ)
    Z_0 = np.zeros(A.shape[0])
    
    # Ensure V_0 matrices are square before block_diag
    if V_0.shape[0] != V_0.shape[1]:
        V_0 = np.diag(np.diag(V_0))
    if initViM.shape[0] != initViM.shape[1]:
        initViM = np.diag(np.diag(initViM))
    if initViQ.shape[0] != initViQ.shape[1]:
        initViQ = np.diag(np.diag(initViQ))
    
    V_0 = block_diag(V_0, initViM, initViQ)
    
    # Diagnostic check: Verify all outputs are finite before returning
    output_params = {
        'A': A, 'C': C, 'Q': Q, 'R': R, 'Z_0': Z_0, 'V_0': V_0
    }
    
    for param_name, param_value in output_params.items():
        nan_count = np.sum(np.isnan(param_value)) if param_value.size > 0 else 0
        inf_count = np.sum(np.isinf(param_value)) if param_value.size > 0 else 0
        
        if nan_count > 0:
            _logger.warning(
                f"init_conditions: Output {param_name} contains {nan_count} NaN values "
                f"(shape: {param_value.shape})"
            )
        if inf_count > 0:
            _logger.warning(
                f"init_conditions: Output {param_name} contains {inf_count} Inf values "
                f"(shape: {param_value.shape})"
            )
        
        # Attempt to clean NaN/Inf values if detected
        if nan_count > 0 or inf_count > 0:
            if param_name in ['A', 'Q', 'V_0']:
                # Covariance matrices: regularize
                output_params[param_name] = _clean_matrix(param_value, 'covariance', default_nan=0.0)
            elif param_name == 'R':
                # Diagonal matrix: clean diagonal
                output_params[param_name] = _clean_matrix(param_value, 'diagonal', default_nan=1e-4)
            elif param_name == 'C':
                # Loading matrix: replace with zeros
                output_params[param_name] = _clean_matrix(param_value, 'loading')
            elif param_name == 'Z_0':
                # Initial state: reset to zeros
                output_params[param_name] = np.zeros_like(param_value)
    
    # Update variables with cleaned values
    A, C, Q, R, Z_0, V_0 = output_params['A'], output_params['C'], output_params['Q'], \
                           output_params['R'], output_params['Z_0'], output_params['V_0']
    
    return A, C, Q, R, Z_0, V_0


def em_step(y: np.ndarray, A: np.ndarray, C: np.ndarray, Q: np.ndarray,
           R: np.ndarray, Z_0: np.ndarray, V_0: np.ndarray,
           r: np.ndarray, p: int, R_mat: Optional[np.ndarray], q: Optional[np.ndarray],
           nQ: Optional[int], i_idio: np.ndarray, blocks: np.ndarray,
           tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
           clock: str = 'm',
           frequencies: Optional[np.ndarray] = None,
           config: Optional[DFMConfig] = None) -> Tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray, np.ndarray,
                                                                      np.ndarray, np.ndarray, float]:
    """Apply EM algorithm for parameter reestimation.
    
    This function implements one iteration of the Expectation-Maximization (EM) algorithm
    for dynamic factor models. It consists of:
    
    1. **E-step**: Run Kalman filter and smoother to obtain smoothed factor estimates
       and covariances: E[Z_t | data], E[Z_t Z_t' | data], etc.
    
    2. **M-step**: Update model parameters (C, R, A, Q) by maximizing the expected
       complete-data log-likelihood given the smoothed estimates.
    
    The function handles:
    - Monthly and quarterly series separately
    - Block structure for factors
    - Quarterly aggregation constraints
    - Idiosyncratic components (monthly and quarterly)
    - Missing data (via selection matrices Wt)
    
    Parameters
    ----------
    y : np.ndarray
        Data matrix (n x T), where n is number of series and T is time periods.
        Note: y is TRANSPOSED relative to input X (which is T x N).
        Missing values should be NaN. Data should be standardized.
    A : np.ndarray
        Current transition matrix (m x m). Describes factor dynamics.
    C : np.ndarray
        Current observation/loading matrix (n x m). Maps factors to observed series.
    Q : np.ndarray
        Current covariance matrix for transition residuals (m x m).
    R : np.ndarray
        Current covariance matrix for observation residuals (n x n). Typically diagonal.
    Z_0 : np.ndarray
        Current initial state vector (m,).
    V_0 : np.ndarray
        Current initial covariance matrix (m x m).
    r : np.ndarray
        Number of factors per block (n_blocks,). Typically all ones.
    p : int
        Number of lags in transition equation. Typically 1.
    R_mat : np.ndarray
        Quarterly aggregation constraints matrix (4 x 5). Same as Rcon in init_conditions.
    q : np.ndarray
        Constraints vector (4,). Typically zeros.
    nQ : int
        Number of quarterly variables.
    i_idio : np.ndarray
        Logical array (n,) indicating idiosyncratic components: 1 for monthly, 0 for quarterly.
    blocks : np.ndarray
        Block loading structure (n x n_blocks). See init_conditions() for details.
        
    Returns
    -------
    C_new : np.ndarray
        Updated observation matrix (n x m). Loadings are re-estimated via least squares
        using smoothed factor estimates.
    R_new : np.ndarray
        Updated observation residual covariance (n x n). Diagonal matrix containing
        updated idiosyncratic variances.
    A_new : np.ndarray
        Updated transition matrix (m x m). AR coefficients are re-estimated via OLS
        using smoothed factor estimates.
    Q_new : np.ndarray
        Updated transition residual covariance (m x m). Contains innovation variances.
    Z_0 : np.ndarray
        Updated initial state (m,). Typically unchanged from input.
    V_0_new : np.ndarray
        Updated initial covariance (m x m). Computed from smoothed state estimates.
    loglik : float
        Log-likelihood of the data under the current parameters. This is the
        incomplete-data log-likelihood computed by the Kalman filter.
        
    Raises
    ------
    ValueError
        If inputs are invalid or matrix dimensions don't match. Also raised if
        numerical issues occur (e.g., singular matrices, NaN/Inf propagation).
        
    Notes
    -----
    - This is a computationally intensive function that involves multiple matrix
      inversions and large matrix operations. Performance scales with O(T * m^3).
    - The function handles missing data via diagonal selection matrices Wt that
      zero out missing observations.
    - For quarterly series, loadings must satisfy aggregation constraints enforced
      via constrained least squares.
    - Some parameter updates may fail silently (e.g., singular matrices) and use
      previous values. These cases are handled gracefully to allow EM to continue.
      
    Examples
    --------
    >>> # Get initial parameters from init_conditions()
    >>> A, C, Q, R, Z_0, V_0 = init_conditions(...)
    >>> # Prepare data (transpose to n x T)
    >>> y = x.T  # n x T
    >>> # Run one EM step
    >>> C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
    ...     y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks
    ... )
    >>> # Update parameters for next iteration
    >>> A, C, Q, R, Z_0, V_0 = A_new, C_new, Q_new, R_new, Z_0_new, V_0_new
    """
    # Uncomment if debugging: _validate_em_step_inputs(y, A, C, Q, R, Z_0, V_0, r, p, blocks, nQ)
    
    # Numerical stability: check and clean parameter matrices before Kalman filter
    # This prevents NaN/Inf from propagating into the filter
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        _logger.warning("em_step: A contains NaN/Inf, using regularized identity")
        A = np.eye(A.shape[0]) * 0.9 + _clean_matrix(A, 'loading') * 0.1
    
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        _logger.warning("em_step: Q contains NaN/Inf, regularizing")
        Q = _clean_matrix(Q, 'covariance', default_nan=1e-6)
    
    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        _logger.warning("em_step: R contains NaN/Inf, regularizing")
        R = _clean_matrix(R, 'diagonal', default_nan=1e-4, default_inf=1e4)
    
    if np.any(np.isnan(C)) or np.any(np.isinf(C)):
        _logger.warning("em_step: C contains NaN/Inf, regularizing")
        C = _clean_matrix(C, 'loading')
    
    if np.any(np.isnan(Z_0)) or np.any(np.isinf(Z_0)):
        _logger.warning("em_step: Z_0 contains NaN/Inf, resetting to zeros")
        Z_0 = np.zeros_like(Z_0)
    
    if np.any(np.isnan(V_0)) or np.any(np.isinf(V_0)):
        _logger.warning("em_step: V_0 contains NaN/Inf, using regularized identity")
        V_0 = np.eye(V_0.shape[0]) * 0.1
    
    n, T = y.shape  # n series, T time periods
    
    # Determine nQ if not provided
    if nQ is None and frequencies is not None:
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        nQ = sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_hierarchy)
    elif nQ is None:
        nQ = 0
    
    # Determine pC from tent_weights_dict if available, otherwise use R_mat
    pC = 1
    if tent_weights_dict is not None and len(tent_weights_dict) > 0:
        # Find maximum tent kernel size
        for tent_weights in tent_weights_dict.values():
            if tent_weights is not None and len(tent_weights) > pC:
                pC = len(tent_weights)
    elif R_mat is not None:
        pC = R_mat.shape[1]
    
    ppC = int(max(p, pC))  # Ensure integer
    num_blocks = blocks.shape[1]
    
    # Get config defaults if not provided
    if config is None:
        from types import SimpleNamespace
        config = SimpleNamespace(
            clip_ar_coefficients=True,
            ar_clip_min=-0.99,
            ar_clip_max=0.99,
            warn_on_ar_clip=True,
            clip_data_values=True,
            data_clip_threshold=100.0,
            warn_on_data_clip=True,
            use_regularization=True,
            regularization_scale=1e-6,
            min_eigenvalue=1e-8,
            max_eigenvalue=1e6,
            warn_on_regularization=True,
            use_damped_updates=True,
            damping_factor=0.8,
            warn_on_damped_update=True
        )
    
    # E-step: Run Kalman filter and smoother
    # Kalman filter expects k x nobs, so we pass y as-is
    zsmooth, vsmooth, vvsmooth, loglik = run_kf(y, A, C, Q, R, Z_0, V_0)
    
    # zsmooth is m x (T+1) from run_kf (factors x time)
    # MATLAB: Zsmooth = runKF(y, A, C, Q, R, Z_0, V_0)' - transpose makes it (T+1) x m
    # So Zsmooth(t+1, bl_idx_same_freq(i,:)) selects time t+1, factors indicated by bl_idx_same_freq(i,:)
    # In Python: transpose to match MATLAB: (T+1) x m (time x factors)
    # Keep initial state (column 0) since MATLAB uses t+1 indexing (skips initial)
    Zsmooth = zsmooth.T  # m x (T+1) -> (T+1) x m
    
    # M-step: Update parameters
    # Initialize output
    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_0.copy()
    
    # Update factor parameters for each block
    for i in range(num_blocks):
        r_i = int(r[i])
        factor_lag_size = r_i * p  # Number of lagged factors for this block
        # Calculate starting index for this block's factors in the state vector
        # Sum of factors from all previous blocks, scaled by max_lag_size
        factor_start_idx = int(np.sum(r[:i]) * ppC)  # Python 0-based: sum of r[0] to r[i-1]
        # Subset of state vector for this block's factors (current and lagged)
        b_subset = slice(factor_start_idx, factor_start_idx + factor_lag_size)  # Python 0-based
        t_start = factor_start_idx  # Starting index for this block
        t_end = int(factor_start_idx + r_i * ppC)  # Ending index for this block
        
        # E[f_t * f_t' | Omega_T] - Expected value of factor outer product at time t
        # Zsmooth[b_subset, 1:] selects factors for this block, skipping initial state (column 0)
        expected_factor_outer = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, 1:].T + np.sum(vsmooth[b_subset, :, :][:, b_subset, 1:], axis=2)
        
        # E[f_{t-1} * f_{t-1}' | Omega_T] - Expected value of lagged factor outer product
        # Zsmooth[b_subset, :-1] selects factors for this block, excluding last time period
        expected_factor_lag_outer = Zsmooth[b_subset, :-1] @ Zsmooth[b_subset, :-1].T + np.sum(vsmooth[b_subset, :, :][:, b_subset, :-1], axis=2)
        
        # E[f_t * f_{t-1}' | Omega_T] - Expected value of factor-lag cross product
        expected_factor_lag_cross = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, :-1].T + np.sum(vvsmooth[b_subset, :, :][:, b_subset, :], axis=2)
        
        # Clean expected values before matrix operations
        expected_factor_lag_outer = _clean_matrix(expected_factor_lag_outer, 'covariance', default_nan=0.0)
        expected_factor_lag_cross = _clean_matrix(expected_factor_lag_cross, 'general', default_nan=0.0)
        
        # Update transition and innovation matrices for block i
        block_transition = A[t_start:t_end, t_start:t_end].copy()
        block_innovation_cov = Q[t_start:t_end, t_start:t_end].copy()
        
        try:
            # Check expected_factor_lag_outer before inversion
            expected_lag_sub = expected_factor_lag_outer[:factor_lag_size, :factor_lag_size]
            if np.any(np.isnan(expected_lag_sub)) or np.any(np.isinf(expected_lag_sub)):
                raise ValueError("Invalid values in expected_factor_lag_outer")
            
            # Regularize if needed (ensure positive definite)
            min_eigenval = config.min_eigenvalue if config else 1e-8
            warn_reg = config.warn_on_regularization if config else True
            expected_lag_sub, _ = _ensure_positive_definite(expected_lag_sub, min_eigenval, warn_reg)
            
            # Check if matrix is too ill-conditioned
            try:
                eigenvals = np.linalg.eigvals(expected_lag_sub)
                max_eigenval = np.max(eigenvals)
                min_eigenval = np.min(eigenvals)
                
                if max_eigenval > 0:
                    cond_num = max_eigenval / max(min_eigenval, 1e-12)
                    if cond_num > 1e12:
                        # Use pseudo-inverse for ill-conditioned matrices
                        try:
                            expected_lag_inv = pinv(expected_lag_sub, cond=1e-8)
                        except TypeError:
                            # Fallback for older scipy versions
                            expected_lag_inv = pinv(expected_lag_sub)
                    else:
                        expected_lag_inv = inv(expected_lag_sub)
                else:
                    # All eigenvalues are zero or negative - use identity
                    expected_lag_inv = np.eye(factor_lag_size) * (1.0 / max(1e-8, np.trace(expected_lag_sub) / factor_lag_size))
            except (np.linalg.LinAlgError, ValueError):
                # Eigendecomposition failed - use pseudo-inverse
                expected_lag_inv = pinv(expected_lag_sub)
            
            # Compute AR coefficients: block_transition = E[f_t * f_{t-1}'] * inv(E[f_{t-1} * f_{t-1}'])
            transition_update = expected_factor_lag_cross[:r_i, :factor_lag_size] @ expected_lag_inv
            
            # Cap AR coefficients to reasonable bounds (configurable)
            transition_update, _ = _apply_ar_clipping(transition_update, config)
            block_transition[:r_i, :factor_lag_size] = transition_update
            
            # Compute innovation covariance: block_innovation_cov = (E[f_t * f_t'] - block_transition * E[f_t * f_{t-1}']) / T
            block_innovation_cov[:r_i, :r_i] = (
                expected_factor_outer[:r_i, :r_i] - 
                block_transition[:r_i, :factor_lag_size] @ expected_factor_lag_cross[:r_i, :factor_lag_size].T
            ) / T
            
            # Clean innovation covariance and ensure positive semi-definite
            block_innovation_cov = _clean_matrix(block_innovation_cov, 'covariance', default_nan=0.0)
            
            # Ensure positive semi-definite for the factor block
            min_eigenval = config.min_eigenvalue if config else 1e-8
            innovation_cov_reg, reg_stats = _apply_regularization(
                block_innovation_cov[:r_i, :r_i], 'covariance', config
            )
            block_innovation_cov[:r_i, :r_i] = innovation_cov_reg
            
            # Cap maximum eigenvalue to prevent explosion
            max_eigenval = config.max_eigenvalue if config else 1e6
            block_innovation_cov[:r_i, :r_i] = _cap_max_eigenvalue(
                block_innovation_cov[:r_i, :r_i], max_eigenval=max_eigenval
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            # If update fails, use damped version of previous block_transition to prevent collapse
            if np.allclose(block_transition[:r_i, :factor_lag_size], 0):
                # If all zeros, initialize with small random values
                block_transition[:r_i, :factor_lag_size] = np.random.randn(r_i, factor_lag_size) * 0.1
            else:
                # Damp previous values slightly
                block_transition[:r_i, :factor_lag_size] = block_transition[:r_i, :factor_lag_size] * 0.95
            _logger.debug(f"em_step: Transition update failed for block {i+1}, using fallback: {type(e).__name__}")
        
        # Clean NaN/Inf from block_transition with reasonable bounds
        if np.any(~np.isfinite(block_transition)):
            block_transition = _clean_matrix(block_transition, 'loading', default_nan=0.0, default_inf=0.99)
            # Ensure AR coefficients are within stability bounds
            block_transition, _ = _apply_ar_clipping(block_transition, config)
        
        # Place updated block matrices in output
        A_new[t_start:t_end, t_start:t_end] = block_transition
        Q_new[t_start:t_end, t_start:t_end] = block_innovation_cov
        V_0_block = vsmooth[t_start:t_end, t_start:t_end, 0]
        # Ensure V_0_block is positive semi-definite
        V_0_block = _clean_matrix(V_0_block, 'covariance', default_nan=0.0)
        min_eigenval = config.min_eigenvalue if config else 1e-8
        warn_reg = config.warn_on_regularization if config else True
        V_0_block, _ = _ensure_positive_definite(V_0_block, min_eigenval, warn_reg)
        V_0_new[t_start:t_end, t_start:t_end] = V_0_block
    
    # Update idiosyncratic component
    # i_idio indicates which series have idiosyncratic components
    # For clock frequency: 1 (have idiosyncratic)
    # For slower frequencies: 0 (no direct idiosyncratic, use tent kernel structure)
    # Calculate starting index for idiosyncratic components in state vector
    # All factors come first, so idiosyncratic starts after all block factors
    idio_start_idx = int(np.sum(r) * ppC)
    n_idio = int(np.sum(i_idio))  # All series with idiosyncratic components
    idio_block_start = idio_start_idx
    i_subset = slice(idio_block_start, idio_block_start + n_idio)
    
    # E[f_t*f_t' | Omega_T] for idiosyncratic (diagonal only)
    # MATLAB: Zsmooth(t_start:end, 2:end) * Zsmooth(t_start:end, 2:end)'
    # Zsmooth is (T+1) x m (time x factors), so Zsmooth(t_start:end, 2:end) selects:
    #   - Rows: t_start to end (factor indices for idiosyncratic)
    #   - Columns: 2 to end (time periods 1 to T, skipping initial state)
    # Result is (m-t_start+1) x T
    # Then Zsmooth(...) * Zsmooth(...)' gives (m-t_start+1) x (m-t_start+1)
    # We want diagonal only: diag(diag(...))
    # 
    # t_start is the start of idiosyncratic component index
    # i_subset = idio_block_start:idio_block_start+n_idio selects the clock-frequency idiosyncratic factors only
    # So we should use i_subset to select rows, not t_start:end
    i_subset_slice = slice(i_subset.start, i_subset.stop)
    Z_idio = Zsmooth[i_subset_slice, 1:]  # n_idio x T, clock-frequency idiosyncratic factors only
    n_idio_actual = Z_idio.shape[0]  # Number of idiosyncratic factors (should be n_idio)
    
    # Compute expected values for idiosyncratic components (diagonal only, since they're independent)
    # E[z_t^2 | Omega_T] = sum(Z_idio^2) + diagonal of smoothed covariance
    expected_idio_current_sq = np.sum(Z_idio**2, axis=1)  # Sum over time
    vsmooth_idio_sum = np.sum(vsmooth[i_subset_slice, :, :][:, i_subset_slice, 1:], axis=2)
    vsmooth_idio_diag = np.diag(vsmooth_idio_sum)
    expected_idio_current_sq = expected_idio_current_sq + vsmooth_idio_diag
    
    # E[z_{t-1}^2 | Omega_T] for idiosyncratic (periods 0 to T-1)
    Z_idio_lag = Zsmooth[i_subset_slice, :-1]  # Time periods 0 to T-1
    expected_idio_lag_sq = np.sum(Z_idio_lag**2, axis=1)  # Sum over time
    vsmooth_lag_sum = np.sum(vsmooth[i_subset_slice, :, :][:, i_subset_slice, :-1], axis=2)
    vsmooth_lag_diag = np.diag(vsmooth_lag_sum)
    expected_idio_lag_sq = expected_idio_lag_sq + vsmooth_lag_diag
    
    # E[z_t * z_{t-1} | Omega_T] for idiosyncratic (element-wise product, sum over time)
    min_cols = min(Z_idio.shape[1], Z_idio_lag.shape[1])
    expected_idio_cross = np.sum(Z_idio[:, :min_cols] * Z_idio_lag[:, :min_cols], axis=1)
    vvsmooth_sum = np.sum(vvsmooth[i_subset_slice, :, :][:, i_subset_slice, :], axis=2)
    vvsmooth_diag = np.diag(vvsmooth_sum)
    expected_idio_cross = expected_idio_cross + vvsmooth_diag
    
    # Update A and Q for idiosyncratic components
    # Idiosyncratic components are independent AR(1) processes, so A_i and Q_i are diagonal
    # Theoretical formula: A_i[i,i] = E[z_t * z_{t-1}] / E[z_{t-1}^2] for each series i
    
    # Estimate AR coefficients using helper function
    ar_coeffs_diag, _ = _estimate_ar_coefficient(
        expected_idio_cross, expected_idio_lag_sq, vsmooth_sum=vsmooth_lag_diag
    )
    
    # Create diagonal transition matrix
    block_transition_idio = np.diag(ar_coeffs_diag)
    
    # Compute innovation covariance: Q_i[i,i] = (E[z_t^2] - A_i[i,i] * E[z_t * z_{t-1}]) / T
    # Clean expected values before computation
    expected_idio_current_sq = np.nan_to_num(expected_idio_current_sq, nan=0.0, posinf=1e6, neginf=-1e6)
    innovation_cov_diag = (expected_idio_current_sq - ar_coeffs_diag * expected_idio_cross) / T
    
    # Ensure innovation covariance is positive (variance must be non-negative)
    innovation_cov_diag = np.maximum(innovation_cov_diag, 1e-8)
    
    # Create diagonal innovation covariance matrix
    block_innovation_cov_idio = np.diag(innovation_cov_diag)
    
    # Place in output matrices (only clock-frequency idiosyncratic)
    # block_transition_idio and block_innovation_cov_idio are n_idio_actual x n_idio_actual
    if n_idio_actual == n_idio:
        A_new[i_subset, i_subset] = block_transition_idio
        Q_new[i_subset, i_subset] = block_innovation_cov_idio
    else:
        # If sizes don't match, take the appropriate slice
        A_new[i_subset, i_subset] = block_transition_idio[:n_idio, :n_idio]
        Q_new[i_subset, i_subset] = block_innovation_cov_idio[:n_idio, :n_idio]
    
    # V_0_new: MATLAB uses diag(diag(Vsmooth(i_subset, i_subset, 1)))
    # Extract diagonal only and place on diagonal of submatrix
    vsmooth_sub = vsmooth[i_subset_slice, :, :][:, i_subset_slice, 0]  # n_idio_actual x n_idio_actual
    vsmooth_diag = np.diag(vsmooth_sub)  # n_idio_actual values
    # Place diagonal values on the diagonal of V_0_new submatrix
    for idx, val in enumerate(vsmooth_diag):
        V_0_new[i_subset.start + idx, i_subset.start + idx] = val
    
    # Update observation matrix C and R
    # MATLAB: Z_0 = Zsmooth(:,1) - selects column 1 (initial state, 1-based)
    # In Python: Zsmooth is (T+1) x m (time x factors), so Zsmooth[0, :] is initial state
    Z_0 = Zsmooth[0, :].copy()  # Initial state (first row)
    
    # Set missing data to 0
    nanY = np.isnan(y)
    y_clean = y.copy()
    y_clean[nanY] = 0
    
    # Blocks - get unique block loading patterns
    bl = np.unique(blocks, axis=0)  # Gives unique loadings
    n_bl = bl.shape[0]  # Number of unique loadings
    
    # Initialize indices: These help with subsetting
    # Initialize as None - will be set to numpy arrays on first iteration
    bl_idx_same_freq = None  # Indicator for same-frequency factor loadings
    bl_idx_slower_freq = None  # Indicator for slower-frequency factor loadings
    R_con_list = []  # List to build block diagonal
    
    # Loop through each block to build indices
    # MATLAB: for i = 1:num_blocks
    #   bl_idx_slower_freq = [bl_idx_slower_freq repmat(bl(:,i),1,r(i)*ppC)];
    #   bl_idx_same_freq = [bl_idx_same_freq repmat(bl(:,i),1,r(i)) zeros(n_bl,r(i)*(ppC-1))];
    # end
    # This builds column by column - each column represents a factor position
    # bl(:,i) is n_bl x 1, repmat makes it n_bl x r(i)*ppC
    for i in range(num_blocks):
        r_i = int(r[i])
        # bl[:, i] repeated r(i) times horizontally for clock frequency (lag 0 only)
        # Then zeros for remaining ppC-1 lags
        bl_col_clock_freq = np.repeat(bl[:, i:i+1], r_i, axis=1)  # n_bl x r_i
        bl_col_clock_freq = np.hstack([bl_col_clock_freq, np.zeros((n_bl, r_i * (ppC - 1)))])  # n_bl x (r_i * ppC)
        
        # bl[:, i] repeated r(i)*ppC times for slower frequencies (all lags)
        bl_col_slower_freq = np.repeat(bl[:, i:i+1], r_i * ppC, axis=1)  # n_bl x (r_i * ppC)
        
        # Append columns (MATLAB uses horizontal concatenation)
        if bl_idx_same_freq is None:
            bl_idx_same_freq = bl_col_clock_freq
            bl_idx_slower_freq = bl_col_slower_freq
        else:
            bl_idx_same_freq = np.hstack([bl_idx_same_freq, bl_col_clock_freq])
            bl_idx_slower_freq = np.hstack([bl_idx_slower_freq, bl_col_slower_freq])
        
        # Build constraint matrix: kron(R_mat, eye(r(i))) - only if R_mat is provided
        if R_mat is not None:
            R_con_list.append(np.kron(R_mat, np.eye(r_i)))
    
    # Convert to boolean (MATLAB: logical)
    # Handle case where no blocks were processed (shouldn't happen, but be safe)
    if bl_idx_same_freq is not None:
        bl_idx_same_freq = bl_idx_same_freq.astype(bool)
        bl_idx_slower_freq = bl_idx_slower_freq.astype(bool)
    else:
        # No blocks processed - create empty boolean arrays
        bl_idx_same_freq = np.array([]).reshape(n_bl, 0).astype(bool)
        bl_idx_slower_freq = np.array([]).reshape(n_bl, 0).astype(bool)
    
    # Build block diagonal constraint matrix
    if len(R_con_list) > 0:
        R_con = block_diag(*R_con_list)
    else:
        R_con = np.array([])
    
    # Set up constraint vector
    if R_mat is not None and q is not None:
        q_con = np.zeros((np.sum(r.astype(int)) * R_mat.shape[0], 1))
    else:
        q_con = np.array([])
    
    # Idiosyncratic indicators
    # i_idio indicates which series have idiosyncratic components
    # For clock frequency: 1 (have idiosyncratic)
    # For slower frequencies: 0 (no direct idiosyncratic, use tent kernel structure)
    i_idio_same = i_idio  # All series with idiosyncratic components
    n_idio_same = int(np.sum(i_idio_same))
    c_i_idio = np.cumsum(i_idio.astype(int))
    
    # Initialize C_new
    C_new = C.copy()
    
    # Loop through unique block patterns
    for i in range(n_bl):
        bl_i = bl[i, :]
        rs = int(np.sum(r[bl_i.astype(bool)]))  # Total factors for this block pattern
        idx_i = np.where((blocks == bl_i).all(axis=1))[0]  # Series with this pattern
        
        # Group series by their actual frequency
        freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
        idx_clock_freq = freq_groups.get(clock, np.array([], dtype=int))  # Same frequency as clock
        
        n_i = len(idx_clock_freq)
        
        if n_i == 0:
            continue
        
        # Get bl_idx_same_freq indices for this block pattern (same frequency: lag 0 only)
        bl_idx_same_freq_i = np.where(bl_idx_same_freq[i, :])[0]
        if len(bl_idx_same_freq_i) == 0:
            continue
        
        # Verify rs matches bl_idx_same_freq_i length
        rs_actual = len(bl_idx_same_freq_i)
        if rs_actual != rs:
            # If mismatch, use actual length
            rs = rs_actual
        
        # Initialize sums for equation 13 (BGR 2010)
        # denom should be (n_i * rs) x (n_i * rs)
        denom_size = n_i * rs
        denom = np.zeros((denom_size, denom_size))
        nom = np.zeros((n_i, rs))
        
        # Clock-frequency indices for idiosyncratic
        i_idio_i = i_idio_same[idx_clock_freq]
        i_idio_ii = c_i_idio[idx_clock_freq]
        i_idio_ii = i_idio_ii[i_idio_i.astype(bool)]
        
        # Loop through each period for clock-frequency variables
        for t in range(T):
            # Wt is diagonal selection matrix: n_i x n_i
            nan_mask = ~nanY[idx_clock_freq, t]
            Wt = np.diag(nan_mask.astype(float))
            
            # Ensure Wt is correct size
            if Wt.shape[0] != n_i:
                Wt = np.diag(np.ones(n_i))
            
            # E[f_t*f_t' | Omega_T]
            # Zsmooth(bl_idx_same_freq(i, :), t+1) selects row t+1, columns where bl_idx_same_freq(i,:) is True
            # This gives 1 x rs (row vector)
            # Then Zsmooth(...)' * Zsmooth(...) = (rs x 1) * (1 x rs) = rs x rs
            
            # Zsmooth is (T+1) x m, so Zsmooth[t+1, bl_idx_same_freq_i] gives 1 x rs
            if t + 1 < Zsmooth.shape[0]:
                Z_block_same_freq_row = Zsmooth[t + 1, bl_idx_same_freq_i]  # 1 x rs (row vector)
                ZZZ = Z_block_same_freq_row.reshape(-1, 1) @ Z_block_same_freq_row.reshape(1, -1)  # rs x 1 * 1 x rs = rs x rs
            else:
                ZZZ = np.zeros((rs, rs))
            
            # V_block_same_freq: vsmooth is m x m x (T+1), so vsmooth[bl_idx_same_freq_i, bl_idx_same_freq_i, t+1] is rs x rs
            if t + 1 < vsmooth.shape[2]:
                V_block_same_freq = vsmooth[np.ix_(bl_idx_same_freq_i, bl_idx_same_freq_i, [t + 1])]
                if V_block_same_freq.ndim == 3:
                    V_block_same_freq = V_block_same_freq[:, :, 0]  # Extract 2D slice
                if V_block_same_freq.shape != (rs, rs):
                    V_block_same_freq = np.zeros((rs, rs))
            else:
                V_block_same_freq = np.zeros((rs, rs))
            
            # kron should produce (n_i * rs) x (n_i * rs)
            # Wt is n_i x n_i, ZZZ + V_block_same_freq is rs x rs
            # kron((rs x rs), (n_i x n_i)) = (rs * n_i) x (rs * n_i) = (n_i * rs) x (n_i * rs)
            # Ensure dimensions match before kron
            expected_shape = (denom_size, denom_size)
            if ZZZ.shape == (rs, rs) and V_block_same_freq.shape == (rs, rs) and Wt.shape == (n_i, n_i):
                try:
                    kron_result = np.kron(ZZZ + V_block_same_freq, Wt)
                    if kron_result.shape == expected_shape:
                        denom += kron_result
                    else:
                        # Dimension mismatch - skip this term
                        pass
                except ValueError:
                    # If kron fails, skip this term
                    pass
            
            # E[y_t*f_t' | Omega_T]
            # y is N x T, so y(idx_clock_freq, t) is n_i x 1 (column vector)
            # Zsmooth(bl_idx_same_freq(i, :), t+1) is 1 x rs, so transpose gives rs x 1
            # y(idx_clock_freq, t) * Zsmooth(...)' gives n_i x 1 * rs x 1 = n_i x rs
            # Actually: (n_i x 1) * (1 x rs) = n_i x rs
            if t + 1 < Zsmooth.shape[0]:
                y_vec = y_clean[idx_clock_freq, t].reshape(-1, 1)  # n_i x 1
                Z_vec_row = Zsmooth[t + 1, bl_idx_same_freq_i].reshape(1, -1)  # 1 x rs
                y_term = y_vec @ Z_vec_row  # n_i x 1 * 1 x rs = n_i x rs
            else:
                y_term = np.zeros((len(idx_clock_freq), rs_actual))
            
            # Idiosyncratic component
            # Wt(:, i_idio_i) * (Zsmooth(idio_start_idx + i_idio_ii, t+1) * Zsmooth(bl_idx_same_freq(i, :), t+1)' + ...)
            # Zsmooth(idio_start_idx + i_idio_ii, t+1) is len(i_idio_ii) x 1, Zsmooth(bl_idx_same_freq(i, :), t+1)' is rs x 1
            # So product is len(i_idio_ii) x rs
            if len(i_idio_ii) > 0 and t + 1 < Zsmooth.shape[0]:
                idio_idx = (idio_start_idx + i_idio_ii).astype(int)
                if idio_idx.max() < Zsmooth.shape[1]:  # Zsmooth is (T+1) x m
                    idio_Z_col = Zsmooth[t + 1, idio_idx].reshape(-1, 1)  # len(i_idio_ii) x 1
                    idio_Z_outer = idio_Z_col @ Z_vec_row  # len(i_idio_ii) x 1 * 1 x rs = len(i_idio_ii) x rs
                    
                    if t + 1 < vsmooth.shape[2]:
                        idio_V = vsmooth[np.ix_(idio_idx, bl_idx_same_freq_i, [t + 1])]
                        if idio_V.ndim == 3:
                            idio_V = idio_V[:, :, 0]  # len(i_idio_ii) x rs
                    else:
                        idio_V = np.zeros((len(i_idio_ii), rs_actual))
                    
                    idio_term = Wt[:, i_idio_i.astype(bool)] @ (idio_Z_outer + idio_V)  # n_i x len(i_idio_i) * len(i_idio_i) x rs = n_i x rs
                else:
                    idio_term = np.zeros((len(idx_clock_freq), rs_actual))
            else:
                idio_term = np.zeros((len(idx_clock_freq), rs_actual))
            
            nom += y_term - idio_term
        
        # Solve for C using regularized least squares
        # Add ridge regularization to prevent extreme loadings while maintaining theoretical soundness
        # Regularization parameter based on data scale
        try:
            # Compute regularization parameter based on trace of denom (data scale)
            scale_factor = config.regularization_scale if config else 1e-6
            warn_reg = config.warn_on_regularization if config else True
            reg_param, reg_stats = _compute_regularization_param(denom, scale_factor, warn_reg)
            
            # Regularized solution: (denom + lambda*I)^(-1) @ nom
            denom_reg = denom + np.eye(denom.shape[0]) * reg_param
            vec_C = inv(denom_reg) @ nom.flatten()
            
            # Assign using explicit indexing to avoid read-only issues
            C_update = vec_C.reshape(n_i, rs)
            
            # Clean invalid values (preserve scale, don't hard clip)
            C_update = _clean_matrix(C_update, 'loading', default_nan=0.0, default_inf=0.0)
            
            # Only clip extreme outliers (beyond reasonable range for standardized data)
            # Use adaptive bounds based on data scale, not fixed values
            C_scale = np.std(C_update[C_update != 0]) if np.any(C_update != 0) else 1.0
            C_max = max(10.0, C_scale * 5)  # Adaptive bound: 5 standard deviations or 10, whichever is larger
            C_update = np.clip(C_update, -C_max, C_max)
            
            for i, row_idx in enumerate(idx_clock_freq):
                for j, col_idx in enumerate(bl_idx_same_freq_i):
                    C_new[row_idx, col_idx] = C_update[i, j]
        except (np.linalg.LinAlgError, ValueError) as e:
            # Matrix inversion failed or dimension mismatch - skip this block
            # This can happen if denom is singular or shapes don't match
            # Log would be helpful but keeping silent for now to match MATLAB behavior
            pass
        
        # Process series with frequencies different from clock (need tent kernel)
        # Process each frequency group separately
        for freq, idx_iFreq in freq_groups.items():
            if freq == clock:
                continue  # Already processed above
            # Get tent weights for this frequency
            if tent_weights_dict is not None and freq in tent_weights_dict:
                tent_weights = tent_weights_dict[freq]
            else:
                tent_weights = get_tent_weights_for_pair(freq, clock)
                if tent_weights is None:
                    # Fallback: generate symmetric tent weights based on frequency gap
                    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                    freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                    n_periods_est = freq_hierarchy - clock_hierarchy + 1
                    if n_periods_est > 0 and n_periods_est <= 12:
                        tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                        _logger.warning(
                            f"em_step: No tent weights found for frequency '{freq}' "
                            f"with clock '{clock}', generating symmetric tent weights with {n_periods_est} periods."
                        )
                    else:
                        raise ValueError(
                            f"em_step: Cannot determine tent weights for frequency '{freq}' "
                            f"with clock '{clock}'. Please provide tent_weights_dict or ensure "
                            f"the frequency pair is supported."
                        )
            
            pC_freq = len(tent_weights)
            rps = rs * pC_freq
            
            # Generate R_mat for this frequency
            R_mat_freq, q_freq = generate_R_mat(tent_weights)
            
            # Build constraint matrix for this frequency and block pattern
            R_con_i = np.kron(R_mat_freq, np.eye(int(rs)))
            q_con_i = np.kron(q_freq, np.zeros(int(rs)))
            
            # Get bl_idx_slower_freq indices for this frequency
            # bl_idx_slower_freq contains all slower frequency factors (all lags)
            # We need to select the appropriate columns for this frequency
            # For now, we'll use the full bl_idx_slower_freq and let the constraint matrix handle it
            if i < bl_idx_slower_freq.shape[0]:
                # Select columns corresponding to this block pattern and frequency
                # bl_idx_slower_freq[i, :] gives all slower frequency factors for block pattern i
                # We need to select the first rps columns (for this frequency)
                bl_idx_slower_freq_i = np.where(bl_idx_slower_freq[i, :])[0]
                if len(bl_idx_slower_freq_i) >= rps:
                    bl_idx_slower_freq_i = bl_idx_slower_freq_i[:rps]  # Take first rps columns
                elif len(bl_idx_slower_freq_i) > 0:
                    # Pad if needed
                    bl_idx_slower_freq_i = np.pad(bl_idx_slower_freq_i, (0, rps - len(bl_idx_slower_freq_i)), mode='edge')
                else:
                    # No slower frequency factors - skip
                    continue
            else:
                continue
            
            # Remove empty constraints
            if R_con_i.size > 0:
                no_c = ~np.any(R_con_i, axis=1)
                R_con_i = R_con_i[~no_c, :]
                q_con_i = q_con_i[~no_c]
            
            # Loop through series of this frequency in this block pattern
            for j in idx_iFreq:
                # Determine actual size based on bl_idx_slower_freq_i
                if len(bl_idx_slower_freq_i) > 0:
                    rps_actual = len(bl_idx_slower_freq_i)
                else:
                    rps_actual = rps
                
                denom = np.zeros((rps_actual, rps_actual))
                nom = np.zeros((1, rps_actual))
                
                # Find ordinal position of this slower frequency series
                # Count how many slower frequency series come before this one
                idx_j_slower = sum(1 for k in idx_iFreq if k < j and 
                                 (frequencies is None or k >= len(frequencies) or frequencies[k] == freq))
                
                # Location of factor structure for slower frequency var residuals
                # Each slower frequency series has n_periods idiosyncratic components
                # Count total idiosyncratic components for all previous slower frequency series
                n_idio_slower_before = 0
                for prev_freq, prev_idx_list in slower_freqs.items():
                    if prev_freq == freq:
                        # Count only series before j in this frequency group
                        n_idio_slower_before += sum(1 for k in prev_idx_list if k < j) * pC_freq
                        break
                    else:
                        # Count all series in previous frequency groups
                        if tent_weights_dict is not None and prev_freq in tent_weights_dict:
                            prev_tent = tent_weights_dict[prev_freq]
                        else:
                            prev_tent = get_tent_weights_for_pair(prev_freq, clock)
                            if prev_tent is None:
                                # Fallback: generate symmetric tent weights
                                clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
                                prev_freq_hierarchy = FREQUENCY_HIERARCHY.get(prev_freq, 3)
                                n_periods_est = prev_freq_hierarchy - clock_hierarchy + 1
                                if n_periods_est > 0 and n_periods_est <= 12:
                                    prev_tent = generate_tent_weights(n_periods_est, 'symmetric')
                                else:
                                    # If cannot generate, use empty array (will cause error downstream)
                                    prev_tent = np.array([])
                        n_idio_slower_before += len(prev_idx_list) * len(prev_tent)
                
                i_idio_jQ = np.arange(idio_start_idx + n_idio_same + n_idio_slower_before, 
                                      idio_start_idx + n_idio_same + n_idio_slower_before + pC_freq)
                
                # Update V_0, A, Q for slower frequency idiosyncratic
                # Each slower frequency series has n_periods idiosyncratic components (tent structure)
                # These components represent the same underlying slower frequency idiosyncratic process,
                # so they should share the same AR coefficient, estimated from their aggregate behavior
                
                if len(i_idio_jQ) > 0 and i_idio_jQ[-1] < V_0_new.shape[0]:
                    # Update V_0 for slower frequency idiosyncratic
                    vsmooth_Q = vsmooth[i_idio_jQ, :, :][:, i_idio_jQ, 0]  # Extract submatrix
                    # Place values explicitly to avoid broadcasting issues
                    for idx1, i1 in enumerate(i_idio_jQ):
                        for idx2, i2 in enumerate(i_idio_jQ):
                            if i1 < V_0_new.shape[0] and i2 < V_0_new.shape[1]:
                                V_0_new[i1, i2] = vsmooth_Q[idx1, idx2]
                    
                    # Estimate AR coefficient for this slower frequency series from its smoothed idiosyncratic factors
                    # Extract smoothed idiosyncratic factors for this slower frequency series (n_periods components)
                    # Zsmooth is (T+1) x m, so Zsmooth[:, i_idio_jQ] gives (T+1) x n_periods
                    if i_idio_jQ[0] < Zsmooth.shape[1] and i_idio_jQ[-1] < Zsmooth.shape[1]:
                        Z_idio_Q = Zsmooth[1:, i_idio_jQ]  # T x n_periods, skip initial state
                        Z_idio_Q_BB = Zsmooth[:-1, i_idio_jQ]  # T x n_periods, periods 0 to T-1
                        
                        # Aggregate the n_periods components using tent weights to get a single slower frequency idiosyncratic series
                        # This represents the underlying slower frequency process
                        tent_weights_norm = tent_weights / np.sum(tent_weights)  # Normalize tent weights
                        z_Q_agg = (Z_idio_Q @ tent_weights_norm)  # T x 1, aggregated slower frequency idiosyncratic
                        z_Q_agg_BB = (Z_idio_Q_BB @ tent_weights_norm)  # T x 1, lagged
                        
                        # Estimate AR coefficient: A_Q = E[z_t * z_{t-1}] / E[z_{t-1}^2]
                        # This is the same formula as for same frequency idiosyncratic, but applied to aggregated slower frequency
                        EZZ_BB_Q = np.sum(z_Q_agg_BB**2)  # Scalar
                        EZZ_FB_Q = np.sum(z_Q_agg * z_Q_agg_BB)  # Scalar
                        
                        # Add variance from smoothed covariance
                        vsmooth_Q_sum = np.sum(vsmooth[i_idio_jQ, :, :][:, i_idio_jQ, :-1], axis=(0, 1, 2))
                        EZZ_BB_Q += vsmooth_Q_sum * np.sum(tent_weights_norm**2)  # Scale by normalized tent weights
                        
                        # Estimate AR coefficient for slower frequency series
                        # Regularize denominator to prevent division by zero
                        min_denom_Q = max(abs(EZZ_BB_Q) * 1e-6, 1e-10)
                        EZZ_BB_Q = max(EZZ_BB_Q, min_denom_Q)
                        
                        # Compute AR coefficient: A_Q = E[z_t * z_{t-1}] / E[z_{t-1}^2]
                        A_Q = EZZ_FB_Q / EZZ_BB_Q
                        A_Q, _ = _apply_ar_clipping(A_Q, config)  # Stability bounds
                        
                        # Compute innovation variance: Q_Q = (E[z_t^2] - A_Q * E[z_t * z_{t-1}]) / T
                        EZZ_Q = np.sum(z_Q_agg**2)
                        vsmooth_Q_sum_current = np.sum(vsmooth[i_idio_jQ, :, :][:, i_idio_jQ, 1:], axis=(0, 1, 2))
                        EZZ_Q += vsmooth_Q_sum_current * np.sum(tent_weights_norm**2)
                        Q_Q = (EZZ_Q - A_Q * EZZ_FB_Q) / T
                        Q_Q = max(Q_Q, 1e-8)  # Ensure positive
                        
                        # Apply the same AR coefficient and variance to all n_periods components of this slower frequency series
                        # This is theoretically correct: they represent the same underlying process
                        for idx_Q in i_idio_jQ:
                            if idx_Q < A_new.shape[0] and idx_Q < A_new.shape[1]:
                                A_new[idx_Q, idx_Q] = A_Q
                                Q_new[idx_Q, idx_Q] = Q_Q
                
                # Loop through each period for slower frequency variables
                for t in range(T):
                    # MATLAB: Wt = diag(~nanY(j, t)) where j is scalar, nanY(j, t) is scalar
                    # In MATLAB, diag(scalar) gives 1x1 matrix
                    # In Python, np.diag() requires 1D array, so wrap scalar in array
                    nan_val = ~nanY[j, t]
                    if np.isscalar(nan_val):
                        # Scalar case: create 1x1 matrix
                        Wt = np.array([[float(nan_val)]])
                    else:
                        # Array case: use np.diag() normally
                        Wt = np.diag(nan_val.astype(float))
                    
                    # Zsmooth(bl_idx_slower_freq(i,:), t+1) - Zsmooth is (T+1) x m
                    # Ensure bl_idx_slower_freq_i indices are valid
                    if len(bl_idx_slower_freq_i) == 0:
                        # No slower frequency factors for this block - skip
                        continue
                    
                    # Ensure indices are within bounds
                    valid_bl_idx_slower_freq_i = bl_idx_slower_freq_i[bl_idx_slower_freq_i < Zsmooth.shape[1]]
                    if len(valid_bl_idx_slower_freq_i) == 0:
                        # No valid indices - skip
                        continue
                    
                    if t + 1 < Zsmooth.shape[0]:
                        Z_block_slower_freq_row = Zsmooth[t + 1, valid_bl_idx_slower_freq_i]  # 1 x rps_actual
                        Z_block_slower_freq_col = Z_block_slower_freq_row.reshape(-1, 1)  # rps_actual x 1
                        ZZZ_slower_freq = Z_block_slower_freq_col @ Z_block_slower_freq_row.reshape(1, -1)  # rps_actual x 1 * 1 x rps_actual = rps_actual x rps_actual
                        
                        # Ensure vsmooth indices are valid
                        valid_vsmooth_idx = valid_bl_idx_slower_freq_i[valid_bl_idx_slower_freq_i < vsmooth.shape[0]]
                        if len(valid_vsmooth_idx) > 0:
                            V_block_slower_freq = vsmooth[np.ix_(valid_vsmooth_idx, valid_vsmooth_idx, [t + 1])]
                            if V_block_slower_freq.ndim == 3:
                                V_block_slower_freq = V_block_slower_freq[:, :, 0]
                            # Ensure V_block_slower_freq matches ZZZ_slower_freq size
                            if V_block_slower_freq.shape != ZZZ_slower_freq.shape:
                                # Resize to match
                                min_size = min(V_block_slower_freq.shape[0], ZZZ_slower_freq.shape[0])
                                V_block_slower_freq = V_block_slower_freq[:min_size, :min_size]
                                ZZZ_slower_freq = ZZZ_slower_freq[:min_size, :min_size]
                        else:
                            V_block_slower_freq = np.zeros_like(ZZZ_slower_freq)
                    else:
                        # t+1 is out of bounds - use zeros
                        Z_block_slower_freq_row = np.zeros(rps_actual)
                        Z_block_slower_freq_col = np.zeros((rps_actual, 1))
                        ZZZ_slower_freq = np.zeros((rps_actual, rps_actual))
                        V_block_slower_freq = np.zeros((rps_actual, rps_actual))
                    
                    # E[f_t*f_t' | Omega_T]
                    # kron(Zsmooth(...)*Zsmooth(...)' + Vsmooth(...), Wt)
                    # Wt is 1x1 matrix for single slower frequency series, so kron((rps x rps), (1 x 1)) = rps x rps
                    # Since Wt is 1x1, kron(A, Wt) = A * Wt[0,0]
                    if Wt.shape == (1, 1):
                        denom += (ZZZ_slower_freq + V_block_slower_freq) * Wt[0, 0]
                    else:
                        denom += np.kron(ZZZ_slower_freq + V_block_slower_freq, Wt)
                    
                    # E[y_t*f_t' | Omega_T]
                    # y(j, t) * Zsmooth(bl_idx_slower_freq(i,:), t+1)' - y is scalar, Zsmooth(...)' is rps x 1
                    # Result is 1 x rps
                    nom += y_clean[j, t] * Z_block_slower_freq_row.reshape(1, -1)  # 1 x rps
                    
                    # Subtract idiosyncratic component
                    # Use the tent weights for this frequency
                    if len(i_idio_jQ) > 0 and t + 1 < Zsmooth.shape[0] and i_idio_jQ[-1] < Zsmooth.shape[1]:
                        # Use tent weights for this frequency (already determined above)
                        tent = tent_weights.reshape(1, -1)
                        
                        # Ensure tent matches i_idio_jQ length
                        n_periods_actual = len(i_idio_jQ)
                        if tent.shape[1] != n_periods_actual:
                            # Adjust tent to match actual periods (truncate or pad)
                            if tent.shape[1] > n_periods_actual:
                                tent = tent[:, :n_periods_actual]
                            else:
                                # Pad with last value
                                pad_width = n_periods_actual - tent.shape[1]
                                tent = np.hstack([tent, np.tile(tent[:, -1:], (1, pad_width))])
                        
                        idio_Z_col = Zsmooth[t + 1, i_idio_jQ].reshape(-1, 1)  # n_periods x 1
                        idio_term_scalar = (tent @ idio_Z_col) * Z_block_slower_freq_row  # scalar * 1 x rps = 1 x rps
                        
                        if t + 1 < vsmooth.shape[2]:
                            idio_V = vsmooth[np.ix_(i_idio_jQ, bl_idx_slower_freq_i, [t + 1])]
                            if idio_V.ndim == 3:
                                idio_V = idio_V[:, :, 0]  # n_periods x rps
                            idio_term_V = tent @ idio_V  # 1 x n_periods * n_periods x rps = 1 x rps
                        else:
                            idio_term_V = np.zeros((1, len(bl_idx_slower_freq_i)))
                        
                        # Wt is 1x1 matrix for slower frequency, extract scalar value
                        wt_scalar = Wt[0, 0] if Wt.shape == (1, 1) else Wt
                        nom -= wt_scalar * (idio_term_scalar + idio_term_V)  # Scalar * 1 x rps
                
                # Solve for C (with constraints if applicable)
                # Use regularized least squares for better numerical stability
                try:
                    # Compute regularization parameter based on data scale
                    scale_factor = config.regularization_scale if config else 1e-6
                    warn_reg = config.warn_on_regularization if config else True
                    reg_param, reg_stats = _compute_regularization_param(denom, scale_factor, warn_reg)
                    
                    # Regularized solution before constraints
                    denom_reg = denom + np.eye(denom.shape[0]) * reg_param
                    C_i = inv(denom_reg) @ nom.T
                    
                    # Clean invalid values
                    C_i = _clean_matrix(C_i, 'loading', default_nan=0.0, default_inf=0.0)
                    
                    # Apply constraints (tent kernel aggregation)
                    if R_con_i.size > 0 and len(q_con_i) == R_con_i.shape[0]:
                        denom_inv = inv(denom_reg)  # Use regularized inverse for constraints too
                        constraint_term = denom_inv @ R_con_i.T @ inv(R_con_i @ denom_inv @ R_con_i.T) @ (R_con_i @ C_i - q_con_i)
                        C_i_constr = C_i - constraint_term
                    else:
                        C_i_constr = C_i
                    
                    # Clean and apply adaptive bounds after constraints
                    C_i_constr = _clean_matrix(C_i_constr, 'loading', default_nan=0.0, default_inf=0.0)
                    C_scale = np.std(C_i_constr[C_i_constr != 0]) if np.any(C_i_constr != 0) else 1.0
                    C_max = max(10.0, C_scale * 5)
                    C_i_constr = np.clip(C_i_constr, -C_max, C_max)
                    
                    # Place in output matrix
                    # Assign using explicit indexing to avoid read-only issues
                    if len(bl_idx_slower_freq_i) > 0:
                        C_update = C_i_constr.flatten()[:len(bl_idx_slower_freq_i)]
                        for k, col_idx in enumerate(bl_idx_slower_freq_i):
                            C_new[j, col_idx] = C_update[k]
                except (np.linalg.LinAlgError, ValueError, IndexError) as e:
                    # Matrix inversion failed, dimension mismatch, or index error - skip this series
                    # This can happen if denom is singular, constraints are inconsistent, or indices invalid
                    pass
    
    # Update R (covariance of observation residuals)
    # R is diagonal, so we compute diagonal elements directly (more efficient and theoretically cleaner)
    # Theoretical formula: R[i,i] = (1/T_i) * sum_{t: obs available} [ (y_i(t) - C_i @ Z_t)^2 + C_i @ V_t @ C_i' ]
    # where T_i is number of non-missing observations for series i
    # This is the EM update for observation residual variance
    
    R_diag = np.zeros(n)
    n_obs_per_series = np.zeros(n, dtype=int)  # Count non-missing observations per series
    
    for t in range(T):
        # Get smoothed factor and covariance at time t
        Z_t = Zsmooth[t + 1, :].reshape(-1, 1)  # m x 1
        vsmooth_t = vsmooth[:, :, t + 1]  # m x m
        
        # Compute prediction for all series: y_pred = C_new @ Z_t gives n x 1
        y_pred = C_new @ Z_t  # Use current C, not C_new, for consistency in this iteration
        y_pred = y_pred.flatten()  # n x 1 -> (n,)
        
        # For each series i, compute R[i,i] contribution
        for i in range(n):
            # Skip if observation is missing
            if nanY[i, t]:
                continue
            
            # Increment observation count
            n_obs_per_series[i] += 1
            
            # Prediction error squared: (y_i(t) - C_i @ Z_t)^2
            resid_i = y_clean[i, t] - y_pred[i]
            resid_sq = resid_i**2
            
            # Uncertainty from factor estimation: C_i @ V_t @ C_i'
            # This accounts for uncertainty in the smoothed factor estimates
            C_i = C_new[i, :].reshape(1, -1)  # 1 x m
            var_factor = (C_i @ vsmooth_t @ C_i.T)[0, 0]  # Scalar
            
            # Add to R[i,i] accumulator
            R_diag[i] += resid_sq + var_factor
    
    # Average over time periods (only count non-missing observations)
    # Avoid division by zero - if no observations, keep previous R value
    n_obs_per_series = np.maximum(n_obs_per_series, 1)
    R_diag = R_diag / n_obs_per_series
    
    # For series with no observations, use previous R value
    no_obs_mask = n_obs_per_series == 1  # Only 1 because we set minimum to 1 above
    if np.any(no_obs_mask):
        R_prev_diag = np.diag(R)
        R_diag[no_obs_mask] = R_prev_diag[no_obs_mask]
    
    # Ensure R_diag is positive (variance must be non-negative)
    # Use adaptive minimum based on data scale, not arbitrary fixed value
    # Minimum variance should be proportional to the typical scale of prediction errors
    # Use a small fraction of the mean variance as minimum
    mean_var = np.mean(R_diag[R_diag > 0]) if np.any(R_diag > 0) else 1e-4
    min_var = np.maximum(mean_var * 1e-6, 1e-8)
    R_diag = np.maximum(R_diag, min_var)
    
    # Clean any invalid values (preserve scale)
    valid_mask = np.isfinite(R_diag) & (R_diag > 0)
    if np.any(valid_mask):
        # Use median of valid values as fallback for invalid ones
        median_var = np.median(R_diag[valid_mask])
        R_diag = np.where(valid_mask, R_diag, median_var)
    else:
        # All invalid - use default
        R_diag.fill(1e-4)
    
    # Create diagonal R matrix
    R_new = np.diag(R_diag)
    
    # Final validation: Ensure Q_new is positive semi-definite
    Q_new = _clean_matrix(Q_new, 'covariance', default_nan=0.0)
    min_eigenval = config.min_eigenvalue if config else 1e-8
    warn_reg = config.warn_on_regularization if config else True
    Q_new, _ = _apply_regularization(Q_new, 'covariance', config)
    
    # Cap maximum eigenvalue to prevent explosion
    max_eigenval = config.max_eigenvalue if config else 1e6
    Q_new = _cap_max_eigenvalue(Q_new, max_eigenval=max_eigenval)
    
    # Final validation: Ensure V_0_new is positive semi-definite
    V_0_new = _clean_matrix(V_0_new, 'covariance', default_nan=0.0)
    V_0_new, _ = _apply_regularization(V_0_new, 'covariance', config)
    
    return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik


def dfm(X: np.ndarray, config: DFMConfig, threshold: Optional[float] = None, max_iter: Optional[int] = None) -> DFMResult:
    """Estimate dynamic factor model using EM algorithm.
    
    This is the main function for estimating a Dynamic Factor Model (DFM). It implements
    the complete EM algorithm workflow:
    
    1. **Initialization**: Compute initial parameter estimates via PCA and OLS
    2. **EM Iterations**: Iteratively update parameters until convergence
    3. **Final Smoothing**: Run Kalman smoother with final parameters to obtain
       smoothed factors and data
    
    The DFM models observed time series as:
    
    .. math::
        y_t = C Z_t + e_t,   e_t \\sim N(0, R)
        Z_t = A Z_{t-1} + v_t,   v_t \\sim N(0, Q)
    
    where:
    - y_t is the n x 1 vector of observed series at time t
    - Z_t is the m x 1 vector of unobserved factors
    - C is the n x m loading matrix
    - A is the m x m transition matrix
    - R and Q are covariance matrices
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (T x N), where T is time periods and N is number of series.
        Data can contain missing values (NaN), which are handled via spline interpolation.
        Missing values are allowed but excessive missing data (>50%) will trigger warnings.
    config : DFMConfig
        Unified DFM configuration object containing:
        - Model structure: Blocks (N x n_blocks), Frequency (per series), 
          Transformation (per series), factors_per_block
        - Estimation parameters: ar_lag, threshold, max_iter, nan_method, nan_k
        Typically obtained from `load_config()`.
    threshold : float, optional
        EM convergence threshold. If None, uses config.threshold (default: 1e-5).
        Convergence is declared when:
        |loglik_current - loglik_previous| / avg(|loglik_current|, |loglik_previous|) < threshold.
        Smaller values require more iterations but provide more precise convergence.
        Typical range: 1e-5 to 1e-3.
    max_iter : int, optional
        Maximum EM iterations. If None, uses config.max_iter (default: 5000).
        For testing, use smaller values like 10-20.
        
    Returns
    -------
    DFMResult
        Dataclass containing all estimation results:
        - x_sm, X_sm: Smoothed data (standardized and unstandardized)
        - Z: Smoothed factor estimates
        - C, A, Q, R: Estimated parameters
        - Mx, Wx: Standardization parameters
        - Z_0, V_0: Initial state and covariance
        - r, p: Model structure parameters
        
    Raises
    ------
    ValueError
        If inputs are invalid (dimensions, data quality, parameters).
        Also raised during EM iterations if numerical issues occur (e.g., NaN/Inf).
    TypeError
        If input types are incorrect (e.g., X is not numpy array).
        
    Notes
    -----
    - The function automatically standardizes data: x = (X - mean) / std
    - Initial conditions are computed via `init_conditions()`
    - EM iterations continue until convergence or max_iter=5000
    - Missing data is handled by the Kalman filter during estimation
    - Convergence messages and progress are printed during execution
    
    Examples
    --------
    >>> from dfm_python import load_config, load_data, dfm
    >>> import pandas as pd
    >>> # Load configuration from YAML or create DFMConfig directly
    >>> config = load_config('config.yaml')
    >>> # Load data from file
    >>> X, Time, Z = load_data('data.csv', config, sample_start=pd.Timestamp('2000-01-01'))
    >>> # Estimate DFM
    >>> Res = dfm(X, config, threshold=1e-4)
    >>> # Access results
    >>> factors = Res.Z  # (T x m) factor estimates
    >>> loadings = Res.C  # (N x m) factor loadings
    >>> smoothed_data = Res.X_sm  # (T x N) smoothed data
    >>> # Compute common factor (first factor)
    >>> common_factor = Res.Z[:, 0]
    >>> # Project factor onto a series
    >>> series_idx = 0
    >>> series_factor = Res.Z @ Res.C[series_idx, :].T
    """
    print('Estimating the dynamic factor model (DFM) ... \n')
    
    # Additional check: ensure no NaN/Inf in input data that could cause issues
    nan_mask = np.isnan(X)
    inf_mask = np.isinf(X)
    if np.any(inf_mask):
        # Replace Inf with NaN (will be handled by missing data mechanisms)
        X = np.where(inf_mask, np.nan, X)
        warnings.warn("Data contains Inf values, replaced with NaN", UserWarning)
    
    # Store model parameters
    blocks = config.Blocks
    # nQ will be computed from frequencies if needed, but we use frequencies directly
    
    # Get configurable parameters from unified DFMConfig
    p = config.ar_lag
    r = (np.array(config.factors_per_block) 
         if config.factors_per_block is not None 
         else np.ones(blocks.shape[1]))
    nan_method = config.nan_method
    nan_k = config.nan_k
    threshold = threshold if threshold is not None else config.threshold
    max_iter = max_iter if max_iter is not None else config.max_iter
    
    # Display blocks (Block Loading Structure)
    try:
        print('\n\n\n')
        print('Block Loading Structure')
        import pandas as pd
        # Create DataFrame similar to MATLAB's array2table
        df = pd.DataFrame(blocks,
                         index=[name.replace(' ', '_') for name in config.SeriesName],
                         columns=config.BlockNames if hasattr(config, 'BlockNames') and len(config.BlockNames) == blocks.shape[1] 
                                else [f'Block_{i+1}' for i in range(blocks.shape[1])])
        print(df.to_string())
        print('\n\n\n')
    except Exception as e:
        # Display error - non-critical, continue execution
        # Fallback: show basic info
        print(f'Blocks shape: {blocks.shape}')
        print('\n\n\n')
    
    T, N = X.shape
    
    # Get clock from config (defaults to 'm' for monthly)
    clock = getattr(config, 'clock', 'm')
    
    # Get aggregation structure based on clock
    agg_info = get_aggregation_structure(config, clock=clock)
    
    # Extract tent weights for use in em_step
    tent_weights_dict = agg_info.get('tent_weights', {})
    
    # Determine which series need tent kernels (slower than clock)
    # Get frequencies array for use in init_conditions and em_step
    frequencies = np.array(config.Frequency) if hasattr(config, 'Frequency') else None
    
    # Find the slowest frequency that has a tent kernel
    # This determines the R_mat and pC to use
    R_mat = None
    q = None
    pC = 1
    
    # Check structures for tent kernels (they're keyed as (slower_freq, clock))
    if agg_info['structures']:
        # Use the first available structure (typically quarterly->monthly)
        # In practice, we'll use the one with the most periods (most complex)
        max_periods = 0
        for (slower_freq, clock_freq), (R, q_vec) in agg_info['structures'].items():
            if R is not None:
                n_periods = R.shape[1]
                if n_periods > max_periods:
                    max_periods = n_periods
                    R_mat = R
                    q = q_vec
                    pC = n_periods
    
    # Prepare data with robust standardization
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0)
    
    # Handle zero/near-zero standard deviations (constant or near-constant series)
    # Set minimum standard deviation to prevent division by zero
    min_std = 1e-6
    Wx = np.maximum(Wx, min_std)
    
    # Handle NaN standard deviations (all NaN series)
    nan_std_mask = np.isnan(Wx) | np.isnan(Mx)
    if np.any(nan_std_mask):
        _logger.warning(
            f"Series with NaN mean/std detected: {np.sum(nan_std_mask)}. "
            f"Setting Wx=1.0, Mx=0.0 for these series."
        )
        Wx[nan_std_mask] = 1.0
        Mx[nan_std_mask] = 0.0
    
    # Standardize series
    xNaN = (X - Mx) / Wx
    
    # Clip extreme values if enabled in config
    clip_threshold = getattr(config, 'data_clip_threshold', 100.0)
    clip_enabled = getattr(config, 'clip_data_values', True)
    warn_on_clip = getattr(config, 'warn_on_data_clip', True)
    
    if clip_enabled:
        n_clipped_before = np.sum(np.abs(xNaN) > clip_threshold)
        xNaN = np.clip(xNaN, -clip_threshold, clip_threshold)
        if warn_on_clip and n_clipped_before > 0:
            pct_clipped = 100.0 * n_clipped_before / xNaN.size
            _logger.warning(
                f"Data value clipping applied: {n_clipped_before} values ({pct_clipped:.2f}%) "
                f"clipped beyond ±{clip_threshold} standard deviations. "
                f"This may remove important outliers. Consider investigating extreme values "
                f"or disabling clipping (clip_data_values: false) if this is frequent."
            )
    
    # Replace any remaining NaN/Inf with 0 (after clipping)
    xNaN = np.nan_to_num(xNaN, nan=0.0, posinf=clip_threshold if clip_enabled else 100, 
                        neginf=-clip_threshold if clip_enabled else -100)
    
    # Initial conditions - use configurable nan_method and nan_k
    optNaN = {'method': nan_method, 'k': nan_k}
    
    # Compute i_idio: 1 for clock frequency, 0 for slower frequencies
    # Slower frequencies use tent kernel structure instead of direct idiosyncratic components
    if frequencies is not None:
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
        i_idio = np.array([
            1 if j >= len(frequencies) or FREQUENCY_HIERARCHY.get(frequencies[j], 3) <= clock_hierarchy
            else 0
            for j in range(N)
        ])
    else:
        # Fallback: assume all series have idiosyncratic components
        i_idio = np.ones(N)
    
    nQ = N - np.sum(i_idio) if frequencies is not None else 0
    
    A, C, Q, R, Z_0, V_0 = init_conditions(
        xNaN, r, p, blocks, optNaN, R_mat, q, nQ, i_idio,
        clock=clock, tent_weights_dict=tent_weights_dict, frequencies=frequencies
    )
    
    # Verify initial conditions are valid (should already be cleaned by init_conditions)
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(C)) or np.any(~np.isfinite(Q)) or np.any(~np.isfinite(R)):
        _logger.warning("Initial conditions contain NaN/Inf - this should not happen after init_conditions()")
    
    # Initialize EM loop
    previous_loglik = -np.inf
    num_iter = 0
    LL = [-np.inf]
    converged = False
    
    # y for estimation is WITH missing data (n x T)
    y = xNaN.T  # Transpose to get n x T
    
    # Remove leading and ending NaNs for estimation
    optNaN_est = {'method': 3, 'k': nan_k}  # Use separate dict to avoid modifying optNaN
    xNaN_est, _ = _rem_nans_spline(xNaN, method=optNaN_est['method'], k=optNaN_est['k'])
    y_est = xNaN_est.T  # n x T
    
    # EM loop with likelihood monitoring
    # In proper EM, likelihood should be non-decreasing. Handle decreases gracefully.
    while num_iter < max_iter and not converged:
        C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
            y_est, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks,
            tent_weights_dict=tent_weights_dict,
            clock=clock,
            frequencies=frequencies,
            config=config
        )
        
        # Check if likelihood decreased significantly (more than numerical precision)
        # Small decreases (< 1e-3) are acceptable due to numerical precision
        # Larger decreases indicate numerical issues - use damped update
        if num_iter > 0 and loglik < previous_loglik - 1e-3:
            # Likelihood decreased - use damped update to preserve stability
            # Damping factor: use 80% of new, 20% of old (conservative)
            damping = 0.8
            
            C_prev = C.copy()
            R_prev = R.copy()
            A_prev = A.copy()
            Q_prev = Q.copy()
            Z_0_prev = Z_0.copy()
            V_0_prev = V_0.copy()
            
            # Try damped update
            C_damped = damping * C_new + (1 - damping) * C_prev
            R_damped = damping * R_new + (1 - damping) * R_prev
            A_damped = damping * A_new + (1 - damping) * A_prev
            Q_damped = damping * Q_new + (1 - damping) * Q_prev
            Z_0_damped = damping * Z_0_new + (1 - damping) * Z_0_prev
            V_0_damped = damping * V_0_new + (1 - damping) * V_0_prev
            
            # Recompute likelihood with damped parameters to verify improvement
            # This is expensive, so only do it if decrease was significant
            if loglik < previous_loglik - 0.1:  # Significant decrease
                try:
                    # Quick likelihood check with damped parameters
                    _, _, _, loglik_damped = run_kf(y_est, A_damped, C_damped, Q_damped, R_damped, Z_0_damped, V_0_damped)
                    if loglik_damped > previous_loglik:
                        # Damped update improves - use it
                        C = C_damped
                        R = R_damped
                        A = A_damped
                        Q = Q_damped
                        Z_0 = Z_0_damped
                        V_0 = V_0_damped
                        loglik = loglik_damped
                    else:
                        # Damped update still worse - keep previous parameters
                        # Don't update - parameters remain as C_prev, etc.
                        loglik = previous_loglik  # Don't decrease likelihood
                except:
                    # Likelihood recomputation failed - use damped update anyway
                    C = C_damped
                    R = R_damped
                    A = A_damped
                    Q = Q_damped
                    Z_0 = Z_0_damped
                    V_0 = V_0_damped
            else:
                # Small decrease - use damped update without recomputing likelihood
                C = C_damped
                R = R_damped
                A = A_damped
                Q = Q_damped
                Z_0 = Z_0_damped
                V_0 = V_0_damped
        else:
            # Normal update - likelihood increased or small decrease
            C = C_new
            R = R_new
            A = A_new
            Q = Q_new
            Z_0 = Z_0_new
            V_0 = V_0_new
        
        if num_iter > 2:
            converged, decreased = em_converged(loglik, previous_loglik, threshold, True)
            # If likelihood decreased significantly and we're not near convergence, 
            # it may indicate numerical issues - but we've already handled it above
        
        if (num_iter % 10 == 0) and (num_iter > 0):
            print(f'Now running the {num_iter}th iteration of max {max_iter}')
            print('  Loglik   (% Change)')
            pct_change = 100 * ((loglik - previous_loglik) / abs(previous_loglik)) if previous_loglik != 0 else 0
            print(f'{loglik:.6f}   ({pct_change:6.2f}%)')
        
        LL.append(loglik)
        previous_loglik = loglik
        num_iter += 1
    
    if num_iter < max_iter:
        print(f'Successful: Convergence at {num_iter} iterations')
    else:
        print('Stopped because maximum iterations reached')
    
    # Final run of Kalman filter and smoother
    # MATLAB: Zsmooth = runKF(y, A, C, Q, R, Z_0, V_0)';
    # run_kf returns m x (T+1), transpose to (T+1) x m
    zsmooth, _, _, _ = run_kf(y, A, C, Q, R, Z_0, V_0)
    Zsmooth = zsmooth.T  # m x (T+1) -> (T+1) x m
    
    # Get smoothed X (standardized)
    # MATLAB: x_sm = Zsmooth(2:end,:) * C';
    # Zsmooth is (T+1) x m, C is N x m, C' is m x N
    # Zsmooth(2:end,:) selects rows 2 to end (skip initial state), gives T x m
    # Then Zsmooth(2:end,:) * C' gives T x m * m x N = T x N
    x_sm = Zsmooth[1:, :] @ C.T  # T x N (standardized smoothed data)
    
    # Unstandardized smoothed data
    # MATLAB: X_sm = repmat(Wx,T,1) .* x_sm + repmat(Mx,T,1)
    # Handle NaN in Wx and Mx (can occur if series had all NaN values)
    Wx_clean = np.where(np.isnan(Wx), 1.0, Wx)
    Mx_clean = np.where(np.isnan(Mx), 0.0, Mx)
    # Element-wise multiplication: (T x N) .* (T x N) + (T x N)
    # Use broadcasting: x_sm is T x N, Wx_clean is (N,), so x_sm * Wx_clean broadcasts correctly
    X_sm = x_sm * Wx_clean + Mx_clean  # Broadcasting: T x N * (N,) + (N,) = T x N
    
    # Calculate RMSE: compare smoothed data to original data (where available)
    # Only use non-missing observations for comparison
    rmse_overall, rmse_per_series = calculate_rmse(X, X_sm, mask=None)
    
    # Calculate standardized RMSE for better comparison across series
    # Standardize both actual and predicted using the same Wx and Mx
    X_std = (X - Mx_clean) / np.where(Wx_clean != 0, Wx_clean, 1.0)
    x_sm_std = x_sm  # Already standardized
    rmse_std_overall, rmse_std_per_series = calculate_rmse(X_std, x_sm_std, mask=None)
    
    # Create result structure
    Res = DFMResult(
        x_sm=x_sm,
        X_sm=X_sm,
        Z=Zsmooth[1:, :],  # T x m (skip initial state)
        C=C,
        R=R,
        A=A,
        Q=Q,
        Mx=Mx,
        Wx=Wx,
        Z_0=Z_0,
        V_0=V_0,
        r=r,
        p=p,
        converged=converged,
        num_iter=num_iter,
        loglik=loglik,
        rmse=rmse_overall,
        rmse_per_series=rmse_per_series,
        rmse_std=rmse_std_overall,
        rmse_std_per_series=rmse_std_per_series
    )
    
    # Display output tables (optional - can be disabled for cleaner output)
    try:
        _display_dfm_tables(Res, config, nQ)
    except Exception:
        pass  # Skip if display fails
    
    return Res


def _display_dfm_tables(Res: DFMResult, config, nQ: int) -> None:
    """Display DFM estimation output tables."""
    # nQ: number of slower-frequency series (e.g., quarterly when clock is monthly)
    # n_same_freq: number of same-frequency series (e.g., monthly when clock is monthly)
    n_same_freq = len(config.SeriesID) - nQ
    nLags = max(Res.p, 5)  # Minimum 5 lags for mixed-frequency aggregation
    nFactors = int(np.sum(Res.r))
    
    try:
        print('\n\n\n')
        
        # Factor Loadings for Same-Frequency Series
        print('Factor Loadings for Same-Frequency Series')
        # Only select lag(0) terms - every 5th column starting from 0
        C_same_freq = Res.C[:n_same_freq, ::5][:, :nFactors]
        try:
            import pandas as pd
            df = pd.DataFrame(C_same_freq, 
                            index=[name.replace(' ', '_') for name in config.SeriesName[:n_same_freq]],
                            columns=config.BlockNames[:nFactors] if len(config.BlockNames) >= nFactors else [f'Block{i+1}' for i in range(nFactors)])
            print(df.to_string())
        except (ValueError, IndexError, AttributeError) as e:
            # DataFrame creation or display failed - show shape instead
            # This can happen if indices/columns don't match or pandas unavailable
            print(f'Same-frequency loadings shape: {C_same_freq.shape}')
        print('\n\n\n')
        
        # Slower-Frequency Loadings Sample (First Factor)
        print('Slower-Frequency Loadings Sample (First Factor)')
        # Select only slower-frequency series and first 5 columns (lag 0-4)
        C_slower_freq = Res.C[-nQ:, :5]
        try:
            n_lags = min(5, C_slower_freq.shape[1])
            lag_cols = [f'factor1_lag{i}' for i in range(n_lags)]
            df = pd.DataFrame(C_slower_freq,
                            index=[name.replace(' ', '_') for name in config.SeriesName[-nQ:]],
                            columns=lag_cols)
            print(df.to_string())
        except (ValueError, IndexError, AttributeError) as e:
            # DataFrame creation or display failed - show shape instead
            print(f'Slower-frequency loadings shape: {C_slower_freq.shape}')
        print('\n\n\n')
        
        # Autoregressive Coefficients on Factors
        print('Autoregressive Coefficients on Factors')
        A_terms = np.diag(Res.A)
        Q_terms = np.diag(Res.Q)
        # Only select lag(0) terms - every 5th element
        A_terms_factors = A_terms[::5][:nFactors]
        Q_terms_factors = Q_terms[::5][:nFactors]
        try:
            df = pd.DataFrame({
                'AR_Coefficient': A_terms_factors,
                'Variance_Residual': Q_terms_factors
            }, index=[name.replace(' ', '_') for name in config.BlockNames[:nFactors]])
            print(df.to_string())
        except (ValueError, IndexError, AttributeError) as e:
            # DataFrame creation or display failed - show values instead
            print(f'Factor AR coefficients: {A_terms_factors}')
        print('\n\n\n')
        
        # Autoregressive Coefficients on Idiosyncratic Component
        print('Autoregressive Coefficients on Idiosyncratic Component')
        rp1 = nFactors * 5  # Start of idiosyncratic component (0-indexed)
        # Same-frequency idiosyncratic: indices rp1 to rp1+n_same_freq-1 (inclusive)
        same_freq_idx = np.arange(rp1, rp1 + n_same_freq)
        # Slower-frequency idiosyncratic: every 5th index starting from rp1+n_same_freq
        slower_freq_idx = np.arange(rp1 + n_same_freq, len(A_terms), 5)
        combined_idx = np.concatenate([same_freq_idx, slower_freq_idx])
        
        # Ensure indices are within bounds
        combined_idx = combined_idx[combined_idx < len(A_terms)]
        
        A_idio = A_terms[combined_idx]
        Q_idio = Q_terms[combined_idx]
        try:
            # Map indices back to series names
            # Same-frequency series: direct mapping (0 to n_same_freq-1)
            # Slower-frequency series: need to map to slower-frequency series index
            series_indices = []
            for idx in combined_idx:
                if idx < idio_start_display + n_same_freq:
                    # Same-frequency idiosyncratic - map to same-frequency series
                    series_idx = idx - idio_start_display
                    if series_idx < n_same_freq:
                        series_indices.append(series_idx)
                    else:
                        series_indices.append(None)
                else:
                    # Slower-frequency idiosyncratic - map to slower-frequency series
                    # (idx - (idio_start_display + n_same_freq)) / 5 gives which slower-frequency series
                    slower_idx = (idx - (idio_start_display + n_same_freq)) // 5
                    if slower_idx < nQ:
                        series_indices.append(n_same_freq + slower_idx)
                    else:
                        series_indices.append(None)
            
            # Filter out None and get valid series names
            valid_indices = [i for i in series_indices if i is not None and i < len(config.SeriesName)]
            series_names_list = [config.SeriesName[i].replace(' ', '_') for i in valid_indices[:len(A_idio)]]
            
            df = pd.DataFrame({
                'AR_Coefficient': A_idio[:len(series_names_list)],
                'Variance_Residual': Q_idio[:len(series_names_list)]
            }, index=series_names_list)
            print(df.to_string())
        except (ValueError, IndexError, AttributeError) as e:
            # DataFrame creation or display failed - show values instead
            print(f'Idiosyncratic AR coefficients (first 10): {A_idio[:min(10, len(A_idio))]}')
            if len(A_idio) > 10:
                print(f'... (total {len(A_idio)} coefficients)')
        print('\n\n\n')
        
        # Model Fit Statistics (RMSE)
        if Res.rmse is not None and not np.isnan(Res.rmse):
            print('Model Fit Statistics')
            print(f'  Overall RMSE (original scale): {Res.rmse:.6f}')
            if Res.rmse_std is not None and not np.isnan(Res.rmse_std):
                print(f'  Overall RMSE (standardized scale): {Res.rmse_std:.6f}')
            if Res.rmse_per_series is not None and len(Res.rmse_per_series) > 0:
                print('\n  RMSE per Series (Original Scale):')
                try:
                    series_names = config.SeriesName if hasattr(config, 'SeriesName') else [f"Series_{i}" for i in range(len(Res.rmse_per_series))]
                    for i, (name, rmse_val) in enumerate(zip(series_names, Res.rmse_per_series)):
                        if not np.isnan(rmse_val):
                            # Calculate RMSE as percentage of mean if available
                            mean_val = Res.Mx[i] if i < len(Res.Mx) else np.nan
                            if not np.isnan(mean_val) and abs(mean_val) > 1e-6:
                                pct = 100.0 * rmse_val / abs(mean_val)
                                print(f'    {name:40s}: {rmse_val:.6f} ({pct:.2f}% of mean)')
                            else:
                                print(f'    {name:40s}: {rmse_val:.6f}')
                except Exception:
                    # Fallback if series names not available
                    for i, rmse_val in enumerate(Res.rmse_per_series):
                        if not np.isnan(rmse_val):
                            print(f'    Series {i:3d}: {rmse_val:.6f}')
            
            # Show standardized RMSE per series
            if Res.rmse_std_per_series is not None and len(Res.rmse_std_per_series) > 0:
                print('\n  RMSE per Series (Standardized Scale):')
                try:
                    series_names = config.SeriesName if hasattr(config, 'SeriesName') else [f"Series_{i}" for i in range(len(Res.rmse_std_per_series))]
                    for i, (name, rmse_std_val) in enumerate(zip(series_names, Res.rmse_std_per_series)):
                        if not np.isnan(rmse_std_val):
                            print(f'    {name:40s}: {rmse_std_val:.6f} std dev')
                except Exception:
                    # Fallback if series names not available
                    for i, rmse_std_val in enumerate(Res.rmse_std_per_series):
                        if not np.isnan(rmse_std_val):
                            print(f'    Series {i:3d}: {rmse_std_val:.6f} std dev')
            
            # Diagnostic warnings for high RMSE
            if Res.rmse_per_series is not None and Res.Mx is not None:
                print('\n  Diagnostic Warnings:')
                try:
                    series_names = config.SeriesName if hasattr(config, 'SeriesName') else [f"Series_{i}" for i in range(len(Res.rmse_per_series))]
                    warnings_count = 0
                    for i, (name, rmse_val) in enumerate(zip(series_names, Res.rmse_per_series)):
                        if not np.isnan(rmse_val) and i < len(Res.Mx):
                            mean_val = Res.Mx[i]
                            std_val = Res.Wx[i] if i < len(Res.Wx) else np.nan
                            if not np.isnan(mean_val) and abs(mean_val) > 1e-6:
                                pct_of_mean = 100.0 * rmse_val / abs(mean_val)
                                # Warn if RMSE > 50% of mean or > 10 standard deviations
                                if pct_of_mean > 50.0 or (not np.isnan(std_val) and rmse_val > 10.0 * std_val):
                                    warnings_count += 1
                                    if warnings_count <= 5:  # Limit to first 5 warnings
                                        print(f'    ⚠ {name:40s}: RMSE is {pct_of_mean:.1f}% of mean')
                                        if not np.isnan(std_val):
                                            print(f'      (RMSE={rmse_val:.2e}, Mean={mean_val:.2e}, Std={std_val:.2e})')
                    if warnings_count > 5:
                        print(f'    ... and {warnings_count - 5} more series with high RMSE')
                except Exception:
                    pass
            print('\n\n\n')
        
    except Exception as e:
        print(f'Error displaying tables: {e}')
        pass


def diagnose_series(Res: DFMResult, config: DFMConfig, series_name: Optional[str] = None, 
                    series_idx: Optional[int] = None) -> Dict[str, Any]:
    """Diagnose model fit issues for a specific series.
    
    This function provides detailed diagnostics to help identify why a series
    may have high RMSE, including factor loadings, standardization values,
    and reconstruction errors.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results
    config : DFMConfig
        DFM configuration
    series_name : str, optional
        Name of the series to diagnose. If provided, series_idx is ignored.
    series_idx : int, optional
        Index of the series to diagnose (0-based). Used if series_name is not provided.
        
    Returns
    -------
    dict
        Dictionary containing diagnostic information:
        - 'series_name': Name of the series
        - 'series_idx': Index of the series
        - 'rmse_original': RMSE on original scale
        - 'rmse_standardized': RMSE on standardized scale
        - 'mean': Series mean (Mx)
        - 'std': Series standard deviation (Wx)
        - 'rmse_pct_of_mean': RMSE as percentage of mean
        - 'rmse_in_std_devs': RMSE in standard deviations
        - 'factor_loadings': Factor loadings for this series
        - 'max_loading': Maximum absolute loading
        - 'loading_sum_sq': Sum of squared loadings
        - 'reconstruction_error_mean': Mean reconstruction error
        - 'reconstruction_error_std': Std of reconstruction error
        
    Examples
    --------
    >>> Res = dfm(X, config)
    >>> diag = diagnose_series(Res, config, series_name="Series_A")
    >>> print(f"Standardized RMSE: {diag['rmse_standardized']:.4f}")
    """
    # Find series index
    if series_name is not None:
        try:
            series_names = config.SeriesName if hasattr(config, 'SeriesName') else []
            if series_name in series_names:
                series_idx = series_names.index(series_name)
            else:
                # Try case-insensitive match
                series_idx = next((i for i, name in enumerate(series_names) 
                                 if name.lower() == series_name.lower()), None)
                if series_idx is None:
                    raise ValueError(f"Series '{series_name}' not found in configuration")
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Cannot find series '{series_name}': {e}")
    
    if series_idx is None:
        raise ValueError("Must provide either series_name or series_idx")
    
    if series_idx < 0 or series_idx >= Res.C.shape[0]:
        raise ValueError(f"Series index {series_idx} out of range [0, {Res.C.shape[0]})")
    
    # Get series name
    try:
        series_names = config.SeriesName if hasattr(config, 'SeriesName') else []
        name = series_names[series_idx] if series_idx < len(series_names) else f"Series_{series_idx}"
    except (AttributeError, IndexError):
        name = f"Series_{series_idx}"
    
    # Get RMSE values
    rmse_original = None
    rmse_standardized = None
    if Res.rmse_per_series is not None and series_idx < len(Res.rmse_per_series):
        rmse_original = Res.rmse_per_series[series_idx]
    if Res.rmse_std_per_series is not None and series_idx < len(Res.rmse_std_per_series):
        rmse_standardized = Res.rmse_std_per_series[series_idx]
    
    # Get standardization values
    mean_val = Res.Mx[series_idx] if series_idx < len(Res.Mx) else np.nan
    std_val = Res.Wx[series_idx] if series_idx < len(Res.Wx) else np.nan
    
    # Calculate RMSE metrics
    rmse_pct_of_mean = None
    rmse_in_std_devs = None
    if not np.isnan(rmse_original) and not np.isnan(mean_val) and abs(mean_val) > 1e-6:
        rmse_pct_of_mean = 100.0 * rmse_original / abs(mean_val)
    if not np.isnan(rmse_original) and not np.isnan(std_val) and std_val > 1e-6:
        rmse_in_std_devs = rmse_original / std_val
    
    # Get factor loadings
    factor_loadings = Res.C[series_idx, :] if series_idx < Res.C.shape[0] else np.array([])
    max_loading = np.max(np.abs(factor_loadings)) if len(factor_loadings) > 0 else np.nan
    loading_sum_sq = np.sum(factor_loadings ** 2) if len(factor_loadings) > 0 else np.nan
    
    # Calculate reconstruction errors (if we have original data)
    reconstruction_error_mean = None
    reconstruction_error_std = None
    # This would require passing X to the function or storing it in Res
    
    return {
        'series_name': name,
        'series_idx': series_idx,
        'rmse_original': rmse_original,
        'rmse_standardized': rmse_standardized,
        'mean': mean_val,
        'std': std_val,
        'rmse_pct_of_mean': rmse_pct_of_mean,
        'rmse_in_std_devs': rmse_in_std_devs,
        'factor_loadings': factor_loadings,
        'max_loading': max_loading,
        'loading_sum_sq': loading_sum_sq,
        'reconstruction_error_mean': reconstruction_error_mean,
        'reconstruction_error_std': reconstruction_error_std
    }


def print_series_diagnosis(Res: DFMResult, config: DFMConfig, 
                          series_name: Optional[str] = None, 
                          series_idx: Optional[int] = None) -> None:
    """Print a formatted diagnosis report for a specific series.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results
    config : DFMConfig
        DFM configuration
    series_name : str, optional
        Name of the series to diagnose
    series_idx : int, optional
        Index of the series to diagnose
        
    Examples
    --------
    >>> Res = dfm(X, config)
    >>> print_series_diagnosis(Res, config, series_name="Series_A")
    """
    diag = diagnose_series(Res, config, series_name=series_name, series_idx=series_idx)
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT: {diag['series_name']}")
    print(f"{'='*70}\n")
    
    print("RMSE Statistics:")
    print(f"  Original scale:     {diag['rmse_original']:.6e}" if diag['rmse_original'] is not None else "  Original scale:     N/A")
    print(f"  Standardized scale: {diag['rmse_standardized']:.6f} std dev" if diag['rmse_standardized'] is not None else "  Standardized scale: N/A")
    if diag['rmse_pct_of_mean'] is not None:
        print(f"  As % of mean:       {diag['rmse_pct_of_mean']:.2f}%")
    if diag['rmse_in_std_devs'] is not None:
        print(f"  In std deviations: {diag['rmse_in_std_devs']:.2f}x")
    
    print("\nStandardization Values:")
    print(f"  Mean:  {diag['mean']:.6e}" if not np.isnan(diag['mean']) else "  Mean:  N/A")
    print(f"  Std:   {diag['std']:.6e}" if not np.isnan(diag['std']) else "  Std:   N/A")
    
    print("\nFactor Loadings:")
    if len(diag['factor_loadings']) > 0:
        print(f"  Number of loadings: {len(diag['factor_loadings'])}")
        print(f"  Max absolute:       {diag['max_loading']:.6f}" if not np.isnan(diag['max_loading']) else "  Max absolute:       N/A")
        print(f"  Sum of squares:     {diag['loading_sum_sq']:.6f}" if not np.isnan(diag['loading_sum_sq']) else "  Sum of squares:     N/A")
        # Show top 5 loadings
        abs_loadings = np.abs(diag['factor_loadings'])
        top_indices = np.argsort(abs_loadings)[-5:][::-1]
        print(f"  Top 5 loadings:")
        for idx in top_indices:
            print(f"    Factor {idx:3d}: {diag['factor_loadings'][idx]:8.4f}")
    else:
        print("  No loadings available")
    
    print(f"\n{'='*70}\n")


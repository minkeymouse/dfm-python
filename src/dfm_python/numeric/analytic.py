"""Analytical computation functions for matrices and arrays.

This module provides functions for analytical computations including
matrix operations, division operations, covariance/variance estimation,
forecast metrics, and convergence checking.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any, Union
import torch
from torch import Tensor

from ..logger import get_logger
from ..config.constants import (
    MIN_FACTOR_VARIANCE,
)

_logger = get_logger(__name__)

# Numerical stability constants
MIN_VARIANCE_COVARIANCE = MIN_FACTOR_VARIANCE
DEFAULT_VARIANCE_FALLBACK = 1.0


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = 0.0
) -> np.ndarray:
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
    min_eigenval: float = 1e-8,
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
    min_eigenval : float, default 1e-8
        Minimum eigenvalue to enforce for positive definiteness
    fallback_to_identity : bool, default True
        If True, fall back to identity matrix on failure
        
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N)
    """
    if data.size == 0:
        if fallback_to_identity:
            return np.eye(1) if data.ndim == 1 else np.eye(data.shape[1] if rowvar else data.shape[0])
        raise ValueError("Cannot compute covariance: data is empty")
    
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
        
        from .stability import ensure_positive_definite
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
        return 0.0
    
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
    # Mask for non-missing values
    mask = ~np.isnan(y_actual)
    
    # Compute MSE on non-missing values
    y_prev_valid = y_prev[mask]
    y_now_valid = y_now[mask]
    y_actual_valid = y_actual[mask]
    
    loss_prev = float(np.mean((y_actual_valid - y_prev_valid) ** 2))
    loss_now = float(np.mean((y_actual_valid - y_now_valid) ** 2))
    
    # Relative change
    from ..config.constants import MIN_FACTOR_VARIANCE
    if loss_prev < MIN_FACTOR_VARIANCE:
        delta = float(abs(loss_now - loss_prev))
    else:
        delta = float(abs(loss_now - loss_prev) / loss_prev)
    
    return delta, loss_now


def safe_matrix_power(
    matrix: Union[np.ndarray, Tensor],
    power: int,
    max_power: int = 1000,
    check_stability: bool = True
) -> Union[np.ndarray, Tensor]:
    """Safely compute matrix power with stability checks.
    
    This function computes matrix powers with numerical stability checks,
    useful for IRF computation where matrix powers are computed for large
    horizons. Supports both NumPy arrays and PyTorch tensors.
    
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
    from ..utils.errors import NumericalError
    
    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}")
    if power > max_power:
        raise ValueError(
            f"power {power} exceeds maximum {max_power}. "
            f"This may indicate a configuration error."
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
            identity = np.eye(n, dtype=matrix.dtype)
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
    'safe_divide',
    'compute_var_safe',
    'compute_cov_safe',
    'mse_missing_numpy',
    'convergence_checker',
    'safe_matrix_power',
    'extract_matrix_block',
    'compute_forecast_metrics',
]


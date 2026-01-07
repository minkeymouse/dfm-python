"""Statistical utilities for debugging and diagnostics.

This module provides statistical computation and diagnostic utilities,
primarily for debugging model behavior (e.g., variance collapse detection).
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from ..config.types import to_numpy
from ..config.constants import (
    DEFAULT_VARIANCE_COLLAPSE_THRESHOLD,
    DEFAULT_FACTOR_COLLAPSE_THRESHOLD,
    DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD,
    DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN,
    DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX,
    DEFAULT_SCALE_RATIO_MIN,
    DEFAULT_STANDARDIZATION_MEAN_THRESHOLD,
    DEFAULT_STANDARDIZATION_STD_MIN,
    DEFAULT_STANDARDIZATION_STD_MAX,
    DEFAULT_TARGET_PREDICTION_STD,
)


# ============================================================================
# Basic Statistics Functions
# ============================================================================

def compute_variance_mean(variance_array: Optional[np.ndarray]) -> Optional[float]:
    """Compute mean of variance array for logging.
    
    Consolidates duplicate pattern: float(np.mean(variance_array)) used for
    prediction_std and factor_std logging.
    
    Parameters
    ----------
    variance_array : np.ndarray, optional
        Variance array (prediction_std or factor_std)
        
    Returns
    -------
    float, optional
        Mean of variance array, or None if array is None
    """
    if variance_array is None:
        return None
    return float(np.mean(variance_array))


def compute_array_stats(array: np.ndarray, use_nan: bool = False) -> Tuple[float, float, float, float]:
    """Compute statistics (mean, std, min, max) for numpy array.
    
    Parameters
    ----------
    array : np.ndarray
        Input array to compute statistics for
    use_nan : bool, default False
        Whether to use NaN-aware functions (nanmean, nanstd, etc.)
        
    Returns
    -------
    Tuple[float, float, float, float]
        (mean, std, min, max) statistics
    """
    if use_nan:
        return (
            float(np.nanmean(array)),
            float(np.nanstd(array)),
            float(np.nanmin(array)),
            float(np.nanmax(array))
        )
    else:
        return (
            float(np.mean(array)),
            float(np.std(array)),
            float(np.min(array)),
            float(np.max(array))
        )


def compute_tensor_stats(tensor: Tensor, dim: int = 0, unbiased: bool = False) -> Tuple[Tensor, Tensor]:
    """Compute mean and std of tensor along specified dimension.
    
    Consolidates duplicate pattern: tensor.mean(dim=0) and tensor.std(dim=0) used
    for predictions_full_tensor and factors_tensor statistics.
    
    **CRITICAL**: Uses `unbiased=False` by default to match TensorFlow's `tf.reduce_std` behavior.
    TensorFlow's `tf.reduce_std` uses population std (unbiased=False) by default, while
    PyTorch's `tensor.std()` defaults to sample std (unbiased=True, Bessel's correction).
    This mismatch can cause systematic differences in prediction_std computation, especially
    with small MC sample counts (n_mc_samples=10), potentially explaining variance collapse.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape (n_samples, ...) where first dimension is samples dimension
    dim : int, default 0
        Dimension along which to compute statistics
    unbiased : bool, default False
        Whether to use unbiased std (Bessel's correction). Default False to match TensorFlow.
            
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (mean, std) computed along specified dimension, where std uses population std (unbiased=False)
        by default to match TensorFlow's tf.reduce_std behavior
    """
    return tensor.mean(dim=dim), tensor.std(dim=dim, unbiased=unbiased)


def average_3d_array(array: np.ndarray, axis: int = 0) -> np.ndarray:
    """Average 3D array along specified axis, or return as-is if not 3D.
    
    Common pattern for averaging MC samples from factors or predictions.
    If array is 3D, averages along the specified axis. Otherwise returns as-is.
    
    Parameters
    ----------
    array : np.ndarray
        Input array (2D or 3D)
    axis : int, default 0
        Axis along which to average if array is 3D
        
    Returns
    -------
    np.ndarray
        Averaged array (2D) if input was 3D, otherwise original array
    """
    if array.ndim == 3:
        return np.mean(array, axis=axis)
    return array


# ============================================================================
# Debugging and Diagnostic Functions
# ============================================================================

def extract_batchnorm_statistics(
    encoder: nn.Module,
    decoder: nn.Module,
    compute_variance_mean_fn: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Extract BatchNorm statistics (running_mean, running_var) from encoder/decoder.
    
    Consolidates BatchNorm statistics inspection pattern used in variance collapse diagnostics.
    
    Parameters
    ----------
    encoder : nn.Module
        Encoder module to extract BatchNorm statistics from
    decoder : nn.Module
        Decoder module to extract BatchNorm statistics from
    compute_variance_mean_fn : callable, optional
        Function to compute variance mean. If None, uses compute_variance_mean from this module.
        
    Returns
    -------
    list[dict]
        List of BatchNorm statistics dictionaries: {
            'module': str ('encoder' or 'decoder'),
            'layer': str (layer name),
            'running_mean_abs': float,
            'running_var_mean': float
        }
    """
    if compute_variance_mean_fn is None:
        compute_variance_mean_fn = compute_variance_mean
    
    batchnorm_stats = []
    for module_name, module in [('encoder', encoder), ('decoder', decoder)]:
        for name, submodule in module.named_modules():
            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d)):
                running_mean = to_numpy(submodule.running_mean) if submodule.running_mean.numel() > 0 else None
                running_var = to_numpy(submodule.running_var) if submodule.running_var.numel() > 0 else None
                if running_mean is not None and running_var is not None:
                    mean_abs = float(np.mean(np.abs(running_mean)))
                    var_mean = compute_variance_mean_fn(running_var)
                    batchnorm_stats.append({
                        'module': module_name,
                        'layer': name,
                        'running_mean_abs': mean_abs,
                        'running_var_mean': var_mean
                    })
    return batchnorm_stats


def diagnose_variance_collapse(
    prediction_std: np.ndarray,
    prediction_mean: np.ndarray,
    factors_mean: np.ndarray,
    y_actual: np.ndarray,
    target_scaler: Optional[Any],
    encoder: nn.Module,
    decoder: nn.Module,
    factors_std: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Diagnose root cause of variance collapse.
    
    Detects when prediction std is too low (std ~DEFAULT_VARIANCE_COLLAPSE_STD vs target ~DEFAULT_TARGET_PREDICTION_STD).
    Uses constants DEFAULT_VARIANCE_COLLAPSE_STD and DEFAULT_TARGET_PREDICTION_STD for target values in diagnostics.
    
    Provides actionable diagnostics to identify why prediction variance is too low.
    Checks: (1) decoder output scale vs data scale, (2) BatchNorm statistics,
    (3) factor magnitudes, (4) per-time-step variance patterns.
    
    Parameters
    ----------
    prediction_std : np.ndarray
        Prediction std across MC samples (shape: (T, N))
    prediction_mean : np.ndarray
        Prediction mean across MC samples (shape: (T, N))
    factors_mean : np.ndarray
        Factor mean across MC samples (shape: (T, m))
    y_actual : np.ndarray
        Actual target values (shape: (T, N))
    target_scaler : object, optional
        Target scaler (e.g., StandardScaler) for checking standardization
    encoder : nn.Module
        Encoder module for BatchNorm statistics extraction
    decoder : nn.Module
        Decoder module for BatchNorm statistics extraction
    factors_std : np.ndarray, optional
        Factor std across MC samples (shape: (T, m))
        
    Returns
    -------
    dict
        Diagnostic information: {
            'prediction_std_mean': float,
            'data_std_mean': float,
            'scale_ratio': float,
            'factors_mean_abs': float,
            'factors_std_mean': float (if factors_std provided),
            'variance_collapse_detected': bool,
            'warnings': list[str]
        }
    """
    from sklearn.preprocessing import StandardScaler
    
    diagnostics = {
        'prediction_std_mean': None,
        'variance_collapse_detected': False,
        'warnings': []
    }
    
    # Validate prediction_std before computing mean to avoid numpy warnings
    if not isinstance(prediction_std, np.ndarray):
        diagnostics['warnings'].append(f"Invalid prediction_std type: {type(prediction_std).__name__}, expected np.ndarray")
        return diagnostics
    
    if prediction_std.ndim == 0 or prediction_std.size == 0:
        diagnostics['warnings'].append(f"Invalid prediction_std shape: {prediction_std.shape}, expected 2D array (T, N)")
        return diagnostics
    
    # Handle 1D arrays (single time step or single series)
    if prediction_std.ndim == 1:
        diagnostics['warnings'].append(f"1D prediction_std array (shape: {prediction_std.shape}), per-time-step analysis skipped")
        return diagnostics
    
    # Validate 2D array shape
    if prediction_std.ndim != 2:
        diagnostics['warnings'].append(f"Invalid prediction_std dimensions: {prediction_std.ndim}, expected 2D array (T, N)")
        return diagnostics
    
    diagnostics['prediction_std_mean'] = compute_variance_mean(prediction_std)
    
    # Check 1: Decoder output scale vs data scale
    data_mean, data_std, _, _ = compute_array_stats(y_actual)
    diagnostics['data_std_mean'] = data_std
    diagnostics['data_mean_abs'] = abs(data_mean)
    
    # Check if data is standardized (mean ≈ 0, std ≈ DEFAULT_TARGET_PREDICTION_STD)
    is_standardized = (
        isinstance(target_scaler, StandardScaler) and
        abs(data_mean) < DEFAULT_STANDARDIZATION_MEAN_THRESHOLD and
        DEFAULT_STANDARDIZATION_STD_MIN <= data_std <= DEFAULT_STANDARDIZATION_STD_MAX
    )
    diagnostics['is_standardized'] = is_standardized
    
    if not is_standardized:
        diagnostics['warnings'].append(
            f"Data standardization assumption not verified: data_mean={data_mean:.6f}, data_std={data_std:.6f} "
            f"(expected: mean≈0, std≈{DEFAULT_TARGET_PREDICTION_STD} for StandardScaler). Diagnostics may be inaccurate."
        )
        target_std = data_std
    else:
        target_std = DEFAULT_TARGET_PREDICTION_STD
    
    diagnostics['scale_ratio'] = diagnostics['prediction_std_mean'] / data_std if data_std > 0 else float('inf')
    
    if diagnostics['prediction_std_mean'] < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD:
        diagnostics['variance_collapse_detected'] = True
        diagnostics['warnings'].append(
            f"Variance collapse detected: prediction_std={diagnostics['prediction_std_mean']:.6f} << target ~{target_std:.6f}"
        )
    
    if diagnostics['scale_ratio'] < DEFAULT_SCALE_RATIO_MIN:
        diagnostics['warnings'].append(f"Scale mismatch: prediction_std/data_std={diagnostics['scale_ratio']:.6f} << target ~{target_std:.6f}")
    
    factors_mean_abs = float(np.mean(np.abs(factors_mean)))
    diagnostics['factors_mean_abs'] = factors_mean_abs
    if factors_mean_abs < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
        diagnostics['warnings'].append(f"Factor collapse: |factors_mean|={factors_mean_abs:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
    
    if factors_std is not None:
        factors_std_mean = compute_variance_mean(factors_std)
        diagnostics['factors_std_mean'] = factors_std_mean
        if factors_std_mean < DEFAULT_FACTOR_COLLAPSE_THRESHOLD:
            diagnostics['warnings'].append(f"Factor variance collapse: factors_std={factors_std_mean:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}")
    
    batchnorm_stats = extract_batchnorm_statistics(encoder, decoder)
    for stat in batchnorm_stats:
        if stat['running_var_mean'] < DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD:
            diagnostics['warnings'].append(
                f"BatchNorm signal suppression in {stat['module']}.{stat['layer']}: running_var={stat['running_var_mean']:.6f} << expected ~{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN}-{DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX}"
            )
    diagnostics['batchnorm_stats'] = batchnorm_stats
    
    if len(prediction_std) > 1:
        per_timestep_std = np.mean(prediction_std, axis=1)
        timestep_collapse_count = int(np.sum(per_timestep_std < DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD))
        timestep_collapse_ratio = timestep_collapse_count / len(per_timestep_std)
        diagnostics['timestep_collapse_count'] = timestep_collapse_count
        diagnostics['timestep_collapse_ratio'] = timestep_collapse_ratio
        if timestep_collapse_ratio > DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD:
            diagnostics['warnings'].append(
                f"Localized variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD} "
                f"(ratio={timestep_collapse_ratio:.2%})"
            )
        elif timestep_collapse_count > 0:
            diagnostics['warnings'].append(
                f"Partial variance collapse: {timestep_collapse_count}/{len(per_timestep_std)} time steps have std < {DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD}"
            )
    
    return diagnostics


__all__ = [
    # Basic statistics functions
    'compute_variance_mean',
    'compute_array_stats',
    'compute_tensor_stats',
    'average_3d_array',
    # Debugging and diagnostic functions
    'extract_batchnorm_statistics',
    'diagnose_variance_collapse',
]


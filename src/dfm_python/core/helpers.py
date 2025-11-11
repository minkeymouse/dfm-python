"""Helper functions for DFM estimation."""
from typing import Optional, Tuple, Any
import numpy as np

def safe_get_method(config, method_name, default=None):
    if config is None:
        return default
    method = getattr(config, method_name, None)
    if method is not None and callable(method):
        return method()
    return default

def safe_get_attr(config, attr_name, default=None):
    if config is None:
        return default
    return getattr(config, attr_name, default)

def resolve_param(override, default):
    return override if override is not None else default

def safe_mean_std(X, clip_data_values=False, data_clip_threshold=10.0):
    """Compute mean and std with optional clipping."""
    if clip_data_values:
        X = np.clip(X, -data_clip_threshold, data_clip_threshold)
    Mx = np.nanmean(X, axis=0)
    Wx = np.nanstd(X, axis=0, ddof=0)
    Wx = np.where(Wx < 1e-8, 1.0, Wx)
    return Mx, Wx

def standardize_data(X, clip_data_values=False, data_clip_threshold=10.0):
    """Standardize data: x = (X - mean) / std."""
    Mx, Wx = safe_mean_std(X, clip_data_values, data_clip_threshold)
    x = (X - Mx) / Wx
    return x, Mx, Wx

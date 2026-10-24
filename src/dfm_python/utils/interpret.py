"""Helpers shared by the models' interpretation API (``loadings_``, ``explained_variance_``, ``decompose``)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

__all__ = ["affine_scaler_params", "explained_variance"]


def affine_scaler_params(scaler: Any, n: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(offset, scale)`` with ``data = offset + scale * scaled`` per column, or None for a non-affine scaler.

    None (no scaler) and the fitted ``StandardScaler``, ``RobustScaler``,
    ``MinMaxScaler`` and ``MaxAbsScaler`` (exact classes: a subclass may
    change ``inverse_transform``) are affine; ``n`` is the number of columns.
    """
    zeros, ones = np.zeros(n), np.ones(n)
    if scaler is None:
        return zeros, ones
    kind = type(scaler)
    if kind is StandardScaler:
        return (scaler.mean_ if scaler.with_mean else zeros), (scaler.scale_ if scaler.with_std else ones)
    if kind is RobustScaler:
        return (scaler.center_ if scaler.with_centering else zeros), (scaler.scale_ if scaler.with_scaling else ones)
    if kind is MinMaxScaler:
        return -scaler.min_ / scaler.scale_, 1.0 / scaler.scale_
    if kind is MaxAbsScaler:
        return zeros, scaler.scale_
    return None


def explained_variance(data: np.ndarray, common: np.ndarray) -> np.ndarray:
    """``1 - var(data - common) / var(data)`` per column over the finite entries of ``data``.

    A column with fewer than two finite entries or no variance gives NaN,
    without warnings.
    """
    data = np.where(np.isfinite(data), data, np.nan)
    observed = np.isfinite(data)
    count = observed.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(observed, data, 0.0).sum(axis=0) / count
        total = np.where(observed, (data - mean) ** 2, 0.0).sum(axis=0) / count
        residual = np.where(observed, data - common, np.nan)
        r_mean = np.where(observed, residual, 0.0).sum(axis=0) / count
        r_var = np.where(observed, (residual - r_mean) ** 2, 0.0).sum(axis=0) / count
        return np.where((count >= 2) & (total > 0), 1.0 - r_var / total, np.nan)

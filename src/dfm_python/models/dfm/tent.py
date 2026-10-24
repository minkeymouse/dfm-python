"""Tent kernels and constraint matrices of slower-frequency series."""

import numpy as np
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

from ...config.constants import (
    FREQUENCY_HIERARCHY,
    MAX_TENT_SIZE,
    DEFAULT_HIERARCHY_VALUE,
)
from ...logger import get_logger
from ...utils.errors import DataValidationError

if TYPE_CHECKING:
    from ...config.schema.model import _DFMEngineConfig

_logger = get_logger(__name__)

__all__ = [
    'generate_tent_weights',
    'generate_R_mat',
    'get_slower_freq_tent_weights',
    'get_agg_structure',
    'group_by_freq',
]


def generate_tent_weights(n_periods: int, tent_type: str = 'symmetric') -> np.ndarray:
    """Tent weights over ``n_periods`` base periods (``[1, 2, 3, 2, 1]`` for 5).

    Parameters
    ----------
    n_periods : int
        Base periods per slower period.
    tent_type : {'symmetric', 'linear', 'exponential'}, default 'symmetric'
        Symmetric weights are integers; the others are float32 (``linear``
        is all ones, ``exponential`` decays from the center and sums to
        ``n_periods``).
    """
    if tent_type == 'symmetric':
        # Odd: 1, ..., peak, ..., 1. Even: 1, ..., half, half, ..., 1.
        half = n_periods // 2
        weights = np.concatenate([np.arange(1, half + 1 + n_periods % 2), np.arange(half, 0, -1)])
        return weights.astype(int)
    if tent_type == 'linear':
        weights = np.ones(n_periods)
    elif tent_type == 'exponential':
        center = n_periods / 2
        weights = np.exp(-np.abs(np.arange(n_periods) - center) / (n_periods / 4))
        weights = weights / weights.sum() * n_periods
    else:
        raise DataValidationError(
            f"Unknown tent_type: {tent_type}. Must be 'symmetric', 'linear', or 'exponential'",
            details=f"Invalid tent_type: {tent_type}"
        )
    return weights.astype(np.float32)


def generate_R_mat(tent_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Constraint ``R_mat c = q`` that ties the loadings to the tent pattern.

    Row ``i`` enforces ``w[i+1] * c0 - w[0] * c(i+1) = 0``, so the loadings on
    the lags are proportional to the weights, ``c = (c0 / w[0]) * w``; any
    positive multiple of a kernel gives the same restriction. For
    ``[1, 2, 3, 2, 1]`` (FRBNY ``dfm.m``), ``R_mat = [[2, -1, 0, 0, 0],
    [3, 0, -1, 0, 0], [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]]``.

    Returns
    -------
    R_mat : np.ndarray
        ``(n-1, n)`` for ``n`` weights.
    q : np.ndarray
        Zeros, ``(n-1,)``.
    """
    if len(tent_weights) == 0:
        raise DataValidationError(
            "Tent weights array is empty",
            details="tent_weights must have at least one element"
        )
    if np.any(tent_weights <= 0):
        raise DataValidationError(
            f"Tent weights must be positive, but got: {tent_weights}",
            details="All tent weights must be > 0"
        )
    n = len(tent_weights)
    R_mat = np.zeros((n - 1, n))
    q = np.zeros(n - 1)
    for i in range(n - 1):
        R_mat[i, 0] = tent_weights[i + 1]
        R_mat[i, i + 1] = -tent_weights[0]
    return R_mat, q


def get_slower_freq_tent_weights(
    slower_freq: str,
    clock: str,
    tent_kernel_size: int,
    dtype: type = np.float32
) -> np.ndarray:
    """Symmetric tent weights of ``tent_kernel_size``, the fallback when the config gives none.

    ``slower_freq`` and ``clock`` are accepted for the caller's convenience
    and do not affect the result.
    """
    return generate_tent_weights(tent_kernel_size, 'symmetric').astype(dtype)


def get_agg_structure(
    config: '_DFMEngineConfig',
    clock: Optional[str] = None
) -> Dict[str, Any]:
    """Tent kernels and constraints of every frequency in ``config`` slower than ``clock``.

    The weights come from ``config.tent_weights`` under the key ``"freq:clock"``
    or ``"freq"``; a slower frequency without them raises ``DataValidationError``,
    as does a kernel longer than ``MAX_TENT_SIZE``.

    Returns
    -------
    dict
        ``structures``: ``{(freq, clock): (R_mat, q)}``; ``tent_weights``:
        ``{freq: weights}``; ``n_periods``: ``{freq: kernel size}``; ``clock``.
    """
    config_tent_weights = getattr(config, 'tent_weights', None)
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    slower_freqs = [freq for freq in set(config.get_frequencies() or [])
                    if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy]
    if slower_freqs and config_tent_weights is None:
        raise DataValidationError(
            f"Mixed-frequency data detected (slower frequencies: {slower_freqs}) but no tent_weights "
            f"specified in config. Please specify tent_weights in config for all frequency pairs, "
            f"e.g., tent_weights: {{'m:w': [1, 2, 1]}}",
            details=f"Slower frequencies detected: {slower_freqs}, clock: {clock}"
        )

    structures = {}
    tent_weights = {}
    n_periods_map = {}
    for freq in slower_freqs:
        pair_key = f"{freq}:{clock}"
        if pair_key in config_tent_weights:
            tent_w = config_tent_weights[pair_key]
        elif freq in config_tent_weights:
            tent_w = config_tent_weights[freq]
        else:
            raise DataValidationError(
                f"Tent weights not found in config for frequency pair ({freq}, {clock}). "
                f"Please specify tent_weights in config, e.g., tent_weights: {{'{pair_key}': [1, 2, 1]}}",
                details=f"Frequency pair: ({freq}, {clock})"
            )
        if not isinstance(tent_w, np.ndarray):
            tent_w = np.array(tent_w)
        if len(tent_w) > MAX_TENT_SIZE:
            raise DataValidationError(
                f"Tent kernel size {len(tent_w)} exceeds maximum allowed size {MAX_TENT_SIZE} "
                f"for frequency '{freq}'. For frequency gaps larger than {MAX_TENT_SIZE}, "
                f"use the missing data approach instead. Please reduce tent kernel size or "
                f"remove tent_weights for this frequency pair.",
                details=f"Frequency: {freq}, tent_size: {len(tent_w)}, MAX_TENT_SIZE: {MAX_TENT_SIZE}"
            )
        tent_weights[freq] = tent_w
        n_periods_map[freq] = len(tent_w)
        structures[(freq, clock)] = generate_R_mat(tent_w)

    return {
        'structures': structures,
        'tent_weights': tent_weights,
        'n_periods': n_periods_map,
        'clock': clock
    }


def group_by_freq(
    idx_i: np.ndarray,
    frequencies: np.ndarray,
    clock: str
) -> Dict[str, np.ndarray]:
    """Group the series indices ``idx_i`` by their frequency, ``{freq: indices}``.

    Without ``frequencies`` every index is at the clock. A series faster than
    the clock, or an index past the end of ``frequencies``, raises
    ``DataValidationError``.
    """
    if frequencies is None or len(frequencies) == 0:
        return {clock: idx_i.copy()}

    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    freq_groups: Dict[str, list] = {}
    faster_indices = []
    for idx in idx_i:
        if idx >= len(frequencies):
            raise DataValidationError(
                f"Index {idx} is out of bounds for frequencies array (length {len(frequencies)}). "
                f"This indicates a mismatch between series indices and frequency data. "
                f"All indices in idx_i must be < len(frequencies).",
                details=f"idx: {idx}, frequencies length: {len(frequencies)}, idx_i: {idx_i[:10]}..."
            )
        freq = frequencies[idx]
        if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) < clock_hierarchy:
            faster_indices.append(idx)
        else:
            freq_groups.setdefault(freq, []).append(idx)

    if faster_indices:
        raise DataValidationError(
            f"Higher frequencies (daily, weekly) are not supported. "
            f"Found {len(faster_indices)} series with frequency faster than clock '{clock}'. "
            f"Please use monthly, quarterly, semi-annual, or annual frequencies only.",
            details=f"Faster indices: {faster_indices}, clock: {clock}"
        )
    return {freq: np.array(indices, dtype=int) for freq, indices in freq_groups.items()}

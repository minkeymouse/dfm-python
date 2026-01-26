"""State-space model building functions.

This module provides functions for building state-space models,
including observation matrix construction and state-space assembly.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from ..logger import get_logger
from ..config.constants import (
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_DTYPE,
    DEFAULT_FACTOR_ORDER,
)
from .stability import create_scaled_identity

_logger = get_logger(__name__)


def build_dfm_structure(config: Any) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build DFM model structure from configuration.
    
    Parameters
    ----------
    config : Any
        DFMConfig instance with get_blocks_array() method
    
    Returns
    -------
    blocks : np.ndarray
        Block structure array (N x n_blocks)
    r : np.ndarray
        Number of factors per block (n_blocks,)
    num_factors : int
        Total number of factors
    p : int
        VAR lag order (always 1 for factors)
    """
    # Get model structure (stored as NumPy arrays)
    # Cache blocks array to avoid multiple calls to get_blocks_array()
    blocks_array = config.get_blocks_array()
    blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)
    
    # Get factors per block (r)
    factors_per_block = getattr(config, 'factors_per_block', None)
    if factors_per_block is not None:
        r = np.array(factors_per_block, dtype=DEFAULT_DTYPE)
    else:
        r = np.ones(blocks_array.shape[1], dtype=DEFAULT_DTYPE)
    
    # Total number of factors (computed from r to avoid redundancy)
    num_factors = int(np.sum(r))
    
    # AR order (always AR(1) for factors)
    p = DEFAULT_FACTOR_ORDER
    
    return blocks, r, num_factors, p


def build_dfm_blocks(
    blocks: np.ndarray,
    config: Any,
    columns: Optional[List[str]],
    N_actual: int
) -> np.ndarray:
    """Rebuild DFM blocks array to match data dimensions.
    
    Parameters
    ----------
    blocks : np.ndarray
        Current blocks array
    config : Any
        Config object with get_blocks_array() method
    columns : Optional[List[str]]
        Column names if available
    N_actual : int
        Expected number of series
        
    Returns
    -------
    np.ndarray
        Updated blocks array matching data dimensions
    """
    from ..logger.dfm_logger import log_blocks_diagnostics
    
    if columns is not None:
        # Clear cache and rebuild from config
        if hasattr(config, '_cached_blocks'):
            config._cached_blocks = None
        blocks_array = config.get_blocks_array(columns=columns)
        new_blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)
        _logger.info(f"Rebuilt blocks array: shape={new_blocks.shape}")
        log_blocks_diagnostics(new_blocks, columns, N_actual)
        return new_blocks
    else:
        # Fallback: pad or truncate to match dimensions
        n_blocks = blocks.shape[1]
        if blocks.shape[0] < N_actual:
            padding = np.zeros((N_actual - blocks.shape[0], n_blocks), dtype=DEFAULT_DTYPE)
            new_blocks = np.vstack([blocks, padding])
            _logger.warning(f"Padded blocks array with zeros: {N_actual - blocks.shape[0]} rows")
            return new_blocks
        elif blocks.shape[0] > N_actual:
            new_blocks = blocks[:N_actual, :]
            _logger.warning(f"Truncated blocks array: {blocks.shape[0]} -> {N_actual} rows")
            return new_blocks
        else:
            return blocks


def build_dfm_slower_freq_observation_matrix(
    N: int,
    n_clock_freq: int,
    n_slower_freq: int,
    tent_weights: np.ndarray,
    dtype: type = np.float32
) -> np.ndarray:
    """Build observation matrix for slower-frequency idiosyncratic chains.
    
    Parameters
    ----------
    N : int
        Total number of series
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    n_slower_freq : int
        Number of slower-frequency series (series slower than clock frequency, generic)
    tent_weights : np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1])
    dtype : type, default np.float32
        Data type for output matrix
        
    Returns
    -------
    np.ndarray
        Observation matrix (N x (tent_kernel_size * n_slower_freq))
    """
    tent_kernel_size = len(tent_weights)
    C_slower_freq = np.zeros((N, tent_kernel_size * n_slower_freq), dtype=dtype)
    C_slower_freq[n_clock_freq:, :] = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), tent_weights.reshape(1, -1))
    return C_slower_freq


def build_lag_matrix(
    factors: np.ndarray,
    T: int,
    num_factors: int,
    tent_kernel_size: int,
    p: int,
    dtype: type = np.float32
) -> np.ndarray:
    """Build lag matrix for factors.
    
    Parameters
    ----------
    factors : np.ndarray
        Factor matrix (T x num_factors)
    T : int
        Number of time periods
    num_factors : int
        Number of factors
    tent_kernel_size : int
        Tent kernel size
    p : int
        AR lag order
    dtype : type
        Data type
        
    Returns
    -------
    np.ndarray
        Lag matrix (T x (num_factors * num_lags))
    """
    num_lags = max(p + 1, tent_kernel_size)
    lag_matrix = np.zeros((T, num_factors * num_lags), dtype=dtype)
    
    # Vectorized implementation: build all lags at once
    for lag_idx in range(num_lags):
        start_idx = max(0, tent_kernel_size - lag_idx)
        end_idx = T - lag_idx
        if start_idx < end_idx:
            col_start = lag_idx * num_factors
            col_end = col_start + num_factors
            # Use advanced indexing for better performance
            lag_matrix[start_idx:end_idx, col_start:col_end] = factors[start_idx:end_idx, :num_factors].copy()
    
    return lag_matrix


__all__ = [
    'build_dfm_structure',
    'build_dfm_blocks',
    'build_dfm_slower_freq_observation_matrix',
    'build_lag_matrix',
]


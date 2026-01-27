"""Tent kernel matrix generation utilities.

This module provides functions for generating tent-shaped weights and constraint
matrices used in mixed-frequency Dynamic Factor Models.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

from ...logger import get_logger
from ...utils.errors import DataValidationError

if TYPE_CHECKING:
    from ...config.schema import DFMConfig

_logger = get_logger(__name__)

# Import constants
from ...config.constants import (
    FREQUENCY_HIERARCHY,
    MAX_TENT_SIZE,
    DEFAULT_HIERARCHY_VALUE,
)


def generate_tent_weights(n_periods: int, tent_type: str = 'symmetric') -> np.ndarray:
    
    """Generate tent-shaped weights for aggregation.
    
    Parameters
    ----------
    n_periods : int
        Number of base periods to aggregate (e.g., 5 for monthly->quarterly)
    tent_type : str
        Type of tent: 'symmetric' (default), 'linear', 'exponential'
        
    Returns
    -------
    weights : np.ndarray
        Array of weights that sum to a convenient number
    """
    if tent_type == 'symmetric':
        if n_periods % 2 == 1:
            # Odd number: symmetric around middle
            half = n_periods // 2
            weights = np.concatenate([
                np.arange(1, half + 2),      # [1, 2, ..., peak]
                np.arange(half, 0, -1)       # [peak-1, ..., 2, 1]
            ])
        else:
            # Even number: symmetric with two peaks
            half = n_periods // 2
            weights = np.concatenate([
                np.arange(1, half + 1),     # [1, 2, ..., half]
                np.arange(half, 0, -1)       # [half, ..., 2, 1]
            ])
    elif tent_type == 'linear':
        # Linear weights (simple average)
        weights = np.ones(n_periods)
    elif tent_type == 'exponential':
        # Exponential decay from center
        center = n_periods / 2
        weights = np.exp(-np.abs(np.arange(n_periods) - center) / (n_periods / 4))
        weights = weights / weights.sum() * n_periods  # Normalize
    else:
        raise DataValidationError(
            f"Unknown tent_type: {tent_type}. Must be 'symmetric', 'linear', or 'exponential'",
            details=f"Invalid tent_type: {tent_type}"
        )
    
    # Bug fix 1.1: Don't truncate non-integer weights (exponential, linear)
    # Only convert to int if all values are already integers (symmetric case)
    # For exponential/linear, preserve float precision
    if tent_type in ('exponential', 'linear'):
        # Keep as float - tent weights can be non-integer
        return weights.astype(np.float32)
    else:
        # Symmetric tent should produce integers, but validate
        if np.any(weights != np.round(weights)):
            raise DataValidationError(
                f"Symmetric tent weights should be integers, but got non-integer values: {weights}",
                details=f"tent_type: {tent_type}, n_periods: {n_periods}"
            )
        result = weights.astype(int)
        
        
        
        return result


def generate_R_mat(tent_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate constraint matrix R_mat from tent weights.
    
    This function generates constraint matrices matching the FRBNY MATLAB implementation.
    The constraint enforces: tent_weights[i+1] * c0 - 1 * c(i+1) = 0, which means
    c(i+1) = tent_weights[i+1] * c0, ensuring the tent kernel pattern in loadings.
    
    For tent weights [1, 2, 3, 2, 1], this produces:
        R_mat = [[2, -1,  0,  0,  0],   # 2*c0 - 1*c1 = 0 → c1 = 2*c0
                 [3,  0, -1,  0,  0],   # 3*c0 - 1*c2 = 0 → c2 = 3*c0
                 [2,  0,  0, -1,  0],   # 2*c0 - 1*c3 = 0 → c3 = 2*c0
                 [1,  0,  0,  0, -1]]   # 1*c0 - 1*c4 = 0 → c4 = 1*c0
    
    Parameters
    ----------
    tent_weights : np.ndarray
        Tent weights array, e.g., [1, 2, 3, 2, 1] for monthly->quarterly
        
    Returns
    -------
    R_mat : np.ndarray
        Constraint matrix of shape (n-1) × n matching MATLAB pattern
    q : np.ndarray
        Constraint vector of zeros, shape (n-1,)
    """
    # Bug fix 4.4: Validate tent weights
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
    if np.sum(tent_weights) <= 0:
        raise DataValidationError(
            f"Tent weights sum must be > 0, but got sum: {np.sum(tent_weights)}",
            details=f"tent_weights: {tent_weights}"
        )
    
    n = len(tent_weights)
    
    
    
    # Create constraint matrix: (n-1) rows × n columns
    R_mat = np.zeros((n - 1, n))
    q = np.zeros(n - 1)
    
    # Row i: enforces tent_weights[i+1]*c0 - 1*c(i+1) = 0
    # This ensures c(i+1) = tent_weights[i+1] * c0 (tent kernel pattern)
    # Pattern matches FRBNY MATLAB dfm.m lines 85-88
    for i in range(n - 1):
        R_mat[i, 0] = tent_weights[i + 1]  # Weight at index i+1
        R_mat[i, i + 1] = -1  # Always -1 (not -tent_weights[i+1])
    
    
    
    return R_mat, q


def get_slower_freq_tent_weights(
    slower_freq: str, 
    clock: str, 
    tent_kernel_size: int, 
    dtype: type = np.float32
) -> np.ndarray:
    """Get tent weights for slower-frequency idiosyncratic chain structure.
    
    Generates symmetric tent weights of the specified size. Tent weights must be
    explicitly specified in config for production use; this function provides
    a fallback for generating symmetric weights when needed.
    
    Parameters
    ----------
    slower_freq : str
        Slower frequency ('q', 'sa', 'a', etc.)
    clock : str
        Clock frequency ('m', 'q', etc.)
    tent_kernel_size : int
        Expected tent kernel size (used for generating weights)
    dtype : type, default np.float32
        Data type for output array
        
    Returns
    -------
    np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1] for quarterly-monthly)
    """
    # Generate symmetric tent weights as fallback
    tent_weights = generate_tent_weights(tent_kernel_size, 'symmetric').astype(dtype)
    return tent_weights


def get_agg_structure(
    config: 'DFMConfig', 
    clock: Optional[str] = None
) -> Dict[str, Any]:
    """Get aggregation structure for all frequency combinations in config based on clock.
    
    This function determines which series need tent kernels (those with frequencies
    slower than the clock) and generates the corresponding constraint matrices (R_mat)
    and constraint vectors (q) for use in constrained least squares estimation.
    
    Parameters
    ----------
    config : DFMConfig
        Model configuration containing series frequencies and structure
    clock : str, optional
        Base frequency (global clock) for latent factors, by default 'm' (monthly).
        All latent factors will evolve at this frequency.
        
    Returns
    -------
    aggregation_info : Dict[str, Any]
        Dictionary containing:
        - 'structures': Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]
            Maps (slower_freq, clock) tuples to (R_mat, q) constraint pairs
        - 'tent_weights': Dict[str, np.ndarray]
            Maps frequency strings to their tent weight arrays
        - 'n_periods': Dict[str, int]
            Maps frequency strings to tent kernel sizes
        - 'clock': str
            The clock frequency used
    """
    
    
    # Get frequencies from config (new API: frequency dict)
    frequencies_list = config.get_frequencies()
    # Bug fix 4.1: Preserve frequency counts, don't use set() which loses multiplicity
    # We still need unique frequencies for iteration, but we'll track counts separately
    frequencies = set(frequencies_list) if frequencies_list else set()
    # Track frequency counts for validation
    from collections import Counter
    frequency_counts = Counter(frequencies_list) if frequencies_list else Counter()
    
    
    
    structures = {}
    tent_weights = {}
    n_periods_map = {}
    
    # Find series with frequencies slower than clock (need tent kernels)
    # Tent weights must be specified in config.tent_weights
    config_tent_weights = getattr(config, 'tent_weights', None)
    
    if config_tent_weights is None:
        # No tent weights in config - check if any slower frequencies exist
        # Bug fix 4.3: Use DEFAULT_HIERARCHY_VALUE consistently instead of hardcoded 999
        clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
        slower_freqs = [freq for freq in frequencies 
                       if FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) > clock_hierarchy]
        if slower_freqs:
            raise DataValidationError(
                f"Mixed-frequency data detected (slower frequencies: {slower_freqs}) but no tent_weights "
                f"specified in config. Please specify tent_weights in config for all frequency pairs, "
                f"e.g., tent_weights: {{'m:w': [1, 2, 1]}}",
                details=f"Slower frequencies detected: {slower_freqs}, clock: {clock}"
            )
    
    # Bug fix 4.3: Use DEFAULT_HIERARCHY_VALUE consistently
    clock_hierarchy_val = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    for freq in frequencies:
        freq_hierarchy_val = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
        if freq_hierarchy_val > clock_hierarchy_val:
            # This frequency is slower than clock, get tent weights from config
            tent_w = None
            
            # Get tent weights from config (required)
            if config_tent_weights is not None:
                # Try both formats: 'freq:clock' (e.g., 'm:w') or just 'freq' (e.g., 'm')
                pair_key = f"{freq}:{clock}"
                if pair_key in config_tent_weights:
                    tent_w = config_tent_weights[pair_key]
                elif freq in config_tent_weights:
                    tent_w = config_tent_weights[freq]
                
                # Convert list to numpy array if needed
                if tent_w is not None and isinstance(tent_w, list):
                    tent_w = np.array(tent_w)
            
            # Tent weights must be provided in config
            if tent_w is None:
                raise DataValidationError(
                    f"Tent weights not found in config for frequency pair ({freq}, {clock}). "
                    f"Please specify tent_weights in config, e.g., tent_weights: {{'{pair_key}': [1, 2, 1]}}",
                    details=f"Frequency pair: ({freq}, {clock})"
                )
            
            # Process tent weights
            if tent_w is not None:
                # Convert to numpy array if not already
                if not isinstance(tent_w, np.ndarray):
                    tent_w = np.array(tent_w)
                
                # Bug fix 4.2: Warn/raise when tent size exceeds MAX_TENT_SIZE
                if len(tent_w) > MAX_TENT_SIZE:
                    raise DataValidationError(
                        f"Tent kernel size {len(tent_w)} exceeds maximum allowed size {MAX_TENT_SIZE} "
                        f"for frequency '{freq}'. For frequency gaps larger than {MAX_TENT_SIZE}, "
                        f"use the missing data approach instead. Please reduce tent kernel size or "
                        f"remove tent_weights for this frequency pair.",
                        details=f"Frequency: {freq}, tent_size: {len(tent_w)}, MAX_TENT_SIZE: {MAX_TENT_SIZE}"
                    )
                
                # Tent kernel available and within size limit
                tent_weights[freq] = tent_w
                n_periods_map[freq] = len(tent_w)
                # Generate R_mat from tent weights
                R_mat, q = generate_R_mat(tent_w)
                structures[(freq, clock)] = (R_mat, q)
                
                
    
    result = {
        'structures': structures,
        'tent_weights': tent_weights,
        'n_periods': n_periods_map,
        'clock': clock
    }
    
    
    
    return result


def group_by_freq(
    idx_i: np.ndarray,
    frequencies: np.ndarray,
    clock: str
) -> Dict[str, np.ndarray]:
    """Group series indices by their actual frequency.
    
    Groups series by their actual frequency values, allowing each frequency
    to be processed independently. Faster frequencies than clock are rejected.
    
    Parameters
    ----------
    idx_i : np.ndarray
        Array of series indices to group (1D integer array)
    frequencies : np.ndarray
        Array of frequency strings for each series (e.g., 'm', 'q', 'sa', 'a')
        Length should match total number of series
    clock : str
        Clock frequency ('m', 'q', 'sa', 'a') - all factors evolve at this frequency
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping frequency strings to arrays of series indices.
        
    Raises
    ------
    ValueError
        If any series has a frequency faster than the clock frequency
    """
    if frequencies is None or len(frequencies) == 0:
        # Fallback: assume all are same as clock if frequencies not provided
        return {clock: idx_i.copy()}
    
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    
    freq_groups: Dict[str, list] = {}
    faster_indices = []
    
    
    
    for idx in idx_i:
        # Bug fix 5.1: Raise error instead of silently skipping out-of-bounds indices
        if idx >= len(frequencies):
            raise DataValidationError(
                f"Index {idx} is out of bounds for frequencies array (length {len(frequencies)}). "
                f"This indicates a mismatch between series indices and frequency data. "
                f"All indices in idx_i must be < len(frequencies).",
                details=f"idx: {idx}, frequencies length: {len(frequencies)}, idx_i: {idx_i[:10]}..."
            )
        
        freq = frequencies[idx]
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
        
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
        raise DataValidationError(
            f"Higher frequencies (daily, weekly) are not supported. "
            f"Found {len(faster_indices)} series with frequency faster than clock '{clock}'. "
            f"Please use monthly, quarterly, semi-annual, or annual frequencies only.",
            details=f"Faster indices: {faster_indices}, clock: {clock}"
        )
    
    # Convert lists to numpy arrays
    result = {freq: np.array(indices, dtype=int) for freq, indices in freq_groups.items()}
    
    
    
    return result


__all__ = [
    'generate_tent_weights',
    'generate_R_mat',
    'get_slower_freq_tent_weights',
    'get_agg_structure',
    'group_by_freq',
]

"""Aggregation utilities for mixed-frequency data in Dynamic Factor Models.

This module implements the clock-based aggregation framework for handling mixed-frequency
time series in Dynamic Factor Models. The key innovation is the use of deterministic
tent kernels to map lower-frequency observed variables to higher-frequency latent states
within the observation equation.

The module provides:
- Frequency hierarchy definitions
- Tent weight generation and lookup
- Constraint matrix (R_mat) generation from tent weights
- Aggregation structure computation based on global clock

The tent kernel approach connects lower-frequency series to higher-frequency 
latent factors through weighted aggregation constraints. This allows all factors to 
evolve at the same clock frequency while properly handling mixed-frequency observations.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from ..config import DFMConfig

# ============================================================================
# Frequency Hierarchy
# ============================================================================

# Frequency hierarchy (from highest to lowest frequency)
# Used to determine which frequencies are slower/faster than the clock
FREQUENCY_HIERARCHY: Dict[str, int] = {
    'd': 1,   # Daily (highest frequency)
    'w': 2,   # Weekly
    'm': 3,   # Monthly
    'q': 4,   # Quarterly
    'sa': 5,  # Semi-annual
    'a': 6    # Annual (lowest frequency)
}

# ============================================================================
# Tent Kernel Configuration
# ============================================================================

# Maximum tent kernel size (number of periods)
# For frequency gaps larger than this, the missing data approach is used instead
# This prevents excessively large constraint matrices that would be computationally
# expensive and potentially numerically unstable
MAX_TENT_SIZE: int = 12

# Deterministic tent weights lookup for supported frequency pairs
# Format: (slower_freq, faster_freq) -> tent_weights_array
# Example: ('q', 'm'): [1, 2, 3, 2, 1] means a quarterly observation aggregates
# 5 monthly latent states with weights 1, 2, 3, 2, 1 (peaking at the middle month).
TENT_WEIGHTS_LOOKUP: Dict[Tuple[str, str], np.ndarray] = {
    ('q', 'm'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: quarterly -> monthly
    ('sa', 'm'): np.array([1, 2, 3, 4, 3, 2, 1]),             # 7 periods: semi-annual -> monthly
    ('a', 'm'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: annual -> monthly
    ('m', 'w'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: monthly -> weekly
    ('q', 'w'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: quarterly -> weekly
    ('sa', 'q'): np.array([1, 2, 1]),                         # 3 periods: semi-annual -> quarterly
    ('a', 'q'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: annual -> quarterly
    ('a', 'sa'): np.array([1, 2, 1]),                         # 3 periods: annual -> semi-annual
}


def generate_tent_weights(n_periods: int, tent_type: str = 'symmetric') -> np.ndarray:
    """Generate tent-shaped weights for aggregation.
    
    Parameters:
    -----------
    n_periods : int
        Number of base periods to aggregate (e.g., 5 for monthly->quarterly)
    tent_type : str
        Type of tent: 'symmetric' (default), 'linear', 'exponential'
        
    Returns:
    --------
    weights : np.ndarray
        Array of weights that sum to a convenient number
        
    Examples:
    --------
    >>> generate_tent_weights(5, 'symmetric')
    array([1, 2, 3, 2, 1])  # Classic tent for monthly->quarterly
    
    >>> generate_tent_weights(7, 'symmetric')
    array([1, 2, 3, 4, 3, 2, 1])  # Weekly aggregation
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
        raise ValueError(f"Unknown tent_type: {tent_type}. Must be 'symmetric', 'linear', or 'exponential'")
    
    return weights.astype(int)


def generate_R_mat(tent_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate constraint matrix R_mat from tent weights.
    
    Parameters:
    -----------
    tent_weights : np.ndarray
        Tent weights array, e.g., [1, 2, 3, 2, 1] for monthly->quarterly
        
    Returns:
    --------
    R_mat : np.ndarray
        Constraint matrix of shape (n-1) × n
    q : np.ndarray
        Constraint vector of zeros, shape (n-1,)
        
    Examples:
    --------
    >>> weights = np.array([1, 2, 3, 2, 1])
    >>> R_mat, q = generate_R_mat(weights)
    >>> # Returns the classic monthly->quarterly R_mat
    """
    n = len(tent_weights)
    w1 = tent_weights[0]  # First weight (reference)
    
    # Create constraint matrix: (n-1) rows × n columns
    R_mat = np.zeros((n - 1, n))
    q = np.zeros(n - 1)
    
    # Row i: relates w1*c1 to w(i+1)*c(i+1)
    # Constraint: w1*c1 - w(i+1)*c(i+1) = 0
    for i in range(n - 1):
        R_mat[i, 0] = w1              # Coefficient for c1
        R_mat[i, i + 1] = -tent_weights[i + 1]  # Coefficient for c(i+1)
        # All other columns remain 0
    
    return R_mat, q


def get_tent_weights_for_pair(slower_freq: str, faster_freq: str) -> Optional[np.ndarray]:
    """Get deterministic tent weights for a frequency pair.
    
    Parameters:
    -----------
    slower_freq : str
        Slower frequency (e.g., 'q' for quarterly)
    faster_freq : str
        Faster frequency (e.g., 'm' for monthly) - this is the clock
    
    Returns:
    --------
    tent_weights : np.ndarray or None
        Tent weights array if pair is supported, None otherwise
        
    Examples:
    --------
    >>> get_tent_weights_for_pair('q', 'm')
    array([1, 2, 3, 2, 1])  # Quarterly -> monthly
    
    >>> get_tent_weights_for_pair('m', 'd')
    None  # Not supported (too large gap)
    """
    return TENT_WEIGHTS_LOOKUP.get((slower_freq, faster_freq))


def get_aggregation_structure(
    config: DFMConfig, 
    clock: str = 'm'
) -> Dict[str, Any]:
    """Get aggregation structure for all frequency combinations in config based on clock.
    
    This function determines which series need tent kernels (those with frequencies
    slower than the clock) and generates the corresponding constraint matrices (R_mat)
    and constraint vectors (q) for use in constrained least squares estimation.
    
    The aggregation structure follows the clock-based approach:
    - All latent factors evolve at the clock frequency
    - Series with frequencies slower than the clock use tent kernels
    - Series with frequencies faster than the clock use missing data approach
    - If tent kernel size exceeds MAX_TENT_SIZE, missing data approach is used
    
    Parameters
    ----------
    config : DFMConfig
        Model configuration containing series frequencies and structure
    clock : str, optional
        Base frequency (global clock) for nowcasting, by default 'm' (monthly).
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
            
    Examples
    --------
    >>> from dfm_python import load_config
    >>> config = load_config('config.yaml')
    >>> agg_info = get_aggregation_structure(config, clock='m')
    >>> # Check which frequencies need tent kernels
    >>> print(agg_info['tent_weights'])
    {'q': array([1, 2, 3, 2, 1]), 'sa': array([1, 2, 3, 4, 3, 2, 1])}
    """
    # Get frequencies using new API with fallback for legacy configs
    if hasattr(config, 'get_frequencies'):
        frequencies = set(config.get_frequencies())
    elif hasattr(config, 'Frequency'):
        frequencies = set(config.Frequency)
    else:
        frequencies = set()
    structures = {}
    tent_weights = {}
    n_periods_map = {}
    
    # Find series with frequencies slower than clock (need tent kernels)
    for freq in frequencies:
        if FREQUENCY_HIERARCHY.get(freq, 999) > FREQUENCY_HIERARCHY.get(clock, 0):
            # This frequency is slower than clock, check if tent kernel is available
            tent_w = get_tent_weights_for_pair(freq, clock)
            if tent_w is not None and len(tent_w) <= MAX_TENT_SIZE:
                # Tent kernel available and within size limit
                tent_weights[freq] = tent_w
                n_periods_map[freq] = len(tent_w)
                # Generate R_mat from tent weights
                R_mat, q = generate_R_mat(tent_w)
                structures[(freq, clock)] = (R_mat, q)
            # If tent kernel not available or too large, use missing data approach (no structure needed)
    
    return {
        'structures': structures,
        'tent_weights': tent_weights,
        'n_periods': n_periods_map,
        'clock': clock
    }


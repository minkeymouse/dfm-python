"""Block structure configuration for DFM.

This module defines the BlockStructure dataclass used to group block-related
parameters in EM algorithm, replacing long conditional checks with a single
optional parameter.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class BlockStructure:
    """Block structure configuration for EM algorithm.
    
    Groups all block-related parameters together to simplify function signatures
    and replace long conditional checks.
    
    Attributes
    ----------
    blocks : np.ndarray
        Block structure array (N x n_blocks)
    r : np.ndarray
        Number of factors per block (n_blocks,)
    p : int
        VAR lag order
    p_plus_one : int
        State dimension per factor. This is equal to max_lag_size = max(p + 1, tent_kernel_size).
        The name "p_plus_one" is used for backward compatibility with the EM algorithm interface.
        In the model code, this is called "max_lag_size" to more clearly indicate it accounts
        for tent kernel size when tent_kernel_size > p + 1.
    n_clock_freq : int
        Number of clock-frequency series
    idio_indicator : np.ndarray
        Idiosyncratic component indicator (N,)
    R_mat : np.ndarray, optional
        Tent kernel constraint matrix for mixed-frequency data
    q : np.ndarray, optional
        Tent kernel constraint vector for mixed-frequency data
    n_slower_freq : int, optional
        Number of slower-frequency series
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights
    
    Cached indices (computed once, reused across EM iterations):
    _cached_unique_blocks : List[np.ndarray], optional
        Unique block patterns (cached)
    _cached_unique_indices : List[int], optional
        Indices of unique block patterns (cached)
    _cached_bl_idxM : List[np.ndarray], optional
        Clock-frequency factor loadings indices per unique block pattern (cached)
    _cached_bl_idxQ : List[np.ndarray], optional
        Slower-frequency factor loadings indices per unique block pattern (cached)
    _cached_R_con : np.ndarray, optional
        Block diagonal constraint matrix (cached)
    _cached_q_con : np.ndarray, optional
        Constraint vector (cached)
    _cached_total_factor_dim : int, optional
        Total factor state dimension (cached)
    _cached_idio_indicator_M : np.ndarray, optional
        Clock-frequency idiosyncratic component indicator (cached)
    _cached_n_idio_M : int, optional
        Number of clock-frequency idiosyncratic components (cached)
    _cached_c_idio_indicator : np.ndarray, optional
        Cumulative sum of idiosyncratic component indicator (cached)
    _cached_rp1 : int, optional
        Start of idiosyncratic components in state space (cached)
    """
    blocks: np.ndarray
    r: np.ndarray
    p: int
    p_plus_one: int
    n_clock_freq: int
    idio_indicator: np.ndarray
    R_mat: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    n_slower_freq: Optional[int] = None
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
    
    # Cached indices (computed once, reused across EM iterations)
    _cached_unique_blocks: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_unique_indices: Optional[List[int]] = field(default=None, init=False, repr=False)
    _cached_bl_idxM: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_bl_idxQ: Optional[List[np.ndarray]] = field(default=None, init=False, repr=False)
    _cached_R_con: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_q_con: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_total_factor_dim: Optional[int] = field(default=None, init=False, repr=False)
    _cached_idio_indicator_M: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_n_idio_M: Optional[int] = field(default=None, init=False, repr=False)
    _cached_c_idio_indicator: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_rp1: Optional[int] = field(default=None, init=False, repr=False)
    
    def is_valid(self) -> bool:
        """Check if block structure is valid (all required fields are not None)."""
        return (
            self.blocks is not None
            and self.r is not None
            and self.p is not None
            and self.p_plus_one is not None
            and self.n_clock_freq is not None
            and self.idio_indicator is not None
        )
    
    def has_cached_indices(self) -> bool:
        """Check if block structure indices are cached."""
        return self._cached_unique_blocks is not None
    
    def clear_cache(self) -> None:
        """Clear all cached indices to ensure clean state for new training runs."""
        self._cached_unique_blocks = None
        self._cached_unique_indices = None
        self._cached_bl_idxM = None
        self._cached_bl_idxQ = None
        self._cached_R_con = None
        self._cached_q_con = None
        self._cached_total_factor_dim = None
        self._cached_idio_indicator_M = None
        self._cached_n_idio_M = None
        self._cached_c_idio_indicator = None
        self._cached_rp1 = None


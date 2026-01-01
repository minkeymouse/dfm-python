"""Block structure configuration for DFM.

This module defines the BlockStructure dataclass used to group block-related
parameters in EM algorithm, replacing long conditional checks with a single
optional parameter.
"""

from dataclasses import dataclass
from typing import Optional, Dict
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
        p + 1 (state dimension per factor)
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


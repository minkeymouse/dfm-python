"""State-space model building functions.

This module provides functions for building state-space models,
including observation matrix construction and state-space assembly.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..logger import get_logger
from ..utils.errors import ConfigurationError
from ..config.constants import (
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_IDENTITY_SCALE,
)
from .stability import ensure_positive_definite, compute_cov_safe, create_scaled_identity

_logger = get_logger(__name__)


def build_observation_matrix(C: np.ndarray, factor_order: int = 1, N: int = 0) -> np.ndarray:
    """Build observation matrix H including idiosyncratic components.
    
    Constructs the observation matrix H = [C, I] for AR(1), where C loads on factors and I on idio.
    
    Parameters
    ----------
    C : np.ndarray
        Loading matrix (N x m) from decoder
    factor_order : int, default 1
        AR lag order (always 1)
    N : int, default 0
        Number of series (unused parameter)
        
    Returns
    -------
    H : np.ndarray
        Observation matrix (N x state_dim)
    """
    N_series, m = C.shape
    
    # H = [C, I] where C loads on f_t, I loads on eps_t
    H = np.hstack([C, create_scaled_identity(N_series, DEFAULT_IDENTITY_SCALE)])
    
    return H


def build_state_space(
    factors: np.ndarray,
    A_f: np.ndarray,
    Q_f: np.ndarray,
    A_eps: np.ndarray,
    Q_eps: np.ndarray,
    factor_order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space model with companion form.
    
    Constructs the complete state-space model including both factors
    and idiosyncratic components in the state vector.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
    A_f : np.ndarray
        Factor transition matrix (m x m) for AR(1)
    Q_f : np.ndarray
        Factor innovation covariance (m x m)
    A_eps : np.ndarray
        Idiosyncratic AR(1) coefficients (N x N), diagonal
    Q_eps : np.ndarray
        Idiosyncratic innovation covariance (N x N), diagonal
    factor_order : int, default 1
        AR lag order (always 1)
        
    Returns
    -------
    A : np.ndarray
        Complete transition matrix (state_dim x state_dim)
    Q : np.ndarray
        Complete innovation covariance (state_dim x state_dim)
    Z_0 : np.ndarray
        Initial state vector
    V_0 : np.ndarray
        Initial state covariance
    """
    T, m = factors.shape
    N = A_eps.shape[0]
    
    # State: [f_t, eps_t]
    # Transition: f_t = A_f @ f_{t-1} + v_f, eps_t = A_eps @ eps_{t-1} + v_eps
    # Block diagonal structure
    A = np.block([
        [A_f, np.zeros((m, N))],
        [np.zeros((N, m)), A_eps]
    ])
    
    Q = np.block([
        [Q_f, np.zeros((m, N))],
        [np.zeros((N, m)), Q_eps]
    ])
    
    # Initial state: [f_0, eps_0]
    Z_0 = np.concatenate([factors[0, :], np.zeros(N)])
    
    # Initial covariance: block diagonal
    V_f = compute_cov_safe(factors.T, rowvar=True, pairwise_complete=False)
    V_eps = np.diag(np.diag(Q_eps))  # Use Q_eps as initial idio covariance
    V_0 = np.block([
        [V_f, np.zeros((m, N))],
        [np.zeros((N, m)), V_eps]
    ])
    
    return A, Q, Z_0, V_0




def compute_idio_lengths(
    config: Any,
    clock: str,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
) -> np.ndarray:
    """Compute idiosyncratic chain length for each series.
    
    For clock-frequency series: returns 1 (single AR(1) state).
    For slower-frequency series: returns tent length (L) if augment_idio_slow is True, else 0.
    If augment_idio is False: all series return 0.
    
    Parameters
    ----------
    config : DFMConfig
        Model configuration containing series frequencies and idio augmentation flags
    clock : str
        Clock frequency ('d', 'w', 'm', 'q', 'sa', 'a')
    tent_weights_dict : Dict[str, np.ndarray], optional
        Dictionary mapping frequency strings to tent weight arrays.
        If None, will be computed from config using get_agg_structure.
        
    Returns
    -------
    idio_chain_lengths : np.ndarray
        Array of chain lengths, one per series.
        - 0: no idio augmentation
        - 1: clock-frequency series (AR(1) idio)
        - L: slower-frequency series (tent-length chain, where L = len(tent_weights))
    """
    from ..config.constants import FREQUENCY_HIERARCHY
    from .tent import get_agg_structure
    
    # Get frequencies using new API
    frequencies = config.get_frequencies()
    if not frequencies:
        return np.zeros(0, dtype=int)
    
    if not config.augment_idio:
        # Feature disabled: all zeros
        return np.zeros(len(frequencies), dtype=int)
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    
    # Get tent weights if not provided
    if tent_weights_dict is None:
        agg_structure = get_agg_structure(config, clock=clock)
        tent_weights_dict = agg_structure.get('tent_weights', {})
    
    lengths = np.zeros(len(frequencies), dtype=int)
    
    for i, freq in enumerate(frequencies):
        freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE)
        
        if freq_hierarchy == clock_hierarchy:
            # Clock-frequency series: AR(1) idio state
            lengths[i] = 1
        elif freq_hierarchy > clock_hierarchy:
            # Slower-frequency series: tent-length chain (if enabled)
            if config.augment_idio_slow and tent_weights_dict is not None:
                tent_weights = tent_weights_dict.get(freq)
                if tent_weights is not None:
                    lengths[i] = len(tent_weights)
                # If no tent weights available, length stays 0 (no idio for this series)
            # If augment_idio_slow is False, length stays 0
    
    return lengths


__all__ = [
    'build_observation_matrix',
    'build_state_space',
    'compute_idio_lengths',
]


"""State-space model building functions.

This module provides functions for building state-space models,
including observation matrix construction and state-space assembly.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..logger import get_logger
from ..config.constants import (
    AR_CLIP_MIN,
    AR_CLIP_MAX,
    DEFAULT_HIERARCHY_VALUE,
)
from .stability import ensure_positive_definite

_logger = get_logger(__name__)


def clip_ar(
    A: np.ndarray,
    min_val: float = AR_CLIP_MIN,
    max_val: float = AR_CLIP_MAX,
    warn: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients to clip
    min_val : float, default AR_CLIP_MIN
        Minimum allowed value
    max_val : float, default AR_CLIP_MAX
        Maximum allowed value
    warn : bool, default True
        Whether to log warnings
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients
    stats : dict
        Statistics about clipping
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def apply_ar_clipping(
    A: np.ndarray,
    config: Optional[Any] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration.
    
    Parameters
    ----------
    A : np.ndarray
        AR coefficients
    config : object, optional
        Configuration object with clipping parameters
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped AR coefficients
    stats : dict
        Statistics about clipping
    """
    if config is None:
        return clip_ar(A, AR_CLIP_MIN, AR_CLIP_MAX, True)
    
    clip_enabled = getattr(config, 'clip_ar_coefficients', True) if config is not None else True
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = getattr(config, 'ar_clip_min', AR_CLIP_MIN) if config is not None else AR_CLIP_MIN
    max_val = getattr(config, 'ar_clip_max', AR_CLIP_MAX) if config is not None else AR_CLIP_MAX
    warn = getattr(config, 'warn_on_ar_clip', True) if config is not None else True
    return clip_ar(A, min_val, max_val, warn)




def build_observation_matrix(C: np.ndarray, factor_order: int, N: int) -> np.ndarray:
    """Build observation matrix H including idiosyncratic components.
    
    Constructs the observation matrix H = [C, I] for VAR(1) or
    H = [C, 0, I] for VAR(2), where C loads on factors and I on idio.
    
    Parameters
    ----------
    C : np.ndarray
        Loading matrix (N x m) from decoder
    factor_order : int
        VAR lag order (1 or 2)
    N : int
        Number of series
        
    Returns
    -------
    H : np.ndarray
        Observation matrix (N x state_dim)
    """
    N_series, m = C.shape
    
    if factor_order == 1:
        # H = [C, I] where C loads on f_t, I loads on eps_t
        H = np.hstack([C, np.eye(N_series)])
    elif factor_order == 2:
        # H = [C, 0, I] where C loads on f_t, 0 on f_{t-1}, I on eps_t
        H = np.hstack([C, np.zeros((N_series, m)), np.eye(N_series)])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
    return H


def build_state_space(
    factors: np.ndarray,
    A_f: np.ndarray,
    Q_f: np.ndarray,
    A_eps: np.ndarray,
    Q_eps: np.ndarray,
    factor_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space model with companion form.
    
    Constructs the complete state-space model including both factors
    and idiosyncratic components in the state vector.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x m)
    A_f : np.ndarray
        Factor transition matrix (m x m) for VAR(1) or (m x 2m) for VAR(2)
    Q_f : np.ndarray
        Factor innovation covariance (m x m)
    A_eps : np.ndarray
        Idiosyncratic AR(1) coefficients (N x N), diagonal
    Q_eps : np.ndarray
        Idiosyncratic innovation covariance (N x N), diagonal
    factor_order : int
        VAR lag order (1 or 2)
        
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
    
    if factor_order == 1:
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
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))  # Use Q_eps as initial idio covariance
        V_0 = np.block([
            [V_f, np.zeros((m, N))],
            [np.zeros((N, m)), V_eps]
        ])
        
    elif factor_order == 2:
        # State: [f_t, f_{t-1}, eps_t]
        # Transition: f_t = A1 @ f_{t-1} + A2 @ f_{t-2} + v_f
        #            f_{t-1} = f_{t-1} (identity)
        #            eps_t = A_eps @ eps_{t-1} + v_eps
        A1 = A_f[:, :m]
        A2 = A_f[:, m:]
        
        A = np.block([
            [A1, A2, np.zeros((m, N))],
            [np.eye(m), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), A_eps]
        ])
        
        Q = np.block([
            [Q_f, np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), Q_eps]
        ])
        
        # Initial state: [f_0, f_{-1}, eps_0]
        # Use f_0 for both f_0 and f_{-1} (or use first two if available)
        if T >= 2:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        else:
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
        
        # Initial covariance: block diagonal
        V_f = np.cov(factors.T)
        V_eps = np.diag(np.diag(Q_eps))
        V_0 = np.block([
            [V_f, V_f, np.zeros((m, N))],
            [V_f, V_f, np.zeros((m, N))],
            [np.zeros((N, m)), np.zeros((N, m)), V_eps]
        ])
    else:
        raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
    
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
    
    if not config.augment_idio:
        # Feature disabled: all zeros
        return np.zeros(len(config.series), dtype=int)
    
    # Get frequencies for each series
    frequencies = [s.frequency for s in config.series]
    clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, DEFAULT_HIERARCHY_VALUE)
    
    # Get tent weights if not provided
    if tent_weights_dict is None:
        agg_structure = get_agg_structure(config, clock=clock)
        tent_weights_dict = agg_structure.get('tent_weights', {})
    
    lengths = np.zeros(len(config.series), dtype=int)
    
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
    'clip_ar',
    'apply_ar_clipping',
    'build_observation_matrix',
    'build_state_space',
    'compute_idio_lengths',
]


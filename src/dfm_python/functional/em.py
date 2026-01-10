"""EM algorithm implementation for DFM.

This module provides the Expectation-Maximization algorithm for DFM parameter estimation.
Uses pykalman for the E-step (Kalman filter/smoother) and implements the M-step
with block structure preservation.

Includes numerical stability utilities to ensure convergence safety.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
try:
    from scipy.linalg import solve
except ImportError:
    # Fallback to numpy if scipy not available
    solve = np.linalg.solve

from ..ssm.kalman import DFMKalmanFilter
from ..logger import get_logger
from ..config.schema.block import BlockStructure
from ..config.constants import (
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_OBSERVATION_NOISE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_MAX_ITER,
    MAX_EIGENVALUE,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    VAR_STABILITY_THRESHOLD,
    DEFAULT_SLOWER_FREQ_AR_COEF,
    DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR,
    DEFAULT_EXTREME_FORECAST_THRESHOLD,
    DEFAULT_CLEAN_NAN,
    DEFAULT_MAX_VARIANCE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_TENT_KERNEL_SIZE,
)
from ..numeric.stability import (
    ensure_positive_definite,
    cap_max_eigenval,
    ensure_covariance_stable,
    ensure_process_noise_stable,
    solve_regularized_ols,
    create_scaled_identity,
)
from ..numeric.estimator import (
    estimate_var_unified,
    estimate_ar1_unified,
    estimate_constrained_ols_unified,
    estimate_variance_unified,
)
from ..utils.helper import handle_linear_algebra_error

_logger = get_logger(__name__)


@dataclass
class EMConfig:
    """Configuration for EM algorithm parameters."""
    regularization: float = DEFAULT_REGULARIZATION
    min_norm: float = MIN_EIGENVALUE
    max_eigenval: float = VAR_STABILITY_THRESHOLD  # Stability threshold for VAR matrices
    min_variance: float = MIN_DIAGONAL_VARIANCE
    max_variance: float = DEFAULT_MAX_VARIANCE  # Maximum variance cap
    min_iterations_for_convergence_check: int = 2
    convergence_log_interval: int = DEFAULT_LOG_INTERVAL
    progress_log_interval: int = DEFAULT_PROGRESS_LOG_INTERVAL
    small_loglik_threshold: float = MIN_FACTOR_VARIANCE
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    # Initialization constants (used by DFM initialization)
    default_transition_coef: float = DEFAULT_TRANSITION_COEF
    default_process_noise: float = DEFAULT_PROCESS_NOISE
    default_observation_noise: float = MIN_DIAGONAL_VARIANCE
    matrix_regularization: float = DEFAULT_REGULARIZATION
    eigenval_floor: float = MIN_EIGENVALUE
    slower_freq_ar_coef: float = DEFAULT_SLOWER_FREQ_AR_COEF  # AR coefficient for slower-frequency idiosyncratic components
    tent_kernel_size: int = DEFAULT_TENT_KERNEL_SIZE
    slower_freq_variance_denominator: float = DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR  # Variance denominator for slower-frequency series
    extreme_forecast_threshold: float = DEFAULT_EXTREME_FORECAST_THRESHOLD


_DEFAULT_EM_CONFIG = EMConfig()


def _align_blocks_to_data(blocks: np.ndarray, n_series: int) -> np.ndarray:
    """Align blocks array to match the number of data series.
    
    Parameters
    ----------
    blocks : np.ndarray
        Block structure array (may have wrong number of rows)
    n_series : int
        Expected number of series (columns in X)
        
    Returns
    -------
    np.ndarray
        Aligned blocks array with exactly n_series rows
    """
    blocks = blocks.copy()
    n_blocks_cols = blocks.shape[1]
    
    if blocks.shape[0] < n_series:
        # Pad with zeros (series not in any block)
        padding = np.zeros((n_series - blocks.shape[0], n_blocks_cols), dtype=blocks.dtype)
        blocks = np.vstack([blocks, padding])
    elif blocks.shape[0] > n_series:
        # Truncate to match data
        blocks = blocks[:n_series, :].copy()
    
    return blocks


def _compute_and_cache_block_indices(block_structure: BlockStructure, N: int) -> None:
    """Compute and cache block structure indices (computed once, reused across EM iterations).
    
    This function computes all static block structure indices that don't change during EM iterations:
    - Unique block patterns
    - bl_idxM, bl_idxQ (factor loading indices)
    - Constraint matrices (R_con, q_con)
    - Idiosyncratic component indices
    
    Parameters
    ----------
    block_structure : BlockStructure
        Block structure configuration (will be modified in-place with cached indices)
    N : int
        Number of data series (columns in X)
    """
    if block_structure.has_cached_indices():
        return  # Already cached
    
    blocks = block_structure.blocks
    r = block_structure.r
    p_plus_one = block_structure.p_plus_one
    n_blocks = len(r)
    R_mat = block_structure.R_mat
    q = block_structure.q
    n_clock_freq = block_structure.n_clock_freq
    idio_indicator = block_structure.idio_indicator
    
    # Align blocks shape to match X
    blocks_aligned = _align_blocks_to_data(blocks, N)
    
    # Find unique block patterns
    block_tuples = [tuple(row) for row in blocks_aligned]
    unique_blocks = []
    unique_indices = []
    seen = set()
    for i, bt in enumerate(block_tuples):
        if bt not in seen:
            unique_blocks.append(blocks_aligned[i].copy())
            unique_indices.append(i)
            seen.add(bt)
    
    # Build block indices for clock-frequency and slower-frequency factors
    bl_idxM = []
    bl_idxQ = []
    R_con = None
    q_con = None
    
    # Calculate total factor state dimension
    total_factor_dim = int(np.sum(r) * p_plus_one)
    
    if R_mat is not None and q is not None:
        from scipy.linalg import block_diag
        R_con_blocks = []
        q_con_blocks = []
        
        # Build indices for each unique block pattern
        for bl_row in unique_blocks:
            bl_idxQ_row = []
            bl_idxM_row = []
            
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
            
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
            
            # Build constraint matrix for blocks used in this pattern
            pattern_blocks = [block_idx for block_idx in range(n_blocks) if bl_row[block_idx] > 0]
            if pattern_blocks:
                for block_idx in pattern_blocks:
                    R_con_blocks.append(np.kron(R_mat, create_scaled_identity(int(r[block_idx]), DEFAULT_IDENTITY_SCALE)))
                    q_con_blocks.append(np.zeros(R_mat.shape[0] * int(r[block_idx])))
        
        if R_con_blocks:
            R_con = block_diag(*R_con_blocks)
            q_con = np.concatenate(q_con_blocks)
    else:
        # No constraints - simpler indexing
        for bl_row in unique_blocks:
            bl_idxM_row = []
            bl_idxQ_row = []
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
    
    # Convert to boolean arrays
    bl_idxM = [np.array(row, dtype=bool) for row in bl_idxM] if bl_idxM else []
    bl_idxQ = [np.array(row, dtype=bool) for row in bl_idxQ] if bl_idxQ else []
    
    # Idiosyncratic component indices
    idio_indicator_M = idio_indicator[:n_clock_freq]
    n_idio_M = int(np.sum(idio_indicator_M))
    c_idio_indicator = np.cumsum(idio_indicator)
    rp1 = int(np.sum(r) * p_plus_one)  # Start of idiosyncratic components
    
    # Cache all computed indices
    block_structure._cached_unique_blocks = unique_blocks
    block_structure._cached_unique_indices = unique_indices
    block_structure._cached_bl_idxM = bl_idxM
    block_structure._cached_bl_idxQ = bl_idxQ
    block_structure._cached_R_con = R_con
    block_structure._cached_q_con = q_con
    block_structure._cached_total_factor_dim = total_factor_dim
    block_structure._cached_idio_indicator_M = idio_indicator_M
    block_structure._cached_n_idio_M = n_idio_M
    block_structure._cached_c_idio_indicator = c_idio_indicator
    block_structure._cached_rp1 = rp1


def _update_transition_matrix(EZ: np.ndarray, A: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update transition matrix A using OLS regression."""
    T, m = EZ.shape
    if T <= 1:
        return A
    
    def _compute_A() -> np.ndarray:
        Y = EZ[1:, :]  # (T-1, m)
        X = EZ[:-1, :]  # (T-1, m)
        A_new = solve_regularized_ols(X, Y, regularization=config.regularization).T
        return cap_max_eigenval(A_new, max_eigenval=config.max_eigenval, symmetric=False, warn=False)
    
    return handle_linear_algebra_error(
        _compute_A, "transition matrix update",
        fallback_value=A
    )


def _update_transition_matrix_blocked(
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    blocks: np.ndarray,
    r: np.ndarray,
    p: int,
    p_plus_one: int,
    idio_indicator: np.ndarray,
    n_clock_freq: int,
    config: EMConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update transition matrix A and Q block-by-block, matching Nowcasting MATLAB code.
    
    This function implements the block-by-block update for factors (lines 325-367 in dfm.m)
    and the idiosyncratic component update (lines 369-397 in dfm.m).
    
    Parameters
    ----------
    EZ : np.ndarray
        Smoothed state means (T x m), where m is state dimension
    V_smooth : np.ndarray
        Smoothed state covariances (T x m x m)
    VVsmooth : np.ndarray
        Lag-1 cross-covariances (T x m x m)
    A : np.ndarray
        Current transition matrix (m x m)
    Q : np.ndarray
        Current process noise covariance (m x m)
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
    idio_indicator : np.ndarray
        Idiosyncratic component indicator (N,)
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    config : EMConfig
        EM configuration
        
    Returns
    -------
    A_new : np.ndarray
        Updated transition matrix
    Q_new : np.ndarray
        Updated process noise covariance
    V_0_new : np.ndarray
        Updated initial state covariance
    """
    T = EZ.shape[0]
    m = EZ.shape[1]
    n_blocks = len(r)
    
    # Initialize output
    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_smooth[0].copy() if len(V_smooth) > 0 else create_scaled_identity(m, config.min_variance)
    
    # Update factor parameters block-by-block
    for i in range(n_blocks):
        r_i = int(r[i])  # Number of factors in block i
        rp = r_i * p  # State dimension for block i (factors * lags)
        rp1 = int(np.sum(r[:i]) * p_plus_one)  # Cumulative state dimension before block i
        b_subset = slice(rp1, rp1 + rp)  # Indices for block i state
        t_start = rp1  # Transition matrix factor idx start
        t_end = rp1 + r_i * p_plus_one  # Transition matrix factor idx end
        
        # Extract block i states (skip first time step for forward-looking)
        # Note: EZ has shape (T+1, m) where first row is Z_0, so EZ[1:] is Z_1 to Z_T
        b_subset_current = slice(rp1, rp1 + r_i)  # Current factors only (no lags)
        b_subset_all = slice(rp1, rp1 + rp)  # All factors including lags
        
        Zsmooth_block = EZ[1:, b_subset_current]  # (T, r_i) - current factors
        Zsmooth_block_lag = EZ[:-1, b_subset_all]  # (T-1, rp) - lagged factors
        
        # Extract smoothed covariances for this block
        V_smooth_block = V_smooth[1:, b_subset_current, :][:, :, b_subset_current]  # (T, r_i, r_i)
        V_smooth_lag_block = V_smooth[:-1, b_subset_all, :][:, :, b_subset_all]  # (T-1, rp, rp)
        VVsmooth_block = VVsmooth[1:, b_subset_current, :][:, :, b_subset_all]  # (T-1, r_i, rp)
        
        def _compute_block_updates() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            # Use unified VAR estimation with smoothed expectations
            A_i, Q_i = estimate_var_unified(
                y=Zsmooth_block,  # Current factors (T x r_i)
                x=Zsmooth_block_lag,  # Lagged factors (T-1 x rp)
                V_smooth=V_smooth_block,  # Smoothed covariances for current
                VVsmooth=VVsmooth_block,  # Cross-covariances
                regularization=config.regularization,
                min_variance=config.min_variance,
                dtype=np.float32
            )
            
            # Ensure correct shape: A_i should be (r_i x rp)
            if A_i.shape != (r_i, rp):
                A_i_new = np.zeros((r_i, rp), dtype=np.float32)
                min_rows = min(A_i.shape[0], r_i)
                min_cols = min(A_i.shape[1], rp)
                A_i_new[:min_rows, :min_cols] = A_i[:min_rows, :min_cols]
                A_i = A_i_new
            
            # Return updated results for assignment
            return A_i, Q_i, V_smooth[0, t_start:t_end, t_start:t_end]
        
        updates = handle_linear_algebra_error(
            _compute_block_updates, f"block {i} update",
            fallback_func=lambda: None
        )
        if updates is not None:
            A_i, Q_i, V_0_block = updates
            # Place updated results in output matrix
            A_new[t_start:t_end, t_start:t_end] = DEFAULT_ZERO_VALUE  # Clear block
            A_new[t_start:t_start+r_i, t_start:t_start+rp] = A_i
            Q_new[t_start:t_end, t_start:t_end] = DEFAULT_ZERO_VALUE  # Clear block
            Q_new[t_start:t_start+r_i, t_start:t_start+r_i] = Q_i
            V_0_new[t_start:t_end, t_start:t_end] = V_0_block
    
    # Update idiosyncratic component parameters
    rp1 = int(np.sum(r) * p_plus_one)  # Column size of factor portion
    niM = int(np.sum(idio_indicator[:n_clock_freq]))  # Number of clock-frequency idiosyncratic components
    t_start = rp1  # Start of idiosyncratic component index
    i_subset = slice(t_start, t_start + niM)  # Indices for monthly idiosyncratic components
    
    if niM > 0:
        # Extract idiosyncratic states
        Zsmooth_idio = EZ[1:, i_subset]  # (T, niM)
        Zsmooth_idio_lag = EZ[:-1, i_subset]  # (T-1, niM)
        
        # Extract smoothed covariances for idiosyncratic components
        V_smooth_idio = V_smooth[1:, i_subset, :][:, :, i_subset]  # (T, niM, niM)
        
        def _compute_idiosyncratic_updates() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            # Use unified AR(1) estimation with smoothed expectations
            A_diag, Q_diag = estimate_ar1_unified(
                y=Zsmooth_idio,  # Current idio (T x niM)
                x=Zsmooth_idio_lag,  # Lagged idio (T-1 x niM)
                V_smooth=V_smooth_idio,  # Smoothed covariances
                regularization=config.regularization,
                min_variance=config.min_variance,
                default_ar_coef=DEFAULT_TRANSITION_COEF,
                default_noise=DEFAULT_PROCESS_NOISE,
                dtype=np.float32
            )
            
            # Return updated results for diagonal assignment
            return (np.diag(A_diag), np.diag(Q_diag), np.diag(np.diag(V_smooth[0, i_subset, i_subset])))
        
        updates = handle_linear_algebra_error(
            _compute_idiosyncratic_updates, "idiosyncratic component update",
            fallback_func=lambda: None
        )
        if updates is not None:
            A_diag_new, Q_diag_new, V_0_diag_new = updates
            A_new[i_subset, i_subset] = A_diag_new
            Q_new[i_subset, i_subset] = Q_diag_new
            V_0_new[i_subset, i_subset] = V_0_diag_new
    
    return A_new, Q_new, V_0_new


def _update_observation_matrix(X: np.ndarray, EZ: np.ndarray, EZZ: np.ndarray, C: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation matrix C using OLS regression."""
    def _compute_C() -> np.ndarray:
        N = X.shape[1]
        m = EZ.shape[1]
        X_clean = np.ma.filled(np.ma.masked_invalid(X), DEFAULT_CLEAN_NAN)
        sum_yEZ = X_clean.T @ EZ  # (N, m)
        sum_EZZ = np.sum(EZZ, axis=0) + create_scaled_identity(m, config.regularization)
        # sum_EZZ is already a covariance matrix, so use use_XTX=False
        C_new = solve_regularized_ols(sum_EZZ, sum_yEZ.T, regularization=DEFAULT_ZERO_VALUE, use_XTX=False).T
        # Normalize columns
        for j in range(m):
            norm = np.linalg.norm(C_new[:, j])
            if norm > config.min_norm:
                C_new[:, j] /= norm
        return C_new
    
    return handle_linear_algebra_error(
        _compute_C, "observation matrix update",
        fallback_value=C
    )


def _update_observation_matrix_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    C: np.ndarray,
    blocks: np.ndarray,
    r: np.ndarray,
    p_plus_one: int,  # Equal to max_lag_size = max(p + 1, tent_kernel_size)
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    n_clock_freq: int,
    n_slower_freq: int,
    idio_indicator: np.ndarray,
    tent_weights_dict: Optional[Dict[str, np.ndarray]],
    config: EMConfig,
    block_structure: Optional[BlockStructure] = None  # Optional: if provided, use cached indices
) -> np.ndarray:
    """Update observation matrix C block-by-block with tent kernel constraints.
    
    This function implements the block-by-block update for observation matrix (lines 438-523 in dfm.m).
    It handles clock-frequency series with standard OLS and slower-frequency series with tent kernel constraints.
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    EZ : np.ndarray
        Smoothed state means (T+1 x m), where first row is Z_0
    V_smooth : np.ndarray
        Smoothed state covariances (T+1 x m x m)
    C : np.ndarray
        Current observation matrix (N x m)
    blocks : np.ndarray
        Block structure array (N x n_blocks)
    r : np.ndarray
        Number of factors per block (n_blocks,)
    p_plus_one : int
        p + 1 (state dimension per factor)
    R_mat : np.ndarray, optional
        Tent kernel constraint matrix
    q : np.ndarray, optional
        Tent kernel constraint vector
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    n_slower_freq : int
        Number of slower-frequency series (series slower than clock frequency, generic)
    idio_indicator : np.ndarray
        Idiosyncratic component indicator (N,)
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights
    config : EMConfig
        EM configuration
        
    Returns
    -------
    C_new : np.ndarray
        Updated observation matrix
    """
    T, N = X.shape
    n_blocks = len(r)
    
    # Align blocks shape to match X (make a copy to avoid modifying original)
    blocks = _align_blocks_to_data(blocks, N)
    
    # Force blocks to have exactly N rows (defensive check)
    # This should already be done by _align_blocks_to_data, but ensure it
    if blocks.shape[0] != N:
        if blocks.shape[0] > N:
            blocks = blocks[:N, :].copy()
        else:
            # Pad if needed
            n_blocks_cols = blocks.shape[1]
            padding = np.zeros((N - blocks.shape[0], n_blocks_cols), dtype=blocks.dtype)
            blocks = np.vstack([blocks, padding])
    
    # Final assertion: blocks must have exactly N rows
    assert blocks.shape[0] == N, f"blocks must have {N} rows, got {blocks.shape[0]}"
    
    # Initialize output
    C_new = C.copy()
    
    # Use cached indices if available (computed once, reused across EM iterations)
    if block_structure is not None and block_structure.has_cached_indices():
        # Use cached indices (fast path - no recomputation)
        unique_blocks = block_structure._cached_unique_blocks
        unique_indices = block_structure._cached_unique_indices
        bl_idxM = block_structure._cached_bl_idxM
        bl_idxQ = block_structure._cached_bl_idxQ
        R_con = block_structure._cached_R_con
        q_con = block_structure._cached_q_con
        total_factor_dim = block_structure._cached_total_factor_dim
        idio_indicator_M = block_structure._cached_idio_indicator_M
        n_idio_M = block_structure._cached_n_idio_M
        c_idio_indicator = block_structure._cached_c_idio_indicator
        rp1 = block_structure._cached_rp1
    else:
        # Compute indices inline (fallback if not cached)
        # Find unique block patterns
        block_tuples = [tuple(row) for row in blocks]
        unique_blocks = []
        unique_indices = []
        seen = set()
        for i, bt in enumerate(block_tuples):
            if bt not in seen:
                unique_blocks.append(blocks[i].copy())
                unique_indices.append(i)
                seen.add(bt)
        
        n_bl = len(unique_blocks)
        
        # Diagnostic logging for block patterns
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(f"Block structure analysis: N={N}, n_blocks={n_blocks}, unique_patterns={n_bl}")
            for i, bl_i in enumerate(unique_blocks):
                n_series_in_pattern = sum(1 for row in blocks if np.array_equal(row, bl_i))
                active_blocks = [j for j, val in enumerate(bl_i) if val > 0]
                _logger.info(f"  Pattern {i}: bl_row={bl_i}, {n_series_in_pattern} series, active blocks: {active_blocks}")
        
        # Build block indices for clock-frequency and slower-frequency factors
        bl_idxM = []
        bl_idxQ = []
        R_con = None
        q_con = None
        
        # Calculate total factor state dimension
        total_factor_dim = int(np.sum(r) * p_plus_one)
        
        if R_mat is not None and q is not None:
            from scipy.linalg import block_diag
            R_con_blocks = []
            q_con_blocks = []
            
            # Build indices for each unique block pattern
            for bl_row in unique_blocks:
                bl_idxQ_row = []
                bl_idxM_row = []
                
                for block_idx in range(n_blocks):
                    if bl_row[block_idx] > 0:
                        bl_idxM_row.extend([True] * int(r[block_idx]))
                        bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                        bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                    else:
                        bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                        bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                
                bl_idxM.append(bl_idxM_row)
                bl_idxQ.append(bl_idxQ_row)
                
                # Build constraint matrix for blocks used in this pattern
                pattern_blocks = [block_idx for block_idx in range(n_blocks) if bl_row[block_idx] > 0]
                if pattern_blocks:
                    for block_idx in pattern_blocks:
                        R_con_blocks.append(np.kron(R_mat, create_scaled_identity(int(r[block_idx]), DEFAULT_IDENTITY_SCALE)))
                        q_con_blocks.append(np.zeros(R_mat.shape[0] * int(r[block_idx])))
            
            if R_con_blocks:
                R_con = block_diag(*R_con_blocks)
                q_con = np.concatenate(q_con_blocks)
        else:
            # No constraints - simpler indexing
            for bl_row in unique_blocks:
                bl_idxM_row = []
                bl_idxQ_row = []
                for block_idx in range(n_blocks):
                    if bl_row[block_idx] > 0:
                        bl_idxM_row.extend([True] * int(r[block_idx]))
                        bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                        bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                    else:
                        bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                        bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                bl_idxM.append(bl_idxM_row)
                bl_idxQ.append(bl_idxQ_row)
        
        # Convert to boolean arrays
        bl_idxM = [np.array(row, dtype=bool) for row in bl_idxM] if bl_idxM else []
        bl_idxQ = [np.array(row, dtype=bool) for row in bl_idxQ] if bl_idxQ else []
        
        # Diagnostic logging for bl_idxQ
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(f"bl_idxQ construction: len={len(bl_idxQ)}, total_factor_dim={total_factor_dim}")
            for i, bl_idxQ_i_arr in enumerate(bl_idxQ):
                n_true = np.sum(bl_idxQ_i_arr) if isinstance(bl_idxQ_i_arr, np.ndarray) else 0
                _logger.info(f"  Pattern {i}: bl_idxQ length={len(bl_idxQ_i_arr)}, True values={n_true}")
        
        # Idiosyncratic component indices
        idio_indicator_M = idio_indicator[:n_clock_freq]
        n_idio_M = int(np.sum(idio_indicator_M))
        c_idio_indicator = np.cumsum(idio_indicator)
        rp1 = int(np.sum(r) * p_plus_one)  # Start of idiosyncratic components
    
    # Handle missing data
    nanY = np.isnan(X)
    X_clean = np.where(nanY, DEFAULT_ZERO_VALUE, X)
    
    # Loop through unique block patterns
    for i, bl_i in enumerate(unique_blocks):
        # Find series indices matching this block pattern
        # blocks is already aligned to exactly N rows
        # Explicitly ensure we only compare first N rows
        n_rows_available = min(blocks.shape[0], N)
        blocks_compare = blocks[:n_rows_available, :]
        pattern_match = (blocks_compare == bl_i).all(axis=1)
        idx_i = np.where(pattern_match)[0]
        
        # Validate all indices are within bounds for X
        # pattern_match has length n_rows_available, so idx_i should only contain indices 0 to n_rows_available-1
        # But add extra validation to be safe
        max_valid_row_idx = min(n_rows_available, X.shape[1]) - 1
        idx_i = idx_i[(idx_i >= 0) & (idx_i <= max_valid_row_idx) & (idx_i < X.shape[1])].astype(int)
        
        # Filter to clock-frequency series
        idx_iM = idx_i[idx_i < n_clock_freq]
        n_i = len(idx_iM)
        
        if n_i == 0:
            continue
        
        # Count factors in this block pattern
        rs = int(np.sum(r[bl_i > 0]))
        
        # Get factor indices for this block pattern
        if i < len(bl_idxM) and len(bl_idxM[i]) > 0:
            bl_idxM_i = np.where(bl_idxM[i])[0]
        else:
            # Fallback: compute from block pattern
            bl_idxM_i = []
            offset = 0
            for block_idx in range(n_blocks):
                if bl_i[block_idx] > 0:
                    bl_idxM_i.extend(range(offset, offset + int(r[block_idx])))
                    offset += int(r[block_idx]) * p_plus_one
                else:
                    offset += int(r[block_idx]) * p_plus_one
            bl_idxM_i = np.array(bl_idxM_i)
        
        # idx_iM is already validated above (line 488), no need to re-validate
        
        # Initialize sums for equation 13 (BGR 2010)
        denom = np.zeros((n_i * rs, n_i * rs))
        nom = np.zeros((n_i, rs))
        
        # Idiosyncratic indices for clock-frequency series
        i_idio_i = idio_indicator_M[idx_iM]
        i_idio_ii = c_idio_indicator[idx_iM]
        i_idio_ii = i_idio_ii[i_idio_i > 0]
        
        # Update clock-frequency variables (VECTORIZED VERSION)
        # EZ has shape (T+1, m), so EZ[1:T+1] corresponds to times 0:T-1
        valid_times = np.arange(min(T, EZ.shape[0] - 1))
        valid_times = valid_times[valid_times < nanY.shape[0]]
        
        if len(valid_times) > 0:
            # Extract all Z_t and V_t at once
            Z_all = EZ[1:len(valid_times)+1, bl_idxM_i]  # (T_valid, rs)
            # Extract V_t for all valid times
            V_all = V_smooth[1:len(valid_times)+1][:, bl_idxM_i, :][:, :, bl_idxM_i]  # (T_valid, rs, rs)
            
            # Compute EZZ_t for all times: EZZ_t = Z_t @ Z_t' + V_t
            # Use einsum for batch outer product: 'ti,tj->tij'
            EZZ_all = np.einsum('ti,tj->tij', Z_all, Z_all) + V_all  # (T_valid, rs, rs)
            
            # Get non-missing indicators for all times: shape (T_valid, n_i)
            nan_mask_all = ~nanY[valid_times, :][:, idx_iM]  # (T_valid, n_i)
            # Pre-convert to float32 once to avoid repeated conversions in loop
            nan_mask_all_f32 = nan_mask_all.astype(np.float32)
            
            # Vectorized kron accumulation: kron(EZZ_t, Wt) where Wt = diag(~nanY[t, idx_iM])
            # kron(EZZ_t, Wt) = kron(EZZ_t, diag(w)) = block_diag([w[0]*EZZ_t, w[1]*EZZ_t, ...])
            # For each time t, we need to compute: sum over i of w[t,i] * kron(EZZ_t, e_i @ e_i')
            # This can be computed as: for each time t, denom += sum_i (w[t,i] * EZZ_t[i,i] ...)
            # Actually, kron(EZZ_t, diag(w)) = block_diag([w[0]*EZZ_t, ..., w[n_i-1]*EZZ_t])
            # So we can compute: for each time t, accumulate w[t,i] * EZZ_t for each i
            
            # FULLY VECTORIZED kron accumulation: kron(EZZ_t, diag(w_t)) = block_diag([w_t[0]*EZZ_t, ..., w_t[n_i-1]*EZZ_t])
            # kron(EZZ_t, diag(w_t)) creates (n_i*rs, n_i*rs) block diagonal where block i is w_t[i] * EZZ_t
            # We can vectorize by: for each (t, i), w_t[i] * EZZ_t -> block (i*rs:(i+1)*rs, i*rs:(i+1)*rs)
            # Using einsum: w_t[t, i] * EZZ_t[t, j, k] -> accumulate to denom[i*rs+j, i*rs+k]
            # Shape: w_t is (T_valid, n_i), EZZ_t is (T_valid, rs, rs)
            # Result: denom is (n_i*rs, n_i*rs) block diagonal
            for i in range(n_i):
                # For each block position i, accumulate: sum_t w_t[t, i] * EZZ_t[t, :, :]
                denom[i*rs:(i+1)*rs, i*rs:(i+1)*rs] += np.einsum('t,tjk->jk', nan_mask_all_f32[:, i], EZZ_all)
            
            # Vectorized nom accumulation: nom = sum_t outer(y_t, Z_t)
            y_all = X_clean[valid_times, :][:, idx_iM]  # (T_valid, n_i)
            # Compute outer products: einsum 'ti,tj->tij' for y_all and Z_all, then sum
            nom = np.einsum('ti,tj->ij', y_all, Z_all)  # (n_i, rs)
            
            # Subtract idiosyncratic component contribution (FULLY VECTORIZED)
            if len(i_idio_ii) > 0:
                idio_idx = (rp1 + i_idio_ii - 1).astype(int)
                idio_mask = i_idio_i > 0
                
                # Extract idiosyncratic states for all times
                Z_idio_all = EZ[1:len(valid_times)+1, idio_idx]  # (T_valid, n_idio)
                # Extract cross-covariances
                V_idio_all = V_smooth[1:len(valid_times)+1][:, idio_idx, :][:, :, bl_idxM_i]  # (T_valid, n_idio, rs)
                
                # Compute outer(Z_idio_t, Z_t) + V_idio_t for all times
                cross_products = np.einsum('ti,tj->tij', Z_idio_all, Z_all) + V_idio_all  # (T_valid, n_idio, rs)
                
                # FULLY VECTORIZED: Subtract sum_t (Wt[:, idio_mask] @ cross_products[t])
                # nan_mask_all is (T_valid, n_i), idio_mask is (n_i,) boolean
                # Extract missing data mask for idiosyncratic series only (reuse pre-converted float32)
                w_idio_all = nan_mask_all_f32[:, idio_mask]  # (T_valid, n_idio_valid)
                
                # Vectorized: sum_t (w_idio_t @ cross_products[t])
                # For each time t: w_idio_t is (n_idio_valid,), cross_products[t] is (n_idio_valid, rs)
                # Use einsum: 'ti,tij->j' where i is n_idio_valid, j is rs
                idio_contribution = np.einsum('ti,tij->j', w_idio_all, cross_products)  # (rs,)
                # Subtract from all rows of nom: (n_i, rs) -= (rs,) broadcasted
                nom[:, :] -= idio_contribution[np.newaxis, :]
        
        # Solve for loadings
        def _compute_clock_freq_loadings() -> np.ndarray:
            denom_reg = denom + create_scaled_identity(n_i * rs, config.regularization)
            # denom_reg is already a covariance matrix, so use use_XTX=False
            vec_C = solve_regularized_ols(denom_reg, nom.flatten(), regularization=DEFAULT_ZERO_VALUE, use_XTX=False)
            return vec_C.reshape(n_i, rs)
        
        loadings = handle_linear_algebra_error(
            _compute_clock_freq_loadings, f"clock-frequency block {i} update",
            fallback_func=lambda: None
        )
        if loadings is not None:
            # Use broadcasting for advanced indexing assignment
            # idx_iM[:, None] creates shape (n_i, 1) and bl_idxM_i creates shape (rs,)
            # This broadcasts to (n_i, rs) matching loadings.shape
            C_new[idx_iM[:, None], bl_idxM_i] = loadings
        
        # Update slower-frequency variables
        # Filter to slower-frequency series and validate indices
        # idx_i is already validated above to be < X.shape[1]
        # Filter to slower-frequency series and validate indices
        idx_iQ = idx_i[(idx_i >= n_clock_freq) & (idx_i < X.shape[1])]
        # Additional validation: ensure all indices are valid for nanY
        idx_iQ = idx_iQ[(idx_iQ >= 0) & (idx_iQ < nanY.shape[1])]
        
        if len(idx_iQ) > 0 and R_mat is not None and q is not None:
            rps = rs * p_plus_one
            
            # Get constraint matrix for this block
            if i < len(bl_idxQ) and len(bl_idxQ[i]) > 0:
                bl_idxQ_i = np.where(bl_idxQ[i])[0]
                if R_con is not None and q_con is not None:
                    R_con_i = R_con[:, bl_idxQ_i]
                    q_con_i = q_con.copy()
                    
                    # Remove zero rows
                    no_c = ~np.any(R_con_i, axis=1)
                    R_con_i = R_con_i[~no_c, :]
                    q_con_i = q_con_i[~no_c]
                else:
                    R_con_i = None
                    q_con_i = None
            else:
                bl_idxQ_i = []
                R_con_i = None
                q_con_i = None
            
            # Get tent kernel size from R_mat or tent_weights_dict (generalized)
            tent_kernel_size = None
            if R_mat is not None:
                tent_kernel_size = R_mat.shape[1]
            elif tent_weights_dict is not None and len(tent_weights_dict) > 0:
                # Use first available tent weights to determine size
                first_weights = next(iter(tent_weights_dict.values()))
                tent_kernel_size = len(first_weights)
            else:
                # Fallback: use default from config
                tent_kernel_size = config.tent_kernel_size
            
            # Get tent weights from tent_weights_dict (generalized for any frequency pair)
            # If multiple slower frequencies exist, use the first one (typically all use same tent structure)
            tent_weights = None
            if tent_weights_dict is not None and len(tent_weights_dict) > 0:
                # Use first available tent weights (all slower-frequency series typically use same structure)
                tent_weights = next(iter(tent_weights_dict.values()))
                if not isinstance(tent_weights, np.ndarray):
                    tent_weights = np.array(tent_weights, dtype=np.float32)
                # Update tent_kernel_size from actual weights
                tent_kernel_size = len(tent_weights)
            
            # Skip if bl_idxQ_i is empty - no factor states to update for this block
            # When bl_idxQ_i is empty, Z_all would have shape (T_valid, 0), leading to denom with shape (0, 0)
            # which causes broadcast error when adding create_scaled_identity(rps, ...)
            if len(bl_idxQ_i) == 0:
                # No block indices for slower-frequency factors in this block, skip all slower-frequency series
                # Continue to next block iteration
                pass
            else:
                # Loop through slower-frequency series (generic, works for any slower frequency)
                # idx_iQ is already filtered above
                for j in idx_iQ:
                    idx_jQ = j - n_clock_freq  # Ordinal position within slower-frequency series
                    
                    # Idiosyncratic component indices for slower-frequency series j
                    # Each slower-frequency series has tent_kernel_size clock-frequency factors
                    i_idio_jQ = np.arange(
                        rp1 + n_idio_M + tent_kernel_size * idx_jQ,
                        rp1 + n_idio_M + tent_kernel_size * (idx_jQ + 1)
                    )
                    
                    # Initialize sums
                    denom = np.zeros((rps, rps))
                    nom = np.zeros(rps)
                    
                    # VECTORIZED VERSION for slower-frequency series
                    # Get valid time indices
                    valid_times = np.arange(min(T, EZ.shape[0] - 1, V_smooth.shape[0] - 1))
                    valid_times = valid_times[(valid_times >= 0) & (valid_times < nanY.shape[0]) & (valid_times < X_clean.shape[0])]
                    
                    # Validate j is a valid index
                    if j < 0 or j >= nanY.shape[1] or j >= X_clean.shape[1]:
                        continue
                    
                    if len(valid_times) > 0:
                        # Extract all Z_t, V_t, y_t at once
                        # Ensure we don't go out of bounds
                        max_idx = min(len(valid_times) + 1, EZ.shape[0])
                        Z_all = EZ[1:max_idx, bl_idxQ_i]  # (T_valid, rps)
                        if Z_all.shape[0] != len(valid_times):
                            # Adjust valid_times if needed
                            valid_times = valid_times[:Z_all.shape[0]]
                        if len(valid_times) == 0:
                            continue
                        
                        V_all = V_smooth[1:len(valid_times)+1][:, bl_idxQ_i, :][:, :, bl_idxQ_i]  # (T_valid, rps, rps)
                        y_all = X_clean[valid_times, j]  # (T_valid,)
                        nan_mask_all = ~nanY[valid_times, j]  # (T_valid,)
                        
                        # Ensure 1D arrays (squeeze in case of unexpected dimensions)
                        y_all = np.squeeze(y_all)
                        nan_mask_all = np.squeeze(nan_mask_all)
                        if nan_mask_all.ndim > 1:
                            nan_mask_all = nan_mask_all.flatten()
                        if y_all.ndim > 1:
                            y_all = y_all.flatten()
                        
                        # Final validation: ensure shapes match
                        # Get minimum length to ensure all arrays are consistent
                        min_len = min(
                            len(valid_times),
                            nan_mask_all.shape[0],
                            y_all.shape[0],
                            Z_all.shape[0],
                            V_all.shape[0] if V_all.ndim >= 1 else len(valid_times)
                        )
                        
                        if min_len < len(valid_times):
                            # Adjust all arrays to match
                            valid_times = valid_times[:min_len]
                            nan_mask_all = nan_mask_all[:min_len]
                            y_all = y_all[:min_len]
                            Z_all = Z_all[:min_len]
                            V_all = V_all[:min_len]
                        
                        if len(valid_times) == 0:
                            continue
                        
                        # Compute EZZ_t for all times - now all arrays guaranteed to have matching first dimension
                        EZZ_all = np.einsum('ti,tj->tij', Z_all, Z_all) + V_all  # (T_valid, rps, rps)
                        
                        # Final check: ensure EZZ_all has correct shape
                        if EZZ_all.shape[0] != len(valid_times):
                            # This shouldn't happen, but handle gracefully
                            min_len_final = min(EZZ_all.shape[0], len(valid_times))
                            EZZ_all = EZZ_all[:min_len_final]
                            nan_mask_all = nan_mask_all[:min_len_final]
                            valid_times = valid_times[:min_len_final]
                            if len(valid_times) == 0:
                                continue
                        
                        # Pre-convert to float32 once to avoid repeated conversions
                        nan_mask_all_f32 = nan_mask_all.astype(np.float32)
                        
                        # Accumulate denom: kron(EZZ_t, Wt) where Wt = diag([~nanY[t,j]]) = scalar * I
                        # kron(EZZ_t, scalar*I) = scalar * EZZ_t (since kron(A, I) for scalar I is just A scaled)
                        # Actually, if Wt = diag([w]), then kron(EZZ_t, Wt) = w * EZZ_t (1x1 case)
                        denom = np.einsum('t,tij->ij', nan_mask_all_f32, EZZ_all)  # (rps, rps)
                        
                        # Accumulate nom: sum_t (y_t * Z_t * w_t)
                        nom = np.einsum('t,t,ti->i', y_all, nan_mask_all_f32, Z_all)  # (rps,)
                        
                        # Subtract idiosyncratic component contribution (vectorized)
                        if tent_weights is not None and len(i_idio_jQ) == len(tent_weights):
                            Z_idio_all = EZ[1:len(valid_times)+1, i_idio_jQ]  # (T_valid, tent_kernel_size)
                            V_idio_all = V_smooth[1:len(valid_times)+1][:, i_idio_jQ, :][:, :, bl_idxQ_i]  # (T_valid, tent_kernel_size, rps)
                            
                            # Compute tent_weights @ (outer(Z_idio_t, Z_t) + V_idio_t) for all times
                            cross_products = np.einsum('ti,tj->tij', Z_idio_all, Z_all) + V_idio_all  # (T_valid, tent_kernel_size, rps)
                            tent_weighted = np.einsum('i,tij->tj', tent_weights, cross_products)  # (T_valid, rps)
                            
                            # Subtract: nom -= sum_t (w_t * tent_weighted[t])
                            nom -= np.einsum('t,tj->j', nan_mask_all_f32, tent_weighted)  # (rps,)
                    
                    def _compute_slower_freq_loading() -> np.ndarray:
                        denom_reg = denom + create_scaled_identity(rps, config.regularization)
                        C_i_unconstrained = solve(denom_reg, nom, overwrite_a=False, overwrite_b=False, check_finite=False)
                        
                        # Apply tent kernel constraints
                        # Note: The unified function expects raw data or full smoothed expectations,
                        # but here we have pre-computed expectations (denom, nom).
                        # So we apply constraints directly using the same algorithm as the unified function
                        if R_con_i is not None and q_con_i is not None and len(R_con_i) > 0:
                            # Constrained OLS: C_i_constr = C_i - inv(denom) * R_con_i' * inv(R_con_i * inv(denom) * R_con_i') * (R_con_i * C_i - q_con_i)
                            # Optimized: use solve instead of inv
                            # Type assertion: q_con_i is guaranteed to be not None by the if condition
                            assert q_con_i is not None
                            constraint_term = R_con_i @ C_i_unconstrained - q_con_i
                            
                            # Solve: denom_reg @ temp1 = R_con_i.T  (instead of computing denom_inv @ R_con_i.T)
                            R_con_denom_inv = solve(denom_reg, R_con_i.T, overwrite_a=False, overwrite_b=False, check_finite=False)
                            R_con_denom = R_con_i @ R_con_denom_inv
                            R_con_denom_reg = R_con_denom + create_scaled_identity(len(R_con_denom), config.regularization)
                            
                            # Solve: R_con_denom_reg @ temp2 = constraint_term
                            temp2 = solve(R_con_denom_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
                            
                            # C_i_constr = C_i_unconstrained - R_con_denom_inv @ temp2
                            C_i_constr = C_i_unconstrained - R_con_denom_inv @ temp2
                        else:
                            C_i_constr = C_i_unconstrained
                        return C_i_constr
                    
                    loading = handle_linear_algebra_error(
                        _compute_slower_freq_loading, f"slower-frequency series {j} update",
                        fallback_func=lambda: None
                    )
                    if loading is not None:
                        C_new[j, bl_idxQ_i] = loading
    
    return C_new


def _update_process_noise(EZ: np.ndarray, A_new: np.ndarray, Q: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update process noise covariance Q from residuals."""
    T, m = EZ.shape
    if T <= 1:
        return Q
    
    residuals = EZ[1:, :] - EZ[:-1, :] @ A_new.T
    if m == 1:
        Q_new = np.array([[np.var(residuals, axis=0)]])
    else:
        Q_new = np.cov(residuals.T)
    Q_new = ensure_process_noise_stable(Q_new, min_eigenval=config.min_variance, warn=True, dtype=np.float32)
    return np.maximum(Q_new, create_scaled_identity(m, config.min_variance))


def _update_observation_noise(X: np.ndarray, EZ: np.ndarray, C_new: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation noise covariance R (diagonal) from residuals."""
    X_clean = np.ma.filled(np.ma.masked_invalid(X), DEFAULT_CLEAN_NAN)
    residuals = X_clean - EZ @ C_new.T
    diag_R = np.var(residuals, axis=0)
    diag_R = np.clip(diag_R, config.min_variance, config.max_variance)
    R_new = np.diag(diag_R)
    return ensure_covariance_stable(R_new, min_eigenval=config.min_variance)


def _update_observation_noise_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    C_new: np.ndarray,
    R: np.ndarray,
    idio_indicator: np.ndarray,
    n_clock_freq: int,
    config: EMConfig
) -> np.ndarray:
    """Update observation noise covariance R with missing data handling.
    
    This function implements the observation noise update with selection matrices
    for missing data (lines 526-541 in dfm.m).
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    EZ : np.ndarray
        Smoothed state means (T+1 x m)
    V_smooth : np.ndarray
        Smoothed state covariances (T+1 x m x m)
    C_new : np.ndarray
        Updated observation matrix (N x m)
    R : np.ndarray
        Current observation noise covariance (N x N)
    idio_indicator : np.ndarray
        Idiosyncratic component indicator (N,)
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    config : EMConfig
        EM configuration
        
    Returns
    -------
    R_new : np.ndarray
        Updated observation noise covariance (diagonal)
    """
    T, N = X.shape
    
    # Handle missing data
    nanY = np.isnan(X)
    X_clean = np.where(nanY, DEFAULT_ZERO_VALUE, X)
    
    # Use unified variance estimation with smoothed expectations
    # Note: The unified function computes R from X, EZ, C, V_smooth
    # But we need to handle missing data with selection matrices, so we compute manually
    # and then use the unified function for the final variance computation
    
    # Initialize covariance of residuals
    R_new = np.zeros((N, N))
    
    # Update using selection matrices (BGR equation 15) - VECTORIZED VERSION
    # EZ has shape (T+1, m) where first row is Z_0, so EZ[1:T+1] corresponds to times 0:T-1
    valid_times = np.arange(min(T, EZ.shape[0] - 1))
    valid_times = valid_times[valid_times < X_clean.shape[0]]
    
    if len(valid_times) > 0:
        # Extract all Z_t and V_t at once
        Z_all = EZ[1:len(valid_times)+1, :]  # (T_valid, m)
        V_all = V_smooth[1:len(valid_times)+1, :, :]  # (T_valid, m, m)
        X_all = X_clean[valid_times, :]  # (T_valid, N)
        nan_mask_all = ~nanY[valid_times, :]  # (T_valid, N)
        
        # Compute residuals for all times: y_t - Wt * C_new * Z_t
        # Wt = diag(~nanY[t, :]), so Wt @ C_new @ Z_t = (~nanY[t, :]) * (C_new @ Z_t)
        CZ_all = np.einsum('ij,tj->ti', C_new, Z_all)  # (T_valid, N)
        residuals = X_all - nan_mask_all * CZ_all  # (T_valid, N)
        
        # Accumulate outer(residual, residual) for all times
        R_new += np.einsum('ti,tj->ij', residuals, residuals)  # (N, N)
        
        # FULLY VECTORIZED: Accumulate Wt @ C_new @ V_t @ C_new' @ Wt for all times
        # Wt = diag(w_t), so: diag(w_t) @ CVCT @ diag(w_t) = w_t[i] * w_t[j] * CVCT[i, j] (element-wise)
        # Shape: w_t is (T_valid, N), CVCT_all is (T_valid, N, N)
        CVCT_all = np.einsum('ij,tjk,kl->til', C_new, V_all, C_new.T)  # (T_valid, N, N)
        # Pre-convert to float32 once to avoid duplicate conversion
        nan_mask_all_f32 = nan_mask_all.astype(np.float32)
        # Vectorized: sum_t (w_t[t, i] * w_t[t, j] * CVCT_all[t, i, j])
        R_new += np.einsum('ti,tj,tij->ij', nan_mask_all_f32, nan_mask_all_f32, CVCT_all)
        
        # FULLY VECTORIZED: Accumulate (I - Wt) @ R @ (I - Wt) for all times
        # (I - Wt) @ R @ (I - Wt) where Wt = diag(w_t)
        # = (I - diag(w_t)) @ R @ (I - diag(w_t))
        # Element (i,j): (1 - w_t[i]) * R[i,j] * (1 - w_t[j])
        I_minus_W_all = 1.0 - nan_mask_all_f32  # (T_valid, N)
        # Vectorized: sum_t (1 - w_t[t, i]) * R[i, j] * (1 - w_t[t, j])
        R_new += np.einsum('ti,ij,tj->ij', I_minus_W_all, R, I_minus_W_all)
    
    R_new = R_new / T
    
    # Extract diagonal and set minimum values using unified variance function
    # For smoothed expectations, we pass the computed R_new as "residuals" (diagonal extraction)
    RR_diag = np.diag(R_new)
    RR_diag = np.maximum(RR_diag, config.min_variance)
    RR_diag = np.where(np.isfinite(RR_diag), RR_diag, config.min_variance)
    
    # Ensure non-zero measurement error for clock-frequency idiosyncratic components
    # idio_indicator has length N, so use it directly for indexing RR_diag
    idio_indicator_M_mask = idio_indicator[:n_clock_freq] > 0
    RR_diag[:n_clock_freq][idio_indicator_M_mask] = np.maximum(RR_diag[:n_clock_freq][idio_indicator_M_mask], MIN_OBSERVATION_NOISE)
    
    # Ensure non-zero for slower-frequency series
    if n_clock_freq < N:
        RR_diag[n_clock_freq:] = np.maximum(RR_diag[n_clock_freq:], MIN_OBSERVATION_NOISE)
    
    # Clip to reasonable range
    RR_diag = np.clip(RR_diag, config.min_variance, config.max_variance)
    
    R_new = np.diag(RR_diag)
    return ensure_covariance_stable(R_new, min_eigenval=config.min_variance)


def em_step(
    X: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    kalman_filter: Optional[DFMKalmanFilter] = None,
    config: Optional[EMConfig] = None,
    block_structure: Optional[BlockStructure] = None,
    num_iter: int = 0  # For timing logs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Optional[DFMKalmanFilter]]:
    """Perform one EM step: pykalman E-step + custom M-step with block constraints.
    
    **Why not use pykalman's built-in `kf.em()`?**
    
    pykalman's `em()` method does unconstrained EM updates that would:
    1. Destroy block structure (factors organized in blocks)
    2. Break mixed-frequency constraints (tent kernel aggregation)
    3. Ignore idiosyncratic component structure
    
    **Our approach:**
    - E-step: Uses pykalman's Kalman filter/smoother (via DFMKalmanFilter wrapper)
      - Handles missing data via masked arrays
      - Provides smoothed state estimates E[Z_t] and covariances
    - M-step: Custom constrained OLS that preserves:
      - Block structure (block-specific loadings)
      - Mixed-frequency constraints (tent kernel aggregation)
      - Idiosyncratic components (per-series state augmentation)
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    A, C, Q, R, Z_0, V_0 : np.ndarray
        Current model parameters
    kalman_filter : DFMKalmanFilter, optional
        Existing Kalman filter instance. If None, creates a new one.
    config : EMConfig, optional
        EM configuration. If None, uses defaults.
    block_structure : BlockStructure, optional
        Block structure configuration. If provided and valid, uses blocked updates.
        
    Returns
    -------
    A_new, C_new, Q_new, R_new, Z_0_new, V_0_new : np.ndarray
        Updated parameters
    loglik : float
        Log-likelihood value
    kalman_filter : DFMKalmanFilter
        Updated Kalman filter instance
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    # Create or update Kalman filter
    if kalman_filter is None:
        kalman_filter = DFMKalmanFilter(
            transition_matrices=A, observation_matrices=C,
            transition_covariance=Q, observation_covariance=R,
            initial_state_mean=Z_0, initial_state_covariance=V_0
        )
    else:
        kalman_filter.update_parameters(A, C, Q, R, Z_0, V_0)
    
    # E-step: pykalman handles missing data via masked arrays
    # E-step consists of: filter (forward pass) + smooth (backward pass)
    # Both are O(T × m³) complexity, but smooth is the actual bottleneck:
    # - Filter: ~1s (highly optimized BLAS/LAPACK with threading)
    # - Smooth: ~5-7 minutes (less optimized, backward iteration)
    # For T=2135, m=183: ~13 billion operations per iteration
    import time as time_module
    e_step_start = time_module.time()
    X_masked = np.ma.masked_invalid(X)
    
    # Log E-step info (progress indicators will show filter and smooth progress)
    verbose_iterations = num_iter < 5
    if verbose_iterations:
        T, m = X.shape[0], kalman_filter._pykalman.transition_matrices.shape[0] if kalman_filter._pykalman.transition_matrices is not None else 0
        N = X.shape[1]
        ops_estimate = T * (m ** 3) / 1e9  # Billion operations estimate
        _logger.info(f"    E-step: Running Kalman filter + smoother (T={T}, N={N}, m={m}, ~{ops_estimate:.1f}B ops)...")
        _logger.info(f"    E-step: Filter is fast (~1s), but smooth may take 5-7 minutes (bottleneck)")
    EZ, V_smooth, VVsmooth, loglik = kalman_filter.filter_and_smooth(X_masked)
    
    # Keep float64 for numerical stability (prevents V_smooth accumulation overflow)
    # Do NOT convert to float32 - maintain float64 precision throughout M-step
    # Float64 is critical for large state spaces (m=183) and long time series (T=2135)
    EZ = EZ.astype(np.float64) if EZ.dtype != np.float64 else EZ
    V_smooth = V_smooth.astype(np.float64) if V_smooth.dtype != np.float64 else V_smooth
    VVsmooth = VVsmooth.astype(np.float64) if VVsmooth.dtype != np.float64 else VVsmooth
    
    e_step_time = time_module.time() - e_step_start
    
    # Always log E-step completion (user needs to see when E-step actually finishes)
    _logger.info(f"    E-step: Completed in {e_step_time:.1f}s, log-likelihood={loglik:.2e}")
    
    # M-step: Use blocked updates if block structure is provided, otherwise use simple updates
    m_step_start = time_module.time()
    
    # Compute and cache block structure indices once (computed once, reused across EM iterations)
    if block_structure is not None and block_structure.is_valid():
        if not block_structure.has_cached_indices():
            # Compute indices once before first M-step (N is number of series/columns in X)
            N = X.shape[1]
            _compute_and_cache_block_indices(block_structure, N)
    
    # Log M-step start for first few iterations
    if verbose_iterations:
        if block_structure is not None:
            n_blocks = len(block_structure.r) if hasattr(block_structure, 'r') and block_structure.r is not None else 0
            _logger.info(f"    M-step: Updating parameters (block structure: {n_blocks} blocks)...")
        else:
            _logger.info(f"    M-step: Updating parameters (unconstrained)...")
    
    if block_structure is not None and block_structure.is_valid():
        # Blocked updates (matching Nowcasting MATLAB)
        n_blocks = len(block_structure.r) if hasattr(block_structure, 'r') and block_structure.r is not None else 0
        n_series = X.shape[1]
        
        if verbose_iterations:
            _logger.info(f"      → Updating transition matrix A and process noise Q...")
        A_new, Q_new, V_0_new = _update_transition_matrix_blocked(
            EZ, V_smooth, VVsmooth, A, Q, block_structure.blocks, block_structure.r,
            block_structure.p, block_structure.p_plus_one, block_structure.idio_indicator,
            block_structure.n_clock_freq, config
        )
        
        # CRITICAL: Cap Q_new after assembly to ensure full matrix stability
        # Individual blocks are capped, but assembled Q_new may still exceed limits
        # due to idiosyncratic components or numerical issues during assembly
        Q_new = ensure_process_noise_stable(Q_new, min_eigenval=config.min_variance, warn=True, dtype=np.float32)
        
        # Blocked observation matrix update
        if verbose_iterations:
            _logger.info(f"      → Updating observation matrix C...")
        C_new = _update_observation_matrix_blocked(
            X, EZ, V_smooth, C, block_structure.blocks, block_structure.r,
            block_structure.p_plus_one, block_structure.R_mat, block_structure.q,
            block_structure.n_clock_freq, block_structure.n_slower_freq or 0,
            block_structure.idio_indicator, block_structure.tent_weights_dict, config,
            block_structure=block_structure  # Pass block_structure to use cached indices
        )
        
        # Blocked observation noise update
        if verbose_iterations:
            _logger.info(f"      → Updating observation noise R...")
        R_new = _update_observation_noise_blocked(
            X, EZ, V_smooth, C_new, R, block_structure.idio_indicator,
            block_structure.n_clock_freq, config
        )
        
        # Update initial state mean
        Z_0_new = EZ[0, :] if EZ.shape[0] > 0 else Z_0
    else:
        # Simple unconstrained updates (backward compatibility)
        # Compute smoothed factor covariances
        EZZ = V_smooth + np.einsum('ti,tj->tij', EZ, EZ)  # (T, m, m)
        
        A_new = _update_transition_matrix(EZ, A, config)
        C_new = _update_observation_matrix(X, EZ, EZZ, C, config)
        Q_new = _update_process_noise(EZ, A_new, Q, config)
        R_new = _update_observation_noise(X, EZ, C_new, config)
        
        # Update initial state
        Z_0_new = EZ[0, :] if EZ.shape[0] > 0 else Z_0
        V_0_new = ensure_covariance_stable(V_smooth[0] if len(V_smooth) > 0 else V_0, min_eigenval=config.min_variance)
    
    m_step_time = time_module.time() - m_step_start
    
    # Log M-step completion for first few iterations
    if num_iter < 5:
        # Log parameter statistics
        A_max = np.max(np.abs(A_new)) if np.isfinite(A_new).all() else np.nan
        C_max = np.max(np.abs(C_new)) if np.isfinite(C_new).all() else np.nan
        Q_max_elem = np.max(np.abs(Q_new)) if np.isfinite(Q_new).all() else np.nan
        # Also check Q max eigenvalue to verify capping is working
        try:
            if np.isfinite(Q_new).all() and Q_new.size > 0:
                Q_eigenvals = np.linalg.eigvalsh(Q_new)
                Q_max_eigval = np.max(np.abs(Q_eigenvals))
            else:
                Q_max_eigval = np.nan
        except (np.linalg.LinAlgError, ValueError):
            Q_max_eigval = np.nan
        R_diag_max = np.max(np.abs(np.diag(R_new))) if np.isfinite(R_new).all() else np.nan
        _logger.info(f"    M-step: Completed in {m_step_time:.1f}s | "
                    f"Max values: |A|={A_max:.3f}, |C|={C_max:.3f}, |Q|={Q_max_elem:.3f} (max_eig={Q_max_eigval:.3f}), |R_diag|={R_diag_max:.3f}")
    
    # Log timing breakdown for all iterations (or at least frequent logging)
    total_time = e_step_time + m_step_time
    # Log timing: always first 3 iterations, then every 5 iterations, or if iteration is slow (>30s)
    should_log_timing = (num_iter < 3) or (num_iter % 5 == 0) or (total_time > 30.0)
    
    if should_log_timing:
        T, m_dim = X.shape[0], EZ.shape[1] if EZ.shape else 0
        e_step_pct = 100*e_step_time/total_time if total_time > 0 else 0
        m_step_pct = 100*m_step_time/total_time if total_time > 0 else 0
        _logger.info(f"  Iteration {num_iter + 1} timing: E-step={e_step_time:.2f}s ({e_step_pct:.1f}%), "
                    f"M-step={m_step_time:.2f}s ({m_step_pct:.1f}%), "
                    f"Total={total_time:.2f}s (T={T}, m={m_dim})")
    
    return A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, kalman_filter


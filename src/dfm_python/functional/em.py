"""EM algorithm implementation for DFM.

This module provides the Expectation-Maximization algorithm for DFM parameter estimation.
Uses pykalman for the E-step (Kalman filter/smoother) and implements the M-step
with block structure preservation.

Includes numerical stability utilities to ensure convergence safety.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass

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
        p + 1 (state dimension per factor)
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
    p_plus_one: int,
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    n_clock_freq: int,
    n_slower_freq: int,
    idio_indicator: np.ndarray,
    tent_weights_dict: Optional[Dict[str, np.ndarray]],
    config: EMConfig
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
    
    # Find unique block patterns
    # Convert blocks to tuples for hashing (use aligned blocks)
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
    
    # Build block indices for clock-frequency and slower-frequency factors
    # bl_idxM and bl_idxQ are built per unique block pattern, not per block
    bl_idxM = []  # Indicator for clock-frequency factor loadings (per unique block pattern)
    bl_idxQ = []  # Indicator for slower-frequency factor loadings (per unique block pattern)
    R_con = None  # Block diagonal constraint matrix
    q_con = None  # Constraint vector
    
    # Calculate total factor state dimension
    total_factor_dim = int(np.sum(r) * p_plus_one)
    
    if R_mat is not None and q is not None:
        from scipy.linalg import block_diag
        R_con_blocks = []
        q_con_blocks = []
        
        # Build indices for each unique block pattern
        for bl_row in unique_blocks:
            # bl_idxQ: all factors for this block pattern (including lags/tent chain)
            bl_idxQ_row = []
            # bl_idxM: only current factors for this block pattern (no lags)
            bl_idxM_row = []
            
            offset = 0
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    # This block is active in this pattern
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                    offset += int(r[block_idx]) * p_plus_one
                else:
                    # This block is not active in this pattern
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    offset += int(r[block_idx]) * p_plus_one
            
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
        
        # Update clock-frequency variables
        for t in range(T):
            # Safety check: ensure t is within valid range for nanY
            if t < 0 or t >= nanY.shape[0]:
                continue
            # Selection matrix for non-missing values
            # Final defensive check: ensure idx_iM indices are valid for nanY
            # Filter out any indices that are out of bounds
            max_valid_col_idx = nanY.shape[1] - 1
            # Use boolean indexing to filter out invalid indices
            is_valid = (idx_iM >= 0) & (idx_iM <= max_valid_col_idx)
            if not np.any(is_valid):
                continue
            idx_iM_valid = idx_iM[is_valid]
            # Additional safety: ensure no out-of-bounds indices remain
            # Double-check bounds before accessing nanY
            idx_iM_valid = idx_iM_valid[(idx_iM_valid >= 0) & (idx_iM_valid < nanY.shape[1])]
            if len(idx_iM_valid) == 0:
                continue
            # nanY has shape (T, N), so index as nanY[t, idx_iM_valid] not nanY[idx_iM_valid, t]
            try:
                Wt = np.diag(~nanY[t, idx_iM_valid])
            except IndexError:
                # If this still fails, there's a deeper issue - log and skip
                _logger.warning(f"IndexError in Wt computation: idx_iM_valid={idx_iM_valid}, nanY.shape={nanY.shape}, t={t}")
                continue
            
            # E[f_t*f_t' | Omega_T]
            # EZ has shape (T+1, m), so EZ[t+1] corresponds to time t (t+1 must be <= T)
            if t + 1 >= EZ.shape[0]:
                continue
            Z_t = EZ[t+1, bl_idxM_i]  # (rs,)
            if t + 1 >= V_smooth.shape[0]:
                continue
            V_t = V_smooth[t+1, bl_idxM_i, :][:, bl_idxM_i]  # (rs, rs)
            EZZ_t = np.outer(Z_t, Z_t) + V_t
            
            denom += np.kron(EZZ_t, Wt)
            
            # E[y_t*f_t' | Omega_T]
            y_t = X_clean[t, idx_iM]  # (n_i,)
            nom += np.outer(y_t, Z_t)
            
            # Subtract idiosyncratic component contribution
            if len(i_idio_ii) > 0 and t + 1 < EZ.shape[0] and t + 1 < V_smooth.shape[0]:
                idio_idx = (rp1 + i_idio_ii - 1).astype(int)  # Convert to 0-based, ensure integer type
                Z_idio_t = EZ[t+1, idio_idx]  # (n_idio,)
                V_idio_t = V_smooth[t+1, idio_idx, :][:, bl_idxM_i]  # (n_idio, rs)
                nom -= Wt[:, i_idio_i > 0] @ (np.outer(Z_idio_t, Z_t) + V_idio_t)
        
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
                
                for t in range(T):
                    # Safety check: ensure t is within valid range for nanY
                    if t < 0 or t >= nanY.shape[0]:
                        continue
                    # Selection matrix
                    # Final safety check: ensure j is valid for nanY
                    # nanY has shape (T, N), so j is column index (series), t is row index (time)
                    if j < 0 or j >= nanY.shape[1]:
                        continue
                    try:
                        Wt = np.diag([~nanY[t, j]])
                    except IndexError:
                        # Skip this iteration if j is out of bounds
                        continue
                    
                    # E[f_t*f_t' | Omega_T]
                    # EZ has shape (T+1, m), so EZ[t+1] corresponds to time t (t+1 must be <= T)
                    if t + 1 >= EZ.shape[0] or t + 1 >= V_smooth.shape[0]:
                        continue
                    Z_t = EZ[t+1, bl_idxQ_i]  # (rps,)
                    V_t = V_smooth[t+1, bl_idxQ_i, :][:, bl_idxQ_i]  # (rps, rps)
                    EZZ_t = np.outer(Z_t, Z_t) + V_t
                    
                    denom += np.kron(EZZ_t, Wt)
                    
                    # E[y_t*f_t' | Omega_T]
                    y_t = X_clean[t, j]
                    nom += y_t * Z_t
                    
                    # Subtract idiosyncratic component contribution
                    if tent_weights is not None and len(i_idio_jQ) == len(tent_weights) and t + 1 < EZ.shape[0] and t + 1 < V_smooth.shape[0]:
                        Z_idio_t = EZ[t+1, i_idio_jQ]  # (tent_kernel_size,)
                        V_idio_t = V_smooth[t+1, i_idio_jQ, :][:, bl_idxQ_i]  # (tent_kernel_size, rps)
                        nom -= Wt[0, 0] * (tent_weights @ (np.outer(Z_idio_t, Z_t) + V_idio_t))
                
                def _compute_slower_freq_loading() -> np.ndarray:
                    denom_reg = denom + create_scaled_identity(rps, config.regularization)
                    C_i_unconstrained = np.linalg.solve(denom_reg, nom)
                    
                    # Apply tent kernel constraints
                    # Note: The unified function expects raw data or full smoothed expectations,
                    # but here we have pre-computed expectations (denom, nom).
                    # So we apply constraints directly using the same algorithm as the unified function
                    if R_con_i is not None and q_con_i is not None and len(R_con_i) > 0:
                        # Constrained OLS: C_i_constr = C_i - inv(denom) * R_con_i' * inv(R_con_i * inv(denom) * R_con_i') * (R_con_i * C_i - q_con_i)
                        denom_inv = np.linalg.inv(denom_reg)
                        R_con_denom = R_con_i @ denom_inv @ R_con_i.T
                        R_con_denom_inv = np.linalg.inv(R_con_denom + create_scaled_identity(len(R_con_denom), config.regularization))
                        # Type assertion: q_con_i is guaranteed to be not None by the if condition
                        assert q_con_i is not None
                        constraint_term = R_con_i @ C_i_unconstrained - q_con_i
                        C_i_constr = C_i_unconstrained - denom_inv @ R_con_i.T @ R_con_denom_inv @ constraint_term
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
    Q_new = ensure_covariance_stable(Q_new, min_eigenval=config.min_variance)
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
    
    # Update using selection matrices (BGR equation 15)
    # EZ has shape (T+1, m) where first row is Z_0, so EZ[t+1] corresponds to time t
    for t in range(T):
        if t + 1 >= EZ.shape[0]:
            continue  # Skip if EZ doesn't have enough rows
        Wt = np.diag(~nanY[t, :])  # Selection matrix: diagonal of non-missing indicators for time t
        
        # Residual: y_t - Wt * C_new * Z_{t+1}
        Z_t = EZ[t+1, :]  # (m,)
        residual = X_clean[t, :] - Wt @ C_new @ Z_t  # (N,)
        
        # R_new += residual * residual' + Wt * C_new * V_{t+1} * C_new' * Wt + (I - Wt) * R * (I - Wt)
        R_new += np.outer(residual, residual)
        if t + 1 < V_smooth.shape[0]:
            R_new += Wt @ C_new @ V_smooth[t+1] @ C_new.T @ Wt
        I_N = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE)
        R_new += (I_N - Wt) @ R @ (I_N - Wt)
    
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
    block_structure: Optional[BlockStructure] = None
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
    X_masked = np.ma.masked_invalid(X)
    EZ, V_smooth, VVsmooth, loglik = kalman_filter.filter_and_smooth(X_masked)
    
    # M-step: Use blocked updates if block structure is provided, otherwise use simple updates
    if block_structure is not None and block_structure.is_valid():
        # Blocked updates (matching Nowcasting MATLAB)
        A_new, Q_new, V_0_new = _update_transition_matrix_blocked(
            EZ, V_smooth, VVsmooth, A, Q, block_structure.blocks, block_structure.r,
            block_structure.p, block_structure.p_plus_one, block_structure.idio_indicator,
            block_structure.n_clock_freq, config
        )
        
        # Blocked observation matrix update
        C_new = _update_observation_matrix_blocked(
            X, EZ, V_smooth, C, block_structure.blocks, block_structure.r,
            block_structure.p_plus_one, block_structure.R_mat, block_structure.q,
            block_structure.n_clock_freq, block_structure.n_slower_freq or 0,
            block_structure.idio_indicator, block_structure.tent_weights_dict, config
        )
        
        # Blocked observation noise update
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
    
    return A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, kalman_filter


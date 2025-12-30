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
from ..config.constants import (
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
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
    DEFAULT_MAX_VARIANCE,
)
from ..numeric.stability import (
    ensure_positive_definite,
    cap_max_eigenval,
    ensure_covariance_stable,
)

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
    convergence_log_interval: int = 10
    progress_log_interval: int = 5
    small_loglik_threshold: float = 1e-10
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    # Initialization constants (used by DFM initialization)
    default_transition_coef: float = DEFAULT_TRANSITION_COEF
    default_process_noise: float = DEFAULT_PROCESS_NOISE
    default_observation_noise: float = MIN_DIAGONAL_VARIANCE
    matrix_regularization: float = DEFAULT_REGULARIZATION
    eigenval_floor: float = MIN_EIGENVALUE
    slower_freq_ar_coef: float = DEFAULT_SLOWER_FREQ_AR_COEF  # AR coefficient for slower-frequency idiosyncratic components
    tent_kernel_size: int = 5
    slower_freq_variance_denominator: float = DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR  # Variance denominator for slower-frequency series
    extreme_forecast_threshold: float = DEFAULT_EXTREME_FORECAST_THRESHOLD


_DEFAULT_EM_CONFIG = EMConfig()


def _update_transition_matrix(EZ: np.ndarray, A: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update transition matrix A using OLS regression."""
    T, m = EZ.shape
    if T <= 1:
        return A
    
    try:
        Y = EZ[1:, :]  # (T-1, m)
        X = EZ[:-1, :]  # (T-1, m)
        XTX = X.T @ X + np.eye(m) * config.regularization
        XTY = X.T @ Y
        A_new = np.linalg.solve(XTX, XTY).T
        return cap_max_eigenval(A_new, max_eigenval=config.max_eigenval, symmetric=False, warn=False)
    except (np.linalg.LinAlgError, ValueError):
        return A


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
    V_0_new = V_smooth[0].copy() if len(V_smooth) > 0 else np.eye(m) * config.min_variance
    
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
        
        # E[f_t*f_t' | Omega_T] - equation from Nowcasting
        EZZ = Zsmooth_block.T @ Zsmooth_block + np.sum(V_smooth[1:, b_subset_current, :][:, :, b_subset_current], axis=0)
        
        # E[f_{t-1}*f_{t-1}' | Omega_T]
        EZZ_BB = Zsmooth_block_lag.T @ Zsmooth_block_lag + np.sum(V_smooth[:-1, b_subset_all, :][:, :, b_subset_all], axis=0)
        
        # E[f_t*f_{t-1}' | Omega_T]
        # VVsmooth[t] is covariance between state at t and t-1
        EZZ_FB = Zsmooth_block[1:].T @ Zsmooth_block_lag + np.sum(VVsmooth[1:, b_subset_current, :][:, :, b_subset_all], axis=0)
        
        # Regularize EZZ_BB
        EZZ_BB_reg = EZZ_BB + np.eye(rp) * config.regularization
        
        try:
            # Equation 6: Estimate VAR(p) for factor
            A_i = np.linalg.solve(EZZ_BB_reg.T, EZZ_FB.T).T  # (r_i, rp)
            
            # Equation 8: Covariance matrix of residuals of VAR
            Q_i = (EZZ - A_i @ EZZ_FB.T) / T  # (r_i, r_i)
            
            # Ensure Q_i is positive definite
            Q_i = ensure_covariance_stable(Q_i, min_eigenval=config.min_variance)
            
            # Place updated results in output matrix
            A_new[t_start:t_end, t_start:t_end] = 0.0  # Clear block
            A_new[t_start:t_start+r_i, t_start:t_start+rp] = A_i
            Q_new[t_start:t_end, t_start:t_end] = 0.0  # Clear block
            Q_new[t_start:t_start+r_i, t_start:t_start+r_i] = Q_i
            V_0_new[t_start:t_end, t_start:t_end] = V_smooth[0, t_start:t_end, t_start:t_end]
        except (np.linalg.LinAlgError, ValueError) as e:
            _logger.warning(f"Block {i} update failed: {e}. Keeping previous values.")
    
    # Update idiosyncratic component parameters
    rp1 = int(np.sum(r) * p_plus_one)  # Column size of factor portion
    niM = int(np.sum(idio_indicator[:n_clock_freq]))  # Number of clock-frequency idiosyncratic components
    t_start = rp1  # Start of idiosyncratic component index
    i_subset = slice(t_start, t_start + niM)  # Indices for monthly idiosyncratic components
    
    if niM > 0:
        # Extract idiosyncratic states
        Zsmooth_idio = EZ[1:, i_subset]  # (T-1, niM)
        Zsmooth_idio_lag = EZ[:-1, i_subset]  # (T-1, niM)
        
        # E[f_t*f_t' | Omega_T] - diagonal only
        EZZ = np.diag(np.diag(Zsmooth_idio.T @ Zsmooth_idio)) + np.diag(np.diag(np.sum(V_smooth[1:, i_subset, :][:, :, i_subset], axis=0)))
        
        # E[f_{t-1}*f_{t-1}' | Omega_T] - diagonal only
        EZZ_BB = np.diag(np.diag(Zsmooth_idio_lag.T @ Zsmooth_idio_lag)) + np.diag(np.diag(np.sum(V_smooth[:-1, i_subset, :][:, :, i_subset], axis=0)))
        
        # E[f_t*f_{t-1}' | Omega_T] - diagonal only
        EZZ_FB = np.diag(np.diag(Zsmooth_idio.T @ Zsmooth_idio_lag)) + np.diag(np.diag(np.sum(VVsmooth[:, i_subset, :][:, :, i_subset], axis=0)))
        
        try:
            # Equation 6: AR(1) for idiosyncratic components
            A_i = EZZ_FB @ np.diag(1.0 / np.diag(EZZ_BB + np.eye(niM) * config.regularization))
            
            # Equation 8: Covariance matrix
            Q_i = (EZZ - A_i @ EZZ_FB.T) / T
            
            # Ensure Q_i is positive definite and diagonal
            Q_i = np.diag(np.maximum(np.diag(Q_i), config.min_variance))
            
            # Place updated results in output matrix
            A_new[i_subset, i_subset] = np.diag(np.diag(A_i))
            Q_new[i_subset, i_subset] = Q_i
            V_0_new[i_subset, i_subset] = np.diag(np.diag(V_smooth[0, i_subset, i_subset]))
        except (np.linalg.LinAlgError, ValueError) as e:
            _logger.warning(f"Idiosyncratic component update failed: {e}. Keeping previous values.")
    
    return A_new, Q_new, V_0_new


def _update_observation_matrix(X: np.ndarray, EZ: np.ndarray, EZZ: np.ndarray, C: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation matrix C using OLS regression."""
    try:
        N = X.shape[1]
        m = EZ.shape[1]
        X_clean = np.ma.filled(np.ma.masked_invalid(X), 0.0)
        sum_yEZ = X_clean.T @ EZ  # (N, m)
        sum_EZZ = np.sum(EZZ, axis=0) + np.eye(m) * config.regularization
        C_new = np.linalg.solve(sum_EZZ.T, sum_yEZ.T).T
        # Normalize columns
        for j in range(m):
            norm = np.linalg.norm(C_new[:, j])
            if norm > config.min_norm:
                C_new[:, j] /= norm
        return C_new
    except (np.linalg.LinAlgError, ValueError):
        return C


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
    
    # Initialize output
    C_new = C.copy()
    
    # Find unique block patterns
    # Convert blocks to tuples for hashing
    block_tuples = [tuple(row) for row in blocks]
    unique_blocks = []
    unique_indices = []
    seen = set()
    for i, bt in enumerate(block_tuples):
        if bt not in seen:
            unique_blocks.append(blocks[i])
            unique_indices.append(i)
            seen.add(bt)
    
    n_bl = len(unique_blocks)
    
    # Build block indices for clock-frequency and slower-frequency factors
    bl_idxM = []  # Indicator for clock-frequency factor loadings
    bl_idxQ = []  # Indicator for slower-frequency factor loadings
    R_con = None  # Block diagonal constraint matrix
    q_con = None  # Constraint vector
    
    if R_mat is not None and q is not None:
        from scipy.linalg import block_diag
        R_con_blocks = []
        q_con_blocks = []
        for i in range(n_blocks):
            # bl_idxQ: all factors in block i (including lags)
            bl_idxQ_row = []
            for bl_row in unique_blocks:
                if bl_row[i] > 0:
                    bl_idxQ_row.extend([True] * (int(r[i]) * p_plus_one))
                else:
                    bl_idxQ_row.extend([False] * (int(r[i]) * p_plus_one))
            
            # bl_idxM: only current factors (no lags)
            bl_idxM_row = []
            for bl_row in unique_blocks:
                if bl_row[i] > 0:
                    bl_idxM_row.extend([True] * int(r[i]))
                    bl_idxM_row.extend([False] * (int(r[i]) * (p_plus_one - 1)))
                else:
                    bl_idxM_row.extend([False] * (int(r[i]) * p_plus_one))
            
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
            
            # Build constraint matrix for block i
            # Check if any unique block pattern uses block i
            if any(bl_row[i] > 0 for bl_row in unique_blocks):
                R_con_blocks.append(np.kron(R_mat, np.eye(int(r[i]))))
                q_con_blocks.append(np.zeros(R_mat.shape[0] * int(r[i])))
        
        if R_con_blocks:
            R_con = block_diag(*R_con_blocks)
            q_con = np.concatenate(q_con_blocks)
    else:
        # No constraints - simpler indexing
        for i in range(n_blocks):
            bl_idxM_row = []
            bl_idxQ_row = []
            for bl_row in unique_blocks:
                if bl_row[i] > 0:
                    bl_idxM_row.extend([True] * int(r[i]))
                    bl_idxM_row.extend([False] * (int(r[i]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[i]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[i]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[i]) * p_plus_one))
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
    X_clean = np.where(nanY, 0.0, X)
    
    # Loop through unique block patterns
    for i, bl_i in enumerate(unique_blocks):
        # Find series with this block pattern
        bl_i_bool = (blocks == bl_i).all(axis=1)
        idx_i = np.where(bl_i_bool)[0]
        idx_iM = idx_i[idx_i < n_clock_freq]  # Clock-frequency series
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
        
        # Initialize sums for equation 13 (BGR 2010)
        denom = np.zeros((n_i * rs, n_i * rs))
        nom = np.zeros((n_i, rs))
        
        # Idiosyncratic indices for clock-frequency series
        i_idio_i = idio_indicator_M[idx_iM]
        i_idio_ii = c_idio_indicator[idx_iM]
        i_idio_ii = i_idio_ii[i_idio_i > 0]
        
        # Update clock-frequency variables
        for t in range(T):
            # Selection matrix for non-missing values
            Wt = np.diag(~nanY[idx_iM, t])
            
            # E[f_t*f_t' | Omega_T]
            Z_t = EZ[t+1, bl_idxM_i]  # (rs,)
            V_t = V_smooth[t+1, bl_idxM_i, :][:, bl_idxM_i]  # (rs, rs)
            EZZ_t = np.outer(Z_t, Z_t) + V_t
            
            denom += np.kron(EZZ_t, Wt)
            
            # E[y_t*f_t' | Omega_T]
            y_t = X_clean[t, idx_iM]  # (n_i,)
            nom += np.outer(y_t, Z_t)
            
            # Subtract idiosyncratic component contribution
            if len(i_idio_ii) > 0:
                idio_idx = rp1 + i_idio_ii - 1  # Convert to 0-based
                Z_idio_t = EZ[t+1, idio_idx]  # (n_idio,)
                V_idio_t = V_smooth[t+1, idio_idx, :][:, bl_idxM_i]  # (n_idio, rs)
                nom -= Wt[:, i_idio_i > 0] @ (np.outer(Z_idio_t, Z_t) + V_idio_t)
        
        # Solve for loadings
        try:
            denom_reg = denom + np.eye(n_i * rs) * config.regularization
            vec_C = np.linalg.solve(denom_reg, nom.flatten())
            C_new[idx_iM, bl_idxM_i] = vec_C.reshape(n_i, rs)
        except (np.linalg.LinAlgError, ValueError) as e:
            _logger.warning(f"Clock-frequency block {i} update failed: {e}. Keeping previous values.")
        
        # Update slower-frequency variables
        idx_iQ = idx_i[idx_i >= n_clock_freq]
        
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
                    # Selection matrix
                    Wt = np.diag([~nanY[j, t]])
                    
                    # E[f_t*f_t' | Omega_T]
                    Z_t = EZ[t+1, bl_idxQ_i]  # (rps,)
                    V_t = V_smooth[t+1, bl_idxQ_i, :][:, bl_idxQ_i]  # (rps, rps)
                    EZZ_t = np.outer(Z_t, Z_t) + V_t
                    
                    denom += np.kron(EZZ_t, Wt)
                    
                    # E[y_t*f_t' | Omega_T]
                    y_t = X_clean[t, j]
                    nom += y_t * Z_t
                    
                    # Subtract idiosyncratic component contribution
                    if tent_weights is not None and len(i_idio_jQ) == len(tent_weights):
                        Z_idio_t = EZ[t+1, i_idio_jQ]  # (tent_kernel_size,)
                        V_idio_t = V_smooth[t+1, i_idio_jQ, :][:, bl_idxQ_i]  # (tent_kernel_size, rps)
                        nom -= Wt[0, 0] * (tent_weights @ (np.outer(Z_idio_t, Z_t) + V_idio_t))
                
                try:
                    denom_reg = denom + np.eye(rps) * config.regularization
                    C_i = np.linalg.solve(denom_reg, nom)
                    
                    # Apply tent kernel constraints
                    if R_con_i is not None and q_con_i is not None and len(R_con_i) > 0:
                        # Constrained OLS: C_i_constr = C_i - inv(denom) * R_con_i' * inv(R_con_i * inv(denom) * R_con_i') * (R_con_i * C_i - q_con_i)
                        denom_inv = np.linalg.inv(denom_reg)
                        R_con_denom = R_con_i @ denom_inv @ R_con_i.T
                        R_con_denom_inv = np.linalg.inv(R_con_denom + np.eye(len(R_con_denom)) * config.regularization)
                        # Type assertion: q_con_i is guaranteed to be not None by the if condition
                        assert q_con_i is not None
                        constraint_term = R_con_i @ C_i - q_con_i
                        C_i_constr = C_i - denom_inv @ R_con_i.T @ R_con_denom_inv @ constraint_term
                    else:
                        C_i_constr = C_i
                    
                    C_new[j, bl_idxQ_i] = C_i_constr
                except (np.linalg.LinAlgError, ValueError) as e:
                    _logger.warning(f"Slower-frequency series {j} update failed: {e}. Keeping previous values.")
    
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
    return np.maximum(Q_new, np.eye(m) * config.min_variance)


def _update_observation_noise(X: np.ndarray, EZ: np.ndarray, C_new: np.ndarray, config: EMConfig) -> np.ndarray:
    """Update observation noise covariance R (diagonal) from residuals."""
    X_clean = np.ma.filled(np.ma.masked_invalid(X), 0.0)
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
    X_clean = np.where(nanY, 0.0, X)
    
    # Initialize covariance of residuals
    R_new = np.zeros((N, N))
    
    # Update using selection matrices (BGR equation 15)
    for t in range(T):
        Wt = np.diag(~nanY[:, t])  # Selection matrix
        
        # Residual: y_t - Wt * C_new * Z_{t+1}
        Z_t = EZ[t+1, :]  # (m,)
        residual = X_clean[t, :] - Wt @ C_new @ Z_t  # (N,)
        
        # R_new += residual * residual' + Wt * C_new * V_{t+1} * C_new' * Wt + (I - Wt) * R * (I - Wt)
        R_new += np.outer(residual, residual)
        R_new += Wt @ C_new @ V_smooth[t+1] @ C_new.T @ Wt
        R_new += (np.eye(N) - Wt) @ R @ (np.eye(N) - Wt)
    
    R_new = R_new / T
    
    # Extract diagonal and set minimum values
    RR = np.diag(R_new)
    
    # Ensure non-zero measurement error for clock-frequency idiosyncratic components
    idio_indicator_M = idio_indicator[:n_clock_freq]
    RR[idio_indicator_M > 0] = np.maximum(RR[idio_indicator_M > 0], 1e-4)
    
    # Ensure non-zero for slower-frequency series
    if n_clock_freq < N:
        RR[n_clock_freq:] = np.maximum(RR[n_clock_freq:], 1e-4)
    
    # Clip to reasonable range
    RR = np.clip(RR, config.min_variance, config.max_variance)
    
    R_new = np.diag(RR)
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
    blocks: Optional[np.ndarray] = None,
    r: Optional[np.ndarray] = None,
    p: Optional[int] = None,
    p_plus_one: Optional[int] = None,
    R_mat: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    n_clock_freq: Optional[int] = None,
    n_slower_freq: Optional[int] = None,
    idio_indicator: Optional[np.ndarray] = None,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
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
    blocks : np.ndarray, optional
        Block structure array (N x n_blocks). If provided, uses blocked updates.
    r : np.ndarray, optional
        Number of factors per block (n_blocks,). Required if blocks is provided.
    p : int, optional
        VAR lag order. Required if blocks is provided.
    p_plus_one : int, optional
        p + 1 (state dimension per factor). Required if blocks is provided.
    R_mat : np.ndarray, optional
        Tent kernel constraint matrix. Required for mixed-frequency data.
    q : np.ndarray, optional
        Tent kernel constraint vector. Required for mixed-frequency data.
    n_clock_freq : int, optional
        Number of clock-frequency series (series at the clock frequency, generic).
        Required if blocks is provided.
    n_slower_freq : int, optional
        Number of slower-frequency series (series slower than clock frequency, generic).
        Required for mixed-frequency data.
    idio_indicator : np.ndarray, optional
        Idiosyncratic component indicator (N,). Required if blocks is provided.
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights (generic, works for any frequency pair).
        
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
    if blocks is not None and r is not None and p is not None and p_plus_one is not None and n_clock_freq is not None and idio_indicator is not None:
        # Blocked updates (matching Nowcasting MATLAB)
        A_new, Q_new, V_0_new = _update_transition_matrix_blocked(
            EZ, V_smooth, VVsmooth, A, Q, blocks, r, p, p_plus_one, idio_indicator, n_clock_freq, config
        )
        
        # Blocked observation matrix update
        C_new = _update_observation_matrix_blocked(
            X, EZ, V_smooth, C, blocks, r, p_plus_one, R_mat, q, n_clock_freq, n_slower_freq or 0, idio_indicator, tent_weights_dict, config
        )
        
        # Blocked observation noise update
        R_new = _update_observation_noise_blocked(
            X, EZ, V_smooth, C_new, R, idio_indicator, n_clock_freq, config
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


def run_em_algorithm(
    X: np.ndarray,
    initial_params: Dict[str, np.ndarray],
    update_params_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None],
    get_params_fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    max_iter: int = DEFAULT_MAX_ITER,
    threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
    config: Optional[EMConfig] = None,
    blocks: Optional[np.ndarray] = None,
    r: Optional[np.ndarray] = None,
    p: Optional[int] = None,
    p_plus_one: Optional[int] = None,
    R_mat: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    n_clock_freq: Optional[int] = None,
    n_slower_freq: Optional[int] = None,
    idio_indicator: Optional[np.ndarray] = None,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Any]:
    """Run full EM algorithm until convergence.
    
    Parameters
    ----------
    X : np.ndarray
        Data array (T x N)
    initial_params : dict
        Initial parameters {'A', 'C', 'Q', 'R', 'Z_0', 'V_0'}
    update_params_fn : callable
        Function to update model parameters: update_params_fn(A, C, Q, R, Z_0, V_0)
    get_params_fn : callable
        Function to get current parameters: get_params_fn() -> (A, C, Q, R, Z_0, V_0)
    max_iter : int
        Maximum iterations
    threshold : float
        Convergence threshold
    config : EMConfig, optional
        EM configuration
    blocks : np.ndarray, optional
        Block structure array (N x n_blocks)
    r : np.ndarray, optional
        Number of factors per block (n_blocks,)
    p : int, optional
        VAR lag order
    p_plus_one : int, optional
        p + 1 (state dimension per factor)
    R_mat : np.ndarray, optional
        Tent kernel constraint matrix
    q : np.ndarray, optional
        Tent kernel constraint vector
    n_clock_freq : int, optional
        Number of clock-frequency series (series at the clock frequency, generic)
    n_slower_freq : int, optional
        Number of slower-frequency series (series slower than clock frequency, generic)
    idio_indicator : np.ndarray, optional
        Idiosyncratic component indicator (N,)
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights (generic)
        
    Returns
    -------
    dict
        Final state with keys: 'A', 'C', 'Q', 'R', 'Z_0', 'V_0', 'loglik', 'num_iter', 'converged'
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    # Initialize parameters
    update_params_fn(
        initial_params['A'],
        initial_params['C'],
        initial_params['Q'],
        initial_params['R'],
        initial_params['Z_0'],
        initial_params['V_0']
    )
    A, C, Q, R, Z_0, V_0 = get_params_fn()
    
    # Initialize state
    previous_loglik = float('-inf')
    num_iter = 0
    converged = False
    loglik = float('-inf')
    change = 0.0
    kalman_filter = None
    
    # EM loop
    while num_iter < max_iter and not converged:
        # E-step + M-step using pykalman
        A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik, kalman_filter = em_step(
            X, A, C, Q, R, Z_0, V_0, kalman_filter=kalman_filter, config=config,
            blocks=blocks, r=r, p=p, p_plus_one=p_plus_one, R_mat=R_mat, q=q,
            n_clock_freq=n_clock_freq, n_slower_freq=n_slower_freq, idio_indicator=idio_indicator, tent_weights_dict=tent_weights_dict
        )
        
        # Check for NaN/Inf (early stopping)
        if not all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                   for p in [A_new, C_new, Q_new, R_new, Z_0_new, V_0_new, loglik]):
            _logger.error(f"EM: NaN/Inf at iteration {num_iter + 1}, stopping")
            break
        
        # Update parameters
        update_params_fn(A_new, C_new, Q_new, R_new, Z_0_new, V_0_new)
        A, C, Q, R, Z_0, V_0 = A_new, C_new, Q_new, R_new, Z_0_new, V_0_new
        
        # Check convergence (relative change in log-likelihood)
        if num_iter >= config.min_iterations_for_convergence_check:
            if abs(previous_loglik) < config.small_loglik_threshold:
                change = abs(loglik - previous_loglik)
            else:
                change = abs((loglik - previous_loglik) / previous_loglik) if previous_loglik != 0.0 else abs(loglik - previous_loglik)
            converged = change < threshold
        else:
            change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else 0.0
        
        previous_loglik = loglik
        num_iter += 1
        
        # Log progress
        if num_iter % config.progress_log_interval == 0 or num_iter == 1:
            status = " ✓" if converged else ""
            _logger.info(f"EM iteration {num_iter}/{max_iter}: loglik={loglik:.4f}, change={change:.2e}{status}")
    
    # Final status
    if converged:
        print(f"\n✓ EM converged after {num_iter} iterations (loglik: {loglik:.6f})")
    else:
        print(f"\n⚠ EM stopped after {num_iter} iterations (loglik: {loglik:.6f}, change: {change:.2e})")
    
    _logger.info(f"EM training completed: converged={converged}, iterations={num_iter}, loglik={loglik:.6f}")
    
    return {
        'A': A,
        'C': C,
        'Q': Q,
        'R': R,
        'Z_0': Z_0,
        'V_0': V_0,
        'loglik': loglik,
        'num_iter': num_iter,
        'converged': converged
    }


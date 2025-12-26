"""Tent kernel utilities for DFM mixed-frequency handling.

This module provides functions for working with tent kernels in DFM initialization,
including building lag matrices, constructing quarterly observation matrices,
and setting up idiosyncratic chains.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from ...config.utils import get_tent_weights, generate_tent_weights
from .em import ensure_covariance_stable


def get_tent_kernel_size(
    R_mat: Optional[np.ndarray] = None,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    default_size: int = 5
) -> int:
    """Determine tent kernel size from available information.
    
    Parameters
    ----------
    R_mat : np.ndarray, optional
        Constraint matrix. If provided, size is inferred from R_mat.shape[1]
    tent_weights_dict : dict, optional
        Dictionary mapping frequency strings to tent weight arrays.
        If provided and contains 'q', uses len(tent_weights_dict['q'])
    default_size : int, default 5
        Default tent kernel size (for quarterly to monthly: 5 periods)
        
    Returns
    -------
    int
        Tent kernel size
    """
    if R_mat is not None:
        return R_mat.shape[1]
    elif tent_weights_dict is not None and 'q' in tent_weights_dict:
        return len(tent_weights_dict['q'])
    return default_size


def get_quarterly_tent_weights(clock: str, tent_kernel_size: int, dtype: type = np.float32) -> np.ndarray:
    """Get tent weights for quarterly idiosyncratic chain structure.
    
    Parameters
    ----------
    clock : str
        Clock frequency ('m', 'q', etc.)
    tent_kernel_size : int
        Expected tent kernel size
    dtype : type, default np.float32
        Data type for output array
        
    Returns
    -------
    np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1] for quarterly-monthly)
    """
    tent_q = get_tent_weights('q', clock) if clock == 'm' else None
    if tent_q is None:
        # Fallback: generate symmetric tent weights
        tent_q = generate_tent_weights(tent_kernel_size, 'symmetric').astype(dtype)
    else:
        tent_q = tent_q.astype(dtype)
    return tent_q


def build_quarterly_observation_matrix(
    N: int,
    nM: int,
    nQ: int,
    tent_weights: np.ndarray,
    dtype: type = np.float32
) -> np.ndarray:
    """Build observation matrix for quarterly idiosyncratic chains.
    
    Parameters
    ----------
    N : int
        Total number of series
    nM : int
        Number of monthly series
    nQ : int
        Number of quarterly series
    tent_weights : np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1])
    dtype : type, default np.float32
        Data type for output matrix
        
    Returns
    -------
    np.ndarray
        Observation matrix (N x (tent_kernel_size * nQ))
    """
    tent_kernel_size = len(tent_weights)
    C_quarterly = np.zeros((N, tent_kernel_size * nQ), dtype=dtype)
    C_quarterly[nM:, :] = np.kron(np.eye(nQ, dtype=dtype), tent_weights.reshape(1, -1))
    return C_quarterly


def build_quarterly_idiosyncratic_chain(
    nQ: int,
    chain_size: int,
    rho0: float,
    sig_e: np.ndarray,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build quarterly idiosyncratic chain transition matrices and covariance.
    
    Parameters
    ----------
    nQ : int
        Number of quarterly series
    chain_size : int
        Chain size (typically 5 for quarterly-monthly)
    rho0 : float
        AR(1) coefficient for quarterly series
    sig_e : np.ndarray
        Observation noise variances for quarterly series (nQ,)
    dtype : type, default np.float32
        Data type for output matrices
        
    Returns
    -------
    BQ : np.ndarray
        Transition matrix for quarterly chains (chain_size * nQ x chain_size * nQ)
    SQ : np.ndarray
        Process noise covariance (chain_size * nQ x chain_size * nQ)
    initViQ : np.ndarray
        Initial covariance (chain_size * nQ x chain_size * nQ)
    """
    if nQ == 0:
        return (
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype),
            np.zeros((0, 0), dtype=dtype)
        )
    
    # Build block structure
    temp = np.zeros((chain_size, chain_size), dtype=dtype)
    temp[0, 0] = 1.0
    SQ = np.kron(np.diag((1 - rho0 ** 2) * sig_e), temp)
    
    BQ_block = np.zeros((chain_size, chain_size), dtype=dtype)
    BQ_block[0, 0] = rho0
    BQ_block[1:, :chain_size-1] = np.eye(chain_size-1, dtype=dtype)
    BQ = np.kron(np.eye(nQ, dtype=dtype), BQ_block)
    
    # Compute initial covariance
    try:
        matrix_regularization = 1e-6  # Default regularization
        kron_BQBQ = np.kron(BQ, BQ)
        eye_kron = np.eye((chain_size * nQ) ** 2, dtype=dtype)
        initViQ_flat = np.linalg.solve(
            eye_kron - kron_BQBQ + eye_kron * matrix_regularization,
            SQ.flatten()
        )
        initViQ = initViQ_flat.reshape(chain_size * nQ, chain_size * nQ)
    except (np.linalg.LinAlgError, ValueError):
        initViQ = SQ.copy()
    
    return BQ, SQ, initViQ


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
    
    for lag_idx in range(num_lags):
        start_idx = max(0, tent_kernel_size - lag_idx)
        end_idx = T - lag_idx
        if start_idx < end_idx:
            col_start = lag_idx * num_factors
            col_end = col_start + num_factors
            lag_matrix[start_idx:end_idx, col_start:col_end] = factors[start_idx:end_idx, :num_factors]
    
    return lag_matrix


def initialize_block_loadings(
    res: np.ndarray,
    data_with_nans: np.ndarray,
    clock_freq_indices: np.ndarray,
    slower_freq_indices: np.ndarray,
    num_factors: int,
    tent_kernel_size: int,
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    N: int,
    max_lag_size: int,
    matrix_regularization: float = 1e-6,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize loadings for a block (clock frequency PCA + slower frequency constrained OLS).
    
    Parameters
    ----------
    res : np.ndarray
        Residuals matrix (T x N)
    data_with_nans : np.ndarray
        Data matrix with NaNs (T x N)
    clock_freq_indices : np.ndarray
        Indices of clock frequency series
    slower_freq_indices : np.ndarray
        Indices of slower frequency series
    num_factors : int
        Number of factors for this block
    tent_kernel_size : int
        Tent kernel size
    R_mat : np.ndarray, optional
        Constraint matrix for tent kernel aggregation
    q : np.ndarray, optional
        Constraint vector for tent kernel aggregation
    N : int
        Total number of series
    max_lag_size : int
        Maximum lag size for loading matrix
    matrix_regularization : float, default 1e-6
        Regularization for matrix operations
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    C_i : np.ndarray
        Loading matrix for this block (N x (num_factors * max_lag_size))
    factors : np.ndarray
        Extracted factors (T x num_factors)
    """
    from ...encoder.pca import compute_principal_components
    
    T = res.shape[0]
    C_i = np.zeros((N, num_factors * max_lag_size), dtype=dtype)
    
    # Clock frequency series: PCA on residuals
    if len(clock_freq_indices) > 0:
        clock_freq_residuals = res[:, clock_freq_indices]
        clock_freq_residuals_centered = clock_freq_residuals - clock_freq_residuals.mean(axis=0, keepdims=True)
        
        # Compute covariance matrix
        if clock_freq_residuals_centered.shape[0] > 1:
            if len(clock_freq_indices) > 1:
                cov_res = np.cov(clock_freq_residuals_centered.T)
                cov_res = (cov_res + cov_res.T) / 2  # Ensure symmetry
            else:
                cov_res = np.atleast_2d(np.var(clock_freq_residuals_centered, axis=0, ddof=0))
        else:
            cov_res = np.eye(len(clock_freq_indices), dtype=dtype)
        
        try:
            _, eigenvectors = compute_principal_components(cov_res, num_factors, block_idx=0)
            loadings = eigenvectors
            loadings_sum = np.sum(loadings, axis=0)
            loadings = np.where(loadings_sum < 0, -loadings, loadings)
        except (RuntimeError, ValueError):
            loadings = np.eye(len(clock_freq_indices), dtype=dtype)[:, :num_factors]
        
        C_i[clock_freq_indices, :num_factors] = loadings
        factors = res[:, clock_freq_indices] @ loadings
    else:
        factors = np.zeros((T, num_factors), dtype=dtype)
    
    # Slower frequency series: constrained least squares
    if R_mat is not None and q is not None and len(slower_freq_indices) > 0:
        constraint_matrix_block = np.kron(R_mat, np.eye(num_factors, dtype=dtype))
        constraint_vector_block = np.kron(q, np.zeros(num_factors, dtype=dtype))
        
        lag_matrix = build_lag_matrix(factors, T, num_factors, tent_kernel_size, 1, dtype)
        slower_freq_factors = lag_matrix[:, :num_factors * tent_kernel_size] if lag_matrix.shape[1] >= num_factors * tent_kernel_size else lag_matrix
        
        for series_idx in slower_freq_indices:
            series_idx_int = int(series_idx)
            series_data = data_with_nans[tent_kernel_size:, series_idx_int]
            non_nan_mask = ~np.isnan(series_data)
            
            # Use clean data if insufficient non-NaN values
            min_required = slower_freq_factors.shape[1] + 2
            if np.sum(non_nan_mask) < min_required:
                series_data = res[tent_kernel_size:, series_idx_int]
                non_nan_mask = np.ones(len(series_data), dtype=bool)
            
            slower_freq_factors_clean = slower_freq_factors[tent_kernel_size:][non_nan_mask, :]
            series_data_clean = series_data[non_nan_mask]
            
            # Skip if insufficient data for regression
            if len(slower_freq_factors_clean) == 0 or slower_freq_factors_clean.shape[0] < slower_freq_factors_clean.shape[1]:
                continue
            
            try:
                factors_cov_inv = np.linalg.pinv(slower_freq_factors_clean.T @ slower_freq_factors_clean)
                loadings_unconstrained = factors_cov_inv @ slower_freq_factors_clean.T @ series_data_clean
                
                # Apply constraints
                constraint_cov_T = constraint_matrix_block @ factors_cov_inv @ constraint_matrix_block.T
                try:
                    reg_matrix = constraint_cov_T + np.eye(constraint_cov_T.shape[0], dtype=dtype) * matrix_regularization
                    constraint_rhs = constraint_matrix_block @ loadings_unconstrained - constraint_vector_block
                    loadings_constrained = loadings_unconstrained - factors_cov_inv @ constraint_matrix_block.T @ np.linalg.solve(reg_matrix, constraint_rhs)
                except (np.linalg.LinAlgError, ValueError):
                    loadings_constrained = loadings_unconstrained
                
                C_i[series_idx_int, :num_factors * tent_kernel_size] = loadings_constrained
            except (np.linalg.LinAlgError, ValueError):
                pass
    
    return C_i, factors


def initialize_block_transition(
    lag_matrix: np.ndarray,
    factors: np.ndarray,
    num_factors: int,
    max_lag_size: int,
    p: int,
    T: int,
    regularization: float = 1e-6,
    default_transition_coef: float = 0.9,
    default_process_noise: float = 0.1,
    matrix_regularization: float = 1e-6,
    eigenval_floor: float = 1e-8,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize transition matrix, process noise, and initial covariance for a block.
    
    Parameters
    ----------
    lag_matrix : np.ndarray
        Lag matrix (T x (num_factors * num_lags))
    factors : np.ndarray
        Factor matrix (T x num_factors)
    num_factors : int
        Number of factors
    max_lag_size : int
        Maximum lag size
    p : int
        AR lag order
    T : int
        Number of time periods
    regularization : float, default 1e-6
        Regularization for OLS
    default_transition_coef : float, default 0.9
        Default transition coefficient
    default_process_noise : float, default 0.1
        Default process noise
    matrix_regularization : float, default 1e-6
        Regularization for matrix operations
    eigenval_floor : float, default 1e-8
        Minimum eigenvalue floor
    dtype : type, default np.float32
        Data type
        
    Returns
    -------
    A_i : np.ndarray
        Transition matrix (block_size x block_size)
    Q_i : np.ndarray
        Process noise (block_size x block_size)
    V_0_i : np.ndarray
        Initial covariance (block_size x block_size)
    """
    block_size = num_factors * max_lag_size
    A_i = np.zeros((block_size, block_size), dtype=dtype)
    
    current_state = lag_matrix[:, :num_factors] if lag_matrix.shape[1] >= num_factors else np.zeros((T, num_factors), dtype=dtype)
    lagged_state = lag_matrix[:, num_factors:num_factors * (p + 1)] if lag_matrix.shape[1] >= num_factors * (p + 1) else np.zeros((T, num_factors * p), dtype=dtype)
    
    # Initialize transition matrix
    default_A_block = np.eye(num_factors, dtype=dtype) * default_transition_coef
    default_shift = np.eye(num_factors * (max_lag_size - 1), dtype=dtype) if num_factors * (max_lag_size - 1) > 0 else np.zeros((0, 0), dtype=dtype)
    
    if T > p and lagged_state.shape[1] > 0:
        try:
            lagged_cov = lagged_state.T @ lagged_state
            lagged_cov_reg = lagged_cov + np.eye(lagged_cov.shape[0], dtype=dtype) * regularization
            lagged_current_cov = lagged_state.T @ current_state
            transition_coef = np.linalg.solve(lagged_cov_reg, lagged_current_cov).T
            
            # Ensure correct shape
            if transition_coef.shape != (num_factors, num_factors * p):
                transition_coef_new = np.zeros((num_factors, num_factors * p), dtype=dtype)
                min_rows = min(transition_coef.shape[0], num_factors)
                min_cols = min(transition_coef.shape[1], num_factors * p)
                transition_coef_new[:min_rows, :min_cols] = transition_coef[:min_rows, :min_cols]
                transition_coef = transition_coef_new
            
            A_i[:num_factors, :num_factors * p] = transition_coef
            if default_shift.shape[0] > 0:
                A_i[num_factors:, :num_factors * (max_lag_size - 1)] = default_shift
        except (np.linalg.LinAlgError, ValueError):
            A_i[:num_factors, :num_factors] = default_A_block
            if default_shift.shape[0] > 0:
                A_i[num_factors:, :num_factors * (max_lag_size - 1)] = default_shift
    else:
        A_i[:num_factors, :num_factors] = default_A_block
        if default_shift.shape[0] > 0:
            A_i[num_factors:, :num_factors * (max_lag_size - 1)] = default_shift
    
    # Initialize process noise
    default_Q_block = np.eye(num_factors, dtype=dtype) * default_process_noise
    Q_i = np.zeros((block_size, block_size), dtype=dtype)
    
    if T > p and lagged_state.shape[1] > 0:
        try:
            residuals = current_state[p:, :] - (lagged_state[p:, :] @ A_i[:num_factors, :num_factors * p].T)
            if residuals.shape[0] > 1:
                if residuals.shape[1] == 1:
                    Q_i[:num_factors, :num_factors] = np.atleast_2d(np.var(residuals, axis=0, ddof=0))
                else:
                    Q_i[:num_factors, :num_factors] = np.cov(residuals.T, ddof=0)
                    Q_i[:num_factors, :num_factors] = (Q_i[:num_factors, :num_factors] + Q_i[:num_factors, :num_factors].T) / 2
            else:
                Q_i[:num_factors, :num_factors] = default_Q_block
        except (np.linalg.LinAlgError, ValueError):
            Q_i[:num_factors, :num_factors] = default_Q_block
    else:
        Q_i[:num_factors, :num_factors] = default_Q_block
    
    # Ensure Q_i is positive definite
    Q_i[:num_factors, :num_factors] = ensure_covariance_stable(
        Q_i[:num_factors, :num_factors], min_eigenval=eigenval_floor
    )
    
    # Initial covariance
    try:
        A_i_block = A_i[:block_size, :block_size]
        kron_AA = np.kron(A_i_block, A_i_block)
        eye_kron = np.eye(block_size ** 2, dtype=dtype)
        initV_i_flat = np.linalg.solve(
            eye_kron - kron_AA + eye_kron * matrix_regularization,
            Q_i[:block_size, :block_size].flatten()
        )
        V_0_i = initV_i_flat.reshape(block_size, block_size)
    except (np.linalg.LinAlgError, ValueError):
        V_0_i = Q_i[:block_size, :block_size].copy()
    
    # Ensure correct size (pad if needed)
    def ensure_size(M: np.ndarray, size: int) -> np.ndarray:
        if M.shape[0] < size or M.shape[1] < size:
            M_new = np.zeros((size, size), dtype=dtype)
            M_new[:M.shape[0], :M.shape[1]] = M
            return M_new
        return M[:size, :size]
    
    return ensure_size(A_i, block_size), ensure_size(Q_i, block_size), ensure_size(V_0_i, block_size)

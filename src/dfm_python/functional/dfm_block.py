"""DFM block initialization utilities.

This module provides functions for initializing DFM blocks, including:
- Block structure parsing and inference from configuration
- Tent kernel utilities for mixed-frequency handling
- Block loadings initialization (PCA + constrained OLS)
- Block transition matrices initialization
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from ..numeric.tent import get_tent_weights, generate_tent_weights
from ..numeric.stability import create_scaled_identity
from ..logger import get_logger
from ..config.constants import (
    DEFAULT_REGULARIZATION,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    MIN_EIGENVALUE,
    DEFAULT_IDENTITY_SCALE,
)
from ..numeric.stability import ensure_covariance_stable, ensure_process_noise_stable
from ..numeric.estimator import (
    estimate_var_unified,
    estimate_constrained_ols_unified,
)
from ..logger import get_logger

_logger = get_logger(__name__)


def build_slower_freq_observation_matrix(
    N: int,
    n_clock_freq: int,
    n_slower_freq: int,
    tent_weights: np.ndarray,
    dtype: type = np.float32
) -> np.ndarray:
    """Build observation matrix for slower-frequency idiosyncratic chains.
    
    Parameters
    ----------
    N : int
        Total number of series
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    n_slower_freq : int
        Number of slower-frequency series (series slower than clock frequency, generic)
    tent_weights : np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1])
    dtype : type, default np.float32
        Data type for output matrix
        
    Returns
    -------
    np.ndarray
        Observation matrix (N x (tent_kernel_size * n_slower_freq))
    """
    tent_kernel_size = len(tent_weights)
    C_slower_freq = np.zeros((N, tent_kernel_size * n_slower_freq), dtype=dtype)
    C_slower_freq[n_clock_freq:, :] = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), tent_weights.reshape(1, -1))
    return C_slower_freq


def build_slower_freq_idiosyncratic_chain(
    n_slower_freq: int,
    chain_size: int,
    rho0: float,
    sig_e: np.ndarray,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build slower-frequency idiosyncratic chain transition matrices and covariance.
    
    Parameters
    ----------
    n_slower_freq : int
        Number of slower-frequency series
    chain_size : int
        Chain size (tent kernel size, e.g., 5 for quarterly-to-monthly)
    rho0 : float
        AR(1) coefficient for slower-frequency series
    sig_e : np.ndarray
        Observation noise variances for slower-frequency series (n_slower_freq,)
    dtype : type, default np.float32
        Data type for output matrices
        
    Returns
    -------
    BQ : np.ndarray
        Transition matrix for slower-frequency chains (chain_size * n_slower_freq x chain_size * n_slower_freq)
    SQ : np.ndarray
        Process noise covariance (chain_size * n_slower_freq x chain_size * n_slower_freq)
    initViQ : np.ndarray
        Initial covariance (chain_size * n_slower_freq x chain_size * n_slower_freq)
    """
    if n_slower_freq == 0:
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
    BQ_block[1:, :chain_size-1] = create_scaled_identity(chain_size-1, DEFAULT_IDENTITY_SCALE, dtype=dtype)
    BQ = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), BQ_block)
    
    # Compute initial covariance: solve (I - BQ ⊗ BQ) vec(V_0) = vec(SQ)
    from ..numeric.estimator import compute_initial_covariance_from_transition
    initViQ = compute_initial_covariance_from_transition(BQ, SQ, regularization=DEFAULT_REGULARIZATION, dtype=dtype)
    
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
    
    # Vectorized implementation: build all lags at once
    for lag_idx in range(num_lags):
        start_idx = max(0, tent_kernel_size - lag_idx)
        end_idx = T - lag_idx
        if start_idx < end_idx:
            col_start = lag_idx * num_factors
            col_end = col_start + num_factors
            # Use advanced indexing for better performance
            lag_matrix[start_idx:end_idx, col_start:col_end] = factors[start_idx:end_idx, :num_factors].copy()
    
    return lag_matrix


def initialize_block_loadings(
    data_for_extraction: np.ndarray,
    data_with_nans: np.ndarray,
    clock_freq_indices: np.ndarray,
    slower_freq_indices: np.ndarray,
    num_factors: int,
    tent_kernel_size: int,
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    N: int,
    max_lag_size: int,
    matrix_regularization: Optional[float] = None,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize loadings for a block (clock frequency PCA + slower frequency constrained OLS).
    
    **Note**: For Block 1, `data_for_extraction` is the original data (after cleaning).
    For subsequent blocks, `data_for_extraction` contains residuals after removing
    previous blocks' contributions.
    
    Parameters
    ----------
    data_for_extraction : np.ndarray
        Data matrix (T x N). For Block 1: original data. For Block 2+: residuals.
    data_with_nans : np.ndarray
        Data matrix with NaNs preserved (T x N)
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
    matrix_regularization : float, default DEFAULT_REGULARIZATION
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
    from ..encoder.pca import compute_principal_components
    
    T = data_for_extraction.shape[0]
    C_i = np.zeros((N, num_factors * max_lag_size), dtype=dtype)
    
    # Clock frequency series: PCA extraction
    # Block 1: PCA on original data
    # Block 2+: PCA on residuals (after removing previous blocks)
    if len(clock_freq_indices) == 0:
        factors = np.zeros((T, num_factors), dtype=dtype)
    else:
        clock_freq_data = data_for_extraction[:, clock_freq_indices]
        
        # Handle missing values for PCA: use nanmean/nanstd for centering
        # NaN values will be handled by Kalman filter during EM, but PCA needs finite values
        clock_freq_data_mean = np.nanmean(clock_freq_data, axis=0, keepdims=True)
        clock_freq_data_centered = clock_freq_data - clock_freq_data_mean
        
        # Replace NaN with 0 after centering for covariance computation
        # (This is only for initialization - EM will use proper masked arrays)
        clock_freq_data_centered_clean = np.where(
            np.isfinite(clock_freq_data_centered),
            clock_freq_data_centered,
            0.0
        )
        
        # Compute covariance matrix (only over valid observations)
        if clock_freq_data_centered_clean.shape[0] <= 1:
            cov_data = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)
        elif len(clock_freq_indices) == 1:
            cov_data = np.atleast_2d(np.nanvar(clock_freq_data_centered, axis=0, ddof=0))
        else:
            # Use nan-aware covariance for proper handling of missing values
            # np.cov with NaN will produce NaN, so compute manually with nan-aware stats
            valid_mask = np.all(np.isfinite(clock_freq_data_centered), axis=1)
            if valid_mask.sum() > 1:
                valid_data = clock_freq_data_centered[valid_mask, :]
                cov_data = np.cov(valid_data.T)
                cov_data = (cov_data + cov_data.T) / 2  # Ensure symmetry
            else:
                # Fallback: use identity if insufficient valid observations
                cov_data = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)
        
        try:
            # PCA can extract at most min(n_series, num_factors) components
            max_extractable = min(len(clock_freq_indices), num_factors)
            _, eigenvectors = compute_principal_components(cov_data, max_extractable, block_idx=0)
            loadings = eigenvectors
            # Ensure positive sign convention
            loadings = np.where(np.sum(loadings, axis=0) < 0, -loadings, loadings)
            
            # Pad loadings to expected shape if PCA returned fewer factors than requested
            if loadings.shape[1] < num_factors:
                padding = np.zeros((loadings.shape[0], num_factors - loadings.shape[1]), dtype=dtype)
                loadings = np.hstack([loadings, padding])
        except (RuntimeError, ValueError):
            loadings = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)[:, :num_factors]
        
        C_i[clock_freq_indices, :num_factors] = loadings
        # Extract only the actual factors (non-zero columns) for computing factors matrix
        # Handle NaN in data_for_extraction: NaN * loadings = NaN (preserved for Kalman filter)
        n_actual_factors = min(len(clock_freq_indices), num_factors)
        factors = data_for_extraction[:, clock_freq_indices] @ loadings[:, :n_actual_factors]
        # NaN values are preserved - will be handled by Kalman filter via masked arrays during EM
        
        # Pad factors matrix to expected shape if needed
        if factors.shape[1] < num_factors:
            padding = np.zeros((factors.shape[0], num_factors - factors.shape[1]), dtype=dtype)
            factors = np.hstack([factors, padding])
    
    # Slower frequency series: constrained least squares
    if R_mat is not None and q is not None and len(slower_freq_indices) > 0:
        constraint_matrix_block = np.kron(R_mat, create_scaled_identity(num_factors, DEFAULT_IDENTITY_SCALE, dtype=dtype))
        constraint_vector_block = np.kron(q, np.zeros(num_factors, dtype=dtype))
        
        # Build lag matrix once (cached for all series in this block)
        lag_matrix = build_lag_matrix(factors, T, num_factors, tent_kernel_size, 1, dtype)
        n_cols = min(num_factors * tent_kernel_size, lag_matrix.shape[1])
        slower_freq_factors = lag_matrix[:, :n_cols]
        
        # Log progress for slower frequency series initialization
        total_slower = len(slower_freq_indices)
        _logger.info(f"    Processing {total_slower} slower-frequency series with constrained OLS...")
        
        for idx, series_idx in enumerate(slower_freq_indices):
            # Log progress every 10 series or at start/end
            if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == total_slower:
                _logger.info(f"      Series {idx + 1}/{total_slower} (index {series_idx})")
            series_idx_int = int(series_idx)
            series_data = data_with_nans[tent_kernel_size:, series_idx_int]
            non_nan_mask = ~np.isnan(series_data)
            
            # Use clean data if insufficient non-NaN values
            min_required = slower_freq_factors.shape[1] + 2
            if np.sum(non_nan_mask) < min_required:
                series_data = data_for_extraction[tent_kernel_size:, series_idx_int]
                non_nan_mask = np.ones(len(series_data), dtype=bool)
            
            slower_freq_factors_clean = slower_freq_factors[tent_kernel_size:][non_nan_mask, :]
            series_data_clean = series_data[non_nan_mask]
            
            # Skip if insufficient data
            if len(slower_freq_factors_clean) < slower_freq_factors_clean.shape[1]:
                continue
            
            try:
                # Use unified constrained OLS estimation
                # Increase regularization for slower-frequency series to handle ill-conditioning
                # Tent kernel factors are highly correlated, requiring much higher regularization
                from ..config.constants import DEFAULT_TENT_KERNEL_REGULARIZATION_MULTIPLIER
                base_reg = matrix_regularization or DEFAULT_REGULARIZATION
                # Use significantly higher regularization for slower-frequency (tent kernel) series
                # Increased multiplier handles extreme ill-conditioning (rcond ~1e-11)
                reg = base_reg * DEFAULT_TENT_KERNEL_REGULARIZATION_MULTIPLIER
                loadings_constrained = estimate_constrained_ols_unified(
                    y=series_data_clean,
                    X=slower_freq_factors_clean,
                    R=constraint_matrix_block,
                    q=constraint_vector_block,
                    V_smooth=None,  # Raw data mode
                    regularization=reg,
                    dtype=dtype
                )
                C_i[series_idx_int, :num_factors * tent_kernel_size] = loadings_constrained
            except (np.linalg.LinAlgError, ValueError) as e:
                _logger.warning(f"Failed to compute constrained loadings for series {series_idx_int}: {e}. Skipping.")
    
    return C_i, factors


def initialize_block_transition(
    lag_matrix: np.ndarray,
    factors: np.ndarray,
    num_factors: int,
    max_lag_size: int,
    p: int,
    T: int,
    regularization: float = DEFAULT_REGULARIZATION,
    default_transition_coef: float = DEFAULT_TRANSITION_COEF,
    default_process_noise: float = DEFAULT_PROCESS_NOISE,
    matrix_regularization: float = DEFAULT_REGULARIZATION,
    eigenval_floor: float = MIN_EIGENVALUE,
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
    regularization : float, default DEFAULT_REGULARIZATION
        Regularization for OLS
    default_transition_coef : float, default DEFAULT_TRANSITION_COEF
        Default transition coefficient
    default_process_noise : float, default DEFAULT_PROCESS_NOISE
        Default process noise
    matrix_regularization : float, default DEFAULT_REGULARIZATION
        Regularization for matrix operations
    eigenval_floor : float, default MIN_EIGENVALUE
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
    
    # Extract current and lagged states
    n_cols = min(num_factors, lag_matrix.shape[1])
    current_state = lag_matrix[:, :n_cols] if n_cols > 0 else np.zeros((T, num_factors), dtype=dtype)
    lag_cols = min(num_factors * (p + 1), lag_matrix.shape[1])
    lagged_state = lag_matrix[:, num_factors:lag_cols] if lag_cols > num_factors else np.zeros((T, num_factors * p), dtype=dtype)
    
    # Initialize transition matrix
    default_A_block = create_scaled_identity(num_factors, default_transition_coef, dtype)
    shift_size = num_factors * (max_lag_size - 1)
    default_shift = create_scaled_identity(shift_size, DEFAULT_IDENTITY_SCALE, dtype=dtype) if shift_size > 0 else np.zeros((0, 0), dtype=dtype)
    
    # Estimate transition coefficients using unified VAR estimation
    if T > p and lagged_state.shape[1] > 0:
        try:
            # Use unified VAR estimation (raw data mode)
            A_transition, Q_transition = estimate_var_unified(
                y=current_state[p:, :],  # Current state (T-p x num_factors)
                x=lagged_state[p:, :],   # Lagged state (T-p x num_factors*p)
                V_smooth=None,  # Raw data mode
                VVsmooth=None,
                regularization=regularization,
                min_variance=eigenval_floor,
                dtype=dtype
            )
            
            # Ensure correct shape
            expected_shape = (num_factors, num_factors * p)
            if A_transition.shape != expected_shape:
                transition_coef_new = np.zeros(expected_shape, dtype=dtype)
                min_rows = min(A_transition.shape[0], num_factors)
                min_cols = min(A_transition.shape[1], num_factors * p)
                transition_coef_new[:min_rows, :min_cols] = A_transition[:min_rows, :min_cols]
                A_transition = transition_coef_new
            
            A_i[:num_factors, :num_factors * p] = A_transition
            Q_i = np.zeros((block_size, block_size), dtype=dtype)
            Q_i[:num_factors, :num_factors] = Q_transition
        except (np.linalg.LinAlgError, ValueError):
            A_i[:num_factors, :num_factors] = default_A_block
            Q_i = np.zeros((block_size, block_size), dtype=dtype)
            Q_i[:num_factors, :num_factors] = create_scaled_identity(num_factors, default_process_noise, dtype)
    else:
        A_i[:num_factors, :num_factors] = default_A_block
        Q_i = np.zeros((block_size, block_size), dtype=dtype)
        Q_i[:num_factors, :num_factors] = create_scaled_identity(num_factors, default_process_noise, dtype=dtype)
    
    # Add shift matrix for lag structure
    if shift_size > 0:
        A_i[num_factors:, :shift_size] = default_shift
    
    # Ensure Q_i is positive definite and bounded (generic process noise stabilization)
    Q_i[:num_factors, :num_factors] = ensure_process_noise_stable(
        Q_i[:num_factors, :num_factors], min_eigenval=eigenval_floor, warn=True, dtype=dtype
    )
    
    # Initial covariance: solve (I - A ⊗ A) vec(V_0) = vec(Q)
    from ..numeric.estimator import compute_initial_covariance_from_transition
    A_i_block = A_i[:block_size, :block_size]
    Q_i_block = Q_i[:block_size, :block_size]
    reg = matrix_regularization or DEFAULT_REGULARIZATION
    V_0_i = compute_initial_covariance_from_transition(A_i_block, Q_i_block, regularization=reg, dtype=dtype)
    
    return A_i, Q_i, V_0_i



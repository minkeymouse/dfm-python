"""DFM block initialization utilities.

This module provides functions for initializing DFM blocks, including:
- Block structure parsing and inference from configuration
- Tent kernel utilities for mixed-frequency handling
- Block loadings initialization (PCA + constrained OLS)
- Block transition matrices initialization
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from ..numeric.tent import get_tent_weights
from ..numeric.tent import generate_tent_weights
from ..config.constants import (
    DEFAULT_REGULARIZATION,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    MIN_EIGENVALUE,
)
from ..numeric.stability import ensure_covariance_stable

# Import DEFAULT_BLOCK_NAME from constants
from ..config.constants import DEFAULT_BLOCK_NAME


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
        Uses first available entry if multiple exist.
    default_size : int, default 5
        Default tent kernel size
        
    Returns
    -------
    int
        Tent kernel size
    """
    if R_mat is not None:
        return R_mat.shape[1]
    if tent_weights_dict:
        # Use first available tent weights
        first_weights = next(iter(tent_weights_dict.values()))
        return len(first_weights)
    return default_size


def get_slower_freq_tent_weights(slower_freq: str, clock: str, tent_kernel_size: int, dtype: type = np.float32) -> np.ndarray:
    """Get tent weights for slower-frequency idiosyncratic chain structure.
    
    Parameters
    ----------
    slower_freq : str
        Slower frequency ('q', 'sa', 'a', etc.)
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
    tent_weights = get_tent_weights(slower_freq, clock)
    if tent_weights is None:
        # Fallback: generate symmetric tent weights
        tent_weights = generate_tent_weights(tent_kernel_size, 'symmetric').astype(dtype)
    else:
        tent_weights = tent_weights.astype(dtype)
    return tent_weights


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
    C_slower_freq[n_clock_freq:, :] = np.kron(np.eye(n_slower_freq, dtype=dtype), tent_weights.reshape(1, -1))
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
    BQ_block[1:, :chain_size-1] = np.eye(chain_size-1, dtype=dtype)
    BQ = np.kron(np.eye(n_slower_freq, dtype=dtype), BQ_block)
    
    # Compute initial covariance: solve (I - BQ ⊗ BQ) vec(V_0) = vec(SQ)
    try:
        kron_BQBQ = np.kron(BQ, BQ)
        eye_kron = np.eye((chain_size * n_slower_freq) ** 2, dtype=dtype)
        initViQ_flat = np.linalg.solve(
            eye_kron - kron_BQBQ + eye_kron * DEFAULT_REGULARIZATION,
            SQ.flatten()
        )
        initViQ = initViQ_flat.reshape(chain_size * n_slower_freq, chain_size * n_slower_freq)
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
        clock_freq_data_centered = clock_freq_data - clock_freq_data.mean(axis=0, keepdims=True)
        
        # Compute covariance matrix
        if clock_freq_data_centered.shape[0] <= 1:
            cov_data = np.eye(len(clock_freq_indices), dtype=dtype)
        elif len(clock_freq_indices) == 1:
            cov_data = np.atleast_2d(np.var(clock_freq_data_centered, axis=0, ddof=0))
        else:
            cov_data = np.cov(clock_freq_data_centered.T)
            cov_data = (cov_data + cov_data.T) / 2  # Ensure symmetry
        
        try:
            _, eigenvectors = compute_principal_components(cov_data, num_factors, block_idx=0)
            loadings = eigenvectors
            # Ensure positive sign convention
            loadings = np.where(np.sum(loadings, axis=0) < 0, -loadings, loadings)
        except (RuntimeError, ValueError):
            loadings = np.eye(len(clock_freq_indices), dtype=dtype)[:, :num_factors]
        
        C_i[clock_freq_indices, :num_factors] = loadings
        factors = data_for_extraction[:, clock_freq_indices] @ loadings
    
    # Slower frequency series: constrained least squares
    if R_mat is not None and q is not None and len(slower_freq_indices) > 0:
        constraint_matrix_block = np.kron(R_mat, np.eye(num_factors, dtype=dtype))
        constraint_vector_block = np.kron(q, np.zeros(num_factors, dtype=dtype))
        
        lag_matrix = build_lag_matrix(factors, T, num_factors, tent_kernel_size, 1, dtype)
        n_cols = min(num_factors * tent_kernel_size, lag_matrix.shape[1])
        slower_freq_factors = lag_matrix[:, :n_cols]
        
        for series_idx in slower_freq_indices:
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
                factors_cov_inv = np.linalg.pinv(slower_freq_factors_clean.T @ slower_freq_factors_clean)
                loadings_unconstrained = factors_cov_inv @ slower_freq_factors_clean.T @ series_data_clean
                
                # Apply constraints
                constraint_cov_T = constraint_matrix_block @ factors_cov_inv @ constraint_matrix_block.T
                reg = matrix_regularization or DEFAULT_REGULARIZATION
                reg_matrix = constraint_cov_T + np.eye(constraint_cov_T.shape[0], dtype=dtype) * reg
                constraint_rhs = constraint_matrix_block @ loadings_unconstrained - constraint_vector_block
                loadings_constrained = loadings_unconstrained - factors_cov_inv @ constraint_matrix_block.T @ np.linalg.solve(reg_matrix, constraint_rhs)
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
    default_A_block = np.eye(num_factors, dtype=dtype) * default_transition_coef
    shift_size = num_factors * (max_lag_size - 1)
    default_shift = np.eye(shift_size, dtype=dtype) if shift_size > 0 else np.zeros((0, 0), dtype=dtype)
    
    # Estimate transition coefficients from data
    if T > p and lagged_state.shape[1] > 0:
        try:
            lagged_cov = lagged_state.T @ lagged_state
            lagged_cov_reg = lagged_cov + np.eye(lagged_cov.shape[0], dtype=dtype) * regularization
            lagged_current_cov = lagged_state.T @ current_state
            transition_coef = np.linalg.solve(lagged_cov_reg, lagged_current_cov).T
            
            # Ensure correct shape
            expected_shape = (num_factors, num_factors * p)
            if transition_coef.shape != expected_shape:
                transition_coef_new = np.zeros(expected_shape, dtype=dtype)
                min_rows = min(transition_coef.shape[0], num_factors)
                min_cols = min(transition_coef.shape[1], num_factors * p)
                transition_coef_new[:min_rows, :min_cols] = transition_coef[:min_rows, :min_cols]
                transition_coef = transition_coef_new
            
            A_i[:num_factors, :num_factors * p] = transition_coef
        except (np.linalg.LinAlgError, ValueError):
            A_i[:num_factors, :num_factors] = default_A_block
    
    # Add shift matrix for lag structure
    if shift_size > 0:
        A_i[num_factors:, :shift_size] = default_shift
    
    # Initialize process noise from residuals
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
    
    # Initial covariance: solve (I - A ⊗ A) vec(V_0) = vec(Q)
    A_i_block = A_i[:block_size, :block_size]
    Q_i_block = Q_i[:block_size, :block_size]
    try:
        kron_AA = np.kron(A_i_block, A_i_block)
        eye_kron = np.eye(block_size ** 2, dtype=dtype)
        reg = matrix_regularization or DEFAULT_REGULARIZATION
        initV_i_flat = np.linalg.solve(
            eye_kron - kron_AA + eye_kron * reg,
            Q_i_block.flatten()
        )
        V_0_i = initV_i_flat.reshape(block_size, block_size)
    except (np.linalg.LinAlgError, ValueError):
        V_0_i = Q_i_block.copy()
    
    return A_i, Q_i, V_0_i


# ============================================================================
# Block Structure Parsing and Inference
# ============================================================================

def parse_blocks_dict(blocks_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Parse blocks from dict format.
    
    Parameters
    ----------
    blocks_data : Dict[str, Any]
        Dictionary mapping block names to block configurations
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping block names to block config dicts
        
    Raises
    ------
    ValueError
        If block config is not a dict
    """
    blocks_dict = {}
    for block_name, block_cfg in blocks_data.items():
        if isinstance(block_cfg, dict):
            blocks_dict[block_name] = block_cfg
        else:
            raise ValueError(f"Invalid block config for {block_name}: {block_cfg}. Must be a dict.")
    return blocks_dict


def infer_blocks(
    series_list: List[Any],
    data: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Infer blocks from configuration data when blocks not explicitly provided.
    
    Note: SeriesConfig no longer contains blocks information.
    Blocks are defined in DFMConfig, not in SeriesConfig.
    
    Parameters
    ----------
    series_list : List[SeriesConfig]
        List of series configurations (blocks information not used)
    data : Dict[str, Any]
        Configuration data (for clock default and block_names)
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping block names to block config dicts
    """
    blocks_dict = {}
    
    # Try to get block_names from data
    if 'block_names' in data:
        block_names_list = data['block_names']
        clock = data.get('clock', DEFAULT_CLOCK_FREQUENCY)
        for block_name in block_names_list:
            blocks_dict[block_name] = {'factors': 1, 'ar_lag': 1, 'clock': clock}
    else:
        # Default: create default block if no blocks specified
        clock = data.get('clock', DEFAULT_CLOCK_FREQUENCY)
        blocks_dict[DEFAULT_BLOCK_NAME] = {'factors': 1, 'ar_lag': 1, 'clock': clock}
    
    return blocks_dict

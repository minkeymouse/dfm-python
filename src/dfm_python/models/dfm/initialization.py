"""DFM parameter initialization functions.

This module contains all functions for initializing DFM state-space parameters
before running the EM algorithm. Initialization uses PCA-based factor extraction
and handles mixed-frequency data with tent kernel aggregation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from scipy.linalg import block_diag

from ...config.constants import (
    DEFAULT_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_IDENTITY_SCALE,
    MIN_EIGENVALUE,
    FREQUENCY_HIERARCHY,
    DEFAULT_REGULARIZATION,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
)
from ...functional.em import _DEFAULT_EM_CONFIG
from ...numeric.builder import build_dfm_slower_freq_observation_matrix, build_lag_matrix
from ...models.dfm.mixed_freq import build_slower_freq_idiosyncratic_chain, find_slower_frequency
from .tent import get_slower_freq_tent_weights
from ...numeric.stability import (
    ensure_covariance_stable,
    ensure_process_noise_stable,
    create_scaled_identity,
)
from ...numeric.estimator import (
    estimate_ar1_unified,
    estimate_variance_unified,
    estimate_var_unified,
    compute_initial_covariance_from_transition,
)
from ...utils.helper import handle_linear_algebra_error
from ...utils.validation import has_shape_with_min_dims
from ...logger import get_logger

_logger = get_logger(__name__)


def impute_for_init(data: np.ndarray) -> np.ndarray:
        """Simple imputation for initialization: forward fill → backward fill → mean.
        
        Following FRBNY pattern: used only when insufficient non-NaN observations
        for regression during initialization. The EM algorithm uses NaN-preserved data.
        
        Parameters
        ----------
        data : np.ndarray
            1D array (T,) with potential NaN values
        
        Returns
        -------
        np.ndarray
            Imputed 1D array with no NaN values
        """
        data_imputed = data.copy()
        mask = np.isnan(data_imputed)
        
        if not mask.any():
            return data_imputed
        
        # Vectorized forward fill using pandas (fast)
        try:
            import pandas as pd
            s = pd.Series(data_imputed)
            data_imputed = s.ffill().bfill().values
        except ImportError:
            # Fallback: numpy-based forward/backward fill (slower but no pandas dependency)
            # Forward fill
            idx = np.where(~mask)[0]
            if len(idx) > 0:
                # Use np.interp to forward fill
                indices = np.arange(len(data_imputed))
                valid_values = data_imputed[~mask]
                valid_indices = indices[~mask]
                if len(valid_values) > 0:
                    # Forward fill: use previous valid value
                    data_imputed = np.interp(indices, valid_indices, valid_values, 
                                           left=valid_values[0] if len(valid_values) > 0 else 0,
                                           right=valid_values[-1] if len(valid_values) > 0 else 0)
        
        # Fill remaining with mean
        mask = np.isnan(data_imputed)
        if mask.any():
            mean_val = np.nanmean(data_imputed)
            if np.isnan(mean_val):
                mean_val = 0.0  # Fallback if all NaN
            data_imputed[mask] = mean_val
        
        return data_imputed


def initialize_clock_freq_idio(
    res: np.ndarray,
    data_with_nans: np.ndarray,
    n_clock_freq: int,
    idio_indicator: Optional[np.ndarray],
    T: int,
    dtype: type = DEFAULT_DTYPE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize clock frequency idiosyncratic components (AR(1) for each series).
        
        Returns
        -------
        BM, SM, initViM
        """
        n_idio_clock = n_clock_freq if idio_indicator is None else int(np.sum(idio_indicator))
        BM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        SM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
        
        idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
        default_ar_coef = _DEFAULT_EM_CONFIG.slower_freq_ar_coef
        default_noise = _DEFAULT_EM_CONFIG.default_process_noise
        
        for i, idx in enumerate(idio_indices):
            res_i = data_with_nans[:, idx]
            non_nan_mask = ~np.isnan(res_i)
            if np.sum(non_nan_mask) > 1:
                first_non_nan = np.where(non_nan_mask)[0][0]
                last_non_nan = np.where(non_nan_mask)[0][-1]
                res_i_clean = res[first_non_nan:last_non_nan + 1, idx]
                
                if len(res_i_clean) > 1:
                    def _estimate_ar1_for_idio() -> np.ndarray:
                        # Use unified AR(1) estimation with raw data
                        y_ar = res_i_clean[1:]
                        x_ar = res_i_clean[:-1].reshape(-1, 1)
                        A_diag, Q_diag = estimate_ar1_unified(
                            y=y_ar.reshape(-1, 1),  # (T-1 x 1)
                            x=x_ar,  # (T-1 x 1)
                            V_smooth=None,  # Raw data mode
                            regularization=_DEFAULT_EM_CONFIG.matrix_regularization,
                            min_variance=default_noise,
                            default_ar_coef=default_ar_coef,
                            default_noise=default_noise,
                            dtype=dtype
                        )
                        return (A_diag[0] if len(A_diag) > 0 else default_ar_coef,
                                Q_diag[0] if len(Q_diag) > 0 else default_noise)
                    
                    BM[i, i], SM[i, i] = handle_linear_algebra_error(
                        _estimate_ar1_for_idio, "AR(1) estimation for idiosyncratic component",
                        fallback_func=lambda: (default_ar_coef, default_noise)
                    )
                else:
                    BM[i, i] = default_ar_coef
                    SM[i, i] = default_noise
            else:
                BM[i, i] = default_ar_coef
                SM[i, i] = default_noise
        
        # Initial covariance for clock frequency idio
        def _compute_initViM() -> np.ndarray:
            eye_BM = create_scaled_identity(n_idio_clock, DEFAULT_IDENTITY_SCALE, dtype=dtype)
            BM_sq = BM ** 2
            diag_inv = DEFAULT_IDENTITY_SCALE / np.diag(eye_BM - BM_sq)
            diag_inv = np.where(np.isfinite(diag_inv), diag_inv, np.full_like(diag_inv, DEFAULT_IDENTITY_SCALE))
            return np.diag(diag_inv) @ SM
        
        initViM = handle_linear_algebra_error(
            _compute_initViM, "initial covariance computation",
            fallback_func=lambda: SM.copy()
        )
        
        return BM, SM, initViM


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
    dtype: type = np.float32,
    impute_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
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
    impute_func : Optional[Callable[[np.ndarray], np.ndarray]], optional
        Function to impute NaN values for initialization fallback.
        Only used when insufficient non-NaN observations for regression.
        
    Returns
    -------
    C_i : np.ndarray
        Loading matrix for this block (N x (num_factors * max_lag_size))
    factors : np.ndarray
        Extracted factors (T x num_factors)
    """
    from ...encoder.pca import compute_principal_components
    from ...numeric.estimator import estimate_constrained_ols_unified
    from ...config.constants import DEFAULT_REGULARIZATION, DEFAULT_TENT_KERNEL_REGULARIZATION_MULTIPLIER
    
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
            
            # Use imputed data if insufficient non-NaN values (following FRBNY pattern)
            min_required = slower_freq_factors.shape[1] + 2
            if np.sum(non_nan_mask) < min_required:
                # Try data_for_extraction first (may already be imputed)
                series_data_attempt = data_for_extraction[tent_kernel_size:, series_idx_int]
                if np.sum(~np.isnan(series_data_attempt)) < min_required and impute_func is not None:
                    # Fallback: use imputation function (only for initialization)
                    series_data = impute_func(series_data)
                    _logger.debug(f"      Using imputed data for series {series_idx_int} (insufficient observations)")
                else:
                    series_data = series_data_attempt
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
                # Validate loadings are finite
                if np.any(~np.isfinite(loadings_constrained)):
                    _logger.warning(f"Computed loadings contain non-finite values for series {series_idx_int}. Skipping.")
                    continue
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
    A_i_block = A_i[:block_size, :block_size]
    Q_i_block = Q_i[:block_size, :block_size]
    reg = matrix_regularization or DEFAULT_REGULARIZATION
    V_0_i = compute_initial_covariance_from_transition(A_i_block, Q_i_block, regularization=reg, dtype=dtype)
    
    return A_i, Q_i, V_0_i


def initialize_block_factors(
        data_for_extraction: np.ndarray,
        data_with_nans: np.ndarray,
        blocks: np.ndarray,
        r: np.ndarray,
        n_blocks: int,
        n_clock_freq: int,
        tent_kernel_size: int,
        p: int,
        R_mat: Optional[np.ndarray],
        q: Optional[np.ndarray],
        N: int,
        T: int,
        indNaN: np.ndarray,
        max_lag_size: int,
        dtype: type = DEFAULT_DTYPE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize factors and transition matrices block-by-block using sequential PCA.
        
        **Estimation Graph Differences:**
        - Clock frequency series (first n_clock_freq columns): Standard factor model
          - Direct factor extraction via PCA on clock frequency data
          - Standard VAR(p) transition dynamics
        - Slower frequency series (last n_slower_freq columns): Tent kernel aggregation
          - Factor extraction with tent kernel aggregation using user-specified weights
          - R_mat and q enforce tent kernel constraints (derived from user-specified tent weights)
          - State dimension expanded by tent_kernel_size (e.g., 5 for weights [1,2,3,2,1])
        
        **Block-by-block extraction process:**
        - Block 1: Extracts factors from original data (data_for_extraction starts as original data)
        - Block 2+: Extracts factors from residuals (data_for_extraction becomes residuals after each block)
        
        This ensures each block captures different variance components, with factors orthogonal across blocks.
        
        Parameters
        ----------
        data_for_extraction : np.ndarray
            Data matrix (T x N). For Block 1, this is the original data (after cleaning).
            For subsequent blocks, this becomes residuals after removing previous blocks' contributions.
        data_with_nans : np.ndarray
            Data matrix with NaNs preserved (T x N)
        blocks : np.ndarray
            Block structure array (N x n_blocks)
        r : np.ndarray
            Number of factors per block (n_blocks,)
        n_blocks : int
            Number of blocks
        n_clock_freq : int
            Number of clock frequency series
        tent_kernel_size : int
            Tent kernel size for mixed-frequency aggregation
        p : int
            VAR lag order
        R_mat : np.ndarray, optional
            Constraint matrix for tent kernel aggregation
        q : np.ndarray, optional
            Constraint vector for tent kernel aggregation
        N : int
            Total number of series
        T : int
            Number of time steps
        indNaN : np.ndarray
            Boolean array indicating missing values
        max_lag_size : int
            Maximum lag size for loading matrix
        dtype : type
            Data type
            
        Returns
        -------
        A_factors : np.ndarray
            Block-diagonal transition matrix for factors
        Q_factors : np.ndarray
            Block-diagonal process noise covariance for factors
        V_0_factors : np.ndarray
            Block-diagonal initial state covariance for factors
        C : np.ndarray
            Observation/loading matrix (N x total_factor_dim)
        """
        C_list = []
        A_list = []
        Q_list = []
        V_0_list = []
        
        # Process each block sequentially
        # Block 1: data_for_extraction = original data
        # Block 2+: data_for_extraction = residuals after previous blocks
        for block_idx in range(n_blocks):
            num_factors_block = int(r[block_idx])
            block_series_indices = np.where(blocks[:, block_idx] > 0)[0]
            clock_freq_indices = block_series_indices[block_series_indices < n_clock_freq]
            slower_freq_indices = block_series_indices[block_series_indices >= n_clock_freq]
            
            _logger.info(f"  Initializing block {block_idx + 1}/{n_blocks}: "
                        f"{num_factors_block} factors, {len(block_series_indices)} series "
                        f"({len(clock_freq_indices)} clock, {len(slower_freq_indices)} slower)")
            
            # Extract factors and loadings for this block
            # Block 1: Uses original data (data_for_extraction = original data)
            # Block 2+: Uses residuals (data_for_extraction = residuals after previous blocks)
            C_i, factors = initialize_block_loadings(
                data_for_extraction, data_with_nans, clock_freq_indices, slower_freq_indices,
                num_factors_block, tent_kernel_size, R_mat, q,
                N, max_lag_size, _DEFAULT_EM_CONFIG.matrix_regularization, dtype,
                impute_func=impute_for_init
            )
            
            # Build lag matrix for transition equation
            lag_matrix = build_lag_matrix(factors, T, num_factors_block, tent_kernel_size, p, dtype)
            slower_freq_factors = lag_matrix[:, :num_factors_block * tent_kernel_size]
            
            # Pad and align factors
            if tent_kernel_size > 1 and slower_freq_factors.shape[0] < T:
                padding = np.zeros((tent_kernel_size - 1, slower_freq_factors.shape[1]), dtype=dtype)
                slower_freq_factors = np.vstack([padding, slower_freq_factors])
                if slower_freq_factors.shape[0] < T:
                    additional_padding = np.zeros((T - slower_freq_factors.shape[0], slower_freq_factors.shape[1]), dtype=dtype)
                    slower_freq_factors = np.vstack([slower_freq_factors, additional_padding])
                slower_freq_factors = slower_freq_factors[:T, :]
            
            # Update data_for_extraction: remove this block's contribution to get residuals for next block
            # After Block 1: data_for_extraction becomes residuals (original_data - Block1_contribution)
            # After Block 2: data_for_extraction becomes residuals (original_data - Block1 - Block2)
            if data_for_extraction.shape[0] != slower_freq_factors.shape[0]:
                slower_freq_factors = slower_freq_factors[:data_for_extraction.shape[0], :]
            data_for_extraction = data_for_extraction - slower_freq_factors @ C_i[:, :num_factors_block * tent_kernel_size].T
            data_with_nans = data_for_extraction.copy()
            data_with_nans[indNaN] = np.nan
            
            C_list.append(C_i)
            
            # Initialize transition matrices
            A_i, Q_i, V_0_i = initialize_block_transition(
                lag_matrix, factors, num_factors_block, max_lag_size, p, T,
                _DEFAULT_EM_CONFIG.regularization, _DEFAULT_EM_CONFIG.default_transition_coef,
                _DEFAULT_EM_CONFIG.default_process_noise, _DEFAULT_EM_CONFIG.matrix_regularization,
                _DEFAULT_EM_CONFIG.eigenval_floor, dtype
            )
            
            A_list.append(A_i)
            Q_list.append(Q_i)
            V_0_list.append(V_0_i)
        
        # Concatenate loadings
        C = np.hstack(C_list) if C_list else np.zeros((N, 0), dtype=dtype)
        
        # Build block-diagonal matrices
        if A_list:
            A_factors = block_diag(*A_list)
            Q_factors = block_diag(*Q_list)
            V_0_factors = block_diag(*V_0_list)
        else:
            empty_matrix = np.zeros((0, 0), dtype=dtype)
            A_factors = Q_factors = V_0_factors = empty_matrix
        
        return A_factors, Q_factors, V_0_factors, C


def add_idiosyncratic_observation_matrix(
        C: np.ndarray,
        N: int,
        n_clock_freq: int,
        n_slower_freq: int,
        idio_indicator: Optional[np.ndarray],
        clock: str,
        tent_kernel_size: int,
        tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
        dtype: type = DEFAULT_DTYPE
    ) -> np.ndarray:
        """Add idiosyncratic components to observation matrix C.
        
        Estimation graph differences:
        - Clock frequency: Identity matrix (direct observation, one idio component per series)
        - Slower frequency: Tent kernel chain observation matrix (uses user-specified tent weights)
        
        Parameters
        ----------
        tent_weights_dict : Optional[Dict[str, np.ndarray]]
            User-specified tent kernel weights (e.g., {'monthly': [1,2,3,2,1]}).
            These are NOT auto-computed - must be provided by user in config.
        
        Returns
        -------
        C : np.ndarray
            Updated observation matrix with idiosyncratic components
        """
        # Clock frequency: identity matrix for each series (direct observation)
        if idio_indicator is not None:
            eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
            idio_indicator_bool = idio_indicator.astype(bool)
            C = np.hstack([C, eyeN[:, idio_indicator_bool]])
        else:
            # Default: all clock frequency series have idiosyncratic components
            if n_clock_freq > 0:
                eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
                C = np.hstack([C, eyeN[:, :n_clock_freq]])
        
        # Slower frequency: tent kernel chain observation matrix
        # Uses user-specified tent weights (e.g., [1,2,3,2,1]) from tent_weights_dict
        if n_slower_freq > 0:
            # Determine slower frequency using helper method
            slower_freq = find_slower_frequency(clock, tent_weights_dict)
            
            # Get user-specified tent weights (NOT auto-computed)
            if tent_weights_dict and slower_freq in tent_weights_dict:
                tent_weights = tent_weights_dict[slower_freq].astype(dtype)
            else:
                # Fallback: generate default weights if user didn't specify (shouldn't happen in practice)
                tent_weights = get_slower_freq_tent_weights(slower_freq or 'q', clock, tent_kernel_size, dtype)
            
            # Build observation matrix using user-specified tent weights
            C_slower_freq = build_dfm_slower_freq_observation_matrix(N, n_clock_freq, n_slower_freq, tent_weights, dtype)
            C = np.hstack([C, C_slower_freq])
        
        return C


def initialize_observation_noise(
        data_with_nans: np.ndarray,
        N: int,
        idio_indicator: Optional[np.ndarray],
        n_clock_freq: int,
        dtype: type = DEFAULT_DTYPE
    ) -> np.ndarray:
        """Initialize observation noise covariance R from residuals.
        
        Missing values (NaN) are handled via nan-aware statistics - only valid observations
        are used for variance estimation. NaN will be handled by Kalman filter during EM.
        
        Returns
        -------
        R : np.ndarray
            Observation noise covariance (N x N, diagonal)
        """
        # Ensure 2D
        if data_with_nans.ndim != 2:
            data_with_nans = data_with_nans.reshape(-1, N) if data_with_nans.size > 0 else np.zeros((1, N), dtype=dtype)
        
        T_res, N_res = data_with_nans.shape
        default_obs_noise = _DEFAULT_EM_CONFIG.default_observation_noise
        
        # Use unified variance estimation with raw residuals (handles NaN via nan-aware stats)
        if T_res <= 1:
            # create_scaled_identity already imported at top
            R = create_scaled_identity(N_res, default_obs_noise, dtype)
        else:
            # Compute residuals (data itself, since we're initializing from raw data)
            # estimate_variance_unified uses nan-aware variance if residuals contain NaN
            R = estimate_variance_unified(
                residuals=data_with_nans,  # Raw data as "residuals" for initialization (may contain NaN)
                X=None,  # Not using smoothed expectations mode
                EZ=None,
                C=None,
                V_smooth=None,
                min_variance=default_obs_noise,
                default_variance=default_obs_noise,
                dtype=dtype
            )
        
        # Set variances for idiosyncratic series to default
        idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
        all_indices = np.unique(np.concatenate([idio_indices, np.arange(n_clock_freq, N, dtype=np.int32)]))
        R[np.ix_(all_indices, all_indices)] = np.diag(np.full(len(all_indices), default_obs_noise, dtype=dtype))
        
        return R


def initialize_parameters(
    x: np.ndarray,
    r: np.ndarray,
    p: int,
    blocks: np.ndarray,
    R_mat: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    n_slower_freq: int = 0,
    idio_indicator: Optional[np.ndarray] = None,
    clock: str = DEFAULT_CLOCK_FREQUENCY,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize DFM state-space parameters.
        
        Setup:
        - Clock frequency oriented dataframe (first n_clock_freq columns)
        - Optional slower frequency series (last n_slower_freq columns, e.g., weekly/monthly, monthly/quarterly)
        - Tent kernel weights are user-specified (e.g., [1,2,3,2,1]) via tent_weights_dict in config
        - Build matrices (A, C, Q, R, Z_0, V_0) and pass to Kalman filter
        
        Estimation Graph Differences:
        - Clock frequency series: Standard factor model estimation (direct observation)
        - Slower frequency series: Tent kernel aggregation with user-specified weights
          (e.g., monthly data aggregated to weekly clock using tent weights [1,2,3,2,1])
        
        Process:
        1. Build factor matrices from clock frequency data (block-by-block PCA)
           - Clock freq: Direct factor extraction
           - Slower freq: Uses tent kernel aggregation with user-specified weights
        2. Add idiosyncratic components to observation matrix C
           - Clock freq: Identity matrix (one idio component per series)
           - Slower freq: Tent kernel chain observation matrix
        3. Build observation noise R
        4. Build idiosyncratic transition matrices
           - Clock freq: AR(1) for each series (BM, SM, initViM)
           - Slower freq: Tent kernel chain (BQ, SQ, initViQ) using user-specified weights
        5. Assemble block-diagonal matrices (A, Q, V_0)
        6. Initialize Z_0 and apply stability checks
    """
    T, N = x.shape
    dtype = DEFAULT_DTYPE
    
    n_blocks = blocks.shape[1]
    n_clock_freq = N - n_slower_freq  # Number of clock frequency series
    
    # Kalman filter handles missing values natively via masked arrays - preserve NaN
    x_clean = np.where(np.isinf(x), np.nan, x)  # Replace Inf with NaN, keep existing NaN
    
    # Check data scale for numerical stability
    # Detect potential RobustScaler issues (IQR≈0) vs StandardScaler (std≈1, mean≈0)
    valid_mask = np.isfinite(x_clean)
    if valid_mask.any():
        data_std = np.nanstd(x_clean)
        data_mean = np.nanmean(x_clean)
        data_median = np.nanmedian(x_clean)
        data_iqr = np.nanpercentile(x_clean, 75) - np.nanpercentile(x_clean, 25)
        
        # Check for RobustScaler issues: IQR≈0 indicates potential scaling problems
        # StandardScaler: mean≈0, std≈1, IQR≈1.35 (for normal distribution)
        # RobustScaler with IQR≈0: can produce extreme values
        has_zero_iqr = data_iqr < 1e-6
        
        # Check for scale mismatch
        has_scale_mismatch = (
            data_std > 10 or abs(data_mean) > 3 or abs(data_median) > 3 or
            (data_std < 0.01 and not has_zero_iqr)  # Very small std (might indicate no scaling)
        )
        
    # Initialize data for factor extraction
    # Block 1: original data. Subsequent blocks: residuals. NaN preserved for Kalman filter
    data_for_extraction = x_clean.copy()
    data_with_nans = x_clean.copy()
    indNaN = np.isnan(x_clean)  # Track NaN positions for initialization (only)
    
    # Determine tent kernel size from user-specified tent weights
    # Tent weights are user-specified (e.g., [1,2,3,2,1] → tent_kernel_size=5)
    if R_mat is not None:
        tent_kernel_size = R_mat.shape[1]  # R_mat derived from user-specified tent weights
    elif tent_weights_dict:
        # Extract size from user-specified tent weights (e.g., {'monthly': [1,2,3,2,1]} → size=5)
        first_weights = next(iter(tent_weights_dict.values()))
        tent_kernel_size = len(first_weights)
    else:
        tent_kernel_size = _DEFAULT_EM_CONFIG.tent_kernel_size  # Fallback default
    # State dimension per factor = max(p + 1, tent_kernel_size)
    # For slower freq: state dimension expanded by tent_kernel_size (reflecting aggregation structure)
    max_lag_size = max(p + 1, tent_kernel_size)
    
    # Set initial observations as NaN for slower-frequency aggregation
    if tent_kernel_size > 1:
        data_with_nans[:tent_kernel_size-1, :] = np.nan
    
    # === BUILD STATE-SPACE MATRICES ===
    # Note: Estimation graph differs between clock and slower frequency series
    # - Clock frequency: Standard factor model (direct observation at clock frequency)
    # - Slower frequency: Tent kernel aggregation using user-specified weights (e.g., [1,2,3,2,1])
    #   Tent weights come from tent_weights_dict in config (user-specified, not auto-computed)
    
    # 1. Build factor matrices (A_factors, Q_factors, V_0_factors, C)
    #    - Clock freq series: Direct factor extraction via PCA
    #    - Slower freq series: Factor extraction with tent kernel aggregation (uses user-specified tent_weights_dict)
    A_factors, Q_factors, V_0_factors, C = initialize_block_factors(
        data_for_extraction, data_with_nans, blocks, r, n_blocks, n_clock_freq, tent_kernel_size,
        p, R_mat, q, N, T, indNaN, max_lag_size, dtype
    )
    
    # 2. Add idiosyncratic components to observation matrix C
    #    - Clock freq: Identity matrix (one idio component per series, direct observation)
    #    - Slower freq: Tent kernel chain observation matrix (uses user-specified tent_weights_dict)
    C = add_idiosyncratic_observation_matrix(
        C, N, n_clock_freq, n_slower_freq, idio_indicator, clock, tent_kernel_size, tent_weights_dict, dtype
    )
    
    # Normalize C columns (vectorized for efficiency)
    norms = np.linalg.norm(C, axis=0)
    valid_mask = norms > MIN_EIGENVALUE
    if np.any(valid_mask):
        # Broadcasting: C[:, valid_mask] is (N, n_valid), norms[valid_mask] is (n_valid,)
        # Divide each column by its norm
        C[:, valid_mask] = C[:, valid_mask] / norms[valid_mask]
    
    # 3. Build observation noise R
    R = initialize_observation_noise(data_with_nans, N, idio_indicator, n_clock_freq, dtype)
    
    # 4. Build idiosyncratic transition matrices
    #    - Clock frequency: AR(1) for each series (standard time series model)
    #    - Slower frequency: Tent kernel chain (uses user-specified tent_weights_dict for aggregation)
    BM, SM, initViM = initialize_clock_freq_idio(
        data_for_extraction, data_with_nans, n_clock_freq, idio_indicator, T, dtype=dtype
    )
    
    # Slower frequency: tent kernel chain (user-specified weights from tent_weights_dict)
    if n_slower_freq == 0:
        BQ = SQ = initViQ = np.zeros((0, 0), dtype=dtype)
    else:
        rho0 = _DEFAULT_EM_CONFIG.slower_freq_ar_coef
        sig_e = np.diag(R[n_clock_freq:, n_clock_freq:]) / _DEFAULT_EM_CONFIG.slower_freq_variance_denominator
        sig_e = np.where(np.isfinite(sig_e), sig_e, _DEFAULT_EM_CONFIG.default_observation_noise)
        BQ, SQ, initViQ = build_slower_freq_idiosyncratic_chain(n_slower_freq, tent_kernel_size, rho0, sig_e, dtype)
    
    # 5. Assemble block-diagonal matrices: A = [A_factors, BM, BQ], Q = [Q_factors, SM, SQ], V_0 = [V_0_factors, initViM, initViQ]
    A = block_diag(A_factors, BM, BQ)
    Q = block_diag(Q_factors, SM, SQ)
    V_0 = block_diag(V_0_factors, initViM, initViQ)
    
    # 6. Initial state: Z_0 = zeros
    m = int(A.shape[0]) if A.size > 0 and has_shape_with_min_dims(A, min_dims=1) else 0
    Z_0 = np.zeros(m, dtype=dtype)
    
    # 7. Stability checks
    Q = ensure_process_noise_stable(Q, min_eigenval=_DEFAULT_EM_CONFIG.eigenval_floor, warn=True, dtype=dtype)
    V_0 = ensure_covariance_stable(V_0, min_eigenval=_DEFAULT_EM_CONFIG.eigenval_floor)
    
    return A, C, Q, R, Z_0, V_0

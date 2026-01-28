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
    estimate_ar1,
    estimate_variance,
    estimate_var,
    compute_initial_covariance_from_transition,
)
from ...utils.helper import handle_linear_algebra_error
from ...utils.validation import has_shape_with_min_dims
from ...utils.errors import NumericalError
from ...logger import get_logger
from ...config.constants import DEFAULT_NAN_K, DEFAULT_NAN_METHOD

_logger = get_logger(__name__)


def remNaNs_spline(x: np.ndarray, optNaN: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Spline interpolation to remove NaNs (matching FRBNY MATLAB remNaNs_spline).
    
    This function creates splined data (xBal) with no NaNs, while preserving
    the original NaN positions (indNaN) for later use. This matches FRBNY's
    approach of using splined data for initialization.
    
    Parameters
    ----------
    x : np.ndarray
        Input data (T x N) with potential NaN values
    optNaN : Dict[str, Any]
        Options for NaN handling:
        - method: int, method for NaN removal (2 = remove leading/trailing, 3 = spline)
        - k: int, spline order (default 3)
    
    Returns
    -------
    xBal : np.ndarray
        Splined data with no NaNs (T x N)
    indNaN : np.ndarray
        Boolean array (T x N) indicating original NaN positions
    """
    T, N = x.shape
    indNaN = np.isnan(x)
    xBal = x.copy()
    
    method = optNaN.get('method', DEFAULT_NAN_METHOD)
    k = optNaN.get('k', DEFAULT_NAN_K)
    
    # Method 2: Remove leading and trailing NaNs, then spline
    if method == 2:
        # For each series, find leading and trailing NaNs
        for j in range(N):
            col = x[:, j]
            valid_mask = ~np.isnan(col)
            
            if not valid_mask.any():
                # All NaN - fill with zeros
                xBal[:, j] = 0.0
                continue
            
            # Find first and last valid indices
            valid_indices = np.where(valid_mask)[0]
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]
            
            # Set leading/trailing NaNs to NaN (will be handled by spline)
            # Keep middle NaNs for spline interpolation
            col_processed = col.copy()
            if first_valid > 0:
                col_processed[:first_valid] = np.nan
            if last_valid < T - 1:
                col_processed[last_valid + 1:] = np.nan
            
            # Spline interpolation for remaining NaNs
            if np.isnan(col_processed).any():
                try:
                    from scipy.interpolate import interp1d
                    valid_mask_processed = ~np.isnan(col_processed)
                    if valid_mask_processed.sum() >= k + 1:  # Need at least k+1 points for spline
                        valid_indices_processed = np.where(valid_mask_processed)[0]
                        valid_values = col_processed[valid_indices_processed]
                        f = interp1d(valid_indices_processed, valid_values, kind='cubic', 
                                    fill_value='extrapolate', bounds_error=False)
                        all_indices = np.arange(T)
                        col_interpolated = f(all_indices)
                        # Only fill NaNs, keep existing values
                        nan_mask = np.isnan(col_processed)
                        col_processed[nan_mask] = col_interpolated[nan_mask]
                    else:
                        # Not enough points for spline - use linear
                        from scipy.interpolate import interp1d
                        valid_indices_processed = np.where(valid_mask_processed)[0]
                        valid_values = col_processed[valid_indices_processed]
                        f = interp1d(valid_indices_processed, valid_values, kind='linear',
                                    fill_value='extrapolate', bounds_error=False)
                        all_indices = np.arange(T)
                        col_interpolated = f(all_indices)
                        nan_mask = np.isnan(col_processed)
                        col_processed[nan_mask] = col_interpolated[nan_mask]
                except Exception:
                    # Fallback: forward/backward fill
                    import pandas as pd
                    s = pd.Series(col_processed)
                    col_processed = s.ffill().bfill().fillna(0.0).values
            
            xBal[:, j] = col_processed
    
    # Method 3: Spline all NaNs (used during EM)
    elif method == 3:
        for j in range(N):
            col = x[:, j]
            valid_mask = ~np.isnan(col)
            
            if not valid_mask.any():
                xBal[:, j] = 0.0
                continue
            
            if valid_mask.sum() >= k + 1:
                try:
                    from scipy.interpolate import interp1d
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = col[valid_indices]
                    f = interp1d(valid_indices, valid_values, kind='cubic',
                                fill_value='extrapolate', bounds_error=False)
                    all_indices = np.arange(T)
                    xBal[:, j] = f(all_indices)
                except Exception:
                    # Fallback: linear interpolation
                    from scipy.interpolate import interp1d
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = col[valid_indices]
                    f = interp1d(valid_indices, valid_values, kind='linear',
                                fill_value='extrapolate', bounds_error=False)
                    all_indices = np.arange(T)
                    xBal[:, j] = f(all_indices)
            else:
                # Not enough points - forward/backward fill
                import pandas as pd
                s = pd.Series(col)
                xBal[:, j] = s.ffill().bfill().fillna(0.0).values
    else:
        # Default: simple forward/backward fill
        import pandas as pd
        df = pd.DataFrame(x)
        xBal = df.ffill().bfill().fillna(0.0).values
    
    # Ensure no NaNs remain
    if np.isnan(xBal).any():
        xBal = np.nan_to_num(xBal, nan=0.0)
    
    return xBal, indNaN


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
            # Bug fix 1.1: Use same array for mask computation and slicing
            # Compute mask from res (data_for_extraction) to match the array we'll slice
            res_i = res[:, idx]
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
                        A_diag, Q_diag = estimate_ar1(
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
        # Bug fix 1.2: Correct AR(1) initial variance formula: Var(u) = σ²/(1-ρ²)
        # For diagonal case: initViM[i,i] = SM[i,i] / (1 - BM[i,i]²)
        def _compute_initViM() -> np.ndarray:
            initViM = np.zeros_like(SM)
            for i in range(n_idio_clock):
                denominator = 1.0 - BM[i, i] ** 2
                if abs(denominator) > 1e-10 and np.isfinite(denominator):
                    initViM[i, i] = SM[i, i] / denominator
                else:
                    # Fallback for near-unity AR coefficient
                    initViM[i, i] = SM[i, i] * DEFAULT_IDENTITY_SCALE
            return initViM
        
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
    from ...layer.pca import compute_principal_components
    from ...numeric.estimator import estimate_constrained_ols
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
        
        # Compute covariance matrix efficiently
        # Use cleaned data directly (NaN already replaced with 0) for faster computation
        if clock_freq_data_centered_clean.shape[0] <= 1:
            cov_data = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)
        elif len(clock_freq_indices) == 1:
            cov_data = np.atleast_2d(np.nanvar(clock_freq_data_centered, axis=0, ddof=0))
        else:
            # Use cleaned data directly for covariance (faster than filtering)
            # Since NaN was replaced with 0, we can compute covariance directly
            cov_data = np.cov(clock_freq_data_centered_clean.T)
            cov_data = (cov_data + cov_data.T) / 2  # Ensure symmetry
        
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
            
            # FRBNY pattern: Always use splined data (data_for_extraction) for initialization
            # If insufficient non-NaN observations in original data, use splined data
            series_data_original = data_with_nans[tent_kernel_size:, series_idx_int]
            series_data_splined = data_for_extraction[tent_kernel_size:, series_idx_int]  # Already splined
            
            # Check if we have enough non-NaN observations in original data
            non_nan_mask = ~np.isnan(series_data_original)
            min_required = slower_freq_factors.shape[1] + 2
            
            if np.sum(non_nan_mask) < min_required:
                # Use splined data (matches MATLAB line 704: xx_j = res(pC:end,j))
                series_data = series_data_splined
                # Still align with original data pattern (use non_nan_mask for alignment)
                # This ensures tent kernel factors align with weeks where monthly data actually exists
                slower_freq_factors_clean = slower_freq_factors[tent_kernel_size:][non_nan_mask, :]
                series_data_clean = series_data[non_nan_mask]
            else:
                # Use original data (with NaNs) - align with actual observations
                series_data = series_data_original
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
                loadings_constrained = estimate_constrained_ols(
                    y=series_data_clean,
                    X=slower_freq_factors_clean,
                    R=constraint_matrix_block,
                    q=constraint_vector_block,
                    V_smooth=None,  # Raw data mode
                    regularization=reg,
                    dtype=dtype
                )
                # Validate loadings are finite - raise error if not (no fallback)
                if np.any(~np.isfinite(loadings_constrained)):
                    raise NumericalError(
                        f"Constrained OLS returned non-finite values for series {series_idx_int}. "
                        f"This indicates numerical instability. Check data quality and tent kernel configuration.",
                        details=f"Series index: {series_idx_int}, tent_kernel_size: {tent_kernel_size}, "
                                f"num_factors: {num_factors}, regularization: {reg}"
                    )
                C_i[series_idx_int, :num_factors * tent_kernel_size] = loadings_constrained
            except (np.linalg.LinAlgError, ValueError) as e:
                # No fallback - raise error to surface the problem
                raise NumericalError(
                    f"Constrained OLS failed for series {series_idx_int}: {e}. "
                    f"This indicates the matrix is too ill-conditioned even with splined data. "
                    f"Check data quality, tent kernel configuration, or increase regularization.",
                    details=f"Series index: {series_idx_int}, tent_kernel_size: {tent_kernel_size}, "
                            f"num_factors: {num_factors}, regularization: {reg}, error: {str(e)}"
                ) from e
    
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
    # Bug fix 1.3: Correct VAR(p) regressor construction
    # For VAR(p): y_t = A_1*y_{t-1} + ... + A_p*y_{t-p} + e_t
    # lag_matrix structure: [lag_0, lag_1, ..., lag_{num_lags-1}]
    # where each lag_i occupies columns [i*num_factors : (i+1)*num_factors]
    # Current state (y_t) = lag_0 = columns 0:num_factors
    # Lagged states [y_{t-1}, ..., y_{t-p}] = columns num_factors:(p+1)*num_factors
    n_cols = min(num_factors, lag_matrix.shape[1])
    current_state = lag_matrix[:, :n_cols] if n_cols > 0 else np.zeros((T, num_factors), dtype=dtype)
    # For VAR(p), we need p lags, so lagged_state should be columns num_factors to (p+1)*num_factors
    lag_cols_end = min(num_factors * (p + 1), lag_matrix.shape[1])
    lagged_state = lag_matrix[:, num_factors:lag_cols_end] if lag_cols_end > num_factors else np.zeros((T, num_factors * p), dtype=dtype)
    
    # Initialize transition matrix
    default_A_block = create_scaled_identity(num_factors, default_transition_coef, dtype)
    shift_size = num_factors * (max_lag_size - 1)
    default_shift = create_scaled_identity(shift_size, DEFAULT_IDENTITY_SCALE, dtype=dtype) if shift_size > 0 else np.zeros((0, 0), dtype=dtype)
    
    # Estimate transition coefficients using unified VAR estimation
    if T > p and lagged_state.shape[1] > 0:
        try:
            # Use unified VAR estimation (raw data mode)
            A_transition, Q_transition = estimate_var(
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
            
            # Check for NaN/Inf values in estimated matrices
            if np.any(~np.isfinite(A_transition)) or np.any(~np.isfinite(Q_transition)):
                # Fallback to default if estimation produced non-finite values
                A_i[:num_factors, :num_factors] = default_A_block
                Q_i = np.zeros((block_size, block_size), dtype=dtype)
                Q_i[:num_factors, :num_factors] = create_scaled_identity(num_factors, default_process_noise, dtype)
            else:
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
    
    # Ensure A_i doesn't contain NaN/Inf before computing initial covariance
    if np.any(~np.isfinite(A_i)):
        # Replace NaN/Inf with default values
        A_i = np.where(np.isfinite(A_i), A_i, 0.0).astype(dtype)
        # Reset to default if A_i is all zeros or invalid
        A_i[:num_factors, :num_factors] = default_A_block
    
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
                N, max_lag_size, _DEFAULT_EM_CONFIG.matrix_regularization, dtype
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
            # 
            # CRITICAL TIME-ALIGNMENT ASSERTION:
            # With tent kernels, factor_t contributes to y_{t+k} (not y_t directly).
            # However, for block residualization, we assume factor_t contributes to y_t.
            # This is valid because:
            # 1. Factors are extracted at clock frequency (aligned with clock-freq data)
            # 2. Tent kernel expansion happens in state dimension, not time dimension
            # 3. The residualization uses the factor values directly, which are already time-aligned
            # 
            # Assertion: factors and data must have same time dimension for valid residualization
            if data_for_extraction.shape[0] != slower_freq_factors.shape[0]:
                # Truncate factors to match data time dimension
                if slower_freq_factors.shape[0] > data_for_extraction.shape[0]:
                    _logger.warning(
                        f"Time alignment: Truncating slower_freq_factors from {slower_freq_factors.shape[0]} "
                        f"to {data_for_extraction.shape[0]} timesteps for block residualization. "
                        f"This may indicate timing misalignment."
                    )
                slower_freq_factors = slower_freq_factors[:data_for_extraction.shape[0], :]
            
            # Assert: factors and data are time-aligned (same T dimension)
            # CRITICAL: With tent kernels, factor_t contributes to y_{t+k}, but for block residualization
            # we assume factor_t contributes to y_t. This is valid because factors are extracted at
            # clock frequency and are already time-aligned with clock-freq data.
            if data_for_extraction.shape[0] != slower_freq_factors.shape[0]:
                _logger.error(
                    f"TIME ALIGNMENT VIOLATION in block residualization: "
                    f"data_for_extraction has {data_for_extraction.shape[0]} timesteps, "
                    f"slower_freq_factors has {slower_freq_factors.shape[0]} timesteps. "
                    f"This breaks the assumption that factor_t contributes to y_t."
                )
            assert data_for_extraction.shape[0] == slower_freq_factors.shape[0], (
                f"Time alignment violation: data_for_extraction has {data_for_extraction.shape[0]} timesteps, "
                f"but slower_freq_factors has {slower_freq_factors.shape[0]} timesteps. "
                f"This breaks the assumption that factor_t contributes to y_t in residualization."
            )
            _logger.debug(
                f"Time alignment validated: data_for_extraction and slower_freq_factors both have "
                f"{data_for_extraction.shape[0]} timesteps (block residualization safe)"
            )
            
            # Compute residualization: data - factors @ C.T
            # This assumes factor_t directly contributes to y_t (valid for clock-aligned factors)
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
            # estimate_variance uses nan-aware variance if residuals contain NaN
            R = estimate_variance(
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
        
    # FRBNY pattern: Spline data first, then preserve NaN positions
    # This ensures we have complete data for initialization (especially for slower-frequency series)
    optNaN = {
        'method': DEFAULT_NAN_METHOD,  # Method 2: remove leading/trailing, then spline
        'k': DEFAULT_NAN_K  # Spline order (default 3, matches MATLAB)
    }
    xBal, indNaN = remNaNs_spline(x_clean, optNaN)
    
    # Create two versions:
    # - xBal: Splined data (no NaNs) - used for factor extraction and initialization
    # - xNaN: Original data with NaNs preserved - used for alignment and EM algorithm
    data_for_extraction = xBal.copy()  # Splined data for initialization
    data_with_nans = x_clean.copy()  # Original data with NaNs (for EM algorithm)
    
    # Determine tent kernel size from user-specified tent weights
    # Tent weights are user-specified (e.g., [1,2,3,2,1] → tent_kernel_size=5)
    if R_mat is not None:
        tent_kernel_size = R_mat.shape[1]  # R_mat derived from user-specified tent weights
    elif tent_weights_dict:
        # Extract size from user-specified tent weights (e.g., {'monthly': [1,2,3,2,1]} → size=5)
        first_weights = next(iter(tent_weights_dict.values()))
        tent_kernel_size = len(first_weights)
    else:
        # Single-frequency (no tent weights): no aggregation kernel.
        # Using a large default here unnecessarily inflates the state dimension (max_lag_size),
        # masks initial observations, and slows EM for daily/weekly data.
        tent_kernel_size = 1
    # State dimension per factor = max(p + 1, tent_kernel_size)
    # For slower freq: state dimension expanded by tent_kernel_size (reflecting aggregation structure)
    max_lag_size = max(p + 1, tent_kernel_size)
    
    # Set initial observations as NaN only when tent kernel aggregation is actually used.
    if tent_kernel_size > 1 and (R_mat is not None or n_slower_freq > 0):
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
    # NOTE: This normalization happens BEFORE Q, V_0, Z_0 are built,
    # so invariance rescaling is not needed here. The normalized C will be used
    # to build Q, V_0, Z_0, ensuring consistency from the start.
    # In EM updates, normalization happens AFTER Q, V_0, Z_0 exist, so rescaling is required.
    norms = np.linalg.norm(C, axis=0)
    valid_mask = norms > MIN_EIGENVALUE
    if np.any(valid_mask):
        # Broadcasting: C[:, valid_mask] is (N, n_valid), norms[valid_mask] is (n_valid,)
        # Divide each column by its norm
        C[:, valid_mask] = C[:, valid_mask] / norms[valid_mask]
        n_normalized = np.sum(valid_mask)
        _logger.debug(
            f"Initialization: Normalized {n_normalized}/{C.shape[1]} C columns "
            f"(before Q, V_0, Z_0 construction - no rescaling needed)"
        )
    
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

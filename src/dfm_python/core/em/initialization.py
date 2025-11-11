"""EM initialization routines.

This module contains:
- init_conditions: compute initial A, C, Q, R, Z_0, V_0 via PCA and OLS
"""

from typing import Optional, Tuple, Dict, Any, TypedDict
import logging
import warnings
import numpy as np
from scipy.linalg import inv, pinv, block_diag

# Common exception types for numerical operations
_NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)

from ...config import DFMConfig
from ..numeric import (
    _ensure_symmetric,
    _ensure_square_matrix,
    _compute_principal_components,
    _compute_covariance_safe,
    _compute_variance_safe,
    _clean_matrix,
    _ensure_positive_definite,
    _ensure_innovation_variance_minimum,
    _compute_regularization_param,
    _apply_ar_clipping,
    _cap_max_eigenvalue,
    _estimate_ar_coefficient,
    _safe_divide,
    _check_finite,
    MIN_DIAGONAL_VARIANCE,
)
from ..helpers import (
    safe_get_attr,
    update_block_diag,
    estimate_ar_coefficients_ols,
    compute_innovation_covariance,
    compute_sufficient_stats,
    validate_params,
    get_tent_weights,
    infer_nQ,
    stabilize_cov,
    reg_inv,
    update_loadings,
    safe_time_index,
    extract_3d_matrix_slice,
    compute_obs_cov,
    clean_variance_array,
    safe_array_operation,
    get_block_indices,
    compute_block_slice_indices,
    extract_block_matrix,
    update_block_in_matrix,
    append_or_initialize,
    create_empty_matrix,
    reshape_to_column_vector,
    reshape_to_row_vector,
    pad_matrix_to_shape,
    safe_numerical_operation,
    get_matrix_shape,
    has_valid_data,
    ensure_minimum_size,
)
from ...utils.aggregation import group_series_by_frequency
from ...data_loader import rem_nans_spline
from ...utils.aggregation import (
    FREQUENCY_HIERARCHY,
    generate_R_mat,
)
_logger = logging.getLogger(__name__)
# Numerical stability constants
DEFAULT_AR_COEFFICIENT = 0.1  # Default AR coefficient when estimation fails
DEFAULT_INNOVATION_VARIANCE = 0.1  # Default innovation variance
DEFAULT_IDIO_AR = 0.9  # Default AR coefficient for idiosyncratic
DEFAULT_IDIO_VAR = 0.1  # Default innovation variance for idiosyncratic
DEFAULT_IDIO_COV = 0.1  # Default initial covariance for idiosyncratic
DEFAULT_OBSERVATION_VARIANCE = 1e-4  # Default observation variance (R diagonal)
MIN_INNOVATION_VARIANCE = 1e-8  # Minimum innovation variance for factor evolution
MIN_OBSERVATION_VARIANCE = 1e-8  # Minimum observation variance
MIN_EIGENVALUE_THRESHOLD = 1e-8  # Minimum eigenvalue for positive definiteness
MIN_VARIANCE_THRESHOLD = 1e-10  # Minimum variance threshold for validation

# Initialization constants
MIN_DATA_COVERAGE_RATIO = 0.5  # Minimum ratio of series required for block initialization (50%)
MIN_EIGENVALUE_ABSOLUTE = 0.1  # Absolute minimum eigenvalue for Block_Global (ensures factor can evolve)
MIN_EIGENVALUE_RELATIVE = 0.1  # Relative minimum eigenvalue (10% of max)
MIN_LOADING = 0.1  # Minimum absolute loading before scaling
MAX_LOADING = 0.5  # Target maximum absolute loading after scaling
MIN_AR = 0.1  # Minimum AR coefficient for Block_Global
FALLBACK_AR = 0.9  # Fallback transition coefficient
FALLBACK_SCALE = 0.1  # Scale for random fallback

class NaNHandlingOptions(TypedDict, total=False):
    """Options for handling missing data (NaN values).
    
    Attributes
    ----------
    method : int
        Method for handling NaNs:
        - 1: Spline interpolation (recommended)
        - 2: Forward fill then backward fill
        - 3: Mean imputation
    k : int
        Spline interpolation order (only used if method=1).
        Typically 3 (cubic spline).
    """
    method: int
    k: int


def init_conditions(
    x: np.ndarray,
    r: np.ndarray,
    p: int,
    blocks: np.ndarray,
    opt_nan: NaNHandlingOptions,
    Rcon: Optional[np.ndarray],
    q: Optional[np.ndarray],
    nQ: Optional[int],
    i_idio: np.ndarray,
    clock: str = 'm',
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    frequencies: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute initial parameter estimates for DFM via PCA and OLS.
    
    This function computes initial values for the DFM parameters:
    - A: Transition matrix (via AR regression on factors)
    - C: Loading matrix (via PCA on data residuals)
    - Q: Innovation covariance (via residual variance)
    - R: Observation covariance (via idiosyncratic variance)
    - Z_0: Initial state (via unconditional mean)
    - V_0: Initial covariance (via stationary covariance)
    
    Parameters
    ----------
    x : np.ndarray
        Standardized data matrix (T x N)
    r : np.ndarray
        Number of factors per block (n_blocks,)
    p : int
        AR lag order (typically 1)
    blocks : np.ndarray
        Block structure (N x n_blocks)
    opt_nan : NaNHandlingOptions
        Options for NaN handling with 'method' (int) and 'k' (int) keys.
        See NaNHandlingOptions for details.
    Rcon : np.ndarray, optional
        Constraint matrix for tent kernel aggregation
    q : np.ndarray, optional
        Constraint vector for tent kernel aggregation
    nQ : int, optional
        Number of slower-frequency series
    i_idio : np.ndarray
        Indicator array (1 for clock frequency, 0 for slower)
    clock : str
        Clock frequency ('m', 'q', 'sa', 'a')
    tent_weights_dict : dict, optional
        Dictionary mapping frequency pairs to tent weights
    frequencies : np.ndarray, optional
        Array of frequencies for each series
        
    Returns
    -------
    A : np.ndarray
        Initial transition matrix (m x m)
    C : np.ndarray
        Initial loading matrix (N x m)
    Q : np.ndarray
        Initial innovation covariance (m x m)
    R : np.ndarray
        Initial observation covariance (N x N)
    Z_0 : np.ndarray
        Initial state vector (m,)
    V_0 : np.ndarray
        Initial covariance matrix (m x m)
    """
    # Determine pC (tent length)
    if Rcon is None or q is None:
        pC = 1
    else:
        pC = Rcon.shape[1]
    ppC = int(max(p, pC))
    n_blocks = blocks.shape[1]

    # Balance NaNs using standard interpolation (spline-based)
    # Remaining missing values will be handled by Kalman Filter during estimation
    # This follows standard practice in DFM (Mariano & Murasawa 2003)
    xBal, _ = rem_nans_spline(x, method=opt_nan['method'], k=opt_nan['k'])
    T, N = xBal.shape

    # Determine pC from tent weights if provided
    pC = 1
    if tent_weights_dict:
        for tent_weights in tent_weights_dict.values():
            if tent_weights is not None and len(tent_weights) > pC:
                pC = len(tent_weights)
    elif Rcon is not None:
        pC = Rcon.shape[1]

    # Use generic helper to infer nQ from frequencies
    if nQ is None:
        nQ = infer_nQ(frequencies, clock)

    # Track missing data locations
    missing_data_mask = np.isnan(xBal)
    data_with_nan = xBal.copy()
    data_with_nan[missing_data_mask] = np.nan  # Explicitly mark as NaN
    data_residuals = xBal
    residuals_with_nan = data_with_nan.copy()

    C = None
    A = None
    Q = None
    V_0 = None

    if pC > 1:
        missing_data_mask[:pC - 1, :] = True

    for i in range(n_blocks):
        r_i = int(r[i])  # Store as int once to avoid repeated conversions
        r_i_int = r_i  # Alias for clarity when used in array indexing
        F_lag = None
        ar_coeffs = None
        F = None  # Initialize F at block level to avoid UnboundLocalError

        C_block = np.zeros((N, r_i_int * ppC))  # Loading matrix for current block
        # Use generic helper for block index extraction
        idx_i = get_block_indices(blocks, i)

        freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
        idx_freq = freq_groups.get(clock, np.array([], dtype=int))
        n_freq = len(idx_freq)

        if n_freq > 0:
            try:
                res = data_residuals[:, idx_freq].copy()
                # For Block_Global (first block), ensure we have sufficient variation
                if i == 0 and n_freq > 1:
                    # For global block, allow missing data but require sufficient pairwise observations
                    n_obs_per_time = np.sum(np.isfinite(res), axis=1)
                    min_series_required = max(2, int(n_freq * MIN_DATA_COVERAGE_RATIO))
                    valid_times = n_obs_per_time >= min_series_required
                    if np.sum(valid_times) < max(10, n_freq + 1):
                        finite_rows = np.any(np.isfinite(res), axis=1)
                    else:
                        finite_rows = valid_times
                else:
                    finite_rows = np.all(np.isfinite(res), axis=1)
                n_finite = int(np.sum(finite_rows))
                if n_finite < max(2, n_freq + 1):
                    _logger.warning(
                        f"init_conditions: Block {i+1} has insufficient data; using identity covariance."
                    )
                    raise ValueError("insufficient data")
                res_clean = res[finite_rows, :]
                # For Block_Global, fill remaining NaNs with column median
                if i == 0 and n_freq > 1:
                    for col_idx in range(res_clean.shape[1]):
                        col_data = res_clean[:, col_idx]
                        nan_mask = np.isnan(col_data)
                        if np.any(nan_mask) and np.any(~nan_mask):
                            col_median = np.nanmedian(col_data)
                            if np.isfinite(col_median):
                                res_clean[nan_mask, col_idx] = col_median
                            else:
                                res_clean[nan_mask, col_idx] = 0.0
                # Use pairwise complete for Block_Global robustness
                use_pairwise = (i == 0 and n_freq > 1)
                cov_res = _compute_covariance_safe(
                    res_clean,
                    rowvar=True,  # Each row is a time period, each column is a series
                    pairwise_complete=use_pairwise,
                    min_eigenval=MIN_EIGENVALUE_THRESHOLD,
                    fallback_to_identity=True
                )
                # For Block_Global, ensure minimum variance in first principal component
                d, v = _compute_principal_components(cov_res, r_i_int, block_idx=i)
                # Ensure first eigenvalue is not too small for Block_Global
                if i == 0 and len(d) > 0:
                    # For Block_Global, ensure first eigenvalue is meaningful
                    # Use absolute minimum to ensure factor can evolve
                    d_min_absolute = MIN_EIGENVALUE_ABSOLUTE
                    d_min_relative = np.max(d) * MIN_EIGENVALUE_RELATIVE  # At least MIN_EIGENVALUE_RELATIVE of max eigenvalue
                    d_min = max(d_min_absolute, d_min_relative)
                    d = np.maximum(d, d_min)
                    # Log if we're using the absolute minimum
                    if d[0] == d_min_absolute and d_min_absolute > d_min_relative:
                        _logger.warning(
                            f"init_conditions: Block_Global first eigenvalue too small "
                            f"(max={np.max(d):.6e}), using absolute minimum {d_min_absolute}"
                        )
                # Align with MATLAB: use eigenvectors directly as loadings
                # MATLAB: C_i(idx_iM,1:r_i) = v; (line 683)
                # Flip sign for cleaner output (MATLAB line 677-679)
                if np.sum(v) < 0:
                    v = -v
                # Use eigenvectors directly as loadings (no arbitrary scaling)
                # This preserves natural scale from eigendecomposition
                C_block[idx_freq, :r_i_int] = v
                # Compute factors by direct projection (MATLAB line 684: f = res(:,idx_iM)*v)
                f = data_residuals[:, idx_freq] @ v
                # MATLAB does not scale factors - use them directly
                # EM algorithm will adjust factors and loadings appropriately
                F = None
                max_lag = max(p + 1, pC)
                for kk in range(max_lag):
                    lag_data = f[pC - kk:T - kk, :]
                    # Use generic helper for append or initialize
                    F = append_or_initialize(F, lag_data, axis=1)
                if F is not None:
                    F_lag = F[:, :r_i_int * pC]
                else:
                    F_lag = None
                for freq, idx_iFreq in freq_groups.items():
                    if freq == clock:
                        continue
                    # Use generic helper for tent weight retrieval
                    tent_weights = get_tent_weights(freq, clock, tent_weights_dict, _logger)
                    R_mat_freq, q_freq = generate_R_mat(tent_weights)
                    pC_freq = len(tent_weights)
                    if F_lag.shape[1] < r_i_int * pC_freq:
                        factor_projection_freq = np.hstack([
                            F_lag,
                            np.zeros((F_lag.shape[0], r_i_int * pC_freq - F_lag.shape[1]))
                        ])
                    else:
                        factor_projection_freq = F_lag[:, :r_i_int * pC_freq]
                    Rcon_i = np.kron(R_mat_freq, np.eye(r_i_int))
                    q_i = np.kron(q_freq, np.zeros(r_i_int))
                    for j in idx_iFreq:
                        series_data = residuals_with_nan[pC_freq:, j]
                        if len(series_data) < factor_projection_freq.shape[0] and len(series_data) > 0:
                            series_data_padded = np.full(factor_projection_freq.shape[0], np.nan)
                            series_data_padded[:len(series_data)] = series_data
                            series_data = series_data_padded
                        if np.sum(~np.isnan(series_data)) < factor_projection_freq.shape[1] + 2:
                            series_data = data_residuals[pC_freq:, j]
                        finite_mask = ~np.isnan(series_data)
                        factor_projection_clean = factor_projection_freq[finite_mask, :]
                        series_data_clean = series_data[finite_mask]
                        # Use generic helper for data validation
                        if has_valid_data(series_data_clean) and get_matrix_shape(factor_projection_clean, dim=0):
                            try:
                                gram = factor_projection_clean.T @ factor_projection_clean
                                gram_inv = inv(gram)
                                loadings = gram_inv @ factor_projection_clean.T @ series_data_clean
                                if Rcon_i is not None and q_i is not None and has_valid_data(Rcon_i):
                                    constraint_term = gram_inv @ Rcon_i.T @ inv(Rcon_i @ gram_inv @ Rcon_i.T) @ (Rcon_i @ loadings - q_i)
                                    loadings = loadings - constraint_term
                                C_block[j, :pC_freq * r_i_int] = loadings[:pC_freq * r_i_int]
                            except _NUMERICAL_EXCEPTIONS:
                                C_block[j, :pC_freq * r_i_int] = 0.0
            except _NUMERICAL_EXCEPTIONS:
                _logger.warning(
                    f"init_conditions: Block {i+1} initialization failed due to numerical error; "
                    f"using fallback (identity loadings)."
                )
                if n_freq > 0:
                    C_block[idx_freq, :r_i_int] = np.eye(n_freq, r_i_int)[:n_freq, :r_i_int]
        if F_lag is not None:
            expected_width = pC * r_i_int
            actual_width = F_lag.shape[1]
            if actual_width != expected_width:
                _logger.warning(
                    f"init_conditions: Factor projection width mismatch for block {i+1} "
                    f"(got {actual_width}, expected {expected_width}); resetting."
                )
                F_lag = None
        if F_lag is not None:
            # Pad top with zeros, then pad to target shape
            top_padding = np.zeros((pC - 1, pC * r_i_int))
            factor_projection_padded = np.vstack([top_padding, F_lag])
            # Use generic helper for padding to target shape
            factor_projection_padded = pad_matrix_to_shape(
                factor_projection_padded, (T, pC * r_i_int), pad_value=0.0, pad_axis=0
            )
        else:
            factor_projection_padded = np.zeros((T, pC * r_i_int))
        C_block_residual = C_block[:, :pC * r_i_int]  # Residual loadings (without lags)
        if factor_projection_padded.shape[0] != data_residuals.shape[0]:
            if factor_projection_padded.shape[0] > data_residuals.shape[0]:
                factor_projection_padded = factor_projection_padded[:data_residuals.shape[0], :]
            else:
                # Use generic helper for padding to target shape
                factor_projection_padded = pad_matrix_to_shape(
                    factor_projection_padded,
                    (data_residuals.shape[0], factor_projection_padded.shape[1]),
                    pad_value=0.0,
                    pad_axis=0
                )
        data_residuals[:, idx_i] = data_residuals[:, idx_i] - factor_projection_padded @ C_block_residual[idx_i, :].T
        residuals_with_nan = data_residuals.copy()
        residuals_with_nan[missing_data_mask] = np.nan
        # Use generic helper for append or initialize
        C = append_or_initialize(C, C_block, axis=1)
        if n_freq > 0 and F is not None:
            # Align with MATLAB: A_temp = inv(Z'*Z)*Z'*z; (line 737)
            z = F[:, :r_i_int]
            Z_lag = F[:, r_i_int:r_i_int * (p + 1)] if F.shape[1] > r_i_int else np.zeros((F.shape[0], r_i_int * p))
            A_block = np.zeros((r_i_int * ppC, r_i_int * ppC))  # Transition matrix for current block
            # Use generic helper for shape checking
            if (get_matrix_shape(Z_lag, dim=0) and get_matrix_shape(Z_lag, dim=1)):
                # Use generic OLS helper for AR coefficient estimation
                ar_coeffs, _ = estimate_ar_coefficients_ols(z, Z_lag, use_pinv=False)
                A_block[:r_i_int, :r_i_int * p] = ar_coeffs
            if r_i_int * (ppC - 1) > 0:
                A_block[r_i_int:, :r_i_int * (ppC - 1)] = np.eye(r_i_int * (ppC - 1))
            Q_block = np.zeros((ppC * r_i_int, ppC * r_i_int))  # Innovation covariance for current block
            if len(z) > 0:
                # Compute innovation residuals
                innovation_residuals = z - Z_lag @ ar_coeffs.T if ar_coeffs is not None else z
                # Clean residuals for numerical stability
                innovation_residuals = _clean_matrix(innovation_residuals, 'general', default_nan=0.0, default_inf=0.0)
                # Use generic helper for innovation covariance computation
                Q_block_computed = compute_innovation_covariance(innovation_residuals, default_variance=DEFAULT_INNOVATION_VARIANCE)
                # Ensure Q_block_computed has correct shape
                if Q_block_computed.shape[0] != r_i_int:
                    Q_block_computed = np.eye(r_i_int) * (Q_block_computed[0, 0] if has_valid_data(Q_block_computed) else DEFAULT_INNOVATION_VARIANCE)
                Q_block[:r_i_int, :r_i_int] = Q_block_computed
            A_block_clean = _clean_matrix(A_block, 'loading')
            Q_block_clean = _clean_matrix(Q_block, 'covariance', default_nan=0.0)
            try:
                kron_transition = np.kron(A_block_clean, A_block_clean)
                identity_kron = np.eye((r_i_int * ppC) ** 2) - kron_transition
                innovation_cov_flat = Q_block_clean.flatten()
                init_cov_block = np.reshape(inv(identity_kron) @ innovation_cov_flat, (r_i_int * ppC, r_i_int * ppC))
                if np.any(~np.isfinite(init_cov_block)):
                    raise ValueError("invalid init_cov_block")
            except _NUMERICAL_EXCEPTIONS:
                _logger.warning(
                    f"init_conditions: Initial covariance computation failed for block {i+1} "
                    f"(kron/inv failed); using diagonal fallback ({DEFAULT_IDIO_COV} * I)."
                )
                init_cov_block = np.eye(r_i_int * ppC) * DEFAULT_IDIO_COV
            # Use generic helper for block diagonal matrix updates
            A, Q, V_0 = update_block_diag(
                A, Q, V_0, A_block, Q_block, init_cov_block
            )
        else:
            # Fallback: use identity matrices if block processing failed
            A_block = np.eye(r_i_int * ppC) * FALLBACK_AR
            Q_block = np.eye(r_i_int * ppC) * DEFAULT_INNOVATION_VARIANCE
            init_cov_block = np.eye(r_i_int * ppC) * DEFAULT_IDIO_COV
            # Use generic helper for block diagonal matrix updates
            A, Q, V_0 = update_block_diag(
                A, Q, V_0, A_block, Q_block, init_cov_block
            )

    eyeN = np.eye(N)[:, i_idio.astype(bool)]
    # Use generic helper for append or initialize
    C = append_or_initialize(C, eyeN, axis=1)

    if nQ > 0 and frequencies is not None:
        clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
        slower_indices = [j for j in range(N) if j < len(frequencies) and FREQUENCY_HIERARCHY.get(frequencies[j], 3) > clock_h]
        # Suppress warning when ddof >= number of non-NaN values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Rdiag = np.nanvar(residuals_with_nan, axis=0)
        # Use generic helper for variance cleaning
        Rdiag = clean_variance_array(
            Rdiag,
            default_value=DEFAULT_OBSERVATION_VARIANCE,
            min_value=DEFAULT_OBSERVATION_VARIANCE
        )
        slower_idio_blocks = []
        slower_series_count: Dict[str, int] = {}
        slower_series_indices: Dict[str, list] = {}
        slower_idio_dims = []  # Track dimensions for BQ
        for j in range(N):
            if j < len(frequencies):
                freq = frequencies[j]
                if FREQUENCY_HIERARCHY.get(freq, 3) > clock_h:
                    slower_series_count[freq] = slower_series_count.get(freq, 0) + 1
                    slower_series_indices.setdefault(freq, []).append(j)
        for freq, idx_list in slower_series_indices.items():
            # Use generic helper for tent weight retrieval
            tent_weights = get_tent_weights(freq, clock, tent_weights_dict, _logger)
            n_periods = len(tent_weights)
            idio_block = np.zeros((N, n_periods * len(idx_list)))
            for idx, j in enumerate(idx_list):
                col_start = idx * n_periods
                idio_block[j, col_start:col_start + n_periods] = tent_weights
            slower_idio_blocks.append(idio_block)
            slower_idio_dims.append(n_periods * len(idx_list))  # Track dimension for this frequency
        if slower_idio_blocks:
            slower_idio_full = np.hstack(slower_idio_blocks)
            C = np.hstack([C, slower_idio_full])
        # Use generic helper for variance cleaning (includes negative check)
        Rdiag = clean_variance_array(
            Rdiag,
            default_value=DEFAULT_OBSERVATION_VARIANCE,
            min_value=DEFAULT_OBSERVATION_VARIANCE,
            replace_negative=True
        )
        R = np.diag(Rdiag)
    else:
        # Suppress warning when ddof >= number of non-NaN values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            var_values = np.nanvar(residuals_with_nan, axis=0)
        # Use generic helper for variance cleaning
        var_values = clean_variance_array(
            var_values,
            default_value=DEFAULT_OBSERVATION_VARIANCE,
            min_value=DEFAULT_OBSERVATION_VARIANCE
        )
        R = np.diag(var_values)
        slower_idio_dims = []

    ii_idio = np.where(i_idio)[0]
    n_idio = len(ii_idio)
    BM = np.zeros((n_idio, n_idio))
    SM = np.zeros((n_idio, n_idio))
    for idx, i in enumerate(ii_idio):
        R[i, i] = DEFAULT_OBSERVATION_VARIANCE
        series_residuals_full = data_residuals[:, i]
        series_residuals_with_nan = residuals_with_nan[:, i]
        # Find leading and trailing NaN positions
        leading_nan_count = int(np.argmax(~np.isnan(series_residuals_with_nan))) if np.any(~np.isnan(series_residuals_with_nan)) else 0
        trailing_nan_count = int(np.argmax(~np.isnan(series_residuals_with_nan[::-1]))) if np.any(~np.isnan(series_residuals_with_nan)) else 0
        # Truncate to remove leading/trailing NaNs
        if trailing_nan_count > 0:
            series_residuals_truncated = series_residuals_full[leading_nan_count:T - trailing_nan_count]
        else:
            series_residuals_truncated = series_residuals_full[leading_nan_count:]
        if len(series_residuals_truncated) > 1:
            series_data = series_residuals_truncated.copy()
            X_ar = series_data[:-1].reshape(-1, 1)
            y_ar = series_data[1:].reshape(-1, 1)
            XTX = X_ar.T @ X_ar
            if XTX.size == 1 and XTX[0, 0] != 0 and np.isfinite(XTX[0, 0]):
                try:
                    # Use generic OLS helper for AR coefficient estimation
                    # Align with MATLAB: BM(i,i) = inv(res_i(1:end-1)'*res_i(1:end-1))*res_i(1:end-1)'*res_i(2:end,:);
                    ar_coeffs, _ = estimate_ar_coefficients_ols(
                        series_residuals_truncated[1:].reshape(-1, 1),
                        series_residuals_truncated[:-1].reshape(-1, 1),
                        use_pinv=False
                    )
                    BM[idx, idx] = ar_coeffs[0, 0] if has_valid_data(ar_coeffs) else 0.0
                    # Use generic helper for innovation covariance
                    # Align with MATLAB: SM(i,i) = cov(res_i(2:end)-res_i(1:end-1)*BM(i,i)); (line 786)
                    innovation_residuals = series_residuals_truncated[1:] - series_residuals_truncated[:-1] * BM[idx, idx]
                    Q_idio = compute_innovation_covariance(innovation_residuals, default_variance=DEFAULT_INNOVATION_VARIANCE)
                    SM[idx, idx] = Q_idio[0, 0] if has_valid_data(Q_idio) else DEFAULT_INNOVATION_VARIANCE
                except _NUMERICAL_EXCEPTIONS:
                    # MATLAB doesn't have fallback - set to 0 if OLS fails
                    BM[idx, idx] = 0.0
                    SM[idx, idx] = DEFAULT_INNOVATION_VARIANCE
            else:
                BM[idx, idx] = 0.0
                SM[idx, idx] = DEFAULT_INNOVATION_VARIANCE
        else:
            BM[idx, idx] = 0.0
            SM[idx, idx] = DEFAULT_INNOVATION_VARIANCE
    # Monthly idio init covariance
    eye_diag = np.diag(np.eye(BM.shape[0]))
    BM_diag_sq = np.diag(BM) ** 2
    denom_val = np.where(np.abs(eye_diag - BM_diag_sq) < MIN_VARIANCE_THRESHOLD, 1.0, eye_diag - BM_diag_sq)
    denom_viM = np.diag(1.0 / denom_val)
    initViM = denom_viM @ SM
    # Slower idio init: create BQ and SQ with proper dimensions
    n_slower_idio = sum(slower_idio_dims) if slower_idio_dims else 0
    if n_slower_idio > 0:
        # Create transition and innovation covariance for slower idio components
        # Each slower idio component follows an AR(1) process with tent weights
        BQ = np.eye(n_slower_idio) * DEFAULT_IDIO_AR  # AR coefficient for slower idio
        SQ = np.eye(n_slower_idio) * DEFAULT_IDIO_VAR  # Innovation variance
        initViQ = np.eye(n_slower_idio) * DEFAULT_IDIO_COV  # Initial covariance
    else:
        # Use generic helper for empty matrix creation
        BQ = create_empty_matrix((0, 0))
        SQ = create_empty_matrix((0, 0))
        initViQ = create_empty_matrix((0, 0))
    # Use generic helper for final block diagonal assembly
    # Note: update_block_diag handles None case, but here we have existing matrices
    # Use has_valid_data for consistent size checking
    A = (block_diag(A, BM, BQ) if has_valid_data(A) else 
         block_diag(BM, BQ) if has_valid_data(BM) else BQ)
    Q = (block_diag(Q, SM, SQ) if has_valid_data(Q) else 
         block_diag(SM, SQ) if has_valid_data(SM) else SQ)
    # Align with MATLAB: Z_0 = zeros(size(A,1),1) (line 809)
    # MATLAB updates Z_0 in EM step to Zsmooth(:,1), which we also do
    Z_0 = np.zeros(A.shape[0])
    # Ensure all covariance matrices are square
    V_0 = _ensure_square_matrix(V_0, method='diag')
    initViM = _ensure_square_matrix(initViM, method='diag')
    initViQ = _ensure_square_matrix(initViQ, method='diag')
    V_0 = block_diag(V_0, initViM, initViQ)
    # Final clean/validate using _check_finite for consistency
    outputs = {'A': A, 'C': C, 'Q': Q, 'R': R, 'Z_0': Z_0, 'V_0': V_0}
    for param_name, param_value in outputs.items():
        if has_valid_data(param_value) and not _check_finite(param_value, param_name):
            # Clean based on parameter type
            if param_name in ['A', 'Q', 'V_0']:
                outputs[param_name] = _clean_matrix(param_value, 'covariance', default_nan=0.0)
            elif param_name == 'R':
                outputs[param_name] = _clean_matrix(param_value, 'diagonal', default_nan=DEFAULT_OBSERVATION_VARIANCE)
            elif param_name == 'C':
                outputs[param_name] = _clean_matrix(param_value, 'loading')
            elif param_name == 'Z_0':
                outputs[param_name] = np.zeros_like(param_value)
    # Final safeguard: ensure Q diagonal has minimum value after all cleaning
    if has_valid_data(outputs['Q']):
        outputs['Q'] = _ensure_innovation_variance_minimum(outputs['Q'], min_variance=MIN_INNOVATION_VARIANCE)
    return outputs['A'], outputs['C'], outputs['Q'], outputs['R'], outputs['Z_0'], outputs['V_0']

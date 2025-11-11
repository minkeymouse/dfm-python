"""Expectation-Maximization core routines.

This module contains:
- init_conditions: compute initial A, C, Q, R, Z_0, V_0 via PCA and OLS
- em_step: perform one EM iteration (E-step via Kalman, M-step updates)
- em_converged: check EM algorithm convergence
"""

from typing import Optional, Tuple, Dict, Any, TypedDict
from dataclasses import dataclass
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

from ...kalman import run_kf
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
from ...data.utils import rem_nans_spline
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

# Convergence and stability constants
MIN_LOG_LIKELIHOOD_DELTA = -1e-3  # Threshold for detecting likelihood decrease
DAMPING = 0.95  # Damping factor when numerical errors occur
MAX_LOADING_REPLACE = 0.99  # Replacement for Inf in loadings


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


@dataclass
class EMStepParams:
    """Parameters for a single EM step execution.
    
    This dataclass groups all parameters required for performing one EM iteration,
    reducing function parameter count and improving readability.
    
    All parameters are required (no optional fields) since the EM step
    needs all of them to execute.
    """
    # Data
    y: np.ndarray
    
    # Model parameters
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    r: np.ndarray
    p: int
    
    # Structure parameters
    R_mat: Optional[np.ndarray]
    q: Optional[np.ndarray]
    nQ: Optional[int]
    i_idio: np.ndarray
    blocks: np.ndarray
    tent_weights_dict: Optional[Dict[str, np.ndarray]]
    clock: str
    frequencies: Optional[np.ndarray]
    
    # Config
    config: Optional[DFMConfig]


def em_step(
    params: EMStepParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform one EM iteration (E-step + M-step) and return updated parameters.
    
    This function performs a single iteration of the Expectation-Maximization (EM) algorithm
    for Dynamic Factor Model estimation. The EM algorithm alternates between:
    
    1. **E-step (Expectation)**: Run Kalman filter and smoother to compute expected
       sufficient statistics given current parameters:
       - E[Z_t | Y] (smoothed factor estimates)
       - E[Z_t Z_t' | Y] (smoothed factor covariances)
       - E[Z_t Z_{t-1}' | Y] (smoothed factor cross-covariances)
    
    2. **M-step (Maximization)**: Update parameters to maximize expected log-likelihood:
       - **C**: Loading matrix via regression of data on factors
       - **R**: Observation covariance via residual variance
       - **A**: Transition matrix via AR regression on factors
       - **Q**: Innovation covariance via innovation variance
       - **V_0**: Initial covariance via stationary covariance
    
    The algorithm follows the standard DFM EM approach (Doz et al. 2011) with robust
    numerical stability checks and missing data handling.
    
    Parameters
    ----------
    params : EMStepParams
        Dataclass containing all parameters for the EM step:
        - y: Observation matrix (n x T), where n is number of series and T is time periods.
          Missing values should be NaN. These are handled automatically by the Kalman filter.
        - A: Current transition matrix estimate (m x m), where m is state dimension.
        - C: Current loading matrix estimate (n x m).
        - Q: Current innovation covariance estimate (m x m).
        - R: Current observation covariance estimate (n x n), typically diagonal.
        - Z_0: Current initial state estimate (m,).
        - V_0: Current initial covariance estimate (m x m).
        - r: Number of factors per block, shape (n_blocks,).
        - p: AR lag order for factor dynamics. Typically p=1.
        - R_mat: Constraint matrix for tent kernel aggregation in mixed-frequency models.
        - q: Constraint vector for tent kernel aggregation.
        - nQ: Number of slower-frequency series. If None, inferred from frequencies.
        - i_idio: Indicator array (n,) where 1 indicates clock frequency, 0 indicates slower.
        - blocks: Block structure matrix (n x n_blocks).
        - tent_weights_dict: Dictionary mapping frequency strings to tent weight arrays.
        - clock: Clock frequency for latent factors ('m', 'q', 'sa', 'a').
        - frequencies: Array of frequency strings (n,) for each series.
        - config: Configuration object for numerical stability parameters.
        
    Returns
    -------
    C_new : np.ndarray
        Updated loading matrix (n x m). Each row corresponds to a series,
        each column to a factor.
    R_new : np.ndarray
        Updated observation covariance (n x n), typically diagonal.
        Diagonal elements represent idiosyncratic variances.
    A_new : np.ndarray
        Updated transition matrix (m x m). Describes factor dynamics.
    Q_new : np.ndarray
        Updated innovation covariance (m x m). Diagonal elements represent
        innovation variances for each factor.
    Z_0_new : np.ndarray
        Updated initial state (m,). Typically unchanged from input.
    V_0_new : np.ndarray
        Updated initial covariance (m x m). Computed from stationary covariance.
    loglik : float
        Log-likelihood value for this iteration. Used for convergence checking.
        
    Raises
    ------
    ValueError
        If inputs are invalid or numerical issues occur during computation.
        
    Notes
    -----
    - All input parameters are validated and cleaned before use
    - Missing data is handled automatically by the Kalman filter
    - Q diagonal elements are enforced to have minimum value (MIN_INNOVATION_VARIANCE) 
      for factor evolution. This is critical: Q[i,i] = 0 means factor i cannot evolve.
      Enforcement occurs after block updates and as final safeguard.
    - All covariance matrices are regularized to ensure positive semi-definiteness
    - AR coefficients are clipped to stability bounds if configured
    
    Examples
    --------
    >>> import numpy as np
    >>> from dfm_python.core.em import em_step, EMStepParams
    >>> # Observation matrix with missing data
    >>> y = np.random.randn(10, 100)
    >>> y[0, 10:20] = np.nan  # Some missing values
    >>> # Current parameter estimates
    >>> A = np.eye(3) * 0.9
    >>> C = np.random.randn(10, 3) * 0.5
    >>> Q = np.eye(3) * 0.1
    >>> R = np.eye(10) * 0.5
    >>> Z_0 = np.zeros(3)
    >>> V_0 = np.eye(3) * 0.1
    >>> r = np.array([1, 2])
    >>> blocks = np.column_stack([np.ones(10), np.ones(10)])
    >>> # Create parameters dataclass
    >>> params = EMStepParams(
    ...     y=y, A=A, C=C, Q=Q, R=R, Z_0=Z_0, V_0=V_0, r=r, p=1,
    ...     R_mat=None, q=None, nQ=0, i_idio=np.ones(10),
    ...     blocks=blocks, tent_weights_dict=None, clock='m', frequencies=None, config=None
    ... )
    >>> # Perform one EM iteration
    >>> C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(params)
    >>> assert C_new.shape == C.shape
    >>> assert np.isfinite(loglik)
    """
    # Extract parameters from dataclass
    y = params.y
    A = params.A
    C = params.C
    Q = params.Q
    R = params.R
    Z_0 = params.Z_0
    V_0 = params.V_0
    r = params.r
    p = params.p
    R_mat = params.R_mat
    q = params.q
    nQ = params.nQ
    i_idio = params.i_idio
    blocks = params.blocks
    tent_weights_dict = params.tent_weights_dict
    clock = params.clock
    frequencies = params.frequencies
    config = params.config
    
    # Validate and clean input parameters using generic helper
    A, Q, R, C, Z_0, V_0 = validate_params(
        A, Q, R, C, Z_0, V_0,
        fallback_transition_coeff=FALLBACK_AR,
        min_innovation_variance=MIN_INNOVATION_VARIANCE,
        default_observation_variance=DEFAULT_OBSERVATION_VARIANCE,
        default_idio_init_covariance=DEFAULT_IDIO_COV
    )

    n, T = y.shape
    # Use generic helper to infer nQ from frequencies
    if nQ is None:
        nQ = infer_nQ(frequencies, clock)

    pC = 1
    if tent_weights_dict:
        for tent_weights in tent_weights_dict.values():
            if tent_weights is not None and len(tent_weights) > pC:
                pC = len(tent_weights)
    elif R_mat is not None:
        pC = R_mat.shape[1]
    ppC = int(max(p, pC))
    num_blocks = blocks.shape[1]

    if config is None:
        from types import SimpleNamespace
        config = SimpleNamespace(
            clip_ar_coefficients=True,
            ar_clip_min=-0.99,
            ar_clip_max=0.99,
            warn_on_ar_clip=True,
            clip_data_values=True,
            data_clip_threshold=100.0,
            warn_on_data_clip=True,
            use_regularization=True,
            regularization_scale=1e-5,
            min_eigenvalue=1e-8,
            max_eigenvalue=1e6,
            warn_on_regularization=True,
            use_damped_updates=True,
            damping_factor=0.8,
            warn_on_damped_update=True,
        )

    zsmooth, vsmooth, vvsmooth, loglik = run_kf(y, A, C, Q, R, Z_0, V_0)
    Zsmooth = zsmooth.T

    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_0.copy()

    for i in range(num_blocks):
        r_i_int = int(r[i])  # Store as int once to avoid repeated conversions
        factor_lag_size = r_i_int * p
        # Use generic helper for block slice indices
        t_start, t_end = compute_block_slice_indices(r, i, ppC)
        factor_start_idx = t_start  # Alias for clarity
        b_subset = slice(factor_start_idx, factor_start_idx + factor_lag_size)

        # Use generic helper for computing expected sufficient statistics
        # Note: Zsmooth is (T+1) x m, vsmooth is m x m x (T+1), vvsmooth is m x m x T
        # We need to transpose Zsmooth to get m x (T+1) format
        Zsmooth_T = Zsmooth.T  # Convert to m x (T+1)
        EZZ, EZZ_lag, EZZ_cross = compute_sufficient_stats(Zsmooth_T, vsmooth, vvsmooth, b_subset, T)

        EZZ_lag = _clean_matrix(EZZ_lag, 'covariance', default_nan=0.0)
        EZZ_cross = _clean_matrix(EZZ_cross, 'general', default_nan=0.0)

        # Use generic helpers for block matrix extraction
        A_block = extract_block_matrix(A, t_start, t_end)
        Q_block = extract_block_matrix(Q, t_start, t_end)
        try:
            EZZ_lag_sub = EZZ_lag[:factor_lag_size, :factor_lag_size]
            min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
            warn_reg = safe_get_attr(config, "warn_on_regularization", True)
            EZZ_lag_sub, _ = _ensure_positive_definite(EZZ_lag_sub, min_eigenval, warn_reg)
            try:
                eigenvals = np.linalg.eigvals(EZZ_lag_sub)
                cond_num = (np.max(eigenvals) / max(np.min(eigenvals), 1e-12)) if np.max(eigenvals) > 0 else 1.0
                EZZ_lag_inv = pinv(EZZ_lag_sub, cond=1e-8) if cond_num > 1e12 else inv(EZZ_lag_sub)
            except _NUMERICAL_EXCEPTIONS:
                EZZ_lag_inv = pinv(EZZ_lag_sub)
            # Align with MATLAB: A_i(1:r_i,1:rp) = EZZ_FB(1:r_i,1:rp) * inv(EZZ_BB(1:rp,1:rp)); (line 357)
            transition_update = EZZ_cross[:r_i_int, :factor_lag_size] @ EZZ_lag_inv
            # Apply AR clipping only if configured (for numerical stability, not arbitrary constraint)
            transition_update, _ = _apply_ar_clipping(transition_update, config)
            A_block[:r_i_int, :factor_lag_size] = transition_update
            # Compute innovation covariance: Q = (E[Z_t Z_t'] - A @ E[Z_t Z_{t-1}']) / T
            Q_block[:r_i_int, :r_i_int] = (
                EZZ[:r_i_int, :r_i_int] -
                A_block[:r_i_int, :factor_lag_size] @ EZZ_cross[:r_i_int, :factor_lag_size].T
            ) / T
            # Use generic helper for stability operations
            Q_block[:r_i_int, :r_i_int] = stabilize_cov(
                Q_block[:r_i_int, :r_i_int],
                config,
                min_variance=MIN_INNOVATION_VARIANCE
            )
        except _NUMERICAL_EXCEPTIONS:
            if np.allclose(A_block[:r_i_int, :factor_lag_size], 0):
                A_block[:r_i_int, :factor_lag_size] = np.random.randn(r_i_int, factor_lag_size) * FALLBACK_SCALE
            else:
                A_block[:r_i_int, :factor_lag_size] *= DAMPING
        if np.any(~np.isfinite(A_block)):
            A_block = _clean_matrix(A_block, 'loading', default_nan=0.0, default_inf=MAX_LOADING_REPLACE)
            A_block, _ = _apply_ar_clipping(A_block, config)
        # Use generic helpers for block matrix updates
        update_block_in_matrix(A_new, A_block, t_start, t_end)
        update_block_in_matrix(Q_new, Q_block, t_start, t_end)
        V_0_block = _clean_matrix(vsmooth[t_start:t_end, t_start:t_end, 0], 'covariance', default_nan=0.0)
        V_0_block, _ = _ensure_positive_definite(V_0_block, min_eigenval, warn_reg)
        update_block_in_matrix(V_0_new, V_0_block, t_start, t_end)

    idio_start_idx = int(np.sum(r) * ppC)
    n_idio = int(np.sum(i_idio))
    i_subset = slice(idio_start_idx, idio_start_idx + n_idio)
    i_subset_slice = slice(i_subset.start, i_subset.stop)
    Z_idio = Zsmooth[i_subset_slice, 1:]
    n_idio_actual = Z_idio.shape[0]
    expected_idio_current_sq = np.sum(Z_idio**2, axis=1)
    vsmooth_idio_block = vsmooth[i_subset_slice, :, :][:, i_subset_slice, 1:]
    vsmooth_idio_sum = np.sum(vsmooth_idio_block, axis=2)
    vsmooth_idio_diag = np.diag(vsmooth_idio_sum[:n_idio_actual, :n_idio_actual]) if vsmooth_idio_sum.ndim == 2 else np.zeros(n_idio_actual)
    expected_idio_current_sq = expected_idio_current_sq + vsmooth_idio_diag
    Z_idio_lag = Zsmooth[i_subset_slice, :-1]
    expected_idio_lag_sq = np.sum(Z_idio_lag**2, axis=1)
    vsmooth_lag_block = vsmooth[i_subset_slice, :, :][:, i_subset_slice, :-1]
    vsmooth_lag_sum = np.sum(vsmooth_lag_block, axis=2)
    vsmooth_lag_diag = np.diag(vsmooth_lag_sum[:n_idio_actual, :n_idio_actual]) if vsmooth_lag_sum.ndim == 2 else np.zeros(n_idio_actual)
    expected_idio_lag_sq = expected_idio_lag_sq + vsmooth_lag_diag
    min_cols = min(Z_idio.shape[1], Z_idio_lag.shape[1])
    expected_idio_cross = np.sum(Z_idio[:, :min_cols] * Z_idio_lag[:, :min_cols], axis=1)
    vvsmooth_block = vvsmooth[i_subset_slice, :, :][:, i_subset_slice, :]
    vvsmooth_sum = np.sum(vvsmooth_block, axis=2)
    vvsmooth_diag = np.diag(vvsmooth_sum[:n_idio_actual, :n_idio_actual]) if vvsmooth_sum.ndim == 2 else np.zeros(n_idio_actual)
    expected_idio_cross = expected_idio_cross + vvsmooth_diag
    ar_coeffs_diag, _ = _estimate_ar_coefficient(expected_idio_cross, expected_idio_lag_sq, vsmooth_sum=vsmooth_lag_diag)
    A_block_idio = np.diag(ar_coeffs_diag)
    innovation_cov_diag = (np.maximum(expected_idio_current_sq, 0.0) - ar_coeffs_diag * expected_idio_cross) / T
    innovation_cov_diag = np.maximum(innovation_cov_diag, MIN_INNOVATION_VARIANCE)
    Q_block_idio = np.diag(innovation_cov_diag)
    i_subset_size = i_subset.stop - i_subset.start
    if n_idio_actual == i_subset_size:
        A_new[i_subset, i_subset] = A_block_idio
        Q_new[i_subset, i_subset] = Q_block_idio
    elif n_idio_actual < i_subset_size:
        A_new[i_subset.start:i_subset.start + n_idio_actual, i_subset.start:i_subset.start + n_idio_actual] = A_block_idio
        Q_new[i_subset.start:i_subset.start + n_idio_actual, i_subset.start:i_subset.start + n_idio_actual] = Q_block_idio
    else:
        A_new[i_subset, i_subset] = A_block_idio[:i_subset_size, :i_subset_size]
        Q_new[i_subset, i_subset] = Q_block_idio[:i_subset_size, :i_subset_size]
    vsmooth_sub = vsmooth[i_subset_slice, :, :][:, i_subset_slice, 0]
    vsmooth_diag = np.diag(vsmooth_sub[:n_idio_actual, :n_idio_actual]) if vsmooth_sub.ndim == 2 else np.zeros(n_idio_actual)
    for idx in range(min(n_idio_actual, i_subset_size)):
        V_0_new[i_subset.start + idx, i_subset.start + idx] = vsmooth_diag[idx] if idx < len(vsmooth_diag) else 0.0

    Z_0 = Zsmooth[0, :].copy()
    nanY = np.isnan(y)
    y_clean = y.copy()
    y_clean[nanY] = 0
    bl = np.unique(blocks, axis=0)
    n_bl = bl.shape[0]
    bl_idx_same_freq = None
    bl_idx_slower_freq = None
    R_con_list = []
    for i in range(num_blocks):
        r_i_int = int(r[i])  # Store as int once to avoid repeated conversions
        bl_col_clock_freq = np.repeat(bl[:, i:i+1], r_i_int, axis=1)
        bl_col_clock_freq = np.hstack([bl_col_clock_freq, np.zeros((n_bl, r_i_int * (ppC - 1)))])
        bl_col_slower_freq = np.repeat(bl[:, i:i+1], r_i_int * ppC, axis=1)
        if bl_idx_same_freq is None:
            bl_idx_same_freq = bl_col_clock_freq
            bl_idx_slower_freq = bl_col_slower_freq
        else:
            bl_idx_same_freq = np.hstack([bl_idx_same_freq, bl_col_clock_freq])
            bl_idx_slower_freq = np.hstack([bl_idx_slower_freq, bl_col_slower_freq])
        if R_mat is not None:
            R_con_list.append(np.kron(R_mat, np.eye(r_i_int)))
    if bl_idx_same_freq is not None:
        bl_idx_same_freq = bl_idx_same_freq.astype(bool)
        bl_idx_slower_freq = bl_idx_slower_freq.astype(bool)
    else:
        bl_idx_same_freq = np.array([]).reshape(n_bl, 0).astype(bool)
        bl_idx_slower_freq = np.array([]).reshape(n_bl, 0).astype(bool)
    R_con = block_diag(*R_con_list) if len(R_con_list) > 0 else np.array([])
    q_con = np.zeros((np.sum(r.astype(int)) * R_mat.shape[0], 1)) if (R_mat is not None and q is not None) else np.array([])

    i_idio_same = i_idio
    n_idio_same = int(np.sum(i_idio_same))
    c_i_idio = np.cumsum(i_idio.astype(int))
    C_new = C.copy()
    for i in range(n_bl):
        bl_i = bl[i, :]
        # rs = total factors across blocks in this block group (sum of r[bl_i])
        # Note: r_i (used elsewhere) = factors per individual block (from r[i])
        rs = int(np.sum(r[bl_i.astype(bool)]))
        idx_i = np.where((blocks == bl_i).all(axis=1))[0]
        freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
        idx_freq = freq_groups.get(clock, np.array([], dtype=int))
        n_freq = len(idx_freq)
        if n_freq == 0:
            continue
        bl_idx_same_freq_i = np.where(bl_idx_same_freq[i, :])[0]
        if len(bl_idx_same_freq_i) == 0:
            continue
        rs_actual = len(bl_idx_same_freq_i)
        if rs_actual != rs:
            rs = rs_actual
        denom_size = n_freq * rs
        denom = np.zeros((denom_size, denom_size))
        nom = np.zeros((n_freq, rs))
        i_idio_i = i_idio_same[idx_freq]
        i_idio_ii = np.cumsum(i_idio.astype(int))[idx_freq]
        i_idio_ii = i_idio_ii[i_idio_i.astype(bool)]
        for t in range(T):
            nan_mask = ~nanY[idx_freq, t]
            Wt = np.diag(nan_mask.astype(float))
            # Use generic helpers for time-indexed access
            if safe_time_index(t, Zsmooth.shape[0], offset=1):
                Z_block_same_freq_row = Zsmooth[t + 1, bl_idx_same_freq_i]
                ZZZ = Z_block_same_freq_row.reshape(-1, 1) @ Z_block_same_freq_row.reshape(1, -1)
            else:
                ZZZ = np.zeros((rs, rs))
            # Use generic helper for 3D matrix slice extraction
            V_block_same_freq = extract_3d_matrix_slice(
                vsmooth, bl_idx_same_freq_i, bl_idx_same_freq_i, t + 1
            )
            if V_block_same_freq.shape != (rs, rs):
                V_block_same_freq = np.zeros((rs, rs))
            expected_shape = (denom_size, denom_size)
            try:
                kron_result = np.kron(ZZZ + V_block_same_freq, Wt)
                if kron_result.shape == expected_shape:
                    denom += kron_result
            except _NUMERICAL_EXCEPTIONS:
                pass
            # Use generic helper for time-indexed access
            if safe_time_index(t, Zsmooth.shape[0], offset=1):
                y_vec = y_clean[idx_freq, t].reshape(-1, 1)
                Z_vec_row = Zsmooth[t + 1, bl_idx_same_freq_i].reshape(1, -1)
                y_term = y_vec @ Z_vec_row
            else:
                y_term = np.zeros((n_freq, rs_actual))
            # Handle idiosyncratic terms
            if len(i_idio_ii) > 0 and safe_time_index(t, Zsmooth.shape[0], offset=1):
                idio_idx = (idio_start_idx + i_idio_ii).astype(int)
                if idio_idx.max() < Zsmooth.shape[1]:
                    idio_Z_col = Zsmooth[t + 1, idio_idx].reshape(-1, 1)
                    idio_Z_outer = idio_Z_col @ Z_vec_row
                    # Use generic helper for 3D matrix slice extraction
                    idio_V = extract_3d_matrix_slice(
                        vsmooth, idio_idx, bl_idx_same_freq_i, t + 1
                    )
                    if idio_V.shape != (len(i_idio_ii), rs_actual):
                        idio_V = np.zeros((len(i_idio_ii), rs_actual))
                    idio_term = Wt[:, i_idio_i.astype(bool)] @ (idio_Z_outer + idio_V)
                else:
                    idio_term = np.zeros((n_freq, rs_actual))
            else:
                idio_term = np.zeros((n_freq, rs_actual))
            nom += y_term - idio_term
        # Use generic helper for regularized inverse computation
        vec_C, success = reg_inv(denom, nom, config)
        if success:
            C_update = vec_C.reshape(n_freq, rs)
            # Clean NaN/Inf for numerical stability
            C_update = _clean_matrix(C_update, 'loading', default_nan=0.0, default_inf=0.0)
            # Use generic helper for loading matrix update
            update_loadings(C_new, C_update, idx_freq, bl_idx_same_freq_i)
        for freq, idx_iFreq in freq_groups.items():
            if freq == clock:
                continue
            # Use generic helper for tent weight retrieval
            tent_weights = get_tent_weights(freq, clock, tent_weights_dict, _logger)
            pC_freq = len(tent_weights)
            rs_full = rs * pC_freq
            R_mat_freq, q_freq = generate_R_mat(tent_weights)
            R_con_i = np.kron(R_mat_freq, np.eye(int(rs)))
            q_con_i = np.kron(q_freq, np.zeros(int(rs)))
            if i < bl_idx_slower_freq.shape[0]:
                bl_idx_slower_freq_i = np.where(bl_idx_slower_freq[i, :])[0]
                if len(bl_idx_slower_freq_i) >= rs_full:
                    bl_idx_slower_freq_i = bl_idx_slower_freq_i[:rs_full]
                elif len(bl_idx_slower_freq_i) > 0:
                    bl_idx_slower_freq_i = np.pad(bl_idx_slower_freq_i, (0, rs_full - len(bl_idx_slower_freq_i)), mode='edge')
                else:
                    continue
            else:
                continue
            if has_valid_data(R_con_i):
                no_c = ~np.any(R_con_i, axis=1)
                R_con_i = R_con_i[~no_c, :]
                q_con_i = q_con_i[~no_c]
            for j in idx_iFreq:
                rps_actual = len(bl_idx_slower_freq_i) if len(bl_idx_slower_freq_i) > 0 else rs_full
                denom = np.zeros((rps_actual, rps_actual))
                nom = np.zeros((1, rps_actual))
                idx_j_slower = sum(1 for k in idx_iFreq if k < j and (frequencies is None or k >= len(frequencies) or frequencies[k] == freq))
                for t in range(T):
                    nan_val = ~nanY[j, t]
                    Wt = np.array([[float(nan_val)]]) if np.isscalar(nan_val) else np.diag(nan_val.astype(float))
                    if len(bl_idx_slower_freq_i) == 0:
                        continue
                    valid_bl_idx = bl_idx_slower_freq_i[bl_idx_slower_freq_i < Zsmooth.shape[1]]
                    if len(valid_bl_idx) == 0:
                        continue
                    # Use generic helper for time-indexed access
                    if safe_time_index(t, Zsmooth.shape[0], offset=1):
                        Z_row = Zsmooth[t + 1, valid_bl_idx]
                        Z_col = Z_row.reshape(-1, 1)
                        ZZZ = Z_col @ Z_row.reshape(1, -1)
                        valid_vs_idx = valid_bl_idx[valid_bl_idx < vsmooth.shape[0]]
                        if len(valid_vs_idx) > 0:
                            # Use generic helper for 3D matrix slice extraction
                            V_block = extract_3d_matrix_slice(
                                vsmooth, valid_vs_idx, valid_vs_idx, t + 1
                            )
                            if V_block.shape != ZZZ.shape:
                                min_size = min(V_block.shape[0], ZZZ.shape[0])
                                V_block = V_block[:min_size, :min_size]
                                ZZZ = ZZZ[:min_size, :min_size]
                        else:
                            V_block = np.zeros_like(ZZZ)
                    else:
                        Z_row = np.zeros(rps_actual)
                        Z_col = np.zeros((rps_actual, 1))
                        ZZZ = np.zeros((rps_actual, rps_actual))
                        V_block = np.zeros((rps_actual, rps_actual))
                    if Wt.shape == (1, 1):
                        denom += (ZZZ + V_block) * Wt[0, 0]
                    else:
                        denom += np.kron(ZZZ + V_block, Wt)
                    nom += y_clean[j, t] * Z_row.reshape(1, -1)
                # Use generic helper for regularized inverse computation
                C_i, success = reg_inv(denom, nom.T, config)
                if success:
                    # Clean NaN/Inf for numerical stability
                    C_i = _clean_matrix(C_i, 'loading', default_nan=0.0, default_inf=0.0)
                    if len(bl_idx_slower_freq_i) > 0:
                        C_update = C_i.flatten()[:len(bl_idx_slower_freq_i)]
                        # Use vectorized assignment for efficiency
                        row_idx_array = np.array([j])
                        C_new[np.ix_(row_idx_array, bl_idx_slower_freq_i)] = C_update.reshape(1, -1)

    # Align with MATLAB: No arbitrary safeguards for loadings
    # MATLAB: vec_C = inv(denom)*nom(:); C_new(idx_iM,bl_idxM(i,:)) = reshape(vec_C, n_i, rs);
    # Uses computed loadings directly without scaling or minimum enforcement

    # Use generic helper for R diagonal computation
    R_diag = compute_obs_cov(
        y, C_new, Zsmooth, vsmooth,
        default_variance=DEFAULT_OBSERVATION_VARIANCE,
        min_variance=MIN_OBSERVATION_VARIANCE,
        min_diagonal_variance_ratio=MIN_DIAGONAL_VARIANCE
    )
    R_new = np.diag(R_diag)
    # Align with MATLAB: No arbitrary safeguards for AR coefficients or Q
    # MATLAB computes Q_i(1:r_i,1:r_i) = (EZZ(1:r_i,1:r_i) - A_i(1:r_i,1:rp)* EZZ_FB(1:r_i,1:rp)') / T;
    # Apply numerical stability operations (for numerical stability, not arbitrary constraints)
    Q_new = stabilize_cov(Q_new, config, min_variance=MIN_INNOVATION_VARIANCE)
    # Clean and stabilize V_0
    V_0_new = _clean_matrix(V_0_new, 'covariance', default_nan=0.0)
    min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
    V_0_new, _ = _ensure_positive_definite(V_0_new, min_eigenval, warn_reg)
    return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik


def em_converged(
    loglik: float,
    previous_loglik: float,
    threshold: float = 1e-4,
    check_decreased: bool = True
) -> Tuple[bool, bool]:
    """Check whether EM algorithm has converged.
    
    Convergence is determined by relative change in log-likelihood:
    |loglik - previous_loglik| / avg(|loglik|, |previous_loglik|) < threshold
    
    Parameters
    ----------
    loglik : float
        Current iteration log-likelihood value.
    previous_loglik : float
        Previous iteration log-likelihood value.
    threshold : float, default 1e-4
        Convergence threshold for relative change in log-likelihood.
        Algorithm converges when relative change falls below this value.
    check_decreased : bool, default True
        If True, check for likelihood decreases and log warning.
        Useful for detecting numerical issues or convergence problems.
        
    Returns
    -------
    converged : bool
        True if algorithm has converged (relative change < threshold).
    decreased : bool
        True if likelihood decreased significantly (only if check_decreased=True).
        A decrease may indicate numerical issues or convergence problems.
    
    Notes
    -----
    - Convergence criterion matches MATLAB implementation (Nowcasting/functions/dfm.m).
    - Formula from Numerical Recipes in C (pg. 423).
    - Default threshold (1e-4) matches MATLAB default.
    - Relative change formula: delta / avg(|loglik|, |previous_loglik|).
    
    Examples
    --------
    >>> loglik_prev = -1000.0
    >>> loglik_curr = -1000.01
    >>> converged, decreased = em_converged(loglik_curr, loglik_prev)
    >>> assert converged == True  # Small relative change
    """
    converged = False
    decrease = False
    
    if check_decreased and (loglik - previous_loglik) < MIN_LOG_LIKELIHOOD_DELTA:
        _logger.warning(f"Likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}")
        decrease = True
    
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease


__all__ = [
    'em_converged',
    'init_conditions',
    'em_step',
    'EMStepParams',
    'NaNHandlingOptions',
    # Constants
    'DEFAULT_AR_COEFFICIENT',
    'DEFAULT_INNOVATION_VARIANCE',
    'DEFAULT_IDIO_AR',
    'DEFAULT_IDIO_VAR',
    'DEFAULT_IDIO_COV',
    'DEFAULT_OBSERVATION_VARIANCE',
    'MIN_INNOVATION_VARIANCE',
    'MIN_OBSERVATION_VARIANCE',
    'MIN_EIGENVALUE_THRESHOLD',
    'MIN_VARIANCE_THRESHOLD',
    'MIN_DATA_COVERAGE_RATIO',
    'MIN_EIGENVALUE_ABSOLUTE',
    'MIN_EIGENVALUE_RELATIVE',
    'MIN_LOADING',
    'MAX_LOADING',
    'MIN_AR',
    'FALLBACK_AR',
    'FALLBACK_SCALE',
    'MIN_LOG_LIKELIHOOD_DELTA',
    'DAMPING',
    'MAX_LOADING_REPLACE',
]

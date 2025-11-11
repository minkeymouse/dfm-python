"""Expectation-Maximization core routines (initialization and one EM iteration).

This module contains:
- init_conditions: compute initial A, C, Q, R, Z_0, V_0
- em_step: perform one EM iteration (E-step via Kalman, M-step updates)
- em_converged: check EM convergence
"""

from typing import Optional, Tuple, Dict, Any, TypedDict
import logging
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

from ..kalman import run_kf
from ..config import DFMConfig
from .numeric import (
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
from .helpers import safe_get_attr
from .grouping import group_series_by_frequency
from ..utils.data_utils import rem_nans_spline
from ..utils.aggregation import (
    FREQUENCY_HIERARCHY,
    generate_R_mat,
    get_tent_weights_for_pair,
    generate_tent_weights,
)

_logger = logging.getLogger(__name__)

# Numerical stability constants
DEFAULT_AR_COEFFICIENT = 0.1  # Default AR coefficient when estimation fails
DEFAULT_INNOVATION_VARIANCE = 0.1  # Default innovation variance
DEFAULT_IDIO_AR_COEFFICIENT = 0.9  # Default AR coefficient for idiosyncratic components
DEFAULT_IDIO_INNOVATION_VARIANCE = 0.1  # Default innovation variance for idiosyncratic
DEFAULT_IDIO_INIT_COVARIANCE = 0.1  # Default initial covariance for idiosyncratic
DEFAULT_OBSERVATION_VARIANCE = 1e-4  # Default observation variance (R diagonal)
MIN_INNOVATION_VARIANCE = 1e-8  # Minimum innovation variance for factor evolution
MIN_OBSERVATION_VARIANCE = 1e-8  # Minimum observation variance
MIN_EIGENVALUE_THRESHOLD = 1e-8  # Minimum eigenvalue for positive definiteness
MIN_VARIANCE_THRESHOLD = 1e-10  # Minimum variance threshold for validation

# Initialization constants
MIN_DATA_COVERAGE_RATIO = 0.5  # Minimum ratio of series required for block initialization (50%)
MIN_EIGENVALUE_ABSOLUTE = 0.1  # Absolute minimum eigenvalue for Block_Global (ensures factor can evolve)
MIN_EIGENVALUE_RELATIVE = 0.1  # Relative minimum eigenvalue (10% of max)
MIN_LOADING_ABS_THRESHOLD = 0.1  # Minimum absolute loading value before scaling
TARGET_LOADING_ABS_MAX = 0.5  # Target maximum absolute loading after scaling
MIN_AR_COEFF_ABSOLUTE = 0.1  # Minimum AR coefficient for Block_Global
FALLBACK_TRANSITION_COEFF = 0.9  # Fallback transition matrix coefficient
FALLBACK_RANDOM_SCALE = 0.1  # Scale for random fallback values

# Convergence and stability constants
LIKELIHOOD_DECREASE_THRESHOLD = -1e-3  # Threshold for detecting likelihood decrease (warns if decrease > this)
TRANSITION_DAMPING_FACTOR = 0.95  # Damping factor for transition matrix when numerical errors occur
DEFAULT_LOADING_INF_REPLACEMENT = 0.99  # Default replacement value for Inf in loading matrices (close to AR clip max)


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


def em_converged(loglik: float, previous_loglik: float, threshold: float = 1e-4,
                 check_decreased: bool = True) -> Tuple[bool, bool]:
    """Check whether EM algorithm has converged.
    
    Convergence is determined by relative change in log-likelihood:
    |loglik - previous_loglik| / avg(|loglik|, |previous_loglik|) < threshold
    
    Parameters
    ----------
    loglik : float
        Current iteration log-likelihood
    previous_loglik : float
        Previous iteration log-likelihood
    threshold : float, default 1e-4
        Convergence threshold (relative change)
    check_decreased : bool, default True
        If True, check for likelihood decreases and log warning
        
    Returns
    -------
    converged : bool
        True if algorithm has converged
    decreased : bool
        True if likelihood decreased (only if check_decreased=True)
    
    Notes
    -----
    - Convergence criterion matches MATLAB implementation (Nowcasting/functions/dfm.m)
    - Formula from Numerical Recipes in C (pg. 423)
    - Default threshold (1e-4) matches MATLAB default
    """
    converged = False
    decrease = False
    
    if check_decreased and (loglik - previous_loglik) < LIKELIHOOD_DECREASE_THRESHOLD:
        _logger.warning(f"Likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}")
        decrease = True
    
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease


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

    # Infer nQ from frequencies if needed
    if nQ is None and frequencies is not None:
        clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
        nQ = sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_h)
    elif nQ is None:
        nQ = 0

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
        r_i = int(r[i])
        factor_projection_lagged = None
        ar_coeffs = None
        F = None  # Initialize F at block level to avoid UnboundLocalError

        block_loadings = np.zeros((N, int(r_i * ppC)))
        idx_i = np.where(blocks[:, i] == 1)[0]

        freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
        idx_clock_freq = freq_groups.get(clock, np.array([], dtype=int))

        if len(idx_clock_freq) > 0:
            try:
                block_residuals = data_residuals[:, idx_clock_freq].copy()
                # For Block_Global (first block), ensure we have sufficient variation
                # Use pairwise complete observations instead of requiring all series
                if i == 0 and len(idx_clock_freq) > 1:
                    # For global block, allow missing data but require sufficient pairwise observations
                    # Calculate how many series have data at each time point
                    n_obs_per_time = np.sum(np.isfinite(block_residuals), axis=1)
                    # Require at least MIN_DATA_COVERAGE_RATIO of series to have data (or minimum 2 series)
                    min_series_required = max(2, int(len(idx_clock_freq) * MIN_DATA_COVERAGE_RATIO))
                    valid_times = n_obs_per_time >= min_series_required
                    if np.sum(valid_times) < max(10, len(idx_clock_freq) + 1):
                        # Fallback: use all available data with pairwise complete
                        finite_rows = np.any(np.isfinite(block_residuals), axis=1)
                    else:
                        finite_rows = valid_times
                else:
                    # For other blocks, require all series to be finite (original behavior)
                    finite_rows = np.all(np.isfinite(block_residuals), axis=1)
                n_finite = int(np.sum(finite_rows))
                n_total = len(finite_rows)
                if n_finite < max(2, len(idx_clock_freq) + 1):
                    _logger.warning(
                        f"init_conditions: Block {i+1} has insufficient data; using identity covariance."
                    )
                    raise ValueError("insufficient data")
                block_residuals_clean = block_residuals[finite_rows, :]
                # For Block_Global, fill remaining NaNs with column median for covariance calculation
                if i == 0 and len(idx_clock_freq) > 1:
                    for col_idx in range(block_residuals_clean.shape[1]):
                        col_data = block_residuals_clean[:, col_idx]
                        nan_mask = np.isnan(col_data)
                        if np.any(nan_mask) and np.any(~nan_mask):
                            col_median = np.nanmedian(col_data)
                            if np.isfinite(col_median):
                                block_residuals_clean[nan_mask, col_idx] = col_median
                            else:
                                block_residuals_clean[nan_mask, col_idx] = 0.0
                # Compute covariance safely with appropriate method
                # For Block_Global (i == 0), use pairwise complete for robustness.
                # This allows covariance computation even when no single time point
                # has all series observed, which is common in mixed-frequency or
                # sparse data scenarios. Pairwise complete uses all available
                # pairwise observations, making it more robust than listwise deletion.
                use_pairwise = (i == 0 and len(idx_clock_freq) > 1)
                cov_res = _compute_covariance_safe(
                    block_residuals_clean,
                    rowvar=True,  # Each row is a time period, each column is a series
                    pairwise_complete=use_pairwise,
                    min_eigenval=MIN_EIGENVALUE_THRESHOLD,
                    fallback_to_identity=True
                )
                # For Block_Global, ensure minimum variance in first principal component
                d, v = _compute_principal_components(cov_res, int(r_i), block_idx=i)
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
                if np.sum(v) < 0:
                    v = -v
                d_pos = np.maximum(d, MIN_EIGENVALUE_THRESHOLD)
                sqrt_d = np.sqrt(d_pos)
                v_scaled = v * float(sqrt_d) if np.isscalar(sqrt_d) or (isinstance(sqrt_d, np.ndarray) and sqrt_d.ndim == 0) else v @ np.diag(sqrt_d)
                if v_scaled.ndim == 2 and v_scaled.shape[1] > 0:
                    for col_idx in range(v_scaled.shape[1]):
                        col_norm = np.linalg.norm(v_scaled[:, col_idx])
                        if col_norm > 0:
                            d_val = float(d_pos) if np.isscalar(d_pos) or (isinstance(d_pos, np.ndarray) and d_pos.ndim == 0) else d_pos[col_idx]
                            v_scaled[:, col_idx] = v_scaled[:, col_idx] / col_norm * np.sqrt(d_val)
                # For Block_Global (first block, i == 0), ensure loadings have meaningful scale
                # Also check if this is the global block by checking if all series load on it
                is_global_block = (i == 0) or (blocks is not None and np.all(blocks[:, i] == 1))
                if is_global_block and len(idx_clock_freq) > 0:
                    # Check if loadings are too small
                    max_loading_abs = np.max(np.abs(v_scaled))
                    if max_loading_abs < MIN_LOADING_ABS_THRESHOLD:
                        # Scale up loadings to have at least TARGET_LOADING_ABS_MAX max absolute value
                        scale_factor = TARGET_LOADING_ABS_MAX / max_loading_abs if max_loading_abs > 0 else 1.0
                        v_scaled = v_scaled * scale_factor
                        _logger.warning(
                            f"init_conditions: Block {i+1} (Global) loadings too small "
                            f"(max_abs={max_loading_abs:.6e}), scaling by {scale_factor:.2f} to {np.max(np.abs(v_scaled)):.6e}"
                        )
                # For all blocks, ensure loadings are not all zero
                max_loading_abs_all = np.max(np.abs(v_scaled)) if v_scaled.size > 0 else 0.0
                if max_loading_abs_all < MIN_VARIANCE_THRESHOLD and len(idx_clock_freq) > 0 and int(r_i) > 0:
                    # If all loadings are essentially zero, use uniform loadings as fallback
                    v_scaled = np.ones((len(idx_clock_freq), int(r_i))) * MIN_LOADING_ABS_THRESHOLD / np.sqrt(len(idx_clock_freq))
                    _logger.warning(
                        f"init_conditions: Block {i+1} loadings all zero (max_abs={max_loading_abs_all:.6e}), "
                        f"using uniform fallback (shape={v_scaled.shape})"
                    )
                block_loadings[idx_clock_freq, :int(r_i)] = v_scaled
                f = data_residuals[:, idx_clock_freq] @ v_scaled
                F = None
                max_lag = max(p + 1, pC)
                for kk in range(max_lag):
                    lag_data = f[pC - kk:T - kk, :]
                    F = lag_data if F is None else np.hstack([F, lag_data])
                if F is not None:
                    factor_projection_lagged = F[:, :int(r_i * pC)]
                else:
                    factor_projection_lagged = None
                for freq, idx_iFreq in freq_groups.items():
                    if freq == clock:
                        continue
                    tent_weights = tent_weights_dict.get(freq) if (tent_weights_dict and freq in tent_weights_dict) else get_tent_weights_for_pair(freq, clock)
                    if tent_weights is None:
                        clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
                        freq_h = FREQUENCY_HIERARCHY.get(freq, 3)
                        n_periods_est = freq_h - clock_h + 1
                        if 0 < n_periods_est <= 12:
                            tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                            _logger.warning(f"init_conditions: generated symmetric tent weights for '{freq}'")
                        else:
                            raise ValueError(f"init_conditions: cannot determine tent weights for '{freq}'")
                    R_mat_freq, q_freq = generate_R_mat(tent_weights)
                    pC_freq = len(tent_weights)
                    if factor_projection_lagged.shape[1] < r_i * pC_freq:
                        factor_projection_freq = np.hstack([
                            factor_projection_lagged,
                            np.zeros((factor_projection_lagged.shape[0], r_i * pC_freq - factor_projection_lagged.shape[1]))
                        ])
                    else:
                        factor_projection_freq = factor_projection_lagged[:, :int(r_i * pC_freq)]
                    Rcon_i = np.kron(R_mat_freq, np.eye(int(r_i)))
                    q_i = np.kron(q_freq, np.zeros(int(r_i)))
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
                        if len(series_data_clean) > 0 and factor_projection_clean.shape[0] > 0:
                            try:
                                gram = factor_projection_clean.T @ factor_projection_clean
                                gram_inv = inv(gram)
                                loadings = gram_inv @ factor_projection_clean.T @ series_data_clean
                                if Rcon_i is not None and q_i is not None and Rcon_i.size > 0:
                                    constraint_term = gram_inv @ Rcon_i.T @ inv(Rcon_i @ gram_inv @ Rcon_i.T) @ (Rcon_i @ loadings - q_i)
                                    loadings = loadings - constraint_term
                                block_loadings[j, :int(pC_freq * r_i)] = loadings[:int(pC_freq * r_i)]
                            except _NUMERICAL_EXCEPTIONS:
                                block_loadings[j, :int(pC_freq * r_i)] = 0.0
            except _NUMERICAL_EXCEPTIONS:
                _logger.warning(
                    f"init_conditions: Block {i+1} initialization failed due to numerical error; "
                    f"using fallback (identity loadings)."
                )
                if len(idx_clock_freq) > 0:
                    block_loadings[idx_clock_freq, :int(r_i)] = np.eye(len(idx_clock_freq), int(r_i))[:len(idx_clock_freq), :int(r_i)]
        if factor_projection_lagged is not None:
            expected_width = int(pC * r_i)
            actual_width = factor_projection_lagged.shape[1]
            if actual_width != expected_width:
                _logger.warning(
                    f"init_conditions: Factor projection width mismatch for block {i+1} "
                    f"(got {actual_width}, expected {expected_width}); resetting."
                )
                factor_projection_lagged = None
        if factor_projection_lagged is not None:
            factor_projection_padded = np.vstack([np.zeros((pC - 1, int(pC * r_i))), factor_projection_lagged])
            if factor_projection_padded.shape[0] < T:
                factor_projection_padded = np.vstack([factor_projection_padded, np.zeros((T - factor_projection_padded.shape[0], int(pC * r_i)))])
        else:
            factor_projection_padded = np.zeros((T, int(pC * r_i)))
        block_loadings_residual = block_loadings[:, :int(pC * r_i)]
        if factor_projection_padded.shape[0] != data_residuals.shape[0]:
            if factor_projection_padded.shape[0] > data_residuals.shape[0]:
                factor_projection_padded = factor_projection_padded[:data_residuals.shape[0], :]
            else:
                factor_projection_padded = np.vstack([factor_projection_padded, np.zeros((data_residuals.shape[0] - factor_projection_padded.shape[0], factor_projection_padded.shape[1]))])
        data_residuals[:, idx_i] = data_residuals[:, idx_i] - factor_projection_padded @ block_loadings_residual[idx_i, :].T
        residuals_with_nan = data_residuals.copy()
        residuals_with_nan[missing_data_mask] = np.nan
        C = block_loadings if C is None else np.hstack([C, block_loadings])
        if len(idx_clock_freq) > 0 and F is not None:
            z = F[:, :int(r_i)]
            Z_lag = F[:, int(r_i):int(r_i * (p + 1))] if F.shape[1] > int(r_i) else np.zeros((F.shape[0], int(r_i * p)))
            block_transition = np.zeros((int(r_i * ppC), int(r_i * ppC)))
            if Z_lag.shape[0] > 0 and Z_lag.shape[1] > 0:
                try:
                    ar_coeffs = inv(Z_lag.T @ Z_lag) @ Z_lag.T @ z
                    block_transition[:int(r_i), :int(r_i * p)] = ar_coeffs.T
                    # For Block_Global (i == 0), ensure AR coefficient is not too small
                    if i == 0 and r_i > 0 and p > 0:
                        ar_coeff_0 = block_transition[0, 0]
                        if np.abs(ar_coeff_0) < MIN_AR_COEFF_ABSOLUTE:
                            # Use default AR coefficient if estimated value is too small
                            block_transition[0, 0] = DEFAULT_AR_COEFFICIENT if ar_coeff_0 >= 0 else -DEFAULT_AR_COEFFICIENT
                            _logger.warning(
                                f"init_conditions: Block_Global AR coefficient too small "
                                f"({ar_coeff_0:.6e}), using default {block_transition[0, 0]:.2f}"
                            )
                except _NUMERICAL_EXCEPTIONS:
                    block_transition[:int(r_i), :int(r_i * p)] = 0.0
                    # For Block_Global, use default AR coefficient instead of 0
                    if i == 0 and r_i > 0 and p > 0:
                        block_transition[0, 0] = DEFAULT_AR_COEFFICIENT
            if r_i * (ppC - 1) > 0:
                block_transition[int(r_i):, :int(r_i * (ppC - 1))] = np.eye(int(r_i * (ppC - 1)))
            block_innovation_cov = np.zeros((int(ppC * r_i), int(ppC * r_i)))
            if len(z) > 0:
                innovation_residuals = z - Z_lag @ ar_coeffs if ar_coeffs is not None else z
                # Clean residuals using consistent matrix cleaning utility
                innovation_residuals = _clean_matrix(innovation_residuals, 'general', default_nan=0.0, default_inf=0.0)
                if innovation_residuals.shape[1] > 1:
                    try:
                        Q_block = np.cov(innovation_residuals.T)
                        if np.any(~np.isfinite(Q_block)):
                            Q_block = np.eye(int(r_i)) * DEFAULT_INNOVATION_VARIANCE
                    except _NUMERICAL_EXCEPTIONS:
                        Q_block = np.eye(int(r_i)) * DEFAULT_INNOVATION_VARIANCE
                else:
                    # Single series case: compute variance safely
                    variance = _compute_variance_safe(
                        innovation_residuals,
                        ddof=0,
                        min_variance=MIN_INNOVATION_VARIANCE,
                        default_variance=DEFAULT_INNOVATION_VARIANCE
                    )
                    Q_block = np.array([[variance]]) if innovation_residuals.ndim > 1 else np.eye(int(r_i)) * variance
                block_innovation_cov[:int(r_i), :int(r_i)] = Q_block
            # Ensure Q diagonal has minimum value - critical for factor evolution
            block_innovation_cov[:int(r_i), :int(r_i)] = _ensure_innovation_variance_minimum(
                block_innovation_cov[:int(r_i), :int(r_i)], min_variance=MIN_INNOVATION_VARIANCE
            )
            block_transition_clean = _clean_matrix(block_transition, 'loading')
            block_innovation_cov_clean = _clean_matrix(block_innovation_cov, 'covariance', default_nan=0.0)
            # Re-ensure minimum after cleaning (in case cleaning set values to 0)
            block_innovation_cov_clean[:int(r_i), :int(r_i)] = _ensure_innovation_variance_minimum(
                block_innovation_cov_clean[:int(r_i), :int(r_i)], min_variance=MIN_INNOVATION_VARIANCE
            )
            try:
                kron_transition = np.kron(block_transition_clean, block_transition_clean)
                identity_kron = np.eye(int((r_i * ppC) ** 2)) - kron_transition
                innovation_cov_flat = block_innovation_cov_clean.flatten()
                init_cov_block = np.reshape(inv(identity_kron) @ innovation_cov_flat, (int(r_i * ppC), int(r_i * ppC)))
                if np.any(~np.isfinite(init_cov_block)):
                    raise ValueError("invalid init_cov_block")
            except _NUMERICAL_EXCEPTIONS:
                _logger.warning(
                    f"init_conditions: Initial covariance computation failed for block {i+1} "
                    f"(kron/inv failed); using diagonal fallback ({DEFAULT_IDIO_INIT_COVARIANCE} * I)."
                )
                init_cov_block = np.eye(int(r_i * ppC)) * DEFAULT_IDIO_INIT_COVARIANCE
            if A is None:
                A, Q, V_0 = block_transition, block_innovation_cov, init_cov_block
            else:
                A = block_diag(A, block_transition)
                Q = block_diag(Q, block_innovation_cov)
                V_0 = block_diag(V_0, init_cov_block)
        else:
            block_transition = np.eye(int(r_i * ppC)) * FALLBACK_TRANSITION_COEFF
            block_innovation_cov = np.eye(int(r_i * ppC)) * DEFAULT_INNOVATION_VARIANCE
            init_cov_block = np.eye(int(r_i * ppC)) * DEFAULT_IDIO_INIT_COVARIANCE
            if A is None:
                A, Q, V_0 = block_transition, block_innovation_cov, init_cov_block
            else:
                A = block_diag(A, block_transition)
                Q = block_diag(Q, block_innovation_cov)
                V_0 = block_diag(V_0, init_cov_block)

    eyeN = np.eye(N)[:, i_idio.astype(bool)]
    C = eyeN if C is None else np.hstack([C, eyeN])

    if nQ > 0 and frequencies is not None:
        clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
        slower_indices = [j for j in range(N) if j < len(frequencies) and FREQUENCY_HIERARCHY.get(frequencies[j], 3) > clock_h]
        Rdiag = np.nanvar(residuals_with_nan, axis=0)
        # Replace NaN/Inf with default variance (1D array - use np.where for efficiency)
        Rdiag = np.where((np.isnan(Rdiag) | np.isinf(Rdiag)), DEFAULT_OBSERVATION_VARIANCE, Rdiag)
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
            tent_weights = tent_weights_dict.get(freq) if (tent_weights_dict and freq in tent_weights_dict) else get_tent_weights_for_pair(freq, clock)
            if tent_weights is None:
                clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
                freq_h = FREQUENCY_HIERARCHY.get(freq, 3)
                n_periods_est = freq_h - clock_h + 1
                if 0 < n_periods_est <= 12:
                    tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                else:
                    raise ValueError(f"init_conditions: cannot determine tent weights for '{freq}'")
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
        Rdiag = np.where((np.isnan(Rdiag) | np.isinf(Rdiag) | (Rdiag < 0)), DEFAULT_OBSERVATION_VARIANCE, Rdiag)
        R = np.diag(Rdiag)
    else:
        var_values = np.nanvar(residuals_with_nan, axis=0)
        var_values = np.where((np.isnan(var_values) | np.isinf(var_values)), DEFAULT_OBSERVATION_VARIANCE, var_values)
        var_values = np.maximum(var_values, DEFAULT_OBSERVATION_VARIANCE)
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
                    ar_coeff = (1.0 / XTX[0, 0]) * (X_ar.T @ y_ar)
                    ar_coeff_clean = ar_coeff[0, 0] if ar_coeff.size > 0 else DEFAULT_AR_COEFFICIENT
                    ar_coeff_clean = DEFAULT_AR_COEFFICIENT if (np.isnan(ar_coeff_clean) or np.isinf(ar_coeff_clean)) else ar_coeff_clean
                    BM[idx, idx] = ar_coeff_clean
                    innovation_residuals = series_residuals_truncated[1:] - series_residuals_truncated[:-1] * BM[idx, idx]
                    innovation_variance = _compute_variance_safe(
                        innovation_residuals,
                        ddof=0,
                        min_variance=MIN_INNOVATION_VARIANCE,
                        default_variance=DEFAULT_INNOVATION_VARIANCE
                    )
                    SM[idx, idx] = innovation_variance
                except _NUMERICAL_EXCEPTIONS:
                    BM[idx, idx] = DEFAULT_AR_COEFFICIENT
                    SM[idx, idx] = DEFAULT_INNOVATION_VARIANCE
            else:
                BM[idx, idx] = DEFAULT_AR_COEFFICIENT
                SM[idx, idx] = DEFAULT_INNOVATION_VARIANCE
        else:
            BM[idx, idx] = DEFAULT_AR_COEFFICIENT
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
        BQ = np.eye(n_slower_idio) * DEFAULT_IDIO_AR_COEFFICIENT  # AR coefficient for slower idio
        SQ = np.eye(n_slower_idio) * DEFAULT_IDIO_INNOVATION_VARIANCE  # Innovation variance
        initViQ = np.eye(n_slower_idio) * DEFAULT_IDIO_INIT_COVARIANCE  # Initial covariance
    else:
        BQ = np.array([]).reshape(0, 0)
        SQ = np.array([]).reshape(0, 0)
        initViQ = np.array([]).reshape(0, 0)
    A = block_diag(A, BM, BQ)
    Q = block_diag(Q, SM, SQ)
    # Ensure Q diagonal has minimum value - critical for factor evolution
    if Q.size > 0:
        Q = _ensure_innovation_variance_minimum(Q, min_variance=MIN_INNOVATION_VARIANCE)
    Z_0 = np.zeros(A.shape[0])
    # Ensure all covariance matrices are square
    V_0 = _ensure_square_matrix(V_0, method='diag')
    initViM = _ensure_square_matrix(initViM, method='diag')
    initViQ = _ensure_square_matrix(initViQ, method='diag')
    V_0 = block_diag(V_0, initViM, initViQ)
    # Final clean/validate using _check_finite for consistency
    outputs = {'A': A, 'C': C, 'Q': Q, 'R': R, 'Z_0': Z_0, 'V_0': V_0}
    for param_name, param_value in outputs.items():
        if param_value.size > 0 and not _check_finite(param_value, param_name):
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
    if outputs['Q'].size > 0:
        outputs['Q'] = _ensure_innovation_variance_minimum(outputs['Q'], min_variance=MIN_INNOVATION_VARIANCE)
    return outputs['A'], outputs['C'], outputs['Q'], outputs['R'], outputs['Z_0'], outputs['V_0']


def em_step(
    y: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Z_0: np.ndarray,
    V_0: np.ndarray,
    r: np.ndarray,
    p: int,
    R_mat: Optional[np.ndarray],
    q: Optional[np.ndarray],
    nQ: Optional[int],
    i_idio: np.ndarray,
    blocks: np.ndarray,
    tent_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    clock: str = 'm',
    frequencies: Optional[np.ndarray] = None,
    config: Optional[DFMConfig] = None
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
    y : np.ndarray
        Observation matrix (n x T), where n is number of series and T is time periods.
        Missing values should be NaN. These are handled automatically by the Kalman filter.
    A : np.ndarray
        Current transition matrix estimate (m x m), where m is state dimension.
    C : np.ndarray
        Current loading matrix estimate (n x m).
    Q : np.ndarray
        Current innovation covariance estimate (m x m).
    R : np.ndarray
        Current observation covariance estimate (n x n), typically diagonal.
    Z_0 : np.ndarray
        Current initial state estimate (m,).
    V_0 : np.ndarray
        Current initial covariance estimate (m x m).
    r : np.ndarray
        Number of factors per block, shape (n_blocks,).
    p : int
        AR lag order for factor dynamics. Typically p=1.
    R_mat : np.ndarray, optional
        Constraint matrix for tent kernel aggregation in mixed-frequency models.
    q : np.ndarray, optional
        Constraint vector for tent kernel aggregation.
    nQ : int, optional
        Number of slower-frequency series. If None, inferred from frequencies.
    i_idio : np.ndarray
        Indicator array (n,) where 1 indicates clock frequency, 0 indicates slower.
    blocks : np.ndarray
        Block structure matrix (n x n_blocks).
    tent_weights_dict : dict, optional
        Dictionary mapping frequency strings to tent weight arrays.
    clock : str, default 'm'
        Clock frequency for latent factors ('m', 'q', 'sa', 'a').
    frequencies : np.ndarray, optional
        Array of frequency strings (n,) for each series.
    config : DFMConfig, optional
        Configuration object for numerical stability parameters:
        - min_eigenvalue: Minimum eigenvalue threshold for regularization
        - max_eigenvalue: Maximum eigenvalue threshold for capping
        - warn_on_regularization: Whether to warn when regularization is applied
        
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
    >>> from dfm_python.core.em import em_step
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
    >>> # Perform one EM iteration
    >>> C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
    ...     y, A, C, Q, R, Z_0, V_0, r, p=1,
    ...     R_mat=None, q=None, nQ=0, i_idio=np.ones(10),
    ...     blocks=blocks, clock='m'
    ... )
    >>> assert C_new.shape == C.shape
    >>> assert np.isfinite(loglik)
    """
    # Validate and clean input parameters
    if not _check_finite(A, "A"):
        _logger.warning(f"em_step: A contains NaN/Inf, applying regularization ({FALLBACK_TRANSITION_COEFF}*I + {1-FALLBACK_TRANSITION_COEFF}*cleaned)")
        A = np.eye(A.shape[0]) * FALLBACK_TRANSITION_COEFF + _clean_matrix(A, 'loading') * (1 - FALLBACK_TRANSITION_COEFF)
    if not _check_finite(Q, "Q"):
        _logger.warning(f"em_step: Q contains NaN/Inf, applying regularization (min={MIN_INNOVATION_VARIANCE})")
        Q = _clean_matrix(Q, 'covariance', default_nan=MIN_INNOVATION_VARIANCE)
    if not _check_finite(R, "R"):
        _logger.warning(f"em_step: R contains NaN/Inf, applying regularization (min={DEFAULT_OBSERVATION_VARIANCE})")
        R = _clean_matrix(R, 'diagonal', default_nan=DEFAULT_OBSERVATION_VARIANCE, default_inf=1e4)
    if not _check_finite(C, "C"):
        _logger.warning("em_step: C contains NaN/Inf, cleaning matrix")
        C = _clean_matrix(C, 'loading')
    if not _check_finite(Z_0, "Z_0"):
        _logger.warning("em_step: Z_0 contains NaN/Inf, resetting to zeros")
        Z_0 = np.zeros_like(Z_0)
    if not _check_finite(V_0, "V_0"):
        _logger.warning(f"em_step: V_0 contains NaN/Inf, using regularized identity ({DEFAULT_IDIO_INIT_COVARIANCE}*I)")
        V_0 = np.eye(V_0.shape[0]) * DEFAULT_IDIO_INIT_COVARIANCE

    n, T = y.shape
    if nQ is None and frequencies is not None:
        clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
        nQ = sum(1 for f in frequencies if FREQUENCY_HIERARCHY.get(f, 3) > clock_h)
    elif nQ is None:
        nQ = 0

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
        r_i = int(r[i])
        factor_lag_size = r_i * p
        factor_start_idx = int(np.sum(r[:i]) * ppC)
        t_start = factor_start_idx
        t_end = int(factor_start_idx + r_i * ppC)
        b_subset = slice(factor_start_idx, factor_start_idx + factor_lag_size)

        expected_factor_outer = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, 1:].T + np.sum(vsmooth[b_subset, :, :][:, b_subset, 1:], axis=2)
        expected_factor_lag_outer = Zsmooth[b_subset, :-1] @ Zsmooth[b_subset, :-1].T + np.sum(vsmooth[b_subset, :, :][:, b_subset, :-1], axis=2)
        expected_factor_lag_cross = Zsmooth[b_subset, 1:] @ Zsmooth[b_subset, :-1].T + np.sum(vvsmooth[b_subset, :, :][:, b_subset, :], axis=2)

        expected_factor_lag_outer = _clean_matrix(expected_factor_lag_outer, 'covariance', default_nan=0.0)
        expected_factor_lag_cross = _clean_matrix(expected_factor_lag_cross, 'general', default_nan=0.0)

        block_transition = A[t_start:t_end, t_start:t_end].copy()
        block_innovation_cov = Q[t_start:t_end, t_start:t_end].copy()
        try:
            expected_lag_sub = expected_factor_lag_outer[:factor_lag_size, :factor_lag_size]
            min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
            warn_reg = safe_get_attr(config, "warn_on_regularization", True)
            expected_lag_sub, _ = _ensure_positive_definite(expected_lag_sub, min_eigenval, warn_reg)
            try:
                eigenvals = np.linalg.eigvals(expected_lag_sub)
                cond_num = (np.max(eigenvals) / max(np.min(eigenvals), 1e-12)) if np.max(eigenvals) > 0 else 1.0
                expected_lag_inv = pinv(expected_lag_sub, cond=1e-8) if cond_num > 1e12 else inv(expected_lag_sub)
            except _NUMERICAL_EXCEPTIONS:
                expected_lag_inv = pinv(expected_lag_sub)
            transition_update = expected_factor_lag_cross[:r_i, :factor_lag_size] @ expected_lag_inv
            transition_update, _ = _apply_ar_clipping(transition_update, config)
            block_transition[:r_i, :factor_lag_size] = transition_update
            block_innovation_cov[:r_i, :r_i] = (
                expected_factor_outer[:r_i, :r_i] -
                block_transition[:r_i, :factor_lag_size] @ expected_factor_lag_cross[:r_i, :factor_lag_size].T
            ) / T
            block_innovation_cov = _clean_matrix(block_innovation_cov, 'covariance', default_nan=0.0)
            # Ensure diagonal elements have minimum value to prevent Q from becoming zero
            # This is critical for factor evolution - Q[i,i] = 0 means factor i doesn't evolve
            block_innovation_cov[:r_i, :r_i] = _ensure_innovation_variance_minimum(
                block_innovation_cov[:r_i, :r_i], min_variance=MIN_INNOVATION_VARIANCE
            )
            innovation_cov_reg, _ = _ensure_positive_definite(block_innovation_cov[:r_i, :r_i], min_eigenval, warn_reg)
            block_innovation_cov[:r_i, :r_i] = innovation_cov_reg
            # Re-ensure minimum after positive definite check (it may have changed diagonal)
            block_innovation_cov[:r_i, :r_i] = _ensure_innovation_variance_minimum(
                block_innovation_cov[:r_i, :r_i], min_variance=MIN_INNOVATION_VARIANCE
            )
            max_eig = safe_get_attr(config, "max_eigenvalue", 1e6)
            block_innovation_cov[:r_i, :r_i] = _cap_max_eigenvalue(block_innovation_cov[:r_i, :r_i], max_eigenval=max_eig)
            # Re-ensure minimum after max eigenvalue capping (it may have changed diagonal)
            block_innovation_cov[:r_i, :r_i] = _ensure_innovation_variance_minimum(
                block_innovation_cov[:r_i, :r_i], min_variance=MIN_INNOVATION_VARIANCE
            )
        except _NUMERICAL_EXCEPTIONS:
            if np.allclose(block_transition[:r_i, :factor_lag_size], 0):
                block_transition[:r_i, :factor_lag_size] = np.random.randn(r_i, factor_lag_size) * FALLBACK_RANDOM_SCALE
            else:
                block_transition[:r_i, :factor_lag_size] *= TRANSITION_DAMPING_FACTOR
        if np.any(~np.isfinite(block_transition)):
            block_transition = _clean_matrix(block_transition, 'loading', default_nan=0.0, default_inf=DEFAULT_LOADING_INF_REPLACEMENT)
            block_transition, _ = _apply_ar_clipping(block_transition, config)
        A_new[t_start:t_end, t_start:t_end] = block_transition
        Q_new[t_start:t_end, t_start:t_end] = block_innovation_cov
        V_0_block = _clean_matrix(vsmooth[t_start:t_end, t_start:t_end, 0], 'covariance', default_nan=0.0)
        V_0_block, _ = _ensure_positive_definite(V_0_block, min_eigenval, warn_reg)
        V_0_new[t_start:t_end, t_start:t_end] = V_0_block

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
    block_transition_idio = np.diag(ar_coeffs_diag)
    innovation_cov_diag = (np.maximum(expected_idio_current_sq, 0.0) - ar_coeffs_diag * expected_idio_cross) / T
    innovation_cov_diag = np.maximum(innovation_cov_diag, MIN_INNOVATION_VARIANCE)
    block_innovation_cov_idio = np.diag(innovation_cov_diag)
    i_subset_size = i_subset.stop - i_subset.start
    if n_idio_actual == i_subset_size:
        A_new[i_subset, i_subset] = block_transition_idio
        Q_new[i_subset, i_subset] = block_innovation_cov_idio
    elif n_idio_actual < i_subset_size:
        A_new[i_subset.start:i_subset.start + n_idio_actual, i_subset.start:i_subset.start + n_idio_actual] = block_transition_idio
        Q_new[i_subset.start:i_subset.start + n_idio_actual, i_subset.start:i_subset.start + n_idio_actual] = block_innovation_cov_idio
    else:
        A_new[i_subset, i_subset] = block_transition_idio[:i_subset_size, :i_subset_size]
        Q_new[i_subset, i_subset] = block_innovation_cov_idio[:i_subset_size, :i_subset_size]
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
        r_i = int(r[i])
        bl_col_clock_freq = np.repeat(bl[:, i:i+1], r_i, axis=1)
        bl_col_clock_freq = np.hstack([bl_col_clock_freq, np.zeros((n_bl, r_i * (ppC - 1)))])
        bl_col_slower_freq = np.repeat(bl[:, i:i+1], r_i * ppC, axis=1)
        if bl_idx_same_freq is None:
            bl_idx_same_freq = bl_col_clock_freq
            bl_idx_slower_freq = bl_col_slower_freq
        else:
            bl_idx_same_freq = np.hstack([bl_idx_same_freq, bl_col_clock_freq])
            bl_idx_slower_freq = np.hstack([bl_idx_slower_freq, bl_col_slower_freq])
        if R_mat is not None:
            R_con_list.append(np.kron(R_mat, np.eye(r_i)))
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
        idx_clock_freq = freq_groups.get(clock, np.array([], dtype=int))
        n_i = len(idx_clock_freq)
        if n_i == 0:
            continue
        bl_idx_same_freq_i = np.where(bl_idx_same_freq[i, :])[0]
        if len(bl_idx_same_freq_i) == 0:
            continue
        rs_actual = len(bl_idx_same_freq_i)
        if rs_actual != rs:
            rs = rs_actual
        denom_size = n_i * rs
        denom = np.zeros((denom_size, denom_size))
        nom = np.zeros((n_i, rs))
        i_idio_i = i_idio_same[idx_clock_freq]
        i_idio_ii = np.cumsum(i_idio.astype(int))[idx_clock_freq]
        i_idio_ii = i_idio_ii[i_idio_i.astype(bool)]
        for t in range(T):
            nan_mask = ~nanY[idx_clock_freq, t]
            Wt = np.diag(nan_mask.astype(float))
            if t + 1 < Zsmooth.shape[0]:
                Z_block_same_freq_row = Zsmooth[t + 1, bl_idx_same_freq_i]
                ZZZ = Z_block_same_freq_row.reshape(-1, 1) @ Z_block_same_freq_row.reshape(1, -1)
            else:
                ZZZ = np.zeros((rs, rs))
            if t + 1 < vsmooth.shape[2]:
                V_block_same_freq = vsmooth[np.ix_(bl_idx_same_freq_i, bl_idx_same_freq_i, [t + 1])]
                V_block_same_freq = V_block_same_freq[:, :, 0] if V_block_same_freq.ndim == 3 else V_block_same_freq
                if V_block_same_freq.shape != (rs, rs):
                    V_block_same_freq = np.zeros((rs, rs))
            else:
                V_block_same_freq = np.zeros((rs, rs))
            expected_shape = (denom_size, denom_size)
            try:
                kron_result = np.kron(ZZZ + V_block_same_freq, Wt)
                if kron_result.shape == expected_shape:
                    denom += kron_result
            except _NUMERICAL_EXCEPTIONS:
                pass
            if t + 1 < Zsmooth.shape[0]:
                y_vec = y_clean[idx_clock_freq, t].reshape(-1, 1)
                Z_vec_row = Zsmooth[t + 1, bl_idx_same_freq_i].reshape(1, -1)
                y_term = y_vec @ Z_vec_row
            else:
                y_term = np.zeros((len(idx_clock_freq), rs_actual))
            if len(i_idio_ii) > 0 and t + 1 < Zsmooth.shape[0]:
                idio_idx = (idio_start_idx + i_idio_ii).astype(int)
                if idio_idx.max() < Zsmooth.shape[1]:
                    idio_Z_col = Zsmooth[t + 1, idio_idx].reshape(-1, 1)
                    idio_Z_outer = idio_Z_col @ Z_vec_row
                    if t + 1 < vsmooth.shape[2]:
                        idio_V = vsmooth[np.ix_(idio_idx, bl_idx_same_freq_i, [t + 1])]
                        idio_V = idio_V[:, :, 0] if idio_V.ndim == 3 else idio_V
                    else:
                        idio_V = np.zeros((len(i_idio_ii), rs_actual))
                    idio_term = Wt[:, i_idio_i.astype(bool)] @ (idio_Z_outer + idio_V)
                else:
                    idio_term = np.zeros((len(idx_clock_freq), rs_actual))
            else:
                idio_term = np.zeros((len(idx_clock_freq), rs_actual))
            nom += y_term - idio_term
        try:
            scale_factor = safe_get_attr(config, "regularization_scale", 1e-5)
            warn_reg = safe_get_attr(config, "warn_on_regularization", True)
            reg_param, _ = _compute_regularization_param(denom, scale_factor, warn_reg)
            denom_reg = denom + np.eye(denom.shape[0]) * reg_param
            vec_C = inv(denom_reg) @ nom.flatten()
            C_update = vec_C.reshape(n_i, rs)
            C_update = _clean_matrix(C_update, 'loading', default_nan=0.0, default_inf=0.0)
            C_scale = np.std(C_update[C_update != 0]) if np.any(C_update != 0) else 1.0
            C_max = max(10.0, C_scale * 5)
            C_update = np.clip(C_update, -C_max, C_max)
            for ii, row_idx in enumerate(idx_clock_freq):
                for jj, col_idx in enumerate(bl_idx_same_freq_i):
                    C_new[row_idx, col_idx] = C_update[ii, jj]
        except _NUMERICAL_EXCEPTIONS:
            pass
        for freq, idx_iFreq in freq_groups.items():
            if freq == clock:
                continue
            tent_weights = tent_weights_dict.get(freq) if (tent_weights_dict and freq in tent_weights_dict) else get_tent_weights_for_pair(freq, clock)
            if tent_weights is None:
                clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
                freq_h = FREQUENCY_HIERARCHY.get(freq, 3)
                n_periods_est = freq_h - clock_h + 1
                if 0 < n_periods_est <= 12:
                    tent_weights = generate_tent_weights(n_periods_est, 'symmetric')
                    _logger.warning(f"em_step: generated symmetric tent weights for '{freq}'")
                else:
                    raise ValueError(f"em_step: cannot determine tent weights for '{freq}'")
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
            if R_con_i.size > 0:
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
                    if t + 1 < Zsmooth.shape[0]:
                        Z_row = Zsmooth[t + 1, valid_bl_idx]
                        Z_col = Z_row.reshape(-1, 1)
                        ZZZ = Z_col @ Z_row.reshape(1, -1)
                        valid_vs_idx = valid_bl_idx[valid_bl_idx < vsmooth.shape[0]]
                        if len(valid_vs_idx) > 0:
                            V_block = vsmooth[np.ix_(valid_vs_idx, valid_vs_idx, [t + 1])]
                            V_block = V_block[:, :, 0] if V_block.ndim == 3 else V_block
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
                try:
                    scale_factor = safe_get_attr(config, "regularization_scale", 1e-5)
                    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
                    reg_param, _ = _compute_regularization_param(denom, scale_factor, warn_reg)
                    denom_reg = denom + np.eye(denom.shape[0]) * reg_param
                    C_i = inv(denom_reg) @ nom.T
                    C_i = _clean_matrix(C_i, 'loading', default_nan=0.0, default_inf=0.0)
                    C_scale = np.std(C_i[C_i != 0]) if np.any(C_i != 0) else 1.0
                    C_max = max(10.0, C_scale * 5)
                    C_i = np.clip(C_i, -C_max, C_max)
                    if len(bl_idx_slower_freq_i) > 0:
                        C_update = C_i.flatten()[:len(bl_idx_slower_freq_i)]
                        for k, col_idx in enumerate(bl_idx_slower_freq_i):
                            C_new[j, col_idx] = C_update[k]
                except _NUMERICAL_EXCEPTIONS:
                    pass

    R_diag = np.zeros(n)
    n_obs_per_series = np.zeros(n, dtype=int)
    for t in range(T):
        Z_t = Zsmooth[t + 1, :].reshape(-1, 1)
        vsmooth_t = vsmooth[:, :, t + 1]
        y_pred = (C_new @ Z_t).flatten()
        for i in range(n):
            if np.isnan(y[i, t]):
                continue
            n_obs_per_series[i] += 1
            resid_sq = (y[i, t] - y_pred[i]) ** 2
            C_i = C_new[i, :].reshape(1, -1)
            var_factor = (C_i @ vsmooth_t @ C_i.T)[0, 0]
            R_diag[i] += resid_sq + var_factor
    n_obs_per_series = np.maximum(n_obs_per_series, 1)
    R_diag = R_diag / n_obs_per_series
    mean_var = np.mean(R_diag[R_diag > 0]) if np.any(R_diag > 0) else DEFAULT_OBSERVATION_VARIANCE
    min_var = np.maximum(mean_var * MIN_DIAGONAL_VARIANCE, MIN_OBSERVATION_VARIANCE)
    R_diag = np.maximum(R_diag, min_var)
    valid_mask = np.isfinite(R_diag) & (R_diag > 0)
    if np.any(valid_mask):
        median_var = np.median(R_diag[valid_mask])
        R_diag = np.where(valid_mask, R_diag, median_var)
    else:
        R_diag.fill(DEFAULT_OBSERVATION_VARIANCE)
    R_new = np.diag(R_diag)
    Q_new = _clean_matrix(Q_new, 'covariance', default_nan=0.0)
    min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
    max_eigenval = safe_get_attr(config, "max_eigenvalue", 1e6)
    # Apply stability operations: PSD, eigenvalue capping, then enforce minimum variance
    # Note: We enforce minimum after all operations since PSD and capping may affect diagonal
    Q_new, _ = _ensure_positive_definite(Q_new, min_eigenval, warn_reg)
    Q_new = _cap_max_eigenvalue(Q_new, max_eigenval=max_eigenval)
    # Final enforcement: ensure Q diagonal has minimum value - critical for factor evolution
    Q_new = _ensure_innovation_variance_minimum(Q_new, min_variance=MIN_INNOVATION_VARIANCE)
    V_0_new = _clean_matrix(V_0_new, 'covariance', default_nan=0.0)
    V_0_new, _ = _ensure_positive_definite(V_0_new, min_eigenval, warn_reg)
    return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik



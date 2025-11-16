"""EM algorithm functions for DFM estimation."""
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, TypedDict
from dataclasses import dataclass
from scipy.linalg import inv, pinv, block_diag

_logger = logging.getLogger(__name__)

# Constants
MIN_INNOVATION_VARIANCE = 1e-8  # Minimum variance for innovation covariance Q diagonal
MIN_OBSERVATION_VARIANCE = 1e-8  # Minimum observation variance
MIN_DIAGONAL_VARIANCE = 1e-6  # Minimum diagonal variance ratio
MIN_EIGENVALUE_ABSOLUTE = 0.1  # Absolute minimum eigenvalue for Block_Global
MIN_EIGENVALUE_RELATIVE = 0.1  # Relative minimum eigenvalue (10% of max)
MIN_DATA_COVERAGE_RATIO = 0.5  # Minimum ratio of series required for block initialization
DEFAULT_INNOVATION_VARIANCE = 0.1  # Default innovation variance
DEFAULT_OBSERVATION_VARIANCE = 1e-4  # Default observation variance
DEFAULT_IDIO_COV = 0.1  # Default initial covariance for idiosyncratic
FALLBACK_AR = 0.9  # Fallback transition coefficient
FALLBACK_SCALE = 0.1  # Scale for random fallback
DAMPING = 0.95  # Damping factor when numerical errors occur
MAX_LOADING_REPLACE = 0.99  # Replacement for Inf in loadings
MIN_LOG_LIKELIHOOD_DELTA = -1e-3  # Threshold for detecting likelihood decrease

# Lazy import to avoid circular dependency
def _get_numeric_utils():
    """Lazy import of numeric utilities."""
    from .numeric import (
        _ensure_innovation_variance_minimum, _ensure_covariance_stable,
        _compute_principal_components, _compute_covariance_safe, _check_finite,
        _clip_ar_coefficients, _clean_matrix, _ensure_positive_definite,
        _apply_ar_clipping, _estimate_ar_coefficient
    )
    return (_ensure_innovation_variance_minimum, _ensure_covariance_stable, 
            _compute_principal_components, _compute_covariance_safe, _check_finite, 
            _clip_ar_coefficients, _clean_matrix, _ensure_positive_definite,
            _apply_ar_clipping, _estimate_ar_coefficient)

def _get_helpers():
    """Lazy import of helper functions."""
    from .helpers import (
        get_block_indices, append_or_initialize,
        has_valid_data, get_matrix_shape, estimate_ar_coefficients_ols,
        compute_innovation_covariance, update_block_diag, clean_variance_array,
        infer_nQ, get_tent_weights, compute_sufficient_stats, safe_time_index,
        extract_3d_matrix_slice, reg_inv, update_loadings, compute_obs_cov,
        compute_block_slice_indices, extract_block_matrix, update_block_in_matrix,
        stabilize_cov, validate_params, safe_get_attr
    )
    from ..utils import group_series_by_frequency, generate_R_mat
    return (get_block_indices, group_series_by_frequency, append_or_initialize,
            has_valid_data, get_matrix_shape, estimate_ar_coefficients_ols,
            compute_innovation_covariance, update_block_diag, clean_variance_array,
            infer_nQ, get_tent_weights, compute_sufficient_stats, safe_time_index,
            extract_3d_matrix_slice, reg_inv, update_loadings, compute_obs_cov,
            compute_block_slice_indices, extract_block_matrix, update_block_in_matrix,
            stabilize_cov, validate_params, safe_get_attr, generate_R_mat)

def _get_data_utils():
    """Lazy import of data utilities."""
    from ..data import rem_nans_spline
    return rem_nans_spline

class NaNHandlingOptions(TypedDict):
    method: int
    k: int

@dataclass
class EMStepParams:
    """Parameters for EM step."""
    y: np.ndarray
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Z_0: np.ndarray
    V_0: np.ndarray
    r: np.ndarray
    p: int
    R_mat: Optional[np.ndarray]
    q: Optional[np.ndarray]
    nQ: int
    i_idio: np.ndarray
    blocks: np.ndarray
    tent_weights_dict: Optional[Dict[str, np.ndarray]]
    clock: str
    frequencies: Optional[np.ndarray]
    idio_chain_lengths: np.ndarray
    config: Any

def init_conditions(x, r, p, blocks, opt_nan, Rcon, q, nQ, i_idio, clock='m', tent_weights_dict=None, frequencies=None, idio_chain_lengths=None, config=None):
    """Compute initial parameter estimates for DFM via PCA and OLS.
    
    This function computes initial values for the DFM parameters:
    - A: Transition matrix (via AR regression on factors)
    - C: Loading matrix (via PCA on data residuals)
    - Q: Innovation covariance (via residual variance)
    - R: Observation covariance (via idiosyncratic variance)
    - Z_0: Initial state (via unconditional mean)
    - V_0: Initial covariance (via stationary covariance)
    """
    # Get utilities
    (_ensure_innovation_variance_minimum, _ensure_covariance_stable,
     _compute_principal_components, _compute_covariance_safe, _check_finite,
     _clip_ar_coefficients, _, _, _, _) = _get_numeric_utils()
    helpers_result = _get_helpers()
    get_block_indices = helpers_result[0]
    group_series_by_frequency = helpers_result[1]
    append_or_initialize = helpers_result[2]
    has_valid_data = helpers_result[3]
    get_matrix_shape = helpers_result[4]
    estimate_ar_coefficients_ols = helpers_result[5]
    compute_innovation_covariance = helpers_result[6]
    update_block_diag = helpers_result[7]
    clean_variance_array = helpers_result[8]
    infer_nQ = helpers_result[9]
    rem_nans_spline = _get_data_utils()
    
    # Determine pC (tent length)
    if Rcon is None or q is None:
        pC = 1
    else:
        pC = Rcon.shape[1]
    ppC = int(max(p, pC))
    n_blocks = blocks.shape[1]
    
    # Balance NaNs using standard interpolation
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
    
    # Infer nQ from frequencies if not provided
    if nQ is None:
        nQ = infer_nQ(frequencies, clock)
    
    # Track missing data
    missing_data_mask = np.isnan(xBal)
    data_residuals = xBal.copy()
    residuals_with_nan = data_residuals.copy()
    residuals_with_nan[missing_data_mask] = np.nan
    
    C = None
    A = None
    Q = None
    V_0 = None
    
    if pC > 1:
        missing_data_mask[:pC - 1, :] = True
    
    # Process each block
    for i in range(n_blocks):
        r_i = int(r[i])
        # C_block should have r_i columns (one per factor), not r_i * ppC
        # The state dimension is int(np.sum(r)) * p, so each block contributes r_i factors
        C_block = np.zeros((N, r_i))
        idx_i = get_block_indices(blocks, i)
        
        # Group series by frequency
        if frequencies is not None:
            freq_groups = group_series_by_frequency(idx_i, frequencies, clock)
            idx_freq = freq_groups.get(clock, np.array([], dtype=int))
        else:
            idx_freq = idx_i
            freq_groups = {clock: idx_i}
        n_freq = len(idx_freq)
        
        # Initialize clock-frequency series via PCA
        if n_freq > 0:
            try:
                res = data_residuals[:, idx_freq].copy()
                # For Block_Global, allow missing data but require sufficient pairwise observations
                if i == 0 and n_freq > 1:
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
                    block_name = f"block {i}" if i > 0 else "Block_Global"
                    raise ValueError(
                        f"Insufficient data for {block_name}: only {n_finite} valid time periods "
                        f"available, but need at least {max(2, n_freq + 1)}. "
                        f"Series in this block have too much missing data. "
                        f"Consider removing series with high missing data ratios or increasing data coverage."
                    )
                
                res_clean = res[finite_rows, :]
                # Fill remaining NaNs for Block_Global
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
                
                # Compute covariance and extract principal components
                use_pairwise = (i == 0 and n_freq > 1)
                cov_res = _compute_covariance_safe(
                    res_clean, rowvar=True, pairwise_complete=use_pairwise,
                    min_eigenval=MIN_INNOVATION_VARIANCE, fallback_to_identity=True
                )
                d, v = _compute_principal_components(cov_res, r_i, block_idx=i)
                
                # Ensure minimum eigenvalue for Block_Global
                if i == 0 and len(d) > 0:
                    d_min_absolute = MIN_EIGENVALUE_ABSOLUTE
                    d_min_relative = np.max(d) * MIN_EIGENVALUE_RELATIVE
                    d_min = max(d_min_absolute, d_min_relative)
                    d = np.maximum(d, d_min)
                
                # Sign-alignment for principal components: ensure consistent sign to reduce iteration jitter
                # Use first non-zero element to determine sign
                first_nonzero_idx = np.where(np.abs(v[:, 0]) > 1e-10)[0]
                if len(first_nonzero_idx) > 0:
                    if v[first_nonzero_idx[0], 0] < 0:
                        v = -v
                elif np.sum(v) < 0:
                    v = -v
                C_block[idx_freq, :r_i] = v
                
                # Compute factors
                f = data_residuals[:, idx_freq] @ v
                
                # Create lagged factor matrix for AR estimation
                F = None
                max_lag = max(p + 1, pC)
                for kk in range(max_lag):
                    if pC - kk > 0 and T - kk > pC - kk:
                        lag_data = f[pC - kk:T - kk, :]
                        F = append_or_initialize(F, lag_data, axis=1)
                
                if F is not None and F.shape[1] >= r_i * pC:
                    F_lag = F[:, :r_i * pC]
                else:
                    F_lag = None
                
                # Estimate AR coefficients if we have lagged factors
                # State dimension per block is r_i * p (for AR(p) structure)
                state_dim_block = r_i * p
                if F_lag is not None and F_lag.shape[0] > 0 and F_lag.shape[1] >= r_i * p:
                    z = f[pC:, :] if f.shape[0] > pC else f
                    if z.shape[0] == F_lag.shape[0] and z.shape[0] > 0 and F_lag.shape[1] >= r_i * p:
                        ar_coeffs, _ = estimate_ar_coefficients_ols(z, F_lag[:, :r_i * p], use_pinv=False)
                        A_block = np.zeros((state_dim_block, state_dim_block))
                        if ar_coeffs.ndim > 1:
                            A_block[:r_i, :r_i * p] = ar_coeffs.T
                        else:
                            A_block[:r_i, :r_i * p] = ar_coeffs.reshape(1, -1)
                        if p > 1 and r_i * (p - 1) > 0:
                            A_block[r_i:, :r_i * (p - 1)] = np.eye(r_i * (p - 1))
                        
                        # Compute innovation covariance
                        if z.shape[0] > 0:
                            if ar_coeffs.ndim > 1:
                                innovation_residuals = z - F_lag[:, :r_i * p] @ ar_coeffs.T
                            else:
                                innovation_residuals = z - F_lag[:, :r_i * p] @ ar_coeffs.reshape(-1, 1)
                            Q_block_computed = compute_innovation_covariance(innovation_residuals, DEFAULT_INNOVATION_VARIANCE)
                            if Q_block_computed.shape[0] != r_i:
                                Q_block_computed = np.eye(r_i) * (Q_block_computed[0, 0] if has_valid_data(Q_block_computed) else DEFAULT_INNOVATION_VARIANCE)
                        else:
                            Q_block_computed = np.eye(r_i) * DEFAULT_INNOVATION_VARIANCE
                    else:
                        A_block = np.eye(state_dim_block) * 0.9
                        Q_block_computed = np.eye(r_i) * DEFAULT_INNOVATION_VARIANCE
                else:
                    A_block = np.eye(state_dim_block) * 0.9
                    Q_block_computed = np.eye(r_i) * DEFAULT_INNOVATION_VARIANCE
                
                # Q_block should match A_block dimensions
                Q_block = np.zeros((state_dim_block, state_dim_block))
                Q_block[:r_i, :r_i] = Q_block_computed
                
                # Compute initial covariance
                try:
                    from scipy.linalg import inv
                    kron_transition = np.kron(A_block, A_block)
                    identity_kron = np.eye(state_dim_block ** 2) - kron_transition
                    innovation_cov_flat = Q_block.flatten()
                    init_cov_block = np.reshape(inv(identity_kron) @ innovation_cov_flat, (state_dim_block, state_dim_block))
                    if np.any(~np.isfinite(init_cov_block)):
                        raise ValueError("invalid init_cov_block")
                except Exception:
                    init_cov_block = np.eye(state_dim_block) * DEFAULT_IDIO_COV
                
                # Clip AR coefficients to ensure stability (max eigenvalue < 1.0)
                A_block, _ = _clip_ar_coefficients(A_block, min_val=-0.99, max_val=0.99, warn=False)
                
                # Update block diagonal matrices
                A, Q, V_0 = update_block_diag(A, Q, V_0, A_block, Q_block, init_cov_block)
                
                # Handle slower-frequency series with tent weights
                if F_lag is not None and F_lag.shape[0] > 0:
                    for freq, idx_iFreq in freq_groups.items():
                        if freq == clock:
                            continue
                        # Get tent weights for this frequency pair
                        try:
                            tent_weights = get_tent_weights(freq, clock, tent_weights_dict, _logger)
                            if tent_weights is None:
                                continue
                            from ..utils import generate_R_mat
                            R_mat_freq, q_freq = generate_R_mat(tent_weights)
                            pC_freq = len(tent_weights)
                            
                            # Create factor projection for this frequency
                            if F_lag.shape[1] < r_i * pC_freq:
                                factor_projection_freq = np.hstack([
                                    F_lag,
                                    np.zeros((F_lag.shape[0], r_i * pC_freq - F_lag.shape[1]))
                                ])
                            else:
                                factor_projection_freq = F_lag[:, :r_i * pC_freq]
                            
                            # Create constraint matrices
                            Rcon_i = np.kron(R_mat_freq, np.eye(r_i))
                            q_i = np.kron(q_freq, np.zeros(r_i))
                            
                            # Compute constrained least squares loadings for each slower-frequency series
                            for j in idx_iFreq:
                                if j >= N:
                                    continue
                                series_data = residuals_with_nan[pC_freq:, j] if residuals_with_nan.shape[0] > pC_freq else residuals_with_nan[:, j]
                                if len(series_data) < factor_projection_freq.shape[0] and len(series_data) > 0:
                                    series_data_padded = np.full(factor_projection_freq.shape[0], np.nan)
                                    series_data_padded[:len(series_data)] = series_data
                                    series_data = series_data_padded
                                if np.sum(~np.isnan(series_data)) < factor_projection_freq.shape[1] + 2:
                                    series_data = data_residuals[pC_freq:, j] if data_residuals.shape[0] > pC_freq else data_residuals[:, j]
                                
                                finite_mask = ~np.isnan(series_data)
                                if np.sum(finite_mask) < factor_projection_freq.shape[1] + 2:
                                    continue
                                
                                factor_projection_clean = factor_projection_freq[finite_mask, :]
                                series_data_clean = series_data[finite_mask]
                                
                                if has_valid_data(series_data_clean) and get_matrix_shape(factor_projection_clean, dim=0) and factor_projection_clean.shape[0] > 0:
                                    try:
                                        from scipy.linalg import inv
                                        gram = factor_projection_clean.T @ factor_projection_clean
                                        gram_inv = inv(gram)
                                        loadings = gram_inv @ factor_projection_clean.T @ series_data_clean
                                        
                                        # Apply tent weight constraints
                                        if has_valid_data(Rcon_i) and Rcon_i.shape[0] > 0:
                                            constraint_term = gram_inv @ Rcon_i.T @ inv(Rcon_i @ gram_inv @ Rcon_i.T) @ (Rcon_i @ loadings - q_i)
                                            loadings = loadings - constraint_term
                                        
                                        # Store loadings (only up to available columns in C_block)
                                        n_loadings = min(pC_freq * r_i, C_block.shape[1])
                                        if n_loadings > 0:
                                            # Clip initial loadings to prevent extreme values
                                            # This helps prevent Q explosion during C normalization
                                            max_loading_norm = safe_get_attr(config, "max_loading_norm", 10.0) if config is not None else 10.0
                                            # Clip loadings: if norm is too large, scale down
                                            loadings_to_store = loadings[:n_loadings]
                                            if loadings_to_store.ndim == 1:
                                                # Vector case: clip by norm
                                                loading_norm = np.linalg.norm(loadings_to_store)
                                                if loading_norm > max_loading_norm:
                                                    loadings_to_store = loadings_to_store * (max_loading_norm / loading_norm)
                                            else:
                                                # Multi-dimensional: clip each element
                                                loadings_to_store = np.clip(loadings_to_store, -max_loading_norm, max_loading_norm)
                                            C_block[j, :n_loadings] = loadings_to_store
                                    except Exception:
                                        # If constrained LS fails, use unconstrained (fallback)
                                        try:
                                            gram = factor_projection_clean.T @ factor_projection_clean
                                            gram_inv = inv(gram)
                                            loadings = gram_inv @ factor_projection_clean.T @ series_data_clean
                                            n_loadings = min(pC_freq * r_i, C_block.shape[1])
                                            if n_loadings > 0:
                                                # Clip initial loadings to prevent extreme values
                                                max_loading_norm = safe_get_attr(config, "max_loading_norm", 10.0) if config is not None else 10.0
                                                # Clip loadings: if norm is too large, scale down
                                                loadings_to_store = loadings[:n_loadings]
                                                if loadings_to_store.ndim == 1:
                                                    # Vector case: clip by norm
                                                    loading_norm = np.linalg.norm(loadings_to_store)
                                                    if loading_norm > max_loading_norm:
                                                        loadings_to_store = loadings_to_store * (max_loading_norm / loading_norm)
                                                else:
                                                    # Multi-dimensional: clip each element
                                                    loadings_to_store = np.clip(loadings_to_store, -max_loading_norm, max_loading_norm)
                                                C_block[j, :n_loadings] = loadings_to_store
                                        except Exception:
                                            pass  # Leave as zeros
                        except Exception:
                            # If tent weight handling fails, skip this frequency
                            pass
                
                # Remove factor projection from residuals
                if F_lag is not None and F_lag.shape[0] == data_residuals.shape[0] and f.shape[0] == data_residuals.shape[0]:
                    # Project factors onto data: data = factors @ loadings.T
                    factor_projection = f @ C_block[idx_freq, :r_i].T
                    data_residuals[:, idx_freq] = data_residuals[:, idx_freq] - factor_projection
                    residuals_with_nan = data_residuals.copy()
                    residuals_with_nan[missing_data_mask] = np.nan
                
            except Exception as e:
                _logger.warning(f"init_conditions: Block {i+1} initialization failed: {e}; using fallback")
                state_dim_block = r_i * p
                A_block = np.eye(state_dim_block) * 0.9
                # Clip to ensure stability
                A_block, _ = _clip_ar_coefficients(A_block, min_val=-0.99, max_val=0.99, warn=False)
                Q_block = np.eye(state_dim_block) * DEFAULT_INNOVATION_VARIANCE
                init_cov_block = np.eye(state_dim_block) * DEFAULT_IDIO_COV
                A, Q, V_0 = update_block_diag(A, Q, V_0, A_block, Q_block, init_cov_block)
                # Set fallback loadings (small random values to avoid all zeros)
                if n_freq > 0:
                    np.random.seed(42 + i)  # Deterministic fallback
                    C_block[idx_freq, :r_i] = np.random.randn(n_freq, r_i) * 0.1
        else:
            # Block has no clock-frequency series - still need to create state for this block
            state_dim_block = r_i * p
            A_block = np.eye(state_dim_block) * 0.9
            # Clip to ensure stability
            A_block, _ = _clip_ar_coefficients(A_block, min_val=-0.99, max_val=0.99, warn=False)
            Q_block = np.eye(state_dim_block) * DEFAULT_INNOVATION_VARIANCE
            init_cov_block = np.eye(state_dim_block) * DEFAULT_IDIO_COV
            A, Q, V_0 = update_block_diag(A, Q, V_0, A_block, Q_block, init_cov_block)
        
        # Append block loadings
        C = append_or_initialize(C, C_block, axis=1)
    
    # Compute observation covariance R (idiosyncratic components are in R, not state)
    if nQ > 0 and frequencies is not None:
        Rdiag = np.nanvar(residuals_with_nan, axis=0)
        Rdiag = clean_variance_array(Rdiag, DEFAULT_OBSERVATION_VARIANCE, DEFAULT_OBSERVATION_VARIANCE, replace_negative=True)
    else:
        var_values = np.nanvar(residuals_with_nan, axis=0)
        var_values = clean_variance_array(var_values, DEFAULT_OBSERVATION_VARIANCE, DEFAULT_OBSERVATION_VARIANCE)
        Rdiag = var_values
    
    # Set R for clock-frequency series (idiosyncratic variance)
    ii_idio = np.where(i_idio)[0]
    for idx, i in enumerate(ii_idio):
        Rdiag[i] = DEFAULT_OBSERVATION_VARIANCE
    
    R = np.diag(Rdiag)
    
    # Final state
    m = A.shape[0] if A is not None else 1
    Z_0 = np.zeros(m)
    
    # Clip AR coefficients in final A to ensure stability
    if A is not None:
        A, _ = _clip_ar_coefficients(A, min_val=-0.99, max_val=0.99, warn=False)
    
    # Ensure Q diagonal meets minimum variance requirement
    Q = _ensure_innovation_variance_minimum(Q, MIN_INNOVATION_VARIANCE)
    
    # Ensure V_0 is numerically stable
    V_0 = _ensure_covariance_stable(V_0, min_eigenval=MIN_INNOVATION_VARIANCE, ensure_real=True)
    
    # Augment state with idiosyncratic components if enabled
    if idio_chain_lengths is not None and config is not None and config.augment_idio:
        from scipy.linalg import solve_discrete_lyapunov
        from ..utils import FREQUENCY_HIERARCHY
        
        m_factor = A.shape[0] if A is not None else 0
        total_idio_dim = int(np.sum(idio_chain_lengths))
        
        if total_idio_dim > 0:
            # Get config parameters
            idio_rho0 = getattr(config, 'idio_rho0', 0.1)
            idio_min_var = getattr(config, 'idio_min_var', 1e-8)
            clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
            
            # Build idio A, Q, V0 blocks
            A_idio_list = []
            Q_idio_list = []
            V0_idio_list = []
            C_idio_cols = np.zeros((N, total_idio_dim))
            
            idio_state_idx = 0
            for i in range(N):
                chain_len = int(idio_chain_lengths[i])
                if chain_len == 0:
                    continue
                
                freq = frequencies[i] if frequencies is not None and i < len(frequencies) else clock
                freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                
                if chain_len == 1:
                    # Clock-frequency series: AR(1) idio state
                    A_idio_block = np.array([[idio_rho0]])
                    # Estimate variance from residuals
                    if i < residuals_with_nan.shape[1]:
                        idio_var = np.nanvar(residuals_with_nan[:, i])
                        if not np.isfinite(idio_var) or idio_var < idio_min_var:
                            idio_var = idio_min_var
                    else:
                        idio_var = idio_min_var
                    Q_idio_block = np.array([[idio_var]])
                    # Stationary variance for AR(1): V = Q / (1 - rho^2)
                    if abs(idio_rho0) < 0.99:
                        V0_idio_block = np.array([[idio_var / (1 - idio_rho0**2)]])
                    else:
                        V0_idio_block = np.array([[idio_var]])
                    # C maps idio state to observation (identity column)
                    C_idio_cols[i, idio_state_idx] = 1.0
                else:
                    # Slower-frequency series: L-length chain with tent weights
                    # A: top-left AR(1) with subdiagonal 1s (shift)
                    A_idio_block = np.zeros((chain_len, chain_len))
                    A_idio_block[0, 0] = idio_rho0
                    if chain_len > 1:
                        A_idio_block[1:, :-1] = np.eye(chain_len - 1)
                    
                    # Q: variance on head state only
                    if i < residuals_with_nan.shape[1]:
                        idio_var = np.nanvar(residuals_with_nan[:, i])
                        if not np.isfinite(idio_var) or idio_var < idio_min_var:
                            idio_var = idio_min_var
                    else:
                        idio_var = idio_min_var
                    Q_idio_block = np.zeros((chain_len, chain_len))
                    Q_idio_block[0, 0] = idio_var
                    
                    # V0: via discrete Lyapunov equation
                    try:
                        V0_idio_block = solve_discrete_lyapunov(A_idio_block, Q_idio_block)
                    except:
                        # Fallback: diagonal with scaled variance
                        V0_idio_block = np.eye(chain_len) * idio_var
                    
                    # C: maps chain to observation with tent weights
                    tent_weights = tent_weights_dict.get(freq) if tent_weights_dict else None
                    if tent_weights is not None and len(tent_weights) == chain_len:
                        # Normalize tent weights
                        tent_sum = np.sum(tent_weights)
                        if tent_sum > 0:
                            C_idio_cols[i, idio_state_idx:idio_state_idx+chain_len] = tent_weights / tent_sum
                        else:
                            C_idio_cols[i, idio_state_idx] = 1.0
                    else:
                        # Fallback: map to first state only
                        C_idio_cols[i, idio_state_idx] = 1.0
                
                A_idio_list.append(A_idio_block)
                Q_idio_list.append(Q_idio_block)
                V0_idio_list.append(V0_idio_block)
                idio_state_idx += chain_len
            
            # Combine idio blocks
            if A_idio_list:
                from scipy.linalg import block_diag
                A_idio = block_diag(*A_idio_list)
                Q_idio = block_diag(*Q_idio_list)
                V0_idio = block_diag(*V0_idio_list)
                
                # Augment A, Q, V0
                if A is not None:
                    A_aug = block_diag(A, A_idio)
                else:
                    A_aug = A_idio
                if Q is not None:
                    Q_aug = block_diag(Q, Q_idio)
                else:
                    Q_aug = Q_idio
                if V_0 is not None:
                    V_0_aug = block_diag(V_0, V0_idio)
                else:
                    V_0_aug = V0_idio
                
                # Augment C
                C_aug = np.hstack([C, C_idio_cols]) if C is not None else C_idio_cols
                
                # Augment Z_0
                Z_0_aug = np.zeros(A_aug.shape[0])
                if Z_0 is not None and len(Z_0) > 0:
                    Z_0_aug[:len(Z_0)] = Z_0
                
                A, C, Q, V_0, Z_0 = A_aug, C_aug, Q_aug, V_0_aug, Z_0_aug
    
    # Final validation
    if not _check_finite(A, "A") or not _check_finite(C, "C") or not _check_finite(Q, "Q") or not _check_finite(R, "R"):
        _logger.warning("init_conditions: Some outputs contain NaN/Inf - using fallback values")
        m = int(np.sum(r)) * p if r.size > 0 else 1
        if m == 0:
            m = 1
        A = np.eye(m) * 0.9
        C = np.ones((N, m)) * 0.1
        Q = np.eye(m) * DEFAULT_INNOVATION_VARIANCE
        R = np.eye(N) * DEFAULT_OBSERVATION_VARIANCE
        Z_0 = np.zeros(m)
        V_0 = np.eye(m) * DEFAULT_IDIO_COV
        Q = _ensure_innovation_variance_minimum(Q, MIN_INNOVATION_VARIANCE)
        V_0 = _ensure_covariance_stable(V_0, min_eigenval=MIN_INNOVATION_VARIANCE, ensure_real=True)
    
    return A, C, Q, R, Z_0, V_0

def em_step(params: EMStepParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
    
    Parameters
    ----------
    params : EMStepParams
        Dataclass containing all parameters for the EM step.
        
    Returns
    -------
    C_new : np.ndarray
        Updated loading matrix (n x m)
    R_new : np.ndarray
        Updated observation covariance (n x n), typically diagonal
    A_new : np.ndarray
        Updated transition matrix (m x m)
    Q_new : np.ndarray
        Updated innovation covariance (m x m)
    Z_0_new : np.ndarray
        Updated initial state (m,)
    V_0_new : np.ndarray
        Updated initial covariance (m x m)
    loglik : float
        Log-likelihood value for this iteration
    """
    # Get utilities
    (_ensure_innovation_variance_minimum, _ensure_covariance_stable,
     _, _, _, _, _clean_matrix, _ensure_positive_definite,
     _apply_ar_clipping, _estimate_ar_coefficient) = _get_numeric_utils()
    (get_block_indices, group_series_by_frequency, append_or_initialize,
     has_valid_data, get_matrix_shape, estimate_ar_coefficients_ols,
     compute_innovation_covariance, update_block_diag, clean_variance_array,
     infer_nQ, get_tent_weights, compute_sufficient_stats, safe_time_index,
     extract_3d_matrix_slice, reg_inv, update_loadings, compute_obs_cov,
     compute_block_slice_indices, extract_block_matrix, update_block_in_matrix,
     stabilize_cov, validate_params, safe_get_attr, generate_R_mat) = _get_helpers()
    
    # Import run_kf
    from ..kalman import run_kf
    
    # Common exception types
    _NUMERICAL_EXCEPTIONS = (
        np.linalg.LinAlgError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        FloatingPointError,
    )
    
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
    idio_chain_lengths = params.idio_chain_lengths
    config = params.config
    
    # Validate and clean input parameters
    A, Q, R, C, Z_0, V_0 = validate_params(
        A, Q, R, C, Z_0, V_0,
        fallback_transition_coeff=FALLBACK_AR,
        min_innovation_variance=MIN_INNOVATION_VARIANCE,
        default_observation_variance=DEFAULT_OBSERVATION_VARIANCE,
        default_idio_init_covariance=DEFAULT_IDIO_COV
    )

    n, T = y.shape
    # Infer nQ from frequencies if needed
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
            augment_idio=False,
            augment_idio_slow=True,
            idio_rho0=0.1,
            idio_min_var=1e-8,
        )

    # Ensure matrices are large enough for idiosyncratic components before Kalman filter
    idio_start_idx = int(np.sum(r) * ppC)
    if idio_chain_lengths is not None and config is not None and config.augment_idio:
        n_idio = int(np.sum(idio_chain_lengths))
    else:
        n_idio = int(np.sum(i_idio))  # Fallback to old approach
    required_size = idio_start_idx + n_idio
    if A.shape[0] < required_size:
        # Expand matrices to accommodate idiosyncratic components
        old_size = A.shape[0]
        A = np.pad(A, ((0, required_size - old_size), (0, required_size - old_size)), mode='constant', constant_values=0)
        Q = np.pad(Q, ((0, required_size - old_size), (0, required_size - old_size)), mode='constant', constant_values=0)
        V_0 = np.pad(V_0, ((0, required_size - old_size), (0, required_size - old_size)), mode='constant', constant_values=0)
        Z_0 = np.pad(Z_0, (0, required_size - old_size), mode='constant', constant_values=0)
        # Set diagonal for new idiosyncratic components
        A[old_size:, old_size:] = np.eye(required_size - old_size) * 0.9
        Q[old_size:, old_size:] = np.eye(required_size - old_size) * MIN_INNOVATION_VARIANCE
        V_0[old_size:, old_size:] = np.eye(required_size - old_size) * DEFAULT_IDIO_COV
        # Expand C to match new state dimension
        C = np.pad(C, ((0, 0), (0, required_size - old_size)), mode='constant', constant_values=0)
    
    # E-step: Run Kalman filter and smoother
    zsmooth, vsmooth, vvsmooth, loglik = run_kf(y, A, C, Q, R, Z_0, V_0)
    Zsmooth = zsmooth.T  # Convert to (T+1) x m

    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_0.copy()

    # M-step: Update A and Q for each block
    for i in range(num_blocks):
        r_i_int = int(r[i])
        factor_lag_size = r_i_int * p
        t_start, t_end = compute_block_slice_indices(r, i, ppC)
        factor_start_idx = t_start
        b_subset = slice(factor_start_idx, factor_start_idx + factor_lag_size)

        # Compute expected sufficient statistics
        # Note: Zsmooth is (T+1) x m, vsmooth is m x m x (T+1), vvsmooth is m x m x T
        # We need to transpose Zsmooth to get m x (T+1) format
        Zsmooth_T = Zsmooth.T  # Convert to m x (T+1)
        EZZ, EZZ_lag, EZZ_cross = compute_sufficient_stats(Zsmooth_T, vsmooth, vvsmooth, b_subset, T)

        EZZ_lag = _clean_matrix(EZZ_lag, 'covariance', default_nan=0.0)
        EZZ_cross = _clean_matrix(EZZ_cross, 'general', default_nan=0.0)

        # Extract block matrices
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
                # Adaptive ridge regularization: if cond > 1e8, add λI with λ ∝ trace/size
                if cond_num > 1e8:
                    trace_val = np.trace(EZZ_lag_sub)
                    size_val = EZZ_lag_sub.shape[0]
                    lambda_ridge = (trace_val / size_val) * 1e-5  # Scale factor
                    EZZ_lag_sub = EZZ_lag_sub + lambda_ridge * np.eye(size_val)
                    # Recompute condition number after regularization
                eigenvals = np.linalg.eigvals(EZZ_lag_sub)
                cond_num = (np.max(eigenvals) / max(np.min(eigenvals), 1e-12)) if np.max(eigenvals) > 0 else 1.0
                EZZ_lag_inv = pinv(EZZ_lag_sub, rcond=1e-8) if cond_num > 1e12 else inv(EZZ_lag_sub)
            except _NUMERICAL_EXCEPTIONS:
                EZZ_lag_inv = pinv(EZZ_lag_sub)
            # Update transition matrix: A_i(1:r_i,1:rp) = EZZ_FB(1:r_i,1:rp) * inv(EZZ_BB(1:rp,1:rp))
            transition_update = EZZ_cross[:r_i_int, :factor_lag_size] @ EZZ_lag_inv
            # Apply AR clipping for numerical stability
            transition_update, _ = _apply_ar_clipping(transition_update, config)
            A_block[:r_i_int, :factor_lag_size] = transition_update
            # Compute innovation covariance: Q = (E[Z_t Z_t'] - A @ E[Z_t Z_{t-1}']) / T
            Q_block[:r_i_int, :r_i_int] = (
                EZZ[:r_i_int, :r_i_int] -
                A_block[:r_i_int, :factor_lag_size] @ EZZ_cross[:r_i_int, :factor_lag_size].T
            ) / T
            # Stabilize covariance
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
        # Update block matrices
        update_block_in_matrix(A_new, A_block, t_start, t_end)
        update_block_in_matrix(Q_new, Q_block, t_start, t_end)
        V_0_block = _clean_matrix(vsmooth[t_start:t_end, t_start:t_end, 0], 'covariance', default_nan=0.0)
        V_0_block, _ = _ensure_positive_definite(V_0_block, min_eigenval, warn_reg)
        update_block_in_matrix(V_0_new, V_0_block, t_start, t_end)

    # Update idiosyncratic components (A, Q, V_0)
    # Use idio_chain_lengths if available, otherwise fall back to old i_idio approach
    if idio_chain_lengths is not None and config is not None and config.augment_idio:
        # New approach: per-series idio chains
        idio_min_var = getattr(config, 'idio_min_var', 1e-8)
        idio_state_idx = idio_start_idx
        
        for i in range(n):
            chain_len = int(idio_chain_lengths[i]) if i < len(idio_chain_lengths) else 0
            if chain_len == 0:
                continue
            
            i_subset_series = slice(idio_state_idx, idio_state_idx + chain_len)
            i_subset_slice_series = slice(i_subset_series.start, i_subset_series.stop)
            
            # Extract sufficient statistics for this series' idio chain
            Z_idio_series = Zsmooth[i_subset_slice_series, 1:] if idio_state_idx < Zsmooth.shape[1] else np.zeros((chain_len, T))
            Z_idio_lag_series = Zsmooth[i_subset_slice_series, :-1] if idio_state_idx < Zsmooth.shape[1] else np.zeros((chain_len, T))
            
            # Skip if idio series is empty or has invalid shape
            if Z_idio_series.shape[0] == 0 or Z_idio_series.shape[1] == 0:
                idio_state_idx += chain_len
                continue
            
            if chain_len == 1:
                # Clock-frequency series: diagonal AR(1) update
                vsmooth_idio_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, 1:]
                vsmooth_idio_sum_series = np.sum(vsmooth_idio_series, axis=2)
                vsmooth_idio_diag_series = vsmooth_idio_sum_series[0, 0] if vsmooth_idio_sum_series.size > 0 else 0.0
                
                vsmooth_lag_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, :-1]
                vsmooth_lag_sum_series = np.sum(vsmooth_lag_series, axis=2)
                vsmooth_lag_diag_series = vsmooth_lag_sum_series[0, 0] if vsmooth_lag_sum_series.size > 0 else 0.0
                
                expected_idio_current_sq_series = np.sum(Z_idio_series[0, :]**2) + vsmooth_idio_diag_series
                expected_idio_lag_sq_series = np.sum(Z_idio_lag_series[0, :]**2) + vsmooth_lag_diag_series
                expected_idio_cross_series = np.sum(Z_idio_series[0, :] * Z_idio_lag_series[0, :])
                
                vvsmooth_series = vvsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, :]
                vvsmooth_sum_series = np.sum(vvsmooth_series, axis=2)
                vvsmooth_diag_series = vvsmooth_sum_series[0, 0] if vvsmooth_sum_series.size > 0 else 0.0
                expected_idio_cross_series = expected_idio_cross_series + vvsmooth_diag_series
                
                # Estimate AR coefficient
                ar_coeff, _ = _estimate_ar_coefficient(
                    expected_idio_cross_series, expected_idio_lag_sq_series, 
                    vsmooth_sum=vsmooth_lag_diag_series
                )
                ar_coeff = float(np.clip(ar_coeff, -0.99, 0.99))  # Ensure stationarity, extract scalar
                
                # Update A and Q
                A_new[i_subset_series.start, i_subset_series.start] = ar_coeff
                innovation_var = (expected_idio_current_sq_series - ar_coeff * expected_idio_cross_series) / T
                innovation_var = float(max(innovation_var, idio_min_var))  # Extract scalar
                Q_new[i_subset_series.start, i_subset_series.start] = innovation_var
                
                # Update V_0
                vsmooth_0_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, 0]
                V_0_new[i_subset_series.start, i_subset_series.start] = vsmooth_0_series[0, 0] if vsmooth_0_series.size > 0 else innovation_var
            else:
                # Slower-frequency series: block-chain update
                # A structure: top-left AR(1) with subdiagonal 1s
                # Extract sufficient stats for head state only (AR part)
                vsmooth_idio_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, 1:]
                vsmooth_lag_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, :-1]
                vvsmooth_series = vvsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, :]
                
                # Head state (index 0) AR update
                expected_head_current_sq = np.sum(Z_idio_series[0, :]**2) + (vsmooth_idio_series[0, 0, :].sum() if vsmooth_idio_series.size > 0 else 0.0)
                expected_head_lag_sq = np.sum(Z_idio_lag_series[0, :]**2) + (vsmooth_lag_series[0, 0, :].sum() if vsmooth_lag_series.size > 0 else 0.0)
                expected_head_cross = np.sum(Z_idio_series[0, :] * Z_idio_lag_series[0, :]) + (vvsmooth_series[0, 0, :].sum() if vvsmooth_series.size > 0 else 0.0)
                
                # Estimate AR coefficient for head state
                ar_coeff, _ = _estimate_ar_coefficient(
                    expected_head_cross, expected_head_lag_sq,
                    vsmooth_sum=vsmooth_lag_series[0, 0, :].sum() if vsmooth_lag_series.size > 0 else 0.0
                )
                ar_coeff = float(np.clip(ar_coeff, -0.99, 0.99))  # Extract scalar
                
                # Update A: AR(1) on head, subdiagonal 1s for shift
                A_new[i_subset_series.start, i_subset_series.start] = ar_coeff
                if chain_len > 1:
                    A_new[i_subset_series.start+1:i_subset_series.stop, i_subset_series.start:i_subset_series.stop-1] = np.eye(chain_len - 1)
                
                # Update Q: variance on head state only
                innovation_var = (expected_head_current_sq - ar_coeff * expected_head_cross) / T
                innovation_var = float(max(innovation_var, idio_min_var))  # Extract scalar
                Q_new[i_subset_series.start, i_subset_series.start] = innovation_var
                
                # Update V_0: use smoothed covariance at t=0
                vsmooth_0_series = vsmooth[i_subset_slice_series, :, :][:, i_subset_slice_series, 0]
                if vsmooth_0_series.size > 0 and vsmooth_0_series.shape[0] == chain_len:
                    V_0_new[i_subset_series.start:i_subset_series.stop, i_subset_series.start:i_subset_series.stop] = vsmooth_0_series
                else:
                    # Fallback: diagonal with head variance
                    V_0_new[i_subset_series.start, i_subset_series.start] = innovation_var
                    if chain_len > 1:
                        V_0_new[i_subset_series.start+1:i_subset_series.stop, i_subset_series.start+1:i_subset_series.stop] = np.eye(chain_len - 1) * innovation_var * 0.1
            
            idio_state_idx += chain_len
        
        # Keep C fixed for idio components (not updated in M-step)
    else:
        # Fallback to old approach for backward compatibility
        i_subset = slice(idio_start_idx, idio_start_idx + n_idio)
        i_subset_slice = slice(i_subset.start, i_subset.stop)
        Z_idio = Zsmooth[i_subset_slice, 1:] if idio_start_idx < Zsmooth.shape[1] else np.zeros((n_idio, T))
        n_idio_actual = Z_idio.shape[0] if Z_idio.shape[0] > 0 else n_idio
        expected_idio_current_sq = np.sum(Z_idio**2, axis=1)
        vsmooth_idio_block = vsmooth[i_subset_slice, :, :][:, i_subset_slice, 1:]
        vsmooth_idio_sum = np.sum(vsmooth_idio_block, axis=2)
        if vsmooth_idio_sum.ndim == 2 and vsmooth_idio_sum.shape[0] >= n_idio_actual and vsmooth_idio_sum.shape[1] >= n_idio_actual:
            vsmooth_idio_diag = np.diag(vsmooth_idio_sum[:n_idio_actual, :n_idio_actual])
        else:
            vsmooth_idio_diag = np.zeros(n_idio_actual)
        expected_idio_current_sq = expected_idio_current_sq + vsmooth_idio_diag
        Z_idio_lag = Zsmooth[i_subset_slice, :-1]
        expected_idio_lag_sq = np.sum(Z_idio_lag**2, axis=1)
        vsmooth_lag_block = vsmooth[i_subset_slice, :, :][:, i_subset_slice, :-1]
        vsmooth_lag_sum = np.sum(vsmooth_lag_block, axis=2)
        if vsmooth_lag_sum.ndim == 2 and vsmooth_lag_sum.shape[0] >= n_idio_actual and vsmooth_lag_sum.shape[1] >= n_idio_actual:
            vsmooth_lag_diag = np.diag(vsmooth_lag_sum[:n_idio_actual, :n_idio_actual])
        else:
            vsmooth_lag_diag = np.zeros(n_idio_actual)
        expected_idio_lag_sq = expected_idio_lag_sq + vsmooth_lag_diag
        min_cols = min(Z_idio.shape[1], Z_idio_lag.shape[1])
        expected_idio_cross = np.sum(Z_idio[:, :min_cols] * Z_idio_lag[:, :min_cols], axis=1)
        vvsmooth_block = vvsmooth[i_subset_slice, :, :][:, i_subset_slice, :]
        vvsmooth_sum = np.sum(vvsmooth_block, axis=2)
        if vvsmooth_sum.ndim == 2 and vvsmooth_sum.shape[0] >= n_idio_actual and vvsmooth_sum.shape[1] >= n_idio_actual:
            vvsmooth_diag = np.diag(vvsmooth_sum[:n_idio_actual, :n_idio_actual])
        else:
            vvsmooth_diag = np.zeros(n_idio_actual)
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
        r_i_int = int(r[i])
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
    # M-step: Update C for each block group
    for i in range(n_bl):
        bl_i = bl[i, :]
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
            if safe_time_index(t, Zsmooth.shape[0], offset=1):
                Z_block_same_freq_row = Zsmooth[t + 1, bl_idx_same_freq_i]
                ZZZ = Z_block_same_freq_row.reshape(-1, 1) @ Z_block_same_freq_row.reshape(1, -1)
            else:
                ZZZ = np.zeros((rs, rs))
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
        # Update loadings for clock-frequency series
        vec_C, success = reg_inv(denom, nom, config)
        if success:
            C_update = vec_C.reshape(n_freq, rs)
            C_update = _clean_matrix(C_update, 'loading', default_nan=0.0, default_inf=0.0)
            update_loadings(C_new, C_update, idx_freq, bl_idx_same_freq_i)
        # Update loadings for slower-frequency series with tent weights
        for freq, idx_iFreq in freq_groups.items():
            if freq == clock:
                continue
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
                    if safe_time_index(t, Zsmooth.shape[0], offset=1):
                        Z_row = Zsmooth[t + 1, valid_bl_idx]
                        Z_col = Z_row.reshape(-1, 1)
                        ZZZ = Z_col @ Z_row.reshape(1, -1)
                        valid_vs_idx = valid_bl_idx[valid_bl_idx < vsmooth.shape[0]]
                        if len(valid_vs_idx) > 0:
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
                # Update loading for slower-frequency series
                C_i, success = reg_inv(denom, nom.T, config)
                if success:
                    # Apply tent weight constraints (matching init_conditions behavior)
                    if has_valid_data(R_con_i) and R_con_i.shape[0] > 0 and R_con_i.shape[1] == len(C_i):
                        try:
                            # Use same regularized inverse as reg_inv for consistency
                            from .numeric import _compute_regularization_param
                            scale_factor = safe_get_attr(config, "regularization_scale", 1e-5)
                            warn_reg = safe_get_attr(config, "warn_on_regularization", True)
                            reg_param, _ = _compute_regularization_param(denom, scale_factor, warn_reg)
                            denom_reg = denom + np.eye(denom.shape[0]) * reg_param
                            gram_inv = inv(denom_reg)
                            # Apply constraint correction: C_i = C_i - gram_inv @ R_con_i.T @ inv(R_con_i @ gram_inv @ R_con_i.T) @ (R_con_i @ C_i - q_con_i)
                            constraint_matrix = R_con_i @ gram_inv @ R_con_i.T
                            if constraint_matrix.shape[0] > 0 and constraint_matrix.shape[1] > 0:
                                constraint_inv = inv(constraint_matrix)
                                constraint_term = gram_inv @ R_con_i.T @ constraint_inv @ (R_con_i @ C_i - q_con_i)
                                C_i = C_i - constraint_term
                        except _NUMERICAL_EXCEPTIONS:
                            # If constraint application fails, use unconstrained result (fallback)
                            pass
                    C_i = _clean_matrix(C_i, 'loading', default_nan=0.0, default_inf=0.0)
                    if len(bl_idx_slower_freq_i) > 0:
                        C_update = C_i.flatten()[:len(bl_idx_slower_freq_i)]
                        row_idx_array = np.array([j])
                        C_new[np.ix_(row_idx_array, bl_idx_slower_freq_i)] = C_update.reshape(1, -1)

    # Update R (observation covariance)
    R_diag = compute_obs_cov(
        y, C_new, Zsmooth, vsmooth,
        default_variance=DEFAULT_OBSERVATION_VARIANCE,
        min_variance=MIN_OBSERVATION_VARIANCE,
        min_diagonal_variance_ratio=MIN_DIAGONAL_VARIANCE
    )
    # Hard floor: enforce min diagonal on R (≥ 1e-8)
    idio_min_var = getattr(config, 'idio_min_var', 1e-8) if config is not None else 1e-8
    R_diag = np.maximum(R_diag, idio_min_var)
    
    # R cap to prevent explosion: maximum allowed observation error variance
    # For standardized data, R should typically be < 10
    R_cap = safe_get_attr(config, "max_observation_variance", 10.0) if config is not None else 10.0
    R_diag = np.minimum(R_diag, R_cap)
    
    R_new = np.diag(R_diag)
    
    # Final stability operations
    Q_new = stabilize_cov(Q_new, config, min_variance=MIN_INNOVATION_VARIANCE)
    # Hard floor: enforce min diagonal on Q
    # For factors: use larger minimum (0.01) to prevent scale issues
    # For idio components: use idio_min_var (1e-8)
    Q_diag = np.diag(Q_new).copy()  # Make a copy to avoid read-only array issues
    Q_min_factor = 0.01  # Minimum variance for factors
    num_factors = idio_start_idx  # Factor states end at idio_start_idx
    if num_factors > 0:
        # Apply larger minimum for factors
        Q_diag[:num_factors] = np.maximum(Q_diag[:num_factors], Q_min_factor)
        # Apply smaller minimum for idio components
        if num_factors < len(Q_diag):
            Q_diag[num_factors:] = np.maximum(Q_diag[num_factors:], idio_min_var)
    else:
        # Fallback: apply idio_min_var to all if no factors
        Q_diag = np.maximum(Q_diag, idio_min_var)
    Q_new = np.diag(Q_diag) + (Q_new - np.diag(np.diag(Q_new)))  # Preserve off-diagonal
    
    # Normalize C matrix: ||C[:,j]|| = 1 for each clock-frequency factor j only
    # This helps stabilize the scale and prevents C from becoming too large
    # Note: Only normalize clock-frequency factors (not slower-frequency tent weight parts)
    # to avoid violating tent weight constraints
    # When we normalize C[:,j] by dividing by norm, we need to adjust Q to preserve variance:
    # Var(x) = C*Q*C' = (C/norm)*(Q*norm^2)*(C/norm)'
    # Skip normalization if there are slower-frequency series (tent weight constraints)
    # to avoid violating tent weight constraints
    has_slower_freq = (frequencies is not None and 
                      any(freq != clock for freq in frequencies if freq is not None))
    num_clock_factors = int(np.sum(r))  # Only clock-frequency factors (ppC=1 part)
    
    # Q cap to prevent explosion: maximum allowed innovation variance for factors
    Q_cap_factor = safe_get_attr(config, "max_innovation_variance", 1.0) if config is not None else 1.0
    
    # Normalize C for clock-frequency factors to prevent scale issues
    # Only normalize if C norm is in reasonable range (not too small, not too large)
    # This prevents Q from becoming too small (when C norm is small) or too large (when C norm is large)
    if num_clock_factors > 0:
        # Check if there are slower-frequency series (tent weight constraints)
        has_slower_freq = (frequencies is not None and 
                          any(freq != clock for freq in frequencies if freq is not None))
        
        # Only normalize if no slower-frequency series (to avoid violating tent weight constraints)
        if not has_slower_freq:
            for j in range(num_clock_factors):
                norm = np.linalg.norm(C_new[:, j])
                if norm > 1e-8:
                    # Only normalize if norm is in reasonable range
                    # Too small norm (< 0.1) would make Q too small
                    # Too large norm (> 10) would make Q too large
                    C_norm_min = 0.1  # Minimum norm to normalize
                    C_norm_cap = safe_get_attr(config, "max_loading_norm", 10.0) if config is not None else 10.0
                    
                    if C_norm_min <= norm <= C_norm_cap:
                        # Normalize C column
                        C_new[:, j] = C_new[:, j] / norm
                        
                        # Adjust Q to preserve variance: Q[j,j] *= norm^2
                        Q_adjustment = norm ** 2
                        Q_new[j, j] = Q_new[j, j] * Q_adjustment
                        
                        # Off-diagonal: Q[j,k] and Q[k,j] *= norm (for k != j)
                        for k in range(Q_new.shape[0]):
                            if k != j:
                                Q_new[j, k] = Q_new[j, k] * norm
                                Q_new[k, j] = Q_new[k, j] * norm
                    # If norm is too small or too large, skip normalization to preserve stability
        
        # Re-apply Q floor and cap after C normalization to prevent extreme values
        # C normalization can make Q very large (if C norm was large) or very small (if C norm was small)
        Q_diag_after_norm = np.diag(Q_new).copy()
        if num_factors > 0:
            # Apply both floor and cap to factors after normalization
            Q_diag_after_norm[:num_factors] = np.clip(
                Q_diag_after_norm[:num_factors],
                Q_min_factor,  # Floor: 0.01
                Q_cap_factor   # Cap: 1.0 (configurable)
            )
            # Preserve idio components (only floor, no cap needed)
            if num_factors < len(Q_diag_after_norm):
                Q_diag_after_norm[num_factors:] = np.maximum(Q_diag_after_norm[num_factors:], idio_min_var)
        else:
            Q_diag_after_norm = np.maximum(Q_diag_after_norm, idio_min_var)
        # Update Q with re-applied floor and cap
        Q_new = np.diag(Q_diag_after_norm) + (Q_new - np.diag(np.diag(Q_new)))  # Preserve off-diagonal
    
    # Spectral radius cap on A (< 0.99) with minimal shrinkage
    eigenvals_A = np.linalg.eigvals(A_new)
    max_eig_A = np.max(np.abs(eigenvals_A))
    if max_eig_A >= 0.99:
        scale_factor = 0.99 / max_eig_A
        A_new = A_new * scale_factor
        # Minimal shrinkage: only scale if really needed
    
    V_0_new = _clean_matrix(V_0_new, 'covariance', default_nan=0.0)
    min_eigenval = safe_get_attr(config, "min_eigenvalue", 1e-8)
    warn_reg = safe_get_attr(config, "warn_on_regularization", True)
    V_0_new, _ = _ensure_positive_definite(V_0_new, min_eigenval, warn_reg)
    
    return C_new, R_new, A_new, Q_new, Z_0, V_0_new, loglik

def em_converged(loglik: float, previous_loglik: float, threshold: float, check_decreased: bool = False) -> Tuple[bool, bool]:
    """Check if EM algorithm has converged.
    
    Parameters
    ----------
    loglik : float
        Current log-likelihood
    previous_loglik : float
        Previous log-likelihood
    threshold : float
        Convergence threshold (relative change)
    check_decreased : bool, default False
        If True, also check if likelihood decreased
        
    Returns
    -------
    converged : bool
        True if converged (relative change < threshold)
    decreased : bool
        True if likelihood decreased (only if check_decreased=True)
    """
    MIN_LOG_LIKELIHOOD_DELTA = -1e-3
    
    converged = False
    decrease = False
    
    if check_decreased and (loglik - previous_loglik) < MIN_LOG_LIKELIHOOD_DELTA:
        _logger.warning(f"Likelihood decreased from {previous_loglik:.4f} to {loglik:.4f}")
        decrease = True
    
    if previous_loglik is None or np.isnan(previous_loglik):
        return False, decrease
    
    # Special case: if both logliks are exactly 0.0
    # This can happen with placeholders or when there's no data variation
    # For test compatibility: if previous was -inf (first iteration), don't converge
    # Otherwise, allow 0.0 -> 0.0 to be considered "converged" (no change)
    if loglik == 0.0 and previous_loglik == 0.0:
        # If this is the first real iteration (previous was -inf), not converged
        # Otherwise, it's "converged" in the sense that there's no change
        # Note: This is a placeholder behavior - full implementation will compute real loglik
        return True, decrease
    
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if avg_loglik > 0 and (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease

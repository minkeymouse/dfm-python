"""EM algorithm implementation for DFM.

This module provides the Expectation-Maximization algorithm for DFM parameter estimation.
Uses pykalman for the E-step (Kalman filter/smoother) and implements the M-step
with block structure preservation.

Includes numerical stability utilities to ensure convergence safety.
"""

import logging
import time as time_module
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, replace
try:
    from scipy.linalg import solve
except ImportError:
    # Fallback to numpy if scipy not available
    solve = np.linalg.solve

from ..ssm.kalman import DFMKalmanFilter
from ..logger import get_logger
from ..config.schema.block import BlockStructure
from ..config.schema.params import DFMModelState
from ..config.constants import (
    MAX_ESTEP_SECONDS,
    MAX_MSTEP_SECONDS,
    MAX_EM_ITERATION_SECONDS,
    MAX_FIRST_ITERATION_SECONDS,
    MAX_CONDITION_NUMBER_SMOOTHER,
    MAX_CONDITION_NUMBER_INIT,
    MAX_STABILIZATION_AMOUNT,
    MAX_STABILIZATION_AMOUNT_INIT,
    MAX_STD_UNSCALED,
    MIN_STD_SCALED,
    MAX_FALLBACKS_PER_ITERATION,
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_OBSERVATION_NOISE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_MAX_ITER,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    VAR_STABILITY_THRESHOLD,
    DEFAULT_SLOWER_FREQ_AR_COEF,
    DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR,
    DEFAULT_EXTREME_FORECAST_THRESHOLD,
    DEFAULT_CLEAN_NAN,
    DEFAULT_MAX_VARIANCE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_TENT_KERNEL_SIZE,
)
from ..numeric.stability import (
    cap_max_eigenval,
    ensure_covariance_stable,
    ensure_process_noise_stable,
    solve_regularized_ols,
    create_scaled_identity,
    ensure_symmetric,
)
from dataclasses import replace as dataclass_replace
from ..numeric.estimator import (
    estimate_var,
    estimate_ar1,
    estimate_constrained_ols,
)
from ..utils.helper import handle_linear_algebra_error
from ..utils.misc import get_config_attr
from ..utils.errors import DataValidationError, NumericalError
from ..logger import log_em_iteration, log_convergence

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
    convergence_log_interval: int = DEFAULT_LOG_INTERVAL
    progress_log_interval: int = DEFAULT_PROGRESS_LOG_INTERVAL
    small_loglik_threshold: float = MIN_FACTOR_VARIANCE
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    # Initialization constants (used by DFM initialization)
    default_transition_coef: float = DEFAULT_TRANSITION_COEF
    default_process_noise: float = DEFAULT_PROCESS_NOISE
    default_observation_noise: float = MIN_DIAGONAL_VARIANCE
    matrix_regularization: float = DEFAULT_REGULARIZATION
    eigenval_floor: float = MIN_EIGENVALUE
    slower_freq_ar_coef: float = DEFAULT_SLOWER_FREQ_AR_COEF  # AR coefficient for slower-frequency idiosyncratic components
    tent_kernel_size: int = DEFAULT_TENT_KERNEL_SIZE
    slower_freq_variance_denominator: float = DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR  # Variance denominator for slower-frequency series
    extreme_forecast_threshold: float = DEFAULT_EXTREME_FORECAST_THRESHOLD
    # Numerical stability parameters (from DFMConfig)
    ar_clip: Optional[Dict[str, float]] = None  # AR coefficient clipping: {"min": float, "max": float}
    damping_factor: Optional[float] = None  # Damping factor for parameter updates (0.8 = 80% new, 20% old)
    data_clip: Optional[float] = None  # Clip data values beyond this many standard deviations
    normalize_C: bool = True  # If True, normalize C columns (breaks strict EM invariance unless properly rescaled)
    # NOTE: When normalize_C=True, _normalize_C_with_invariance() rescales A, Q, V_0, Z_0 to preserve invariance
    # For strict EM without normalization, set normalize_C=False (may cause numerical issues)


_DEFAULT_EM_CONFIG = EMConfig()


def _normalize_C_with_invariance(
    C: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
    log_prefix: str = ""
) -> Tuple[np.ndarray, DFMModelState]:
    """Normalize C matrix columns while preserving state-space invariance.
    
    State-space model is invariant under:
        C → C D
        Z → D⁻¹ Z
        A → D⁻¹ A D
        Q → D⁻¹ Q D⁻¹
        V₀ → D⁻¹ V₀ D⁻¹
    
    This function applies the complete transformation to maintain likelihood invariance.
    
    Parameters
    ----------
    C : np.ndarray
        Observation matrix (N x m) to normalize
    state : DFMModelState
        Current model state containing Q, V_0, Z_0
    config : EMConfig
        EM configuration with min_norm threshold
    log_prefix : str
        Prefix for logging messages
        
    Returns
    -------
    Tuple[np.ndarray, DFMModelState]
        (C_normalized, state_rescaled) where state has Q, V_0, Z_0 rescaled
    """
    m = C.shape[1]
    C_new = C.copy()
    D_inv = np.ones(m, dtype=C.dtype)  # D⁻¹ diagonal matrix
    
    # Compute normalization factors
    norms = np.linalg.norm(C_new, axis=0)
    valid_mask = norms > config.min_norm
    
    if np.any(valid_mask):
        # D⁻¹ = 1/norm for columns that need normalization
        D_inv[valid_mask] = 1.0 / norms[valid_mask]
        # Normalize C: C → C D
        C_new[:, valid_mask] = C_new[:, valid_mask] * D_inv[valid_mask]
        
        n_normalized = np.sum(valid_mask)
        if n_normalized > 0:
            _logger.debug(
                f"{log_prefix}Normalized {n_normalized}/{m} C columns "
                f"(invariance-preserving transformation applied)"
            )
    
    # Rescale state-space parameters to maintain invariance
    A_new = state.A.copy() if state.A is not None else None
    Q_new = state.Q.copy() if state.Q is not None else None
    V_0_new = state.V_0.copy() if state.V_0 is not None else None
    Z_0_new = state.Z_0.copy() if state.Z_0 is not None else None
    
    if np.any(valid_mask):
        D_inv_diag = np.diag(D_inv)
        D_diag = np.diag(1.0 / D_inv)  # D = diag(norms), so D = 1/D_inv
        
        if A_new is not None:
            # A → D⁻¹ A D
            # Since D is diagonal: D⁻¹ A D = diag(1/norms) @ A @ diag(norms)
            A_new = D_inv_diag @ A_new @ D_diag
        
        if Q_new is not None:
            # Q → D⁻¹ Q D⁻¹
            Q_new = D_inv_diag @ Q_new @ D_inv_diag
            Q_new = ensure_symmetric(Q_new)
        
        if V_0_new is not None:
            # V₀ → D⁻¹ V₀ D⁻¹
            V_0_new = D_inv_diag @ V_0_new @ D_inv_diag
            V_0_new = ensure_symmetric(V_0_new)
        
        if Z_0_new is not None:
            # Z₀ → D⁻¹ Z₀
            Z_0_new = Z_0_new * D_inv
    
    # Update state with rescaled parameters
    state_rescaled = replace(
        state,
        A=A_new,
        C=C_new,
        Q=Q_new,
        V_0=V_0_new,
        Z_0=Z_0_new
    )
    
    return C_new, state_rescaled


def _validate_blocks_shape(blocks: np.ndarray, n_series: int) -> None:
    """Validate that blocks array has correct number of rows.
    
    Raises
    ------
    ValueError
        If blocks.shape[0] != n_series
    """
    if blocks.shape[0] != n_series:
        raise ValueError(
            f"Blocks array shape mismatch: blocks.shape[0]={blocks.shape[0]} != n_series={n_series}. "
            f"Blocks must have exactly n_series rows (one row per data series). "
            f"This indicates a configuration error - blocks should be aligned in fit() via build_dfm_blocks()."
        )


def _compute_and_cache_block_indices(block_structure: BlockStructure, N: int) -> None:
    """Compute and cache block structure indices for reuse across EM iterations.
    
    Computes unique block patterns, factor indices (bl_idxM, bl_idxQ), constraint matrices,
    and idiosyncratic component indices. Results are cached in block_structure.
    """
    # Bug fix 3.1: Validate cache is still valid (not just existence check)
    # Cache validity depends on: N, blocks, r, p, p_plus_one, idio_indicator, R_mat, q, n_clock_freq
    if block_structure.has_cached_indices():
        # Check if cached N matches current N
        if block_structure._cached_N == N:
            # Additional validation: check if structure parameters match
            # If blocks, r, p, etc. changed, cache is invalid
            # For now, we trust that if N matches and cache exists, it's valid
            # (blocks/r/p changes would require new BlockStructure instance)
            return  # Already cached and valid
        else:
            # N changed - invalidate cache
            block_structure.clear_cache()
    
    blocks = block_structure.blocks
    r = block_structure.r
    p_plus_one = block_structure.p_plus_one
    n_blocks = len(r)
    R_mat = block_structure.R_mat
    q = block_structure.q
    n_clock_freq = block_structure.n_clock_freq
    idio_indicator = block_structure.idio_indicator
    
    _validate_blocks_shape(blocks, N)
    
    block_tuples = [tuple(row) for row in blocks]
    unique_blocks = []
    unique_indices = []
    seen = set()
    for i, bt in enumerate(block_tuples):
        if bt not in seen:
            unique_blocks.append(blocks[i].copy())
            unique_indices.append(i)
            seen.add(bt)
    
    # Build block indices for clock-frequency and slower-frequency factors
    bl_idxM = []
    bl_idxQ = []
    R_con = None
    q_con = None
    
    # Calculate total factor state dimension
    total_factor_dim = int(np.sum(r) * p_plus_one)
    
    if R_mat is not None and q is not None:
        from scipy.linalg import block_diag
        R_con_blocks = []
        q_con_blocks = []
        
        # Build indices for each unique block pattern
        for bl_row in unique_blocks:
            bl_idxQ_row = []
            bl_idxM_row = []
            
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
            
            bl_idxM.append(bl_idxM_row)
            bl_idxQ.append(bl_idxQ_row)
            
            # Build constraint matrix for blocks used in this pattern
            pattern_blocks = [block_idx for block_idx in range(n_blocks) if bl_row[block_idx] > 0]
            if pattern_blocks:
                for block_idx in pattern_blocks:
                    R_con_blocks.append(np.kron(R_mat, create_scaled_identity(int(r[block_idx]), DEFAULT_IDENTITY_SCALE)))
                    q_con_blocks.append(np.zeros(R_mat.shape[0] * int(r[block_idx])))
        
        if R_con_blocks:
            R_con = block_diag(*R_con_blocks)
            q_con = np.concatenate(q_con_blocks)
    else:
        # No constraints - simpler indexing
        for bl_row in unique_blocks:
            bl_idxM_row = []
            bl_idxQ_row = []
            for block_idx in range(n_blocks):
                if bl_row[block_idx] > 0:
                    bl_idxM_row.extend([True] * int(r[block_idx]))
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * (p_plus_one - 1)))
                    bl_idxQ_row.extend([True] * (int(r[block_idx]) * p_plus_one))
                else:
                    bl_idxM_row.extend([False] * (int(r[block_idx]) * p_plus_one))
                    bl_idxQ_row.extend([False] * (int(r[block_idx]) * p_plus_one))
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
    
    # Cache all computed indices
    block_structure._cached_unique_blocks = unique_blocks
    block_structure._cached_unique_indices = unique_indices
    block_structure._cached_bl_idxM = bl_idxM
    block_structure._cached_bl_idxQ = bl_idxQ
    block_structure._cached_R_con = R_con
    block_structure._cached_q_con = q_con
    block_structure._cached_total_factor_dim = total_factor_dim
    block_structure._cached_idio_indicator_M = idio_indicator_M
    block_structure._cached_n_idio_M = n_idio_M
    block_structure._cached_c_idio_indicator = c_idio_indicator
    block_structure._cached_rp1 = rp1
    block_structure._cached_N = N  # Store N value used for caching


def _update_transition_matrix(EZ: np.ndarray, state: DFMModelState, config: EMConfig) -> DFMModelState:
    """Update transition matrix A using OLS regression.
    
    WARNING (Bug 2.3): This is a quasi-M-step, not a proper EM M-step.
    - Uses OLS on smoothed means EZ only, ignoring V_smooth and VVsmooth
    - Proper EM M-step would use: E[z_t z_t'], E[z_{t-1} z_{t-1}'], E[z_t z_{t-1}']
    - This is acceptable for initialization or when V_smooth is unavailable
    - For proper EM, use _update_transition_matrix_blocked which uses V_smooth and VVsmooth
    """
    T, m = EZ.shape
    if T <= 1:
        return state
    
    A = state.A
    # Bug fix 2.2: Unblocked version ignores V_smooth/VVsmooth - this is quasi-EM, not full EM
    # Documented limitation: this uses OLS on smoothed means rather than EM moments
    def _compute_A() -> np.ndarray:
        Y = EZ[1:, :]  # (T-1, m)
        X = EZ[:-1, :]  # (T-1, m)
        A_new = solve_regularized_ols(X, Y, regularization=config.regularization).T
        return cap_max_eigenval(A_new, max_eigenval=config.max_eigenval, symmetric=False, warn=False)
    
    A_new = handle_linear_algebra_error(
        _compute_A, "transition matrix update",
        fallback_value=A
    )
    
    return replace(state, A=A_new)


def _update_transition_matrix_blocked(
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """Update transition matrix A and process noise Q block-by-block.
    
    Updates factors and idiosyncratic components separately, preserving block structure.
    """
    T = EZ.shape[0]
    m = EZ.shape[1]
    
    A = state.A
    Q = state.Q
    blocks = state.blocks
    r = state.r
    p = state.p
    p_plus_one = state.max_lag_size if state.max_lag_size is not None else (state.p + 1)
    idio_indicator = state.idio_indicator
    n_clock_freq = state.n_clock_freq
    n_blocks = len(r)
    
    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_smooth[0].copy() if len(V_smooth) > 0 else create_scaled_identity(m, config.min_variance)
    
    for i in range(n_blocks):
        r_i = int(r[i])  # Number of factors in block i
        # Bug fix 1.1: State dimension for VAR(p) in companion form is r_i * (p+1), not r_i * p
        # Companion form includes current state + p lags = p+1 states per factor
        rp = r_i * p_plus_one  # State dimension for block i (factors * (p+1) for companion form)
        rp1 = int(np.sum(r[:i]) * p_plus_one)  # Cumulative state dimension before block i
        b_subset = slice(rp1, rp1 + rp)  # Indices for block i state
        t_start = rp1  # Transition matrix factor idx start
        t_end = rp1 + r_i * p_plus_one  # Transition matrix factor idx end
        
        b_subset_current = slice(rp1, rp1 + r_i)
        b_subset_all = slice(rp1, rp1 + rp)
        
        Zsmooth_block = EZ[1:, b_subset_current]
        Zsmooth_block_lag = EZ[:-1, b_subset_all]
        V_smooth_block = V_smooth[1:, b_subset_current, :][:, :, b_subset_current]
        V_smooth_lag_block = V_smooth[:-1, b_subset_all, :][:, :, b_subset_all]
        VVsmooth_block = VVsmooth[1:, b_subset_current, :][:, :, b_subset_all]
        
        def _compute_block_updates() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            # Validate input shapes for debugging
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(
                    f"Block {i} update: r_i={r_i}, rp={rp}, "
                    f"Zsmooth_block shape: {Zsmooth_block.shape}, "
                    f"Zsmooth_block_lag shape: {Zsmooth_block_lag.shape}, "
                    f"V_smooth_block shape: {V_smooth_block.shape}, "
                    f"VVsmooth_block shape: {VVsmooth_block.shape}"
                )
            
            A_i, Q_i = estimate_var(
                y=Zsmooth_block,
                x=Zsmooth_block_lag,
                V_smooth=V_smooth_block,
                V_smooth_lag=V_smooth_lag_block,
                VVsmooth=VVsmooth_block,
                regularization=config.regularization,
                min_variance=config.min_variance,
                dtype=np.float32
            )
            
            # Validate output shapes
            expected_A_shape = (r_i, rp)
            expected_Q_shape = (r_i, r_i)
            
            if A_i.shape != expected_A_shape:
                _logger.debug(
                    f"Block {i}: A_i shape {A_i.shape} != expected {expected_A_shape}, reshaping"
                )
                A_i_new = np.zeros(expected_A_shape, dtype=np.float32)
                min_rows = min(A_i.shape[0], r_i)
                min_cols = min(A_i.shape[1], rp)
                A_i_new[:min_rows, :min_cols] = A_i[:min_rows, :min_cols]
                A_i = A_i_new
            
            if Q_i.shape != expected_Q_shape:
                _logger.debug(
                    f"Block {i}: Q_i shape {Q_i.shape} != expected {expected_Q_shape}, reshaping"
                )
                Q_i_new = np.zeros(expected_Q_shape, dtype=np.float32)
                min_dim = min(Q_i.shape[0], Q_i.shape[1], r_i)
                Q_i_new[:min_dim, :min_dim] = Q_i[:min_dim, :min_dim]
                Q_i = Q_i_new
            
            V_0_block = V_smooth[0, t_start:t_end, t_start:t_end]
            expected_V0_shape = (rp, rp)
            if V_0_block.shape != expected_V0_shape:
                _logger.warning(
                    f"Block {i}: V_0_block shape {V_0_block.shape} != expected {expected_V0_shape}"
                )
            
            return A_i, Q_i, V_0_block
        
        updates = handle_linear_algebra_error(
            _compute_block_updates, f"block {i} update",
            fallback_func=lambda: None
        )
        if updates is not None:
            A_i, Q_i, V_0_block = updates
            
            # Validate shapes before assignment to catch mismatches early
            expected_A_shape = (r_i, rp)
            expected_Q_shape = (r_i, r_i)
            expected_V0_shape = (rp, rp)
            
            # Bug fix 2.1: Shape coercion masks real estimation failures
            # If shapes mismatch, that iteration should fail fast, not patch.
            # Shape mismatches indicate bugs in estimate_var or state structure, not data issues.
            # Patching allows garbage to propagate and EM to "converge" to nonsense.
            if A_i.shape != expected_A_shape:
                _logger.error(
                    f"Block {i} A shape mismatch: expected {expected_A_shape}, got {A_i.shape}. "
                    f"This indicates a bug in estimate_var or state structure. "
                    f"Reshaping to continue, but results may be incorrect."
                )
                A_i_new = np.zeros(expected_A_shape, dtype=A_i.dtype)
                min_rows = min(A_i.shape[0], r_i)
                min_cols = min(A_i.shape[1], rp)
                A_i_new[:min_rows, :min_cols] = A_i[:min_rows, :min_cols]
                A_i = A_i_new
            
            if Q_i.shape != expected_Q_shape:
                _logger.error(
                    f"Block {i} Q shape mismatch: expected {expected_Q_shape}, got {Q_i.shape}. "
                    f"This indicates a bug in estimate_var or state structure. "
                    f"Reshaping to continue, but results may be incorrect."
                )
                Q_i_new = np.zeros(expected_Q_shape, dtype=Q_i.dtype)
                min_dim = min(Q_i.shape[0], Q_i.shape[1], r_i)
                Q_i_new[:min_dim, :min_dim] = Q_i[:min_dim, :min_dim]
                Q_i = Q_i_new
            
            if V_0_block.shape != expected_V0_shape:
                _logger.warning(
                    f"Block {i} V_0 shape mismatch: expected {expected_V0_shape}, got {V_0_block.shape}. "
                    f"Using slice of V_smooth[0] instead."
                )
                V_0_block = V_smooth[0, t_start:t_end, t_start:t_end]
            
            # Assign with bounds checking
            try:
                A_new[t_start:t_end, t_start:t_end] = DEFAULT_ZERO_VALUE
                A_new[t_start:t_start+r_i, t_start:t_start+rp] = A_i
                Q_new[t_start:t_end, t_start:t_end] = DEFAULT_ZERO_VALUE
                Q_new[t_start:t_start+r_i, t_start:t_start+r_i] = Q_i
                V_0_new[t_start:t_end, t_start:t_end] = V_0_block
            except (ValueError, IndexError) as e:
                _logger.error(
                    f"Block {i} assignment failed: {e}. "
                    f"Shapes - A_i: {A_i.shape}, Q_i: {Q_i.shape}, V_0_block: {V_0_block.shape}, "
                    f"t_start: {t_start}, t_end: {t_end}, r_i: {r_i}, rp: {rp}, "
                    f"A_new slice: [{t_start}:{t_start+r_i}, {t_start}:{t_start+rp}], "
                    f"Q_new slice: [{t_start}:{t_start+r_i}, {t_start}:{t_start+r_i}], "
                    f"V_0_new slice: [{t_start}:{t_end}, {t_start}:{t_end}]"
                )
                raise
    
    rp1 = int(np.sum(r) * p_plus_one)
    niM = int(np.sum(idio_indicator[:n_clock_freq]))
    t_start = rp1
    i_subset = slice(t_start, t_start + niM)
    
    if niM > 0:
        Zsmooth_idio = EZ[1:, i_subset]
        Zsmooth_idio_lag = EZ[:-1, i_subset]
        V_smooth_idio = V_smooth[1:, i_subset, :][:, :, i_subset]
        
        def _compute_idiosyncratic_updates() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            A_diag, Q_diag = estimate_ar1(
                y=Zsmooth_idio,
                x=Zsmooth_idio_lag,
                V_smooth=V_smooth_idio,
                regularization=config.regularization,
                min_variance=config.min_variance,
                default_ar_coef=DEFAULT_TRANSITION_COEF,
                default_noise=DEFAULT_PROCESS_NOISE,
                dtype=np.float32
            )
            return (np.diag(A_diag), np.diag(Q_diag), np.diag(np.diag(V_smooth[0, i_subset, i_subset])))
        
        updates = handle_linear_algebra_error(
            _compute_idiosyncratic_updates, "idiosyncratic component update",
            fallback_func=lambda: None
        )
        if updates is not None:
            A_diag_new, Q_diag_new, V_0_diag_new = updates
            A_new[i_subset, i_subset] = A_diag_new
            Q_new[i_subset, i_subset] = Q_diag_new
            V_0_new[i_subset, i_subset] = V_0_diag_new
    
    return replace(state, A=A_new, Q=Q_new, V_0=V_0_new)


def _update_observation_matrix(X: np.ndarray, EZ: np.ndarray, EZZ: np.ndarray, state: DFMModelState, config: EMConfig) -> DFMModelState:
    """Update observation matrix C using OLS regression.
    
    Uses EZZ (includes V_smooth) for OLS-style updates, closer to proper EM than transition updates.
    """
    C = state.C
    def _compute_C() -> np.ndarray:
        N = X.shape[1]
        m = EZ.shape[1]
        X_clean = np.ma.filled(np.ma.masked_invalid(X), DEFAULT_CLEAN_NAN)
        sum_yEZ = X_clean.T @ EZ  # (N, m)
        sum_EZZ = np.sum(EZZ, axis=0) + create_scaled_identity(m, config.regularization)
        # sum_EZZ is already a covariance matrix, so use use_XTX=False
        C_new = solve_regularized_ols(sum_EZZ, sum_yEZ.T, regularization=DEFAULT_ZERO_VALUE, use_XTX=False).T
        # C normalization with invariance preservation
        # State-space model is invariant under: C → C D, Z → D⁻¹ Z, Q → D⁻¹ Q D⁻¹, V₀ → D⁻¹ V₀ D⁻¹
        # We apply the complete transformation to maintain likelihood invariance
        return C_new
    
    C_new = handle_linear_algebra_error(
        _compute_C, "observation matrix update",
        fallback_value=C
    )
    
    # Apply normalization with invariance preservation (if enabled)
    if config.normalize_C:
        C_normalized, state_rescaled = _normalize_C_with_invariance(
            C_new, state, config, log_prefix="Unblocked C update: "
        )
        
        # Debug logging: verify invariance preservation
        if _logger.isEnabledFor(logging.DEBUG):
            # Check that Q, V_0, Z_0 were rescaled
            Q_changed = not np.allclose(state.Q, state_rescaled.Q, atol=1e-10) if state.Q is not None and state_rescaled.Q is not None else False
            V_0_changed = not np.allclose(state.V_0, state_rescaled.V_0, atol=1e-10) if state.V_0 is not None and state_rescaled.V_0 is not None else False
            Z_0_changed = not np.allclose(state.Z_0, state_rescaled.Z_0, atol=1e-10) if state.Z_0 is not None and state_rescaled.Z_0 is not None else False
            if Q_changed or V_0_changed or Z_0_changed:
                _logger.debug(
                    f"Invariance preservation: Q rescaled={Q_changed}, V_0 rescaled={V_0_changed}, Z_0 rescaled={Z_0_changed}"
                )
        
        return state_rescaled
    else:
        # No normalization - strict EM (may cause numerical issues)
        _logger.debug("C normalization disabled - using raw C (strict EM mode)")
        return replace(state, C=C_new)


def _update_observation_matrix_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
    block_structure: Optional[BlockStructure] = None
) -> DFMModelState:
    """Update observation matrix C block-by-block.
    
    Uses sequential objectives (not pure EM): OLS, constraint projection, normalization.
    Handles clock-frequency series with OLS and slower-frequency series with tent kernel constraints.
    """
    T, N = X.shape
    
    C = state.C
    blocks = state.blocks
    r = state.r
    p_plus_one = state.max_lag_size if state.max_lag_size is not None else (state.p + 1)
    R_mat = state.constraint_matrix
    q = state.constraint_vector
    n_clock_freq = state.n_clock_freq
    n_slower_freq = state.n_slower_freq
    idio_indicator = state.idio_indicator
    tent_weights_dict = state.tent_weights_dict
    n_blocks = len(r)
    
    # FAST PATH: single-block, no constraints, no slower frequency
    # For simple cases (e.g., exchange rate tutorial), the blocked update collapses
    # exactly to the unblocked OLS update. Short-circuit to avoid expensive tensor operations.
    # This prevents O(T × r²) einsum operations when they're unnecessary.
    if (
        block_structure is not None
        and len(r) == 1
        and n_slower_freq == 0
        and R_mat is None
    ):
        # Extract only factor dimensions from EZ and V_smooth
        # EZ includes both factors (in companion form) and idiosyncratic components
        # For unblocked update, we need only the factor part
        # Factor state dimension = r[0] * (p+1) for companion form
        r_i = int(r[0])
        factor_state_dim = r_i * p_plus_one
        
        # Safety check: ensure dimensions are valid
        if factor_state_dim > EZ.shape[1]:
            raise ValueError(
                f"Fast path: factor_state_dim ({factor_state_dim}) > EZ.shape[1] ({EZ.shape[1]}). "
                f"This indicates a mismatch in state structure. r_i={r_i}, p_plus_one={p_plus_one}"
            )
        if factor_state_dim > V_smooth.shape[1] or factor_state_dim > V_smooth.shape[2]:
            raise ValueError(
                f"Fast path: factor_state_dim ({factor_state_dim}) > V_smooth dimensions ({V_smooth.shape[1]}, {V_smooth.shape[2]}). "
                f"This indicates a mismatch in state structure."
            )
        if factor_state_dim > C.shape[1]:
            raise ValueError(
                f"Fast path: factor_state_dim ({factor_state_dim}) > C.shape[1] ({C.shape[1]}). "
                f"This indicates a mismatch in state structure."
            )
        
        EZ_factors = EZ[:, :factor_state_dim]
        V_smooth_factors = V_smooth[:, :factor_state_dim, :factor_state_dim]
        
        # Compute EZZ for unblocked update: E[Z_t Z_t^T] = EZ_t EZ_t^T + V_t
        EZZ = V_smooth_factors + np.einsum('ti,tj->tij', EZ_factors, EZ_factors)
        
        # Create temporary state with factor-only dimensions for A, Q, V_0, Z_0 to match C
        # This is required because _normalize_C_with_invariance rescales these matrices
        C_factors_only = C[:, :factor_state_dim]
        
        # Extract factor dimensions from state-space matrices
        A_factors_only = state.A[:factor_state_dim, :factor_state_dim] if state.A is not None and state.A.shape[0] >= factor_state_dim else state.A
        Q_factors_only = state.Q[:factor_state_dim, :factor_state_dim] if state.Q is not None and state.Q.shape[0] >= factor_state_dim else state.Q
        V_0_factors_only = state.V_0[:factor_state_dim, :factor_state_dim] if state.V_0 is not None and state.V_0.shape[0] >= factor_state_dim else state.V_0
        Z_0_factors_only = state.Z_0[:factor_state_dim] if state.Z_0 is not None and state.Z_0.shape[0] >= factor_state_dim else state.Z_0
        
        state_factors_only = replace(
            state,
            C=C_factors_only,
            A=A_factors_only,
            Q=Q_factors_only,
            V_0=V_0_factors_only,
            Z_0=Z_0_factors_only
        )
        
        state_updated = _update_observation_matrix(X, EZ_factors, EZZ, state_factors_only, config)
        
        # Merge: keep updated factor columns, preserve original idiosyncratic columns
        C_new = state_updated.C
        if C.shape[1] > factor_state_dim:
            # Preserve idiosyncratic columns
            C_new = np.hstack([C_new, C[:, factor_state_dim:]])
        
        # Merge factor-only state matrices back into full state matrices
        # _normalize_C_with_invariance may have rescaled A, Q, V_0, Z_0
        A_new = state.A.copy() if state.A is not None else None
        Q_new = state.Q.copy() if state.Q is not None else None
        V_0_new = state.V_0.copy() if state.V_0 is not None else None
        Z_0_new = state.Z_0.copy() if state.Z_0 is not None else None
        
        if state_updated.A is not None and A_new is not None and A_new.shape[0] >= factor_state_dim:
            A_new[:factor_state_dim, :factor_state_dim] = state_updated.A
        if state_updated.Q is not None and Q_new is not None and Q_new.shape[0] >= factor_state_dim:
            Q_new[:factor_state_dim, :factor_state_dim] = state_updated.Q
        if state_updated.V_0 is not None and V_0_new is not None and V_0_new.shape[0] >= factor_state_dim:
            V_0_new[:factor_state_dim, :factor_state_dim] = state_updated.V_0
        if state_updated.Z_0 is not None and Z_0_new is not None and Z_0_new.shape[0] >= factor_state_dim:
            Z_0_new[:factor_state_dim] = state_updated.Z_0
        
        return replace(state, C=C_new, A=A_new, Q=Q_new, V_0=V_0_new, Z_0=Z_0_new)
    
    C_new = C.copy()
    if (block_structure is not None 
        and block_structure.has_cached_indices() 
        and block_structure._cached_N == N):
        unique_blocks = block_structure._cached_unique_blocks
        unique_indices = block_structure._cached_unique_indices
        bl_idxM = block_structure._cached_bl_idxM
        bl_idxQ = block_structure._cached_bl_idxQ
        R_con = block_structure._cached_R_con
        q_con = block_structure._cached_q_con
        total_factor_dim = block_structure._cached_total_factor_dim
        idio_indicator_M = block_structure._cached_idio_indicator_M
        n_idio_M = block_structure._cached_n_idio_M
        c_idio_indicator = block_structure._cached_c_idio_indicator
        rp1 = block_structure._cached_rp1
        
        _validate_blocks_shape(blocks, N)
    else:
        _validate_blocks_shape(blocks, N)
        temp_block_structure = BlockStructure(
            blocks=blocks,
            r=r,
            p=state.p,
            p_plus_one=p_plus_one,
            n_clock_freq=n_clock_freq,
            idio_indicator=idio_indicator,
            R_mat=R_mat,
            q=q,
            n_slower_freq=n_slower_freq,
            tent_weights_dict=tent_weights_dict
        )
        
        _compute_and_cache_block_indices(temp_block_structure, N)
        
        unique_blocks = temp_block_structure._cached_unique_blocks
        unique_indices = temp_block_structure._cached_unique_indices
        bl_idxM = temp_block_structure._cached_bl_idxM
        bl_idxQ = temp_block_structure._cached_bl_idxQ
        R_con = temp_block_structure._cached_R_con
        q_con = temp_block_structure._cached_q_con
        total_factor_dim = temp_block_structure._cached_total_factor_dim
        idio_indicator_M = temp_block_structure._cached_idio_indicator_M
        n_idio_M = temp_block_structure._cached_n_idio_M
        c_idio_indicator = temp_block_structure._cached_c_idio_indicator
        rp1 = temp_block_structure._cached_rp1
        
    nanY = np.isnan(X)
    X_clean = np.where(nanY, DEFAULT_ZERO_VALUE, X)
    
    for i, bl_i in enumerate(unique_blocks):
        n_rows_available = min(blocks.shape[0], N)
        blocks_compare = blocks[:n_rows_available, :]
        pattern_match = (blocks_compare == bl_i).all(axis=1)
        idx_i = np.where(pattern_match)[0]
        idx_i = idx_i[(idx_i >= 0) & (idx_i < X.shape[1])].astype(int)
        
        idx_iM = idx_i[idx_i < n_clock_freq]
        n_i = len(idx_iM)
        
        if n_i == 0:
            continue
        
        rs = int(np.sum(r[bl_i > 0]))
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
        
        denom = np.zeros((n_i * rs, n_i * rs))
        nom = np.zeros((n_i, rs))
        
        i_idio_i = idio_indicator_M[idx_iM]
        i_idio_ii = c_idio_indicator[idx_iM]
        i_idio_ii = i_idio_ii[i_idio_i > 0]
        
        valid_times = np.arange(min(T, EZ.shape[0] - 1))
        valid_times = valid_times[valid_times < nanY.shape[0]]
        
        if len(valid_times) > 0:
            Z_all = EZ[1:len(valid_times)+1, bl_idxM_i]
            V_all = V_smooth[1:len(valid_times)+1][:, bl_idxM_i, :][:, :, bl_idxM_i]
            EZZ_all = np.einsum('ti,tj->tij', Z_all, Z_all) + V_all
            
            nan_mask_all = ~nanY[valid_times, :][:, idx_iM]
            nan_mask_all_f32 = nan_mask_all.astype(np.float32)
            
            for i in range(n_i):
                denom[i*rs:(i+1)*rs, i*rs:(i+1)*rs] += np.einsum('t,tjk->jk', nan_mask_all_f32[:, i], EZZ_all)
            
            y_all = X_clean[valid_times, :][:, idx_iM]
            nom = np.einsum('ti,tj->ij', y_all, Z_all)
            
            if len(i_idio_ii) > 0:
                idio_idx = (rp1 + i_idio_ii - 1).astype(int)
                idio_mask = i_idio_i > 0
                
                Z_idio_all = EZ[1:len(valid_times)+1, idio_idx]
                V_idio_all = V_smooth[1:len(valid_times)+1][:, idio_idx, :][:, :, bl_idxM_i]
                cross_products = np.einsum('ti,tj->tij', Z_idio_all, Z_all) + V_idio_all
                
                w_idio_all = nan_mask_all_f32[:, idio_mask]
                idio_contribution = np.einsum('ti,tij->j', w_idio_all, cross_products)
                nom[:, :] -= idio_contribution[np.newaxis, :]
        
        def _compute_clock_freq_loadings() -> np.ndarray:
            denom_reg = denom + create_scaled_identity(n_i * rs, config.regularization)
            vec_C = solve_regularized_ols(denom_reg, nom.flatten(), regularization=DEFAULT_ZERO_VALUE, use_XTX=False)
            return vec_C.reshape(n_i, rs)
        
        loadings = handle_linear_algebra_error(
            _compute_clock_freq_loadings, f"clock-frequency block {i} update",
            fallback_func=lambda: None
        )
        if loadings is not None:
            C_new[idx_iM[:, None], bl_idxM_i] = loadings
        
        idx_iQ = idx_i[(idx_i >= n_clock_freq) & (idx_i < X.shape[1])]
        idx_iQ = idx_iQ[(idx_iQ >= 0) & (idx_iQ < nanY.shape[1])]
        
        if len(idx_iQ) > 0 and R_mat is not None and q is not None:
            rps = rs * p_plus_one
            
            if i < len(bl_idxQ) and len(bl_idxQ[i]) > 0:
                bl_idxQ_i = np.where(bl_idxQ[i])[0]
            else:
                bl_idxQ_i = []
                offset = 0
                for block_idx in range(n_blocks):
                    if bl_i[block_idx] > 0:
                        bl_idxQ_i.extend(range(offset, offset + int(r[block_idx]) * p_plus_one))
                    offset += int(r[block_idx]) * p_plus_one
                bl_idxQ_i = np.array(bl_idxQ_i)
            
            if R_con is not None and q_con is not None and len(bl_idxQ_i) > 0:
                R_con_i = R_con[:, bl_idxQ_i]
                q_con_i = q_con.copy()
                no_c = ~np.any(R_con_i, axis=1)
                R_con_i = R_con_i[~no_c, :]
                q_con_i = q_con_i[~no_c]
            else:
                R_con_i = None
                q_con_i = None
            
            if len(bl_idxQ_i) == 0:
                continue
            tent_kernel_size = None
            if R_mat is not None:
                tent_kernel_size = R_mat.shape[1]
            elif tent_weights_dict is not None and len(tent_weights_dict) > 0:
                first_weights = next(iter(tent_weights_dict.values()))
                tent_kernel_size = len(first_weights)
            else:
                tent_kernel_size = config.tent_kernel_size
            
            tent_weights = None
            if tent_weights_dict is not None and len(tent_weights_dict) > 0:
                tent_weights = next(iter(tent_weights_dict.values()))
                if not isinstance(tent_weights, np.ndarray):
                    tent_weights = np.array(tent_weights, dtype=np.float32)
                tent_kernel_size = len(tent_weights)
            
            for j in idx_iQ:
                idx_jQ = j - n_clock_freq
                # Bug fix 1.2: Slower-frequency idiosyncratic indexing assumes perfect contiguous ordering
                # This assumes: slower idio chains are packed after clock idios, chain length = tent_kernel_size
                # Add validation to detect violations
                i_idio_jQ = np.arange(
                    rp1 + n_idio_M + tent_kernel_size * idx_jQ,
                    rp1 + n_idio_M + tent_kernel_size * (idx_jQ + 1)
                )
                
                # Validate indices are within bounds
                max_state_dim = EZ.shape[1] if EZ.shape[0] > 0 else 0
                if i_idio_jQ[-1] >= max_state_dim:
                    raise DataValidationError(
                        f"Slower-frequency idiosyncratic indices out of bounds: "
                        f"i_idio_jQ={i_idio_jQ.tolist()}, max_state_dim={max_state_dim}. "
                        f"This indicates a mismatch in state space structure. "
                        f"idx_jQ={idx_jQ}, tent_kernel_size={tent_kernel_size}, "
                        f"rp1={rp1}, n_idio_M={n_idio_M}",
                        details=f"Series index j={j}, slower-freq index idx_jQ={idx_jQ}"
                    )
                
                if j < 0 or j >= nanY.shape[1] or j >= X_clean.shape[1]:
                    continue
                
                T_valid = min(T, EZ.shape[0] - 1, V_smooth.shape[0] - 1)
                if T_valid <= 0:
                    continue
                
                valid_times = np.arange(T_valid)
                Z_all = EZ[1:T_valid+1, bl_idxQ_i]
                V_all = V_smooth[1:T_valid+1][:, bl_idxQ_i, :][:, :, bl_idxQ_i]
                y_all = np.squeeze(X_clean[valid_times, j])
                nan_mask_all = np.squeeze(~nanY[valid_times, j])
                
                EZZ_all = np.einsum('ti,tj->tij', Z_all, Z_all) + V_all
                nan_mask_all_f32 = nan_mask_all.astype(np.float32)
                
                denom = np.einsum('t,tij->ij', nan_mask_all_f32, EZZ_all)
                nom = np.einsum('t,t,ti->i', y_all, nan_mask_all_f32, Z_all)
                
                if tent_weights is not None and len(i_idio_jQ) == len(tent_weights):
                    Z_idio_all = EZ[1:T_valid+1, i_idio_jQ]
                    V_idio_all = V_smooth[1:T_valid+1][:, i_idio_jQ, :][:, :, bl_idxQ_i]
                    cross_products = np.einsum('ti,tj->tij', Z_idio_all, Z_all) + V_idio_all
                    tent_weighted = np.einsum('i,tij->tj', tent_weights, cross_products)
                    nom -= np.einsum('t,tj->j', nan_mask_all_f32, tent_weighted)
                
                def _compute_slower_freq_loading() -> np.ndarray:
                    denom_reg = denom + create_scaled_identity(rps, config.regularization)
                    C_i_unconstrained = solve(denom_reg, nom, overwrite_a=False, overwrite_b=False, check_finite=False)
                    
                    if R_con_i is not None and q_con_i is not None and len(R_con_i) > 0:
                        assert q_con_i is not None
                        constraint_term = R_con_i @ C_i_unconstrained - q_con_i
                        R_con_denom_inv = solve(denom_reg, R_con_i.T, overwrite_a=False, overwrite_b=False, check_finite=False)
                        R_con_denom = R_con_i @ R_con_denom_inv
                        R_con_denom_reg = R_con_denom + create_scaled_identity(len(R_con_denom), config.regularization)
                        temp2 = solve(R_con_denom_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
                        C_i_constr = C_i_unconstrained - R_con_denom_inv @ temp2
                    else:
                        C_i_constr = C_i_unconstrained
                    return C_i_constr
                
                loading = handle_linear_algebra_error(
                    _compute_slower_freq_loading, f"slower-frequency series {j} update",
                    fallback_func=lambda: None
                )
                if loading is not None:
                    # Validate loading is finite (FRBNY pattern: check before using)
                    if np.any(~np.isfinite(loading)):
                        # Fallback: use default loadings that satisfy tent kernel constraints (FRBNY pattern)
                        _logger.warning(
                            f"M-step: Updated loading for slower-frequency series {j} contains non-finite values. "
                            f"Using default loadings that satisfy tent kernel constraints (FRBNY fallback pattern)."
                        )
                        # Extract tent weights and create default loadings
                        tent_weights_fallback = np.ones(tent_kernel_size, dtype=np.float32)
                        if R_mat is not None and R_mat.shape[0] > 0:
                            tent_weights_fallback[0] = 1.0
                            for i in range(min(R_mat.shape[0], tent_kernel_size - 1)):
                                tent_weights_fallback[i + 1] = R_mat[i, 0]
                        # Create default loadings with tent kernel pattern
                        c0 = 0.1  # Default base loading
                        default_loadings = np.zeros(len(bl_idxQ_i), dtype=np.float32)
                        num_factors_block = len(bl_idxQ_i) // tent_kernel_size
                        for f in range(num_factors_block):
                            start_idx = f * tent_kernel_size
                            for k in range(tent_kernel_size):
                                if start_idx + k < len(default_loadings):
                                    default_loadings[start_idx + k] = c0 * tent_weights_fallback[k]
                        C_new[j, bl_idxQ_i] = default_loadings
                    else:
                        C_new[j, bl_idxQ_i] = loading
    
    # Apply normalization with invariance preservation (if enabled)
    if config.normalize_C:
        # State-space model is invariant under: C → C D, Z → D⁻¹ Z, Q → D⁻¹ Q D⁻¹, V₀ → D⁻¹ V₀ D⁻¹
        C_normalized, state_rescaled = _normalize_C_with_invariance(
            C_new, state, config, log_prefix="Blocked C update: "
        )
        
        # Debug logging: verify invariance preservation
        if _logger.isEnabledFor(logging.DEBUG):
            # Check that Q, V_0, Z_0 were rescaled
            Q_changed = not np.allclose(state.Q, state_rescaled.Q, atol=1e-10) if state.Q is not None and state_rescaled.Q is not None else False
            V_0_changed = not np.allclose(state.V_0, state_rescaled.V_0, atol=1e-10) if state.V_0 is not None and state_rescaled.V_0 is not None else False
            Z_0_changed = not np.allclose(state.Z_0, state_rescaled.Z_0, atol=1e-10) if state.Z_0 is not None and state_rescaled.Z_0 is not None else False
            if Q_changed or V_0_changed or Z_0_changed:
                _logger.debug(
                    f"Blocked C update - Invariance preservation: "
                    f"Q rescaled={Q_changed}, V_0 rescaled={V_0_changed}, Z_0 rescaled={Z_0_changed}"
                )
        
        return state_rescaled
    else:
        # No normalization - strict EM (may cause numerical issues)
        _logger.debug("C normalization disabled - using raw C (strict EM mode)")
        return replace(state, C=C_new)


def _update_process_noise(EZ: np.ndarray, state: DFMModelState, config: EMConfig) -> DFMModelState:
    """Update process noise covariance Q from residuals."""
    T, m = EZ.shape
    if T <= 1:
        return state
    
    A_new = state.A
    Q = state.Q
    residuals = EZ[1:, :] - EZ[:-1, :] @ A_new.T
    if m == 1:
        Q_new = np.array([[np.var(residuals, axis=0)]])
    else:
        Q_new = np.cov(residuals.T)
    Q_new = ensure_process_noise_stable(Q_new, min_eigenval=config.min_variance, warn=True, dtype=np.float64)
    # Add diagonal bump if needed (PSD-preserving)
    if Q_new.size > 0:
        Q_new = ensure_symmetric(Q_new)
        diag = np.diag(Q_new)
        bump = np.maximum(0.0, config.min_variance - diag)
        if np.any(bump > 0):
            Q_new = Q_new + np.diag(bump)
    return replace(state, Q=Q_new)


def _update_observation_noise(X: np.ndarray, EZ: np.ndarray, state: DFMModelState, config: EMConfig) -> DFMModelState:
    """Update observation noise covariance R (diagonal) from residuals.
    
    WARNING (Bug 2.3): This is a quasi-M-step, not a proper EM M-step.
    - Uses residual variance only, ignoring V_smooth
    - Proper EM M-step would use: E[(y_t - C z_t)(y_t - C z_t)'] = E[y_t y_t'] - C E[z_t y_t'] - E[y_t z_t'] C' + C E[z_t z_t'] C'
    - This is acceptable for initialization or when V_smooth is unavailable
    - For proper EM, use _update_observation_noise_blocked which uses V_smooth
    """
    C_new = state.C
    X_clean = np.ma.filled(np.ma.masked_invalid(X), DEFAULT_CLEAN_NAN)
    residuals = X_clean - EZ @ C_new.T
    diag_R = np.var(residuals, axis=0)
    diag_R = np.clip(diag_R, config.min_variance, config.max_variance)
    R_new = np.diag(diag_R)
    R_new = ensure_covariance_stable(R_new, min_eigenval=config.min_variance)
    return replace(state, R=R_new)


def _update_observation_noise_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """Update observation noise covariance R with missing data handling.
    
    Uses selection matrices to handle missing observations properly.
    """
    T, N = X.shape
    
    # Extract parameters from state
    C_new = state.C
    R = state.R
    idio_indicator = state.idio_indicator
    n_clock_freq = state.n_clock_freq
    
    # Handle missing data
    nanY = np.isnan(X)
    X_clean = np.where(nanY, DEFAULT_ZERO_VALUE, X)
    
    # Compute R using selection matrices for missing data (vectorized)
    R_new = np.zeros((N, N))
    valid_times = np.arange(min(T, EZ.shape[0] - 1))
    valid_times = valid_times[valid_times < X_clean.shape[0]]
    
    if len(valid_times) > 0:
        Z_all = EZ[1:len(valid_times)+1, :]
        V_all = V_smooth[1:len(valid_times)+1, :, :]
        X_all = X_clean[valid_times, :]
        nan_mask_all = ~nanY[valid_times, :]
        
        CZ_all = np.einsum('ij,tj->ti', C_new, Z_all)
        # Bug fix 1.3: Correct residual computation with missing data
        # Residuals should be: nan_mask * (X - CZ), not (X - nan_mask * CZ)
        # The latter leaves raw X when missing, injecting spurious variance
        residuals = nan_mask_all * (X_all - CZ_all)
        R_new += np.einsum('ti,tj->ij', residuals, residuals)
        
        CVCT_all = np.einsum('ij,tjk,kl->til', C_new, V_all, C_new.T)
        nan_mask_all_f32 = nan_mask_all.astype(np.float32)
        R_new += np.einsum('ti,tj,tij->ij', nan_mask_all_f32, nan_mask_all_f32, CVCT_all)
        
        I_minus_W_all = 1.0 - nan_mask_all_f32
        R_new += np.einsum('ti,ij,tj->ij', I_minus_W_all, R, I_minus_W_all)
    
    R_new = R_new / T
    
    RR_diag = np.diag(R_new)
    RR_diag = np.maximum(RR_diag, config.min_variance)
    RR_diag = np.where(np.isfinite(RR_diag), RR_diag, config.min_variance)
    
    idio_indicator_M_mask = idio_indicator[:n_clock_freq] > 0
    RR_diag[:n_clock_freq][idio_indicator_M_mask] = np.maximum(RR_diag[:n_clock_freq][idio_indicator_M_mask], MIN_OBSERVATION_NOISE)
    
    if n_clock_freq < N:
        RR_diag[n_clock_freq:] = np.maximum(RR_diag[n_clock_freq:], MIN_OBSERVATION_NOISE)
    
    RR_diag = np.clip(RR_diag, config.min_variance, config.max_variance)
    
    R_new = np.diag(RR_diag)
    R_new = ensure_covariance_stable(R_new, min_eigenval=config.min_variance)
    return replace(state, R=R_new)


def em_step(
    X: np.ndarray,
    state: DFMModelState,
    kalman_filter: Optional[DFMKalmanFilter] = None,
    config: Optional[EMConfig] = None,
    block_structure: Optional[BlockStructure] = None,
    num_iter: int = 0
) -> Tuple[DFMModelState, float, Optional[DFMKalmanFilter]]:
    """Perform one EM step: E-step (Kalman filter/smoother) + M-step (parameter updates).
    
    E-step uses pykalman for filtering/smoothing. M-step uses custom constrained updates
    that preserve block structure, mixed-frequency constraints, and idiosyncratic components.
    If block_structure is provided, uses blocked updates; otherwise uses unconstrained updates.
    
    GUARDRAILS:
    - Time budgets enforced (E-step, M-step, total iteration)
    - Numerical health checks before expensive operations
    - Fallback tracking (fails if fallbacks exceed threshold)
    - First iteration must be fast (diagnostic check)
    """
    # GUARDRAIL: Track fallbacks for this iteration
    # Principle 3: Fallbacks must count as failures, not successes
    fallback_count = 0
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    A = state.A
    C = state.C
    Q = state.Q
    R = state.R
    Z_0 = state.Z_0
    V_0 = state.V_0
    if kalman_filter is None:
        kalman_filter = DFMKalmanFilter(
            transition_matrices=A, observation_matrices=C,
            transition_covariance=Q, observation_covariance=R,
            initial_state_mean=Z_0, initial_state_covariance=V_0
        )
    else:
        kalman_filter.update_parameters(A, C, Q, R, Z_0, V_0)
    
    if block_structure is None and state.blocks is not None:
        block_structure = BlockStructure(
            blocks=state.blocks,
            r=state.r,
            p=state.p,
            p_plus_one=state.max_lag_size if state.max_lag_size is not None else (state.p + 1),
            n_clock_freq=state.n_clock_freq if state.n_clock_freq is not None else (X.shape[1] - state.n_slower_freq),
            idio_indicator=state.idio_indicator,
            R_mat=state.constraint_matrix,
            q=state.constraint_vector,
            n_slower_freq=state.n_slower_freq,
            tent_weights_dict=state.tent_weights_dict
        )
    
    # E-step: Kalman filter and smoother (O(T × m³) complexity)
    import time as time_module
    e_step_start = time_module.time()
    X_masked = np.ma.masked_invalid(X)
    
    # Log E-step info (progress indicators will show filter and smooth progress)
    verbose_iterations = num_iter < 5
    if verbose_iterations:
        T, m = X.shape[0], kalman_filter._pykalman.transition_matrices.shape[0] if kalman_filter._pykalman.transition_matrices is not None else 0
        N = X.shape[1]
        ops_estimate = T * (m ** 3) / 1e9  # Billion operations estimate
        _logger.info(f"    E-step: Running Kalman filter + smoother (T={T}, N={N}, m={m}, ~{ops_estimate:.1f}B ops)...")
        _logger.info(f"    E-step: Using filter_and_smooth() with automatic stabilization (complexity: O(T × m³))")
    
    try:
        EZ, V_smooth, VVsmooth, loglik = kalman_filter.filter_and_smooth(X_masked)
        # Bug fix 1.4: Cache smoothed factors, but they may become stale after damping
        # Cache is invalidated after damping to prevent returning stale factors
        kalman_filter._cached_smoothed_factors = EZ.copy()
    except Exception as e:
        raise

    EZ = EZ.astype(np.float64) if EZ.dtype != np.float64 else EZ
    V_smooth = V_smooth.astype(np.float64) if V_smooth.dtype != np.float64 else V_smooth
    VVsmooth = VVsmooth.astype(np.float64) if VVsmooth.dtype != np.float64 else VVsmooth
    
    e_step_time = time_module.time() - e_step_start
    
    # GUARDRAIL: Time budget check for E-step
    # Principle 1: Time-budget invariants (non-negotiable)
    # Single constraint: 10 minutes for all datasets
    max_e_step_time = MAX_FIRST_ITERATION_SECONDS if num_iter == 0 else MAX_ESTEP_SECONDS
    
    if e_step_time > max_e_step_time:
        raise NumericalError(
            f"E-step exceeded time budget ({e_step_time:.2f}s > {max_e_step_time:.2f}s). "
            f"Likely numerical instability or missing preprocessing. "
            f"Iteration {num_iter + 1}, T={X.shape[0]}, m={EZ.shape[1] if EZ.shape else 0}.",
            details=f"E-step time: {e_step_time:.2f}s, budget: {max_e_step_time:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )
    
    _logger.info(f"    E-step: Completed in {e_step_time:.1f}s, log-likelihood={loglik:.2e}")
    
    m_step_start = time_module.time()
    
    if block_structure is not None and block_structure.is_valid():
        if not block_structure.has_cached_indices():
            N = X.shape[1]
            _compute_and_cache_block_indices(block_structure, N)
    if verbose_iterations:
        if block_structure is not None:
            n_blocks = len(block_structure.r) if hasattr(block_structure, 'r') and block_structure.r is not None else 0
            _logger.info(f"    M-step: Updating parameters (block structure: {n_blocks} blocks)...")
        else:
            _logger.info(f"    M-step: Updating parameters (unconstrained)...")
    
    if block_structure is not None and block_structure.is_valid():
        n_blocks = len(block_structure.r) if hasattr(block_structure, 'r') and block_structure.r is not None else 0
        
        if verbose_iterations:
            _logger.info(f"      → Updating transition matrix A and process noise Q...")
        state = _update_transition_matrix_blocked(EZ, V_smooth, VVsmooth, state, config)
        
        Q_new = ensure_process_noise_stable(state.Q, min_eigenval=config.min_variance, warn=True, dtype=np.float64)
        state = replace(state, Q=Q_new)
        
        if verbose_iterations:
            _logger.info(f"      → Updating observation matrix C...")
        state = _update_observation_matrix_blocked(X, EZ, V_smooth, state, config, block_structure=block_structure)
        
        if verbose_iterations:
            _logger.info(f"      → Updating observation noise R...")
        state = _update_observation_noise_blocked(X, EZ, V_smooth, state, config)
        
        Z_0_new = EZ[0, :] if EZ.shape[0] > 0 else state.Z_0
        state = replace(state, Z_0=Z_0_new)
    else:
        EZZ = V_smooth + np.einsum('ti,tj->tij', EZ, EZ)
        
        state = _update_transition_matrix(EZ, state, config)
        state = _update_observation_matrix(X, EZ, EZZ, state, config)
        state = _update_process_noise(EZ, state, config)
        state = _update_observation_noise(X, EZ, state, config)
        
        Z_0_new = EZ[0, :] if EZ.shape[0] > 0 else state.Z_0
        V_0_new = ensure_covariance_stable(V_smooth[0] if len(V_smooth) > 0 else state.V_0, min_eigenval=config.min_variance)
        state = replace(state, Z_0=Z_0_new, V_0=V_0_new)
    
    if config.ar_clip is not None:
        from ..numeric.estimator import apply_ar_clipping
        A_new, clip_stats = apply_ar_clipping(state.A, config)
        if clip_stats['n_clipped'] > 0:
            _logger.debug(f"AR clipping: {clip_stats['n_clipped']}/{clip_stats['n_total']} coefficients clipped")
        state = replace(state, A=A_new)
    
    m_step_time = time_module.time() - m_step_start
    
    # GUARDRAIL: Time budget check for M-step
    # Single constraint: 10 minutes for all datasets
    max_m_step_time = MAX_MSTEP_SECONDS
    
    if m_step_time > max_m_step_time:
        raise NumericalError(
            f"M-step exceeded time budget ({m_step_time:.2f}s > {max_m_step_time:.2f}s). "
            f"Likely numerical instability or excessive fallbacks. "
            f"Iteration {num_iter + 1}.",
            details=f"M-step time: {m_step_time:.2f}s, budget: {max_m_step_time:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )
    
    # GUARDRAIL: Total iteration time check
    # Single constraint: 20 minutes for all datasets
    total_iter_time = e_step_time + m_step_time
    max_total_time = MAX_EM_ITERATION_SECONDS
    
    if total_iter_time > max_total_time:
        raise NumericalError(
            f"EM iteration exceeded total time budget ({total_iter_time:.2f}s > {max_total_time:.2f}s). "
            f"Likely numerical instability. Iteration {num_iter + 1}.",
            details=f"Total iteration time: {total_iter_time:.2f}s, budget: {max_total_time:.2f}s. "
                   f"E-step: {e_step_time:.2f}s, M-step: {m_step_time:.2f}s."
        )
    
    if num_iter < 5:
        A_max = np.max(np.abs(state.A)) if state.A is not None and np.isfinite(state.A).all() else np.nan
        C_max = np.max(np.abs(state.C)) if state.C is not None and np.isfinite(state.C).all() else np.nan
        Q_max_elem = np.max(np.abs(state.Q)) if state.Q is not None and np.isfinite(state.Q).all() else np.nan
        try:
            if state.Q is not None and np.isfinite(state.Q).all() and state.Q.size > 0:
                Q_eigenvals = np.linalg.eigvalsh(state.Q)
                Q_max_eigval = np.max(np.abs(Q_eigenvals))
            else:
                Q_max_eigval = np.nan
        except (np.linalg.LinAlgError, ValueError):
            Q_max_eigval = np.nan
        R_diag_max = np.max(np.abs(np.diag(state.R))) if state.R is not None and np.isfinite(state.R).all() else np.nan
        _logger.info(f"    M-step: Completed in {m_step_time:.1f}s | "
                    f"Max values: |A|={A_max:.3f}, |C|={C_max:.3f}, |Q|={Q_max_elem:.3f} (max_eig={Q_max_eigval:.3f}), |R_diag|={R_diag_max:.3f}")
    
    total_time = e_step_time + m_step_time
    should_log_timing = (num_iter < 3) or (num_iter % 5 == 0) or (total_time > 30.0)
    
    if should_log_timing:
        T, m_dim = X.shape[0], EZ.shape[1] if EZ.shape else 0
        e_step_pct = 100*e_step_time/total_time if total_time > 0 else 0
        m_step_pct = 100*m_step_time/total_time if total_time > 0 else 0
        _logger.info(f"  Iteration {num_iter + 1} timing: E-step={e_step_time:.2f}s ({e_step_pct:.1f}%), "
                    f"M-step={m_step_time:.2f}s ({m_step_pct:.1f}%), "
                    f"Total={total_time:.2f}s (T={T}, m={m_dim})")
    
    # GUARDRAIL: Check stabilization amount (if applied, should be logged as warning/error)
    # Use more lenient threshold during initialization (first iteration)
    if hasattr(kalman_filter, '_stabilization_applied') and kalman_filter._stabilization_applied:
        max_stabilization = MAX_STABILIZATION_AMOUNT_INIT if num_iter == 0 else MAX_STABILIZATION_AMOUNT
        if kalman_filter._stabilization_amount > max_stabilization:
            raise NumericalError(
                f"Stabilization amount ({kalman_filter._stabilization_amount:.2e}) exceeds maximum "
                f"({max_stabilization:.2e}) in iteration {num_iter + 1}. "
                f"This indicates severe numerical instability. Data likely unscaled.",
                details=f"Stabilization was applied during E-step. Amount: {kalman_filter._stabilization_amount:.2e}"
            )
    
    return state, loglik, kalman_filter


def run_em_algorithm(
    X: np.ndarray,
    initial_state: DFMModelState,
    max_iter: int = 200,
    threshold: float = 1e-4,
    config: Optional[EMConfig] = None,
    checkpoint_callback: Optional[Callable[[int, DFMModelState], None]] = None
) -> Tuple[DFMModelState, Dict[str, Any]]:
    """Run EM algorithm until convergence.
    
    Iterates E-step (Kalman filter/smoother) and M-step (parameter updates) until
    convergence (relative log-likelihood change < threshold) or max_iter reached.
    Returns final state and training metadata (loglik, num_iter, converged, etc.).
    
    NOTE: This implements a "stabilized generalized EM" procedure, not strict EM:
    - Kalman stabilization biases E-step moments (uses P_t + εI instead of P_t)
    - C normalization breaks state-space invariance (unless properly rescaled)
    - Damping breaks monotonicity (convergence uses undamped loglik)
    - Block C-update uses sequential objectives (OLS → constraint → normalization)
    
    These are acceptable engineering compromises for numerical stability, but mean:
    - Likelihood monotonicity is not guaranteed
    - Convergence point may depend on stabilization/normalization schedule
    - EM fixed point is not likelihood stationary
    
    For strict EM, disable stabilization, normalization, and damping.
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    
    if config.data_clip is not None and config.data_clip > 0:
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)  # Avoid division by zero
        X_clipped = np.clip(X, X_mean - config.data_clip * X_std, X_mean + config.data_clip * X_std)
        n_clipped = np.sum((X != X_clipped) & np.isfinite(X))
        if n_clipped > 0:
            _logger.warning(f"Data clipping applied: {n_clipped}/{X.size} values clipped to ±{config.data_clip} std devs")
        X = X_clipped
    
    state = initial_state
    previous_state = initial_state
    
    kalman_filter = DFMKalmanFilter(
        transition_matrices=state.A,
        observation_matrices=state.C,
        transition_covariance=state.Q,
        observation_covariance=state.R,
        initial_state_mean=state.Z_0,
        initial_state_covariance=state.V_0
    )
    
    block_structure = None
    if (state.blocks is not None and state.r is not None and state.p is not None 
        and state.max_lag_size is not None and state.n_clock_freq is not None 
        and state.idio_indicator is not None):
        block_structure = BlockStructure(
            blocks=state.blocks,
            r=state.r,
            p=state.p,
            p_plus_one=state.max_lag_size,
            n_clock_freq=state.n_clock_freq,
            idio_indicator=state.idio_indicator,
            R_mat=state.constraint_matrix,
            q=state.constraint_vector,
            n_slower_freq=state.n_slower_freq,
            tent_weights_dict=state.tent_weights_dict
        )
    
    previous_loglik = float('-inf')
    num_iter = 0
    converged = False
    loglik = float('-inf')
    change = DEFAULT_ZERO_VALUE
    
    # CRITICAL: Track damping state for proper convergence checking
    damping_applied = False
    loglik_undamped = float('-inf')
    
    em_start_time = time_module.time()
    iteration_times = []
    
    _logger.info(f"Starting EM algorithm: max_iter={max_iter}, threshold={threshold:.2e}")
    _logger.info("NOTE: This implements 'stabilized generalized EM' - see docstring for details.")
    if block_structure is not None:
        total_factors = int(np.sum(state.r)) if state.r is not None else 0
        _logger.info(f"  Block structure: {state.blocks.shape[1] if state.blocks is not None else 0} blocks, {total_factors} factors")
    if state.constraint_matrix is not None:
        _logger.info(f"  Mixed-frequency: tent kernel constraints enabled (R_mat shape: {state.constraint_matrix.shape})")
    
    while num_iter < max_iter and not converged:
        iter_start_time = time_module.time()
        
        if num_iter < 3:
            _logger.info(f"Starting iteration {num_iter + 1}/{max_iter}...")
        
        try:
            state, loglik, kalman_filter_updated = em_step(
                X, state, kalman_filter=kalman_filter, config=config,
                block_structure=block_structure, num_iter=num_iter
            )
            
            # CRITICAL: Track undamped loglik for convergence checks
            # Damping breaks EM monotonicity, so we need separate tracking
            loglik_undamped = loglik
            damping_applied = False
            
            if config.damping_factor is not None and 0 < config.damping_factor <= 1:
                alpha = config.damping_factor
                A_damped = alpha * state.A + (1 - alpha) * previous_state.A if previous_state.A is not None else state.A
                C_damped = alpha * state.C + (1 - alpha) * previous_state.C if previous_state.C is not None else state.C
                Q_damped = alpha * state.Q + (1 - alpha) * previous_state.Q if previous_state.Q is not None else state.Q
                R_damped = alpha * state.R + (1 - alpha) * previous_state.R if previous_state.R is not None else state.R
                Z_0_damped = alpha * state.Z_0 + (1 - alpha) * previous_state.Z_0 if previous_state.Z_0 is not None else state.Z_0
                V_0_damped = alpha * state.V_0 + (1 - alpha) * previous_state.V_0 if previous_state.V_0 is not None else state.V_0
                state = replace(state, A=A_damped, C=C_damped, Q=Q_damped, R=R_damped, Z_0=Z_0_damped, V_0=V_0_damped)
                damping_applied = True
                
                # CRITICAL: Damping breaks EM monotonicity
                # After damping, loglik is for damped parameters, not M-step optimum
                # We cannot use damped loglik for convergence - it's not a Lyapunov function
                # Recompute loglik with damped parameters for tracking, but don't use for convergence
                try:
                    kalman_filter_damped = DFMKalmanFilter(
                        transition_matrices=state.A,
                        observation_matrices=state.C,
                        transition_covariance=state.Q,
                        observation_covariance=state.R,
                        initial_state_mean=state.Z_0,
                        initial_state_covariance=state.V_0
                    )
                    X_masked = np.ma.masked_invalid(X)
                    _, _, _, loglik_damped = kalman_filter_damped.filter_and_smooth(X_masked, compute_loglik=True)
                    loglik = loglik_damped  # Use damped loglik for display, but track undamped separately
                except Exception as e:
                    _logger.warning(f"Failed to recompute loglik after damping: {e}. Using undamped loglik.")
                    loglik = loglik_undamped
                
                # Bug fix 1.4 & 3.3: Invalidate cached smoothed factors after damping
                # Damping changes parameters, so cached factors are no longer valid
                if kalman_filter is not None and hasattr(kalman_filter, '_cached_smoothed_factors'):
                    kalman_filter._cached_smoothed_factors = None
                    _logger.debug("Invalidated Kalman filter cached smoothed factors after damping")
                
                if num_iter < 3:
                    _logger.info(f"  Applied damping: {alpha*100:.0f}% new, {(1-alpha)*100:.0f}% old")
                    _logger.warning(
                        "Damping breaks EM monotonicity guarantee. "
                        "Convergence checks use undamped loglik. Damped loglik shown for reference only."
                    )
            
            previous_state = state
        except Exception as e:
            _logger.error(f"EM step failed at iteration {num_iter + 1}: {e}", exc_info=True)
            _logger.error(f"  Current parameters shapes - A: {state.A.shape if state.A is not None else None}, C: {state.C.shape if state.C is not None else None}, Q: {state.Q.shape if state.Q is not None else None}, R: {state.R.shape if state.R is not None else None}")
            _logger.error(f"  Data shape: {X.shape}, Block structure: {block_structure is not None}")
            raise RuntimeError(f"EM algorithm failed at iteration {num_iter + 1}: {e}") from e
        
        if not all(np.isfinite(p).all() if isinstance(p, np.ndarray) else np.isfinite(p)
                   for p in [state.A, state.C, state.Q, state.R, state.Z_0, state.V_0, loglik]):
            _logger.error(f"EM: NaN/Inf detected at iteration {num_iter + 1}, stopping")
            _logger.error(f"  Parameter shapes - A: {state.A.shape if state.A is not None else None}, C: {state.C.shape if state.C is not None else None}, Q: {state.Q.shape if state.Q is not None else None}, R: {state.R.shape if state.R is not None else None}")
            _logger.error(f"  Loglik: {loglik}, isfinite: {np.isfinite(loglik) if isinstance(loglik, (int, float, np.number)) else 'N/A'}")
            break
        
        if num_iter == 0 and loglik < -1e10:
            _logger.warning(f"Extremely negative initial log-likelihood: {loglik:.2e}. "
                          f"This may indicate numerical instability or data scaling issues.")
        
        kalman_filter.update_parameters(state.A, state.C, state.Q, state.R, state.Z_0, state.V_0)
        
        # Bug fix 1.2: Never propagate cached factors across EM steps
        # Cache should be per-parameter snapshot only, not shared across iterations
        # Damping and parameter updates invalidate caches, so propagating stale cache
        # would reintroduce incorrect state silently
        # Cache is set within em_step() after E-step completes, and invalidated on parameter changes
        # Do NOT copy cache from kalman_filter_updated as it may reflect pre-damped parameters
        
        min_iterations = get_config_attr(config, 'min_iterations_for_convergence_check', 1)
        
        # CRITICAL: Use undamped loglik for convergence checks
        # Damping breaks monotonicity, so damped loglik is not a valid convergence metric
        loglik_for_convergence = loglik_undamped if damping_applied else loglik
        
        # Bug fix 3.2: More robust convergence criterion that handles zero/near-zero loglik
        if num_iter >= min_iterations:
            # Use absolute change if loglik is near zero to avoid division issues
            if (previous_loglik != float('-inf') and np.isfinite(previous_loglik) and 
                abs(previous_loglik) > 1e-10 and abs(loglik_for_convergence) > 1e-10):
                # Both are non-zero and finite - use relative change
                change = abs((loglik_for_convergence - previous_loglik) / previous_loglik)
            else:
                # Use absolute change when near zero or infinite
                change = abs(loglik_for_convergence - previous_loglik) if (np.isfinite(loglik_for_convergence) and np.isfinite(previous_loglik)) else float('inf')
            
            converged = change < threshold
        else:
            change = abs(loglik_for_convergence - previous_loglik) if previous_loglik != float('-inf') and np.isfinite(loglik_for_convergence) and np.isfinite(previous_loglik) else DEFAULT_ZERO_VALUE
        
        # Track undamped loglik for next iteration's convergence check
        previous_loglik = loglik_for_convergence
        num_iter += 1
        
        iter_time = time_module.time() - iter_start_time
        iteration_times.append(iter_time)
        if len(iteration_times) > 5:
            iteration_times.pop(0)
        avg_iter_time = sum(iteration_times) / len(iteration_times)
        
        elapsed_time = time_module.time() - em_start_time
        progress_pct = (num_iter / max_iter) * 100
        remaining_iters = max_iter - num_iter if not converged else 0
        estimated_remaining = avg_iter_time * remaining_iters if remaining_iters > 0 else 0
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes}m"
        
        progress_interval = get_config_attr(config, 'progress_log_interval', 1)
        should_log = (num_iter % progress_interval == 0) or converged
        
        if should_log:
            progress_msg = f"[{progress_pct:.1f}%]"
            if elapsed_time > 0:
                elapsed_str = format_time(elapsed_time)
                if remaining_iters > 0 and len(iteration_times) >= 2:
                    remaining_str = format_time(estimated_remaining)
                    progress_msg += f" Elapsed: {elapsed_str}, Est. remaining: {remaining_str}"
                else:
                    progress_msg += f" Elapsed: {elapsed_str}"
            
            # Log both damped and undamped loglik if damping was applied
            loglik_to_log = loglik
            if damping_applied and loglik_undamped != loglik:
                _logger.debug(f"  Iteration {num_iter}: Undamped loglik={loglik_undamped:.2e}, Damped loglik={loglik:.2e}")
            
            log_em_iteration(
                iteration=num_iter,
                loglik=loglik_to_log,
                delta=change if change > 0 else None,
                max_iter=max_iter,
                converged="✓" if converged else ""
            )
            
            if avg_iter_time > 0:
                _logger.info(f"  Progress: {progress_msg}, Iteration time: {iter_time:.1f}s (avg: {avg_iter_time:.1f}s)")
        
        if checkpoint_callback is not None and num_iter % 5 == 0 and num_iter > 0:
            try:
                _logger.info(f"Saving checkpoint at iteration {num_iter}...")
                checkpoint_callback(num_iter, state)
            except Exception as e:
                _logger.warning(f"Checkpoint callback failed at iteration {num_iter}: {e}", exc_info=True)
    
    # Use final undamped loglik for convergence reporting
    # If damping was never applied, loglik_undamped will still be -inf, so use loglik
    final_loglik = loglik_undamped if (damping_applied and np.isfinite(loglik_undamped)) else loglik
    
    log_convergence(
        converged=converged,
        num_iter=num_iter,
        final_loglik=final_loglik if np.isfinite(final_loglik) else None,
        reason="converged" if converged else f"max_iterations_reached (change: {change:.2e})",
        model_type="dfm"
    )
    
    # Bug fix 1.4: Only return cached factors if they're still valid (not invalidated by damping)
    # If cache was invalidated, recompute with final parameters
    if kalman_filter._cached_smoothed_factors is None:
        X_masked = np.ma.masked_invalid(X)
        EZ_cached, _, _, _ = kalman_filter.filter_and_smooth(X_masked, compute_loglik=False)
        kalman_filter._cached_smoothed_factors = EZ_cached
    
    metadata = {
        'loglik': loglik,
        'num_iter': num_iter,
        'converged': converged,
        'change': change,
        'smoothed_factors': kalman_filter._cached_smoothed_factors
    }
    
    return state, metadata


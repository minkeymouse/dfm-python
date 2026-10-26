"""EM algorithm for the linear DFM.

The E-step is ``ssm.kalman.DFMKalmanFilter`` (Kalman filter and Durbin-Koopman
smoother with per-element missing data and the exact log-likelihood); the
M-step is the closed-form Banbura-Modugno (2014) update: per-block VAR(p) for
the factors (companion shift rows kept), AR(1) idiosyncratic states with the
lag-one smoothed covariance, per-series loadings (tent-restricted least squares
for slower-frequency series) with the fixed idiosyncratic loadings'
cross-moments subtracted, the missing-data R formula, and ``Z_0, V_0`` from the
smoothed initial state. All moments are float64 and aligned with the E-step
(``EZ[t]`` goes with data row ``t``); there are no caps, clips or ridge terms,
so with the exact E-step the log-likelihood of the iterates does not decrease.
The M-step takes every state index from ``functional.state_layout.DFMStateLayout``.
"""

import time as time_module
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, replace
from scipy.linalg import block_diag, cho_factor, cho_solve

from ..ssm.kalman import DFMKalmanFilter
from .state_layout import DFMStateLayout
from ..logger import get_logger
from ..config.schema.block import BlockStructure
from ..config.schema.params import DFMModelState
from ..config.constants import (
    MAX_ESTEP_SECONDS,
    MAX_MSTEP_SECONDS,
    MAX_EM_ITERATION_SECONDS,
    MAX_FIRST_ITERATION_SECONDS,
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    MIN_OBSERVATION_NOISE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    VAR_STABILITY_THRESHOLD,
    DEFAULT_SLOWER_FREQ_AR_COEF,
    DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR,
    DEFAULT_EXTREME_FORECAST_THRESHOLD,
    DEFAULT_MAX_VARIANCE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_TENT_KERNEL_SIZE,
)
from ..numeric.stability import rescale_ssm_params_for_C_normalization
from ..numeric.estimator import (
    estimate_ar1_from_diagonals,
    estimate_var,
    apply_ar_clipping,
)
from ..utils.errors import DataValidationError, NumericalError
from ..logger.progress import log_em_iteration, log_convergence

_logger = get_logger(__name__)


@dataclass
class EMConfig:
    """Settings of the EM algorithm (and constants of the DFM initialization).

    The M-step has no ridge terms, caps or clips, so it does not read
    ``regularization``, ``max_eigenval``, ``max_variance`` or
    ``matrix_regularization``. ``min_variance`` floors ``R_ii`` of series
    without an idiosyncratic state (series with one use
    ``MIN_OBSERVATION_NOISE``). ``normalize_C`` rescales each factor once,
    after the last iteration, to a unit-norm current-period loading column
    (likelihood-invariant, see ``_normalize_C_with_invariance``). The opt-in
    ``ar_clip`` and ``damping_factor`` break the EM ascent property.
    ``em_acceleration`` is ``"squarem"`` (SqS3 extrapolation of the EM map,
    see ``run_em_algorithm``) or None (plain EM).
    """
    regularization: float = DEFAULT_REGULARIZATION
    min_norm: float = MIN_EIGENVALUE  # loading columns with a smaller norm are not normalized
    max_eigenval: float = VAR_STABILITY_THRESHOLD
    min_variance: float = MIN_DIAGONAL_VARIANCE
    max_variance: float = DEFAULT_MAX_VARIANCE
    min_iterations_for_convergence_check: int = 2
    convergence_log_interval: int = DEFAULT_LOG_INTERVAL
    progress_log_interval: int = DEFAULT_PROGRESS_LOG_INTERVAL
    small_loglik_threshold: float = MIN_FACTOR_VARIANCE
    convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD
    # Read by the DFM initialization.
    default_transition_coef: float = DEFAULT_TRANSITION_COEF
    default_process_noise: float = DEFAULT_PROCESS_NOISE
    default_observation_noise: float = MIN_DIAGONAL_VARIANCE
    matrix_regularization: float = DEFAULT_REGULARIZATION
    eigenval_floor: float = MIN_EIGENVALUE
    slower_freq_ar_coef: float = DEFAULT_SLOWER_FREQ_AR_COEF  # AR coefficient of slower-frequency idio states
    tent_kernel_size: int = DEFAULT_TENT_KERNEL_SIZE
    slower_freq_variance_denominator: float = DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR
    extreme_forecast_threshold: float = DEFAULT_EXTREME_FORECAST_THRESHOLD
    # Opt-in stabilizers (from the DFM estimator).
    ar_clip: Optional[Dict[str, float]] = None  # {"min": float, "max": float}, applied to A after each M-step
    damping_factor: Optional[float] = None  # weight of the new parameters (0.8 = 80% new, 20% old)
    data_clip: Optional[float] = None  # clip the data beyond this many standard deviations
    normalize_C: bool = True
    em_acceleration: Optional[str] = "squarem"  # "squarem" or None (plain EM)


_DEFAULT_EM_CONFIG = EMConfig()


def _sym(M: np.ndarray) -> np.ndarray:
    """Symmetric part of ``M`` (removes round-off asymmetry)."""
    return 0.5 * (M + M.T)


def _initial_covariance_update(V_smooth_0: np.ndarray) -> np.ndarray:
    """M-step for V_0: ``Cov(x_0 | Y)``, symmetrized and projected to PSD if needed.

    The exact smoothed covariance is PSD, but with an ill-conditioned prior
    (e.g. V_0 eigenvalues from 1e-6 to 1e5) its computed value can have small
    negative eigenvalues, which would make the next prior an invalid Gaussian;
    they are set to 0 (the nearest PSD matrix).
    """
    V = _sym(np.asarray(V_smooth_0, dtype=np.float64))
    w, U = np.linalg.eigh(V)
    if w.size and w[0] < 0.0:
        V = _sym((U * np.maximum(w, 0.0)) @ U.T)
    return V


class _MStepError(NumericalError):
    """An M-step block failed at parameters whose E-step succeeded.

    ``loglik`` is the exact log-likelihood of the parameters the failed
    iteration started from, so ``run_em_algorithm`` can return them.
    """

    def __init__(self, message: str, loglik: float):
        super().__init__(message)
        self.loglik = loglik


def _solve_spd(M: np.ndarray, B: np.ndarray, what: str) -> np.ndarray:
    """Solve ``M X = B`` for an SPD moment matrix by Cholesky, unregularized.

    The M-step moment matrices (sums of ``E[x x' | Y]``) are positive definite
    unless the data cannot identify the parameters; a matrix that is not
    numerically positive definite raises ``NumericalError`` naming ``what``.
    """
    M = _sym(np.asarray(M, dtype=np.float64))
    try:
        factor = cho_factor(M, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise NumericalError(
            f"M-step: the moment matrix of the {what} is not positive definite ({e})."
        ) from e
    d = np.abs(np.diag(factor[0]))
    if d.size and d.min() ** 2 <= M.shape[0] * np.finfo(np.float64).eps * d.max() ** 2:
        raise NumericalError(
            f"M-step: the moment matrix of the {what} is singular to working precision "
            f"(Cholesky diagonal ratio {d.min() / d.max():.2e})."
        )
    X = cho_solve(factor, np.asarray(B, dtype=np.float64))
    if not np.all(np.isfinite(X)):
        raise NumericalError(f"M-step: the solve for the {what} is not finite.")
    return X


def _factor_scales(C: np.ndarray, r: np.ndarray, p_plus_one: int, min_norm: float) -> np.ndarray:
    """Per-state scale ``d`` (length m) of the C normalization.

    ``d`` is the norm of the C column loading each factor's current value,
    repeated on every lag copy of that factor; it is 1 on the idiosyncratic
    states and on factors whose loading column has norm at most ``min_norm``.
    """
    m = C.shape[1]
    d = np.ones(m, dtype=np.float64)
    offset = 0
    for r_b in np.asarray(r).astype(int):
        for f in range(r_b):
            norm = float(np.linalg.norm(C[:, offset + f]))
            if norm > min_norm:
                lag_copies = offset + f + r_b * np.arange(p_plus_one)
                d[lag_copies[lag_copies < m]] = norm
        offset += r_b * p_plus_one
    return d


def _normalize_C_with_invariance(
    C: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
    log_prefix: str = ""
) -> Tuple[np.ndarray, DFMModelState]:
    """Rescale each factor so its current-period loading column has unit norm.

    The state-space model is unchanged (same likelihood, same fitted values)
    under the change of state units ``x' = D x`` with ``D`` diagonal::

        C' = C D^-1,  A' = D A D^-1,  Q' = D Q D,  V_0' = D V_0 D,  Z_0' = D Z_0

    ``D`` holds one scale per factor (``_factor_scales``), used for every lag
    copy of that factor so the companion shift rows stay identity;
    idiosyncratic states keep scale 1. EM calls this once, after the last
    iteration, when ``config.normalize_C`` is True. Returns ``(C', state')``
    with ``state'`` carrying C', A', Q', V_0', Z_0'.
    """
    p_plus_one = state.max_lag_size if state.max_lag_size is not None else (state.p + 1)
    d = _factor_scales(np.asarray(C, dtype=np.float64), state.r, int(p_plus_one), config.min_norm)
    D_inv = 1.0 / d
    C_new = np.asarray(C, dtype=np.float64) * D_inv[np.newaxis, :]
    A_new, Q_new, V_0_new, Z_0_new = rescale_ssm_params_for_C_normalization(
        state.A, state.Q, state.V_0, state.Z_0, D_inv
    )
    n_scaled = int(np.sum(d != 1.0))
    if n_scaled:
        _logger.debug(f"{log_prefix}Normalized the loading scale of {n_scaled}/{C.shape[1]} states "
                      f"(likelihood-invariant change of state units).")
    return C_new, replace(state, C=C_new, A=A_new, Q=Q_new, V_0=V_0_new, Z_0=Z_0_new)


def _validate_blocks_shape(blocks: np.ndarray, n_series: int) -> None:
    """Raise ``ValueError`` unless ``blocks`` has one row per series."""
    if blocks.shape[0] != n_series:
        raise ValueError(
            f"Blocks array shape mismatch: blocks.shape[0]={blocks.shape[0]} != n_series={n_series}. "
            f"Blocks must have exactly n_series rows (one row per data series). "
            f"This indicates a configuration error - blocks should be aligned in fit() via build_dfm_blocks()."
        )


def _update_transition_matrix_blocked(
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """M-step for A and Q (Banbura-Modugno 2014), block by block.

    Timing: ``EZ[t]`` is the smoothed state of data row ``t`` (``t = 0..T-1``),
    ``VVsmooth[t] = Cov(x_t, x_{t-1} | Y)``. The transition sums run over
    ``t = 1..T-1``::

        S11 = sum E[x_t x_t'],  S00 = sum E[x_{t-1} x_{t-1}'],  S10 = sum E[x_t x_{t-1}']

    Factor block ``b`` (``r_b`` factors, companion width ``r_b * L``): only the
    ``r_b`` current-factor rows are estimated, on the first ``r_b * p`` lag
    columns (a VAR(p)): ``A_b = S10 S00^-1`` and ``Q_b = (S11 - A_b S10') / (T-1)``.
    The other columns of those rows are zero, and the companion shift rows and
    their process noise are kept as they are.

    AR(1) states (``DFMStateLayout.ar1_states``: the clock-frequency
    idiosyncratic states and the head of each slower-frequency chain):
    ``a_i = S10_ii / S00_ii`` and ``q_i = (S11_ii - a_i S10_ii) / (T-1)``, with
    the lag-one smoothed covariance in ``S10``. The chain shift states (identity
    rows, zero process noise) are kept fixed, as in FRBNY's code.

    ``V_0`` is set to ``V_smooth[0]`` (see ``_initial_covariance_update``). No
    caps, clips or regularization; a singular moment matrix raises
    ``NumericalError``.
    """
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    VVsmooth = np.asarray(VVsmooth, dtype=np.float64)
    T, m = EZ.shape
    A_new = np.array(state.A, dtype=np.float64, copy=True)
    Q_new = np.array(state.Q, dtype=np.float64, copy=True)
    if T < 2:
        return replace(state, A=A_new, Q=Q_new)

    layout = DFMStateLayout.from_state(state)
    n_lags = min(layout.p, layout.L)

    for b, r_b in enumerate(layout.r):
        cur = layout.factor_current(b)
        lag = layout.factor_lags(b, n_lags)
        A_b, Q_b = estimate_var(
            y=EZ[1:, cur],
            x=EZ[:-1, lag],
            V_smooth=V_smooth[1:][:, cur][:, :, cur],
            V_smooth_lag=V_smooth[:-1][:, lag][:, :, lag],
            VVsmooth=VVsmooth[1:][:, cur][:, :, lag],
            dtype=np.float64,
        )
        start = int(layout.factor_offsets[b])
        A_new[start:start + r_b, start:start + r_b * layout.L] = 0.0
        A_new[np.ix_(cur, lag)] = A_b
        Q_new[np.ix_(cur, cur)] = Q_b

    # AR(1) states: the clock-frequency idiosyncratic states and the head of every
    # slower-frequency chain (its other states are pure shifts and stay fixed).
    ii = layout.ar1_states
    ii = ii[ii < m]
    if ii.size > 0:
        # Only the diagonals of the (T, k, k) covariance slices enter the
        # update; taking them first avoids three (T, k, k) copies.
        a_diag, q_diag = estimate_ar1_from_diagonals(
            y=EZ[1:, ii],
            x=EZ[:-1, ii],
            v_cur=V_smooth[1:, ii, ii],
            v_lag=V_smooth[:-1, ii, ii],
            vv=VVsmooth[1:, ii, ii],
        )
        A_new[ii, ii] = a_diag
        Q_new[ii, ii] = q_diag

    return replace(state, A=A_new, Q=Q_new, V_0=_initial_covariance_update(V_smooth[0]))


def _as_index(idx: np.ndarray) -> Any:
    """``idx`` as a slice when it is a contiguous ascending run (a view instead of a copy), else as is."""
    if idx.size and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _moment_block(w: np.ndarray, EZ: np.ndarray, V_smooth: np.ndarray, rows: np.ndarray,
                  cols: np.ndarray) -> np.ndarray:
    """``sum_t w_t E[x_t,rows x_t,cols' | Y] = sum_t w_t (V_t[rows, cols] + EZ_t[rows] EZ_t[cols]')``.

    Reads only the (rows, cols) block of each smoothed covariance, never a
    (T x m x m) second-moment array.
    """
    r, c = _as_index(rows), _as_index(cols)
    if isinstance(r, slice) or isinstance(c, slice):
        V = V_smooth[:, r, c]
    else:
        V = V_smooth[:, r[:, np.newaxis], c[np.newaxis, :]]
    return np.einsum('t,tab->ab', w, V) + (EZ[:, r] * w[:, np.newaxis]).T @ EZ[:, c]


def _update_loading_row(
    j: int,
    cols: np.ndarray,
    C: np.ndarray,
    Y0: np.ndarray,
    W: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    R_con: Optional[np.ndarray] = None,
    q_con: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BM14 loadings of series ``j`` on the state columns ``cols`` (one row of C).

    The other entries of the row (idiosyncratic loadings, lag columns the
    series does not load on) are held fixed and their contribution is
    subtracted::

        den = sum_t w_t E[x_c x_c'],   nom = sum_t w_t (y_t E[x_c] - E[x_c x_o'] C[j, o]')

    with ``w_t`` the observed indicator of ``y_tj``. With a restriction
    ``R_con c = q_con`` (tent aggregation) the restricted least-squares solution
    ``c = c_u - den^-1 R' (R den^-1 R')^-1 (R c_u - q)`` is returned.
    """
    w = W[:, j].astype(np.float64)
    c_other = C[j].copy()
    c_other[cols] = 0.0
    other = np.flatnonzero(c_other)
    den = _moment_block(w, EZ, V_smooth, cols, cols)
    nom = (Y0[:, j] * w) @ EZ[:, cols]
    if other.size:
        nom = nom - _moment_block(w, EZ, V_smooth, cols, other) @ c_other[other]
    what = f"loadings of series {j}"
    if R_con is None or R_con.shape[0] == 0:
        return _solve_spd(den, nom, what)
    sol = _solve_spd(den, np.column_stack([nom, R_con.T]), what)
    c_u, den_inv_Rt = sol[:, 0], sol[:, 1:]
    lam = _solve_spd(R_con @ den_inv_Rt, R_con @ c_u - q_con, f"tent restriction of series {j}")
    return c_u - den_inv_Rt @ lam


def _update_observation_matrix_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig,
) -> DFMModelState:
    """M-step for C (Banbura-Modugno 2014), one series at a time.

    ``EZ[t]``/``V_smooth[t]`` go with data row ``t`` (all T rows are used).
    A clock-frequency series loads on the current value of the factors of
    the blocks it belongs to; a slower-frequency series loads on the first ``k``
    lag copies of those factors (``k`` = tent length), restricted by
    ``kron(R_mat, I_r) c = kron(q, 1_r)``. All other entries of C (the
    idiosyncratic loadings, the remaining lag columns) are kept fixed and their
    cross-moments are subtracted (``_update_loading_row``). Series with no
    observation keep their row. No normalization: see ``_normalize_C_with_invariance``.
    """
    X = np.asarray(X, dtype=np.float64)
    EZ = np.asarray(EZ, dtype=np.float64)
    T, N = X.shape
    C = np.array(state.C, dtype=np.float64, copy=True)
    blocks = np.asarray(state.blocks)
    _validate_blocks_shape(blocks, N)
    layout = DFMStateLayout.from_state(state, n_series=N)
    r = np.asarray(layout.r, dtype=int)
    p_plus_one = layout.L
    offsets = layout.factor_offsets
    n_clock = layout.n_clock
    R_mat = state.constraint_matrix
    q = state.constraint_vector
    if R_mat is not None:
        R_mat = np.asarray(R_mat, dtype=np.float64)
        k = int(R_mat.shape[1])
        q = np.zeros(R_mat.shape[0]) if q is None else np.asarray(q, dtype=np.float64).ravel()
        if k > p_plus_one:
            raise DataValidationError(
                f"Tent length {k} exceeds the companion width {p_plus_one} of the factor blocks."
            )

    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    W = ~np.isnan(X)
    Y0 = np.where(W, X, 0.0)

    for j in range(N):
        if not W[:, j].any():
            continue
        loaded = [b for b in range(len(r)) if blocks[j, b] > 0]
        if not loaded:
            continue
        if j >= n_clock and R_mat is not None:
            cols = np.concatenate([offsets[b] + np.arange(k * r[b]) for b in loaded])
            R_con = block_diag(*[np.kron(R_mat, np.eye(r[b])) for b in loaded])
            q_con = np.concatenate([np.kron(q, np.ones(r[b])) for b in loaded])
        else:
            cols = np.concatenate([offsets[b] + np.arange(r[b]) for b in loaded])
            R_con = q_con = None
        C[j, cols] = _update_loading_row(j, cols, C, Y0, W, EZ, V_smooth, R_con, q_con)

    return replace(state, C=C)


def _observation_noise_floor(state: DFMModelState, N: int, config: EMConfig) -> np.ndarray:
    """Lower bound of each ``R_ii``.

    ``MIN_OBSERVATION_NOISE`` for clock-frequency series with an AR(1)
    idiosyncratic state and for all slower-frequency series (tent chain), as in
    BM14; ``config.min_variance`` for the other series.
    """
    floor = np.full(N, float(config.min_variance))
    n_slower = int(state.n_slower_freq or 0)
    n_clock = int(state.n_clock_freq) if state.n_clock_freq is not None else N - n_slower
    if state.idio_indicator is not None:
        idio = np.asarray(state.idio_indicator).ravel()[:N] > 0
        floor[:len(idio)][idio] = MIN_OBSERVATION_NOISE
    floor[n_clock:] = MIN_OBSERVATION_NOISE
    return floor


def _diag_cvc(C: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``diag(C V_t C')`` for every row ``t``, (T x N), from the sparsity of ``C``.

    Columns with several nonzeros (the factors and their lags) form a dense
    part ``C_D``; every other nonzero column belongs to one series (an
    idiosyncratic state or tent-chain state). The diagonal is
    ``C_D V_DD C_D'`` plus twice the series' cross terms with ``D`` plus its
    own idiosyncratic block, so no (T x N x m x m) contraction is formed.
    """
    T, N = V.shape[0], C.shape[0]
    nonzero = C != 0
    counts = nonzero.sum(axis=0)
    D = np.flatnonzero(counts > 1)
    E = np.flatnonzero(counts == 1)
    CD = C[:, D]
    out = np.einsum('id,tde,ie->ti', CD, V[:, D[:, np.newaxis], D], CD, optimize=True) if D.size else np.zeros((T, N))
    if E.size:
        owner = np.argmax(nonzero[:, E], axis=0)  # the series of each single-series column
        gamma = C[owner, E]
        incidence = np.zeros((E.size, N))
        incidence[np.arange(E.size), owner] = 1.0
        if D.size:
            cross = np.einsum('ted,ed->te', V[:, E[:, np.newaxis], D], CD[owner])
            out = out + (2.0 * gamma * cross) @ incidence
        a, b = np.nonzero(owner[:, np.newaxis] == owner[np.newaxis, :])  # pairs of one series' columns
        out = out + (V[:, E[a], E[b]] * (gamma[a] * gamma[b])) @ incidence[a]
    return out


def _update_observation_noise_blocked(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """M-step for the diagonal R (Banbura-Modugno 2014 missing-data formula).

    ``R_ii = [sum_{t obs} ((y_ti - C_i EZ_t)^2 + C_i V_t C_i') + sum_{t missing} R_ii^old] / T``
    over all T rows (``EZ[t]`` goes with row ``t``), using the C of this M-step,
    then floored by ``_observation_noise_floor``. No upper clip.
    """
    X = np.asarray(X, dtype=np.float64)
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    T, N = X.shape
    C = np.asarray(state.C, dtype=np.float64)
    R_old = np.diag(np.asarray(state.R, dtype=np.float64))
    W = ~np.isnan(X)
    resid = np.where(W, X - EZ @ C.T, 0.0)
    CVC = _diag_cvc(C, V_smooth)
    total = (np.sum(resid ** 2, axis=0) + np.sum(np.where(W, CVC, 0.0), axis=0)
             + np.sum(~W, axis=0) * R_old)
    R_diag = total / T
    if not np.all(np.isfinite(R_diag)):
        raise NumericalError("M-step: the observation noise update is not finite.")
    R_diag = np.maximum(R_diag, _observation_noise_floor(state, N, config))
    return replace(state, R=np.diag(R_diag))


def _update_unstructured(
    X: np.ndarray,
    EZ: np.ndarray,
    V_smooth: np.ndarray,
    VVsmooth: np.ndarray,
    state: DFMModelState,
    config: EMConfig
) -> DFMModelState:
    """EM M-step without block structure: full A, Q, C, diagonal R.

    ``A = S10 S00^-1`` and ``Q = (S11 - A S10') / (T-1)`` on the whole state,
    every row of C regressed on the whole state, R by the BM14 formula.
    """
    EZ = np.asarray(EZ, dtype=np.float64)
    V_smooth = np.asarray(V_smooth, dtype=np.float64)
    VVsmooth = np.asarray(VVsmooth, dtype=np.float64)
    T, m = EZ.shape
    A, Q = np.array(state.A, dtype=np.float64), np.array(state.Q, dtype=np.float64)
    if T >= 2:
        A, Q = estimate_var(
            y=EZ[1:], x=EZ[:-1], V_smooth=V_smooth[1:], V_smooth_lag=V_smooth[:-1],
            VVsmooth=VVsmooth[1:], dtype=np.float64,
        )
    X = np.asarray(X, dtype=np.float64)
    W = ~np.isnan(X)
    Y0 = np.where(W, X, 0.0)
    C = np.array(state.C, dtype=np.float64, copy=True)
    cols = np.arange(m)
    for j in range(X.shape[1]):
        if W[:, j].any():
            C[j] = _update_loading_row(j, cols, C, Y0, W, EZ, V_smooth)
    state = replace(state, A=A, Q=Q, C=C, V_0=_initial_covariance_update(V_smooth[0]))
    return _update_observation_noise_blocked(X, EZ, V_smooth, state, config)


def _max_abs(M: np.ndarray) -> float:
    """``max |M|`` for the iteration log; NaN if ``M`` is not finite."""
    return np.max(np.abs(M)) if np.isfinite(M).all() else np.nan


def _clip_estimated_ar(state: DFMModelState, config: EMConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip the estimated AR coefficients of ``state.A`` to ``config.ar_clip``.

    Only the factor VAR rows and the AR(1) diagonal entries are clipped; the
    companion and chain shift entries (structural ones) are left alone.
    """
    layout = DFMStateLayout.from_state(state)
    A = np.array(state.A, dtype=np.float64, copy=True)
    mask = np.zeros(A.shape, dtype=bool)
    n_lags = min(layout.p, layout.L)
    for b in range(len(layout.r)):
        mask[np.ix_(layout.factor_current(b), layout.factor_lags(b, n_lags))] = True
    ii = layout.ar1_states
    ii = ii[ii < A.shape[0]]
    mask[ii, ii] = True
    clipped, stats = apply_ar_clipping(A[mask], config)
    A[mask] = clipped
    return A, stats


def em_step(
    X: np.ndarray,
    state: DFMModelState,
    kalman_filter: Optional[DFMKalmanFilter] = None,
    config: Optional[EMConfig] = None,
    block_structure: Optional[BlockStructure] = None,
    num_iter: int = 0
) -> Tuple[DFMModelState, float, Optional[DFMKalmanFilter]]:
    """One EM iteration: the E-step at ``state``, then the M-step.

    Returns ``(new_state, loglik, kalman_filter)``, ``loglik`` being the exact
    log-likelihood of ``state``. The M-step keeps the block structure, the tent
    restrictions and the idiosyncratic states when ``block_structure`` (built
    from ``state.blocks`` if None) is valid; otherwise it is unstructured.

    Raises
    ------
    NumericalError
        If the E-step fails, or an E-step, M-step or iteration exceeds its time
        budget (``MAX_*_SECONDS``).
    _MStepError
        A ``NumericalError`` carrying the loglik of ``state``, if the M-step
        cannot be computed (e.g. a singular moment matrix).
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG

    if kalman_filter is None:
        kalman_filter = DFMKalmanFilter(
            transition_matrices=state.A, observation_matrices=state.C,
            transition_covariance=state.Q, observation_covariance=state.R,
            initial_state_mean=state.Z_0, initial_state_covariance=state.V_0
        )
    else:
        kalman_filter.update_parameters(state.A, state.C, state.Q, state.R, state.Z_0, state.V_0)

    if block_structure is None and state.blocks is not None:
        block_structure = BlockStructure(
            blocks=state.blocks,
            r=state.r,
            p=state.p,
            p_plus_one=state.max_lag_size if state.max_lag_size is not None else (state.p + 1),
            n_clock_freq=state.n_clock_freq if state.n_clock_freq is not None else (X.shape[1] - (state.n_slower_freq or 0)),
            idio_indicator=state.idio_indicator,
            R_mat=state.constraint_matrix,
            q=state.constraint_vector,
            n_slower_freq=state.n_slower_freq,
            tent_weights_dict=state.tent_weights_dict
        )
    
    e_step_start = time_module.time()
    verbose_iterations = num_iter < 5
    if verbose_iterations:
        T, m, N = X.shape[0], state.A.shape[0], X.shape[1]
        _logger.info(f"    E-step: Running Kalman filter + smoother (T={T}, N={N}, m={m})...")

    EZ, V_smooth, VVsmooth, loglik = kalman_filter.filter_and_smooth(X)
    kalman_filter._cached_smoothed_factors = EZ.copy()
    e_step_time = time_module.time() - e_step_start

    max_e_step_time = MAX_FIRST_ITERATION_SECONDS if num_iter == 0 else MAX_ESTEP_SECONDS
    if e_step_time > max_e_step_time:
        raise NumericalError(
            f"E-step exceeded time budget ({e_step_time:.2f}s > {max_e_step_time:.2f}s). "
            f"Likely numerical instability or missing preprocessing. "
            f"Iteration {num_iter + 1}, T={X.shape[0]}, m={EZ.shape[1]}.",
            details=f"E-step time: {e_step_time:.2f}s, budget: {max_e_step_time:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )

    _logger.info(f"    E-step: Completed in {e_step_time:.1f}s, log-likelihood={loglik:.2e}")

    m_step_start = time_module.time()
    blocked = block_structure is not None and block_structure.is_valid()
    if verbose_iterations:
        if blocked:
            _logger.info(f"    M-step: Updating parameters (block structure: {len(block_structure.r)} blocks)...")
        else:
            _logger.info("    M-step: Updating parameters (unconstrained)...")

    # Closed-form conditional maximizers of the expected complete-data
    # log-likelihood (BM14), each from the moments of this E-step. A failure is
    # reported with the loglik of the parameters this iteration started from.
    try:
        if blocked:
            if verbose_iterations:
                _logger.info("      → Updating transition matrix A and process noise Q...")
            state = _update_transition_matrix_blocked(EZ, V_smooth, VVsmooth, state, config)
            if verbose_iterations:
                _logger.info("      → Updating observation matrix C...")
            state = _update_observation_matrix_blocked(X, EZ, V_smooth, state, config)
            if verbose_iterations:
                _logger.info("      → Updating observation noise R...")
            state = _update_observation_noise_blocked(X, EZ, V_smooth, state, config)
        else:
            state = _update_unstructured(X, EZ, V_smooth, VVsmooth, state, config)
    except (NumericalError, np.linalg.LinAlgError) as e:
        raise _MStepError(f"M-step failed at iteration {num_iter + 1}: {e}", loglik=loglik) from e

    # The maximizers Z_0 = E[x_0 | Y] and V_0 = Cov(x_0 | Y) (projected to PSD).
    state = replace(state, Z_0=np.array(EZ[0], dtype=np.float64), V_0=_initial_covariance_update(V_smooth[0]))

    if config.ar_clip is not None:
        A_new, clip_stats = _clip_estimated_ar(state, config)
        if clip_stats['n_clipped'] > 0:
            _logger.debug(f"AR clipping: {clip_stats['n_clipped']}/{clip_stats['n_total']} coefficients clipped")
        state = replace(state, A=A_new)

    m_step_time = time_module.time() - m_step_start
    if m_step_time > MAX_MSTEP_SECONDS:
        raise NumericalError(
            f"M-step exceeded time budget ({m_step_time:.2f}s > {MAX_MSTEP_SECONDS:.2f}s). "
            f"Likely numerical instability or excessive fallbacks. "
            f"Iteration {num_iter + 1}.",
            details=f"M-step time: {m_step_time:.2f}s, budget: {MAX_MSTEP_SECONDS:.2f}s. "
                   f"This indicates pathological regime - check data scaling and model configuration."
        )

    total_time = e_step_time + m_step_time
    if total_time > MAX_EM_ITERATION_SECONDS:
        raise NumericalError(
            f"EM iteration exceeded total time budget ({total_time:.2f}s > {MAX_EM_ITERATION_SECONDS:.2f}s). "
            f"Likely numerical instability. Iteration {num_iter + 1}.",
            details=f"Total iteration time: {total_time:.2f}s, budget: {MAX_EM_ITERATION_SECONDS:.2f}s. "
                   f"E-step: {e_step_time:.2f}s, M-step: {m_step_time:.2f}s."
        )

    if verbose_iterations:
        # After the M-step A, C, Q and R are arrays; only their finiteness is checked.
        try:
            Q_max_eigval = (np.max(np.abs(np.linalg.eigvalsh(state.Q)))
                            if np.isfinite(state.Q).all() and state.Q.size > 0 else np.nan)
        except (np.linalg.LinAlgError, ValueError):
            Q_max_eigval = np.nan
        R_diag_max = np.max(np.abs(np.diag(state.R))) if np.isfinite(state.R).all() else np.nan
        _logger.info(f"    M-step: Completed in {m_step_time:.1f}s | "
                     f"Max values: |A|={_max_abs(state.A):.3f}, |C|={_max_abs(state.C):.3f}, "
                     f"|Q|={_max_abs(state.Q):.3f} (max_eig={Q_max_eigval:.3f}), |R_diag|={R_diag_max:.3f}")

    if num_iter < 3 or num_iter % 5 == 0 or total_time > 30.0:
        e_step_pct = 100*e_step_time/total_time if total_time > 0 else 0
        m_step_pct = 100*m_step_time/total_time if total_time > 0 else 0
        _logger.info(f"  Iteration {num_iter + 1} timing: E-step={e_step_time:.2f}s ({e_step_pct:.1f}%), "
                     f"M-step={m_step_time:.2f}s ({m_step_pct:.1f}%), "
                     f"Total={total_time:.2f}s (T={X.shape[0]}, m={EZ.shape[1]})")

    return state, loglik, kalman_filter


#: Relative tolerance of the EM monotonicity guard: a point of the EM path whose exact
#: log-likelihood is below the best accepted one by more than this (relative) stops EM.
EM_MONOTONICITY_RTOL = 1e-6

#: Values of ``EMConfig.em_acceleration``.
EM_ACCELERATIONS = ("squarem", None)

#: Factor by which the SQUAREM step-length bound grows after an accepted step at
#: the bound and shrinks (not below 1) after a fallback from one, as in the
#: SQUAREM package of Varadhan and Roland (``mstep = 4``, initial bound 1).
SQUAREM_STEP_FACTOR = 4.0

#: The parameters EM estimates; SQUAREM extrapolates them stacked into one vector.
_SSM_PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")


def _params_finite(state: DFMModelState) -> bool:
    return all(p is not None and np.all(np.isfinite(p)) for p in
               (state.A, state.C, state.Q, state.R, state.Z_0, state.V_0))


def _is_psd(M: np.ndarray) -> bool:
    """``M`` is symmetric and positive semidefinite, both up to rounding relative to ``max |M|``."""
    M = np.asarray(M, dtype=np.float64)
    tol = 10.0 * M.shape[0] * np.finfo(np.float64).eps * np.max(np.abs(M), initial=0.0)
    return bool(np.max(np.abs(M - M.T), initial=0.0) <= tol and np.linalg.eigvalsh(_sym(M))[0] >= -tol)


def _squarem_point(theta0: DFMModelState, theta1: DFMModelState, theta2: DFMModelState,
                   step_max: float) -> Tuple[Optional[DFMModelState], float]:
    """SqS3 extrapolation (Varadhan and Roland 2008) from ``theta1 = F(theta0)`` and ``theta2 = F(theta1)``.

    With ``r = theta1 - theta0`` and ``v = theta2 - 2 theta1 + theta0`` over the
    stacked parameters (A, C, Q, R, Z_0, V_0), the step length is
    ``s = ||r|| / ||v||`` clipped to ``[1, step_max]`` (``alpha = -s``) and the
    point is ``theta0 - 2 alpha r + alpha^2 v``; ``s = 1`` gives ``theta2``,
    which is returned as is. The point is an affine combination of the three
    iterates (weights ``(1 + alpha)^2``, ``-2 alpha (1 + alpha)``,
    ``alpha^2``, summing to one), so entries equal in all three (structural
    zeros, the companion and chain shift entries, the fixed idiosyncratic
    loadings) are kept exactly and the linear tent restrictions on C hold to
    rounding. Returns ``(point, s)``, the point None unless every parameter is
    finite and Q, R and V_0 are symmetric positive semidefinite.
    """
    r = [getattr(theta1, n) - getattr(theta0, n) for n in _SSM_PARAMS]
    v = [getattr(theta2, n) - 2.0 * getattr(theta1, n) + getattr(theta0, n) for n in _SSM_PARAMS]
    r_norm = np.sqrt(sum(np.sum(d * d) for d in r))
    v_norm = np.sqrt(sum(np.sum(d * d) for d in v))
    step = float(np.clip(r_norm / v_norm, 1.0, step_max)) if v_norm > 0 else 1.0
    if not step > 1.0:
        return theta2, 1.0
    alpha = -step
    point = replace(theta0, **{n: getattr(theta0, n) - 2.0 * alpha * dr + alpha ** 2 * dv
                               for n, dr, dv in zip(_SSM_PARAMS, r, v)})
    if not (_params_finite(point) and all(_is_psd(getattr(point, n)) for n in ("Q", "R", "V_0"))):
        return None, step
    return point, step


def _relative_change(loglik: float, previous: float) -> float:
    """``|L_k - L_{k-1}| / |L_{k-1}|`` (absolute change when ``|L_{k-1}|`` is ~0)."""
    if not (np.isfinite(loglik) and np.isfinite(previous)):
        return float('inf')
    if abs(previous) > 1e-10:
        return abs((loglik - previous) / previous)
    return abs(loglik - previous)


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


class _EMRun:
    """The iterations of one ``run_em_algorithm`` call and their bookkeeping.

    Holds the E-step count ``num_iter``, the best accepted parameters with
    their exact log-likelihood, the convergence state, the log-likelihoods of
    the accepted points (``loglik_path``), the SQUAREM fallbacks and step-length
    bound, and the stop reason. ``plain`` and ``squarem`` return the parameters
    EM ends on; when ``stop_reason`` is set, ``run_em_algorithm`` returns
    ``best_state`` instead. With plain EM every iterate is an accepted point.
    """

    def __init__(self, X: np.ndarray, kalman_filter: DFMKalmanFilter, config: EMConfig,
                 block_structure: Optional[BlockStructure], max_iter: int, threshold: float,
                 checkpoint_callback: Optional[Callable[[int, DFMModelState], None]]):
        self.X = X
        self.kalman_filter = kalman_filter
        self.config = config
        self.block_structure = block_structure
        self.max_iter = max_iter
        self.threshold = threshold
        self.checkpoint_callback = checkpoint_callback
        self.num_iter = 0
        self.best_state: Optional[DFMModelState] = None
        self.best_loglik = float('-inf')
        self.previous_loglik = float('-inf')
        self.converged = False
        self.stop_reason: Optional[str] = None
        self.change = DEFAULT_ZERO_VALUE
        self.loglik_path: list = []
        self.squarem_fallbacks = 0
        self.step_max = 1.0
        self._start_time = time_module.time()
        self._iter_start = self._start_time
        self._iteration_times: list = []

    def _em_map(self, state: DFMModelState) -> Tuple[DFMModelState, float]:
        """``(F(state), L(state))``: ``em_step`` at ``state``, damped when ``config.damping_factor`` is set."""
        k = self.num_iter
        self._iter_start = time_module.time()
        if k % 10 == 0:
            _logger.info(f"EM heartbeat: iteration {k}/{self.max_iter}")
        if k < 3:
            _logger.info(f"Starting iteration {k + 1}/{self.max_iter}...")
        new_state, loglik, _ = em_step(
            self.X, state, kalman_filter=self.kalman_filter, config=self.config,
            block_structure=self.block_structure, num_iter=k
        )
        alpha = self.config.damping_factor
        if alpha is not None and 0 < alpha <= 1:
            new_state = replace(
                new_state,
                **{name: alpha * getattr(new_state, name) + (1 - alpha) * getattr(state, name)
                   for name in _SSM_PARAMS}
            )
            if k < 3:
                _logger.info(f"  Applied damping: {alpha*100:.0f}% new, {(1-alpha)*100:.0f}% old")
                _logger.warning("Damping breaks the EM monotonicity guarantee.")
        return new_state, loglik

    def _keep_best(self, state: DFMModelState, loglik: float) -> None:
        if loglik > self.best_loglik or self.best_state is None:
            self.best_state, self.best_loglik = state, float(loglik)

    def step(self, state: DFMModelState, accepted: bool = True) -> Optional[Tuple[DFMModelState, float]]:
        """E-step and EM map at a point of the EM path: ``(F(state), L(state))``, or None when EM stops.

        Applies the failure handling and the monotonicity guard of
        ``run_em_algorithm`` against the best accepted log-likelihood
        (``stop_reason`` is set when it returns None); ``accepted`` points
        update the best parameters.
        """
        k = self.num_iter
        try:
            new_state, loglik = self._em_map(state)
        except _MStepError as e:
            if k == 0:
                _logger.error(f"EM step failed at iteration 1: {e}", exc_info=True)
                raise RuntimeError(f"EM algorithm failed at iteration 1: {e}") from e
            # The E-step at state succeeded (its loglik is e.loglik); no update
            # could be computed from it.
            _logger.error(f"EM: {e}; stopping.")
            if accepted and np.isfinite(e.loglik):
                self._keep_best(state, e.loglik)
            self.stop_reason = "M-step failed"
            return None
        except NumericalError as e:
            if k == 0:
                _logger.error(f"EM step failed at iteration 1: {e}", exc_info=True)
                raise RuntimeError(f"EM algorithm failed at iteration 1: {e}") from e
            # The filter is not usable at state (the previous M-step output).
            _logger.error(f"EM: the E-step failed at iteration {k + 1} ({e}); stopping.")
            self.stop_reason = "E-step failed"
            return None
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:  # other errors are bugs: not wrapped
            _logger.error(f"EM step failed at iteration {k + 1}: {e}", exc_info=True)
            _logger.error(f"  Current parameters shapes - A: {state.A.shape if state.A is not None else None}, C: {state.C.shape if state.C is not None else None}, Q: {state.Q.shape if state.Q is not None else None}, R: {state.R.shape if state.R is not None else None}")
            _logger.error(f"  Data shape: {self.X.shape}, Block structure: {self.block_structure is not None}")
            raise RuntimeError(f"EM algorithm failed at iteration {k + 1}: {e}") from e

        if not np.isfinite(loglik):
            if k == 0:
                raise RuntimeError(f"EM algorithm failed at iteration 1: the log-likelihood of the "
                                   f"initial parameters is not finite ({loglik!r}).")
            _logger.error(f"EM: the log-likelihood is not finite at iteration {k + 1}; stopping.")
            self.stop_reason = "non-finite log-likelihood"
            return None
        if self.best_state is not None and loglik < self.best_loglik - EM_MONOTONICITY_RTOL * abs(self.best_loglik):
            _logger.warning(
                f"EM: the exact log-likelihood decreased at iteration {k + 1} "
                f"({self.best_loglik:.6f} -> {loglik:.6f}); stopping and returning the best parameters."
            )
            self.stop_reason = "log-likelihood decreased"
            return None
        if accepted:
            self._keep_best(state, loglik)
        if k == 0 and loglik < -1e10:
            _logger.warning(f"Extremely negative initial log-likelihood: {loglik:.2e}. "
                            f"This may indicate numerical instability or data scaling issues.")
        return new_state, loglik

    def accept(self, loglik: float) -> None:
        """Record the exact log-likelihood of an accepted point and test convergence on it.

        From E-step ``config.min_iterations_for_convergence_check`` on, EM has
        converged when the relative change from the previous accepted point is
        below ``threshold``.
        """
        if self.num_iter >= self.config.min_iterations_for_convergence_check:
            self.change = _relative_change(loglik, self.previous_loglik)
            self.converged = self.change < self.threshold
        elif np.isfinite(self.previous_loglik):
            self.change = abs(loglik - self.previous_loglik)
        self.previous_loglik = loglik
        self.loglik_path.append(float(loglik))

    def advance(self, state: DFMModelState, loglik: float) -> None:
        """Count the E-step just run, log progress and checkpoint ``state`` (the parameters EM would end on now)."""
        self.num_iter += 1
        num_iter, max_iter = self.num_iter, self.max_iter
        iter_time = time_module.time() - self._iter_start
        self._iteration_times.append(iter_time)
        if len(self._iteration_times) > 5:
            self._iteration_times.pop(0)
        avg_iter_time = sum(self._iteration_times) / len(self._iteration_times)
        if (num_iter % self.config.progress_log_interval == 0) or self.converged:
            elapsed_time = time_module.time() - self._start_time
            progress_msg = f"[{(num_iter / max_iter) * 100:.1f}%] Elapsed: {_format_time(elapsed_time)}"
            remaining_iters = max_iter - num_iter if not self.converged else 0
            if remaining_iters > 0 and len(self._iteration_times) >= 2:
                progress_msg += f", Est. remaining: {_format_time(avg_iter_time * remaining_iters)}"
            log_em_iteration(
                iteration=num_iter,
                loglik=loglik,
                delta=self.change if self.change > 0 else None,
                max_iter=max_iter,
                converged="✓" if self.converged else ""
            )
            _logger.info(f"  Progress: {progress_msg}, Iteration time: {iter_time:.1f}s (avg: {avg_iter_time:.1f}s)")

        if self.checkpoint_callback is not None and num_iter % 5 == 0:
            try:
                _logger.info(f"Saving checkpoint at iteration {num_iter}...")
                self.checkpoint_callback(num_iter, state)
            except Exception as e:  # documented boundary: a failing checkpoint must not abort a long EM run
                _logger.warning(f"Checkpoint callback failed at iteration {num_iter}: {e}", exc_info=True)

    def _finite(self, state: DFMModelState) -> bool:
        """True if every parameter of ``state`` is finite; otherwise stop EM."""
        if _params_finite(state):
            return True
        _logger.error(f"EM: non-finite parameters after iteration {self.num_iter}; stopping.")
        self.stop_reason = "non-finite parameters"
        return False

    def plain(self, state: DFMModelState) -> DFMModelState:
        """Iterate the EM map from ``state``; every iterate is an accepted point."""
        while self.num_iter < self.max_iter:
            out = self.step(state)
            if out is None:
                break
            state, loglik = out
            self.accept(loglik)
            self.advance(state, loglik)
            if self.converged or not self._finite(state):
                break
        return state

    def squarem(self, state: DFMModelState) -> DFMModelState:
        """SqS3 cycles from ``state``: two EM maps, the extrapolation, one stabilizing EM map.

        Each cycle starts at an accepted point ``theta0`` (convergence is tested
        on its log-likelihood, and on convergence EM ends on ``F(theta0)``, as
        plain EM does), maps it to ``theta1`` and ``theta2`` and extrapolates
        with the step length bounded by ``step_max`` (``_squarem_point``). The
        next accepted point is ``F(theta')`` when ``theta'`` is valid, its
        E-step and M-step succeed and ``L(theta') >= L(theta0)``: the
        log-likelihood the stabilizing E-step gives, and with an exact E-step
        ``L(F(theta')) >= L(theta')``, so the accepted log-likelihoods do not
        decrease without an extra E-step. Otherwise the next accepted point is
        the plain EM point ``theta2`` (a fallback). The bound ``step_max``
        starts at 1 (a cycle of three plain EM maps) and adapts as in the
        SQUAREM package (``_stabilize``), so a slowly converging EM is not
        extrapolated far past the likelihood's ascent region. Every E-step
        counts towards ``max_iter``.
        """
        theta0 = state
        while self.num_iter < self.max_iter:
            out = self.step(theta0)
            if out is None:
                break
            theta1, loglik0 = out
            self.accept(loglik0)
            self.advance(theta1, loglik0)
            state = theta1
            if self.converged or not self._finite(theta1) or self.num_iter >= self.max_iter:
                break
            out = self.step(theta1, accepted=False)
            if out is None:
                break
            state, loglik1 = out
            self.advance(state, loglik1)
            if not self._finite(state) or self.num_iter >= self.max_iter:
                break
            point, step = _squarem_point(theta0, theta1, state, self.step_max)
            theta0 = state = self._stabilize(point, step, loglik0, fallback=state)
        return state

    def _stabilize(self, point: Optional[DFMModelState], step: float, loglik0: float,
                   fallback: DFMModelState) -> DFMModelState:
        """``F(point)`` if the extrapolated ``point`` is accepted (see ``squarem``), else ``fallback``.

        An invalid ``point`` (None) falls back without an E-step. ``step`` is
        the step length of ``point``; when it is the bound ``step_max``, the
        bound is multiplied by ``SQUAREM_STEP_FACTOR`` on acceptance and divided
        by it (not below 1) on a fallback.
        """
        new_state, loglik = None, float('nan')
        if point is None:
            _logger.debug("EM: the SQUAREM point is not a valid parameter set; using the EM point.")
        else:
            try:
                new_state, loglik = self._em_map(point)
            except (NumericalError, np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                _logger.debug(f"EM: the E-step or M-step at the SQUAREM point failed ({e}); using the EM point.")
        accepted = new_state is not None and loglik >= loglik0 and _params_finite(new_state)
        if step == self.step_max:
            self.step_max = (self.step_max * SQUAREM_STEP_FACTOR if accepted
                             else max(1.0, self.step_max / SQUAREM_STEP_FACTOR))
        if not accepted:
            if point is not None:
                _logger.debug(f"EM: the SQUAREM point does not raise the log-likelihood "
                              f"({loglik0:.6f} -> {loglik:.6f}); using the EM point.")
            self.squarem_fallbacks += 1
            new_state = fallback
        if point is not None:
            self.advance(new_state, loglik)
        return new_state


def run_em_algorithm(
    X: np.ndarray,
    initial_state: DFMModelState,
    max_iter: int = 200,
    threshold: float = 1e-4,
    config: Optional[EMConfig] = None,
    checkpoint_callback: Optional[Callable[[int, DFMModelState], None]] = None
) -> Tuple[DFMModelState, Dict[str, Any]]:
    """Run EM from ``initial_state``; return the fitted state and its metadata.

    Each E-step at ``theta`` gives the exact smoothed moments and
    log-likelihood ``L(theta)``, and the M-step the EM map ``F(theta)``
    (``em_step``, then ``config.damping_factor``); with the exact E-step
    ``L(F(theta)) >= L(theta)``. ``max_iter`` bounds the number of E-steps.

    - ``config.em_acceleration=None`` (plain EM): ``theta_{k+1} = F(theta_k)``;
      every iterate is an accepted point.
    - ``"squarem"``: the SqS3 scheme of Varadhan and Roland (2008), see
      ``_EMRun.squarem``: three E-steps per cycle, the accepted points being
      the cycle starts; an extrapolation that would lower the log-likelihood
      falls back to the plain EM point. ``config.ar_clip`` and
      ``damping_factor`` apply to every EM map. With ``max_iter=1`` both
      settings run the same single EM map.

    From E-step ``config.min_iterations_for_convergence_check`` on, EM stops
    when the relative change ``|L_k - L_{k-1}| / |L_{k-1}|`` between
    consecutive accepted points is below ``threshold`` and returns the EM map
    of the last one; otherwise it returns the parameters it ends on after
    ``max_iter`` E-steps.

    Guards on the points of the EM path (the accepted points and, with
    SQUAREM, ``theta1``; each guard stops EM with ``converged=False`` and
    returns the accepted parameters with the best exact log-likelihood):

    - ``L`` is not finite, or ``L < L_best - EM_MONOTONICITY_RTOL * |L_best|``
      with ``L_best`` the best accepted log-likelihood (a decrease, which an
      exact EM cannot produce; a warning is logged);
    - the E-step fails (non-finite filter, an innovation covariance that is not
      positive definite) or the M-step cannot be computed (singular moment
      matrix) after the first E-step; in the first one either raises
      ``RuntimeError``;
    - the parameters to be returned fail the same checks in the final pass.

    The returned parameters get one more filter/smoother pass, after the
    optional likelihood-invariant factor rescaling (``config.normalize_C``).
    Metadata: ``loglik`` (of the returned parameters, from that pass),
    ``smoothed_factors`` (its smoothed states), ``num_iter`` (E-steps run),
    ``converged``, ``change``, ``stop_reason`` (None unless a guard stopped EM),
    ``best_loglik`` (of the best accepted point), ``loglik_path`` (the exact log-likelihoods of the accepted
    points, in order) and ``squarem_fallbacks`` (SQUAREM cycles that fell back
    to the plain EM point). ``checkpoint_callback(num_iter, state)`` runs every
    5 E-steps; its exceptions are logged and ignored. The opt-in
    ``config.damping_factor`` and ``config.ar_clip`` can break monotonicity;
    the guard applies to them too.
    """
    if config is None:
        config = _DEFAULT_EM_CONFIG
    if config.em_acceleration not in EM_ACCELERATIONS:
        raise ValueError(f"em_acceleration must be one of {EM_ACCELERATIONS}, got {config.em_acceleration!r}")

    if config.data_clip is not None and config.data_clip > 0:
        X_mean = np.nanmean(X, axis=0, keepdims=True)
        X_std = np.nanstd(X, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-10, 1.0, X_std)
        X_clipped = np.clip(X, X_mean - config.data_clip * X_std, X_mean + config.data_clip * X_std)
        n_clipped = np.sum((X != X_clipped) & np.isfinite(X))
        if n_clipped > 0:
            _logger.warning(f"Data clipping applied: {n_clipped}/{X.size} values clipped to ±{config.data_clip} std devs")
        X = X_clipped

    state = initial_state
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

    _logger.info(f"Starting EM algorithm: max_iter={max_iter}, threshold={threshold:.2e}, "
                 f"acceleration={config.em_acceleration}")
    if block_structure is not None:
        _logger.info(f"  Block structure: {state.blocks.shape[1]} blocks, {int(np.sum(state.r))} factors")
    if state.constraint_matrix is not None:
        _logger.info(f"  Mixed-frequency: tent kernel constraints enabled (R_mat shape: {state.constraint_matrix.shape})")

    run = _EMRun(X, kalman_filter, config, block_structure, max_iter, threshold, checkpoint_callback)
    state = run.squarem(state) if config.em_acceleration == "squarem" else run.plain(state)
    best_state, best_loglik, stop_reason, converged = run.best_state, run.best_loglik, run.stop_reason, run.converged

    def _final(candidate: DFMModelState) -> Tuple[DFMModelState, float, np.ndarray]:
        """Rescale (if configured), then the final filter/smoother pass at ``candidate``."""
        if config.normalize_C and candidate.C is not None and candidate.r is not None:
            _, candidate = _normalize_C_with_invariance(candidate.C, candidate, config, log_prefix="EM: ")
        kalman_filter.update_parameters(candidate.A, candidate.C, candidate.Q, candidate.R,
                                        candidate.Z_0, candidate.V_0)
        EZ_final, _, _, ll_final = kalman_filter.filter_and_smooth(X)
        if not np.isfinite(ll_final):
            raise NumericalError(f"the log-likelihood of the returned parameters is not finite ({ll_final!r})")
        return candidate, float(ll_final), EZ_final

    # The parameters to return: the ones EM ended on, unless EM stopped on a
    # guard; they must also pass the guard (finite, not below the best loglik).
    result = None
    if stop_reason is None:
        try:
            result = _final(state)
            if best_state is not None and result[1] < best_loglik - EM_MONOTONICITY_RTOL * abs(best_loglik):
                _logger.warning(
                    f"EM: the exact log-likelihood of the last M-step output is below the best one seen "
                    f"({best_loglik:.6f} -> {result[1]:.6f}); returning the best parameters."
                )
                stop_reason = "log-likelihood decreased"
                result = None
        except (NumericalError, ValueError) as e:
            _logger.error(f"EM: the final filter pass failed at the last M-step output ({e}); "
                          f"returning the best parameters.")
            stop_reason = "final E-step failed"
    if result is None:
        converged = False
        if best_state is None:  # max_iter == 0
            best_state = initial_state
        try:
            result = _final(best_state)
        except (NumericalError, ValueError) as e:
            _logger.error(f"EM: the final filter pass failed at the best parameters ({e}); "
                          f"no smoothed factors are cached.")
            result = (best_state, best_loglik, None)
    state, final_loglik, EZ_final = result

    log_convergence(
        converged=converged,
        num_iter=run.num_iter,
        final_loglik=final_loglik if np.isfinite(final_loglik) else None,
        reason=("converged" if converged else
                (stop_reason if stop_reason is not None else f"max_iterations_reached (change: {run.change:.2e})")),
        model_type="dfm"
    )

    metadata = {
        'loglik': final_loglik,
        'num_iter': run.num_iter,
        'converged': converged,
        'change': run.change,
        'smoothed_factors': EZ_final,
        'stop_reason': stop_reason,
        'best_loglik': best_loglik,
        'loglik_path': run.loglik_path,
        'squarem_fallbacks': run.squarem_fallbacks,
    }
    return state, metadata

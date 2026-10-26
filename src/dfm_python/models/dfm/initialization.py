"""DFM state-space initialization: block-by-block PCA with tent-kernel aggregation of slower series."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy import linalg as sla
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

from ...config.constants import (
    DEFAULT_DTYPE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_IDENTITY_SCALE,
    MIN_EIGENVALUE,
    DEFAULT_REGULARIZATION,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_PROCESS_NOISE,
    MIN_OBSERVATION_NOISE,
)
from ...functional.em import _DEFAULT_EM_CONFIG
from ...functional.state_layout import DFMStateLayout, bm14_idio_indicator
from ...numeric.builder import build_dfm_slower_freq_observation_matrix, build_lag_matrix
from ...models.dfm.mixed_freq import build_slower_freq_idiosyncratic_chain, find_slower_frequency
from .tent import get_slower_freq_tent_weights
from ...numeric.stability import (
    ensure_process_noise_stable,
    create_scaled_identity,
)
from ...numeric.estimator import (
    estimate_ar1,
    estimate_variance,
    estimate_var,
)
from ...utils.helper import handle_linear_algebra_error
from ...utils.errors import NumericalError
from ...logger import get_logger
from ...config.constants import DEFAULT_NAN_K, DEFAULT_NAN_METHOD

_logger = get_logger(__name__)


#: Variance of the diffuse part of V_0 on the (near) unit-root subspace of A.
DIFFUSE_INITIAL_VARIANCE = 1e4
#: Eigenvalues of A with modulus >= 1 - this are treated as unit roots for V_0.
UNIT_ROOT_TOLERANCE = 1e-3


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _psd(M: np.ndarray) -> np.ndarray:
    """Symmetric part of ``M``, with negative eigenvalues (round-off) set to 0 only if there are any."""
    M = _sym(np.asarray(M, dtype=np.float64))
    if M.size == 0:
        return M
    w, U = np.linalg.eigh(M)
    if w[0] < 0.0:
        M = _sym((U * np.maximum(w, 0.0)) @ U.T)
    return M


def initial_state_covariance(
    A: np.ndarray,
    Q: np.ndarray,
    kappa: float = DIFFUSE_INITIAL_VARIANCE,
    unit_root_tol: float = UNIT_ROOT_TOLERANCE,
) -> np.ndarray:
    """Initial state covariance ``V_0`` for the transition ``(A, Q)``.

    If every eigenvalue of ``A`` has modulus below ``1 - unit_root_tol``, ``V_0``
    is the stationary covariance, the exact solution of the discrete Lyapunov
    equation ``V = A V A' + Q`` (no regularization). Otherwise, or if that
    solve fails or is not finite, ``V_0`` is the stationary covariance on the
    stable invariant subspace of ``A`` plus a finite diffuse ``kappa * I`` on
    the (near) unit-root subspace: with the ordered real Schur form
    ``A = U T U'`` (unit roots first), ``V_0 = U diag(kappa I, P_22) U'`` where
    ``P_22`` solves the Lyapunov equation of the stable block ``T_22``. Both
    branches are deterministic. The result is symmetric and PSD (negative
    round-off eigenvalues are set to 0).
    """
    A = np.asarray(A, dtype=np.float64)
    Q = _sym(np.asarray(Q, dtype=np.float64))
    m = A.shape[0]
    if m == 0:
        return np.zeros((0, 0))
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho < 1.0 - unit_root_tol:
        try:
            V = sla.solve_discrete_lyapunov(A, Q)
            if np.all(np.isfinite(V)):
                return _psd(V)
        except (np.linalg.LinAlgError, ValueError):
            pass
    tol = 1.0 - unit_root_tol
    T_s, U, k = sla.schur(A, output="real", sort=lambda re, im: abs(complex(re, im)) >= tol)
    P = np.zeros((m, m))
    if k < m:
        Qs = U.T @ Q @ U
        P[k:, k:] = sla.solve_discrete_lyapunov(T_s[k:, k:], _sym(Qs[k:, k:]))
    if k > 0:
        P[:k, :k] = kappa * np.eye(k)
    _logger.info(
        f"Initial state covariance: spectral radius of A is {rho:.4f}; {k} of {m} directions "
        f"get the diffuse variance {kappa:g}, the rest the stationary covariance."
    )
    return _psd(U @ P @ U.T)


def tent_restricted_least_squares(
    y: np.ndarray,
    X: np.ndarray,
    R_con: np.ndarray,
    q_con: np.ndarray,
) -> np.ndarray:
    """``argmin ||y - X c||`` subject to ``R_con c = q_con``, with the restriction exact.

    ``c = c_p + N z`` with ``c_p`` the minimum-norm solution of the restriction
    and ``N`` an orthonormal basis of the null space of ``R_con``; ``z`` is the
    least-squares fit of ``y - X c_p`` on ``X N``. ``R_con c - q_con`` is then
    zero up to rounding, whatever the conditioning of ``X'X``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    R_con = np.asarray(R_con, dtype=np.float64)
    q_con = np.asarray(q_con, dtype=np.float64).ravel()
    c_p = np.linalg.lstsq(R_con, q_con, rcond=None)[0]
    basis = sla.null_space(R_con)
    z = np.linalg.lstsq(X @ basis, y - X @ c_p, rcond=None)[0]
    return c_p + basis @ z


def _interpolate(col: np.ndarray, kind: str) -> np.ndarray:
    """``col`` interpolated (and extrapolated) over its NaN entries, at every index."""
    valid_indices = np.where(~np.isnan(col))[0]
    f = interp1d(valid_indices, col[valid_indices], kind=kind, fill_value='extrapolate', bounds_error=False)
    return f(np.arange(len(col)))


def remNaNs_spline(x: np.ndarray, optNaN: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the NaNs of ``x`` (T x N) by splines (FRBNY ``remNaNs_spline``).

    ``optNaN['method']`` 2 (the default) interpolates the gaps of each series
    (cubic with at least ``k + 1`` observations, else linear, extrapolating at
    the ends; forward/backward fill if that fails); 3 splines every series the
    same way, falling back to linear and then to forward/backward fill; any
    other value forward/backward fills. ``optNaN['k']`` is the spline order.
    An all-NaN series becomes 0.

    Returns
    -------
    xBal : np.ndarray
        The filled data (T x N).
    indNaN : np.ndarray
        Boolean (T x N), True where ``x`` was NaN.
    """
    T, N = x.shape
    indNaN = np.isnan(x)
    xBal = x.copy()

    method = optNaN.get('method', DEFAULT_NAN_METHOD)
    k = optNaN.get('k', DEFAULT_NAN_K)

    if method == 2:
        for j in range(N):
            col = x[:, j]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                xBal[:, j] = 0.0
                continue
            col_processed = col.copy()
            if not valid_mask.all():
                try:
                    kind = 'cubic' if valid_mask.sum() >= k + 1 else 'linear'
                    col_processed[~valid_mask] = _interpolate(col_processed, kind)[~valid_mask]
                except (ValueError, np.linalg.LinAlgError):  # too few points, or a singular spline system
                    col_processed = pd.Series(col_processed).ffill().bfill().fillna(0.0).values
            xBal[:, j] = col_processed
    elif method == 3:
        for j in range(N):
            col = x[:, j]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                xBal[:, j] = 0.0
            elif valid_mask.sum() >= k + 1:
                try:
                    xBal[:, j] = _interpolate(col, 'cubic')
                except (ValueError, np.linalg.LinAlgError):  # as in method 2
                    xBal[:, j] = _interpolate(col, 'linear')
            else:
                xBal[:, j] = pd.Series(col).ffill().bfill().fillna(0.0).values
    else:
        xBal = pd.DataFrame(x).ffill().bfill().fillna(0.0).values

    if np.isnan(xBal).any():
        xBal = np.nan_to_num(xBal, nan=0.0)
    return xBal, indNaN


def impute_for_init(data: np.ndarray) -> np.ndarray:
    """Forward fill, then backward fill, then 0 (an all-NaN series) of a 1-D array."""
    data_imputed = data.copy()
    mask = np.isnan(data_imputed)
    if not mask.any():
        return data_imputed
    # A writable copy: the fill below writes into it, and under pandas
    # copy-on-write (the default from pandas 3) ``.values`` is read-only.
    data_imputed = pd.Series(data_imputed).ffill().bfill().to_numpy(copy=True)
    # NaN survives the fills only when every value is NaN; the mean is then
    # undefined and 0 is used.
    mask = np.isnan(data_imputed)
    if mask.any():
        data_imputed[mask] = 0.0
    return data_imputed


def initialize_clock_freq_idio(
    res: np.ndarray,
    data_with_nans: np.ndarray,
    n_clock_freq: int,
    idio_indicator: Optional[np.ndarray],
    T: int,
    dtype: type = DEFAULT_DTYPE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AR(1) idiosyncratic states of the clock-frequency series from their residuals ``res`` (T x N).

    Each series with an idiosyncratic state (``idio_indicator > 0``; all of
    the first ``n_clock_freq`` when None) gets an AR(1) fitted on its residual
    between its first and last observation, or the EM defaults with fewer
    than two observations. ``data_with_nans`` and ``T`` are unused.

    Returns
    -------
    BM, SM, initViM : np.ndarray
        Diagonal transition, process noise and stationary initial covariance
        ``SM / (1 - BM^2)`` (``SM * DEFAULT_IDENTITY_SCALE`` at a unit root).
    """
    n_idio_clock = n_clock_freq if idio_indicator is None else int(np.sum(idio_indicator))
    BM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)
    SM = np.zeros((n_idio_clock, n_idio_clock), dtype=dtype)

    idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
    default_ar_coef = _DEFAULT_EM_CONFIG.slower_freq_ar_coef
    default_noise = _DEFAULT_EM_CONFIG.default_process_noise

    for i, idx in enumerate(idio_indices):
        non_nan_mask = ~np.isnan(res[:, idx])
        if np.sum(non_nan_mask) > 1:
            first_non_nan = np.where(non_nan_mask)[0][0]
            last_non_nan = np.where(non_nan_mask)[0][-1]
            res_i_clean = res[first_non_nan:last_non_nan + 1, idx]

            def _estimate_ar1_for_idio() -> np.ndarray:
                A_diag, Q_diag = estimate_ar1(
                    y=res_i_clean[1:].reshape(-1, 1),
                    x=res_i_clean[:-1].reshape(-1, 1),
                    V_smooth=None,
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

    def _compute_initViM() -> np.ndarray:
        initViM = np.zeros_like(SM)
        for i in range(n_idio_clock):
            denominator = 1.0 - BM[i, i] ** 2
            if abs(denominator) > 1e-10 and np.isfinite(denominator):
                initViM[i, i] = SM[i, i] / denominator
            else:
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
    """Loadings of one block: PCA on the clock series, tent-restricted least squares on the slower ones.

    Parameters
    ----------
    data_for_extraction : np.ndarray
        (T x N) splined data for the first block, residuals after the
        earlier blocks for the next ones.
    data_with_nans : np.ndarray
        (T x N) the same with the missing entries NaN.
    clock_freq_indices, slower_freq_indices : np.ndarray
        Series of the block at the clock and at the slower frequency.
    num_factors : int
        Factors of the block.
    tent_kernel_size : int
        Length of the tent kernel.
    R_mat, q : np.ndarray, optional
        Tent constraint of one factor; ``kron(R_mat, I)`` restricts the block.
    N : int
        Number of series.
    max_lag_size : int
        Lags per factor in the state.
    matrix_regularization : float, optional
        Unused: the slower-frequency loadings are the exact tent-restricted
        least-squares fit (``tent_restricted_least_squares``).
    dtype : type, default np.float32

    Returns
    -------
    C_i : np.ndarray
        (N x num_factors * max_lag_size); the rows of the slower series
        satisfy ``kron(R_mat, I) c = 0``. A slower series with fewer
        observations than regressors keeps zero loadings.
    factors : np.ndarray
        (T x num_factors), the PCA factors (NaN where the data is NaN).
    """
    from ...layer.pca import compute_principal_components

    T = data_for_extraction.shape[0]
    C_i = np.zeros((N, num_factors * max_lag_size), dtype=dtype)

    if len(clock_freq_indices) == 0:
        factors = np.zeros((T, num_factors), dtype=dtype)
    else:
        clock_freq_data = data_for_extraction[:, clock_freq_indices]
        # PCA needs finite values: center with nan-aware means, then zero the NaNs.
        clock_freq_data_mean = np.nanmean(clock_freq_data, axis=0, keepdims=True)
        clock_freq_data_centered = clock_freq_data - clock_freq_data_mean
        clock_freq_data_centered_clean = np.where(
            np.isfinite(clock_freq_data_centered),
            clock_freq_data_centered,
            0.0
        )

        if clock_freq_data_centered_clean.shape[0] <= 1:
            cov_data = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)
        elif len(clock_freq_indices) == 1:
            cov_data = np.atleast_2d(np.nanvar(clock_freq_data_centered, axis=0, ddof=0))
        else:
            cov_data = np.cov(clock_freq_data_centered_clean.T)
            cov_data = (cov_data + cov_data.T) / 2

        try:
            # PCA extracts at most min(n_series, num_factors) components.
            max_extractable = min(len(clock_freq_indices), num_factors)
            _, eigenvectors = compute_principal_components(cov_data, max_extractable, block_idx=0)
            loadings = eigenvectors
            loadings = np.where(np.sum(loadings, axis=0) < 0, -loadings, loadings)
            if loadings.shape[1] < num_factors:
                padding = np.zeros((loadings.shape[0], num_factors - loadings.shape[1]), dtype=dtype)
                loadings = np.hstack([loadings, padding])
        except (RuntimeError, ValueError):
            loadings = create_scaled_identity(len(clock_freq_indices), DEFAULT_IDENTITY_SCALE, dtype=dtype)[:, :num_factors]

        C_i[clock_freq_indices, :num_factors] = loadings
        # NaN in the data gives NaN factors, which the Kalman filter handles.
        n_actual_factors = min(len(clock_freq_indices), num_factors)
        factors = data_for_extraction[:, clock_freq_indices] @ loadings[:, :n_actual_factors]
        if factors.shape[1] < num_factors:
            padding = np.zeros((factors.shape[0], num_factors - factors.shape[1]), dtype=dtype)
            factors = np.hstack([factors, padding])

    if R_mat is not None and q is not None and len(slower_freq_indices) > 0:
        constraint_matrix_block = np.kron(R_mat, create_scaled_identity(num_factors, DEFAULT_IDENTITY_SCALE, dtype=dtype))
        constraint_vector_block = np.kron(q, np.zeros(num_factors, dtype=dtype))

        lag_matrix = build_lag_matrix(factors, T, num_factors, tent_kernel_size, 1, dtype)
        n_cols = min(num_factors * tent_kernel_size, lag_matrix.shape[1])
        slower_freq_factors = lag_matrix[:, :n_cols]

        total_slower = len(slower_freq_indices)
        _logger.info(f"    Processing {total_slower} slower-frequency series with constrained OLS...")

        for idx, series_idx in enumerate(slower_freq_indices):
            if idx == 0 or (idx + 1) % 10 == 0 or (idx + 1) == total_slower:
                _logger.info(f"      Series {idx + 1}/{total_slower} (index {series_idx})")
            series_idx_int = int(series_idx)

            # The regression rows are the observations of the series (FRBNY). With
            # too few of them the splined data is used instead (MATLAB line 704:
            # xx_j = res(pC:end,j)), still on the rows where the series is observed.
            series_data_original = data_with_nans[tent_kernel_size:, series_idx_int]
            non_nan_mask = ~np.isnan(series_data_original)
            min_required = slower_freq_factors.shape[1] + 2
            if np.sum(non_nan_mask) < min_required:
                series_data = data_for_extraction[tent_kernel_size:, series_idx_int]
            else:
                series_data = series_data_original
            slower_freq_factors_clean = slower_freq_factors[tent_kernel_size:][non_nan_mask, :]
            series_data_clean = series_data[non_nan_mask]

            if len(slower_freq_factors_clean) < slower_freq_factors_clean.shape[1]:
                continue

            try:
                # The tent restriction is imposed exactly (null-space parameterization,
                # no ridge term), as every later M-step does.
                loadings_constrained = tent_restricted_least_squares(
                    series_data_clean, slower_freq_factors_clean,
                    constraint_matrix_block, constraint_vector_block,
                )
                if np.any(~np.isfinite(loadings_constrained)):
                    raise NumericalError(
                        f"Restricted least squares returned non-finite values for series {series_idx_int}. "
                        f"Check data quality and tent kernel configuration.",
                        details=f"Series index: {series_idx_int}, tent_kernel_size: {tent_kernel_size}, "
                                f"num_factors: {num_factors}"
                    )
                C_i[series_idx_int, :num_factors * tent_kernel_size] = loadings_constrained
            except (np.linalg.LinAlgError, ValueError) as e:
                raise NumericalError(
                    f"Restricted least squares failed for series {series_idx_int}: {e}.",
                    details=f"Series index: {series_idx_int}, tent_kernel_size: {tent_kernel_size}, "
                            f"num_factors: {num_factors}, error: {str(e)}"
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
    """VAR(p) transition of one block's factors from their lag matrix, in companion form.

    ``lag_matrix`` (T x num_factors * num_lags) holds ``[f_t, f_{t-1}, ...]``;
    the VAR is fitted by regularized OLS of ``f_t`` on its ``p`` lags over
    rows ``p:``. When ``T <= p``, ``p == 0`` or the fit is not finite (or
    raises), the VAR is ``default_transition_coef * I`` with process noise
    ``default_process_noise * I``. ``factors`` and ``matrix_regularization``
    are unused.

    Returns
    -------
    A_i, Q_i, V_0_i : np.ndarray
        ``(num_factors * max_lag_size,) ** 2`` each: the companion transition
        (identity shifts below the VAR rows), the process noise (nonzero on
        the first ``num_factors`` rows only, floored at ``eigenval_floor``)
        and the initial covariance ``initial_state_covariance(A_i, Q_i)``.
    """
    block_size = num_factors * max_lag_size
    A_i = np.zeros((block_size, block_size), dtype=dtype)

    n_cols = min(num_factors, lag_matrix.shape[1])
    current_state = lag_matrix[:, :n_cols] if n_cols > 0 else np.zeros((T, num_factors), dtype=dtype)
    lag_cols_end = min(num_factors * (p + 1), lag_matrix.shape[1])
    lagged_state = lag_matrix[:, num_factors:lag_cols_end] if lag_cols_end > num_factors else np.zeros((T, num_factors * p), dtype=dtype)

    default_A_block = create_scaled_identity(num_factors, default_transition_coef, dtype)
    shift_size = num_factors * (max_lag_size - 1)

    def _default() -> np.ndarray:
        A_i[:num_factors, :num_factors] = default_A_block
        Q_i = np.zeros((block_size, block_size), dtype=dtype)
        Q_i[:num_factors, :num_factors] = create_scaled_identity(num_factors, default_process_noise, dtype)
        return Q_i

    if T > p and lagged_state.shape[1] > 0:
        try:
            A_transition, Q_transition = estimate_var(
                y=current_state[p:, :],
                x=lagged_state[p:, :],
                V_smooth=None,
                VVsmooth=None,
                regularization=regularization,
                min_variance=eigenval_floor,
                dtype=dtype
            )
            if np.any(~np.isfinite(A_transition)) or np.any(~np.isfinite(Q_transition)):
                Q_i = _default()
            else:
                A_i[:num_factors, :num_factors * p] = A_transition
                Q_i = np.zeros((block_size, block_size), dtype=dtype)
                Q_i[:num_factors, :num_factors] = Q_transition
        except (np.linalg.LinAlgError, ValueError):
            Q_i = _default()
    else:
        Q_i = _default()

    if shift_size > 0:
        A_i[num_factors:, :shift_size] = create_scaled_identity(shift_size, DEFAULT_IDENTITY_SCALE, dtype=dtype)

    Q_i[:num_factors, :num_factors] = ensure_process_noise_stable(
        Q_i[:num_factors, :num_factors], min_eigenval=eigenval_floor, warn=True, dtype=dtype
    )
    V_0_i = initial_state_covariance(A_i, Q_i).astype(dtype)
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
    dtype: type = DEFAULT_DTYPE,
    return_residuals: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Factors, loadings and transitions of every block by sequential PCA.

    Block 1 is extracted from the (splined) data, each later block from the
    residuals after the earlier blocks, so the blocks capture different
    variance. The clock-frequency series (the first ``n_clock_freq`` columns)
    load directly; the slower ones through the tent kernel, with ``R_mat``
    and ``q`` the constraint of one factor.

    Parameters
    ----------
    data_for_extraction : np.ndarray
        (T x N) splined data.
    data_with_nans : np.ndarray
        (T x N) data with the missing entries NaN.
    blocks : np.ndarray
        (N x n_blocks) membership.
    r : np.ndarray
        Factors per block.
    n_blocks, n_clock_freq, tent_kernel_size, p, N, T : int
        Blocks, clock-frequency series, tent kernel size, VAR lag order,
        series and time steps.
    R_mat, q : np.ndarray, optional
        Tent kernel constraint matrix and vector.
    indNaN : np.ndarray
        (T x N) boolean, True where the data is missing.
    max_lag_size : int
        Lags per factor in the state.
    dtype : type, default DEFAULT_DTYPE
    return_residuals : bool, default False
        Also return the residuals after every block.

    Returns
    -------
    A_factors, Q_factors, V_0_factors : np.ndarray
        Block-diagonal transition, process noise and initial covariance.
    C : np.ndarray
        Loadings (N x sum of the blocks' ``num_factors * max_lag_size``).
    residuals, residuals_with_nans : np.ndarray
        Only with ``return_residuals=True``: the data minus every block's
        factor contribution (T x N), splined and with the missing entries
        restored to NaN (FRBNY's ``res`` and ``resNaN``).
    """
    C_list = []
    A_list = []
    Q_list = []
    V_0_list = []

    for block_idx in range(n_blocks):
        num_factors_block = int(r[block_idx])
        block_series_indices = np.where(blocks[:, block_idx] > 0)[0]
        clock_freq_indices = block_series_indices[block_series_indices < n_clock_freq]
        slower_freq_indices = block_series_indices[block_series_indices >= n_clock_freq]

        _logger.info(f"  Initializing block {block_idx + 1}/{n_blocks}: "
                    f"{num_factors_block} factors, {len(block_series_indices)} series "
                    f"({len(clock_freq_indices)} clock, {len(slower_freq_indices)} slower)")

        C_i, factors = initialize_block_loadings(
            data_for_extraction, data_with_nans, clock_freq_indices, slower_freq_indices,
            num_factors_block, tent_kernel_size, R_mat, q,
            N, max_lag_size, _DEFAULT_EM_CONFIG.matrix_regularization, dtype
        )

        # build_lag_matrix returns T rows, so the lagged factors are time-aligned
        # with the data: factor_t is taken to contribute to y_t (the tent kernel
        # widens the state, not the time axis), which is what the residualization
        # below assumes.
        lag_matrix = build_lag_matrix(factors, T, num_factors_block, tent_kernel_size, p, dtype)
        slower_freq_factors = lag_matrix[:, :num_factors_block * tent_kernel_size]
        data_for_extraction = data_for_extraction - slower_freq_factors @ C_i[:, :num_factors_block * tent_kernel_size].T
        data_with_nans = data_for_extraction.copy()
        data_with_nans[indNaN] = np.nan
        C_list.append(C_i)

        A_i, Q_i, V_0_i = initialize_block_transition(
            lag_matrix, factors, num_factors_block, max_lag_size, p, T,
            _DEFAULT_EM_CONFIG.regularization, _DEFAULT_EM_CONFIG.default_transition_coef,
            _DEFAULT_EM_CONFIG.default_process_noise, _DEFAULT_EM_CONFIG.matrix_regularization,
            _DEFAULT_EM_CONFIG.eigenval_floor, dtype
        )
        A_list.append(A_i)
        Q_list.append(Q_i)
        V_0_list.append(V_0_i)

    C = np.hstack(C_list) if C_list else np.zeros((N, 0), dtype=dtype)
    if A_list:
        A_factors = block_diag(*A_list)
        Q_factors = block_diag(*Q_list)
        V_0_factors = block_diag(*V_0_list)
    else:
        empty_matrix = np.zeros((0, 0), dtype=dtype)
        A_factors = Q_factors = V_0_factors = empty_matrix

    if return_residuals:
        return A_factors, Q_factors, V_0_factors, C, data_for_extraction, data_with_nans
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
    """Append the idiosyncratic columns to the loadings ``C`` (N x factor states).

    Clock-frequency series with an idiosyncratic state (``idio_indicator > 0``;
    the first ``n_clock_freq`` when None) get an identity column; the
    slower-frequency series get their tent chain, loaded with the weights of
    ``tent_weights_dict`` (symmetric weights of ``tent_kernel_size`` when the
    slower frequency has none there).
    """
    if idio_indicator is not None:
        eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
        C = np.hstack([C, eyeN[:, idio_indicator.astype(bool)]])
    elif n_clock_freq > 0:
        eyeN = create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=dtype)
        C = np.hstack([C, eyeN[:, :n_clock_freq]])

    if n_slower_freq > 0:
        slower_freq = find_slower_frequency(clock, tent_weights_dict)
        if tent_weights_dict and slower_freq in tent_weights_dict:
            tent_weights = tent_weights_dict[slower_freq].astype(dtype)
        else:
            tent_weights = get_slower_freq_tent_weights(slower_freq or 'q', clock, tent_kernel_size, dtype)
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
    """Diagonal observation noise ``R`` (N x N) from the residuals ``data_with_nans`` (T x N).

    ``R_ii`` is the nan-aware variance of the residuals for series without an
    idiosyncratic state, and ``MIN_OBSERVATION_NOISE`` (1e-4, BM14) for series
    with one (clock series with an AR(1) state and all slower-frequency
    series), the floor the EM M-step keeps them at.
    """
    T_res, N_res = data_with_nans.shape
    default_obs_noise = _DEFAULT_EM_CONFIG.default_observation_noise

    if T_res <= 1:
        R = create_scaled_identity(N_res, default_obs_noise, dtype)
    else:
        R = estimate_variance(
            residuals=data_with_nans,
            X=None,
            EZ=None,
            C=None,
            V_smooth=None,
            min_variance=default_obs_noise,
            default_variance=default_obs_noise,
            dtype=dtype
        )

    idio_indices = np.where(idio_indicator > 0)[0] if idio_indicator is not None else np.arange(n_clock_freq, dtype=np.int32)
    all_indices = np.unique(np.concatenate([idio_indices, np.arange(n_clock_freq, N, dtype=np.int32)]))
    R[np.ix_(all_indices, all_indices)] = np.diag(np.full(len(all_indices), MIN_OBSERVATION_NOISE, dtype=dtype))
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
    """Initial state-space parameters ``(A, C, Q, R, Z_0, V_0)`` of the DFM.

    Banbura-Modugno 2014 / FRBNY ``InitCond``, in the state layout of
    ``functional.state_layout.DFMStateLayout``. ``x`` (T x N) is the scaled
    data, the clock-frequency series first and the ``n_slower_freq``
    slower-frequency series last; the tent kernel is ``R_mat``'s width, else
    the length of the weights in ``tent_weights_dict``, else 1.

    1. Spline the data (``remNaNs_spline``) and mask the first
       ``tent_kernel_size - 1`` rows when a kernel is used.
    2. Block-by-block PCA (``initialize_block_factors``): VAR(p) of each
       block's factors on their lags; tent-restricted least squares, exact,
       for the slower series.
    3. Idiosyncratic columns of C: an AR(1) state per clock series, a tent
       chain loaded with the unnormalized weights per slower series. C is not
       column-normalized; EM rescales the factors once, after it converges.
    4. R: ``MIN_OBSERVATION_NOISE`` on every series with an idiosyncratic
       state (all of them in this layout).
    5. Idiosyncratic transitions from the residuals after the factors (FRBNY
       ``res``/``resNaN``): AR(1) per clock series; a chain per slower series
       whose head has stationary variance residual variance / 19.
    6. Block-diagonal A and Q (Q exactly zero on the companion lag states
       and the chain shift states); ``Z_0 = 0`` and
       ``V_0 = initial_state_covariance(A, Q)``.
    """
    T, N = x.shape
    dtype = DEFAULT_DTYPE
    n_blocks = blocks.shape[1]
    n_clock_freq = N - n_slower_freq

    x_clean = np.where(np.isinf(x), np.nan, x)
    data_for_extraction, indNaN = remNaNs_spline(x_clean, {'method': DEFAULT_NAN_METHOD, 'k': DEFAULT_NAN_K})
    data_with_nans = x_clean

    if R_mat is not None:
        tent_kernel_size = R_mat.shape[1]
    elif tent_weights_dict:
        tent_kernel_size = len(next(iter(tent_weights_dict.values())))
    else:
        # No kernel: a larger default would only inflate the state (max_lag_size),
        # mask initial observations and slow EM.
        tent_kernel_size = 1
    max_lag_size = max(p + 1, tent_kernel_size)

    if tent_kernel_size > 1 and (R_mat is not None or n_slower_freq > 0):
        data_with_nans[:tent_kernel_size-1, :] = np.nan

    if idio_indicator is None:
        idio_indicator = bm14_idio_indicator(n_clock_freq, n_slower_freq)
    idio_indicator = np.asarray(idio_indicator)

    A_factors, Q_factors, V_0_factors, C, res, res_nan = initialize_block_factors(
        data_for_extraction, data_with_nans, blocks, r, n_blocks, n_clock_freq, tent_kernel_size,
        p, R_mat, q, N, T, indNaN, max_lag_size, dtype, return_residuals=True
    )
    C = add_idiosyncratic_observation_matrix(
        C, N, n_clock_freq, n_slower_freq, idio_indicator, clock, tent_kernel_size, tent_weights_dict, dtype
    )

    # Residual variances (FRBNY) before R is set to the floor: the slower chains need them.
    with np.errstate(invalid="ignore", divide="ignore"):
        n_obs = np.sum(np.isfinite(res_nan), axis=0)
        res_var = np.where(n_obs > 1, np.nanvar(np.where(n_obs > 1, res_nan, 0.0), axis=0, ddof=1), np.nan)
    R = initialize_observation_noise(res_nan, N, idio_indicator, n_clock_freq, dtype)

    # Each clock series' AR(1) is fitted on its residual over its observed span.
    res_span = res.copy()
    for j in range(N):
        obs = np.flatnonzero(~indNaN[:, j])
        if obs.size:
            res_span[:obs[0], j] = np.nan
            res_span[obs[-1] + 1:, j] = np.nan
    BM, SM, initViM = initialize_clock_freq_idio(
        res_span, res_nan, n_clock_freq, idio_indicator, T, dtype=dtype
    )

    if n_slower_freq == 0:
        BQ = SQ = initViQ = np.zeros((0, 0), dtype=dtype)
    else:
        rho0 = _DEFAULT_EM_CONFIG.slower_freq_ar_coef
        sig_e = res_var[n_clock_freq:] / _DEFAULT_EM_CONFIG.slower_freq_variance_denominator
        sig_e = np.where(np.isfinite(sig_e) & (sig_e > 0), sig_e, MIN_OBSERVATION_NOISE)
        BQ, SQ, initViQ = build_slower_freq_idiosyncratic_chain(n_slower_freq, tent_kernel_size, rho0, sig_e, dtype)

    A = block_diag(A_factors, BM, BQ)
    Q = _sym(block_diag(Q_factors, SM, SQ))
    layout = DFMStateLayout(
        r=tuple(int(x) for x in np.asarray(r).ravel()), p=int(p), L=int(max_lag_size), n_series=N,
        n_clock=n_clock_freq, idio_series=tuple(int(i) for i in np.flatnonzero(idio_indicator > 0)),
        chain_length=tent_kernel_size if n_slower_freq > 0 else 0,
    )
    if A.shape[0] != layout.m or C.shape != (N, layout.m):
        raise NumericalError(
            f"Initialization built a state of dimension {A.shape[0]} (C {C.shape}), "
            f"but the DFM state layout has {layout.m}.",
            details=f"layout: {layout}"
        )

    Z_0 = np.zeros(layout.m, dtype=dtype)
    V_0 = initial_state_covariance(A, Q).astype(dtype)
    return A, C, Q, R, Z_0, V_0

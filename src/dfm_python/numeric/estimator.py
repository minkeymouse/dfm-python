"""Estimators of state-space parameters: VAR, AR, variances, AR clipping.

``estimate_var`` and ``estimate_ar1`` have a raw data mode (regularized OLS,
used by the initializations) and a smoothed expectations mode (the exact EM
M-step update in float64, used by ``functional.em``).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from scipy import linalg
from scipy.linalg import solve

if TYPE_CHECKING:
    from ..functional.em import EMConfig

from ..logger import get_logger
from ..utils.errors import ConfigurationError, DataValidationError, NumericalError
from ..config.constants import (
    MIN_DIAGONAL_VARIANCE,
    MIN_FACTOR_VARIANCE,
    DEFAULT_REGULARIZATION,
    MIN_EIGENVALUE,
    MAX_EIGENVALUE,
    VAR_STABILITY_THRESHOLD,
    AR_CLIP_MIN,
    AR_CLIP_MAX,
    MIN_Q_FLOOR,
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_ZERO_VALUE,
    DEFAULT_PROCESS_NOISE,
    DEFAULT_TRANSITION_COEF,
    DEFAULT_MIN_OBS,
    DEFAULT_DTYPE,
)
from .stability import (
    compute_var_safe,
    compute_cov_safe,
    solve_regularized_ols,
    stabilize_innovation_covariance,
    create_scaled_identity,
)
from ..utils.misc import get_config_attr
from .validator import validate_ndarray_ndim, validate_no_nan_inf

_logger = get_logger(__name__)


def clip_ar(
    A: np.ndarray,
    min_val: float = AR_CLIP_MIN,
    max_val: float = AR_CLIP_MAX,
    warn: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip every entry of ``A`` to ``[min_val, max_val]``.

    Parameters
    ----------
    A : np.ndarray
        Coefficients to clip.
    min_val : float, default AR_CLIP_MIN
    max_val : float, default AR_CLIP_MAX
    warn : bool, default True
        Log a warning when anything is clipped.

    Returns
    -------
    A_clipped : np.ndarray
    stats : dict
        ``n_clipped``, ``n_total``, ``clipped_indices`` (flat),
        ``min_violations`` and ``max_violations``.
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else DEFAULT_ZERO_VALUE
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def apply_ar_clipping(
    A: np.ndarray,
    config: Optional["EMConfig"] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip ``A`` to the bounds of ``config.ar_clip`` (``{"min": ..., "max": ...}``).

    Missing keys, and a None ``config``, use ``AR_CLIP_MIN``/``AR_CLIP_MAX``;
    ``config.ar_clip`` None clips nothing. Returns ``(A_clipped, stats)`` as
    ``clip_ar``.

    Raises
    ------
    ValueError
        If ``config.ar_clip`` is not a dict.
    """
    if config is None:
        return clip_ar(A, AR_CLIP_MIN, AR_CLIP_MAX, True)

    ar_clip = get_config_attr(config, 'ar_clip', None)
    if ar_clip is None:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    if not isinstance(ar_clip, dict):
        raise ValueError(f"ar_clip must be a dict with 'min' and 'max' keys, got {type(ar_clip)}")

    min_val = ar_clip.get('min', AR_CLIP_MIN)
    max_val = ar_clip.get('max', AR_CLIP_MAX)
    return clip_ar(A, min_val, max_val, warn=True)


def get_idio(eps: np.ndarray, idx_no_missings: np.ndarray, min_obs: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AR(1) statistics of each residual series (the original TensorFlow DDFM's).

    Only rows observed together with the previous row are used; a series with
    fewer than ``min_obs`` of them gets ``Phi = 0``, mean 0 and std 1.

    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals (T x N).
    idx_no_missings : np.ndarray
        Observed mask (T x N).
    min_obs : int, default 5
        Minimum number of usable rows.

    Returns
    -------
    Phi : np.ndarray
        Diagonal AR(1) coefficient matrix (N x N).
    mu_eps, std_eps : np.ndarray
        Mean and standard deviation of each series (N,).
    """
    Phi = np.zeros((eps.shape[1], eps.shape[1]))
    mu_eps = np.zeros(eps.shape[1])
    std_eps = np.zeros(eps.shape[1])
    for j in range(eps.shape[1]):
        to_select = idx_no_missings[:, j]
        to_select = np.hstack((np.array([False]), to_select[:-1] & to_select[1:]))
        if np.sum(to_select) < min_obs:
            std_eps[j] = 1.0  # Phi and mu stay 0
            continue
        this_eps = eps[to_select, j]
        mu_eps[j] = np.mean(this_eps)
        std_eps[j] = np.std(this_eps)
        if std_eps[j] > 0:  # a constant series has no AR(1) coefficient
            cov1_eps = np.cov(this_eps[1:], this_eps[:-1])[0][1]
            Phi[j, j] = cov1_eps / (std_eps[j] ** 2)
    return Phi, mu_eps, std_eps


def get_transition_params(f_t: np.ndarray, eps_t: np.ndarray, bool_no_miss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transition parameters of the DDFM state ``[f_t, eps_t]`` (TensorFlow DDFM's).

    A VAR(1) for the factors (ridge-stabilized least squares) and the
    ``get_idio`` AR(1) for the idiosyncratic components.

    Parameters
    ----------
    f_t : np.ndarray
        Common factors (T x m).
    eps_t : np.ndarray
        Idiosyncratic residuals (T x N).
    bool_no_miss : np.ndarray
        Observed mask (T x N).

    Returns
    -------
    A : np.ndarray
        Block-diagonal transition matrix (VAR(1) factors, AR(1) idio).
    Q : np.ndarray
        Diagonal process noise covariance (``W`` in the original).
    mu_0, Sigma_0 : np.ndarray
        Initial state mean and covariance (no factor-idio covariance, diagonal
        idio block).
    x_t : np.ndarray
        Stacked states ``[f_t, eps_t]``, ((m + N) x T).
    """
    f_past = f_t[:-1, :]
    # Upstream training can produce non-finite factors, which make pinv's SVD
    # fail. To keep forecasting possible they are zeroed, the normal equations
    # get a ridge scaled by their condition number, and lstsq (then the
    # identity) is the fallback.
    f_past_safe = np.nan_to_num(f_past, nan=0.0, posinf=0.0, neginf=0.0)
    f_next_safe = np.nan_to_num(f_t[1:, :], nan=0.0, posinf=0.0, neginf=0.0)

    f_past_std = np.std(f_past_safe, axis=0)
    if np.any(f_past_std < 1e-10):
        _logger.warning(
            f"get_transition_params: Some factors have near-zero variance (min_std={f_past_std.min():.2e}). "
            "This may cause numerical instability in transition matrix estimation."
        )
    
    XtX = f_past_safe.T @ f_past_safe
    Xty = f_past_safe.T @ f_next_safe

    # Ridge from 1e-6 up to 1e-4, growing with log10 of the condition number.
    try:
        cond_num = np.linalg.cond(XtX)
        ridge_scale = max(1e-6, min(1e-4, 1e-6 * (1 + np.log10(max(cond_num, 1)))))
    except (np.linalg.LinAlgError, ValueError):
        ridge_scale = 1e-6

    ridge = ridge_scale * np.eye(XtX.shape[0], dtype=XtX.dtype)

    try:
        A_f = (np.linalg.pinv(XtX + ridge) @ Xty).T
        if not np.all(np.isfinite(A_f)):
            raise ValueError("pinv produced non-finite values")
    except (np.linalg.LinAlgError, ValueError) as e:
        _logger.warning(
            f"get_transition_params: pinv failed ({type(e).__name__}), using lstsq fallback. "
            "This may indicate numerical instability in factor estimation."
        )
        try:
            A_t, _, rank, _ = np.linalg.lstsq(XtX + ridge, Xty, rcond=None)
            A_f = A_t.T
            if rank < XtX.shape[0]:
                _logger.warning(
                    f"get_transition_params: lstsq solution has rank {rank} < {XtX.shape[0]}. "
                    "Matrix may be rank-deficient."
                )
        except np.linalg.LinAlgError as e2:
            _logger.error(
                f"get_transition_params: Both pinv and lstsq failed. Using identity matrix as fallback. "
                f"Errors: {type(e).__name__}, {type(e2).__name__}"
            )
            A_f = np.eye(f_past_safe.shape[1], dtype=f_past_safe.dtype)
    
    Phi, _, _ = get_idio(eps_t, bool_no_miss)
    
    x_t = np.vstack((f_t.T, eps_t.T))
    A = np.vstack((
        np.hstack((A_f, np.zeros((A_f.shape[0], eps_t.shape[1])))),  # VAR factors
        np.hstack((np.zeros((eps_t.shape[1], A_f.shape[1])), Phi))  # AR(1) idio
    ))
    
    w_t = x_t[:, 1:] - A @ x_t[:, :-1]
    Q = np.diag(np.diag(np.cov(w_t)))
    mu_0 = np.mean(x_t, axis=1)
    Sigma_0 = np.cov(x_t)
    Sigma_0[:A_f.shape[1], A_f.shape[1]:] = 0
    Sigma_0[A_f.shape[1]:, :A_f.shape[1]] = 0
    Sigma_0[A_f.shape[1]:, A_f.shape[1]:] = np.diag(np.diag(Sigma_0[A_f.shape[1]:, A_f.shape[1]:]))
    return A, Q, mu_0, Sigma_0, x_t


def forecast_ar1_factors(
    Z_last: np.ndarray,
    A: np.ndarray,
    horizon: int,
    dtype: type = DEFAULT_DTYPE
) -> np.ndarray:
    """Forecast ``f_h = A f_{h-1}`` from ``f_0 = Z_last`` for ``h = 1..horizon``.

    Used by DFM and DDFM.

    Parameters
    ----------
    Z_last : np.ndarray
        Last state, (m,) with m >= 1.
    A : np.ndarray
        Transition matrix, (m, m).
    horizon : int
        Number of steps, >= 1.
    dtype : type, default DEFAULT_DTYPE
        Output dtype.

    Returns
    -------
    np.ndarray
        Forecasts, (horizon, m).

    Raises
    ------
    DataValidationError
        If the shapes or ``horizon`` are invalid, or an input or the forecast
        is not finite.
    NumericalError
        If the computation fails otherwise.
    """
    validate_ndarray_ndim(Z_last, "Z_last", 1)
    validate_ndarray_ndim(A, "A", 2)
    
    m = Z_last.shape[0]
    if m < 1:
        raise DataValidationError(f"Z_last must have at least 1 factor, got m={m}")
    
    if horizon < 1:
        raise DataValidationError(f"horizon must be >= 1, got {horizon}")
    
    validate_no_nan_inf(Z_last, name="Z_last")
    validate_no_nan_inf(A, name="transition matrix A")

    if A.shape != (m, m):
        raise DataValidationError(
            f"A (transition matrix) must have shape ({m}, {m}), got {A.shape}"
        )
    
    try:
        Z_forecast = np.zeros((horizon, m), dtype=dtype)
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        validate_no_nan_inf(Z_forecast, name="forecasted factors")
        return Z_forecast
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
        if isinstance(e, DataValidationError):  # from validate_no_nan_inf (a ValueError)
            raise
        raise NumericalError(
            f"Forecast computation failed: {e}",
            details=(
                f"Z_last shape: {Z_last.shape}, A shape: {A.shape}, "
                f"horizon={horizon}. Check: (1) Matrix dimensions, "
                f"(2) Numerical stability, (3) Input validity."
            )
        ) from e


def estimate_idio_dynamics(
    residuals: np.ndarray,
    missing_mask: np.ndarray,
    min_obs: int = DEFAULT_MIN_OBS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonal AR(1) coefficients and innovation variances of each residual series.

    Uses the rows observed together with the previous row (the ``get_idio``
    selection). With fewer than ``min_obs`` of them the coefficient is 0 and
    the variance that of the observed residuals; coefficients are clipped to
    ``+-VAR_STABILITY_THRESHOLD``.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals of the observation equation (T x N).
    missing_mask : np.ndarray
        True where data is missing (T x N).
    min_obs : int, default DEFAULT_MIN_OBS
        Minimum number of usable rows.

    Returns
    -------
    A_eps, Q_eps : np.ndarray
        Diagonal AR(1) coefficients and innovation variances (N x N).
    """
    T, N = residuals.shape
    A_eps = np.zeros((N, N))
    Q_eps = np.zeros((N, N))

    for j in range(N):
        valid = ~missing_mask[:, j]
        to_select = np.hstack((np.array([False]), valid[:-1] & valid[1:]))

        if np.sum(to_select) < min_obs:
            _logger.warning(
                f"Insufficient observations ({np.sum(to_select)}) for idio AR(1) "
                f"estimation for series {j}. Using zero AR(1) coefficient."
            )
            A_eps[j, j] = DEFAULT_ZERO_VALUE
            if np.sum(valid) > 0:
                Q_eps[j, j] = compute_var_safe(residuals[valid, j], ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)
            else:
                Q_eps[j, j] = MIN_DIAGONAL_VARIANCE
        else:
            this_eps = residuals[to_select, j]
            # phi = Cov(eps_t, eps_{t-1}) / Var(eps_{t-1}) over consecutive selected values.
            if len(this_eps) >= 2:
                cov_matrix = compute_cov_safe(np.vstack([this_eps[1:], this_eps[:-1]]), rowvar=True, pairwise_complete=False)
                cov_eps = cov_matrix[0, 1]
                var_eps = compute_var_safe(this_eps[:-1], ddof=0, min_variance=MIN_FACTOR_VARIANCE)
                if var_eps > MIN_FACTOR_VARIANCE:
                    A_eps[j, j] = cov_eps / var_eps
                else:
                    A_eps[j, j] = DEFAULT_ZERO_VALUE
            else:
                A_eps[j, j] = DEFAULT_ZERO_VALUE
            
            if abs(A_eps[j, j]) >= VAR_STABILITY_THRESHOLD:
                sign = np.sign(A_eps[j, j])
                A_eps[j, j] = sign * VAR_STABILITY_THRESHOLD
                _logger.debug(
                    f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                )
            
            if len(this_eps) >= 2:
                eps_t = this_eps[1:]
                eps_t_1 = this_eps[:-1]
                residuals_ar = eps_t - A_eps[j, j] * eps_t_1
                Q_eps[j, j] = compute_var_safe(residuals_ar, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)
            else:
                Q_eps[j, j] = compute_var_safe(this_eps, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE)

    return A_eps, Q_eps


def _split_smoothed_covariances(
    V_smooth: np.ndarray,
    V_smooth_lag: Optional[np.ndarray],
    n_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Covariances of the current rows and of the lagged rows, float64.

    ``V_smooth`` with ``n_rows + 1`` rows is the whole smoothed sequence (rows
    ``1:`` current, rows ``:-1`` lagged); otherwise it holds the current rows
    and ``V_smooth_lag`` the lagged ones (``V_smooth`` itself if None).
    """
    V = np.asarray(V_smooth, dtype=np.float64)
    if V.ndim == 3 and V.shape[0] == n_rows + 1 and V_smooth_lag is None:
        return V[1:], V[:-1]
    V_lag = V if V_smooth_lag is None else np.asarray(V_smooth_lag, dtype=np.float64)
    return V, V_lag


def _diag_sum(V: np.ndarray) -> np.ndarray:
    """Diagonal of the sum over time of a (T x n x n) or (n x n) covariance array."""
    V = np.asarray(V, dtype=np.float64)
    if V.ndim == 3:
        return np.einsum('tii->i', V)
    return np.diag(V).copy()


def _cov_sum(V: np.ndarray) -> np.ndarray:
    """Sum over time of a (T x a x b) covariance array, or the (a x b) array itself."""
    V = np.asarray(V, dtype=np.float64)
    return V.sum(axis=0) if V.ndim == 3 else V


def _var_from_smoothed_moments(
    y: np.ndarray,
    x: np.ndarray,
    V_smooth: np.ndarray,
    V_smooth_lag: Optional[np.ndarray],
    VVsmooth: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact EM update of a VAR from smoothed moments (float64, no caps).

    ``S11 = y'y + sum V``, ``S00 = x'x + sum V_lag``, ``S10 = y'x + sum VV``;
    ``A = S10 S00^-1`` (Cholesky solve) and ``Q = (S11 - A S10') / n`` with
    ``n`` the number of rows of ``y``, symmetrized. A singular ``S00`` raises
    ``NumericalError``.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n_rows = y.shape[0]
    V_cur, V_lag = _split_smoothed_covariances(V_smooth, V_smooth_lag, n_rows)
    S11 = y.T @ y + _cov_sum(V_cur)
    S00 = x.T @ x + _cov_sum(V_lag)
    S10 = y.T @ x
    if VVsmooth is not None:
        VV = np.asarray(VVsmooth, dtype=np.float64)
        if VV.ndim == 3 and VV.shape[0] == n_rows + 1:
            VV = VV[1:]
        S10 = S10 + _cov_sum(VV)
    S00 = 0.5 * (S00 + S00.T)
    try:
        factor = linalg.cho_factor(S00, lower=True, check_finite=True)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise NumericalError(f"VAR update: the lagged moment matrix is not positive definite ({e}).") from e
    d = np.abs(np.diag(factor[0]))
    if d.size and d.min() ** 2 <= S00.shape[0] * np.finfo(np.float64).eps * d.max() ** 2:
        raise NumericalError(
            f"VAR update: the lagged moment matrix is singular to working precision "
            f"(Cholesky diagonal ratio {d.min() / d.max():.2e})."
        )
    A = linalg.cho_solve(factor, S10.T).T
    Q = (S11 - A @ S10.T) / n_rows
    Q = 0.5 * (Q + Q.T)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(Q))):
        raise NumericalError("VAR update is not finite.")
    return A, Q


def estimate_var(
    y: np.ndarray,
    x: np.ndarray,
    V_smooth: Optional[np.ndarray] = None,
    V_smooth_lag: Optional[np.ndarray] = None,
    VVsmooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    min_variance: float = MIN_EIGENVALUE,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """VAR of ``y`` on ``x``: coefficients ``A`` (m x p) and covariance ``Q`` (m x m).

    With ``V_smooth`` (smoothed expectations mode, the EM M-step) this is the
    exact update ``A = S10 S00^-1``, ``Q = (S11 - A S10') / n`` in float64,
    with no regularization, caps or floors; a singular ``S00`` raises
    ``NumericalError``. Otherwise (raw data mode) ``A`` is ridge-regularized
    OLS and ``Q`` the stabilized residual covariance, cast to ``dtype``; fewer
    than 2 rows give ``DEFAULT_TRANSITION_COEF * I`` and
    ``DEFAULT_PROCESS_NOISE * I``.

    Parameters
    ----------
    y : np.ndarray
        Current values, or smoothed means ``E[z_t]`` (n x m).
    x : np.ndarray
        Lagged values, or smoothed means ``E[z_{t-1}]`` (n x p).
    V_smooth : np.ndarray, optional
        Smoothed covariances of the rows of ``y`` (n x m x m), or of the whole
        sequence (n + 1 rows: ``1:`` for ``y``, ``:-1`` for ``x``).
    V_smooth_lag : np.ndarray, optional
        Smoothed covariances of the rows of ``x`` (n x p x p); needed when
        ``V_smooth`` has n rows and ``p != m``.
    VVsmooth : np.ndarray, optional
        Lag-one covariances ``Cov(y_t, x_t | Y)`` (n x m x p, or n + 1 rows
        with row 0 dropped).
    regularization : float, default DEFAULT_REGULARIZATION
        Ridge of the raw-data OLS.
    min_variance : float, default MIN_EIGENVALUE
        Eigenvalue floor of the raw-data ``Q``.
    dtype : type, default np.float32
        Output dtype of the raw data mode.
    """
    if V_smooth is not None:
        return _var_from_smoothed_moments(y, x, V_smooth, V_smooth_lag, VVsmooth)

    T = y.shape[0]
    m = y.shape[1]
    p = x.shape[1]

    if T < 2:
        # A is m x p: np.eye, since create_scaled_identity is square only.
        A = np.eye(m, p, dtype=dtype) * DEFAULT_TRANSITION_COEF
        Q = create_scaled_identity(m, DEFAULT_PROCESS_NOISE, dtype=dtype)
        return A, Q

    A = solve_regularized_ols(x, y, regularization=regularization, dtype=dtype).T  # (m x p)

    residuals = y - x @ A.T
    if m == 1:
        var_val = compute_var_safe(residuals.flatten(), ddof=0, min_variance=min_variance)
        Q = np.atleast_2d(var_val)
    else:
        # Pairwise-complete covariances still work when few rows are complete.
        Q = compute_cov_safe(residuals.T, rowvar=True, pairwise_complete=True, min_eigenval=min_variance)

    Q = stabilize_innovation_covariance(Q, min_eigenval=min_variance, min_floor=MIN_Q_FLOOR, dtype=dtype)

    return A.astype(dtype), Q.astype(dtype)


def _ar1_from_moment_sums(S11: np.ndarray, S00: np.ndarray, S10: np.ndarray, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """``a = S10 / S00`` and ``q = max((S11 - a S10) / n, 0)`` per state, with the finiteness checks."""
    if np.any(~np.isfinite(S00)) or np.any(S00 <= 0.0):
        raise NumericalError(
            "AR(1) update: the lagged second moment is not positive and finite "
            f"(min {np.min(S00) if S00.size else float('nan'):.3e})."
        )
    A_diag = S10 / S00
    Q_diag = np.maximum((S11 - A_diag * S10) / n_rows, 0.0)
    if not (np.all(np.isfinite(A_diag)) and np.all(np.isfinite(Q_diag))):
        raise NumericalError("AR(1) update is not finite.")
    return A_diag, Q_diag


def estimate_ar1_from_diagonals(
    y: np.ndarray,
    x: np.ndarray,
    v_cur: np.ndarray,
    v_lag: np.ndarray,
    vv: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """The smoothed-expectations AR(1) update of :func:`estimate_ar1` from per-row diagonals.

    ``y`` and ``x`` are the smoothed means ``E[z_t]`` and ``E[z_{t-1}]`` (n x k);
    ``v_cur``, ``v_lag`` and ``vv`` (n x k) hold the diagonals of
    ``Var[z_t]``, ``Var[z_{t-1}]`` and ``Cov(z_t, z_{t-1})``. Same result as
    ``estimate_ar1`` on the full (n x k x k) covariances, without building them.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    # einsum over the rows reproduces ``_diag_sum``'s ``einsum('tii->i')`` bit for bit.
    S11 = np.sum(y * y, axis=0) + np.einsum('ti->i', np.asarray(v_cur, dtype=np.float64))
    S00 = np.sum(x * x, axis=0) + np.einsum('ti->i', np.asarray(v_lag, dtype=np.float64))
    S10 = np.sum(y * x, axis=0) + np.einsum('ti->i', np.asarray(vv, dtype=np.float64))
    return _ar1_from_moment_sums(S11, S00, S10, y.shape[0])


def estimate_ar1(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    V_smooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    min_variance: float = MIN_EIGENVALUE,
    default_ar_coef: float = DEFAULT_TRANSITION_COEF,
    default_noise: float = DEFAULT_PROCESS_NOISE,
    dtype: type = np.float32,
    V_smooth_lag: Optional[np.ndarray] = None,
    VVsmooth: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """AR(1) coefficient and innovation variance of each column of ``y`` on ``x``.

    Smoothed expectations mode (``V_smooth`` given) is the exact EM update of
    diagonal AR(1) states, in float64 with no regularization::

        a_i = (sum y_t x_t + sum VV_t)_ii / (sum x_t^2 + sum V_lag_t)_ii
        q_i = max(((sum y_t^2 + sum V_t)_ii - a_i (sum y_t x_t + sum VV_t)_ii) / n, 0)

    with ``n`` the number of rows of ``y``; a non-positive lagged moment raises
    ``NumericalError``. Raw data mode: per-series ridge OLS on the finite
    pairs, falling back to ``default_ar_coef``/``default_noise``, then clipped
    to ``+-VAR_STABILITY_THRESHOLD`` and ``[min_variance, MAX_EIGENVALUE]``.

    Parameters
    ----------
    y : np.ndarray
        Current values, or smoothed means ``E[z_t]`` (n x k).
    x : np.ndarray, optional
        Lagged values, or smoothed means ``E[z_{t-1}]`` (n x k); None uses
        ``y[:-1]`` (and ``y[1:]`` as ``y``).
    V_smooth : np.ndarray, optional
        Smoothed covariances of the rows of ``y`` (then ``V_smooth_lag`` gives
        those of ``x``), or of the whole sequence with one more row (rows
        ``1:`` for ``y``, ``:-1`` for ``x``).
    regularization : float, default DEFAULT_REGULARIZATION
        Raw data mode only, as the next four.
    min_variance : float, default MIN_EIGENVALUE
    default_ar_coef : float, default DEFAULT_TRANSITION_COEF
    default_noise : float, default DEFAULT_PROCESS_NOISE
    dtype : type, default np.float32
    V_smooth_lag : np.ndarray, optional
        Smoothed covariances of the rows of ``x``.
    VVsmooth : np.ndarray, optional
        Lag-one covariances ``Cov(y_t, x_t | Y)`` of the rows of ``y`` (or of
        the whole sequence with one more row; row 0 is then dropped).

    Returns
    -------
    A_diag, Q_diag : np.ndarray
        AR(1) coefficients and innovation variances (k,).
    """
    if V_smooth is not None:
        y = np.asarray(y, dtype=np.float64)
        if x is None:
            x = y[:-1]
            y = y[1:]
        x = np.asarray(x, dtype=np.float64)
        n_rows = y.shape[0]
        V_cur, V_lag = _split_smoothed_covariances(V_smooth, V_smooth_lag, n_rows)
        S11 = np.sum(y * y, axis=0) + _diag_sum(V_cur)
        S00 = np.sum(x * x, axis=0) + _diag_sum(V_lag)
        S10 = np.sum(y * x, axis=0)
        if VVsmooth is not None:
            VV = np.asarray(VVsmooth, dtype=np.float64)
            if VV.shape[0] == n_rows + 1:
                VV = VV[1:]
            S10 = S10 + _diag_sum(VV)
        return _ar1_from_moment_sums(S11, S00, S10, n_rows)

    if x is None:
        x = y[:-1]
        y = y[1:]

    T, n = y.shape

    if T < 2:
        A_diag = np.full(n, default_ar_coef, dtype=dtype)
        Q_diag = np.full(n, default_noise, dtype=dtype)
        return A_diag, Q_diag

    A_diag = np.zeros(n, dtype=dtype)
    Q_diag = np.zeros(n, dtype=dtype)

    for i in range(n):
        y_i = y[:, i]
        x_i = x[:, i].reshape(-1, 1)

        valid = np.isfinite(y_i) & np.isfinite(x_i.squeeze())
        if np.sum(valid) < 2:
            A_diag[i] = default_ar_coef
            Q_diag[i] = default_noise
            continue

        y_i_clean = y_i[valid]
        x_i_clean = x_i[valid]

        try:
            XTX = x_i_clean.T @ x_i_clean
            XTX_reg = XTX + create_scaled_identity(1, regularization, dtype=dtype)
            A_i = solve(XTX_reg, x_i_clean.T @ y_i_clean, overwrite_a=False, overwrite_b=False, check_finite=False).item()

            residuals = y_i_clean - x_i_clean.squeeze() * A_i
            Q_i = compute_var_safe(residuals, ddof=0, min_variance=min_variance, default_variance=default_noise)

            A_diag[i] = A_i if np.isfinite(A_i) else default_ar_coef
            Q_diag[i] = max(Q_i, min_variance) if np.isfinite(Q_i) else default_noise
        except (np.linalg.LinAlgError, ValueError):
            A_diag[i] = default_ar_coef
            Q_diag[i] = default_noise

    # Finite and bounded before the caller's cast.
    A_diag = np.where(np.isfinite(A_diag), A_diag, default_ar_coef)
    Q_diag = np.where(np.isfinite(Q_diag), Q_diag, default_noise)
    A_diag = np.clip(A_diag, -VAR_STABILITY_THRESHOLD, VAR_STABILITY_THRESHOLD)
    Q_diag = np.clip(Q_diag, min_variance, MAX_EIGENVALUE)

    return A_diag, Q_diag


def estimate_arp_ols(
    factors: np.ndarray,
    order: int,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Ridge OLS AR(p) fit of each factor on its own lags.

    ``f_i[t] = sum_l a_i[l] f_i[t-l] + e_i[t]``. Coefficients whose absolute
    sum reaches ``VAR_STABILITY_THRESHOLD`` are rescaled to sum to 0.9 times
    it. With fewer than ``order + 1`` rows, or when a fit fails, the
    coefficients are 0 and ``B_diag`` is ``DEFAULT_PROCESS_NOISE``.

    Parameters
    ----------
    factors : np.ndarray
        Factors (T x r).
    order : int
        AR order p.
    regularization : float, default DEFAULT_REGULARIZATION
        Ridge of the OLS.
    dtype : type, default np.float32
        Output dtype.

    Returns
    -------
    ar_coeffs : np.ndarray
        AR coefficients (r x p).
    B_diag : np.ndarray
        Residual standard deviation of each factor's fit (r,), the innovation
        scale used as the diagonal of B.
    """
    T, r = factors.shape

    if T < order + 1:
        ar_coeffs = np.zeros((r, order), dtype=dtype)
        B_diag = np.ones(r, dtype=dtype) * DEFAULT_PROCESS_NOISE
        return ar_coeffs, B_diag

    ar_coeffs = np.zeros((r, order), dtype=dtype)
    B_diag = np.zeros(r, dtype=dtype)

    for i in range(r):
        f_i = factors[:, i]

        # Row t of X is [f_i[t-1], ..., f_i[t-p]], the target f_i[t].
        X = np.zeros((T - order, order), dtype=dtype)
        y = f_i[order:]
        for lag in range(order):
            X[:, lag] = f_i[order - lag - 1:T - lag - 1]

        try:
            ar_i = solve_regularized_ols(X, y, regularization=regularization, dtype=dtype)

            # sum |a_l| < 1 is sufficient for stationarity (all roots of
            # 1 - sum a_l z^l outside the unit circle).
            if np.sum(np.abs(ar_i)) >= VAR_STABILITY_THRESHOLD:
                ar_i = ar_i * (VAR_STABILITY_THRESHOLD * 0.9 / np.sum(np.abs(ar_i)))

            ar_coeffs[i, :] = ar_i

            residuals = y - X @ ar_i
            B_diag[i] = np.sqrt(np.maximum(
                compute_var_safe(residuals, ddof=0, min_variance=MIN_DIAGONAL_VARIANCE),
                MIN_DIAGONAL_VARIANCE
            ))
        except (np.linalg.LinAlgError, ValueError) as e:  # the solve's failures (pinv not converging)
            _logger.warning(f"AR({order}) estimation failed for factor {i}: {e}. Using zeros.")
            ar_coeffs[i, :] = 0.0
            B_diag[i] = DEFAULT_PROCESS_NOISE

    return ar_coeffs, B_diag


def estimate_constrained_ols(
    y: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    q: np.ndarray,
    V_smooth: Optional[np.ndarray] = None,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> np.ndarray:
    """Ridge OLS of ``y`` on ``X`` subject to ``R beta = q``.

    ``beta = b - M^-1 R' (R M^-1 R' + reg I)^-1 (R b - q)`` with ``b`` the
    unconstrained solution and ``M`` the regularized moment matrix (solved,
    never inverted); if a solve fails ``b`` is returned.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (T x n), or the moments ``E[y_t f_t']`` in the
        smoothed expectations mode.
    X : np.ndarray
        Regressors (T x p), or the moments ``E[f_t f_t']``.
    R : np.ndarray
        Constraint matrix (n_constraints x p).
    q : np.ndarray
        Constraint vector (n_constraints,).
    V_smooth : np.ndarray, optional
        Selects the smoothed expectations mode (its values are not read).
    regularization : float, default DEFAULT_REGULARIZATION
        Ridge of the moment matrix and of ``R M^-1 R'``.
    dtype : type, default np.float32
        Output dtype.

    Returns
    -------
    np.ndarray
        Constrained coefficients (p,).
    """
    if V_smooth is not None:
        # X is already the moment matrix.
        beta_unconstrained = solve_regularized_ols(X, y, regularization=regularization, use_XTX=False, dtype=dtype)
        try:
            X_reg = X + create_scaled_identity(X.shape[0], regularization, dtype=dtype)
            constraint_term = R @ beta_unconstrained - q
            R_X_reg_inv = solve(X_reg, R.T, overwrite_a=False, overwrite_b=False, check_finite=False)
            R_X_inv_RT = R @ R_X_reg_inv
            R_X_inv_RT_reg = R_X_inv_RT + create_scaled_identity(R_X_inv_RT.shape[0], regularization, dtype=dtype)
            temp2 = solve(R_X_inv_RT_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
            beta_constrained = beta_unconstrained - R_X_reg_inv @ temp2
        except (np.linalg.LinAlgError, ValueError):
            beta_constrained = beta_unconstrained
    else:
        beta_unconstrained = solve_regularized_ols(X, y, regularization=regularization, dtype=dtype)
        try:
            XTX = X.T @ X
            # A larger ridge for an ill-conditioned X'X.
            try:
                eigenvals_XTX = np.linalg.eigvalsh(XTX)
                max_eig = np.max(np.abs(eigenvals_XTX))
                min_eig = np.min(np.abs(eigenvals_XTX[eigenvals_XTX != 0]))
                cond_num = max_eig / max(min_eig, 1e-12)
                if cond_num > 1e10:
                    adaptive_reg = regularization * (cond_num / 1e10)
                    XTX_reg = XTX + create_scaled_identity(XTX.shape[0], adaptive_reg, dtype=dtype)
                else:
                    XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype=dtype)
            except (np.linalg.LinAlgError, ValueError):  # eigvalsh failed, or every eigenvalue is 0
                XTX_reg = XTX + create_scaled_identity(XTX.shape[0], regularization, dtype=dtype)

            constraint_term = R @ beta_unconstrained - q
            # lstsq handles a matrix that is too ill-conditioned for solve.
            try:
                R_XTX_inv = solve(XTX_reg, R.T, overwrite_a=False, overwrite_b=False, check_finite=False)
            except np.linalg.LinAlgError:
                R_XTX_inv, _, _, _ = linalg.lstsq(XTX_reg, R.T, lapack_driver='gelsy', cond=1e-10)

            R_XTX_inv_RT = R @ R_XTX_inv
            R_XTX_inv_RT_reg = R_XTX_inv_RT + create_scaled_identity(R_XTX_inv_RT.shape[0], regularization, dtype=dtype)
            try:
                temp2 = solve(R_XTX_inv_RT_reg, constraint_term, overwrite_a=False, overwrite_b=False, check_finite=False)
            except np.linalg.LinAlgError:
                temp2, _, _, _ = linalg.lstsq(R_XTX_inv_RT_reg, constraint_term, lapack_driver='gelsy', cond=1e-10)

            beta_constrained = beta_unconstrained - R_XTX_inv @ temp2
        except (np.linalg.LinAlgError, ValueError) as e:
            _logger.debug(f"Constrained OLS failed, using unconstrained solution: {e}")
            beta_constrained = beta_unconstrained

    return beta_constrained.astype(dtype)


def estimate_variance(
    residuals: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    EZ: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    V_smooth: Optional[np.ndarray] = None,
    min_variance: float = MIN_EIGENVALUE,
    default_variance: float = DEFAULT_PROCESS_NOISE,
    dtype: type = np.float32
) -> np.ndarray:
    """Diagonal noise covariance (N x N) from residuals or smoothed moments.

    With ``residuals`` (raw data mode): the variance of each series' finite
    values, ``default_variance`` when it is below ``min_variance``, not finite
    or from at most one value (or when ``T <= 1``). Otherwise
    ``mean_t diag((y_t - C z_t)(y_t - C z_t)' + C V_t C')`` with
    ``z_t = EZ[t + 1]`` and ``V_t = V_smooth[t + 1]``, floored at
    ``min_variance`` (``default_variance`` where not finite).

    Parameters
    ----------
    residuals : np.ndarray, optional
        Residuals (T x N); selects the raw data mode.
    X : np.ndarray, optional
        Data (T x N), smoothed expectations mode.
    EZ : np.ndarray, optional
        Smoothed state means (T+1 x m), smoothed expectations mode.
    C : np.ndarray, optional
        Observation matrix (N x m), smoothed expectations mode.
    V_smooth : np.ndarray, optional
        Smoothed state covariances (T+1 x m x m); without them the term
        ``C V_t C'`` is left out.
    min_variance : float, default MIN_EIGENVALUE
    default_variance : float, default DEFAULT_PROCESS_NOISE
    dtype : type, default np.float32
        Output dtype.

    Raises
    ------
    ConfigurationError
        If neither ``residuals`` nor all of ``X``, ``EZ`` and ``C`` are given.
    """
    if residuals is not None:
        T, N = residuals.shape

        if T <= 1:
            R = create_scaled_identity(N, default_variance, dtype=dtype)
            return R

        var_res = np.array([
            compute_var_safe(
                residuals[:, i][np.isfinite(residuals[:, i])],
                ddof=0,
                min_variance=min_variance,
                default_variance=default_variance
            )
            if np.sum(np.isfinite(residuals[:, i])) > 1 else default_variance
            for i in range(N)
        ], dtype=dtype)
        var_res = np.where(np.isfinite(var_res), var_res, default_variance)
        var_res = np.maximum(var_res, min_variance)

        R = np.diag(var_res)
    else:
        if X is None or EZ is None or C is None:
            raise ConfigurationError(
                "X, EZ, and C are required for smoothed expectations mode",
                details=f"Missing required parameters: X={X is None}, EZ={EZ is None}, C={C is None}"
            )

        T, N = X.shape
        # E[(y_t - C z_t)(y_t - C z_t)'] = e_t e_t' + C V_t C' with e_t = y_t - C E[z_t].
        R = np.zeros((N, N), dtype=dtype)
        for t in range(T):
            z_t = EZ[t+1, :]
            residual = X[t, :] - C @ z_t
            R += np.outer(residual, residual)
            if V_smooth is not None:
                V_t = V_smooth[t+1]
                R += C @ V_t @ C.T

        R = R / T

        R_diag = np.diag(R)
        R_diag = np.maximum(R_diag, min_variance)
        R_diag = np.where(np.isfinite(R_diag), R_diag, default_variance)
        R = np.diag(R_diag)

    return R.astype(dtype)


def compute_initial_covariance_from_transition(
    A: np.ndarray,
    Q: np.ndarray,
    regularization: float = DEFAULT_REGULARIZATION,
    dtype: type = np.float32
) -> np.ndarray:
    """Stationary covariance ``V_0 = A V_0 A' + Q`` (n x n) of the state.

    Solves ``(I - A kron A + regularization I) vec(V_0) = vec(Q)``; if the
    solve fails, ``Q`` is returned (with a warning).

    Parameters
    ----------
    A : np.ndarray
        Transition matrix (n x n).
    Q : np.ndarray
        Process noise covariance (n x n).
    regularization : float, default DEFAULT_REGULARIZATION
        Ridge of the linear system.
    dtype : type, default np.float32
        Output dtype.
    """
    n = A.shape[0]
    try:
        kron_AA = np.kron(A, A)
        eye_kron = create_scaled_identity(n ** 2, DEFAULT_IDENTITY_SCALE, dtype=dtype)
        V_0_flat = solve(
            eye_kron - kron_AA + create_scaled_identity(n ** 2, regularization, dtype=dtype),
            Q.flatten(),
            overwrite_a=False,
            overwrite_b=False,
            check_finite=False
        )
        V_0 = V_0_flat.reshape(n, n).astype(dtype)
        return V_0
    except (np.linalg.LinAlgError, ValueError):
        _logger.warning(
            f"Initial covariance computation failed for transition matrix of size {n}x{n}. "
            f"Using process noise Q as fallback."
        )
        return Q.copy().astype(dtype)


__all__ = [
    'clip_ar',
    'apply_ar_clipping',
    'estimate_idio_dynamics',
    'get_idio',
    'get_transition_params',
    'estimate_var',
    'estimate_ar1',
    'estimate_ar1_from_diagonals',
    'estimate_constrained_ols',
    'estimate_variance',
    'compute_initial_covariance_from_transition',
    'stabilize_innovation_covariance',
    'forecast_ar1_factors',
]


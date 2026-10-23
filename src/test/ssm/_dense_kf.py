"""Dense Kalman filter and Durbin-Koopman smoother: the test oracle of ``ssm.kalman``.

A verbatim copy of the dense implementation of
:class:`dfm_python.ssm.kalman.DFMKalmanFilter` as it was before the
structure-aware path (Joseph-form update, dense ``m x m`` products in every
step), kept as :class:`DenseKalmanFilter` so the structured filter can be
checked against it. The original description follows.

:class:`DenseKalmanFilter` is a self-contained float64 Kalman filter and
Durbin-Koopman smoother for the linear Gaussian state-space model::

    x_{t+1} = A x_t + w_t,   w_t ~ N(0, Q)
    y_t     = C x_t + v_t,   v_t ~ N(0, R)

Timing: ``x_0 ~ N(Z_0, V_0)`` is the state AT data row 0, so the prediction
for row 0 is ``(Z_0, V_0)`` and the first transition is applied before row 1.

Missing data are handled per element. At row ``t`` only the observed entries
``o_t`` of ``y_t`` (not NaN and not masked) enter the update, with the rows
``C[o_t]`` and the block ``R[o_t, o_t]``; a row with nothing observed is a pure
prediction step. A row is never dropped because one of its entries is missing.

Forward pass (one multivariate update per row, which works for any ``R``):

- innovation ``e_t = y_o - C_o a_t`` and ``S_t = C_o P_t C_o' + R_oo``,
  factored by Cholesky (a failed factorization raises :class:`NumericalError`);
- gain ``K_t = P_t C_o' S_t^-1`` by ``cho_solve``;
- Joseph-form covariance ``(I - K C_o) P (I - K C_o)' + K R_oo K'``,
  symmetrized;
- the exact Gaussian log-likelihood of the observed entries, accumulated in the
  same pass: ``-0.5 (p_t log 2 pi + log|S_t| + e' S_t^-1 e)`` per row, 0 for
  an empty row.

Backward pass: the Durbin-Koopman ``(r_t, N_t)`` recursion (Durbin and Koopman,
*Time Series Analysis by State Space Methods*, 2nd ed., section 4.4), which
never inverts the predicted covariance and so also works when it is singular
(deterministic lag-shift states with zero process noise). With
``L_t = A (I - K_t C_o)`` (``L_t = A`` on an empty row)::

    r_{t-1} = C_o' S_t^-1 e_t + L_t' r_t,   N_{t-1} = C_o' S_t^-1 C_o + L_t' N_t L_t
    E[x_t | Y]   = a_t + P_t r_{t-1}
    Var[x_t | Y] = P_t - P_t N_{t-1} P_t
    Cov(x_t, x_{t+1} | Y) = P_t L_t' (I - N_t P_{t+1})          (DK eq. 4.69)

The lag-one covariances are returned in the layout the M-step reads:
``VVsmooth[t] = Cov(x_t, x_{t-1} | Y)`` for ``t >= 1`` and ``VVsmooth[0] = 0``.

There is no covariance regularization, eigenvalue capping or state clipping:
the moments and the log-likelihood describe exactly the model given to
:meth:`DenseKalmanFilter.update_parameters`. Non-finite results raise
:class:`~dfm_python.utils.errors.NumericalError`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from dfm_python.config.types import FloatArray
from dfm_python.logger import get_logger
from dfm_python.utils.errors import ModelNotInitializedError, NumericalError

_logger = get_logger(__name__)

_LOG2PI = float(np.log(2.0 * np.pi))

#: Relative tolerance on the most negative eigenvalue of V_0 before it is reported as indefinite.
_PSD_RTOL = 1e-8


def _sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _observed_mask(observations) -> Tuple[np.ndarray, np.ndarray]:
    """``(values, observed)`` of a (T x N) array or masked array.

    Observed means neither masked nor NaN.
    """
    if isinstance(observations, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(observations)
        values = np.asarray(observations.filled(np.nan), dtype=np.float64)
    else:
        values = np.asarray(observations, dtype=np.float64)
        mask = np.zeros(values.shape, dtype=bool)
    if values.ndim == 1:
        values = values[np.newaxis, :]
        mask = np.atleast_2d(mask)
    if values.ndim != 2:
        raise ValueError(f"observations must be a (T x N) array, got shape {values.shape}")
    observed = ~(mask | np.isnan(values))
    return values, observed


class DenseKalmanFilter:
    """Kalman filter and Durbin-Koopman smoother of the DFM.

    See the module docstring. Parameters are set with
    :meth:`update_parameters` (or all six constructor arguments) and are only
    validated (shapes, finiteness) and symmetrized, never regularized.
    """

    def __init__(
        self,
        transition_matrices: Optional[FloatArray] = None,
        observation_matrices: Optional[FloatArray] = None,
        transition_covariance: Optional[FloatArray] = None,
        observation_covariance: Optional[FloatArray] = None,
        initial_state_mean: Optional[FloatArray] = None,
        initial_state_covariance: Optional[FloatArray] = None,
    ) -> None:
        self._initialized = False
        # Smoothed states of the last EM E-step (set by functional.em; cleared when parameters change).
        self._cached_smoothed_factors: Optional[np.ndarray] = None
        if all(p is not None for p in (
            transition_matrices, observation_matrices, transition_covariance,
            observation_covariance, initial_state_mean, initial_state_covariance,
        )):
            self.update_parameters(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
            )

    def update_parameters(
        self,
        transition_matrices: FloatArray,
        observation_matrices: FloatArray,
        transition_covariance: FloatArray,
        observation_covariance: FloatArray,
        initial_state_mean: FloatArray,
        initial_state_covariance: FloatArray,
    ) -> None:
        """Set ``(A, C, Q, R, Z_0, V_0)``.

        The arrays are copied to float64. ``Q``, ``R`` and ``V_0`` are
        symmetrized; nothing else is changed.

        Raises
        ------
        ValueError
            If the shapes are inconsistent.
        NumericalError
            If a parameter is not finite. An indefinite ``V_0`` is logged and
            used unchanged; the filter raises if an innovation covariance is
            then not positive definite.
        """
        A = np.array(transition_matrices, dtype=np.float64)
        C = np.array(observation_matrices, dtype=np.float64)
        Q = np.array(transition_covariance, dtype=np.float64)
        R = np.array(observation_covariance, dtype=np.float64)
        Z_0 = np.array(initial_state_mean, dtype=np.float64).reshape(-1)
        V_0 = np.array(initial_state_covariance, dtype=np.float64)

        m = A.shape[0] if A.ndim == 2 else -1
        N = C.shape[0] if C.ndim == 2 else -1
        expected = {
            "A (transition)": (A, (m, m)),
            "C (observation)": (C, (N, m)),
            "Q (process noise)": (Q, (m, m)),
            "R (observation noise)": (R, (N, N)),
            "Z_0 (initial state mean)": (Z_0, (m,)),
            "V_0 (initial state cov)": (V_0, (m, m)),
        }
        for name, (arr, shape) in expected.items():
            if m < 0 or N < 0 or arr.shape != shape:
                raise ValueError(f"DFMKalmanFilter: {name} has shape {arr.shape}, expected {shape}")
        bad = [name for name, (arr, _) in expected.items() if not np.all(np.isfinite(arr))]
        if bad:
            raise NumericalError(
                f"DFMKalmanFilter: non-finite parameters: {', '.join(bad)}.",
                details="The Kalman filter needs finite A, C, Q, R, Z_0 and V_0.",
            )
        Q, R, V_0 = _sym(Q), _sym(R), _sym(V_0)
        if m > 0:
            eig = np.linalg.eigvalsh(V_0)
            if eig[0] < -_PSD_RTOL * max(1.0, float(np.max(np.abs(eig)))):
                # Used as given (no repair): the filter only needs each S_t to be
                # positive definite, and raises if one is not.
                _logger.warning(
                    "DFMKalmanFilter: V_0 is not positive semi-definite (smallest eigenvalue %.3e); "
                    "it is used unchanged.", eig[0],
                )

        self.transition_matrices = A
        self.observation_matrices = C
        self.transition_covariance = Q
        self.observation_covariance = R
        self.initial_state_mean = Z_0
        self.initial_state_covariance = V_0
        self._initialized = True
        self._cached_smoothed_factors = None

    def _require_parameters(self) -> None:
        if not self._initialized:
            raise ModelNotInitializedError(
                "DFMKalmanFilter parameters not initialized. Call update_parameters() first."
            )

    def _forward(self, observations, *, keep_filtered: bool, keep_gains: bool) -> dict:
        """Filter every row: predicted (and filtered) moments, loglik, smoother inputs."""
        self._require_parameters()
        A = self.transition_matrices
        C = self.observation_matrices
        Q = self.transition_covariance
        R = self.observation_covariance
        Y, observed = _observed_mask(observations)
        T, N_y = Y.shape
        m = A.shape[0]
        if N_y != C.shape[0]:
            raise ValueError(f"observations have {N_y} columns, the model has {C.shape[0]} series")

        pm = np.empty((T, m))
        pP = np.empty((T, m, m))
        fm = np.empty((T, m)) if keep_filtered else None
        fP = np.empty((T, m, m)) if keep_filtered else None
        gains = [None] * T if keep_gains else None
        loglik = 0.0
        I_m = np.eye(m)
        a = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()
        # The per-row matrices are small (m x m); multithreaded BLAS spends
        # its time synchronizing on them (a 4-core run was 28x slower than
        # a single-threaded one at m = 106), so the loops run on one thread.
        with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", invalid="ignore"):
            for t in range(T):
                if t > 0:
                    a = A @ a
                    P = _sym(A @ P @ A.T + Q)
                pm[t] = a
                pP[t] = P
                o = observed[t]
                if o.any():
                    idx = np.flatnonzero(o)
                    Co = C[idx]
                    Ro = R[np.ix_(idx, idx)]
                    e = Y[t, idx] - Co @ a
                    PCt = P @ Co.T
                    S = _sym(Co @ PCt + Ro)
                    if not np.all(np.isfinite(S)):
                        raise NumericalError(
                            f"Kalman filter: the innovation covariance is not finite at row {t}.",
                            details="The predicted state covariance overflowed; the parameters are not usable.",
                        )
                    try:
                        chol = linalg.cho_factor(S, lower=True, check_finite=False)
                    except linalg.LinAlgError as exc:
                        raise NumericalError(
                            f"Kalman filter: the innovation covariance is not positive definite at row {t}.",
                            details=f"Cholesky of S_t failed: {exc}",
                        ) from exc
                    u = linalg.solve_triangular(chol[0], e, lower=True, check_finite=False)
                    loglik += -0.5 * (idx.size * _LOG2PI + 2.0 * float(np.sum(np.log(np.diag(chol[0])))) + float(u @ u))
                    K = linalg.cho_solve(chol, PCt.T, check_finite=False).T  # P C_o' S^-1
                    a = a + K @ e
                    IKC = I_m - K @ Co
                    P = _sym(IKC @ P @ IKC.T + K @ Ro @ K.T)
                    if keep_gains:
                        gains[t] = (idx, e, chol, K)
                if keep_filtered:
                    fm[t] = a
                    fP[t] = P
        if not (np.isfinite(loglik) and np.all(np.isfinite(a)) and np.all(np.isfinite(P))):
            raise NumericalError(
                "Kalman filter: the filtered moments or the log-likelihood are not finite.",
                details=f"loglik={loglik!r}; the parameters make the filter diverge.",
            )
        return dict(pm=pm, pP=pP, fm=fm, fP=fP, gains=gains, loglik=float(loglik))

    def filter(self, observations: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Run the Kalman filter.

        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) with NaN for missing entries, or a masked array.

        Returns
        -------
        filtered_state_means : np.ndarray
            ``E[x_t | y_0..y_t]``, (T x m).
        filtered_state_covariances : np.ndarray
            ``Var[x_t | y_0..y_t]``, (T x m x m).
        """
        out = self._forward(observations, keep_filtered=True, keep_gains=False)
        return out["fm"], out["fP"]

    def loglikelihood(self, observations: FloatArray) -> float:
        """Exact Gaussian log-likelihood of the observed entries of ``observations``."""
        return self._forward(observations, keep_filtered=False, keep_gains=False)["loglik"]

    def filter_and_smooth(
        self,
        observations: FloatArray,
        compute_loglik: bool = True,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
        """Smoothed moments, lag-one covariances and the exact log-likelihood.

        Parameters
        ----------
        observations : np.ndarray
            Observations (T x N) with NaN for missing entries, or a masked array.
        compute_loglik : bool, default True
            Ignored: the log-likelihood comes out of the forward pass at no
            extra cost and is always returned.

        Returns
        -------
        smoothed_state_means : np.ndarray
            ``E[x_t | Y]``, (T x m).
        smoothed_state_covariances : np.ndarray
            ``Var[x_t | Y]``, (T x m x m).
        sigma_pair_smooth : np.ndarray
            ``Cov(x_t, x_{t-1} | Y)`` at index ``t >= 1``; index 0 is zero. (T x m x m).
        loglik : float
            Exact log-likelihood of the observed entries.

        Raises
        ------
        NumericalError
            If the filter or the smoother produce non-finite values, or an
            innovation covariance is not positive definite.
        """
        del compute_loglik  # always computed
        fwd = self._forward(observations, keep_filtered=False, keep_gains=True)
        A = self.transition_matrices
        C = self.observation_matrices
        pm, pP, gains = fwd["pm"], fwd["pP"], fwd["gains"]
        T, m = pm.shape
        I_m = np.eye(m)
        sm = np.empty((T, m))
        sP = np.empty((T, m, m))
        pair = np.zeros((T, m, m))
        r = np.zeros(m)
        N = np.zeros((m, m))
        with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", invalid="ignore"):
            for t in range(T - 1, -1, -1):
                N_next = N  # N_t: information from rows t+1 .. T-1
                step = gains[t]
                if step is None:
                    Lt = A
                    r = A.T @ r
                    N = _sym(A.T @ N @ A)
                else:
                    idx, e, chol, K = step
                    Co = C[idx]
                    Lt = A - (A @ K) @ Co
                    W = linalg.solve_triangular(chol[0], Co, lower=True, check_finite=False)  # chol^-1 C_o
                    r = Co.T @ linalg.cho_solve(chol, e, check_finite=False) + Lt.T @ r
                    N = _sym(W.T @ W + Lt.T @ N @ Lt)
                P = pP[t]
                sm[t] = pm[t] + P @ r
                sP[t] = _sym(P - P @ N @ P)
                if t < T - 1:
                    pair[t + 1] = (P @ Lt.T @ (I_m - N_next @ pP[t + 1])).T
        if not (np.all(np.isfinite(sm)) and np.all(np.isfinite(sP)) and np.all(np.isfinite(pair))):
            raise NumericalError(
                "Kalman smoother: the smoothed moments are not finite.",
                details="The parameters make the smoother diverge.",
            )
        return sm, sP, pair, fwd["loglik"]

    def init_streaming(self) -> None:
        """Initialize the streaming state to ``(Z_0, V_0)``.

        Call once before the first :meth:`filter_step`. The streaming state is
        a filtered state: each :meth:`filter_step` first applies the
        transition.
        """
        self._require_parameters()
        self._stream_state = self.initial_state_mean.copy()
        self._stream_cov = self.initial_state_covariance.copy()

    def filter_step(
        self, y_t: FloatArray, *, reencode: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One predict (and, by default, update) step from the streaming state.

        Parameters
        ----------
        y_t : np.ndarray
            One observation vector, shape (N,); NaN entries are missing and
            skipped. Ignored when ``reencode=False``.
        reencode : bool, default True
            If True, update the predicted state with ``y_t`` (Joseph form). If
            False, only apply the transition.

        Returns
        -------
        state : np.ndarray
            Updated (or predicted) state mean, shape (m,).
        cov : np.ndarray
            Updated (or predicted) state covariance, shape (m, m).
        """
        if not hasattr(self, "_stream_state"):
            raise ModelNotInitializedError(
                "Streaming state not initialized. Call init_streaming() first."
            )
        A = self.transition_matrices
        C = self.observation_matrices
        R = self.observation_covariance
        a = A @ self._stream_state
        P = _sym(A @ self._stream_cov @ A.T + self.transition_covariance)
        if reencode:
            y = np.asarray(y_t, dtype=np.float64).ravel()
            idx = np.flatnonzero(~np.isnan(y))
            if idx.size:
                Co = C[idx]
                Ro = R[np.ix_(idx, idx)]
                PCt = P @ Co.T
                try:
                    chol = linalg.cho_factor(_sym(Co @ PCt + Ro), lower=True)
                except (linalg.LinAlgError, ValueError) as exc:
                    raise NumericalError("Kalman filter_step: the innovation covariance is not positive definite.",
                                         details=str(exc)) from exc
                K = linalg.cho_solve(chol, PCt.T).T
                a = a + K @ (y[idx] - Co @ a)
                IKC = np.eye(A.shape[0]) - K @ Co
                P = _sym(IKC @ P @ IKC.T + K @ Ro @ K.T)
        self._stream_state = a
        self._stream_cov = P
        return a.copy(), P.copy()

    @property
    def streaming_state(self) -> Optional[np.ndarray]:
        """Current streaming state mean, or None if not initialized."""
        return getattr(self, "_stream_state", None)

    @property
    def streaming_cov(self) -> Optional[np.ndarray]:
        """Current streaming state covariance, or None if not initialized."""
        return getattr(self, "_stream_cov", None)

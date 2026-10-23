"""Kalman filter and smoother for the DFM E-step.

:class:`DFMKalmanFilter` is a self-contained float64 Kalman filter and
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
  factored by Cholesky ``S_t = L L'`` (a failed factorization raises
  :class:`NumericalError`);
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

Two implementations of these recursions, equal up to rounding:

- **Structured**, used when :meth:`DFMKalmanFilter.update_parameters` finds
  the DFM sparsity (:class:`_Transition`, :class:`_ObsStructure`): ``A`` is a
  diagonal (AR(1) coefficients), shift rows with one nonzero (factor lags,
  tent chains) and a few dense rows (the factor VAR), and ``C`` a few dense
  columns (the loadings) plus columns with one nonzero (the idiosyncratic
  states). Products with ``A`` and ``C_o`` are scalings, gathers and small
  dense products, and ``A P A'`` costs no ``m x m`` product. The update uses
  the triangular inverse of ``L``: ``W = L^-1 C_o P``, ``P_f = P - W'W`` (a
  symmetric rank-``p_t`` product, no Joseph form) and
  ``a_f = a + W' L^-1 e``. The smoother works from the filtered moments: with
  ``G = A P_f`` and ``H = N_t G``, ``E[x_t | Y] = a_f + P_f A' r_t``,
  ``Var[x_t | Y] = P_f - G'H`` and ``Cov(x_{t+1}, x_t | Y) = G - P_{t+1} H``;
  ``N_{t-1} = C_o' S^-1 C_o + (I - K C_o)' A'N_tA (I - K C_o)`` is expanded
  so that only products with ``K`` are dense. A row then costs three dense
  products in the filter (``L^-1 C_o P``, ``W'W``, ``L^-1``) and five in the
  smoother, against about twenty in the dense path. ``P_f - G'H`` subtracts
  terms of the size of ``P_f``, where ``P - P N P`` subtracts terms of the
  size of ``P``, so the structured smoothed variances lose no accuracy when
  ``R`` is small; the filtered ones can (``P - W'W`` cancels in the
  directions the data pin down, which the Joseph form avoids).
- **Dense** (any other matrices): the products as written above, the gain
  ``K_t = P_t C_o' S_t^-1`` by ``cho_solve`` and the Joseph-form covariance
  ``(I - K C_o) P (I - K C_o)' + K R_oo K'``, symmetrized.

There is no covariance regularization, eigenvalue capping or state clipping:
the moments and the log-likelihood describe exactly the model given to
:meth:`DFMKalmanFilter.update_parameters`. Non-finite results raise
:class:`~dfm_python.utils.errors.NumericalError`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.linalg import lapack
from threadpoolctl import threadpool_limits

from ..config.types import FloatArray
from ..logger import get_logger
from ..utils.errors import ModelNotInitializedError, NumericalError

_logger = get_logger(__name__)

_LOG2PI = float(np.log(2.0 * np.pi))

#: Relative tolerance on the most negative eigenvalue of V_0 before it is reported as indefinite.
_PSD_RTOL = 1e-8


def _sym(M: np.ndarray) -> np.ndarray:
    """``(M + M') / 2``."""
    out = M.T.copy()
    out += M
    out *= 0.5
    return out


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


def _index(idx: np.ndarray) -> Tuple[object, object]:
    """``(rows, block)``: indexers of the rows ``idx`` and of the block ``idx x idx``.

    Slices (indexing then makes views) when ``idx`` is an increasing contiguous range.
    """
    if idx.size and np.all(np.diff(idx) == 1):
        s = slice(int(idx[0]), int(idx[-1]) + 1)
        return s, (s, s)
    return idx, np.ix_(idx, idx)


class _Transition:
    """``A`` as a diagonal plus a few special rows: ``A = diag(d) + E_S' A[S]``.

    The special rows ``S`` are the rows of ``A`` with a nonzero off the
    diagonal; ``d`` is the diagonal of the other rows (AR(1) coefficients) and
    zero on ``S``. A special row with one nonzero (a factor lag or tent-chain
    shift) whose column no earlier such row holds is a gather: its row of
    ``A X`` is ``g * X[c]``. The other special rows ``S_B`` (factor VAR rows)
    come first in ``S`` and give ``B = A[S_B, T]``, ``T`` the columns they
    touch, so a product with ``A`` costs a scaling pass, a gather and a
    ``|S_B| x |T|`` product. ``U`` are the columns the special rows touch
    (``T`` and ``c``); :meth:`sandwich` and :meth:`sandwich_t` are
    ``(d d') * X`` outside the rows ``S`` (``U``), so they are exactly
    symmetric when ``X`` is.
    """

    __slots__ = ("d", "dd", "B", "n_B", "T", "g", "c", "S", "SS", "U", "UU", "T_U", "c_U")

    def __init__(self, A: np.ndarray, S: np.ndarray) -> None:
        nz = A[S] != 0
        c = np.argmax(nz, axis=1)
        single = np.flatnonzero(np.count_nonzero(nz, axis=1) == 1)
        gather = np.zeros(S.size, dtype=bool)
        gather[single[np.unique(c[single], return_index=True)[1]]] = True  # one row per column
        d = np.diag(A).copy()
        d[S] = 0.0
        self.d = d[:, None]
        self.dd = np.outer(d, d)
        T = np.flatnonzero(nz[~gather].any(axis=0))
        self.B = A[np.ix_(S[~gather], T)]
        self.n_B = self.B.shape[0]
        self.g = A[S[gather], c[gather]][:, None]
        U = np.union1d(T, c[gather])
        self.T, self.T_U = _index(T)[0], _index(np.searchsorted(U, T))[0]
        self.c, self.c_U = _index(c[gather])[0], _index(np.searchsorted(U, c[gather]))[0]
        self.S, self.SS = _index(np.concatenate([S[~gather], S[gather]]))
        self.U, self.UU = _index(U)

    @classmethod
    def detect(cls, A: np.ndarray) -> Optional[_Transition]:
        """The representation of ``A``, or None when ``B`` is over half the size of ``A``."""
        off = A != 0
        np.fill_diagonal(off, False)
        op = cls(A, np.flatnonzero(off.any(axis=1)))
        return None if 2 * op.B.size > A.size else op

    def _rows(self, X: np.ndarray) -> np.ndarray:
        """``A[S] @ X`` for a 2-D ``X`` (rows in the order of ``S``)."""
        out = np.empty((self.n_B + self.g.shape[0], X.shape[1]))
        np.matmul(self.B, X[self.T], out=out[: self.n_B])
        np.multiply(self.g, X[self.c], out=out[self.n_B:])
        return out

    def left(self, X: np.ndarray) -> np.ndarray:
        """``A @ X`` for a 2-D ``X``."""
        out = self.d * X
        out[self.S] = self._rows(X)
        return out

    def _add_rows_t(self, out: np.ndarray, Y: np.ndarray, T, c) -> None:
        """``out += A[S]' @ Y`` for ``Y`` with one row per special row.

        ``T`` and ``c`` index the rows of ``out`` that stand for the columns ``T`` and ``c`` of ``A``.
        """
        out[T] += self.B.T @ Y[: self.n_B]
        out[c] += self.g * Y[self.n_B:]

    def left_t(self, X: np.ndarray) -> np.ndarray:
        """``A' @ X`` for a 2-D ``X``."""
        out = self.d * X
        self._add_rows_t(out, X[self.S], self.T, self.c)
        return out

    def sandwich(self, P: np.ndarray) -> np.ndarray:
        """``A P A'`` for a symmetric ``P``."""
        Z = self.left(self._rows(P).T).T  # rows S of A P A'
        out = self.dd * P
        out[self.S] = Z
        out[:, self.S] = Z.T
        out[self.SS] = _sym(Z[:, self.S])
        return out

    def sandwich_t(self, N: np.ndarray) -> np.ndarray:
        """``A' N A`` for a symmetric ``N``."""
        Y = self.d[self.U] * N[self.U]  # rows U of A' N
        self._add_rows_t(Y, N[self.S], self.T_U, self.c_U)
        Z = self.left_t(Y.T).T  # rows U of A' N A
        out = self.dd * N
        out[self.U] = Z
        out[:, self.U] = Z.T
        out[self.UU] = _sym(Z[:, self.U])
        return out


class _ObsStructure:
    """``C`` as dense columns ``cols`` plus entries ``(er, ec, ev)`` in columns with one nonzero.

    The entries are sorted by row; each of their columns appears once.
    ``unit`` says every entry is 1 (the idiosyncratic states of clock-frequency
    series), so products skip the scaling.
    """

    __slots__ = ("m", "C_cols", "cols", "er", "ec", "ev", "unit", "one_each")

    def __init__(self, C: np.ndarray, cols: np.ndarray, single: np.ndarray) -> None:
        self.m = C.shape[1]
        self.C_cols = C[:, cols]
        self.cols = cols
        self.er, self.ec = np.nonzero((C != 0) & single)
        self.ev = C[self.er, self.ec]
        self.unit = bool(np.all(self.ev == 1.0))
        self.one_each = np.array_equal(self.er, np.arange(C.shape[0]))

    @classmethod
    def detect(cls, C: np.ndarray) -> Optional[_ObsStructure]:
        """The representation of ``C``, or None when over half its columns have two or more nonzeros."""
        count = np.count_nonzero(C, axis=0)
        cols = np.flatnonzero(count > 1)
        if 2 * cols.size > C.shape[1]:
            return None
        return cls(C, cols, count == 1)

    def plan(self, observed: np.ndarray) -> _ObsPlan:
        """The observed rows ``C_o`` of one data row (``observed`` is its boolean mask)."""
        idx = np.flatnonzero(observed)
        if self.one_each:  # every series has exactly one entry, so entry j is in row j of C_o
            sel, er, starts = idx, slice(None), None
        else:
            sel = observed[self.er]
            er = (np.cumsum(observed) - 1)[self.er[sel]]
            first = np.ones(er.size, dtype=bool)
            first[1:] = er[1:] != er[:-1]
            starts = None if first.all() else np.flatnonzero(first)
        return _ObsPlan(self, idx, er, starts, self.ec[sel], None if self.unit else self.ev[sel])


class _ObsPlan:
    """Products with ``C_o`` (the observed rows of ``C``) for one data row; see :class:`_ObsStructure`.

    ``er`` are the entries' rows within ``C_o`` (``slice(None)`` when entry
    ``j`` is in row ``j``), ``ec`` and ``ev`` their columns and values
    (``ev`` None for unit entries). With rows holding several entries,
    ``starts`` are the first entry of each row with entries and ``rows`` those
    rows (``slice(None)`` when every row has entries).
    """

    __slots__ = ("idx", "C_dense", "cols", "er", "starts", "rows", "ec", "ev", "m")

    def __init__(self, obs: _ObsStructure, idx, er, starts, ec, ev) -> None:
        self.idx = idx
        self.C_dense = obs.C_cols[idx]
        self.cols = obs.cols
        self.m = obs.m
        self.er = er
        self.starts = starts
        rows = er if starts is None else er[starts]
        self.rows = slice(None) if isinstance(rows, slice) or rows.size == idx.size else rows
        if starts is None and not isinstance(er, slice) and isinstance(self.rows, slice):
            self.er = slice(None)
        self.ec = ec
        self.ev = None if ev is None else ev[:, None]

    def left(self, X: np.ndarray) -> np.ndarray:
        """``C_o @ X`` for a 2-D ``X`` with ``m`` rows."""
        out = self.C_dense @ X[self.cols]
        part = X[self.ec]
        if self.ev is not None:
            part *= self.ev
        if self.starts is not None:
            part = np.add.reduceat(part, self.starts, axis=0)
        out[self.rows] += part
        return out

    def right(self, X: np.ndarray) -> np.ndarray:
        """``X @ C_o'`` for a 2-D ``X`` with ``m`` columns."""
        out = X[:, self.cols] @ self.C_dense.T
        part = X[:, self.ec]
        if self.ev is not None:
            part *= self.ev.T
        if self.starts is not None:
            part = np.add.reduceat(part, self.starts, axis=1)
        out[:, self.rows] += part
        return out

    def left_t(self, X: np.ndarray) -> np.ndarray:
        """``C_o' @ X`` for a 2-D ``X`` with one row per observed series."""
        out = np.zeros((self.m, X.shape[1]))
        out[self.ec] = X[self.er] if self.ev is None else self.ev * X[self.er]
        out[self.cols] += self.C_dense.T @ X
        return out


class _Structure:
    """The DFM sparsity of ``(A, C, R)`` used by the structured path, or None from :meth:`detect`."""

    __slots__ = ("A", "C", "R_diag")

    def __init__(self, A: _Transition, C: _ObsStructure, R_diag: Optional[np.ndarray]) -> None:
        self.A = A
        self.C = C
        self.R_diag = R_diag

    @classmethod
    def detect(cls, A, C, R) -> Optional[_Structure]:
        A_op = _Transition.detect(A)
        C_op = _ObsStructure.detect(C)
        if A_op is None or C_op is None:
            return None
        R_diag = np.diag(R).copy() if np.count_nonzero(R - np.diag(np.diag(R))) == 0 else None
        return cls(A_op, C_op, R_diag)

    def predict(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """``A P A' + Q`` (exactly symmetric when ``P`` is)."""
        out = self.A.sandwich(P)
        out += Q
        return out

    def innovation(self, plan: _ObsPlan, P: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(C_o P, S)`` with ``S = C_o P C_o' + R_oo``; only the lower triangle of ``S`` is used."""
        CP = plan.left(P)
        S = plan.right(CP)
        if self.R_diag is not None:
            S.flat[:: plan.idx.size + 1] += self.R_diag[plan.idx]
        else:
            S += R[np.ix_(plan.idx, plan.idx)]
        return CP, S


def _cholesky_inverse(S: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(L, L^-1)`` for ``S = L L'`` from the lower triangle of ``S`` (both lower triangular).

    Raises NumericalError (the messages of the dense path) when ``S`` is not
    finite or not positive definite.
    """
    if not np.all(np.isfinite(S)):
        raise NumericalError(
            f"Kalman filter: the innovation covariance is not finite at row {t}.",
            details="The predicted state covariance overflowed; the parameters are not usable.",
        )
    L, info = lapack.dpotrf(S, lower=1, clean=1)
    if info != 0:
        raise NumericalError(
            f"Kalman filter: the innovation covariance is not positive definite at row {t}.",
            details=f"Cholesky of S_t failed: LAPACK potrf info={info}",
        )
    L_inv, _ = lapack.dtrtri(L, lower=1)
    return L, L_inv


def _structured_update(a, P, CP, e, L_inv) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(a_f, P_f, W, u)``: ``W = L^-1 C_o P``, ``u = L^-1 e``, ``a_f = a + W'u``, ``P_f = P - W'W``.

    ``W'W`` is computed as a symmetric rank-k product, so ``P_f`` is exactly
    symmetric when ``P`` is.
    """
    W = L_inv @ CP
    u = L_inv @ e
    P_f = W.T @ W
    np.subtract(P, P_f, out=P_f)
    return a + W.T @ u, P_f, W, u


class DFMKalmanFilter:
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
        """Set ``(A, C, Q, R, Z_0, V_0)`` and detect their structure.

        The arrays are copied to float64. ``Q``, ``R`` and ``V_0`` are
        symmetrized; nothing else is changed. The structured path is used when
        ``A`` and ``C`` have the DFM sparsity (module docstring), the
        dense path otherwise.

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
        self._structure = _Structure.detect(A, C, R) if m > 0 else None
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
        Y, observed = _observed_mask(observations)
        if Y.shape[1] != self.observation_matrices.shape[0]:
            raise ValueError(
                f"observations have {Y.shape[1]} columns, the model has {self.observation_matrices.shape[0]} series"
            )
        # The per-row matrices are small (m x m); multithreaded BLAS spends
        # its time synchronizing on them (a 4-core run was 28x slower than
        # a single-threaded one at m = 106), so the loops run on one thread.
        with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", invalid="ignore"):
            if self._structure is None:
                return self._forward_dense(Y, observed, keep_filtered=keep_filtered, keep_gains=keep_gains)
            return self._forward_structured(Y, observed, keep_filtered=keep_filtered, keep_gains=keep_gains)

    @staticmethod
    def _check_innovation(S: np.ndarray, t: int) -> tuple:
        """Cholesky factor ``cho_factor(S, lower=True)``; raises NumericalError if ``S`` is unusable."""
        if not np.all(np.isfinite(S)):
            raise NumericalError(
                f"Kalman filter: the innovation covariance is not finite at row {t}.",
                details="The predicted state covariance overflowed; the parameters are not usable.",
            )
        try:
            return linalg.cho_factor(S, lower=True, check_finite=False)
        except linalg.LinAlgError as exc:
            raise NumericalError(
                f"Kalman filter: the innovation covariance is not positive definite at row {t}.",
                details=f"Cholesky of S_t failed: {exc}",
            ) from exc

    @staticmethod
    def _check_filtered(loglik: float, a: np.ndarray, P: np.ndarray) -> None:
        if not (np.isfinite(loglik) and np.all(np.isfinite(a)) and np.all(np.isfinite(P))):
            raise NumericalError(
                "Kalman filter: the filtered moments or the log-likelihood are not finite.",
                details=f"loglik={loglik!r}; the parameters make the filter diverge.",
            )

    def _forward_dense(self, Y, observed, *, keep_filtered: bool, keep_gains: bool) -> dict:
        """Dense forward pass with the Joseph-form update; ``gains[t] = (idx, e, chol, K)``."""
        A = self.transition_matrices
        C = self.observation_matrices
        Q = self.transition_covariance
        R = self.observation_covariance
        T = Y.shape[0]
        m = A.shape[0]
        pm = np.empty((T, m))
        pP = np.empty((T, m, m))
        fm = np.empty((T, m)) if keep_filtered else None
        fP = np.empty((T, m, m)) if keep_filtered else None
        gains = [None] * T if keep_gains else None
        loglik = 0.0
        I_m = np.eye(m)
        a = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()
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
                chol = self._check_innovation(_sym(Co @ PCt + Ro), t)
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
        self._check_filtered(loglik, a, P)
        return dict(pm=pm, pP=pP, fm=fm, fP=fP, gains=gains, loglik=float(loglik))

    def _forward_structured(self, Y, observed, *, keep_filtered: bool, keep_gains: bool) -> dict:
        """Structured forward pass (``P_f = P - W'W``).

        ``gains[t] = (plan, S^-1 e, S^-1, K')`` with ``K' = S^-1 C_o P``, from
        ``L^-1`` (``S = L L'``): ``S^-1 = L^-T L^-1`` and ``K' = L^-T W``.
        """
        st = self._structure
        A = self.transition_matrices
        Q = self.transition_covariance
        R = self.observation_covariance
        C = self.observation_matrices
        T = Y.shape[0]
        m = A.shape[0]
        pP = np.empty((T, m, m)) if keep_gains else None
        fm = np.empty((T, m)) if keep_filtered else None
        fP = np.empty((T, m, m)) if keep_filtered else None
        gains = [None] * T if keep_gains else None
        loglik = 0.0
        a = self.initial_state_mean.copy()
        P = self.initial_state_covariance.copy()
        for t in range(T):
            if t > 0:
                a = A @ a
                P = st.predict(P, Q)
            if keep_gains:
                pP[t] = P
            o = observed[t]
            if o.any():
                plan = st.C.plan(o)
                e = Y[t, plan.idx] - (C @ a)[plan.idx]
                CP, S = st.innovation(plan, P, R)
                L, L_inv = _cholesky_inverse(S, t)
                a, P, W, u = _structured_update(a, P, CP, e, L_inv)
                loglik += -0.5 * (plan.idx.size * _LOG2PI + 2.0 * float(np.sum(np.log(np.diag(L)))) + float(u @ u))
                if keep_gains:
                    gains[t] = (plan, L_inv.T @ u, L_inv.T @ L_inv, L_inv.T @ W)
            if keep_filtered:
                fm[t] = a
                fP[t] = P
        self._check_filtered(loglik, a, P)
        return dict(pm=None, pP=pP, fm=fm, fP=fP, gains=gains, loglik=float(loglik))

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
        structured = self._structure is not None
        fwd = self._forward(observations, keep_filtered=structured, keep_gains=True)
        with threadpool_limits(limits=1, user_api="blas"), np.errstate(over="ignore", invalid="ignore"):
            sm, sP, pair = self._smooth_structured(fwd) if structured else self._smooth_dense(fwd)
        if not (np.all(np.isfinite(sm)) and np.all(np.isfinite(sP)) and np.all(np.isfinite(pair))):
            raise NumericalError(
                "Kalman smoother: the smoothed moments are not finite.",
                details="The parameters make the smoother diverge.",
            )
        return sm, sP, pair, fwd["loglik"]

    def _smooth_dense(self, fwd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dense DK backward pass over the output of :meth:`_forward_dense`."""
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
        return sm, sP, pair

    def _smooth_structured(self, fwd: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Structured DK backward pass over the output of :meth:`_forward_structured`.

        At row ``t`` (``N``, ``r`` are ``N_t``, ``r_t``), with ``G = A P_f`` and
        ``H = N_t G``: ``E[x_t | Y] = a_f + P_f A' r_t``,
        ``Var[x_t | Y] = P_f - G' H``, ``Cov(x_{t+1}, x_t | Y) = G - P_{t+1} H``
        and ``N_{t-1} = A'N_tA + (Z + Z')`` with ``U = K' A'N_tA`` and
        ``Z = C_o' ((S^-1 + U K) C_o / 2 - U)``, which keeps ``N`` exactly
        symmetric. The smoothed and lag-one covariances overwrite ``pP`` and
        ``fP`` once those are no longer read.
        """
        st = self._structure
        A = self.transition_matrices
        C = self.observation_matrices
        pP, fm, fP, gains = fwd["pP"], fwd["fm"], fwd["fP"], fwd["gains"]
        T, m = fm.shape
        sm = np.empty((T, m))
        w = np.empty(C.shape[0])  # S^-1 (e - C_o P A' r_t) scattered to all series; r_{t-1} = A' r_t + C' w
        r = np.zeros(m)
        N = np.zeros((m, m))
        sP_next = None  # Var[x_{t+1} | Y], written to pP[t + 1] once P_{t+1} is used
        for t in range(T - 1, -1, -1):
            Pf = fP[t]
            G = st.A.left(Pf)
            H = N @ G
            if t < T - 1:
                np.subtract(G, pP[t + 1] @ H, out=fP[t + 1])  # Cov(x_{t+1}, x_t | Y)
                pP[t + 1] = sP_next
            GH = G.T @ H
            np.subtract(Pf, GH, out=GH)
            sP_next = _sym(GH)
            r_A = A.T @ r
            sm[t] = fm[t] + Pf @ r_A
            if t == 0:
                pP[0], fP[0] = sP_next, 0.0
                break
            N = st.A.sandwich_t(N)
            step = gains[t]
            if step is None:
                r = r_A
                continue
            plan, v, S_inv, Kt = step
            w[:] = 0.0
            w[plan.idx] = v - Kt @ r_A
            r = r_A + C.T @ w
            U = Kt @ N
            M = U @ Kt.T
            M += S_inv
            V = np.multiply(plan.left_t(M).T, 0.5, out=np.empty(U.shape))  # (S^-1 + U K) C_o / 2
            V -= U
            Z = plan.left_t(V)
            N += Z + Z.T
        return sm, pP, fP

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
            If True, update the predicted state with ``y_t``. If False, only
            apply the transition.

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
        st = self._structure
        A = self.transition_matrices
        C = self.observation_matrices
        R = self.observation_covariance
        a = A @ self._stream_state
        if st is None:
            P = _sym(A @ self._stream_cov @ A.T + self.transition_covariance)
        else:
            P = st.predict(self._stream_cov, self.transition_covariance)
        if reencode:
            y = np.asarray(y_t, dtype=np.float64).ravel()
            o = ~np.isnan(y)
            if o.any() and st is None:
                idx = np.flatnonzero(o)
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
            elif o.any():
                plan = st.C.plan(o)
                CP, S = st.innovation(plan, P, R)
                try:
                    L_inv = _cholesky_inverse(S, 0)[1]
                except NumericalError as exc:
                    raise NumericalError("Kalman filter_step: the innovation covariance is not positive definite.",
                                         details=exc.details) from exc
                a, P = _structured_update(a, P, CP, y[plan.idx] - (C @ a)[plan.idx], L_inv)[:2]
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

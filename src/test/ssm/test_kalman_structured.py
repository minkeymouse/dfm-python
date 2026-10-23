"""The structured path of ``DFMKalmanFilter`` against the dense oracle (``_dense_kf``).

Random DFM-shaped models (a factor companion block, AR(1) idiosyncratic
states of clock series and a tent chain per slower-frequency series; missing
values, all-NaN rows and a ragged edge) go through the structured path; the
log-likelihood, the smoothed means, covariances and lag-one covariances, the
filter and ``filter_step`` must equal the dense implementation to 1e-9
relative (1e-12 absolute near zero). Generic dense matrices take the dense
path and give the oracle's numbers exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.utils.errors import NumericalError

from ._dense_kf import DenseKalmanFilter

RTOL, ATOL = 1e-9, 1e-12
TENT = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


def _structured_model(seed: int, *, r: int = 2, p: int = 2, n_clock: int = 7, n_slow: int = 2,
                      T: int = 60, R_scale: float = 1e-4, loading_scale: float = 1.0):
    """A DFM in the engine's state layout and data simulated from it (NaN-gapped).

    States: ``r`` factors with ``max(p, 5)`` lags (companion), one AR(1) state
    per clock series, a 5-state tent chain per slower series. A slower series
    is observed every third row; rows 4 and 11 are all-NaN; the last three
    rows are a ragged edge.
    """
    rng = np.random.default_rng(seed)
    lags = max(p, TENT.size)
    k = r * lags
    m = k + n_clock + TENT.size * n_slow
    N = n_clock + n_slow
    A = np.zeros((m, m))
    for j in range(p):
        A[:r, j * r:(j + 1) * r] = rng.uniform(-0.4, 0.4, (r, r)) / p
    A[r:k, :k - r] = np.eye(k - r)
    Q = np.zeros((m, m))
    B = rng.standard_normal((r, r))
    Q[:r, :r] = B @ B.T + 0.5 * np.eye(r)
    C = np.zeros((N, m))
    C[:n_clock, :r] = loading_scale * rng.standard_normal((n_clock, r))
    for i in range(n_clock):
        s = k + i
        A[s, s] = rng.uniform(-0.9, 0.9)
        Q[s, s] = rng.uniform(0.1, 1.0)
        C[i, s] = 1.0
    for q in range(n_slow):
        i = n_clock + q
        s = k + n_clock + q * TENT.size
        lam = loading_scale * rng.standard_normal(r)
        for j, w in enumerate(TENT):
            C[i, j * r:(j + 1) * r] = w * lam
            C[i, s + j] = w
        A[s, s] = rng.uniform(-0.9, 0.9)
        A[s + 1:s + TENT.size, s:s + TENT.size - 1] = np.eye(TENT.size - 1)
        Q[s, s] = rng.uniform(0.1, 1.0)
    R = np.diag(R_scale * rng.uniform(0.5, 2.0, N))
    Z0 = rng.standard_normal(m)
    V = rng.standard_normal((m, m))
    V0 = V @ V.T / m + 0.1 * np.eye(m)

    x = Z0.copy()
    Y = np.empty((T, N))
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(m))
    for t in range(T):
        if t > 0:
            x = A @ x + Lq @ rng.standard_normal(m)
        Y[t] = C @ x + np.sqrt(np.diag(R)) * rng.standard_normal(N)
    Y[rng.random((T, N)) < 0.15] = np.nan
    Y[np.arange(T) % 3 != 2, n_clock:] = np.nan
    Y[[4, 11]] = np.nan
    Y[-3:, : n_clock // 2] = np.nan
    Y[-1, n_clock:] = np.nan
    return (A, C, Q, R, Z0, V0), Y


def _dense_model(seed: int, m: int = 6, N: int = 5, T: int = 40):
    rng = np.random.default_rng(seed)
    A = 0.3 * rng.standard_normal((m, m))
    C = rng.standard_normal((N, m))
    B = rng.standard_normal((m, m))
    Q = B @ B.T / m + 0.1 * np.eye(m)
    D = rng.standard_normal((N, N))
    R = D @ D.T / N + 0.1 * np.eye(N)
    Z0 = rng.standard_normal(m)
    V0 = np.eye(m)
    Y = rng.standard_normal((T, N))
    Y[rng.random((T, N)) < 0.2] = np.nan
    Y[3] = np.nan
    return (A, C, Q, R, Z0, V0), Y


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


MODELS = {
    "clock-only": dict(n_slow=0),
    "tent": dict(),
    "one-slow-series": dict(n_slow=1, r=1, p=1),
    "three-factors-p1": dict(r=3, p=1, n_clock=12),
    "large-loadings": dict(loading_scale=10.0),
    "noisy": dict(R_scale=0.5),
}


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("name", list(MODELS))
def test_smoother_matches_dense(name, seed):
    params, Y = _structured_model(seed, **MODELS[name])
    kf = DFMKalmanFilter(*params)
    assert kf._structure is not None
    sm, sP, pair, ll = kf.filter_and_smooth(Y)
    sm0, sP0, pair0, ll0 = DenseKalmanFilter(*params).filter_and_smooth(Y)
    assert ll == pytest.approx(ll0, rel=RTOL)
    _assert_close(sm, sm0)
    _assert_close(sP, sP0)
    _assert_close(pair, pair0)
    assert np.all(pair[0] == 0.0)


@pytest.mark.parametrize("name", ["tent", "large-loadings"])
def test_filter_and_loglik_match_dense(name):
    params, Y = _structured_model(3, **MODELS[name])
    kf, dense = DFMKalmanFilter(*params), DenseKalmanFilter(*params)
    fm, fP = kf.filter(Y)
    fm0, fP0 = dense.filter(Y)
    _assert_close(fm, fm0)
    _assert_close(fP, fP0)
    assert kf.loglikelihood(Y) == pytest.approx(dense.loglikelihood(Y), rel=RTOL)


def test_filter_step_matches_dense():
    params, Y = _structured_model(4)
    kf, dense = DFMKalmanFilter(*params), DenseKalmanFilter(*params)
    kf.init_streaming()
    dense.init_streaming()
    for t in range(Y.shape[0]):
        reencode = t % 7 != 5
        a, P = kf.filter_step(Y[t], reencode=reencode)
        a0, P0 = dense.filter_step(Y[t], reencode=reencode)
        _assert_close(a, a0)
        _assert_close(P, P0)


def test_ill_conditioned_initial_covariance():
    """``R = 1e-4`` (why the dense path uses the Joseph form) and a V_0 with eigenvalues from 1e-10 to 1e2.

    The absolute tolerance is 1e-11: here the dense oracle's own smoothed
    variances are off by 9e-13 from an extended-precision smoother (the
    structured ones by 4e-14), so 1e-12 would test the oracle's rounding.
    """
    params, Y = _structured_model(5, loading_scale=3.0)
    A, C, Q, R, Z0, _ = params
    rng = np.random.default_rng(5)
    U, _ = np.linalg.qr(rng.standard_normal((A.shape[0], A.shape[0])))
    V0 = (U * np.logspace(-10, 2, A.shape[0])) @ U.T
    params = (A, C, Q, R, Z0, V0)
    sm, sP, pair, ll = DFMKalmanFilter(*params).filter_and_smooth(Y)
    sm0, sP0, pair0, ll0 = DenseKalmanFilter(*params).filter_and_smooth(Y)
    assert ll == pytest.approx(ll0, rel=RTOL)
    for a, b in ((sm, sm0), (sP, sP0), (pair, pair0)):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=1e-11)


def _compare_all(params, Y):
    kf = DFMKalmanFilter(*params)
    assert kf._structure is not None
    out, ref = kf.filter_and_smooth(Y), DenseKalmanFilter(*params).filter_and_smooth(Y)
    assert out[3] == pytest.approx(ref[3], rel=RTOL, abs=1e-12)
    for a, b in zip(out[:3], ref[:3]):
        _assert_close(a, b)


def test_diagonal_transition_and_full_R():
    """``A`` diagonal (no special rows) and a non-diagonal ``R``."""
    rng = np.random.default_rng(8)
    N = 6
    m = 1 + N
    A = np.diag(rng.uniform(-0.8, 0.8, m))
    C = np.hstack([rng.standard_normal((N, 1)), np.eye(N)])
    Q = np.diag(rng.uniform(0.2, 1.0, m))
    D = 0.1 * rng.standard_normal((N, N))
    R = D @ D.T + 1e-3 * np.eye(N)
    Y = rng.standard_normal((30, N))
    Y[rng.random(Y.shape) < 0.2] = np.nan
    Y[7] = np.nan
    _compare_all((A, C, Q, R, rng.standard_normal(m), np.eye(m)), Y)


@pytest.mark.parametrize("rows", ["one", "all-nan", "first-nan"])
def test_edge_samples(rows):
    params, Y = _structured_model(9)
    if rows == "one":
        Y = Y[2:3]
    elif rows == "all-nan":
        Y = np.full((5, Y.shape[1]), np.nan)
    else:
        Y = Y.copy()
        Y[0] = np.nan
    _compare_all(params, Y)


def test_innovation_not_positive_definite_raises():
    params, Y = _structured_model(10)
    A, C, Q, R, Z0, V0 = params
    bad = (A, C, Q, -10.0 * np.eye(R.shape[0]), Z0, V0)
    kf = DFMKalmanFilter(*bad)
    assert kf._structure is not None
    with pytest.raises(NumericalError, match="not positive definite at row 0"):
        kf.filter_and_smooth(Y)
    kf.init_streaming()
    with pytest.raises(NumericalError, match="filter_step"):
        kf.filter_step(Y[0])


def test_masked_input_equals_nan_input():
    params, Y = _structured_model(6)
    kf = DFMKalmanFilter(*params)
    masked = np.ma.masked_invalid(np.where(np.isnan(Y), 0.0, Y))
    masked.mask = np.isnan(Y)
    for a, b in zip(kf.filter_and_smooth(Y)[:3], kf.filter_and_smooth(masked)[:3]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1])
def test_generic_matrices_take_the_dense_path(seed):
    params, Y = _dense_model(seed)
    kf = DFMKalmanFilter(*params)
    assert kf._structure is None
    for a, b in zip(kf.filter_and_smooth(Y), DenseKalmanFilter(*params).filter_and_smooth(Y)):
        np.testing.assert_array_equal(a, b)


def test_structure_follows_parameter_updates():
    params, Y = _structured_model(7)
    kf = DFMKalmanFilter(*params)
    assert kf._structure is not None
    dense, _ = _dense_model(7, m=params[0].shape[0], N=params[1].shape[0])
    kf.update_parameters(*dense)
    assert kf._structure is None
    kf.update_parameters(*params)
    assert kf._structure is not None
    sm, _, _, ll = kf.filter_and_smooth(Y)
    sm0, _, _, ll0 = DenseKalmanFilter(*params).filter_and_smooth(Y)
    assert ll == pytest.approx(ll0, rel=RTOL)
    _assert_close(sm, sm0)

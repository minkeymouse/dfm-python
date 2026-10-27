"""The BM14 state layout of the DFM and the initialization that builds it.

- ``idio_indicator`` is 1 for clock-frequency series and 0 for slower ones: a
  slower series carries only its tent chain (no AR(1) state).
- The chain loadings are the unnormalized tent weights; only a chain's head has
  an AR coefficient and process noise, its other states are pure shifts with
  zero ``Q``; the companion lag states of the factors have zero ``Q`` too.
- Initialization: ``R = MIN_OBSERVATION_NOISE`` on series with an
  idiosyncratic state; the chain head's variance comes from the residual
  variance (not from the overwritten R); the initial slower loadings satisfy the
  tent restriction to 1e-10; ``V_0`` is the exact stationary covariance, or the
  Schur-stationary plus diffuse covariance when ``A`` has a unit root.
- Save/load round-trips on the new layout, and a save written with the old
  layout is refused.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
from scipy import linalg

from dfm_python.config.constants import MIN_OBSERVATION_NOISE
from dfm_python.functional.state_layout import DFMStateLayout, bm14_idio_indicator
from dfm_python.models.dfm.initialization import (
    initial_state_covariance,
    initialize_parameters,
    tent_restricted_least_squares,
)
from dfm_python.models.dfm.tent import generate_R_mat
from dfm_python.numeric.builder import build_lag_matrix


def _quiet_fit(model, panel):
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(panel)
    finally:
        logging.disable(logging.NOTSET)


def _monthly_quarterly():
    from ..functional.test_em_oracle import _monthly_quarterly as make

    return make()


def _weekly_monthly():
    from ..functional.test_em_driver import _weekly_monthly as make

    return make()


# ------------------------------------------------------------------ layout


def test_layout_indices():
    layout = DFMStateLayout.build(r=[2, 1], p=1, L=3, n_series=6, n_clock=4, chain_length=3)
    assert layout.n_factor_states == 9 and layout.n_idio == 4 and layout.n_chain_states == 6
    assert layout.m == 19
    np.testing.assert_array_equal(layout.factor_offsets, [0, 6])
    np.testing.assert_array_equal(layout.factor_current(1), [6])
    np.testing.assert_array_equal(layout.factor_lags(0, 2), [0, 1, 2, 3])
    np.testing.assert_array_equal(layout.idio_states, [9, 10, 11, 12])
    np.testing.assert_array_equal(layout.chain_states(4), [13, 14, 15])
    np.testing.assert_array_equal(layout.chain_states(5), [16, 17, 18])
    np.testing.assert_array_equal(layout.chain_heads, [13, 16])
    np.testing.assert_array_equal(layout.ar1_states, [9, 10, 11, 12, 13, 16])
    np.testing.assert_array_equal(layout.idio_indicator, bm14_idio_indicator(4, 2))
    assert layout.is_bm14()
    with pytest.raises(IndexError):
        layout.chain_states(3)


@pytest.mark.parametrize("make,k", [(_monthly_quarterly, 5), (_weekly_monthly, 3)])
def test_fitted_state_has_the_bm14_layout(make, k):
    model = _quiet_fit(*make())
    engine = model.engine_
    state = engine.training_state
    N = state.C.shape[0]
    n_clock = int(state.n_clock_freq)
    n_slower = N - n_clock
    np.testing.assert_array_equal(state.idio_indicator, np.r_[np.ones(n_clock), np.zeros(n_slower)])
    layout = DFMStateLayout.from_state(state)
    assert layout.is_bm14() and layout.chain_length == k
    m = int(np.sum(state.r)) * int(state.max_lag_size) + n_clock + n_slower * k
    assert state.A.shape == (m, m) and layout.m == m

    weights = np.asarray(next(iter(state.tent_weights_dict.values())), dtype=np.float64)
    A, Q, C = (np.asarray(x, dtype=np.float64) for x in (state.A, state.Q, state.C))
    for j in range(n_clock, N):
        chain = layout.chain_states(j)
        # Loadings: the unnormalized tent weights on its own chain, nothing on the other idio states.
        np.testing.assert_array_equal(C[j, chain], weights)
        others = np.setdiff1d(np.arange(layout.n_factor_states, m), chain)
        np.testing.assert_array_equal(C[j, others], 0.0)
        # Head: AR(1); the rest: identity shift with zero process noise.
        head, rest = chain[0], chain[1:]
        np.testing.assert_array_equal(A[rest][:, chain[:-1]], np.eye(k - 1))
        np.testing.assert_array_equal(np.delete(A[rest], chain[:-1], axis=1), 0.0)
        np.testing.assert_array_equal(Q[rest], 0.0)
        assert Q[head, head] > 0.0
    for j in range(n_clock):
        np.testing.assert_array_equal(C[j, layout.idio_states], np.eye(n_clock)[j])
        np.testing.assert_array_equal(C[j, layout.chain_start:], 0.0)
    # Companion lag states of the factors: zero process noise.
    for b, r_b in enumerate(layout.r):
        start = int(layout.factor_offsets[b])
        np.testing.assert_array_equal(Q[start + r_b:start + r_b * layout.L], 0.0)
    # Every series has an idiosyncratic state, so R is at least the floor.
    assert np.all(np.diag(np.asarray(state.R)) >= MIN_OBSERVATION_NOISE)


# ------------------------------------------------------------ initialization


@pytest.fixture(scope="module")
def mq_init():
    """``initialize_parameters`` on the standardized monthly/quarterly panel."""
    from ..parity._cases_dfm import _MQ_MONTHLY, _MQ_QUARTERLY, _mq_panel

    df = _mq_panel()
    X = df[_MQ_MONTHLY + _MQ_QUARTERLY].to_numpy(np.float64)
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    R_mat, q = generate_R_mat(weights)
    n_slower = len(_MQ_QUARTERLY)
    N = X.shape[1]
    logging.disable(logging.CRITICAL)
    try:
        params = initialize_parameters(
            X, np.array([2.0]), 1, np.ones((N, 1)), R_mat, q, n_slower,
            bm14_idio_indicator(N - n_slower, n_slower), "m", {"q": weights},
        )
    finally:
        logging.disable(logging.NOTSET)
    return X, params, R_mat, n_slower, weights


def test_initial_R_is_the_floor(mq_init):
    X, (A, C, Q, R, Z_0, V_0), *_ = mq_init
    np.testing.assert_array_equal(R, np.diag(np.full(X.shape[1], MIN_OBSERVATION_NOISE)))


def test_initial_slower_loadings_satisfy_the_tent_restriction(mq_init):
    X, (A, C, Q, R, Z_0, V_0), R_mat, n_slower, weights = mq_init
    r, k = 2, len(weights)
    R_con = np.kron(R_mat, np.eye(r))
    for j in range(X.shape[1] - n_slower, X.shape[1]):
        c = C[j, :r * k]
        assert np.abs(c).max() > 1e-3
        assert np.abs(R_con @ c).max() <= 1e-10, R_con @ c
        # Equivalently: the loadings on the lags are the tent weights times one loading per factor.
        np.testing.assert_allclose(c.reshape(k, r), np.outer(weights, c[:r]), rtol=0, atol=1e-10)


def test_initial_chain_variance_comes_from_the_residual_variance(mq_init):
    X, (A, C, Q, R, Z_0, V_0), R_mat, n_slower, weights = mq_init
    N, k = X.shape[1], len(weights)
    layout = DFMStateLayout.build([2], 1, max(2, k), N, N - n_slower, k)
    for j in range(N - n_slower, N):
        head = layout.chain_states(j)[0]
        assert A[head, head] == pytest.approx(0.1)
        # (1 - rho^2) * residual variance / 19: of order 1e-2 for standardized data, not ~5e-10.
        assert 1e-4 < Q[head, head] < 1.0
        np.testing.assert_array_equal(Q[layout.chain_states(j)[1:]], 0.0)


def test_initial_V0_is_the_stationary_covariance(mq_init):
    X, (A, C, Q, R, Z_0, V_0), *_ = mq_init
    assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0 - 1e-3
    np.testing.assert_array_equal(V_0, V_0.T)
    assert np.linalg.eigvalsh(V_0).min() >= 0.0
    np.testing.assert_allclose(A @ V_0 @ A.T + Q, V_0, rtol=0, atol=1e-10 * np.abs(V_0).max())
    np.testing.assert_array_equal(Z_0, 0.0)


def test_initial_state_covariance_stable_and_unit_root():
    rng = np.random.default_rng(0)
    A = 0.5 * rng.standard_normal((4, 4)) / 2.0
    B = rng.standard_normal((4, 4))
    Q = B @ B.T
    V = initial_state_covariance(A, Q)
    np.testing.assert_allclose(V, linalg.solve_discrete_lyapunov(A, Q), rtol=1e-12, atol=1e-12)
    # A unit root: finite, PSD, deterministic, kappa on the unit-root direction.
    A_u = np.diag([1.0, 0.5, 0.2])
    Q_u = np.diag([1.0, 1.0, 1.0])
    V_u = initial_state_covariance(A_u, Q_u, kappa=1e4)
    assert np.all(np.isfinite(V_u)) and np.linalg.eigvalsh(V_u).min() >= 0.0
    np.testing.assert_allclose(np.sort(np.diag(V_u)), np.sort([1e4, 1 / (1 - 0.25), 1 / (1 - 0.04)]), rtol=1e-12)
    np.testing.assert_array_equal(V_u, initial_state_covariance(A_u, Q_u, kappa=1e4))


def test_tent_restricted_least_squares_is_exact():
    rng = np.random.default_rng(1)
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    R_mat, _ = generate_R_mat(weights)
    R_con = np.kron(R_mat, np.eye(2))
    # Nearly collinear regressors (lags of a persistent factor).
    f = np.cumsum(rng.standard_normal((200, 2)), axis=0)
    X = build_lag_matrix(f, 200, 2, 5, 1, np.float64)[5:]
    y = X @ np.kron(weights, [0.3, -0.2]) + 0.1 * rng.standard_normal(X.shape[0])
    c = tent_restricted_least_squares(y, X, R_con, np.zeros(R_con.shape[0]))
    assert np.abs(R_con @ c).max() <= 1e-12
    np.testing.assert_allclose(c, np.kron(weights, [0.3, -0.2]), atol=0.05)


def test_build_lag_matrix_shifts_rows():
    f = np.arange(12.0).reshape(6, 2)
    L = build_lag_matrix(f, 6, 2, 3, 1, np.float64)
    assert L.shape == (6, 6)
    for k in range(3):
        np.testing.assert_array_equal(L[k:, 2 * k:2 * k + 2], f[:6 - k])
        np.testing.assert_array_equal(L[:k, 2 * k:2 * k + 2], 0.0)


# ------------------------------------------------------------- save / load


def test_save_load_round_trip_and_old_layout_refused(tmp_path):
    from dfm_python import DFM
    from dfm_python.utils.errors import ConfigurationError

    model = _quiet_fit(*_monthly_quarterly())
    path = tmp_path / "mq"
    model.save(path)
    loaded = DFM.load(path)
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z"):
        np.testing.assert_array_equal(getattr(loaded.result_, name), getattr(model.result_, name))
    np.testing.assert_array_equal(loaded.engine_.training_state.idio_indicator,
                                  model.engine_.training_state.idio_indicator)
    np.testing.assert_array_equal(loaded.predict(4).to_numpy(), model.predict(4).to_numpy())

    # The pre-fix layout: an AR(1) idio state for every series (idio_indicator all ones).
    with np.load(path / "state.npz", allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["mixed_freq/idio_indicator"] = np.ones_like(arrays["mixed_freq/idio_indicator"])
    np.savez(path / "state.npz", **arrays)
    with pytest.raises(ConfigurationError, match="state layout"):
        DFM.load(path)

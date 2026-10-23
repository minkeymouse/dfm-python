"""SQUAREM acceleration of the DFM EM (``em_acceleration="squarem"``) against plain EM.

Three panels: the README quick-start panel, a monthly/quarterly panel with
missing values and a 300 x 30 synthetic panel.

- At convergence (``tol=1e-5``) SQUAREM uses fewer E-steps than plain EM and
  ends on a log-likelihood at least as high, within 2e-3 relative of plain
  EM's, with close common components (``x_sm = Z C'``, which does not depend
  on the factor rotation). The two do not agree to 1e-6: plain EM approaches
  the maximum sublinearly on these panels (``V_0``, and on the README panel
  the innovation variance of the noiseless quarterly series, shrink towards
  zero like ``1/k``), so it stops short of the maximum at a given ``tol``; on
  the README panel plain EM given four times SQUAREM's E-steps still ends
  below SQUAREM.
- The log-likelihoods of the accepted points do not decrease (the in-house
  exact filter, and the reference filter on the accepted parameters).
- The extrapolation keeps the structural zeros, the shift entries and the
  tent restrictions; the fallback paths (an invalid point, a point that lowers
  the log-likelihood) reproduce plain EM; the guard stops a corrupted run;
  ``max_iter`` counts E-steps; ``update(refit=True)`` is unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ..ssm import _reference_kf as ref

PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")
TENT = {"q:m": [1, 2, 3, 2, 1]}


# ------------------------------------------------------------------ panels


def _readme():
    """The README quick-start panel and model."""
    from dfm_python import PanelData

    rng = np.random.default_rng(0)
    f = np.zeros(120)
    for t in range(1, 120):
        f[t] = 0.8 * f[t - 1] + rng.standard_normal()
    idx = pd.date_range("2015-01-31", periods=120, freq="ME", name="date")
    df = pd.DataFrame({"ip": f, "sales": 0.7 * f, "jobs": -0.5 * f}, index=idx)
    df += 0.5 * rng.standard_normal(df.shape)
    df["gdp"] = np.convolve(f, [1, 2, 3, 2, 1])[:120] / 9
    df.loc[df.index.month % 3 != 0, "gdp"] = np.nan
    return PanelData(y=df, freq={"gdp": "q"}), {"n_factors": 1, "clock": "m", "tent_weights": TENT,
                                                "scaler": "standard"}


def _mixed():
    """Eight monthly series (10% missing, a ragged start) and three quarterly ones on two factors."""
    from dfm_python import PanelData

    T, n_m, n_q, r = 180, 8, 3, 2
    rng = np.random.default_rng(1)
    f = np.zeros((T, r))
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal(r)
    y_m = f @ rng.standard_normal((n_m, r)).T + 0.7 * rng.standard_normal((T, n_m))
    y_m[rng.random(y_m.shape) < 0.1] = np.nan
    tent = np.array([1, 2, 3, 2, 1]) / 9
    f_agg = np.stack([np.convolve(f[:, j], tent)[:T] for j in range(r)], axis=1)
    y_q = f_agg @ rng.standard_normal((n_q, r)).T + 0.3 * rng.standard_normal((T, n_q))
    quarterly = [f"q{i}" for i in range(n_q)]
    df = pd.DataFrame(np.hstack([y_m, y_q]), columns=[f"m{i}" for i in range(n_m)] + quarterly,
                      index=pd.date_range("2005-01-31", periods=T, freq="ME", name="date"))
    df.loc[df.index.month % 3 != 0, quarterly] = np.nan
    df.iloc[:12, 0] = np.nan
    return PanelData(y=df, freq={name: "q" for name in quarterly}), {"n_factors": r, "clock": "m",
                                                                     "tent_weights": TENT, "scaler": "standard"}


def _synthetic():
    """300 x 30: three AR(1) factors, AR(1) idiosyncratic components."""
    from dfm_python import PanelData

    T, N, r = 300, 30, 3
    rng = np.random.default_rng(2)
    f = np.zeros((T, r))
    for t in range(1, T):
        f[t] = np.array([0.9, 0.6, 0.3]) * f[t - 1] + rng.standard_normal(r)
    loadings = rng.standard_normal((N, r))
    e = np.zeros((T, N))
    for t in range(1, T):
        e[t] = 0.3 * e[t - 1] + rng.standard_normal(N)
    df = pd.DataFrame(f @ loadings.T + e, columns=[f"s{i}" for i in range(N)],
                      index=pd.date_range("2000-01-31", periods=T, freq="ME", name="date"))
    return PanelData(y=df), {"n_factors": r, "clock": "m", "scaler": "standard"}


PANELS = {"readme": _readme, "mixed": _mixed, "synthetic": _synthetic}


def _params(obj) -> tuple[np.ndarray, ...]:
    return tuple(np.array(getattr(obj, k), dtype=np.float64) for k in PARAMS)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def fit(panel, params, monkeypatch, **overrides):
    """Fit DFM; return ``(model, metadata, steps)``.

    ``metadata`` is the dict ``run_em_algorithm`` returned and ``steps`` the
    ``(input state, loglik)`` of every E-step.
    """
    from dfm_python import DFM
    import dfm_python.functional.em as em

    meta, steps = {}, []
    original_run, original_step = em.run_em_algorithm, em.em_step

    def run(*args, **kwargs):
        state, metadata = original_run(*args, **kwargs)
        meta.update(metadata)
        return state, metadata

    def step(X, state, *args, **kwargs):
        out = original_step(X, state, *args, **kwargs)
        steps.append((state, out[1]))
        return out

    monkeypatch.setattr(em, "run_em_algorithm", run)
    monkeypatch.setattr(em, "em_step", step)
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DFM(**{**params, **overrides}).fit(panel)
    finally:
        logging.disable(logging.NOTSET)
    monkeypatch.setattr(em, "run_em_algorithm", original_run)
    monkeypatch.setattr(em, "em_step", original_step)
    return model, meta, steps


def _assert_same_fit(a, b) -> None:
    for name in PARAMS + ("Z", "x_sm"):
        assert getattr(a.result_, name).tobytes() == getattr(b.result_, name).tobytes(), name
    assert np.float64(a.loglik_).tobytes() == np.float64(b.loglik_).tobytes()


# ------------------------------------------------------- convergence


@pytest.mark.parametrize("name", list(PANELS))
def test_squarem_converges_in_fewer_esteps_to_at_least_plain_em_loglik(name, monkeypatch):
    panel, params = PANELS[name]()
    plain, _, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=3000, tol=1e-5)
    sq, meta, _ = fit(panel, params, monkeypatch, em_acceleration="squarem", max_iter=3000, tol=1e-5)
    assert plain.converged_ and sq.converged_
    assert sq.n_iter_ < plain.n_iter_
    assert sq.n_iter_ == meta["num_iter"]
    # Plain EM stops short of the maximum (sublinear convergence): SQUAREM ends
    # higher, by at most 2e-3 relative on these panels.
    assert sq.loglik_ >= plain.loglik_
    assert _rel(sq.loglik_, plain.loglik_) < 2e-3
    # Common components after the first year (V_0 converges slowest, so the
    # first rows differ most), relative to their largest value.
    x_p, x_s = np.asarray(plain.result_.x_sm)[12:], np.asarray(sq.result_.x_sm)[12:]
    assert np.max(np.abs(x_s - x_p)) < 2e-2 * np.max(np.abs(x_p))
    np.testing.assert_allclose(np.diag(sq.result_.R), np.diag(plain.result_.R), rtol=0.2, atol=1e-3)


def test_plain_em_with_four_times_the_esteps_stays_below_squarem(monkeypatch):
    panel, params = _readme()
    sq, _, _ = fit(panel, params, monkeypatch, max_iter=3000, tol=1e-5)
    plain, _, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=4 * sq.n_iter_, tol=0.0)
    assert plain.n_iter_ == 4 * sq.n_iter_
    assert plain.loglik_ < sq.loglik_


@pytest.mark.parametrize("name", list(PANELS))
def test_accepted_logliks_never_decrease(name, monkeypatch):
    panel, params = PANELS[name]()
    model, meta, steps = fit(panel, params, monkeypatch, max_iter=60, tol=0.0)
    path = np.asarray(meta["loglik_path"])
    assert len(path) >= 15 and model.n_iter_ == 60
    assert np.all(np.diff(path) >= -1e-10 * np.abs(path[:-1])), path
    # Two or three E-steps per cycle (an invalid point takes no E-step).
    assert len(path) * 2 <= 60 <= len(path) * 3
    if name == "mixed":
        # The reference filter agrees on the accepted parameters.
        X = model.engine_.data_processed
        accepted = [state for state, ll in steps if ll in set(path)]
        exact = [ref.loglik(*_params(state), X) for state in accepted]
        assert len(exact) == len(path)
        assert all(exact[k + 1] >= exact[k] - 1e-9 * abs(exact[k]) for k in range(len(exact) - 1))


# --------------------------------------------------- the extrapolation


def _plain_iterates(monkeypatch, n: int):
    """The first ``n`` M-step outputs of plain EM on the mixed panel, and its engine state."""
    import dfm_python.functional.em as em

    outputs = []
    original = em.em_step

    def step(X, state, *args, **kwargs):
        out = original(X, state, *args, **kwargs)
        outputs.append(out[0])
        return out

    monkeypatch.setattr(em, "em_step", step)
    panel, params = _mixed()
    logging.disable(logging.CRITICAL)
    try:
        from dfm_python import DFM

        model = DFM(**params, em_acceleration=None, max_iter=n, tol=0.0).fit(panel)
    finally:
        logging.disable(logging.NOTSET)
    return outputs, model


def test_extrapolation_keeps_the_structure(monkeypatch):
    from dfm_python.functional.em import _squarem_point
    from dfm_python.functional.state_layout import DFMStateLayout

    outputs, model = _plain_iterates(monkeypatch, 30)
    # The first three consecutive M-step outputs that extrapolate (||r|| > ||v||).
    for k in range(len(outputs) - 2):
        theta0, theta1, theta2 = outputs[k:k + 3]
        point, step = _squarem_point(theta0, theta1, theta2, step_max=1e6)
        if step > 1.0:
            break
    assert point is not None and step > 1.0
    for name in ("A", "C", "Q", "R"):
        fixed = (getattr(theta0, name) == getattr(theta1, name)) & (getattr(theta1, name) == getattr(theta2, name))
        assert fixed.any() and np.array_equal(getattr(point, name)[fixed], getattr(theta0, name)[fixed]), name
    # The companion shift rows (ones) and the structural zeros of A, Q and C.
    assert np.array_equal(point.A == 0, theta0.A == 0) and np.any(point.A == 1.0)
    assert np.array_equal(point.Q == 0, theta0.Q == 0) and np.array_equal(point.C == 0, theta0.C == 0)
    # R stays diagonal; Q and V_0 stay symmetric.
    assert np.count_nonzero(point.R - np.diag(np.diag(point.R))) == 0
    np.testing.assert_array_equal(point.Q, point.Q.T)
    np.testing.assert_array_equal(point.V_0, point.V_0.T)
    # The tent restriction R_mat c = q on every quarterly series' loadings.
    layout = DFMStateLayout.from_state(theta0, n_series=theta0.C.shape[0])
    R_mat = np.asarray(theta0.constraint_matrix, dtype=np.float64)
    q = np.asarray(theta0.constraint_vector, dtype=np.float64).ravel()
    r = int(layout.r[0])
    cols = layout.factor_offsets[0] + np.arange(R_mat.shape[1] * r)
    R_con, q_con = np.kron(R_mat, np.eye(r)), np.kron(q, np.ones(r))
    assert layout.n_clock < theta0.C.shape[0]
    for j in range(layout.n_clock, theta0.C.shape[0]):
        for state in (theta0, theta1, theta2, point):
            np.testing.assert_allclose(R_con @ state.C[j, cols], q_con, atol=1e-10)
        assert not np.allclose(point.C[j, cols], theta2.C[j, cols])


def test_step_length_is_clipped_to_one_and_the_bound(monkeypatch):
    from dfm_python.functional.em import _squarem_point

    outputs, _ = _plain_iterates(monkeypatch, 30)
    theta0, theta1, theta2 = outputs[-3:]
    assert _squarem_point(theta0, theta1, theta2, step_max=1.0) == (theta2, 1.0)
    unbounded = _squarem_point(theta0, theta1, theta2, step_max=np.inf)[1]
    assert unbounded > 1.5
    point, step = _squarem_point(theta0, theta1, theta2, step_max=1.5)
    assert step == 1.5
    expected = theta0.C + 3.0 * (theta1.C - theta0.C) + 2.25 * (theta2.C - 2 * theta1.C + theta0.C)
    np.testing.assert_allclose(point.C, expected, rtol=1e-14, atol=1e-15)
    # An extrapolation past a variance's zero is not a valid parameter set:
    # R = I, I/2, I/10 gives the step 5 and R = (1 - 5 + 2.5) I.
    eye = np.eye(theta0.R.shape[0])
    point, step = _squarem_point(replace(theta2, R=eye), replace(theta2, R=0.5 * eye), replace(theta2, R=0.1 * eye),
                                 step_max=np.inf)
    assert point is None and np.isclose(step, 5.0)


# ------------------------------------------------------- fallbacks


def test_invalid_points_fall_back_to_plain_em(monkeypatch):
    """Every extrapolation invalid: two E-steps per cycle, the iterates of plain EM, bitwise.

    The sixth cycle ends on the E-step budget before its extrapolation.
    """
    import dfm_python.functional.em as em

    panel, params = _readme()
    plain, plain_meta, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=12, tol=0.0)
    monkeypatch.setattr(em, "_squarem_point", lambda t0, t1, t2, step_max: (None, 2.0))
    sq, meta, _ = fit(panel, params, monkeypatch, max_iter=12, tol=0.0)
    assert sq.n_iter_ == 12 and meta["squarem_fallbacks"] == 5
    assert meta["loglik_path"] == plain_meta["loglik_path"][::2]
    _assert_same_fit(sq, plain)


def test_points_that_lower_the_loglik_fall_back_to_plain_em(monkeypatch):
    """Every extrapolated point worse than theta0: three E-steps per cycle, plain EM's iterates."""
    import dfm_python.functional.em as em

    panel, params = _readme()
    plain, plain_meta, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=8, tol=0.0)
    points = []

    def worse(theta0, theta1, theta2, step_max):
        points.append(replace(theta0, R=theta0.R * 50.0))
        return points[-1], step_max

    monkeypatch.setattr(em, "_squarem_point", worse)
    sq, meta, steps = fit(panel, params, monkeypatch, max_iter=12, tol=0.0)
    assert sq.n_iter_ == 12 and meta["squarem_fallbacks"] == 4 and len(points) == 4
    # Every third E-step ran at the rejected point, below the cycle's start.
    for k, point in enumerate(points):
        assert steps[3 * k + 2][0] is point and steps[3 * k + 2][1] < meta["loglik_path"][k]
    assert meta["loglik_path"] == plain_meta["loglik_path"][::2]
    _assert_same_fit(sq, plain)


def test_a_failed_estep_at_the_point_falls_back(monkeypatch):
    import dfm_python.functional.em as em

    panel, params = _readme()
    plain, _, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=4, tol=0.0)
    broken = []

    def failing(theta0, theta1, theta2, step_max):
        broken.append(replace(theta0, R=-theta0.R, Q=np.zeros_like(theta0.Q), V_0=np.zeros_like(theta0.V_0)))
        return broken[-1], step_max

    monkeypatch.setattr(em, "_squarem_point", failing)
    sq, meta, _ = fit(panel, params, monkeypatch, max_iter=6, tol=0.0)
    assert sq.n_iter_ == 6 and meta["squarem_fallbacks"] == 2 and meta["stop_reason"] is None
    _assert_same_fit(sq, plain)


def test_guard_stops_squarem_on_a_non_finite_em_point(monkeypatch):
    """A non-finite theta2 stops EM and returns the best accepted parameters."""
    import dfm_python.functional.em as em

    original = em.em_step
    n_calls = [0]

    def step(X, state, *args, **kwargs):
        new_state, loglik, kf = original(X, state, *args, **kwargs)
        n_calls[0] += 1
        if n_calls[0] == 5:  # the second cycle's theta1 -> theta2 (the first cycle had three E-steps)
            A = np.array(new_state.A)
            A[0, 0] = np.nan
            new_state = replace(new_state, A=A)
        return new_state, loglik, kf

    monkeypatch.setattr(em, "em_step", step)
    panel, params = _readme()
    model, meta, steps = fit(panel, params, monkeypatch, max_iter=50, tol=0.0)
    assert meta["squarem_fallbacks"] == 0 and len(meta["loglik_path"]) == 2
    assert meta["stop_reason"] == "non-finite parameters"
    assert model.n_iter_ == 5 and not model.converged_
    # The returned parameters are the best accepted point, the second cycle's start (E-step 4).
    best_state, best_loglik = steps[3]
    assert best_loglik == meta["loglik_path"][1] == meta["best_loglik"]
    X = model.engine_.data_processed
    assert _rel(float(model.loglik_), best_loglik) < 1e-10
    assert _rel(float(model.loglik_), ref.loglik(*_params(model.result_), X)) < 1e-8
    np.testing.assert_array_equal(np.diag(model.result_.R), np.diag(best_state.R))


# ------------------------------------------------- options and budget


@pytest.mark.parametrize("max_iter", [1, 2, 3, 4, 7])
def test_max_iter_counts_esteps(max_iter, monkeypatch):
    panel, params = _readme()
    model, meta, steps = fit(panel, params, monkeypatch, max_iter=max_iter, tol=0.0)
    assert model.n_iter_ == meta["num_iter"] == len(steps) == max_iter
    if max_iter <= 2:  # one or two EM maps from the start: plain EM's
        plain, _, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=max_iter, tol=0.0)
        _assert_same_fit(model, plain)


def test_ar_clip_applies_to_every_em_map(monkeypatch):
    from dfm_python.functional.state_layout import DFMStateLayout

    panel, params = _mixed()
    model, meta, _ = fit(panel, params, monkeypatch, max_iter=40, tol=0.0, ar_clip={"min": -0.2, "max": 0.2})
    # Clipping breaks the EM ascent; the guard may stop EM on a decrease and
    # return the best accepted point. Either way the returned parameters are an
    # EM map's output, with the estimated AR coefficients clipped (the C
    # normalization keeps the diagonal of A).
    assert meta["stop_reason"] in (None, "log-likelihood decreased") and model.n_iter_ > 3
    state = model.engine_.training_state
    layout = DFMStateLayout.from_state(state)
    ii = layout.ar1_states
    cur = layout.factor_current(0)
    estimated = np.concatenate([state.A[ii, ii], state.A[np.ix_(cur, cur)].diagonal()])
    assert np.all(np.abs(estimated) <= 0.2 + 1e-12)
    assert np.isclose(np.max(np.abs(estimated)), 0.2)


def test_update_refit_is_one_em_map_with_both_settings(monkeypatch):
    from dfm_python import PanelData

    panel, params = _mixed()
    model, _, _ = fit(panel, params, monkeypatch, max_iter=20, tol=0.0)
    T = len(panel.index)
    head = PanelData(y=panel.y.iloc[:T - 6], freq=panel.freq)
    new = PanelData(y=panel.y.iloc[T - 6:], freq=panel.freq)
    model, _, _ = fit(head, params, monkeypatch, max_iter=20, tol=0.0)
    plain = copy.deepcopy(model)
    plain.engine_._config.em_acceleration = None
    logging.disable(logging.CRITICAL)
    try:
        model.update(new, refit=True)
        plain.update(new, refit=True)
    finally:
        logging.disable(logging.NOTSET)
    assert model.n_iter_ == plain.n_iter_ == 1
    _assert_same_fit(model, plain)


def test_invalid_em_acceleration_is_rejected():
    from dfm_python import DFM
    from dfm_python.functional.em import EMConfig, run_em_algorithm
    from dfm_python.utils.errors import ConfigurationError

    panel, params = _readme()
    with pytest.raises(ConfigurationError, match="em_acceleration"):
        DFM(**params, em_acceleration="anderson").fit(panel)
    with pytest.raises(ValueError, match="em_acceleration"):
        run_em_algorithm(np.zeros((5, 2)), None, config=EMConfig(em_acceleration="anderson"))


def test_a_run_saved_without_em_acceleration_loads_as_plain_em(tmp_path, monkeypatch):
    """Runs saved before em_acceleration existed (0.8.x) were fitted by plain EM."""
    from dfm_python import DFM

    panel, params = _readme()
    model, _, _ = fit(panel, params, monkeypatch, em_acceleration=None, max_iter=10)
    model.save(tmp_path / "run")
    for name, key in (("config.json", "params"), ("state.json", "engine_config")):
        path = tmp_path / "run" / name
        content = json.loads(path.read_text())
        del content[key]["em_acceleration"]
        path.write_text(json.dumps(content))
    loaded = DFM.load(tmp_path / "run")
    assert loaded.em_acceleration is None
    assert loaded.predict(6).equals(model.predict(6))

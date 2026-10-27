"""The EM driver (``functional.em.run_em_algorithm``) against the independent reference filter.

- The reported log-likelihood is the exact missing-data log-likelihood of the
  RETURNED parameters (reference filter, rel 1e-8), and ``result_.Z`` is the
  smoothed state of those parameters, on the synthetic, monthly/quarterly and
  weekly/monthly fits.
- The monotonicity guard: when an iterate's exact log-likelihood drops (an M-step
  output corrupted on purpose) or its filter is not finite, EM stops, returns
  the best parameters seen with ``converged=False`` and reports their loglik
  (plain EM, ``em_acceleration=None``; the SQUAREM guards are tested in
  ``test_squarem.py``).
- ``update(refit=True)`` runs one EM iteration on all rows through the same
  driver and reports the exact loglik of what it returns.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace

import numpy as np
import pytest

from ..ssm import _reference_kf as ref

PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")
LOGLIK_RTOL = 1e-8


def _params(obj) -> tuple[np.ndarray, ...]:
    return tuple(np.array(getattr(obj, k), dtype=np.float64) for k in PARAMS)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _synthetic():
    from .test_em_oracle import _synthetic as make

    return make()


def _monthly_quarterly():
    from .test_em_oracle import _monthly_quarterly as make

    return make()


def _weekly_monthly():
    """config/dfm_macro.yaml on the last 400 rows of data/macro.csv (the WM golden panel)."""
    from dfm_python import DFM, PanelData
    from dfm_python.config import DFMConfig, load_yaml

    from ..parity._cases_common import REPO_ROOT, _macro_tail
    from ..parity._cases_dfm import _DFM_WM_ROWS

    sections = load_yaml(REPO_ROOT / "config" / "dfm_macro.yaml")
    panel = PanelData.from_wide(_macro_tail(_DFM_WM_ROWS), **sections["data"])
    return DFM.from_config(DFMConfig.from_dict(sections["model"])), panel


_CASES = {"dfm_synthetic": _synthetic, "dfm_macro_monthly_quarterly": _monthly_quarterly,
          "dfm_macro_weekly_monthly": _weekly_monthly}


def _fit(model, panel):
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(panel)
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("case", list(_CASES))
def test_reported_loglik_is_exact_at_the_returned_parameters(case):
    """result_.loglik equals the reference loglik of result_'s parameters; result_.Z is their smoothed state."""
    model = _fit(*_CASES[case]())
    res = model.result_
    X = model.engine_.data_processed
    params = _params(res)
    expected = ref.filter_smooth(*params, X)
    assert np.isfinite(res.loglik) and res.loglik != -X.shape[0] / 2
    assert _rel(float(res.loglik), expected["ll"]) < LOGLIK_RTOL, (res.loglik, expected["ll"])
    assert float(model.loglik_) == float(res.loglik)
    assert ref.rel_err(np.asarray(res.Z, dtype=np.float64), expected["sm"]) < 1e-6
    # The last smoothed state is the last filtered state x_T|T, where update() and predict() start.
    assert ref.rel_err(np.asarray(res.Z, dtype=np.float64)[-1], expected["fm"][-1]) < 1e-6


# ------------------------------------------------------------ the guards


def _run_with_corruption(monkeypatch, corrupt, at: int, max_iter: int = 8):
    """Fit the synthetic case; ``corrupt(state)`` replaces the M-step output of iteration ``at`` (0-based).

    Returns ``(model, X, inputs, metadata)`` where ``inputs`` are the parameters
    each EM iteration started from and ``metadata`` the dict ``run_em_algorithm``
    returned.
    """
    import dfm_python.functional.em as em

    inputs, calls, out = [], [], {}
    original_step, original_run = em.em_step, em.run_em_algorithm

    def step(X, state, *args, **kwargs):
        inputs.append(_params(state))
        new_state, loglik, kf = original_step(X, state, *args, **kwargs)
        calls.append(loglik)
        if len(calls) - 1 == at:
            new_state = corrupt(new_state)
        return new_state, loglik, kf

    def run(*args, **kwargs):
        state, metadata = original_run(*args, **kwargs)
        out.update(metadata)
        return state, metadata

    monkeypatch.setattr(em, "em_step", step)
    monkeypatch.setattr(em, "run_em_algorithm", run)
    model, panel = _synthetic()
    model.set_params(max_iter=max_iter, em_acceleration=None)
    model = _fit(model, panel)
    return model, model.engine_.data_processed, inputs, out


def _check_returns_best(model, X, inputs, metadata, best_index: int, reason: str):
    res = model.result_
    lls = [ref.loglik(*p, X) for p in inputs[:best_index + 1]]
    assert all(lls[k + 1] >= lls[k] - 1e-6 * abs(lls[k]) for k in range(len(lls) - 1))
    assert metadata["stop_reason"] == reason
    assert not bool(res.converged)
    # The returned parameters are the best iterate (up to the likelihood-invariant
    # factor rescaling), and the reported loglik is theirs.
    exact = ref.loglik(*_params(res), X)
    assert _rel(float(res.loglik), exact) < LOGLIK_RTOL
    assert _rel(exact, max(lls)) < 1e-9, (exact, lls)
    np.testing.assert_allclose(_params(res)[3], inputs[best_index][3], rtol=1e-12)  # R is not rescaled


def test_loglik_decrease_stops_em_and_returns_the_best_parameters(monkeypatch):
    """An M-step output with a lower loglik (R x 50) stops EM: the best earlier iterate is returned."""
    def corrupt(state):
        return replace(state, R=np.asarray(state.R) * 50.0)

    model, X, inputs, metadata = _run_with_corruption(monkeypatch, corrupt, at=2)
    # Iteration 3 starts from the corrupted theta_3, whose loglik is below L_2.
    assert len(inputs) == 4
    assert ref.loglik(*inputs[3], X) < ref.loglik(*inputs[2], X)
    _check_returns_best(model, X, inputs, metadata, best_index=2, reason="log-likelihood decreased")
    assert metadata["num_iter"] == 3


def test_non_finite_parameters_stop_em_and_return_the_best_parameters(monkeypatch):
    """A non-finite M-step output stops EM with the best earlier iterate."""
    def corrupt(state):
        A = np.array(state.A, dtype=np.float64)
        A[0, 0] = np.nan
        return replace(state, A=A)

    model, X, inputs, metadata = _run_with_corruption(monkeypatch, corrupt, at=2)
    assert len(inputs) == 3
    _check_returns_best(model, X, inputs, metadata, best_index=2, reason="non-finite parameters")


def test_failed_filter_stops_em_and_returns_the_best_parameters(monkeypatch):
    """An M-step output whose filter fails (an innovation covariance not positive definite) stops EM."""
    def corrupt(state):
        return replace(state, R=-np.asarray(state.R), V_0=np.zeros_like(np.asarray(state.V_0)),
                       Q=np.zeros_like(np.asarray(state.Q)))

    model, X, inputs, metadata = _run_with_corruption(monkeypatch, corrupt, at=2)
    assert len(inputs) == 4  # em_step is entered with the corrupted parameters; its E-step raises
    _check_returns_best(model, X, inputs, metadata, best_index=2, reason="E-step failed")


def test_final_pass_rejects_a_worse_last_iterate(monkeypatch):
    """With max_iter reached, a last M-step output with a lower loglik is not returned."""
    def corrupt(state):
        return replace(state, R=np.asarray(state.R) * 50.0)

    model, X, inputs, metadata = _run_with_corruption(monkeypatch, corrupt, at=2, max_iter=3)
    assert len(inputs) == 3
    _check_returns_best(model, X, inputs, metadata, best_index=2, reason="log-likelihood decreased")
    assert metadata["num_iter"] == 3


def test_convergence_is_tested_on_the_exact_loglik():
    """A converged run stops when the relative change of the exact loglik is below tol."""
    model, panel = _synthetic()
    model.set_params(max_iter=500, tol=1e-4)
    model = _fit(model, panel)
    assert bool(model.result_.converged)
    assert 3 <= model.n_iter_ < 500
    X = model.engine_.data_processed
    assert _rel(float(model.loglik_), ref.loglik(*_params(model.result_), X)) < LOGLIK_RTOL


# ------------------------------------------------------------ update(refit=True)


def test_refit_runs_one_iteration_and_reports_its_exact_loglik():
    """update(refit=True): one EM iteration on all rows; loglik_ is the exact loglik of the result."""
    from dfm_python import PanelData

    from ..parity._cases_dfm import dfm_synthetic_inputs

    df, new_df, config, _ = dfm_synthetic_inputs()
    cols = list(config["frequency"])
    model, panel = _synthetic()
    model = _fit(model, panel)
    ll_fit = float(model.loglik_)
    new = PanelData(y=new_df.set_index("date"), freq={c: "m" for c in cols})
    logging.disable(logging.CRITICAL)
    try:
        model.update(new, refit=True)
    finally:
        logging.disable(logging.NOTSET)
    X = model.engine_.data_processed
    assert X.shape[0] == len(df) + len(new_df)
    assert model.n_iter_ == 1
    exact = ref.loglik(*_params(model.result_), X)
    assert _rel(float(model.loglik_), exact) < LOGLIK_RTOL
    assert float(model.loglik_) != ll_fit

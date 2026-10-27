"""Oracle tests for the DFM EM loop: the exact log-likelihood of the EM iterates.

Each test fits ``DFM`` with a hook on ``dfm_python.functional.em.em_step`` that
records the parameters every EM iteration starts from (and the data EM sees);
the returned parameters are appended. The exact missing-data log-likelihood of
each iterate is then computed with the independent reference filter
(``src/test/ssm/_reference_kf.py``). EM must not decrease it (tolerance 1e-6
relative): the E-step is exact and every M-step block is a conditional
maximizer of the expected complete-data log-likelihood (see
``test_mstep_qfunction.py`` for the block-by-block check). The fits run plain
EM (``em_acceleration=None``), whose every iterate is an EM map; the SQUAREM
path is tested in ``test_squarem.py``.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pytest

from ..ssm import _reference_kf as ref

MONOTONE_RTOL = 1e-6
PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")


def _params(obj) -> tuple[np.ndarray, ...]:
    return tuple(np.array(getattr(obj, k), dtype=np.float64) for k in PARAMS)


def fit_with_iterates(model, panel, monkeypatch):
    """Fit ``model`` on ``panel``; return ``(model, X, iterates)``.

    ``iterates`` are the parameters each EM iteration started from, followed by
    the returned parameters (``result_``); ``X`` is the data EM ran on.
    """
    import dfm_python.functional.em as em

    seen: list = []
    original = em.em_step

    def recording_em_step(X, state, *args, **kwargs):
        seen.append((np.array(X, dtype=np.float64), _params(state)))
        return original(X, state, *args, **kwargs)

    monkeypatch.setattr(em, "em_step", recording_em_step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(panel)
    assert seen, "EM ran no iteration"
    X = seen[0][0]
    for X_k, _ in seen[1:]:
        np.testing.assert_array_equal(X_k, X)
    return model, X, [p for _, p in seen] + [_params(model.result_)]


def reference_logliks(X: np.ndarray, iterates) -> list[float]:
    out = []
    for p in iterates:
        if not all(np.all(np.isfinite(a)) for a in p):
            out.append(float("nan"))
            continue
        try:
            out.append(ref.loglik(*p, X))
        except np.linalg.LinAlgError:
            out.append(float("nan"))
    return out


def assert_monotone(lls: list[float], label: str) -> None:
    assert all(math.isfinite(v) for v in lls), f"{label}: non-finite loglik of an iterate: {lls}"
    drops = [(k, lls[k], lls[k + 1]) for k in range(len(lls) - 1)
             if lls[k + 1] < lls[k] - MONOTONE_RTOL * abs(lls[k])]
    assert not drops, f"{label}: the exact loglik decreased at (iteration, before, after): {drops}"


# --------------------------------------------------------------------- cases


def _synthetic():
    from dfm_python import DFM, PanelData

    from ..parity._cases_dfm import dfm_synthetic_inputs

    df, _, config, _ = dfm_synthetic_inputs()
    cols = list(config["frequency"])
    panel = PanelData(y=df.set_index("date"), freq={c: "m" for c in cols})
    model = DFM(blocks={"Block_Global": {"n_factors": 2, "series": cols}}, clock="m",
                max_iter=15, tol=0.0, scaler="standard", em_acceleration=None)
    return model, panel


def _monthly_quarterly():
    from dfm_python import DFM, PanelData

    from ..parity._cases_dfm import _MQ_MONTHLY, _MQ_QUARTERLY, _mq_panel

    panel = PanelData(y=_mq_panel().set_index("date"),
                      freq={**{c: "m" for c in _MQ_MONTHLY}, **{c: "q" for c in _MQ_QUARTERLY}})
    model = DFM(blocks={"Block_Global": {"n_factors": 2, "series": _MQ_MONTHLY + _MQ_QUARTERLY}},
                clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]}, max_iter=15, tol=0.0, scaler="standard",
                em_acceleration=None)
    return model, panel


def _weekly_monthly():
    from dfm_python import DFM, PanelData
    from dfm_python.config import DFMConfig, load_yaml

    from ..parity._cases_common import REPO_ROOT, _macro_tail
    from ..parity._cases_dfm import _DFM_WM_ROWS

    sections = load_yaml(REPO_ROOT / "config" / "dfm_macro.yaml")
    panel = PanelData.from_wide(_macro_tail(_DFM_WM_ROWS), **sections["data"])
    model = DFM.from_config(DFMConfig.from_dict(sections["model"]))
    model.set_params(max_iter=15, tol=0.0, em_acceleration=None)
    return model, panel


_CASES = {"dfm_synthetic": _synthetic, "dfm_macro_monthly_quarterly": _monthly_quarterly,
          "dfm_macro_weekly_monthly": _weekly_monthly}


@pytest.mark.parametrize("case", list(_CASES))
def test_em_exact_loglik_is_monotone(case, monkeypatch):
    """The exact loglik of the EM iterates is non-decreasing (15 iterations, tol=0)."""
    model, panel = _CASES[case]()
    logging.disable(logging.CRITICAL)
    try:
        model, X, iterates = fit_with_iterates(model, panel, monkeypatch)
    finally:
        logging.disable(logging.NOTSET)
    assert len(iterates) >= 3
    assert_monotone(reference_logliks(X, iterates), case)


def test_c_normalization_rescale_preserves_loglik():
    """C -> C diag(D_inv) with the rescaled (A, Q, V_0, Z_0) is the same model: same exact loglik."""
    from dfm_python.numeric.stability import rescale_ssm_params_for_C_normalization

    A, C, Q, R, Z0, V0, Y = ref.random_model(m=4, N=6, T=60, nan_frac=0.3, seed=21)
    D_inv = np.array([0.5, 2.0, 1.5, 0.8])
    A2, Q2, V02, Z02 = rescale_ssm_params_for_C_normalization(A, Q, V0, Z0, D_inv)
    C2 = C * D_inv[np.newaxis, :]
    before = ref.loglik(A, C, Q, R, Z0, V0, Y)
    after = ref.loglik(A2, C2, Q2, R, Z02, V02, Y)
    assert abs(after - before) <= 1e-10 * abs(before), (before, after)


# ------------------------------------------------- the macro truncation sweep

#: |EZ| clip bound of the pre-fix EM (numeric.stability.cap_smoothed_states: sqrt(1e3 * T)).
def _clip_bound(T: int) -> float:
    return math.sqrt(1e3 * T)


@pytest.mark.parametrize("rows", [300, 400, 600, "full"])
def test_macro_truncation_sweep(rows, monkeypatch):
    """data/macro.csv with config/dfm_macro.yaml (weekly clock, monthly tent [1,2,1]) on its last rows.

    Finite parameters, smoothed states and forecasts; no state at the clip
    bound; a non-decreasing exact loglik over the EM iterates; the reported
    loglik is the exact loglik of the returned parameters. ``"full"`` is the
    whole panel (2538 weeks).
    """
    import pandas as pd

    from dfm_python import DFM, PanelData
    from dfm_python.config import DFMConfig, load_yaml

    from ..parity._cases_common import MACRO_CSV, REPO_ROOT, _macro_tail

    sections = load_yaml(REPO_ROOT / "config" / "dfm_macro.yaml")
    if rows == "full":
        df = pd.read_csv(MACRO_CSV)
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = _macro_tail(rows)
    panel = PanelData.from_wide(df, **sections["data"])
    model = DFM.from_config(DFMConfig.from_dict(sections["model"]))
    model.set_params(em_acceleration=None)
    logging.disable(logging.CRITICAL)
    try:
        model, X, iterates = fit_with_iterates(model, panel, monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(6).to_numpy()
            factors = model.predict_factors(6).to_numpy()
    finally:
        logging.disable(logging.NOTSET)
    for name, value in zip(PARAMS, _params(model.result_)):
        assert np.all(np.isfinite(value)), f"non-finite {name}"
    Z = np.asarray(model.result_.Z)
    assert np.all(np.isfinite(Z)) and np.all(np.isfinite(forecast)) and np.all(np.isfinite(factors))
    bound = _clip_bound(X.shape[0])
    assert np.max(np.abs(Z)) < bound * (1 - 1e-9), "smoothed state at the clip bound"
    assert np.max(np.abs(model.result_.Z_0)) < bound * (1 - 1e-9), "Z_0 at the clip bound"
    lls = reference_logliks(X, iterates)
    assert_monotone(lls, f"macro last {rows}")
    reported = float(model.result_.loglik)
    assert abs(reported - lls[-1]) <= 1e-8 * abs(lls[-1]), (reported, lls[-1])

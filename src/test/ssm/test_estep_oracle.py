"""Oracle tests for the DFM E-step (``DFMKalmanFilter``) against an independent reference.

The reference (``_reference_kf``) is a dense float64 Kalman filter with
per-element missing data, a Joseph update and the exact Gaussian
log-likelihood, with RTS and Durbin-Koopman smoothers. It is validated here
against a brute-force joint-density log-likelihood and, when installed,
statsmodels' ``KalmanSmoother``.

The engine is then compared with the reference at FIXED parameters: the fitted
parameters of the DFM golden fixtures (on the standardized data the engine fed
to EM, rebuilt from the case recipes), the synthetic fixture with a few NaN,
and random models. The engine must match the reference loglik to 1e-8 and the
smoothed moments to 1e-6 (relative). ``DFM.fit`` must report the exact
loglik of the parameters it returns.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from . import _reference_kf as ref

FIXTURES = ("dfm_synthetic", "dfm_macro_monthly_quarterly", "dfm_macro_weekly_monthly")

LOGLIK_RTOL = 1e-8
MOMENT_RTOL = 1e-6


# --------------------------------------------------------------------- data


def _case_frame_and_config(case: str):
    from ..parity._cases_common import _macro_tail
    from ..parity._cases_dfm import _DFM_WM_ROWS, _dfm_macro_flat_config, _mq_config, _mq_panel, dfm_synthetic_inputs

    if case == "dfm_synthetic":
        df, _, config, _ = dfm_synthetic_inputs()
        return df, config
    if case == "dfm_macro_monthly_quarterly":
        return _mq_panel(), _mq_config()
    if case == "dfm_macro_weekly_monthly":
        return _macro_tail(_DFM_WM_ROWS), _dfm_macro_flat_config()
    raise KeyError(case)


def standardized_X(case: str) -> np.ndarray:
    """The standardized matrix the engine fed to EM for a golden case, rebuilt from its recipe.

    The frame goes through ``DFMDataset`` (the engine's column order) and a
    fresh ``StandardScaler``, as in ``_cases_dfm._dfm_fit``; the result must equal
    the fixture's ``data_processed``.
    """
    from sklearn.preprocessing import StandardScaler

    from dfm_python.config.schema.model import _DFMEngineConfig
    from dfm_python.dataset.dfm_dataset import DFMDataset

    from ..parity._golden import load_golden

    df, config_dict = _case_frame_and_config(case)
    config = _DFMEngineConfig.from_dict(dict(config_dict))
    dataset = DFMDataset(config=config, data=df, time_index="date")
    X = StandardScaler().fit_transform(dataset.variables.to_numpy(np.float64))
    _, arrays = load_golden(case)
    np.testing.assert_allclose(X, arrays["data_processed"], rtol=1e-12, atol=1e-12)
    return X


def fixture_params(case: str) -> tuple[np.ndarray, ...]:
    """The fitted ``(A, C, Q, R, Z_0, V_0)`` of a golden case."""
    from ..parity._golden import load_golden

    _, arrays = load_golden(case)
    return tuple(np.asarray(arrays[k], dtype=np.float64) for k in ("A", "C", "Q", "R", "Z_0", "V_0"))


def _with_nans(X: np.ndarray) -> np.ndarray:
    """``X`` with one NaN at rows 0, 40 and T-1 (a different column each)."""
    X = X.copy()
    T, N = X.shape
    for k, t in enumerate((0, 40, T - 1)):
        X[t, k % N] = np.nan
    return X


def _random_full():
    A, C, Q, R, Z0, V0, Y = ref.random_model(m=4, N=6, T=60, nan_frac=0.0, seed=11)
    Y = Y.copy()
    Y[7] = np.nan  # random_model blanks row 7; refill so the panel is fully observed
    rng = np.random.default_rng(12)
    Y[7] = C @ rng.standard_normal(4) + rng.standard_normal(6)
    return A, C, Q, R, Z0, V0, Y


def model_and_data(name: str):
    """``(A, C, Q, R, Z_0, V_0, X)`` of an oracle case."""
    if name in FIXTURES:
        return (*fixture_params(name), standardized_X(name))
    if name == "dfm_synthetic_nan":
        return (*fixture_params("dfm_synthetic"), _with_nans(standardized_X("dfm_synthetic")))
    if name == "random_40pct_nan":
        return ref.random_model(m=4, N=5, T=60, nan_frac=0.4, seed=3)
    if name == "random_full":
        return _random_full()
    raise KeyError(name)


def engine_estep(A, C, Q, R, Z0, V0, X):
    """``DFMKalmanFilter.update_parameters`` + ``filter_and_smooth`` on the masked data: ``(EZ, V, VV, loglik)``."""
    from dfm_python.ssm.kalman import DFMKalmanFilter

    kf = DFMKalmanFilter()
    kf.update_parameters(A, C, Q, R, Z0, V0)
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            EZ, V, VV, ll = kf.filter_and_smooth(np.ma.masked_invalid(X))
    finally:
        logging.disable(logging.NOTSET)
    return np.asarray(EZ), np.asarray(V), np.asarray(VV), float(ll)


def _rel_scalar(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300) if np.isfinite(a) and np.isfinite(b) else float("inf")


# --------------------------------------------- the reference against oracles


def test_reference_matches_brute_force_loglik():
    """Filter loglik equals the joint density of the observed entries (random m=4, N=5, T=12, 40% NaN)."""
    model = ref.random_model(m=4, N=5, T=12, nan_frac=0.4, seed=0)
    Y = model[-1]
    missing = np.isnan(Y)
    assert 0.2 < missing.mean() < 0.6  # partially observed rows
    assert missing.all(axis=1).any()  # and one row with nothing observed
    ll = ref.loglik(*model)
    bf = ref.brute_force_loglik(*model)
    assert _rel_scalar(ll, bf) < 1e-10, (ll, bf)


def test_reference_smoothers_agree():
    """RTS (solve on the predicted covariance) and DK (no inverse) give the same moments."""
    model = ref.random_model(m=5, N=6, T=80, nan_frac=0.4, seed=4)
    dk = ref.filter_smooth(*model, smoother="dk")
    rts = ref.filter_smooth(*model, smoother="rts")
    for key in ("sm", "sP", "pair"):
        assert ref.rel_err(rts[key], dk[key]) < 1e-10, key


@pytest.mark.parametrize("name", ["random_40pct_nan", "random_full", *FIXTURES])
def test_reference_matches_statsmodels(name):
    """loglik, smoothed states and covariances and the lag-one covariance match statsmodels' KalmanSmoother."""
    pytest.importorskip("statsmodels")
    *params, X = model_and_data(name)
    mine = ref.filter_smooth(*params, X)
    sm = ref.statsmodels_smooth(*params, X)
    assert _rel_scalar(mine["ll"], sm["ll"]) < 1e-10, (mine["ll"], sm["ll"])
    assert ref.rel_err(mine["sm"], sm["sm"]) < 1e-8
    assert ref.rel_err(mine["sP"], sm["sP"]) < 1e-8
    # statsmodels' autocov[:, :, t] = Cov(x_{t+1}, x_t | Y) = pair[t + 1]
    assert ref.rel_err(mine["pair"][1:], sm["pair"][1:]) < 1e-8


# ----------------------------------------------- the engine at fixed parameters

_ENGINE_CASES = ["random_full", "dfm_synthetic", "dfm_synthetic_nan", *FIXTURES[1:], "random_40pct_nan"]


@pytest.mark.parametrize("name", _ENGINE_CASES)
def test_engine_loglik_matches_reference(name):
    """The E-step's loglik is the exact missing-data loglik of the given parameters (rel 1e-8), never -T/2."""
    *params, X = model_and_data(name)
    expected = ref.loglik(*params, X)
    *_, ll = engine_estep(*params, X)
    T = X.shape[0]
    assert ll != -T / 2
    assert _rel_scalar(ll, expected) < LOGLIK_RTOL, (ll, expected)


@pytest.mark.parametrize("name", _ENGINE_CASES)
def test_engine_moments_match_reference(name):
    """EZ, V_smooth and the lag-one VVsmooth equal the reference smoother's (rel 1e-6)."""
    *params, X = model_and_data(name)
    expected = ref.filter_smooth(*params, X)
    EZ, V, VV, _ = engine_estep(*params, X)
    assert ref.rel_err(EZ, expected["sm"]) < MOMENT_RTOL
    assert ref.rel_err(V, expected["sP"]) < MOMENT_RTOL
    # The M-step's layout: VV[t] = Cov(x_t, x_{t-1} | Y) for t >= 1
    assert VV.shape == expected["pair"].shape
    assert ref.rel_err(VV[1:], expected["pair"][1:]) < MOMENT_RTOL


# ----------------------------------------------------- the fitted loglik


def test_fit_loglik_is_exact_on_mixed_frequency():
    """DFM.fit on the monthly/quarterly case reports the exact loglik of its returned parameters."""
    from dfm_python import DFM, PanelData

    from ..parity._cases_dfm import _MQ_MONTHLY, _MQ_QUARTERLY, _mq_panel

    df = _mq_panel()
    panel = PanelData(y=df.set_index("date"), freq={**{c: "m" for c in _MQ_MONTHLY}, **{c: "q" for c in _MQ_QUARTERLY}})
    model = DFM(
        blocks={"Block_Global": {"n_factors": 2, "series": _MQ_MONTHLY + _MQ_QUARTERLY}},
        clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]}, max_iter=10, tol=1e-5, scaler="standard",
    ).fit(panel)
    res = model.result_
    X = model.engine_.data_processed
    T = X.shape[0]
    expected = ref.loglik(res.A, res.C, res.Q, res.R, res.Z_0, res.V_0, X)
    assert np.isfinite(res.loglik)
    assert res.loglik != -T / 2
    assert _rel_scalar(float(res.loglik), expected) < LOGLIK_RTOL, (res.loglik, expected)

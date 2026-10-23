"""Golden tests for DFM.

``test_golden_dfm_engine`` runs the engine (the pre-redesign class, now
``DFMEngine``) exactly as the fixtures were captured, update path included
(the fixtures were captured with a test-only import shim around ``update``,
whose broken import the engine move fixed), and must reproduce every output
bitwise on the capture platform.

``test_golden_dfm_estimator`` runs the public ``DFM`` estimator on the same
data as ``PanelData`` (the series in the user's column order, their
frequencies in ``freq``), built from the constructor and from a YAML file with
the recorded hyperparameters under the canonical names. The engine it builds
must hold the recorded engine config (every field, the EM config, the block
structure) when EM starts and reproduce every golden output of the fit
(bitwise on the capture platform, within the ``_golden`` tolerances
elsewhere); the estimator's ``result_``, ``factors_``, ``scaler_`` and
diagnostics must equal that engine's numbers bitwise, ``internal_feature_names_``
the fixture's columns, ``predict(6)`` the engine's forecast reordered to the
user's columns, and ``predict_factors(6)`` its state forecast. The permuted case passes
the series in the permuted order of its fixture; the estimator's internal
order is the fixture's dataset order. For the cases with update rows, the
fixture's update path is reproduced through the estimator: ``predict(6,
data=new_rows)`` (without changing the model's state hash), ``update(new_rows)``
on a copy (the engine's appended smoothed state and data, then the forecast)
and ``update(new_rows, refit=True)`` on another copy (the refitted parameters,
diagnostics, smoothed state and forecast).

``test_golden_dfm_save_load`` fits the estimator, saves it, loads it back
(``DFM.load`` and ``dfm_python.load``, without data) and requires the loaded
model to report the same numbers: ``predict``, ``predict_factors``,
``predict(data=)`` and ``result_`` bitwise equal to the fitted model's, every
golden output of the fit reproduced from the loaded engine alone, the saved
last smoothed covariance equal to the fixture's last filtered covariance, and
the golden update and refit outputs from the loaded model.

Every DFM case is compared across platforms (within the ``_golden``
tolerances off the capture platform); none is registered with
``cross_platform=False``.

The DFM fixtures were recaptured after the numerical-stability rewrite (exact
per-element Kalman filter and DK smoother with the exact log-likelihood, the
Banbura-Modugno M-step, the BM14 state layout, the EM driver on the exact
log-likelihood, and update/predict(data=) continuing the training filter); see
the CHANGELOG. The DDFM, iVDFM and AFM fixtures were not recaptured.
"""

from __future__ import annotations

import copy
import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

import dfm_python
from dfm_python import DFM, PanelData
from dfm_python.config import DFMConfig
from dfm_python.models.dfm._engine import DFMEngine
from dfm_python.testing import _state_hash

from ._cases import (
    _DFM_WM_ROWS,
    _MQ_MONTHLY,
    _MQ_PERMUTED,
    _MQ_QUARTERLY,
    _MQ_UPDATE_ROWS,
    DFM_HORIZON,
    DFM_UPDATE_KEYS,
    _dfm_hyper,
    _digest,
    _macro_tail,
    _mq_panel,
    cases_for,
    check_case,
    dfm_fit_outputs,
    dfm_synthetic_inputs,
    normalized,
)
from ._golden import assert_golden, load_golden

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_engine(case, pinned):
    check_case(case.name, case.run())


def _frame(case: str, recipe: dict) -> pd.DataFrame:
    """The frame the case fitted (a ``date`` column, then the series in the case's order)."""
    frames = {
        "dfm_synthetic": lambda: dfm_synthetic_inputs()[0],
        "dfm_macro_weekly_monthly": lambda: _macro_tail(_DFM_WM_ROWS),
        "dfm_macro_monthly_quarterly": _mq_panel,
        "dfm_macro_monthly_quarterly_permuted": lambda: _mq_panel()[["date"] + _MQ_PERMUTED],
        "dfm_macro_monthly_quarterly_update": lambda: _mq_panel().iloc[:-_MQ_UPDATE_ROWS],
    }
    df = frames[case]()
    assert _digest(df) == recipe["input_sha256"], f"{case}: the frame differs from the fixture's input"
    return df


def _as_panel(df: pd.DataFrame, manifest: dict) -> PanelData:
    y = df.set_index("date")
    frequency = manifest["hyperparameters"]["config"]["frequency"]
    return PanelData(y=y, freq={name: frequency[name] for name in y.columns})


def _panel(case: str, manifest: dict) -> PanelData:
    """The fitted frame as PanelData: the series in the case's column order, their frequencies in ``freq``."""
    return _as_panel(_frame(case, manifest["data"]), manifest)


def _new_rows(case: str, manifest: dict) -> PanelData | None:
    """The case's update rows as PanelData (same columns and freq as the fitted data), or None."""
    recipe = manifest["data"]
    if case == "dfm_synthetic":
        df = dfm_synthetic_inputs()[1]
        assert _digest(df) == recipe["update_rows"]["input_sha256"]
    elif case == "dfm_macro_monthly_quarterly_update":
        df = _mq_panel().iloc[-_MQ_UPDATE_ROWS:]
        assert _digest(df) == recipe["update_input_sha256"]
    else:
        return None
    return _as_panel(df, manifest)


def _internal(frame: pd.DataFrame, columns: list) -> np.ndarray:
    """A forecast frame's values with the series in the engine's (fixture) order."""
    return frame[columns].to_numpy()


def _assert_outputs(outputs: dict, case: str, fixture) -> None:
    """Compare ``outputs`` with the fixture's outputs of the same names (bitwise on the capture platform)."""
    assert_golden(outputs, case, exclude=[key for key in fixture if key not in outputs])


def _check_update_path(model: DFM, new: PanelData, case: str, fixture) -> None:
    """The fixture's update path through the estimator: predict(data=), update and update(refit=True)."""
    columns = [str(c) for c in fixture["columns"]]
    before = _state_hash(model)
    conditioned = model.predict(DFM_HORIZON, data=new)
    conditioned_factors = model.predict_factors(DFM_HORIZON, data=new)
    assert _state_hash(model) == before
    assert conditioned.index[0] > new.index[-1] and conditioned.index.equals(conditioned_factors.index)
    _assert_outputs({"predict_data_X": _internal(conditioned, columns),
                     "predict_data_Z": conditioned_factors.to_numpy()}, case, fixture)

    updated = copy.deepcopy(model)
    result, factors = updated.result_, updated.factors_.copy()
    assert updated.update(new) is updated
    assert updated.result_ is result and updated.factors_.equals(factors)
    assert updated.conditioning_index_.equals(model.conditioning_index_.append(new.index))
    _assert_outputs({
        "update_Z": updated.engine_.get_result().Z,
        "update_data_processed": updated.engine_.data_processed,
        "update_predict_X": _internal(updated.predict(DFM_HORIZON), columns),
        "update_predict_Z": updated.predict_factors(DFM_HORIZON).to_numpy(),
    }, case, fixture)

    refitted = copy.deepcopy(model)
    assert refitted.update(new, refit=True) is refitted
    engine = refitted.engine_
    refit = engine.get_result()
    _assert_outputs({
        "refit_A": engine.A, "refit_C": engine.C, "refit_Q": engine.Q, "refit_R": engine.R,
        "refit_Z_0": engine.Z_0, "refit_V_0": engine.V_0,
        "refit_loglik": np.float64(refit.loglik),
        "refit_num_iter": np.int64(refit.num_iter),
        "refit_converged": np.bool_(refit.converged),
        "refit_result_Z": refitted.result_.Z,
        "refit_predict_X": _internal(refitted.predict(DFM_HORIZON), columns),
        "refit_predict_Z": refitted.predict_factors(DFM_HORIZON).to_numpy(),
    }, case, fixture)
    # The estimator reports the refitted engine's diagnostics (compared with the fixture above).
    assert refitted.n_iter_ == refit.num_iter and refitted.converged_ == refit.converged
    assert np.float64(refitted.loglik_).tobytes() == np.float64(refit.loglik).tobytes()
    assert refitted.factors_.index.equals(refitted.conditioning_index_)


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the config the engine held in the fixture."""
    config = hyper["config"]
    assert hyper["scaler"] == "StandardScaler"
    return {
        "blocks": {name: {"n_factors": block["num_factors"], "series": list(block["series"])}
                   for name, block in config["blocks"].items()},
        "clock": config["clock"],
        "tent_weights": config["tent_weights"],
        "max_iter": config["max_iter"],
        "tol": config["threshold"],
        "ar_clip": config["ar_clip"],
        "data_clip": config["data_clip"],
        "regularization": config["regularization"],
        "damping_factor": config["damping_factor"],
        "em_acceleration": None,  # the fixtures were captured with plain EM
        "scaler": "standard",
    }


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's config and structure (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = DFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_dfm_hyper(self)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFMEngine, "fit", fit)
    return seen


def _build(build: str, params: dict, tmp_path) -> DFM:
    if build == "constructor":
        return DFM(**params)
    if not _YAML:
        pytest.skip("YAML needs omegaconf or PyYAML")
    path = tmp_path / "dfm.yaml"
    DFMConfig(**params).to_yaml(path)
    return DFM.from_config(DFMConfig.from_yaml(path))


@pytest.mark.parametrize("build", ["constructor", "yaml"])
@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_estimator(case, build, pinned, engine_at_fit, tmp_path):
    manifest, fixture = load_golden(case.name)
    assert manifest["raised"] is None
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data = _panel(case.name, manifest)
    user_order = list(data.series_names)
    if case.name == "dfm_macro_monthly_quarterly_permuted":
        assert user_order == _MQ_PERMUTED
    elif case.name.startswith("dfm_macro_monthly_quarterly"):
        assert user_order == _MQ_MONTHLY + _MQ_QUARTERLY

    model = _build(build, params, tmp_path)
    assert model.get_params() == {**DFM().get_params(), **params}
    assert list(model.blocks) == list(params["blocks"])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert model.fit(data) is model
    # Every golden fit is stationary (spectral radius of A below 1): no non-stationarity warning.
    assert float(np.max(np.abs(np.linalg.eigvals(model.result_.A)))) < 1.0
    assert [str(w.message) for w in record if "spectral radius" in str(w.message)] == []

    # The engine held the recorded config, EM config and block structure when EM started.
    assert engine_at_fit == [hyper]

    # The engine reproduces every pinned output of the fit (the update path is checked below).
    engine = model.engine_
    X, Z = engine.predict(horizon=DFM_HORIZON)
    outputs = dfm_fit_outputs(engine, X, Z)
    outputs["columns"] = np.array(list(model.internal_feature_names_), dtype=str)
    assert_golden(outputs, case.name, exclude=[key for key in DFM_UPDATE_KEYS if key in fixture])

    # The internal order is the fixture's dataset order.
    columns = [str(c) for c in fixture["columns"]]
    assert list(model.internal_feature_names_) == columns
    assert list(model.feature_names_in_) == user_order

    # result_ is a copy of the engine's result, in the internal order: the same
    # numbers, bitwise, as the engine outputs compared with the fixture above.
    result = model.result_
    assert result is not engine.get_result()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0"):
        np.testing.assert_array_equal(getattr(result, name), outputs[name], strict=True)
    np.testing.assert_array_equal(result.Z, outputs["result_Z"], strict=True)
    np.testing.assert_array_equal(result.x_sm, outputs["result_x_sm"], strict=True)
    assert np.float64(result.loglik).tobytes() == outputs["loglik"].tobytes()
    np.testing.assert_array_equal(model.factors_.to_numpy(), outputs["result_Z"].astype(np.float64), strict=True)
    assert list(model.factors_.columns) == [f"state_{i}" for i in range(fixture["result_Z"].shape[1])]
    assert model.factors_.index.equals(data.index)
    np.testing.assert_array_equal(model.scaler_.mean_, outputs["scaler_mean"], strict=True)
    np.testing.assert_array_equal(model.scaler_.scale_, outputs["scaler_scale"], strict=True)
    assert model.n_iter_ == int(outputs["num_iter"])
    assert model.converged_ == bool(outputs["converged"])
    assert np.float64(model.loglik_).tobytes() == outputs["loglik"].tobytes()
    assert model.n_factors_ == hyper["num_factors"]
    assert model.freq_ == {name: hyper["config"]["frequency"][name] for name in user_order}

    # predict(h): the engine's forecast (compared with the fixture above) with the columns in the user's order.
    forecast = model.predict(DFM_HORIZON)
    expected = pd.DataFrame(outputs["predict_X"], columns=columns)[user_order]
    assert list(forecast.columns) == user_order
    assert forecast.to_numpy().tobytes() == expected.to_numpy().tobytes()
    assert all(dtype == np.float64 for dtype in forecast.dtypes)
    assert forecast.index[0] > data.index[-1] and len(forecast.index) == DFM_HORIZON

    factors = model.predict_factors(DFM_HORIZON)
    np.testing.assert_array_equal(factors.to_numpy(), outputs["predict_Z"], strict=True)
    assert list(factors.columns) == list(model.factors_.columns)
    assert factors.index.equals(forecast.index)

    new = _new_rows(case.name, manifest)
    assert (new is None) == ("update_Z" not in fixture)
    if new is not None:
        _check_update_path(model, new, case.name, fixture)


@pytest.mark.parametrize("loader", ["DFM.load", "dfm_python.load"])
@pytest.mark.parametrize("case", cases_for("dfm"), ids=repr)
def test_golden_dfm_save_load(case, loader, pinned, tmp_path):
    manifest, fixture = load_golden(case.name)
    data = _panel(case.name, manifest)
    new = _new_rows(case.name, manifest)
    model = DFM(**_estimator_params(manifest["hyperparameters"])).fit(data)
    outputs = [model.predict(DFM_HORIZON), model.predict_factors(DFM_HORIZON)]
    if new is not None:
        outputs += [model.predict(DFM_HORIZON, data=new), model.predict_factors(DFM_HORIZON, data=new)]
    path = tmp_path / "run"
    model.save(path)
    files = {p.name for p in path.iterdir()}
    assert "scaler.pkl" not in files and "weights.pt" not in files and "scaler.json" in files
    with np.load(path / "state.npz", allow_pickle=False) as npz:
        # The covariance update() starts from is the fixture's last filtered covariance.
        _assert_outputs({"last_filtered_cov": npz["data/last_smoothed_cov"]}, case.name, fixture)

    loaded = (DFM.load if loader == "DFM.load" else dfm_python.load)(path)
    assert type(loaded) is DFM and loaded.get_params() == model.get_params()
    reloaded = [loaded.predict(DFM_HORIZON), loaded.predict_factors(DFM_HORIZON)]
    if new is not None:
        reloaded += [loaded.predict(DFM_HORIZON, data=new), loaded.predict_factors(DFM_HORIZON, data=new)]
    for got, expected in zip(reloaded, outputs):
        assert got.index.equals(expected.index) and list(got.columns) == list(expected.columns)
        assert got.to_numpy().tobytes() == expected.to_numpy().tobytes() and got.attrs == expected.attrs
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z", "x_sm"):
        assert getattr(loaded.result_, name).tobytes() == getattr(model.result_, name).tobytes(), name
    assert np.float64(loaded.result_.loglik).tobytes() == np.float64(model.result_.loglik).tobytes()
    assert loaded.result_.num_iter == model.result_.num_iter and loaded.result_.converged == model.result_.converged
    assert loaded.factors_.equals(model.factors_)
    del model

    # The loaded engine alone reproduces every golden output of the fit.
    engine = loaded.engine_
    X, Z = engine.predict(horizon=DFM_HORIZON)
    fit_outputs = dfm_fit_outputs(engine, X, Z)
    fit_outputs["columns"] = np.array(list(loaded.internal_feature_names_), dtype=str)
    assert_golden(fit_outputs, case.name, exclude=[key for key in DFM_UPDATE_KEYS if key in fixture])
    # ... and the loaded model the golden update path.
    if new is not None:
        _check_update_path(loaded, new, case.name, fixture)

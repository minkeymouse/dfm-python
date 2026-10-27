"""Golden tests for DDFM.

``test_golden_ddfm_engine`` runs the engine (the pre-redesign class, now
``DDFMEngine``) as the fixtures were captured, minus the removed entry points:
no ``config`` argument (the capture passed the values it read explicitly) and
``_update(dataset)`` followed by ``predict(horizon)`` for the old mutating
``predict(horizon, data=dataset)``, which ran the same calls in that order.
``test_golden_ddfm_estimator`` runs the public ``DDFM`` estimator, built from
the constructor and from a YAML file, on the same data as ``PanelData``
(covariates in ``exog``) with the recorded hyperparameters mapped to the
canonical names (``device="cpu"``). The engine it builds must hold the
recorded hyperparameters when training starts and reproduce every golden
output of the fit (bitwise on the capture platform, within the ``_golden``
tolerances elsewhere); the estimator's ``predict``, ``predict_factors``,
``result_.C`` and ``factors_`` must equal that engine's forecasts, loadings and
factors bitwise. For the cases with update rows (``ddfm_exchange`` and
``ddfm_finance``: the last 12 rows of the panel, after the fitted ones), the
fixture's update path (the engine's ``_update`` on those rows) is reproduced
twice: by the pure ``predict(h, data=new_rows)``, which leaves the model
unchanged, and by ``update(new_rows)`` followed by ``predict(h)``, which must
also give the fixture's appended factors and the unchanged target scaler.
``ddfm_macro`` has no rows after the fitted ones; there an update with the
fitted rows must raise ``DataValidationError``.
``test_golden_ddfm_save_load`` fits the estimator, saves it, loads it back
(``DDFM.load`` and ``dfm_python.load``) and requires the loaded model to report
the same numbers: ``predict`` and ``predict(data=...)`` bitwise equal to the
fitted model's, every golden output of the fit reproduced from the loaded
engine alone, and the golden update outputs from an update of the loaded model.

The update outputs (``DDFM_UPDATE_KEYS``) were recaptured before the 0.8.0
release, when the update stopped refitting the target scaler and started
taking only the rows after the conditioning data; the fit outputs are
unchanged.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import dfm_python
from dfm_python import DDFM, PanelData
from dfm_python.config import DDFMConfig
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.testing import _state_hash
from dfm_python.utils.errors import DataValidationError

from ._cases import cases_for, check_case
from ._cases_common import _attrs, _digest, _macro_tail, _reset_global_rngs, normalized
from ._cases_ddfm import (
    _DDFM_ATTRS,
    DDFM_UPDATE_KEYS,
    _DDFM_HOLDOUT,
    _DDFM_HORIZON,
    _DDFM_ROWS,
    _FINANCE_TARGET,
    _ddfm_hyper,
    ddfm_engine_outputs,
    ddfm_exchange_panel,
    ddfm_finance_panel,
)
from ._golden import assert_golden, load_golden

_SCALER_NAMES = {None: None, "StandardScaler": "standard"}
_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the hyperparameters the engine held in the fixture."""
    encoder_size = hyper["encoder_size"]
    return {
        "n_factors": encoder_size[-1],
        "encoder_layers": tuple(encoder_size[:-1]),
        "decoder_type": hyper["decoder_type"],
        "activation": hyper["activation"],
        "learning_rate": hyper["learning_rate"],
        "optimizer": hyper["optimizer_type"],
        "n_mc_samples": hyper["n_mc_samples"],
        "window_size": hyper["window_size"],
        "max_iter": hyper["max_iter"],
        "tol": hyper["tolerance"],
        "interpolation_method": hyper["interpolation_method"],
        "interpolation_limit": hyper["interpolation_limit"],
        "interpolation_limit_direction": hyper["interpolation_limit_direction"],
        "scaler": _SCALER_NAMES[hyper["scaler"]],
        "random_state": hyper["initializer_seed"],
        "device": "cpu",
    }


def _panels(case: str, manifest: dict) -> tuple[PanelData, PanelData | None]:
    """The data the case fitted and the new rows of its update (None without), as PanelData.

    The targets are in ``y`` and the covariates in ``exog``. ddfm_macro has no
    rows after the 400 it fitted; the other cases fit on all but the last rows
    of their panel and update with those rows.
    """
    recipe = manifest["data"]
    if case == "ddfm_macro":
        df = _macro_tail(_DDFM_ROWS)
        assert _digest(df) == recipe["input_sha256"]
        frame = df.set_index("date")
        return PanelData(y=frame[[recipe["target"]]], exog=frame[recipe["covariates"]]), None
    if case == "ddfm_exchange":
        panel = ddfm_exchange_panel()
        fit_df, new_df = panel.iloc[:-_DDFM_HOLDOUT], panel.iloc[-_DDFM_HOLDOUT:]
        assert _digest(fit_df) == recipe["fit_input_sha256"] and _digest(new_df) == recipe["update_input_sha256"]
        assert recipe["covariates"] == [] and recipe["targets"] == list(fit_df.columns)
        return PanelData(y=fit_df), PanelData(y=new_df)
    if case == "ddfm_finance":
        panel, covariates = ddfm_finance_panel()
        fit_df, new_df = panel.iloc[:-_DDFM_HOLDOUT], panel.iloc[-_DDFM_HOLDOUT:]
        assert _digest(fit_df) == recipe["fit_input_sha256"] and _digest(new_df) == recipe["update_input_sha256"]
        assert recipe["targets"] == [_FINANCE_TARGET] and recipe["covariates"] == covariates

        def as_panel(df):
            frame = df.set_index("date")
            return PanelData(y=frame[[_FINANCE_TARGET]], exog=frame[covariates])

        return as_panel(fit_df), as_panel(new_df)
    raise AssertionError(f"no PanelData recipe for {case}")


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's hyperparameters (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = DDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_ddfm_hyper(self)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "fit", fit)
    return seen


def _build(build: str, params: dict, tmp_path) -> DDFM:
    if build == "constructor":
        return DDFM(**params)
    if not _YAML:
        pytest.skip("YAML needs omegaconf or PyYAML")
    path = tmp_path / "ddfm.yaml"
    DDFMConfig(**params).to_yaml(path)
    return DDFM.from_config(DDFMConfig.from_yaml(path))


def _as_float64_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values).astype(np.float64).tobytes()


def _update_outputs(model: DDFM) -> dict:
    """The fixture's update outputs, read from a model after ``update(new_rows)``."""
    engine = model.network_
    X, Z = engine._forecast(_DDFM_HORIZON)
    return {
        "update_predict_X": X,
        "update_predict_Z": Z,
        "update_factors": engine.factors,
        "update_scaler_mean": model.scaler_.mean_,
        "update_scaler_scale": model.scaler_.scale_,
    }


def _fit_keys(fixture) -> list:
    return [key for key in fixture if key not in DDFM_UPDATE_KEYS]


def _update_keys(fixture) -> list:
    return [key for key in DDFM_UPDATE_KEYS if key in fixture]


@pytest.mark.parametrize("build", ["constructor", "yaml"])
@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_estimator(case, build, pinned, engine_at_fit, tmp_path):
    manifest, fixture = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data, new = _panels(case.name, manifest)
    assert (new is None) == (not _update_keys(fixture))

    model = _build(build, params, tmp_path)
    assert model.get_params() == {**DDFM().get_params(), **params}

    _reset_global_rngs()
    assert model.fit(data) is model

    # Every hyperparameter reached the engine unchanged (read when its training starts).
    assert engine_at_fit == [hyper]

    # The engine reproduces every pinned output of the fit; the update outputs
    # (an engine _update, which mutates it) are left out here.
    outputs = ddfm_engine_outputs(model.network_)
    assert_golden(outputs, case.name, exclude=_update_keys(fixture))

    # The estimator's forecasts are the engine's (compared with the fixture above), in data units.
    forecast = model.predict(_DDFM_HORIZON)
    assert forecast.to_numpy().tobytes() == _as_float64_bytes(outputs["predict_X"])
    assert forecast.shape == fixture["predict_X"].shape
    assert list(forecast.columns) == list(data.series_names)
    assert all(dtype == np.float64 for dtype in forecast.dtypes)
    factors = model.predict_factors(_DDFM_HORIZON)
    assert factors.to_numpy().tobytes() == _as_float64_bytes(outputs["predict_Z"])
    assert factors.shape == fixture["predict_Z"].shape
    assert forecast.index.equals(factors.index) and forecast.index[0] > data.index[-1]

    # result_ and the fitted attributes come from the same run.
    np.testing.assert_array_equal(model.result_.C, outputs["H"], strict=True)
    np.testing.assert_array_equal(model.result_.Z, outputs["result_Z"], strict=True)
    np.testing.assert_array_equal(
        model.factors_.to_numpy(), outputs["result_Z"].astype(np.float64), strict=True
    )
    assert model.factors_.index.equals(data.index)
    assert model.n_iter_ == int(outputs["num_iter"])
    assert model.converged_ == bool(outputs["converged"])
    assert model.n_factors_ == hyper["encoder_size"][-1]
    np.testing.assert_array_equal(model.scaler_.mean_, outputs["scaler_mean"], strict=True)
    np.testing.assert_array_equal(model.scaler_.scale_, outputs["scaler_scale"], strict=True)

    if new is None:
        # No rows after the fitted ones: an update with the fitted rows overlaps them.
        before = _state_hash(model)
        with pytest.raises(DataValidationError, match="overlaps"):
            model.update(data)
        with pytest.raises(DataValidationError, match="overlaps"):
            model.predict(_DDFM_HORIZON, data=data)
        assert _state_hash(model) == before
        return

    # The fixture's update path, without mutating the model: predict(h, data=new rows).
    before = _state_hash(model)
    conditioned = model.predict(_DDFM_HORIZON, data=new)
    conditioned_factors = model.predict_factors(_DDFM_HORIZON, data=new)
    assert _state_hash(model) == before
    assert conditioned.index[0] > new.index[-1] and conditioned.index.equals(conditioned_factors.index)
    np.testing.assert_array_equal(model.scaler_.mean_, outputs["scaler_mean"], strict=True)

    # ... and with it: update(new rows), then predict(h). The pure predict(data=)
    # above gave the same numbers, bitwise, as the update compared with the fixture.
    result, factors = model.result_, model.factors_.copy()
    assert model.update(new) is model
    updated = _update_outputs(model)
    assert_golden(updated, case.name, exclude=_fit_keys(fixture))
    assert conditioned.to_numpy().tobytes() == _as_float64_bytes(updated["update_predict_X"])
    assert conditioned_factors.to_numpy().tobytes() == _as_float64_bytes(updated["update_predict_Z"])
    assert model.predict(_DDFM_HORIZON).equals(conditioned)
    assert model.predict_factors(_DDFM_HORIZON).equals(conditioned_factors)
    assert model.result_ is result and model.result_.target_scaler is model.scaler_
    assert model.factors_.equals(factors) and model.conditioning_index_.equals(data.index.append(new.index))


@pytest.mark.parametrize("loader", ["DDFM.load", "dfm_python.load"])
@pytest.mark.parametrize("case", cases_for("ddfm"), ids=repr)
def test_golden_ddfm_save_load(case, loader, pinned, tmp_path):
    manifest, fixture = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    data, new = _panels(case.name, manifest)

    _reset_global_rngs()
    model = DDFM(**_estimator_params(hyper)).fit(data)
    forecast = model.predict(_DDFM_HORIZON)
    pairs = [(lambda m: m.predict(_DDFM_HORIZON), forecast)]
    if new is not None:
        pairs.append((lambda m: m.predict(_DDFM_HORIZON, data=new), model.predict(_DDFM_HORIZON, data=new)))
    model.save(tmp_path / "run")
    del model

    loaded = (DDFM.load if loader == "DDFM.load" else dfm_python.load)(tmp_path / "run", device="cpu")
    assert type(loaded) is DDFM
    for got, expected in ((call(loaded), expected) for call, expected in pairs):
        assert got.index.equals(expected.index) and list(got.columns) == list(expected.columns)
        assert got.to_numpy().tobytes() == expected.to_numpy().tobytes()
        assert got.attrs == expected.attrs

    # The loaded engine alone reproduces the golden numbers of the fit and holds the recorded hyperparameters.
    engine = loaded.network_
    assert_golden(ddfm_engine_outputs(engine), case.name, exclude=_update_keys(fixture))
    assert normalized({**_attrs(engine, _DDFM_ATTRS), "scaler": type(engine.scaler).__name__}) == hyper
    # An update of the loaded model reproduces the golden update (it interpolates against the saved frame
    # and scales with the saved scalers).
    if new is not None:
        loaded.update(new)
        assert_golden(_update_outputs(loaded), case.name, exclude=_fit_keys(fixture))


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_values_reach_the_engine(tmp_path, engine_at_fit):
    """Spec section 14: a DDFM YAML learning rate reaches the optimizer, and every value the engine."""
    path = tmp_path / "ddfm.yaml"
    path.write_text(
        "model:\n"
        "  n_factors: 2\n  encoder_layers: [6, 4]\n  decoder_type: mlp\n  learning_rate: 0.0123\n"
        "  optimizer: AdamW\n  n_mc_samples: 2\n  window_size: 16\n  max_iter: 2\n  tol: 0.001\n"
        "  interpolation_limit: 7\n  interpolation_limit_direction: forward\n  scaler: robust\n"
        "  random_state: 5\n  device: cpu\n",
        encoding="utf-8",
    )
    model = DDFM.from_config(DDFMConfig.from_yaml(path))
    assert model.encoder_layers == (6, 4) and model.learning_rate == 0.0123
    y = np.random.default_rng(0).standard_normal((40, 3))
    engine = model.fit(PanelData.from_arrays(y)).network_
    held = engine_at_fit[0]
    assert held["learning_rate"] == 0.0123 and held["encoder_size"] == [6, 4, 2] and held["decoder_size"] == [4, 6]
    assert held["optimizer_type"] == "AdamW" and held["tolerance"] == 0.001 and held["initializer_seed"] == 5
    assert held["interpolation_limit"] == 7 and held["interpolation_limit_direction"] == "forward"
    assert held["n_mc_samples"] == 2 and held["window_size"] == 16 and held["max_iter"] == 2
    assert held["scaler"] == "RobustScaler"
    assert type(engine.optimizer).__name__ == "AdamW"
    assert [group["initial_lr"] for group in engine.optimizer.param_groups] == [0.0123]
    assert engine.scheduler.base_lrs == [0.0123]

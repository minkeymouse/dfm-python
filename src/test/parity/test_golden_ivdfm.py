"""Golden tests for iVDFM.

``test_golden_ivdfm_engine`` runs the engine (the pre-redesign class, now
``iVDFMEngine``) exactly as the fixtures were captured.
``test_golden_ivdfm_estimator`` runs the public ``iVDFM`` estimator, built from
the constructor and from a YAML file, on the same data with the recorded
hyperparameters (mapped to the canonical names, ``device="cpu"``). The engine
it builds must hold the recorded hyperparameters when training starts,
reproduce every golden output (bitwise on the capture platform, within the
``_golden`` tolerances elsewhere), and the estimator's ``predict`` must equal
that engine's model-space forecast inverse-transformed with ``scaler_``, bitwise.
``test_golden_ivdfm_save_load`` fits the estimator, saves it, loads it back
(``iVDFM.load`` and ``dfm_python.load``) and requires the loaded model to
report the same numbers: ``predict`` bitwise equal to the fitted model's, and
every golden output reproduced from the loaded engine alone.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

import dfm_python
from dfm_python import PanelData, iVDFM
from dfm_python.config import iVDFMConfig
from dfm_python.models.ivdfm._engine import iVDFMEngine

from ._cases import cases_for, check_case
from ._cases_common import _attrs, _macro_tail, _reset_global_rngs, normalized
from ._cases_ivdfm import (
    _IVDFM_ATTRS,
    IVDFM_EXOG_HORIZON,
    _IVDFM_ROWS,
    _ivdfm_hyper,
    _ivdfm_outputs,
    ivdfm_exog_context_inputs,
    ivdfm_time_features,
)
from ._golden import assert_golden, load_golden

_SCALER_NAMES = {None: None, "StandardScaler": "standard"}
_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm_engine(case, pinned):
    check_case(case.name, case.run())


def _estimator_params(hyper: dict) -> dict:
    """Constructor parameters for the hyperparameters the engine held in the fixture."""
    ds = hyper["dataset"]
    window = hyper["window"]
    return {
        "n_factors": hyper["latent_dim"],
        "window": window,
        # stride=None is window // 2, the dataset default.
        "stride": None if ds["stride"] == max(1, ds["window"] // 2) else ds["stride"],
        "time_context": hyper["time_context"],
        "encoder_hidden_dim": hyper["encoder_hidden_dim"],
        "encoder_n_hidden_layers": hyper["encoder_n_hidden_layers"],
        "decoder_type": hyper["decoder_type"],
        "decoder_hidden_dim": hyper["decoder_hidden_dim"],
        "decoder_n_hidden_layers": hyper["decoder_n_hidden_layers"],
        "prior_hidden_dim": hyper["prior_hidden_dim"],
        "prior_n_hidden_layers": hyper["prior_n_hidden_layers"],
        "activation": hyper["activation"],
        "slope": hyper["slope"],
        "factor_order": hyper["factor_order"],
        "innovation_distribution": hyper["innovation_distribution"],
        "decoder_var": hyper["decoder_var"],
        "n_regimes": hyper["num_regimes"],
        "regime_temperature": hyper["regime_temperature"],
        "beta_kl": hyper["beta_kl"],
        "use_layer_norm": hyper["use_layer_norm"],
        "mixing": hyper["mixing"],
        "learning_rate": hyper["learning_rate"],
        "optimizer": hyper["optimizer_type"],
        "optimizer_weight_decay": hyper["optimizer_weight_decay"],
        "optimizer_momentum": hyper["optimizer_momentum"],
        "batch_size": hyper["batch_size"],
        "max_epochs": hyper["max_epochs"],
        "tol": hyper["tolerance"],
        "patience": hyper["patience"],
        "scheduler_type": hyper["scheduler_type"],
        "scheduler_step_size": hyper["scheduler_step_size"],
        "scheduler_gamma": hyper["scheduler_gamma"],
        "scheduler_patience": hyper["scheduler_patience"],
        "scheduler_factor": hyper["scheduler_factor"],
        "scheduler_min_lr": hyper["scheduler_min_lr"],
        "scaler": _SCALER_NAMES[ds["scaler"]],
        "random_state": hyper["seed"],
        "device": "cpu",
    }


def _panel(case: str, manifest: dict) -> tuple[PanelData, Any]:
    """The case's data as PanelData, and the future exog (or None)."""
    recipe = manifest["data"]
    if case == "ivdfm_macro":
        # The case also passed covariates, which the dataset drops from the
        # targets and context; the arrays it builds are the same without them.
        df = _macro_tail(_IVDFM_ROWS).set_index("date")
        assert recipe["rows"] == f"last {_IVDFM_ROWS} rows"
        return PanelData(y=df[recipe["targets"]]), None
    if case == "ivdfm_regimes":
        y = np.random.default_rng(recipe["seed"]).standard_normal((recipe["T"], recipe["N"]))
        return PanelData.from_arrays(y), None
    if case == "ivdfm_exog_context":
        df, x_future, data = ivdfm_exog_context_inputs()
        assert normalized(data) == recipe
        df = df.set_index("date")
        return PanelData(y=df[["y0", "y1", "y2"]], exog=df[["x0", "x1"]]), x_future
    raise AssertionError(f"no PanelData recipe for {case}")


@pytest.fixture
def engine_at_fit(monkeypatch):
    """Record the engine's hyperparameters (as the cases record them) when its fit starts."""
    seen: list[dict] = []
    original = iVDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.append(normalized(_ivdfm_hyper(self, self._dataset)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(iVDFMEngine, "fit", fit)
    return seen


def _build(build: str, params: dict, tmp_path) -> iVDFM:
    if build == "constructor":
        return iVDFM(**params)
    if not _YAML:
        pytest.skip("YAML needs omegaconf or PyYAML")
    path = tmp_path / "ivdfm.yaml"
    iVDFMConfig(**params).to_yaml(path)
    return iVDFM.from_config(iVDFMConfig.from_yaml(path))


def _engine_outputs(engine: iVDFMEngine, hyper: dict, x_future: Any) -> dict:
    """The golden outputs of a fitted (or loaded) engine."""
    outputs = _ivdfm_outputs(engine, horizon=IVDFM_EXOG_HORIZON)
    if x_future is not None:
        T_train = engine._factor_path().shape[0]
        context = np.hstack([ivdfm_time_features(IVDFM_EXOG_HORIZON, hyper["time_context"], T_train), x_future])
        outputs["predict_futr"] = engine.predict(horizon=IVDFM_EXOG_HORIZON, context_data=context)
    return outputs


@pytest.mark.parametrize("build", ["constructor", "yaml"])
@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm_estimator(case, build, pinned, engine_at_fit, tmp_path):
    manifest, _ = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    params = _estimator_params(hyper)
    data, x_future = _panel(case.name, manifest)

    model = _build(build, params, tmp_path)
    assert model.get_params() == {**iVDFM().get_params(), **params}

    _reset_global_rngs()
    assert model.fit(data) is model

    # Every hyperparameter reached the engine unchanged (read when its training starts).
    assert engine_at_fit == [hyper]

    outputs = _engine_outputs(model.network_, hyper, x_future)
    assert_golden(outputs, case.name)

    # The estimator's forecasts are the engine's (compared with the fixture above), in data units.
    def data_units(model_space: np.ndarray) -> np.ndarray:
        values = model_space.astype(np.float64)
        return values if model.scaler_ is None else model.scaler_.inverse_transform(values)

    if x_future is None:
        forecast = model.predict(IVDFM_EXOG_HORIZON)
    else:
        with pytest.warns(UserWarning, match="futr_exog"):
            forecast = model.predict(IVDFM_EXOG_HORIZON)
        futr = model.predict(IVDFM_EXOG_HORIZON, futr_exog=pd.DataFrame(x_future, columns=["x0", "x1"]))
        assert futr.to_numpy().tobytes() == data_units(outputs["predict_futr"]).tobytes()
        np.testing.assert_array_equal(
            model.predict(IVDFM_EXOG_HORIZON, futr_exog=x_future).to_numpy(), futr.to_numpy(), strict=True
        )
    assert forecast.to_numpy().tobytes() == data_units(outputs["predict"]).tobytes()
    assert list(forecast.columns) == list(data.series_names)
    assert all(dtype == np.float64 for dtype in forecast.dtypes)

    # Fitted attributes come from the same run.
    assert model.n_iter_ == int(outputs["num_iter"])
    assert model.converged_ == bool(outputs["converged"])
    assert model.n_factors_ == hyper["latent_dim"]
    assert (model.window_, model.stride_) == (hyper["dataset"]["window"], hyper["dataset"]["stride"])
    assert model.context_dim_ == hyper["dataset"]["context_dim"]
    assert model.time_context_ == hyper["time_context"]
    n_windows, window, r = outputs["factors"].shape
    assert model.window_factors_.shape == (n_windows * window, r)
    # window_factors_ holds the engine's windows, reordered chronologically.
    rows = model.window_factors_.to_numpy().reshape(n_windows, window, r)
    engine_rows = {outputs["factors"][k].astype(np.float64).tobytes() for k in range(n_windows)}
    assert {rows[w].tobytes() for w in range(n_windows)} == engine_rows
    # factors_ is the engine's factor path on the training index.
    assert model.factors_.index.equals(data.index)
    assert model.factors_.to_numpy().tobytes() == model.network_.factor_path.astype(np.float64).tobytes()


@pytest.mark.parametrize("loader", ["iVDFM.load", "dfm_python.load"])
@pytest.mark.parametrize("case", cases_for("ivdfm"), ids=repr)
def test_golden_ivdfm_save_load(case, loader, pinned, tmp_path):
    manifest, _ = load_golden(case.name)
    hyper = manifest["hyperparameters"]
    data, x_future = _panel(case.name, manifest)
    futr = None if x_future is None else pd.DataFrame(x_future, columns=["x0", "x1"])

    _reset_global_rngs()
    model = iVDFM(**_estimator_params(hyper)).fit(data)
    forecast = model.predict(IVDFM_EXOG_HORIZON, futr_exog=futr)
    model.save(tmp_path / "run")
    fitted_config = dataclasses.asdict(model.network_._config)
    del model

    loaded = (iVDFM.load if loader == "iVDFM.load" else dfm_python.load)(tmp_path / "run", device="cpu")
    assert type(loaded) is iVDFM
    reloaded = loaded.predict(IVDFM_EXOG_HORIZON, futr_exog=futr)
    assert reloaded.index.equals(forecast.index) and list(reloaded.columns) == list(forecast.columns)
    assert reloaded.to_numpy().tobytes() == forecast.to_numpy().tobytes()
    assert reloaded.attrs == forecast.attrs

    # The loaded engine alone reproduces the golden numbers and holds the recorded hyperparameters.
    engine = loaded.network_
    assert_golden(_engine_outputs(engine, hyper, x_future), case.name)
    # time_context was recorded when training started; fit then sets it to the width of u_t.
    held = normalized(_attrs(engine, _IVDFM_ATTRS))
    assert held.pop("time_context") == hyper["dataset"]["context_dim"]
    assert held == {k: v for k, v in hyper.items() if k not in ("dataset", "time_context")}
    assert dataclasses.asdict(engine._config) == fitted_config


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_values_reach_the_engine(tmp_path, engine_at_fit):
    """Spec section 14: a YAML with factor_order: 3 runs AR(3), and patience: null disables early stopping."""
    path = tmp_path / "ivdfm.yaml"
    path.write_text(
        "model:\n"
        "  n_factors: 2\n  window: 12\n  factor_order: 3\n  patience: null\n  scheduler_type: null\n"
        "  encoder_hidden_dim: [8, 6]\n  decoder_hidden_dim: 8\n  prior_hidden_dim: 8\n"
        "  batch_size: 4\n  max_epochs: 2\n  random_state: 0\n  device: cpu\n",
        encoding="utf-8",
    )
    model = iVDFM.from_config(iVDFMConfig.from_yaml(path))
    assert model.factor_order == 3 and model.patience is None and model.encoder_hidden_dim == (8, 6)
    y = np.random.default_rng(0).standard_normal((30, 3))
    engine = model.fit(PanelData.from_arrays(y)).network_
    assert engine_at_fit[0]["factor_order"] == 3 and engine_at_fit[0]["patience"] is None
    assert engine.factor_order == 3 and engine._config.factor_order == 3
    assert engine.patience is None and engine._config.patience is None
    assert engine.scheduler_type is None and engine.scheduler is None
    assert engine.encoder_hidden_dim == [8, 6]

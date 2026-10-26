"""Tests for the public DDFM estimator (dfm_python.models.ddfm.estimator)."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import inspect
import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import dfm_python
import dfm_python.models
import dfm_python.models.ddfm
from dfm_python import PanelData
from dfm_python.config import DDFMConfig, load_yaml
from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.config.schema.params import DDFMModelState
from dfm_python.config.schema.results import DDFMResult
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.models.ddfm.estimator import _TIME_COLUMN, _TRAINING_BUFFERS, DDFM
from dfm_python.registry import all_estimators
from dfm_python.testing import _state_hash
from dfm_python.utils.errors import (
    ConfigurationError,
    DataValidationError,
    ModelNotInitializedError,
    NotFittedError,
    UnsupportedInputError,
)

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None

_REPO = Path(__file__).resolve().parents[3]

TINY = {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 2,
        "random_state": 0, "device": "cpu"}


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _panel(T: int = 48, N: int = 3, K: int = 1, *, index: str = "datetime", seed: int = 0) -> PanelData:
    rng = np.random.default_rng(seed)
    if index == "datetime":
        idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    else:
        idx = pd.RangeIndex(T)
    y = pd.DataFrame(rng.standard_normal((T, N)), index=idx, columns=[f"s{i}" for i in range(N)])
    y.iloc[5, 0] = np.nan
    exog = None
    if K:
        exog = pd.DataFrame(rng.standard_normal((T, K)), index=idx, columns=[f"x{k}" for k in range(K)])
    return PanelData(y=y, exog=exog)


@pytest.fixture(scope="module")
def fitted() -> DDFM:
    with torch.random.fork_rng():
        return DDFM(**TINY, scaler="standard").fit(_panel())


# -- construction ---------------------------------------------------------------


def test_defaults_are_the_values_that_ran_when_omitted():
    assert DDFM().get_params() == {
        "n_factors": 4, "encoder_layers": (16,), "decoder_type": "linear", "activation": "relu",
        "learning_rate": 0.005, "optimizer": "Adam", "n_mc_samples": 10, "window_size": 100, "max_iter": 200,
        "tol": 5e-4, "max_time": None, "interpolation_method": "linear", "interpolation_limit": 10,
        "interpolation_limit_direction": "both", "scaler": "standard", "random_state": 3, "device": "auto", "verbose": 0,
    }


def test_constructor_only_stores_and_is_keyword_only():
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    est = DDFM(n_factors=3, encoder_layers=[8, 4], device="cuda:7", scaler="nonsense")
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert vars(est) == est.get_params()
    assert est.encoder_layers == [8, 4]
    with pytest.raises(TypeError):
        DDFM(4)  # noqa
    for old in ("encoder_size", "seed", "tolerance", "disp", "dataset", "config", "max_epoch_pre_train",
                "pretrain_epochs", "min_target_interporate_ratio"):
        with pytest.raises(TypeError):
            DDFM(**{old: 1})


def test_tags():
    tags = DDFM.get_tags()
    assert tags["task"] == "forecast" and tags["backend"] == "torch"
    for cap in ("exog", "missing_values", "update", "factor_forecast", "device", "interpret"):
        assert tags[f"capability:{cap}"] is True
    assert not tags["capability:series_exog"] and not tags["capability:mixed_frequency"]
    assert not tags["capability:pred_int"]
    assert tags["update_kind"] == "encoder_forward" and tags["predict_data"] == "new_rows"
    assert tags["non_deterministic"] is False
    assert DDFM().get_tags()["non_deterministic"] is False
    assert DDFM(random_state=None).get_tags()["non_deterministic"] is True


def test_exports():
    assert dfm_python.DDFM is DDFM
    assert dfm_python.models.DDFM is DDFM
    assert dfm_python.models.ddfm.DDFM is DDFM
    assert issubclass(DDFM, dfm_python.BaseFactorModel) and DDFM is not DDFMEngine


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params, name",
    [
        ({"n_factors": 0}, "n_factors"), ({"n_factors": 2.0}, "n_factors"), ({"n_factors": True}, "n_factors"),
        ({"n_mc_samples": 0}, "n_mc_samples"), ({"window_size": -1}, "window_size"), ({"max_iter": 0}, "max_iter"),
        ({"encoder_layers": 16}, "encoder_layers"), ({"encoder_layers": (8, 0)}, "encoder_layers"),
        ({"encoder_layers": (8.0,)}, "encoder_layers"), ({"encoder_layers": "16"}, "encoder_layers"),
        ({"encoder_layers": (), "decoder_type": "mlp"}, "encoder_layers"),
        ({"decoder_type": "cnn"}, "decoder_type"), ({"activation": "gelu"}, "activation"),
        ({"optimizer": "RMSprop"}, "optimizer"), ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"), ({"tol": -1.0}, "tol"), ({"tol": float("inf")}, "tol"),
        ({"interpolation_limit": 0}, "interpolation_limit"), ({"interpolation_limit": 2.5}, "interpolation_limit"),
        ({"interpolation_limit_direction": "up"}, "interpolation_limit_direction"),
        ({"interpolation_method": ""}, "interpolation_method"), ({"interpolation_method": None}, "interpolation_method"),
        ({"scaler": "zscore"}, "scaler"), ({"scaler": object()}, "scaler"),
        ({"random_state": 1.5}, "random_state"), ({"verbose": -1}, "verbose"), ({"verbose": 1.0}, "verbose"),
        ({"device": "tpu"}, "device"),
    ],
)
def test_invalid_parameters_raise_configuration_error_naming_them(params, name, monkeypatch):
    def no_engine(*args, **kwargs):
        raise AssertionError("the engine must not be built")

    monkeypatch.setattr("dfm_python.models.ddfm.estimator.DDFMEngine", no_engine)
    est = DDFM(**{**TINY, **params})
    with pytest.raises(ConfigurationError, match=name):
        est.fit(_panel())
    assert not hasattr(est, "result_")


def test_valid_edge_parameters_are_accepted(monkeypatch):
    built = []

    class Stop(Exception):
        pass

    def engine(*args, **kwargs):
        built.append(kwargs)
        raise Stop

    monkeypatch.setattr("dfm_python.models.ddfm.estimator.DDFMEngine", engine)
    for params in ({"encoder_layers": ()}, {"encoder_layers": [np.int64(4)]}, {"interpolation_limit": None},
                   {"tol": 0}, {"random_state": None}, {"scaler": "Standard"}, {"scaler": RobustScaler()},
                   {"device": torch.device("cpu")}, {"n_factors": np.int64(2)}):
        with pytest.raises(Stop):
            DDFM(**{**TINY, **params}).fit(_panel())
    assert len(built) == 9


def test_unexpected_fit_keywords_raise():
    with pytest.raises(TypeError, match="unexpected"):
        DDFM(**TINY).fit(_panel(), epochs=3)


def test_series_exog_and_mixed_frequency_are_rejected():
    panel = _panel(K=0)
    rng = np.random.default_rng(1)
    with_series_exog = PanelData.from_arrays(rng.standard_normal((20, 2)), series_exog=rng.standard_normal((20, 2, 3)))
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        DDFM(**TINY).fit(with_series_exog)
    mixed = PanelData(y=panel.y, freq={"s0": "q"})
    with pytest.raises(UnsupportedInputError, match="mixed"):
        DDFM(**TINY).fit(mixed)


# -- the adapter and the engine call ----------------------------------------------


def test_every_hyperparameter_reaches_the_engine(monkeypatch):
    seen = {}
    original = DDFMEngine.__init__

    def init(self, *args, **kwargs):
        seen.update(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "__init__", init)
    params = {
        "n_factors": 3, "encoder_layers": (6, 5), "decoder_type": "mlp", "activation": "tanh",
        "learning_rate": 0.01, "optimizer": "AdamW", "n_mc_samples": 2, "window_size": 12, "max_iter": 1,
        "tol": 1e-3, "interpolation_method": "nearest", "interpolation_limit": 3,
        "interpolation_limit_direction": "forward", "scaler": "minmax", "random_state": 11, "device": "cpu",
    }
    model = DDFM(**params).fit(_panel())
    dataset = seen.pop("dataset")
    assert isinstance(dataset, DDFMDataset)
    assert seen == {
        "encoder_size": (6, 5, 3), "decoder_type": "mlp", "seed": 11, "activation": "tanh",
        "learning_rate": 0.01, "optimizer": "AdamW", "n_mc_samples": 2, "window_size": 12, "max_iter": 1,
        "tolerance": 1e-3, "disp": 10, "interpolation_method": "nearest", "interpolation_limit": 3,
        "interpolation_limit_direction": "forward", "device": torch.device("cpu"),
    }
    engine = model.network_
    assert engine.encoder_size == (6, 5, 3) and engine.decoder_size == (5, 6)
    assert engine.optimizer_type == "AdamW" and engine.tolerance == 1e-3 and engine.initializer_seed == 11
    assert model.scaler_ is engine.scaler and type(model.scaler_) is MinMaxScaler
    assert model.n_factors_ == 3 and model.factors_.shape == (48, 3)


def test_adapter_builds_exog_then_y_on_a_range_index():
    est = DDFM(**TINY, scaler="standard")
    # y listed first in PanelData; the engine's frame is still [exog | y].
    panel = _panel(N=2, K=2)
    ds = est._adapter(panel)
    assert list(ds.data.columns) == ["x0", "x1", "s0", "s1"]
    assert ds.covariates == ["x0", "x1"] and ds.target_series == ["s0", "s1"]
    assert isinstance(ds.data.index, pd.RangeIndex) and ds.time_idx == _TIME_COLUMN
    assert isinstance(ds.scaler, StandardScaler)
    # Missing values pass through to the engine (it interpolates and imputes them).
    assert ds.missing_y[5, 0] and ds.missing_y.sum() == 1
    # The inputs are not mutated.
    assert np.isnan(panel.y.iloc[5, 0]) and isinstance(panel.y.index, pd.DatetimeIndex)


def test_adapter_time_column_never_collides():
    y = pd.DataFrame(np.random.default_rng(0).standard_normal((20, 2)), columns=[_TIME_COLUMN, "b"])
    ds = DDFM(**TINY)._adapter(PanelData(y=y))
    assert ds.target_series == [_TIME_COLUMN, "b"] and ds.covariates == []
    assert ds.time_idx == _TIME_COLUMN + "_"


def test_adapter_without_exog_and_without_scaler():
    ds = DDFM(**TINY, scaler=None)._adapter(_panel(K=0))
    assert list(ds.data.columns) == ["s0", "s1", "s2"] and ds.covariates == [] and ds.scaler is None


def test_fit_matches_the_engine_on_the_same_frame():
    """The estimator runs the engine on [exog | y]: same numbers as the engine on that frame."""
    panel = _panel(N=2, K=2)
    torch.manual_seed(123)
    model = DDFM(**TINY, scaler="standard").fit(panel)
    frame = pd.concat([panel.exog, panel.y], axis=1).reset_index(drop=True)
    torch.manual_seed(123)
    dataset = DDFMDataset(data=frame, time_idx="none", covariates=["x0", "x1"], scaler=StandardScaler())
    engine = DDFMEngine(dataset, encoder_size=(4, 2), n_mc_samples=2, window_size=16, max_iter=2, seed=0,
                        device="cpu")
    engine.fit()
    engine.build_state_space()
    np.testing.assert_array_equal(model.network_.factors, engine.factors, strict=True)
    np.testing.assert_array_equal(model.result_.C, engine.training_state.H, strict=True)
    for (k, a), (k2, b) in zip(model.network_.state_dict().items(), engine.state_dict().items()):
        assert k == k2
        assert torch.equal(a, b), k
    X, Z = engine.predict(horizon=4)
    assert model.predict(4).to_numpy().tobytes() == X.tobytes()
    assert model.predict_factors(4).to_numpy().tobytes() == Z.tobytes()


def test_scaler_instance_is_cloned_and_a_fitted_one_warns():
    scaler = StandardScaler()
    model = DDFM(**{**TINY, "scaler": scaler}).fit(_panel())
    assert model.scaler is scaler and not hasattr(scaler, "mean_")
    assert model.scaler_ is not scaler and isinstance(model.scaler_, StandardScaler)

    fitted_scaler = StandardScaler().fit(np.ones((3, 3)))
    with pytest.warns(UserWarning, match="already fitted") as record:
        model = DDFM(**{**TINY, "scaler": fitted_scaler}).fit(_panel())
    assert record[0].filename == __file__
    np.testing.assert_array_equal(fitted_scaler.mean_, np.ones(3))
    assert model.scaler_ is not fitted_scaler


def test_device_cpu_overrides_cuda(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_available", lambda: True)
        assert DDFM(device="auto")._resolve_device() == torch.device("cuda")
        assert DDFM(device="cpu")._resolve_device() == torch.device("cpu")
    model = DDFM(**TINY).fit(_panel())
    assert model.device_ == torch.device("cpu") and model.network_.device == torch.device("cpu")
    assert all(p.device.type == "cpu" for p in model.network_.parameters())


# -- fitted state -----------------------------------------------------------------


def test_fitted_attributes(fitted):
    engine = fitted.network_
    assert isinstance(engine, DDFMEngine) and isinstance(fitted.result_, DDFMResult)
    assert fitted.scaler_ is engine.scaler and isinstance(fitted.scaler_, StandardScaler)
    assert fitted.n_iter_ == engine._num_iter == 2 and fitted.converged_ is False
    assert fitted.n_factors_ == 2 and fitted.random_state_ == 0
    assert list(fitted.feature_names_in_) == ["s0", "s1", "s2"] and list(fitted.exog_names_in_) == ["x0"]
    assert fitted.n_features_in_ == 3
    assert fitted.conditioning_index_.equals(fitted.index_) and fitted.index_.equals(_panel().index)
    assert fitted.factors_.index.equals(fitted.index_)
    assert list(fitted.factors_.columns) == ["factor_0", "factor_1"]
    assert all(dtype == np.float64 for dtype in fitted.factors_.dtypes)
    np.testing.assert_array_equal(fitted.factors_.to_numpy(), fitted.result_.Z.astype(np.float64))


def test_result_c_is_series_by_factors(fitted):
    H = fitted.network_.training_state.H
    assert fitted.result_.C.shape == (3, 2) == H.shape
    np.testing.assert_array_equal(fitted.result_.C, H)
    np.testing.assert_allclose(fitted.result_.x_sm, fitted.result_.Z @ fitted.result_.C.T, rtol=1e-6)


def test_engine_keeps_no_training_data(fitted):
    engine = fitted.network_
    assert engine._dataset is None
    for name in _TRAINING_BUFFERS:
        assert name not in vars(engine), name
    # The scaled training frame stays: update interpolates the new rows against it.
    assert list(engine.data.columns) == ["x0", "s0", "s1", "s2"]
    assert engine.target_series == ["s0", "s1", "s2"] and engine.covariates == ["x0"]


def test_mlp_decoder_fit_drops_its_last_iteration_datasets():
    model = DDFM(**{**TINY, "decoder_type": "mlp"}).fit(_panel())
    assert "_last_iter_datasets" not in vars(model.network_)
    assert model.network_.last_neurons.shape == (2, 48, 4)
    assert model.predict(2).shape == (2, 3)


def test_unfitted_access_raises_not_fitted():
    est = DDFM(**TINY)
    assert not hasattr(est, "result_") and not hasattr(est, "network_")
    for call in (lambda: est.predict(3), lambda: est.predict_factors(3)):
        with pytest.raises(NotFittedError):
            call()
    with pytest.raises(NotFittedError):
        _ = est.result_


def test_failed_fit_leaves_the_model_unfitted():
    model = DDFM(**TINY).fit(_panel())
    model.set_params(n_factors=0)
    with pytest.raises(ConfigurationError):
        model.fit(_panel())
    assert not hasattr(model, "result_") and not hasattr(model, "network_")


def test_fit_is_repeatable():
    a = DDFM(**TINY, scaler="standard").fit(_panel())
    b = DDFM(**TINY, scaler="standard").fit(_panel())
    np.testing.assert_array_equal(a.result_.Z, b.result_.Z)
    assert a.predict(3).equals(b.predict(3))


def test_fit_logs_nothing_below_warning_with_verbose_zero(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DDFM(**TINY).fit(_panel())
    assert [r for r in caplog.records if r.levelno < logging.WARNING] == []


def test_fit_with_verbose_logs_the_engine_progress(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DDFM(**{**TINY, "verbose": 1}).fit(_panel())
    assert any(r.name.startswith("dfm_python.models.ddfm") and r.levelno == logging.INFO for r in caplog.records)


# -- forecasting --------------------------------------------------------------------


def test_predict_frame(fitted):
    fc = fitted.predict(4)
    assert fc.shape == (4, 3) and list(fc.columns) == ["s0", "s1", "s2"]
    assert all(dtype == np.float64 for dtype in fc.dtypes)
    expected_index = pd.date_range("2014-01-31", periods=4, freq="ME", name="date")
    assert fc.index.equals(expected_index)
    assert fc.attrs == {"model": "DDFM", "kind": "forecast", "origin": fitted.index_[-1]}
    assert np.isfinite(fc.to_numpy()).all()

    zf = fitted.predict_factors(4)
    assert zf.shape == (4, 2) and list(zf.columns) == ["factor_0", "factor_1"]
    assert zf.index.equals(expected_index)
    assert zf.attrs == {"model": "DDFM", "kind": "factors", "origin": fitted.index_[-1]}


def test_predict_is_the_engine_forecast_in_data_units(fitted):
    y_forecast, z_forecast = fitted.network_._forecast(5)
    H = fitted.network_.training_state.H
    model_space = z_forecast @ H.T
    np.testing.assert_array_equal(fitted.predict(5).to_numpy(), y_forecast)
    np.testing.assert_allclose(fitted.predict(5).to_numpy(), fitted.scaler_.inverse_transform(model_space))
    np.testing.assert_array_equal(fitted.predict_factors(5).to_numpy(), z_forecast)


def test_predict_does_not_change_the_model(fitted):
    factors = fitted.network_.factors.copy()
    first = fitted.predict(3)
    fitted.predict_factors(3)
    assert fitted.predict(3).equals(first)
    np.testing.assert_array_equal(fitted.network_.factors, factors)


def test_scaler_none_forecasts_in_model_space():
    model = DDFM(**TINY, scaler=None).fit(_panel(index="range"))
    assert model.scaler_ is None
    fc = model.predict(3)
    _, z_forecast = model.network_._forecast(3)
    np.testing.assert_array_equal(fc.to_numpy(), z_forecast @ model.network_.training_state.H.T)
    assert fc.index.equals(pd.RangeIndex(48, 51, name="time"))


@pytest.mark.parametrize("h", [2.0, "3", None, True, np.float64(2)])
def test_horizon_must_be_an_integer(fitted, h):
    with pytest.raises(TypeError, match="h must be"):
        fitted.predict(h)
    with pytest.raises(TypeError, match="h must be"):
        fitted.predict_factors(h)


def test_horizon_must_be_positive(fitted):
    with pytest.raises(ValueError):
        fitted.predict(0)
    assert fitted.predict(np.int64(2)).shape == (2, 3)


# -- update ---------------------------------------------------------------------


def _weights(est: DDFM) -> dict:
    return {k: v.detach().clone() for k, v in est.network_.state_dict().items()}


def _same_weights(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def _full(T: int = 60, K: int = 1) -> PanelData:
    """A panel of 60 rows on the training rows' dates followed by 12 new ones (other values than _panel)."""
    return _panel(T=T, K=K, seed=1)


def _new(T: int = 60, K: int = 1, start: int = 48) -> PanelData:
    """The rows of :func:`_full` after the training data (the 12 months after 2013-12-31 by default)."""
    full = _full(T=T, K=K)
    index = full.index[start:]
    return PanelData(y=full.y.loc[index], exog=None if full.exog is None else full.exog.loc[index])


def test_update_appends_only_the_new_rows_and_keeps_the_scalers(fitted):
    est = copy.deepcopy(fitted)
    weights, result, factors = _weights(est), est.result_, est.factors_.copy()
    engine = est.network_
    scaler = copy.deepcopy(est.scaler_)
    new = _new()
    assert est.update(new) is est
    # Only the 12 new rows are encoded and appended to each MC sample's factors.
    assert engine.factors.shape == (2, 48 + 12, 2)
    np.testing.assert_array_equal(engine.factors[:, :48], fitted.network_.factors)
    # The scalers fitted at fit time are not refitted.
    assert est.scaler_ is engine.scaler and est.result_.target_scaler is est.scaler_
    for attr in ("mean_", "scale_", "var_", "n_samples_seen_"):
        np.testing.assert_array_equal(getattr(est.scaler_, attr), getattr(scaler, attr))
    # The fit is unchanged; the forecasts now start after the new rows.
    assert _same_weights(weights, _weights(est))
    assert est.result_ is result and est.factors_.equals(factors) and est.index_.equals(_panel().index)
    assert est.conditioning_index_.equals(_full().index)
    assert est.conditioning_index_.freq == "ME"
    forecast = est.predict(3)
    assert forecast.index.equals(pd.date_range(new.index[-1], periods=4, freq="ME", name="date")[1:])
    assert forecast.attrs["origin"] == new.index[-1] and np.isfinite(forecast.to_numpy()).all()
    assert est.get_params() == fitted.get_params()


def test_update_scales_the_new_rows_with_the_training_statistics(fitted):
    """Targets with scaler_ and covariates with the covariate scaler fitted on the training exog; no refit."""
    est = copy.deepcopy(fitted)
    engine = est.network_
    train = _panel()
    covariate_scaler = engine.covariate_scaler
    assert type(covariate_scaler) is StandardScaler and covariate_scaler is not est.scaler_
    np.testing.assert_array_equal(covariate_scaler.mean_, StandardScaler().fit(train.exog.to_numpy()).mean_)
    np.testing.assert_array_equal(est.scaler_.mean_, StandardScaler().fit(train.y.to_numpy()).mean_)
    new = _new()
    est.update(new)
    appended = engine.data.iloc[48:]
    assert len(engine.data) == 60 and list(engine.data.columns) == ["x0", "s0", "s1", "s2"]
    np.testing.assert_array_equal(appended[["s0", "s1", "s2"]].to_numpy(), est.scaler_.transform(new.y.to_numpy()))
    np.testing.assert_array_equal(appended[["x0"]].to_numpy(), covariate_scaler.transform(new.exog.to_numpy()))
    # The encoder sees exactly those rows (no missing values in them). The engine feeds it a
    # C-contiguous array (to_tensor); a Fortran-ordered frame view can differ by an ulp on some CPUs.
    with torch.no_grad():
        rows = np.ascontiguousarray(appended.to_numpy())
        expected = engine.encoder(torch.as_tensor(rows, dtype=torch.float32)).numpy()
    np.testing.assert_array_equal(engine.factors[0, 48:], expected)
    np.testing.assert_array_equal(engine.factors[1, 48:], expected)


@pytest.mark.parametrize("data", ["full_panel", "training_panel", "partial_overlap", "last_row"])
def test_update_and_predict_data_with_overlapping_rows_raise(fitted, data):
    full = _full()
    panels = {
        "full_panel": full,
        "training_panel": _panel(),
        "partial_overlap": _new(start=40),
        "last_row": PanelData(y=full.y.iloc[47:48], exog=full.exog.iloc[47:48]),
    }
    est = copy.deepcopy(fitted)
    before = _state_hash(est)
    for call in (lambda: est.update(panels[data]), lambda: est.predict(3, data=panels[data]),
                 lambda: est.predict_factors(3, data=panels[data])):
        with pytest.raises(DataValidationError, match="overlaps the conditioning data"):
            call()
    assert _state_hash(est) == before
    # After an update, the rows it appended are conditioning data too.
    est.update(_new(T=54))
    with pytest.raises(DataValidationError, match="overlaps"):
        est.update(_new(T=60, start=50))


def test_update_rejects_another_index_type_and_a_gap_in_a_range_index():
    est = DDFM(**TINY).fit(_panel(index="range"))
    full = _panel(T=60, index="range", seed=1)
    with pytest.raises(DataValidationError, match="RangeIndex must continue"):
        est.update(PanelData(y=full.y.iloc[50:], exog=full.exog.iloc[50:]))
    with pytest.raises(DataValidationError, match="kind of index"):
        est.update(_new())
    est.update(PanelData(y=full.y.iloc[48:], exog=full.exog.iloc[48:]))
    assert est.conditioning_index_.equals(pd.RangeIndex(60)) and est.network_.factors.shape == (2, 60, 2)
    assert est.predict(2).index.equals(pd.RangeIndex(60, 62, name="time"))


@pytest.mark.parametrize("scaler", ["minmax", "robust", "quantile", "standard"])
def test_update_forecasts_are_in_the_data_range(scaler):
    """Positive-level series: the old update refitted minmax/robust/... scalers on mixed units (e.g. 11.5 here).

    ("maxabs" leaves level data uncentred in [0.7, 1], and this tiny network
    then fits a degenerate model whose forecasts are off even without update.)
    """
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-01-31", periods=96, freq="ME", name="date")
    common = np.cumsum(rng.standard_normal(96)) * 0.3
    y = pd.DataFrame({f"s{i}": 50 + 5 * i + common + rng.standard_normal(96) for i in range(3)}, index=idx)
    x = pd.DataFrame({"x0": 100 + 10 * rng.standard_normal(96)}, index=idx)
    params = {**TINY, "encoder_layers": (8,), "max_iter": 3, "scaler": scaler}
    est = DDFM(**params).fit(PanelData(y=y.iloc[:84], exog=x.iloc[:84]))
    new = PanelData(y=y.iloc[84:], exog=x.iloc[84:])
    lo, hi = y.to_numpy().min(), y.to_numpy().max()
    margin = 0.5 * (hi - lo)
    forecast = est.predict(3, data=new).to_numpy()
    assert np.isfinite(forecast).all()
    assert ((forecast > lo - margin) & (forecast < hi + margin)).all(), (scaler, forecast, lo, hi)
    # Each target's forecast is near its own level, not another unit's.
    assert (np.abs(forecast[0] - y.iloc[84:].mean().to_numpy()) < margin).all(), (scaler, forecast[0])
    assert copy.deepcopy(est).update(new).predict(3).equals(est.predict(3, data=new))


def test_missing_targets_and_long_covariate_gaps_in_new_rows_are_filled():
    """Missing targets in the new rows never reach the encoder as NaN, so no NaN factors and no zero-state forecast."""
    est = DDFM(**{**TINY, "interpolation_limit": 2}, scaler="standard").fit(_panel())
    full = _full()
    y, x = full.y.iloc[48:].copy(), full.exog.iloc[48:].copy()
    y.iloc[-1, :] = np.nan  # every target missing in the last new row (a ragged edge)
    y.iloc[3:6, 1] = np.nan  # an interior target gap
    x.iloc[2:11, 0] = np.nan  # a covariate gap longer than interpolation_limit
    updated = copy.deepcopy(est).update(PanelData(y=y, exog=x))
    factors = updated.network_.factors
    assert np.isfinite(factors).all()
    # The last new row is encoded from filled inputs: the last observed targets carried forward.
    filled = updated.network_.data.iloc[48:].copy()
    assert filled.iloc[-1].isna().sum() == 3 and filled.isna().to_numpy().sum() == 3 + 3 + 9
    carried = y.copy()
    carried.iloc[-1, :] = carried.iloc[-2, :]
    reference = copy.deepcopy(est).update(PanelData(y=carried, exog=x)).network_.factors
    np.testing.assert_array_equal(factors[:, -1], reference[:, -1])
    # The forecast starts from the last new row's factors, not from a zero state.
    z_last = factors.mean(axis=0)[-1]
    assert np.abs(z_last).sum() > 0
    np.testing.assert_allclose(
        updated.predict_factors(1).to_numpy()[0], z_last @ updated.network_.training_state.F.T, rtol=1e-5, atol=1e-7
    )
    forecast = updated.predict(3)
    assert np.isfinite(forecast.to_numpy()).all()
    assert not np.allclose(forecast.to_numpy(), est.scaler_.inverse_transform(np.zeros((3, 3))))
    assert est.predict(3, data=PanelData(y=y, exog=x)).equals(forecast)


def test_new_rows_are_interpolated_across_the_boundary_with_the_conditioning_rows(fitted):
    """A gap at the start of the new rows is interpolated from the last conditioning row, as fit would."""
    new = _new()
    y = new.y.copy()
    y.iloc[0, 0] = np.nan
    est = copy.deepcopy(fitted).update(PanelData(y=y, exog=new.exog))
    data = est.network_.data
    assert np.isnan(data.iloc[48]["s0"])  # the conditioning frame keeps the missing value
    expected = 0.5 * (data.iloc[47]["s0"] + data.iloc[49]["s0"])
    row = data.iloc[48].copy()
    row["s0"] = expected
    with torch.no_grad():
        z = est.network_.encoder(torch.as_tensor(row.to_numpy(dtype=np.float64)[None], dtype=torch.float32)).numpy()
    np.testing.assert_allclose(est.network_.factors[0, 48], z[0], rtol=1e-6, atol=1e-7)


def test_successive_updates_equal_one_update_with_all_rows(fitted):
    new = _new()
    first = PanelData(y=new.y.iloc[:6], exog=new.exog.iloc[:6])
    second = PanelData(y=new.y.iloc[6:], exog=new.exog.iloc[6:])
    one = copy.deepcopy(fitted).update(new)
    two = copy.deepcopy(fitted).update(first).update(second)
    np.testing.assert_array_equal(one.network_.factors, two.network_.factors)
    assert one.network_.data.equals(two.network_.data)
    assert one.conditioning_index_.equals(two.conditioning_index_)
    assert one.predict(3).equals(two.predict(3))


def test_update_without_a_scaler_and_without_exog():
    est = DDFM(**TINY, scaler=None).fit(_panel(K=0, index="range"))
    assert est.network_.covariate_scaler is None
    new = _panel(T=60, K=0, index="range", seed=1)
    new = PanelData(y=new.y.iloc[48:])
    est.update(new)
    assert est.scaler_ is None and est.network_.factors.shape == (2, 48 + 12, 2)
    np.testing.assert_array_equal(est.network_.data.iloc[48:].to_numpy(), new.y.to_numpy())
    assert est.predict(2).index.equals(pd.RangeIndex(60, 62, name="time"))


def test_update_refit_raises(fitted):
    est = copy.deepcopy(fitted)
    with pytest.raises(UnsupportedInputError, match="refit"):
        est.update(_new(), refit=True)
    assert est.conditioning_index_.equals(fitted.conditioning_index_)
    np.testing.assert_array_equal(est.network_.factors, fitted.network_.factors)


@pytest.mark.parametrize("change", ["series", "order", "exog", "no_exog"])
def test_update_and_predict_data_check_the_names(fitted, change):
    panel = _new()
    if change == "series":
        panel = PanelData(y=panel.y.rename(columns={"s0": "other"}), exog=panel.exog)
    elif change == "order":
        panel = PanelData(y=panel.y[["s1", "s0", "s2"]], exog=panel.exog)
    elif change == "exog":
        panel = _new(K=2)
    else:
        panel = PanelData(y=panel.y)
    est = copy.deepcopy(fitted)
    for call in (lambda: est.update(panel), lambda: est.predict(3, data=panel), lambda: est.predict_factors(3, data=panel)):
        with pytest.raises(DataValidationError):
            call()
    assert est.conditioning_index_.equals(fitted.conditioning_index_)


def test_update_rejects_excluded_inputs(fitted):
    rng = np.random.default_rng(0)
    with_series_exog = PanelData.from_arrays(rng.standard_normal((20, 3)), series_exog=rng.standard_normal((20, 3, 2)))
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        copy.deepcopy(fitted).update(with_series_exog)


def test_update_before_fit_raises():
    """Ported from the old engine test: update before fit raises (now NotFittedError)."""
    with pytest.raises(NotFittedError):
        DDFM(**TINY).update(_new())


# -- predict(data=...) ----------------------------------------------------------


def test_predict_data_is_update_on_a_copy(fitted):
    est = fitted
    new = _new()
    before = _state_hash(est)
    weights, factors, mean = _weights(est), est.network_.factors.copy(), est.scaler_.mean_.copy()
    state = torch.get_rng_state()
    forecast = est.predict(4, data=new)
    factor_forecast = est.predict_factors(4, data=new)
    assert torch.equal(state, torch.get_rng_state())
    assert est.predict(4, data=new).equals(forecast)
    assert _state_hash(est) == before
    assert _same_weights(weights, _weights(est)) and np.array_equal(est.network_.factors, factors)
    np.testing.assert_array_equal(est.scaler_.mean_, mean)
    assert est.conditioning_index_.equals(_panel().index) and len(est.network_.data) == 48

    reference = copy.deepcopy(est).update(new)
    assert forecast.to_numpy().tobytes() == reference.predict(4).to_numpy().tobytes()
    assert forecast.index.equals(reference.predict(4).index) and forecast.attrs == reference.predict(4).attrs
    assert forecast.index[0] > new.index[-1] and forecast.attrs["origin"] == new.index[-1]
    assert factor_forecast.equals(reference.predict_factors(4))
    assert not forecast.equals(est.predict(4))


# -- save / load ----------------------------------------------------------------


def _assert_same_model(loaded: DDFM, est: DDFM) -> None:
    assert type(loaded) is DDFM and loaded.get_params() == est.get_params()
    assert _same_weights(_weights(loaded), _weights(est))
    for name in ("n_features_in_", "n_iter_", "converged_", "n_factors_", "random_state_"):
        assert getattr(loaded, name) == getattr(est, name), name
    for name in ("feature_names_in_", "exog_names_in_"):
        assert list(getattr(loaded, name)) == list(getattr(est, name)), name
    assert loaded.index_.equals(est.index_) and getattr(loaded.index_, "freq", None) == getattr(est.index_, "freq", None)
    assert loaded.conditioning_index_.equals(est.conditioning_index_)
    assert loaded.factors_.equals(est.factors_)
    assert loaded.device_ == torch.device("cpu") and loaded.network_.device == torch.device("cpu")
    engine, original = loaded.network_, est.network_
    for name in ("factors", "eps", "last_neurons"):
        a, b = getattr(engine, name), getattr(original, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    for name in ("F", "H", "Q", "R", "mu_0", "Sigma_0"):
        a, b = getattr(engine.training_state, name), getattr(original.training_state, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    assert engine.data.equals(original.data) and list(engine.data.columns) == list(original.data.columns)
    assert isinstance(engine.data.index, pd.RangeIndex)
    assert engine.target_series == original.target_series and engine.covariates == original.covariates
    np.testing.assert_array_equal(engine.target_indices, original.target_indices, strict=True)
    assert engine._num_iter == original._num_iter
    assert engine._converged == getattr(original, "_converged", False)
    assert getattr(engine, "loss_now", None) == getattr(original, "loss_now", None)
    assert type(getattr(engine, "loss_now", None)) is type(getattr(original, "loss_now", None))
    for name in ("encoder_size", "decoder_size", "input_dim", "output_dim", "num_series", "initializer_seed",
                 "interpolation_limit", "learning_rate", "tolerance", "disp"):
        assert getattr(engine, name) == getattr(original, name), name
    assert engine.scaler is loaded.scaler_
    result, fitted_result = loaded.result_, est.result_
    assert result.target_scaler is loaded.scaler_
    for f in dataclasses.fields(DDFMResult):
        if f.name == "target_scaler":
            continue
        a, b = getattr(result, f.name), getattr(fitted_result, f.name)
        if isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f.name
        else:
            assert a == b and type(a) is type(b), f.name


def test_save_load_round_trip(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"config.json", "data_spec.json", "state.npz", "state.json", "result.npz", "result.json",
                     "weights.pt", "scaler.json"}
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["num_iter"] == 2 and state["loss_now"] is None and state["converged"] is False
    assert state["encoder_size"] == [4, 2] and state["decoder_size"] is None
    assert state["data_columns"] == ["x0", "s0", "s1", "s2"]
    assert state["target_series"] == ["s0", "s1", "s2"] and state["covariates"] == ["x0"]
    assert (state["input_dim"], state["output_dim"], state["num_series"]) == (4, 3, 4)
    assert state["conditioning_index"]["kind"] == "datetime"
    with np.load(tmp_path / "run" / "state.npz") as npz:
        keys = set(npz.files)
        assert np.isnan(npz["data/frame"]).sum() == 1  # the training frame keeps its missing value
    assert {"factors", "eps", "last_neurons", "data/frame"} | {f"ssm/{n}" for n in ("F", "H", "Q", "R", "mu_0", "Sigma_0")} <= keys
    assert "target_scaler" in json.loads((tmp_path / "run" / "result.json").read_text())["excluded"]
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["model_type"] == "DDFM" and config["params"]["random_state"] == 0
    assert state["covariate_scaler"]["class"] == "sklearn.preprocessing.StandardScaler"
    assert {"covariate_scaler/mean_", "covariate_scaler/scale_"} <= keys
    new = _new()
    for loader in (DDFM.load, dfm_python.load):
        loaded = loader(tmp_path / "run", device="cpu")
        _assert_same_model(loaded, fitted)
        assert loaded.predict(3).equals(fitted.predict(3))
        assert loaded.predict_factors(3).equals(fitted.predict_factors(3))
        assert loaded.predict(3, data=new).equals(fitted.predict(3, data=new))
        assert isinstance(loaded.scaler_, StandardScaler)


def test_save_load_after_update(fitted, tmp_path):
    est = copy.deepcopy(fitted).update(_new())
    est.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run")
    _assert_same_model(loaded, est)
    assert loaded.network_.factors.shape == (2, 60, 2) and len(loaded.network_.data) == 60
    assert loaded.predict(3).equals(est.predict(3))
    np.testing.assert_array_equal(loaded.scaler_.mean_, est.scaler_.mean_)
    # Updating the loaded model further works like updating the original.
    more = _new(T=72, start=60)
    y = more.y.copy()
    y.iloc[0, 1] = np.nan  # interpolated against the saved conditioning frame
    more = PanelData(y=y, exog=more.exog)
    assert loaded.update(more).predict(3).equals(copy.deepcopy(est).update(more).predict(3))


class _HalfScaler(StandardScaler):
    """A transformer outside the five known scalers (saved with pickle)."""


@pytest.mark.parametrize("scaler", ["minmax", "robust", "quantile", _HalfScaler()])
def test_save_load_keeps_the_covariate_scaler(scaler, tmp_path):
    est = DDFM(**{**TINY, "scaler": scaler}).fit(_panel(K=2))
    assert type(est.network_.covariate_scaler) is type(est.scaler_)
    est.save(tmp_path / "run")
    pickled = isinstance(scaler, _HalfScaler)
    if pickled:
        with pytest.raises(ConfigurationError, match="allow_pickle"):
            DDFM.load(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run", allow_pickle=pickled)
    covariate_scaler = loaded.network_.covariate_scaler
    assert type(covariate_scaler) is type(est.network_.covariate_scaler)
    new = _new(K=2)
    expected = est.network_.covariate_scaler.transform(new.exog.to_numpy())
    np.testing.assert_array_equal(covariate_scaler.transform(new.exog.to_numpy()), expected)
    assert loaded.predict(3, data=new).equals(est.predict(3, data=new))


def test_a_run_saved_without_the_covariate_scaler_cannot_update(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    path = tmp_path / "run" / "state.json"
    state = json.loads(path.read_text())
    del state["covariate_scaler"]
    path.write_text(json.dumps(state))
    loaded = DDFM.load(tmp_path / "run")
    assert loaded.network_.covariate_scaler is None
    assert loaded.predict(3).equals(fitted.predict(3))
    with pytest.raises(ModelNotInitializedError, match="covariate scaler"):
        loaded.update(_new())


def test_save_load_mlp_without_scaler_or_exog(tmp_path):
    est = DDFM(**{**TINY, "decoder_type": "mlp", "encoder_layers": (6, 4), "max_iter": 3, "scaler": None}).fit(
        _panel(K=0, index="range")
    )
    assert isinstance(est.network_.loss_now, float)
    est.save(tmp_path / "run")
    assert json.loads((tmp_path / "run" / "state.json").read_text())["decoder_size"] == [4, 6]
    loaded = DDFM.load(tmp_path / "run")
    _assert_same_model(loaded, est)
    assert loaded.scaler_ is None and loaded.network_.last_neurons.shape == (2, 48, 6)
    assert loaded.predict(3).equals(est.predict(3))
    new = PanelData(y=_panel(T=60, K=0, index="range", seed=1).y.iloc[48:])
    assert loaded.predict(3, data=new).equals(est.predict(3, data=new))


def test_load_does_not_touch_the_global_rngs(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    DDFM.load(tmp_path / "run")
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))


def test_from_saved_sets_every_attribute_init_sets(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run").network_
    built = DDFMEngine(dataset=fitted._adapter(_panel()), device="cpu", **fitted._engine_params())
    missing = sorted(set(vars(built)) - set(vars(loaded)))
    assert not missing
    for name in ("encoder_size", "decoder_size", "decoder_type", "activation", "n_mc_samples", "window_size",
                 "max_epoch_pre_train", "max_iter", "tolerance", "disp", "min_target_interporate_ratio",
                 "interpolation_method", "interpolation_limit", "interpolation_limit_direction", "num_series",
                 "input_dim", "output_dim", "initializer_seed", "learning_rate", "optimizer_type", "device",
                 "lags_input", "target_series", "covariates", "_config"):
        assert getattr(loaded, name) == getattr(built, name), name
    assert loaded._dataset is None and loaded.optimizer is None and loaded.scheduler is None
    assert isinstance(loaded.training_state, DDFMModelState) and loaded.training_state.F is not None
    assert dict(loaded.named_modules()).keys() == dict(fitted.network_.named_modules()).keys()
    assert list(loaded.state_dict()) == list(fitted.network_.state_dict())
    assert not loaded.autoencoder.training


def test_from_saved_rejects_unknown_engine_parameters(fitted):
    dims = {"input_dim": 4, "output_dim": 3, "num_series": 4, "target_series": ["s0", "s1", "s2"], "covariates": ["x0"]}
    for bad in ({"dataset": None}, {"device": "cpu"}, {"pretrain_epochs": 3}, {"config": None}):
        with pytest.raises(TypeError, match="unexpected"):
            DDFMEngine._from_saved({**fitted._engine_params(), **bad}, dims, "cpu")


def test_load_rejects_a_state_that_does_not_match_the_config(fitted, tmp_path):
    fitted.save(tmp_path / "run")
    path = tmp_path / "run" / "state.json"
    state = json.loads(path.read_text())
    state["encoder_size"] = [8, 2]
    path.write_text(json.dumps(state))
    with pytest.raises(ConfigurationError, match="encoder"):
        DDFM.load(tmp_path / "run")


def test_single_file_checkpoint_is_rejected(fitted, tmp_path):
    """Replaces the tests of the removed single-file helpers (infer_ddfm_input_dim and
    infer_ddfm_params_from_state_dict): an old torch.save checkpoint is refused, not read."""
    path = tmp_path / "old_ddfm_model.pt"
    torch.save({"state_dict": fitted.network_.state_dict(), "encoder_size": (4, 2), "decoder_type": "linear"}, path)
    with pytest.raises(ConfigurationError, match="single-file"):
        DDFM.load(path)
    with pytest.raises(ConfigurationError, match="single-file"):
        dfm_python.load(path)


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(NotFittedError):
        DDFM(**TINY).save(tmp_path / "run")
    assert not (tmp_path / "run").exists()


# -- config mirror and registry -------------------------------------------------


def test_config_mirrors_the_constructor():
    cfg = DDFM().get_config()
    assert isinstance(cfg, DDFMConfig) and cfg.to_dict() == DDFM().get_params()
    assert DDFM._config_class is DDFMConfig and dfm_python.DDFMConfig is DDFMConfig
    assert DDFMConfig.field_names() == tuple(inspect.signature(DDFM.__init__).parameters)[1:]
    est = DDFM.from_config({"n_factors": 3, "encoder_layers": [16, 8], "scaler": "robust", "tol": 1e-3})
    assert est.n_factors == 3 and est.encoder_layers == (16, 8) and est.scaler == "robust" and est.tol == 1e-3


@pytest.mark.parametrize(
    "old",
    ["encoder_size", "num_factors", "tolerance", "seed", "disp", "max_epoch", "use_batch_norm", "use_idiosyncratic",
     "min_obs_idio", "lags_input", "max_epoch_pre_train", "frequency", "clock"],
)
def test_config_rejects_old_names(old):
    with pytest.raises(ConfigurationError, match=old):
        DDFMConfig.from_dict({old: 1})


@pytest.mark.parametrize("params", [{"encoder_layers": [64, 32]}, {"model_type": "ddfm"}, {"learning_rate": 0.01}])
def test_dfm_config_from_dict_points_ddfm_configs_to_the_mirror(params):
    with pytest.raises(ConfigurationError, match=r"DDFMConfig\.from_dict.*DDFM\.from_config"):
        _DFMEngineConfig.from_dict({"clock": "m", **params})


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_interpolation_values_reach_the_engine(tmp_path, monkeypatch):
    """Ported from the engine test of config interpolation precedence: what the YAML sets is what runs."""
    seen = {}
    original = DDFMEngine.fit

    def fit(self, *args, **kwargs):
        seen.update(method=self.interpolation_method, limit=self.interpolation_limit,
                    direction=self.interpolation_limit_direction)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DDFMEngine, "fit", fit)
    path = tmp_path / "ddfm.yaml"
    path.write_text(
        "model:\n  n_factors: 2\n  encoder_layers: [4]\n  n_mc_samples: 2\n  window_size: 16\n  max_iter: 2\n"
        "  interpolation_method: nearest\n  interpolation_limit: 7\n  interpolation_limit_direction: forward\n"
        "  random_state: 0\n  device: cpu\n",
        encoding="utf-8",
    )
    model = DDFM.from_config(DDFMConfig.from_yaml(path))
    assert model.interpolation_limit == 7
    engine = model.fit(_panel()).network_
    assert seen == {"method": "nearest", "limit": 7, "direction": "forward"}
    assert engine._config is None and engine.interpolation_limit == 7


def test_ddfm_is_registered():
    assert ("DDFM", DDFM) in all_estimators()


# -- workflow (ported from the old test_ddfm.py) -------------------------------------


def _five_series(T: int = 50, *, start: str = "2010-01-31", seed: int = 42) -> PanelData:
    """Five target series and no covariates, as the old engine tests used."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=T, freq="ME", name="date")
    return PanelData(y=pd.DataFrame(rng.randn(T, 5), index=idx, columns=[f"series_{i}" for i in range(5)]))


def test_fit_creates_the_network_components():
    model = DDFM(max_iter=1, device="cpu")
    assert not hasattr(model, "network_")
    model.fit(_five_series())
    engine = model.network_
    assert engine.autoencoder is not None
    assert engine.encoder is not None
    assert engine.decoder is not None
    assert engine.optimizer is not None
    assert engine.scheduler is not None
    assert engine.factors is not None
    assert engine._training_time > 0


def test_fit_builds_the_state_space():
    model = DDFM(max_iter=1, device="cpu").fit(_five_series())
    state = model.network_.training_state
    for name in ("F", "H", "Q", "R", "mu_0", "Sigma_0"):
        assert getattr(state, name) is not None, name
    r = model.n_factors_
    assert model.result_.A.shape == (r, r) == state.F.shape
    assert model.result_.C.shape == (5, r) == state.H.shape


def test_result_and_predict_are_available_right_after_fit():
    """The old engine needed build_state_space() before predict and get_result(); fit now builds it."""
    model = DDFM(max_iter=1, scaler="standard", device="cpu").fit(_five_series())
    for name in ("Z", "C", "A", "x_sm"):
        assert getattr(model.result_, name) is not None, name
    forecast = model.predict(5)
    assert forecast.shape == (5, 5) and np.isfinite(forecast.to_numpy()).all()


def test_full_workflow():
    data = _five_series()
    model = DDFM(max_iter=2, scaler="standard", device="cpu")  # encoder (16,) and 4 factors: the defaults
    assert model.fit(data) is model
    assert model.network_.factors.shape[-1] == 4
    assert model.factors_.shape == (50, 4) and model.result_.Z.shape[-1] == 4
    assert model.predict(5).shape == (5, 5)
    assert model.predict_factors(5).shape == (5, 4)
    assert model.result_.converged is not None and model.result_.num_iter >= 0
    assert model.n_iter_ == 2


def test_update_with_new_rows():
    model = DDFM(max_iter=1, device="cpu").fit(_five_series())
    old = model.network_.factors.copy()
    new = _five_series(T=10, start="2014-03-31", seed=0)  # the ten months after the training data
    assert model.update(new) is model
    factors = model.network_.factors
    assert factors.shape[:-2] == old.shape[:-2]
    assert factors.shape[-2] == old.shape[-2] + 10 and factors.shape[-1] == old.shape[-1]
    np.testing.assert_array_equal(factors[..., : old.shape[-2], :], old)
    forecast = model.predict(3)
    assert forecast.index[0] > new.index[-1]
    assert forecast.index.equals(pd.date_range(new.index[-1], periods=4, freq="ME", name="date")[1:])


# -- shipped YAMLs and tutorials -------------------------------------------------

_MACRO_EXOG = ["KORFBEV", "KORWHLS", "KORMCMP", "KORELEC", "KORCNST", "KORMACH", "KOROCOINF", "KOEMPTOTO",
               "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOWRCCNSE", "KOIMPCONA"]
_EXCHANGE = ["AUD_USD", "GBP_USD", "CAD_USD", "CHF_USD", "CNY_USD", "JPY_USD", "SGD_USD", "NZD_USD"]
_FINANCE_TARGET = "market_forward_excess_returns"


def _finance_exog() -> list[str]:
    """Every column of finance.csv except the target, forward_returns and risk_free_rate, in CSV order."""
    header = pd.read_csv(_REPO / "data" / "finance.csv", nrows=0).columns
    return [c for c in header if c not in (_FINANCE_TARGET, "forward_returns", "risk_free_rate")]


_SHIPPED = {
    "macro": {
        "model": {"n_factors": 2, "encoder_layers": (64, 32), "decoder_type": "linear", "activation": "relu",
                  "learning_rate": 0.001, "optimizer": "Adam", "n_mc_samples": 1, "window_size": 100,
                  "max_iter": 200, "tol": 5e-4, "interpolation_method": "linear", "interpolation_limit": 10,
                  "interpolation_limit_direction": "both", "scaler": "standard", "random_state": 3},
        "data": {"time_col": "date", "target_cols": ["KOEQUIPTE"], "exog_cols": _MACRO_EXOG},
        "csv": "macro.csv",
    },
    "finance": {
        "model": {"n_factors": 5, "encoder_layers": (64, 32), "decoder_type": "mlp", "activation": "relu",
                  "learning_rate": 0.001, "optimizer": "Adam", "n_mc_samples": 10, "window_size": 200,
                  "max_iter": 10, "tol": 5e-4, "interpolation_method": "linear", "interpolation_limit": 10,
                  "interpolation_limit_direction": "both", "scaler": "standard", "random_state": 3},
        "data": {"time_col": "date", "target_cols": [_FINANCE_TARGET], "exog_cols": _finance_exog()},
        "csv": "finance.csv",
    },
    "exchange": {
        "model": {"n_factors": 4, "encoder_layers": (16,), "decoder_type": "linear", "activation": "relu",
                  "learning_rate": 0.005, "optimizer": "Adam", "n_mc_samples": 10, "window_size": 200,
                  "max_iter": 10, "tol": 5e-4, "interpolation_method": "linear", "interpolation_limit": 10,
                  "interpolation_limit_direction": "both", "scaler": "standard", "random_state": 3},
        "data": {"time_col": "date", "target_cols": _EXCHANGE, "exog_cols": []},
        "csv": "exchange_rate.csv",
    },
}


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_shipped_yaml_builds_the_estimator(name):
    path = _REPO / "config" / f"ddfm_{name}.yaml"
    expected = {**DDFM().get_params(), **_SHIPPED[name]["model"]}
    cfg = DDFMConfig.from_yaml(path)
    assert cfg.to_dict() == expected
    assert DDFM.from_config(cfg).get_params() == expected
    sections = load_yaml(path)
    assert sections["data"] == _SHIPPED[name]["data"]
    # Every column the data: section names is in the shipped CSV (finance.csv gets its 'date' in the tutorial).
    header = set(pd.read_csv(_REPO / "data" / _SHIPPED[name]["csv"], nrows=0).columns) | {"date"}
    data = sections["data"]
    assert set(data["target_cols"]) | set(data["exog_cols"]) <= header


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", ["macro", "finance"])
def test_shipped_yaml_covariates_are_the_ones_the_tutorials_ran(name):
    """The exog columns, in order, are the engine covariates the golden fixtures recorded from the old tutorials."""
    manifest = json.loads((_REPO / "src" / "test" / "parity" / "fixtures" / f"ddfm_{name}.json").read_text())
    data = load_yaml(_REPO / "config" / f"ddfm_{name}.yaml")["data"]
    assert data["exog_cols"] == manifest["data"]["covariates"]


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_tutorial_runs_on_truncated_data(name, tmp_path):
    """The tutorial runs end to end: YAML, PanelData.from_wide, fit, predict, predict_factors, save and load."""
    path = _REPO / "tutorial" / f"tutorial_{name}_ddfm.py"
    spec = importlib.util.spec_from_file_location(f"_tutorial_{name}_ddfm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.main(["--rows", "120", "--max-iter", "1", "--device", "cpu", "--run-dir", str(tmp_path / "run")])
    text = out.getvalue()
    assert "6-step forecast (data units)" in text and "reloaded forecast is identical" in text
    loaded = DDFM.load(tmp_path / "run")
    expected = {**_SHIPPED[name]["model"], "max_iter": 1, "device": "cpu"}
    assert {k: loaded.get_params()[k] for k in expected} == expected
    assert loaded.n_iter_ == 1 and loaded.index_.size == 120
    assert list(loaded.feature_names_in_) == _SHIPPED[name]["data"]["target_cols"]
    assert list(loaded.exog_names_in_) == _SHIPPED[name]["data"]["exog_cols"]
    assert np.isfinite(loaded.predict(6).to_numpy()).all()

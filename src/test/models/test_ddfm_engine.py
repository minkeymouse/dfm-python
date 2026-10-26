"""Tests for the private DDFM engine (models.ddfm._engine.DDFMEngine).

The engine is the pre-redesign DDFM class with three edits that change no
math: a ``device`` keyword that overrides the automatic CUDA/CPU choice, the
pure ``_forecast`` extraction from ``predict``, and ``target_series`` /
``covariates`` copied from the dataset so that ``_update`` needs no dataset.
The removed pre-redesign entry points (``config``, ``save``, ``load``, the
public ``update``, ``predict``'s ``data``/``update``/``return_*`` flags) stay
removed, and the engine no longer inherits the old abstract base.

The engine internals that the old ``test_ddfm.py`` covered (encoder sizes,
the pre-training inputs, the decoder helpers, the variance diagnostics and the
dimension constants) are tested here against the engine; its workflow tests
moved to ``test_ddfm_estimator.py`` and the generic ``infer_input_dim_*``
helpers to ``src/test/utils/test_checkpoint.py``.
"""

import copy
import inspect
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from dfm_python.config.constants import (
    DEFAULT_DISP,
    DEFAULT_ENCODER_LAYERS,
    DEFAULT_MIN_TARGET_INTERPOLATE_RATIO,
    DEFAULT_PRETRAIN_EPOCHS,
    MIN_DDFM_TIME_STEPS,
    MIN_VARIABLES,
)
from dfm_python import DDFM, PanelData
from dfm_python.config.schema.params import DDFMModelState
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.models.ddfm._engine import DDFMEngine
from dfm_python.numeric.statistic import diagnose_variance_collapse
from dfm_python.utils.errors import (
    ConfigurationError,
    DataValidationError,
    ModelNotInitializedError,
    ModelNotTrainedError,
)


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _frame(T: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((T, 4)), columns=["cov", "a", "b", "c"])


def _dataset(frame: pd.DataFrame, scaler="standard") -> DDFMDataset:
    return DDFMDataset(
        data=frame,
        time_idx="index",
        covariates=["cov"],
        scaler=StandardScaler() if scaler == "standard" else scaler,
    )


def _engine(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    params = dict(encoder_size=(4, 2), n_mc_samples=2, max_iter=2, device="cpu")
    params.update(kwargs)
    return DDFMEngine(dataset, **params)


def _fitted(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    engine = _engine(dataset, **kwargs)
    engine.fit()
    engine.build_state_space()
    return engine


def test_target_series_and_covariates_are_copied_from_the_dataset():
    dataset = _dataset(_frame())
    engine = _engine(dataset)
    assert engine.target_series == ["a", "b", "c"]
    assert engine.covariates == ["cov"]
    assert engine.target_series is not dataset.target_series
    assert engine.covariates is not dataset.covariates


def test_device_keyword_overrides_the_automatic_choice():
    dataset = _dataset(_frame())
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        engine = _engine(dataset, device="cpu")
    assert engine.device == torch.device("cpu")
    # Fit outside the patch: torch's optimizer step itself queries CUDA when
    # it believes CUDA is available, which fails on a machine without it.
    engine.fit()
    params = list(engine.parameters())
    assert params
    assert all(p.device.type == "cpu" for p in params)


def test_device_none_keeps_the_automatic_choice():
    dataset = _dataset(_frame())
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert _engine(dataset, device=None).device == torch.device("cpu")
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        assert _engine(dataset, device=None).device == torch.device("cuda")


def test_forecast_equals_predict_bitwise():
    engine = _fitted(_dataset(_frame()))
    y_forecast, Z_forecast = engine._forecast(6)
    y_pred, Z_pred = engine.predict(horizon=6)
    assert y_forecast.shape == (6, 3)
    assert Z_forecast.shape == (6, 2)
    for got, want in ((y_forecast, y_pred), (Z_forecast, Z_pred)):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()


def test_forecast_without_scaler_is_the_identity_and_predict_still_raises():
    engine = _fitted(_dataset(_frame(), scaler=None))
    assert engine.scaler is None
    y_forecast, Z_forecast = engine._forecast(6)
    expected = np.nan_to_num(engine.decode(Z_forecast))  # the trained decoder, bias included
    assert y_forecast.tobytes() == expected.tobytes()
    with pytest.raises(ConfigurationError):
        engine.predict(horizon=6)


class TestEngineSurface:
    """The pre-redesign entry points were removed, not deprecated; they must not come back."""

    @pytest.mark.parametrize("name", ["save", "load", "update", "_create_minimal_dataset_from_metadata",
                                      "_get_interpolation_params", "_restore_checkpoint_metadata"])
    def test_old_entry_point_is_removed(self, name):
        assert not hasattr(DDFMEngine, name)
        assert not hasattr(_engine(_dataset(_frame())), name)

    def test_load_state_dict_is_the_module_one(self):
        assert DDFMEngine.load_state_dict is torch.nn.Module.load_state_dict

    def test_engine_does_not_inherit_the_old_base(self):
        import dfm_python.models.base

        assert not hasattr(dfm_python.models.base, "BaseFactorModel")
        assert DDFMEngine.__bases__ == (torch.nn.Module,)

    def test_init_sets_the_old_base_attributes(self):
        dataset = _dataset(_frame())
        engine = _engine(dataset)
        assert engine._config is None and engine._result is None and engine.data_processed is None
        assert engine._dataset is dataset and isinstance(engine.training_state, DDFMModelState)

    def test_config_parameter_is_removed(self):
        assert "config" not in inspect.signature(DDFMEngine.__init__).parameters
        with pytest.raises(TypeError, match="config"):
            DDFMEngine(_dataset(_frame()), config=None)

    def test_dead_parameters_keep_their_recorded_defaults(self):
        # The golden manifests record them; the engine never reads them.
        engine = _engine(_dataset(_frame()))
        assert engine.max_epoch_pre_train == DEFAULT_PRETRAIN_EPOCHS
        assert engine.min_target_interporate_ratio == DEFAULT_MIN_TARGET_INTERPOLATE_RATIO
        assert engine.disp == DEFAULT_DISP and engine.lags_input == 0

    def test_interpolation_arguments_are_used_directly(self):
        engine = _engine(_dataset(_frame()), interpolation_method="nearest", interpolation_limit=3,
                         interpolation_limit_direction="forward")
        assert (engine.interpolation_method, engine.interpolation_limit,
                engine.interpolation_limit_direction) == ("nearest", 3, "forward")

    def test_predict_takes_only_the_horizon(self):
        assert list(inspect.signature(DDFMEngine.predict).parameters) == ["self", "horizon"]
        engine = _fitted(_dataset(_frame()))
        with pytest.raises(TypeError):
            engine.predict(horizon=6, data=_dataset(_frame()))
        y_pred, z_pred = engine.predict(horizon=6)
        assert y_pred.shape == (6, 3) and z_pred.shape == (6, 2)

    def test_predict_before_the_state_space_raises(self):
        engine = _engine(_dataset(_frame()))
        engine.fit()
        with pytest.raises(ModelNotInitializedError, match="state-space model has not been built"):
            engine.predict(horizon=3)


def test_update_on_a_loaded_engine_needs_no_dataset(tmp_path):
    """A loaded engine has no _dataset; _update runs on it and matches the fitted engine's update."""
    frame = _frame(T=72)
    fit_panel = PanelData(y=frame[["a", "b", "c"]].iloc[:60], exog=frame[["cov"]].iloc[:60])
    est = DDFM(n_factors=2, encoder_layers=(4,), n_mc_samples=2, max_iter=2, scaler="standard",
               random_state=0, device="cpu").fit(fit_panel)
    est.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run")
    engine = loaded.network_
    assert engine._dataset is None

    new_rows = frame.iloc[60:].reset_index(drop=True)
    reference = copy.deepcopy(est.network_)
    reference._update(new_rows)
    engine._update(new_rows)

    assert engine._dataset is None
    assert engine.factors.shape[-2] == 60 + 12
    assert engine.factors.tobytes() == reference.factors.tobytes()
    assert engine.data.equals(reference.data)
    assert engine.scaler.mean_.tobytes() == est.scaler_.mean_.tobytes()


def test_update_needs_no_dataset():
    frame = _frame(T=72)
    engine = _fitted(_dataset(frame.iloc[:60]))
    detached = copy.deepcopy(engine)
    detached._dataset = None
    mean = engine.scaler.mean_.copy()

    new_rows = frame.iloc[60:].reset_index(drop=True)
    engine._update(new_rows)
    detached._update(new_rows)

    assert engine.factors.shape[-2] == 60 + 12
    assert detached.factors.tobytes() == engine.factors.tobytes()
    assert engine.scaler.mean_.tobytes() == mean.tobytes()


def test_update_takes_the_training_columns_in_order():
    frame = _frame(T=72)
    engine = _fitted(_dataset(frame.iloc[:60]))
    new_rows = frame.iloc[60:].reset_index(drop=True)
    for bad in (new_rows[["a", "cov", "b", "c"]], new_rows[["cov", "a", "b"]], new_rows.iloc[:0]):
        with pytest.raises(DataValidationError):
            engine._update(bad)
    assert engine.factors.shape[-2] == 60 and len(engine.data) == 60


def test_dataset_keeps_the_fitted_covariate_scaler():
    frame = _frame()
    dataset = _dataset(frame)
    assert type(dataset.covariate_scaler) is StandardScaler and dataset.covariate_scaler is not dataset.scaler
    np.testing.assert_array_equal(dataset.covariate_scaler.mean_, frame[["cov"]].mean().to_numpy())
    np.testing.assert_array_equal(dataset.X, dataset.covariate_scaler.transform(frame[["cov"]].to_numpy()))
    assert _engine(dataset).covariate_scaler is dataset.covariate_scaler
    assert _dataset(frame, scaler=None).covariate_scaler is None
    assert DDFMDataset(data=frame[["a", "b"]], time_idx="index", scaler=StandardScaler()).covariate_scaler is None


# -- engine internals (ported from the old test_ddfm.py) ----------------------------


def _all_targets(num_series: int = 5, time_steps: int = 10, scaler=None, seed: int = 42) -> DDFMDataset:
    """A dataset whose columns are all targets (no covariates), as the old tests built it."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame(rng.randn(time_steps, num_series), columns=[f"series_{i}" for i in range(num_series)])
    return DDFMDataset(data=data, time_idx="index", scaler=scaler)


def _default_engine(dataset: DDFMDataset, **kwargs) -> DDFMEngine:
    return DDFMEngine(dataset=dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS), device="cpu", **kwargs)


@pytest.mark.parametrize("encoder_size", [(16, 4), (64, 32, 4), (4,)])
def test_initialization_with_encoder_sizes(encoder_size):
    engine = DDFMEngine(dataset=_all_targets(), encoder_size=encoder_size, device="cpu")
    assert engine.encoder_size == encoder_size
    assert engine.num_series == 5
    assert engine.encoder_size[-1] == encoder_size[-1]  # the last width is the number of factors


def test_window_size_and_n_mc_samples_are_stored():
    engine = _default_engine(_all_targets(), n_mc_samples=10, window_size=100)
    assert engine.window_size == 100
    assert engine.n_mc_samples == 10


def test_build_inputs_for_pretrain_is_the_data_optionally_interpolated():
    """_build_inputs_for_pretrain returns dataset.data, interpolated on request, without feature engineering."""
    data = pd.DataFrame(np.random.RandomState(0).randn(10, 5), columns=[f"series_{i}" for i in range(5)])
    data.iloc[1, 0] = np.nan
    data.iloc[3, 2] = np.nan
    dataset = DDFMDataset(data=data, time_idx="index", scaler=None)
    engine = _default_engine(dataset, interpolation_method="linear", interpolation_limit=10,
                             interpolation_limit_direction="both", max_iter=1)

    raw = engine._build_inputs_for_pretrain(interpolate=False)
    assert raw.shape == dataset.data.shape
    assert list(raw.columns) == list(dataset.data.columns)
    assert raw.isna().sum().sum() > 0

    filled = engine._build_inputs_for_pretrain(interpolate=True)
    assert filled.shape == dataset.data.shape
    assert list(filled.columns) == list(dataset.data.columns)
    assert filled.isna().sum().sum() == 0


def test_get_result_and_predict_before_fit_raise_not_trained():
    engine = _default_engine(_all_targets())
    with pytest.raises(ModelNotTrainedError):
        engine.get_result()
    assert getattr(engine, "_result", None) is None
    with pytest.raises(ModelNotTrainedError, match="model has not been trained"):
        engine.predict(horizon=5)


def test_dimension_constants():
    assert MIN_VARIABLES == 1
    assert MIN_DDFM_TIME_STEPS == 2


def test_decoder_helpers():
    engine = _default_engine(_all_targets(time_steps=50), max_iter=1)
    engine.fit()
    linear = engine.decoder.get_last_linear_layer()
    assert isinstance(linear, torch.nn.Linear)
    assert linear in [m for m in engine.decoder.modules() if isinstance(m, torch.nn.Linear)]
    assert engine.decoder_type == "linear"
    assert engine.decoder.get_intermediate() is None  # the linear decoder has no hidden layer


def test_variance_diagnostics_attributes_after_fit():
    engine = _default_engine(_all_targets(time_steps=50), max_iter=1)
    engine.fit()
    assert hasattr(engine, "prediction_std")
    assert hasattr(engine, "factor_std")


def test_variance_diagnostics_report_standardization():
    dataset_std = _all_targets(time_steps=50, scaler=StandardScaler())
    dataset_std.scaler.fit(dataset_std.data.values)
    dataset_std.data = pd.DataFrame(dataset_std.scaler.transform(dataset_std.data.values),
                                    columns=dataset_std.data.columns, index=dataset_std.data.index)
    engine_std = _default_engine(dataset_std, max_iter=1)
    engine_std.fit()

    dataset_robust = _all_targets(time_steps=50, scaler=RobustScaler())
    engine_robust = _default_engine(dataset_robust, max_iter=1)
    engine_robust.fit()

    rng = np.random.RandomState(42)
    prediction_std = rng.rand(50, 5) * 0.1
    prediction_mean = rng.rand(50, 5)
    factors_mean = rng.rand(50, 4)
    for engine, dataset in ((engine_std, dataset_std), (engine_robust, dataset_robust)):
        diagnostics = diagnose_variance_collapse(
            prediction_std=prediction_std, prediction_mean=prediction_mean, factors_mean=factors_mean,
            y_actual=engine.y_actual, target_scaler=dataset.scaler,
            encoder=engine.encoder, decoder=engine.decoder,
        )
        assert "is_standardized" in diagnostics


def test_variance_diagnostics_validate_shapes():
    dataset = _all_targets(time_steps=50)
    engine = _default_engine(dataset, max_iter=1)
    engine.fit()

    rng = np.random.RandomState(42)
    prediction_mean = rng.rand(50, 5)
    factors_mean = rng.rand(50, 4)

    def diagnose(prediction_std):
        return diagnose_variance_collapse(
            prediction_std=prediction_std, prediction_mean=prediction_mean, factors_mean=factors_mean,
            y_actual=engine.y_actual, target_scaler=dataset.scaler,
            encoder=engine.encoder, decoder=engine.decoder,
        )

    invalid = diagnose([[0.1] * 5] * 50)
    assert any("Invalid prediction_std type" in w for w in invalid["warnings"])
    one_d = diagnose(rng.rand(50))
    assert any("1D prediction_std array" in w for w in one_d["warnings"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        empty = diagnose(np.array([]).reshape(0, 5))
    assert any("Invalid prediction_std shape" in w for w in empty["warnings"])
    valid = diagnose(rng.rand(50, 5) * 0.1)
    assert "prediction_std_mean" in valid
    assert "variance_collapse_detected" in valid

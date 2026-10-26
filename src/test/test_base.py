"""``dfm_python.base.BaseFactorModel``: tags, config mirror, fitted-state guard, tag-gated defaults and fit."""

import copy
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.utils.validation import check_is_fitted

import dfm_python
from dfm_python.base import DEFAULT_TAGS, BaseFactorModel
from dfm_python.dataset.panel import PanelData
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

from .contract._dummies import _DummyAllocator, _DummyForecaster, _DummyForecasterConfig


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    index = pd.date_range("2001-01-31", periods=12, freq="ME", name="date")
    return PanelData(pd.DataFrame(rng.standard_normal((12, 2)), index=index, columns=["a", "b"]))


class _Bare(BaseFactorModel):
    def __init__(self, *, alpha=1.0):
        self.alpha = alpha


class _Child(_DummyForecaster):
    _tags = {"capability:exog": True}


# -- top-level export -------------------------------------------------------


def test_top_level_base_is_the_estimator_base():
    assert dfm_python.BaseFactorModel is BaseFactorModel
    assert "BaseFactorModel" in dfm_python.__all__


def test_old_engine_base_is_gone():
    # The old abstract engine base in models.base is deleted; that module keeps
    # only the network component bases of DDFM and iVDFM.
    import dfm_python.models
    import dfm_python.models.base as engine_base

    assert not hasattr(engine_base, "BaseFactorModel")
    assert hasattr(engine_base, "BaseEncoder") and hasattr(engine_base, "BaseDecoder")
    assert "BaseFactorModel" not in dfm_python.models.__all__
    assert not hasattr(dfm_python.models, "BaseFactorModel")


@pytest.mark.parametrize(
    "model, name", [("afm", "AFMEngine"), ("ddfm", "DDFMEngine"), ("dfm", "DFMEngine"), ("ivdfm", "iVDFMEngine")]
)
def test_no_engine_has_the_old_base_or_its_entry_points(model, name):
    import importlib

    import torch

    module = importlib.import_module(f"dfm_python.models.{model}._engine")
    engine = getattr(module, name)
    assert not issubclass(engine, BaseFactorModel)
    assert all(base in (object, torch.nn.Module) for base in engine.__bases__), engine.__bases__
    for name in ("save", "load", "reset", "_restore_checkpoint_metadata", "load_config"):
        assert not hasattr(engine, name), (engine.__name__, name)
    for name in ("fit", "predict", "get_result"):
        assert hasattr(engine, name), (engine.__name__, name)
    assert callable(module._init_engine_state)


# -- tags -------------------------------------------------------------------


def test_default_tags():
    assert _Bare.get_tags() == DEFAULT_TAGS
    assert DEFAULT_TAGS["task"] == "forecast" and DEFAULT_TAGS["backend"] == "numpy"
    assert DEFAULT_TAGS["update_kind"] is None and DEFAULT_TAGS["predict_data"] is None
    assert not any(v for k, v in DEFAULT_TAGS.items() if k.startswith("capability:"))
    assert DEFAULT_TAGS["non_deterministic"] is False


def test_get_tags_merges_along_the_mro_base_first():
    tags = _Child.get_tags()
    assert tags["capability:exog"] is True  # the child's own override
    assert tags["capability:update"] is True and tags["predict_data"] == "full_panel"  # from the parent
    assert tags["backend"] == "numpy"  # from the base defaults
    assert _DummyForecaster.get_tags()["capability:exog"] is False  # the parent is unchanged


def test_get_tags_works_on_instances_and_returns_a_copy():
    tags = _Child().get_tags()
    tags["task"] = "allocation"
    assert _Child.get_tags()["task"] == "forecast"


class _SeedTagged(_Bare):
    def __init__(self, *, alpha=1.0, random_state=None):
        self.alpha = alpha
        self.random_state = random_state

    def _instance_tags(self):
        return {"non_deterministic": self.random_state is None}


def test_instance_tags_apply_only_on_instances():
    assert _SeedTagged.get_tags()["non_deterministic"] is False  # class tags
    assert _SeedTagged().get_tags()["non_deterministic"] is True
    assert _SeedTagged(random_state=0).get_tags()["non_deterministic"] is False
    assert _SeedTagged().get_tags()["task"] == "forecast"  # the class tags are still merged in


def test_unknown_instance_tag_raises():
    class _BadInstanceTag(_Bare):
        def _instance_tags(self):
            return {"non_determinstic": True}

    assert _BadInstanceTag.get_tags() == DEFAULT_TAGS
    with pytest.raises(ConfigurationError, match="'non_deterministic'"):
        _BadInstanceTag().get_tags()


def test_unknown_tag_raises_with_closest_name():
    class _Typo(_Bare):
        _tags = {"capability:exogg": True}

    with pytest.raises(ConfigurationError, match="'capability:exog'"):
        _Typo.get_tags()


# -- parameters and config --------------------------------------------------


def test_sklearn_parameter_machinery():
    est = _DummyForecaster(shrink=0.5, lags=(2,))
    assert est.get_params() == {"shrink": 0.5, "window": None, "lags": (2,), "verbose": 0}
    assert clone(est).get_params() == est.get_params()
    assert est.set_params(window=3) is est and est.window == 3
    assert "shrink=0.5" in repr(est)


def test_set_params_unknown_name_names_closest():
    with pytest.raises(ConfigurationError, match="'shrink'") as info:
        _DummyForecaster().set_params(shrnk=1.0)
    assert isinstance(info.value, ValueError)


def test_get_config_and_from_config():
    est = _DummyForecaster(shrink=0.5, window=3)
    cfg = est.get_config()
    assert isinstance(cfg, _DummyForecasterConfig)
    assert cfg.to_dict() == est.get_params()
    rebuilt = _DummyForecaster.from_config(cfg)
    assert type(rebuilt) is _DummyForecaster and rebuilt.get_params() == est.get_params()


def test_from_config_accepts_a_strict_mapping():
    assert _DummyForecaster.from_config({"window": 2}).window == 2
    with pytest.raises(ConfigurationError, match="'window'"):
        _DummyForecaster.from_config({"windw": 2})


def test_from_config_rejects_another_config_type():
    with pytest.raises(ConfigurationError, match="_DummyForecasterConfig"):
        _DummyForecaster.from_config(_DummyAllocator().get_config())


def test_get_config_without_config_class():
    with pytest.raises(NotImplementedError, match="_config_class"):
        _Bare().get_config()


def test_default_test_params():
    assert _Bare.get_test_params() == [{}]


# -- fitted-state guard -----------------------------------------------------


def test_fitted_attributes_raise_not_fitted_before_fit():
    est = _DummyForecaster()
    assert not hasattr(est, "result_")
    with pytest.raises(NotFittedError, match="result_.*fit") as info:
        _ = est.result_
    assert isinstance(info.value, AttributeError) and isinstance(info.value, SklearnNotFittedError)
    with pytest.raises(SklearnNotFittedError):
        check_is_fitted(est)


def test_missing_attributes_after_fit_are_plain_attribute_errors(panel):
    est = _DummyForecaster().fit(panel)
    check_is_fitted(est)
    with pytest.raises(AttributeError) as info:
        _ = est.device_
    assert not isinstance(info.value, NotFittedError)
    with pytest.raises(AttributeError) as info:
        _ = _DummyForecaster()._private_
    assert not isinstance(info.value, NotFittedError)


def test_check_is_fitted_names_the_method():
    with pytest.raises(NotFittedError, match=r"before predict\(\)"):
        _DummyForecaster().predict(2)


def test_copy_and_pickle_keep_the_guard(panel):
    est = _DummyForecaster(window=3)
    assert not hasattr(copy.deepcopy(est), "result_")
    assert not hasattr(pickle.loads(pickle.dumps(est)), "result_")
    fitted = _DummyForecaster().fit(panel)
    restored = pickle.loads(pickle.dumps(fitted))
    assert np.array_equal(restored.mean_, fitted.mean_)


# -- tag-gated default methods ----------------------------------------------


def test_predict_on_an_allocator_is_a_type_error():
    with pytest.raises(TypeError, match="allocate"):
        _DummyAllocator().predict(3)


def test_allocate_on_a_forecaster_is_a_type_error():
    with pytest.raises(TypeError, match=r"predict\(h\)"):
        _DummyForecaster().allocate()


def test_capability_gated_methods_name_the_tag(panel):
    with pytest.raises(UnsupportedInputError, match="capability:factor_forecast"):
        _Bare().predict_factors(2)
    with pytest.raises(UnsupportedInputError, match="capability:update"):
        _Bare().update(panel)


def test_enabled_but_unimplemented_methods():
    class _Enabled(_Bare):
        _tags = {"capability:factor_forecast": True, "capability:update": True, "update_kind": "filter"}

    with pytest.raises(NotImplementedError):
        _Enabled().predict(1)
    with pytest.raises(NotImplementedError):
        _Enabled().predict_factors(1)
    with pytest.raises(NotImplementedError):
        _Enabled().update(None)


@pytest.mark.parametrize("h", [1, np.int64(4)])
def test_validate_horizon_accepts_integers(h):
    assert BaseFactorModel._validate_horizon(h) == int(h)


@pytest.mark.parametrize("h", [1.0, "3", None, True, np.bool_(True)])
def test_validate_horizon_rejects_non_integers(h):
    with pytest.raises(TypeError, match="positive integer"):
        BaseFactorModel._validate_horizon(h)


@pytest.mark.parametrize("h", [0, -2])
def test_validate_horizon_rejects_non_positive(h):
    with pytest.raises(ValueError, match="positive integer"):
        BaseFactorModel._validate_horizon(h)


# -- data validation --------------------------------------------------------


def _frame(index, columns, value=0.0):
    return pd.DataFrame(value, index=index, columns=columns)


def test_validate_data_coerces(panel):
    est = _Bare()
    assert est._validate_data(panel) is panel
    assert est._validate_data(panel.y).equals(panel)
    assert est._validate_data(panel.y.to_numpy()).n_timesteps == 12
    with pytest.raises(TypeError):
        est._validate_data([1, 2, 3])


def test_validate_data_rejects_excluded_slots(panel):
    est = _Bare()
    with_exog = PanelData(panel.y, exog=_frame(panel.index, ["x"]))
    with pytest.raises(UnsupportedInputError, match="capability:exog") as info:
        est._validate_data(with_exog)
    assert "'x'" in str(info.value)

    columns = pd.MultiIndex.from_product([["c"], ["a", "b"]], names=["feature", "series"])
    with_series_exog = PanelData(panel.y, series_exog=_frame(panel.index, columns))
    with pytest.raises(UnsupportedInputError, match="capability:series_exog"):
        est._validate_data(with_series_exog)

    mixed = PanelData(panel.y, freq={"a": "q", "b": "m"})
    with pytest.raises(UnsupportedInputError, match="capability:mixed_frequency"):
        est._validate_data(mixed)

    y = panel.y
    y.iloc[0, 1] = np.nan
    with pytest.raises(UnsupportedInputError, match="capability:missing_values.*y contains NaN"):
        est._validate_data(PanelData(y))

    exog = _frame(panel.index, ["x"])
    exog.iloc[2, 0] = np.nan

    class _ExogOK(_Bare):
        _tags = {"capability:exog": True}

    with pytest.raises(UnsupportedInputError, match="exog contains NaN"):
        _ExogOK()._validate_data(PanelData(panel.y, exog=exog))


def test_validate_data_allows_what_the_tags_enable(panel):
    class _All(_Bare):
        _tags = {
            "capability:exog": True,
            "capability:series_exog": True,
            "capability:mixed_frequency": True,
            "capability:missing_values": True,
        }

    y = panel.y
    y.iloc[0, 0] = np.nan
    columns = pd.MultiIndex.from_product([["c"], ["a", "b"]], names=["feature", "series"])
    data = PanelData(y, exog=_frame(panel.index, ["x"]), series_exog=_frame(panel.index, columns), freq={"a": "q"})
    assert _All()._validate_data(data) is data


def test_single_frequency_freq_is_not_mixed(panel):
    data = PanelData(panel.y, freq={"a": "m", "b": "m"})
    assert _Bare()._validate_data(data) is data


# -- fit ----------------------------------------------------------------------


def test_fit_returns_self_and_sets_data_attributes(panel):
    est = _DummyForecaster()
    assert est.fit(panel) is est
    assert est.n_features_in_ == 2
    assert est.feature_names_in_.dtype == object and list(est.feature_names_in_) == ["a", "b"]
    assert est.exog_names_in_.shape == (0,)
    assert est.index_.equals(panel.index)
    assert "series_exog_names_in_" not in vars(est)


def test_fit_sets_series_exog_names_when_capable(panel):
    columns = pd.MultiIndex.from_product([["c", "d"], ["a", "b"]], names=["feature", "series"])
    data = PanelData(panel.y * 0.01, series_exog=_frame(panel.index, columns))
    est = _DummyAllocator(hist_len=2).fit(data)
    assert list(est.series_exog_names_in_) == ["c", "d"]


def test_fit_accepts_a_dataframe(panel):
    est = _DummyForecaster().fit(panel.y)
    assert list(est.feature_names_in_) == ["a", "b"]


def test_refit_starts_from_scratch(panel):
    est = _DummyForecaster().fit(panel)
    est.stale_ = 1
    est.fit(panel)
    assert "stale_" not in vars(est)


def test_failed_fit_leaves_the_model_unfitted(panel):
    class _Fails(_DummyForecaster):
        def _fit(self, data, **fit_kwargs):
            self.partial_ = 1
            raise RuntimeError("boom")

    est = _Fails()
    with pytest.raises(RuntimeError, match="boom"):
        est.fit(panel)
    assert not est._is_fitted() and not hasattr(est, "index_")


def test_rejected_data_leaves_a_fitted_model_untouched(panel):
    est = _DummyForecaster().fit(panel)
    mean = est.mean_
    with pytest.raises(UnsupportedInputError):
        est.fit(PanelData(panel.y, exog=_frame(panel.index, ["x"])))
    assert est.mean_ is mean


def test_callback_reaches_fit(panel):
    class _Kw(_DummyForecaster):
        def _fit(self, data, *, callback=None):
            super()._fit(data)
            self.callback_ = callback

    def callback(info):
        return None

    assert _Kw().fit(panel, callback=callback).callback_ is callback
    assert _Kw().fit(panel).callback_ is None
    with pytest.raises(TypeError, match="callback must be None or a callable"):
        _Kw().fit(panel, callback="cb")


def test_a_max_time_stop_warns(panel):
    def timed(reason):
        class _Timed(_DummyForecaster):
            max_time = 2.0

            def _fit(self, data, *, callback=None):
                super()._fit(data)
                self.n_iter_ = 3
                self.stop_reason_ = reason

        return _Timed()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        timed("max_iter").fit(panel)
    with pytest.warns(ConvergenceWarning, match=r"_Timed: training stopped after 3 iteration\(s\).*max_time=2\.0"):
        timed("max_time").fit(panel)


def test_unimplemented_fit(panel):
    with pytest.raises(NotImplementedError, match="_fit"):
        _Bare().fit(panel)


def test_fit_is_silent_with_verbose_0(panel, capsys, caplog):
    class _Progress(_DummyForecaster):
        def _fit(self, data, **fit_kwargs):
            super()._fit(data)
            if self.verbose:
                logging.getLogger("dfm_python.test_progress").info("fitted %d rows", data.n_timesteps)

    with caplog.at_level(logging.INFO):
        _Progress().fit(panel)
    assert capsys.readouterr().out == "" and not caplog.records

    _Progress(verbose=1).fit(panel)
    assert "fitted 12 rows" in capsys.readouterr().out
    package_logger = logging.getLogger("dfm_python")
    assert not [h for h in package_logger.handlers if getattr(h, "_dfm_python_fit_handler", False)]


def test_invalid_verbose_raises(panel):
    with pytest.raises(ConfigurationError, match="verbose"):
        _DummyForecaster(verbose=-1).fit(panel)


# -- forecast index -----------------------------------------------------------


def test_forecast_index_uses_freq_then_inferred_freq():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME", name="date")
    expected = pd.date_range("2020-06-30", periods=3, freq="ME", name="date")
    assert BaseFactorModel._forecast_index(idx, 3).equals(expected)
    no_freq = pd.DatetimeIndex(list(idx), name="date")
    assert no_freq.freq is None
    out = BaseFactorModel._forecast_index(no_freq, 3)
    assert out.equals(expected) and out.name == "date"


def test_forecast_index_irregular_datetime_uses_the_median_step():
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-08", "2020-01-14", "2020-01-22", "2020-01-29"])
    out = BaseFactorModel._forecast_index(idx, 2)
    assert list(out) == [pd.Timestamp("2020-02-05"), pd.Timestamp("2020-02-12")]
    assert out.name == "time"


def test_forecast_index_range_and_period():
    assert BaseFactorModel._forecast_index(pd.RangeIndex(0, 10, 2), 3).equals(pd.RangeIndex(10, 16, 2))
    assert BaseFactorModel._forecast_index(pd.RangeIndex(5), 2).name == "time"
    per = pd.period_range("2020Q1", periods=4, freq="Q", name="q")
    assert BaseFactorModel._forecast_index(per, 2).equals(pd.period_range("2021Q1", periods=2, freq="Q", name="q"))


def test_forecast_index_single_stamp_without_freq_raises():
    with pytest.raises(DataValidationError, match="single time stamp"):
        BaseFactorModel._forecast_index(pd.DatetimeIndex(["2020-01-01"]), 2)

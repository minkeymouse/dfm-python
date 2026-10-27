"""The new base classes (``dfm_python._estimator``) through dummy models and ``_testing.check_estimator``."""

from __future__ import annotations

import copy
import inspect
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError as SklearnNotFittedError

import dfm_python
from dfm_python import PanelData
from dfm_python._estimator import STOP_REASONS, TrainingLog
from dfm_python._testing import check_estimator, make_panel
from dfm_python.errors import ConfigurationError, DataValidationError, NotFittedError
from dfm_python.io import loading

from ._base_dummies import DummyAllocator, DummyForecaster, DummyTorchForecaster

ALL_CHECKS = ["parameters", "construction", "unfitted", "fit", "fit_attributes", "callback", "namespaces", "outputs",
              "purity", "update", "silence", "save_load"]


def _fit(cls=DummyForecaster, data=None, **params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cls(**params).fit(make_panel(cls) if data is None else data)


# ---------------------------------------------------------------------------
# The harness on the dummies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [DummyForecaster, DummyAllocator, DummyTorchForecaster], ids=lambda c: c.__name__)
def test_dummies_pass_check_estimator(cls):
    assert check_estimator(cls) == ALL_CHECKS


def test_instance_is_checked_with_its_own_params():
    assert "save_load" in check_estimator(DummyForecaster(max_iter=2))


def test_check_estimator_rejects_other_classes():
    with pytest.raises(TypeError, match="_Forecaster or _Allocator"):
        check_estimator(object)


class _ForgetsData(DummyForecaster):
    def _moments(self, stamps, exog, coverage):
        mean, lower, upper = super()._moments(stamps, exog, coverage)
        return mean + 1.0, lower, upper


def test_base_reports_data_at_observed_cells_whatever_the_hook_returns():
    # The base, not the model, puts the data at observed in-sample cells.
    assert check_estimator(_ForgetsData)[-1] == "save_load"


class _ExtraAttribute(DummyForecaster):
    def _condition(self, panel, previous):
        super()._condition(panel, previous)
        self.cache = 1


class _ChangesParams(DummyForecaster):
    def _condition(self, panel, previous):
        super()._condition(panel, previous)
        if previous is not None:
            self.params_ = self.params_.__class__("p", [], {})


@pytest.mark.parametrize("cls, check", [(_ExtraAttribute, "fit_attributes"), (_ChangesParams, "update")])
def test_broken_models_fail_their_check(cls, check):
    with pytest.raises(AssertionError, match=f"check {check!r} failed"):
        check_estimator(cls)


# ---------------------------------------------------------------------------
# sklearn shell
# ---------------------------------------------------------------------------


def test_keyword_only_constructor_and_clone():
    params = [p for p in inspect.signature(DummyForecaster.__init__).parameters.values()][1:]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
    with pytest.raises(TypeError):
        DummyForecaster(3)
    model = _fit(max_iter=2, sign_anchors={"factor_0": "s1"})
    cloned = clone(model)
    assert cloned.get_params() == model.get_params() and not cloned._is_fitted()
    assert cloned.sign_anchors is not model.sign_anchors and cloned.sign_anchors == model.sign_anchors


def test_set_params_names_the_closest_parameter():
    model = DummyForecaster()
    assert model.set_params(max_iter=5) is model and model.max_iter == 5
    with pytest.raises(ConfigurationError, match=r"'max_itr' \(did you mean 'max_iter'\?\)"):
        model.set_params(max_itr=5)


class _Spec(BaseEstimator):
    def __init__(self, *, width=3):
        self.width = width


class _NestedForecaster(DummyForecaster):
    """A forecaster with a spec hyperparameter, like the AFM specs (``factors__embed_dim``)."""

    def __init__(self, *, max_iter=3, tol=0.0, sign_anchors=None, clock=None, verbose=0, spec=None):
        super().__init__(max_iter=max_iter, tol=tol, sign_anchors=sign_anchors, clock=clock, verbose=verbose)
        self.spec = spec


def test_set_params_checks_nested_names():
    model = _NestedForecaster(spec=_Spec())
    assert model.set_params(spec__width=5).spec.width == 5
    with pytest.raises(ConfigurationError, match=r"'spec__widht' \(did you mean 'spec__width'\?\)"):
        model.set_params(max_iter=9, spec__widht=5)
    assert model.max_iter == 3  # nothing is set when a name is unknown
    # a spec passed in the same call is the one checked
    assert _NestedForecaster().set_params(spec=_Spec(), spec__width=7).spec.width == 7
    with pytest.raises(ConfigurationError, match=r"'spec__width'.*hold no estimator.*\['spec'\]"):
        _NestedForecaster().set_params(spec__width=5)


def test_no_metadata_request_setters():
    for cls in (DummyForecaster, DummyAllocator):
        assert not [n for n in dir(cls()) if n.startswith("set_") and n.endswith("_request")]
        assert cls().get_metadata_routing() is not None


def test_not_fitted_errors():
    model = DummyForecaster()
    for name in ("params_", "states_", "factors_", "index_"):
        assert not hasattr(model, name)
    with pytest.raises(NotFittedError, match=r"not fitted yet, so it has no attribute 'params_'"):
        _ = model.params_
    with pytest.raises(SklearnNotFittedError):
        model.predict()
    with pytest.raises(AttributeError):
        model.nowcast()
    with pytest.raises(NotFittedError, match=r"before allocate\(\)"):
        DummyAllocator().allocate()
    with pytest.raises(AttributeError, match="no attribute 'other'"):
        _ = model.other


def test_invalid_hyperparameters_and_data_leave_the_model_unfitted():
    model = _fit()
    with pytest.raises(ConfigurationError, match="max_iter"):
        model.set_params(max_iter=0).fit(make_panel(DummyForecaster))
    assert not model._is_fitted()


# ---------------------------------------------------------------------------
# fit: diagnostics, callback, warnings
# ---------------------------------------------------------------------------


def test_history_and_diagnostics():
    model = _fit(max_iter=3)
    assert model.n_iter_ == 3 == len(model.history_)
    assert model.stop_reason_ == "max_iter" and model.converged_ is False
    assert list(model.history_.columns) == ["objective", "elapsed"] and model.history_.index.name == "iteration"
    assert (model.history_.dtypes == np.float64).all() and model.history_["elapsed"].is_monotonic_increasing
    allocator = _fit(DummyAllocator)
    assert list(allocator.history_.columns) == ["objective", "stage", "elapsed"]
    idle = _fit(DummyAllocator, max_iter=0)
    assert idle.n_iter_ == 0 and idle.stop_reason_ == "not_trainable" and idle.converged_
    assert list(idle.history_.columns) == ["objective", "elapsed"] and len(idle.history_) == 0


def test_max_iter_warns_for_models_that_declare_it():
    with pytest.warns(ConvergenceWarning, match=r"stopped after 3 iteration\(s\) because max_iter") as record:
        DummyForecaster(max_iter=3).fit(make_panel(DummyForecaster))
    assert record[0].filename == __file__  # the warning points at the caller of fit
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        DummyForecaster(max_iter=50, tol=1.0).fit(make_panel(DummyForecaster))  # converged: no warning
        DummyAllocator().fit(make_panel(DummyAllocator))  # allocators do not declare it


def test_callback_info_and_stop():
    infos = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = DummyForecaster(max_iter=3).fit(make_panel(DummyForecaster), callback=infos.append)
    assert [info["iteration"] for info in infos] == [1, 2, 3]
    assert all(set(info) == {"iteration", "objective", "elapsed"} for info in infos)
    assert [info["objective"] for info in infos] == model.history_["objective"].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stopped = DummyForecaster(max_iter=5).fit(make_panel(DummyForecaster), callback=lambda info: info["iteration"] == 2)
    assert stopped.stop_reason_ == "callback" and stopped.n_iter_ == 2 and not stopped.converged_
    with pytest.raises(TypeError, match="callback must be None or a callable"):
        DummyForecaster().fit(make_panel(DummyForecaster), callback="print")


def test_raising_callback_leaves_the_model_unfitted():
    model = _fit()

    class Stop(Exception):
        pass

    def callback(info):
        raise Stop

    with pytest.raises(Stop):
        model.fit(make_panel(DummyForecaster), callback=callback)
    assert not model._is_fitted() and not hasattr(model, "stop_reason_") and "_panel_" not in vars(model)


def test_training_log():
    seen = []
    log = TrainingLog(lambda info: seen.append(info) or info["iteration"] == 2)
    assert log(1.5, kl=np.float32(0.25)) is False
    assert log(np.nan, kl=0.5) is True
    frame = log.frame()
    assert list(frame.columns) == ["objective", "kl", "elapsed"] and np.isnan(frame["objective"].iloc[1])
    assert seen[0]["iteration"] == 1 and seen[0]["kl"] == 0.25 and type(seen[0]["kl"]) is float
    assert set(STOP_REASONS) == {"converged", "early_stopping", "max_iter", "callback", "numerical_guard", "not_trainable"}


def test_fit_time_seed():
    torch_model = DummyTorchForecaster(device="cpu", random_state=7)
    assert torch_model._draw_seed() == DummyTorchForecaster(random_state=7)._draw_seed()
    assert DummyTorchForecaster(random_state=None)._draw_seed() != DummyTorchForecaster(random_state=None)._draw_seed()
    assert DummyForecaster()._draw_seed() is None  # no random_state parameter
    with pytest.raises(ConfigurationError, match="random_state"):
        DummyTorchForecaster(random_state=-1)._draw_seed()


# ---------------------------------------------------------------------------
# Data checks at fit (spec section 4.12)
# ---------------------------------------------------------------------------


def _frame(index, n_series=3, seed=0):
    values = np.random.default_rng(seed).standard_normal((len(index), n_series))
    return pd.DataFrame(values, index=index, columns=[f"s{i}" for i in range(n_series)])


def test_gaps_and_irregular_indexes_are_rejected_with_the_reindex_hint():
    index = pd.date_range("2020-01-31", periods=30, freq="ME", name="date").delete([4, 5])
    with pytest.raises(DataValidationError, match=r"DummyForecaster.fit: the time index has gaps.*df.reindex"):
        DummyForecaster().fit(_frame(index))


def test_clock_must_match_the_index():
    index = pd.date_range("2020-01-31", periods=30, freq="ME")
    with pytest.raises(DataValidationError, match="clock='q' does not match"):
        DummyForecaster(clock="q").fit(_frame(index))
    with pytest.raises(ConfigurationError, match="clock must be one of"):
        DummyForecaster(clock="monthly").fit(_frame(index))


def test_slower_series_must_sit_at_their_anchors():
    index = pd.date_range("2020-01-31", periods=30, freq="ME")
    df = _frame(index)
    df.loc[index.month % 3 != 1, "s0"] = np.nan  # January, April, ...: not quarter ends
    with pytest.raises(DataValidationError, match=r"'s0'.*last stamp of each quarter"):
        DummyForecaster().fit(PanelData(df, freq={"s0": "q"}))
    with pytest.raises(DataValidationError, match="faster than the clock"):
        DummyForecaster().fit(PanelData(_frame(index), freq={"s0": "w"}))
    with pytest.raises(DataValidationError, match="explicit clock"):
        DummyForecaster().fit(PanelData(_frame(pd.RangeIndex(30)), freq={"s0": "q"}))


def test_mixed_frequency_is_rejected_by_models_that_do_not_accept_it():
    panel = make_panel(DummyForecaster)
    with pytest.raises(DataValidationError, match="does not accept mixed-frequency data"):
        DummyTorchForecaster(device="cpu").fit(panel)


def test_slots_and_infinite_values():
    panel = make_panel(DummyForecaster)
    exog = pd.DataFrame({"x": 1.0}, index=panel.index)
    with pytest.raises(DataValidationError, match=r"DummyForecaster does not accept exog; got exog columns \['x'\]"):
        DummyForecaster().fit(PanelData(panel.y, exog=exog, freq=panel.freq))
    chars = make_panel(DummyAllocator)
    with pytest.raises(DataValidationError, match="does not accept characteristics"):
        DummyForecaster().fit(PanelData(chars.y, characteristics=chars.characteristics))
    y = panel.y
    y.iloc[5, 1] = np.inf
    with pytest.raises(DataValidationError, match=r"y contains infinite values in \['s1'\]"):
        DummyForecaster().fit(PanelData(y, freq=panel.freq))


def test_degenerate_targets_are_rejected():
    df = _frame(pd.date_range("2020-01-31", periods=30, freq="ME"))
    df["s1"] = 2.5
    df["s2"] = np.nan
    with pytest.raises(DataValidationError, match=r"'s1' is constant.*'s2' has no observed value"):
        DummyForecaster().fit(df)


# ---------------------------------------------------------------------------
# predict family and nowcast
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed():
    """A DummyForecaster on the month-end fixture with a quarterly s0, fitted through 2003-12."""
    panel = make_panel(DummyForecaster)
    return _fit(data=panel._take(slice(0, 47))), panel


def test_in_sample_observed_cells_equal_the_data(mixed):
    model, panel = mixed
    past = model.predict(0, start=-11)
    data = panel.y.to_numpy()[35:47]
    observed = ~np.isnan(data)
    np.testing.assert_array_equal(past.to_numpy()[observed], data[observed])
    bands = model.predict_interval(0, start=-11)
    for bound in ("lower", "upper"):
        np.testing.assert_array_equal(bands.xs(bound, axis=1, level="bound").to_numpy()[observed], data[observed])
    # a missing in-sample cell gets the model estimate
    assert np.isnan(panel.y.iloc[17, 2]) and np.isfinite(model.predict(17 - 46, start=17 - 46).iloc[0, 2])


def test_slower_series_are_reported_at_period_ends_only(mixed):
    model, _ = mixed
    frame = model.predict(6, start=-5)
    quarter_end = frame.index.month % 3 == 0
    assert frame["s0"][quarter_end].notna().all() and frame["s0"][~quarter_end].isna().all()
    assert frame[["s1", "s2"]].notna().all().all()
    bands = model.predict_interval(6, start=-5)
    assert bands["s0"][~quarter_end].isna().all().all()


def test_targets_and_labels(mixed):
    model, _ = mixed
    assert model.index_[-1] == pd.Timestamp("2003-11-30")
    assert list(model.predict("2004Q1").index.strftime("%Y-%m")) == ["2003-12", "2004-01", "2004-02", "2004-03"]
    frame = model.predict(2)
    assert frame.index.name == "date" and frame.columns.name == "series" and frame.index.freq == "ME"
    factors = model.predict_factors(2, start=-1)
    assert list(factors.columns) == ["factor_0"] and factors.columns.name == "factor" and len(factors) == 4
    np.testing.assert_array_equal(factors.iloc[:2].to_numpy(), model.factors_.iloc[-2:].to_numpy())
    with pytest.raises(TypeError):
        model.predict(1.0)
    with pytest.raises(ConfigurationError, match="coverage"):
        model.predict_interval(coverage=1.0)
    with pytest.raises(TypeError):
        model.predict_interval(1, 0.8)  # coverage is keyword-only


def test_exog_must_be_none_for_models_that_do_not_use_it(mixed):
    model, _ = mixed
    with pytest.raises(ConfigurationError, match="takes no exog here: DummyForecaster's forecasts do not use"):
        model.predict(1, exog=pd.DataFrame({"x": [1.0]}, index=model.predict().index))


def test_nowcast_table(mixed):
    model, panel = mixed
    table = model.nowcast()  # the slower series by default
    assert table.index.names == ["series", "period"]
    assert list(table.index.get_level_values("series").unique()) == ["s0"]
    labels = [str(p) for p in table.index.get_level_values("period")]
    assert labels == ["2003Q3", "2003Q4", "2004Q1"]  # the latest release, the nowcast quarter, the next one
    assert list(table["status"]) == ["released", "nowcast", "forecast"]
    released = table.iloc[0]
    assert released["estimate"] == released["lower"] == released["upper"] == released["actual"]
    point = model.predict(4, start=-1)["s0"]
    assert table.loc[("s0", pd.Period("2003Q4")), "estimate"] == point.loc["2003-12-31"]
    assert table.loc[("s0", pd.Period("2004Q1")), "estimate"] == point.loc["2004-03-31"]
    clock = model.nowcast(["s2", "s1"], end=0, start=-2)
    assert list(clock.index.get_level_values("series")) == ["s1"] * 3 + ["s2"] * 3  # the user's order
    assert set(clock["status"]) <= {"released", "nowcast", "backcast"}
    both = model.nowcast(["s1", "s0"], end=0, start=-1)  # a quarterly and a monthly series together
    assert list(both.index.get_level_values("series")) == ["s0"] * 2 + ["s1"] * 2
    assert [str(p) for p in both.index.get_level_values("period")] == ["2003Q3", "2003Q4", "2003-10", "2003-11"]
    assert both.loc[("s0", pd.Period("2003Q4")), "estimate"] == table.loc[("s0", pd.Period("2003Q4")), "estimate"]
    assert both.loc["s1"].equals(model.nowcast("s1", end=0, start=-1).loc["s1"])
    with pytest.raises(ConfigurationError, match="unknown series"):
        model.nowcast("gdp")


def test_nowcast_labels_on_a_range_index_are_the_predict_row_labels():
    # the period-end stamp value, not its position from index_[0] (38, 39, 40 here)
    model = _fit(data=_frame(pd.RangeIndex(100, 180, 2)))
    table = model.nowcast("s1", end=1, start=-1)
    assert list(table.index.get_level_values("period")) == [176, 178, 180]
    point = model.predict(1, start=-1)["s1"]
    assert list(point.index) == [176, 178, 180]
    np.testing.assert_array_equal(table["estimate"].to_numpy(), point.to_numpy())


def test_nowcast_backcast_status_for_an_unreleased_past_period():
    panel = make_panel(DummyForecaster)
    y = panel.y
    y.loc[["2003-09-30", "2003-12-31"], "s0"] = np.nan  # 2003Q3 and 2003Q4 not yet released
    model = _fit(data=PanelData(y, freq=panel.freq))
    assert model.index_[-1] == pd.Timestamp("2003-12-31")
    table = model.nowcast(end=0)  # start: the latest released quarter, 2003Q2
    assert [str(p) for p in table.index.get_level_values("period")] == ["2003Q2", "2003Q3", "2003Q4"]
    assert list(table["status"]) == ["released", "backcast", "nowcast"]  # 2003Q4 ends at the origin


class _ExogForecaster(DummyForecaster):
    """A forecaster whose forecast uses future exog: the exog sum is added to every series after the origin."""

    _accepts_exog = True
    _accepts_mixed_frequency = False
    _forecast_uses_exog = True

    def _moments(self, stamps, exog, coverage):
        mean, lower, upper = super()._moments(stamps, exog, coverage)
        if exog is not None:
            n = len(self.index_)
            positions = self._target_positions(stamps)
            future = positions >= n
            shift = exog.sum(axis=1)[positions[future] - n][:, None]
            mean[future] += shift
            if lower is not None:
                lower[future] += shift
                upper[future] += shift
        return mean, lower, upper


def _exog_panel():
    index = pd.date_range("2020-01-31", periods=30, freq="ME", name="date")
    exog = pd.DataFrame({"u": np.linspace(0, 1, 30), "v": 1.0}, index=index)
    return PanelData(_frame(index), exog=exog), index


def test_future_exog_rules():
    panel, index = _exog_panel()
    model = _fit(_ExogForecaster, data=panel)
    assert model.exog_names_in_ == ("u", "v")
    future = pd.DataFrame({"v": [1.0, 2.0, 3.0], "u": 0.0}, index=pd.date_range("2022-07-31", periods=3, freq="ME"))
    with pytest.raises(DataValidationError, match=r"exog needed for 2022-07-31..2022-08-31.*\['u', 'v'\]"):
        model.predict(2)
    shifted = model.predict(2, exog=future) - model.predict(2, exog=future * 0.0)
    np.testing.assert_allclose(shifted.to_numpy(), [[1.0] * 3, [2.0] * 3])
    with pytest.raises(DataValidationError, match="lacks the stamps 2022-09-30"):
        model.predict(3, exog=future.iloc[:2])
    with pytest.raises(DataValidationError, match=r"lacks the columns \['u'\]"):
        model.predict(1, exog=future[["v"]])
    with pytest.raises(DataValidationError, match="missing or infinite"):
        model.predict(1, exog=future.assign(u=np.nan))
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict(1, exog=future.to_numpy())
    with pytest.raises(ConfigurationError, match="every target lies in the conditioning data"):
        model.predict(0, exog=future)
    assert model.predict(0).shape == (1, 3)  # in-sample targets use the conditioning exog
    unexog = _fit(_ExogForecaster, data=PanelData(panel.y))
    with pytest.raises(ConfigurationError, match="fitted without exog"):
        unexog.predict(1, exog=future)
    nowcast = model.nowcast("s0", end=1, start=0, exog=future)
    assert list(nowcast["status"]) == ["released", "forecast"]
    with pytest.raises(DataValidationError, match="repeats the stamps 2022-07-31"):
        model.predict(1, exog=pd.concat([future, future.iloc[:1]]))


def test_future_exog_on_range_index_data_matches_any_integer_index():
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.standard_normal((30, 3)), columns=["s0", "s1", "s2"])
    model = _fit(_ExogForecaster, data=PanelData(y, exog=pd.DataFrame({"u": rng.standard_normal(30)})))
    expected = model.predict(2, exog=pd.DataFrame({"u": [1.0, 2.0]}, index=pd.RangeIndex(30, 32)))
    for index in ([30, 31], np.arange(30, 32), np.array([30, 31], dtype=np.uint32)):
        frame = model.predict(2, exog=pd.DataFrame({"u": [1.0, 2.0]}, index=index))
        pd.testing.assert_frame_equal(frame, expected)
    # extra rows in any order are ignored
    frame = model.predict(2, exog=pd.DataFrame({"u": [9.0, 2.0, 1.0]}, index=[40, 31, 30]))
    pd.testing.assert_frame_equal(frame, expected)
    with pytest.raises(DataValidationError, match="lacks the stamps 31"):
        model.predict(2, exog=pd.DataFrame({"u": [1.0, 2.0]}, index=[30, 32]))
    with pytest.raises(DataValidationError, match="lacks the stamps 30, 31"):
        model.predict(2, exog=pd.DataFrame({"u": [1.0, 2.0]}, index=pd.date_range("2020-01-31", periods=2)))


# ---------------------------------------------------------------------------
# update: the merge rule and the conditioning
# ---------------------------------------------------------------------------


def test_update_merges_and_replace_conditions_on_exactly_the_data(mixed):
    model, panel = mixed
    model = _fit(data=panel._take(slice(0, 40)))
    params = model.params_
    model.update(panel._take(slice(40, 48)))
    assert model.index_.equals(panel.index) and model.params_ is params
    # a revision of one cell inside the conditioning data
    revised = pd.DataFrame({"s0": np.nan, "s1": 99.0, "s2": np.nan}, index=panel.index[[10]])
    model.update(PanelData(revised, freq=panel.freq))
    assert model._panel_.y.iloc[10]["s1"] == 99.0 and model._panel_.y.iloc[10]["s2"] == panel.y.iloc[10]["s2"]
    assert model.predict(-37, start=-37)["s1"].iloc[0] == 99.0
    # a NaN never erases an observed value
    model.update(PanelData(panel.y * np.nan, freq=panel.freq))
    assert model._panel_.y.iloc[10]["s1"] == 99.0
    # replace=True conditions on exactly the data; starting before index_[0] needs it
    with pytest.raises(DataValidationError, match="replace=True"):
        _fit(data=panel._take(slice(10, 48))).update(panel)
    model.update(panel._take(slice(0, 20)), replace=True)
    assert model.index_.equals(panel.index[:20]) and model._panel_.y.iloc[10]["s1"] == panel.y.iloc[10]["s1"]


def test_update_checks_the_grid_and_the_anchors(mixed):
    model, panel = mixed
    gap = PanelData(_frame(pd.DatetimeIndex(["2004-02-29"], name="date")), freq=panel.freq)
    with pytest.raises(DataValidationError, match="DummyForecaster.update: the time index has gaps"):
        model.update(gap)
    off = panel._take(slice(47, 48)).y
    off.iloc[0, 0] = np.nan
    late = pd.DataFrame({"s0": [1.0], "s1": [1.0], "s2": [1.0]}, index=pd.DatetimeIndex(["2004-01-31"], name="date"))
    with pytest.raises(DataValidationError, match="'s0'.*last stamp of each quarter"):
        model.update(PanelData(pd.concat([off, late]), freq=panel.freq))
    with pytest.raises(DataValidationError, match="misses"):
        model.update(panel.y[["s0", "s1"]])
    with pytest.raises(TypeError, match="replace"):
        model.update(panel, replace="yes")
    assert model.index_[-1] == pd.Timestamp("2003-11-30")  # a failed update changes nothing
    ranged = _fit(data=PanelData(make_panel(DummyForecaster).y[["s1", "s2"]].reset_index(drop=True)))
    with pytest.raises(DataValidationError, match=r"misses 2 row\(s\), first 48, 49"):
        ranged.update(pd.DataFrame({"s1": [1.0], "s2": [1.0]}, index=pd.RangeIndex(50, 51)))


def test_failed_conditioning_leaves_the_model_as_it_was():
    class Fragile(DummyForecaster):
        def _condition(self, panel, previous):
            if previous is not None and panel.n_periods > 45:
                raise RuntimeError("boom")
            super()._condition(panel, previous)

    panel = make_panel(DummyForecaster)
    model = _fit(Fragile, data=panel._take(slice(0, 40)))
    before = model.factors_
    with pytest.raises(RuntimeError, match="boom"):
        model.update(panel)
    assert model.factors_ is before and model.index_.equals(panel.index[:40])


def test_allocator_update_adds_new_assets_and_allocate_uses_the_forward_stamp():
    panel = make_panel(DummyAllocator)
    model = _fit(DummyAllocator, data=panel._take(slice(0, 70)))
    weights = model.allocate()
    assert weights.index[0] == model.index_[-1] + pd.offsets.BDay() and weights.index.name == "date"
    np.testing.assert_allclose(np.abs(weights.to_numpy()).sum(), 1.0)
    pd.testing.assert_frame_equal(weights, model.states_.weights.iloc[[-1]] * 0.0 + weights.to_numpy())
    new = panel._take(slice(70, 80))
    y = new.y.assign(a6=0.01)
    chars = new.characteristics
    extra = pd.DataFrame(0.0, index=new.index, columns=pd.MultiIndex.from_product([["c0", "c1"], ["a6"]],
                                                                                    names=["characteristic", "series"]))
    model.update(PanelData(y, exog=new.exog, characteristics=pd.concat([chars, extra], axis=1)))
    assert list(model.allocate().columns) == [f"a{i}" for i in range(7)]
    assert list(model.feature_names_in_) == [f"a{i}" for i in range(6)]  # the training assets
    assert np.isnan(model._panel_.y["a6"].iloc[:70]).all()


def test_custom_business_day_allocator_round_trip(tmp_path):
    cbd = pd.offsets.CustomBusinessDay(holidays=["2024-01-15", "2024-02-19"])
    index = pd.date_range("2024-01-02", periods=40, freq=cbd, name="day")
    returns = 0.01 * np.random.default_rng(3).standard_normal((40, 4))
    panel = PanelData.from_arrays(returns, index=index)
    model = _fit(DummyAllocator, data=panel)
    assert model.allocate().index[0] == index[-1] + cbd
    model.save(tmp_path / "run")
    loaded = DummyAllocator._load(tmp_path / "run")
    assert loaded.index_.freq == cbd and loaded.allocate().index.equals(model.allocate().index)


# ---------------------------------------------------------------------------
# save / load (format 2)
# ---------------------------------------------------------------------------


def test_format_2_file_set_and_config(tmp_path, mixed):
    model, _ = mixed
    model.save(tmp_path / "run")
    files = sorted(p.name for p in (tmp_path / "run").iterdir())
    assert files == ["config.json", "data.json", "data.npz", "fitted.json", "fitted.npz"]
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["format_version"] == 2 and config["model_type"] == "DummyForecaster"
    assert config["library_version"] == dfm_python.__version__
    assert config["params"] == {"clock": None, "max_iter": 3, "sign_anchors": None, "tol": 0.0, "verbose": 0}
    assert config["runtime"] == {"numpy_version": np.__version__}
    fitted = json.loads((tmp_path / "run" / "fitted.json").read_text())
    assert fitted["base"]["stop_reason_"] == "max_iter" and fitted["base"]["slower"] == [["s0", "q"]]
    assert fitted["base"]["grid"] == "ME" and fitted["base"]["clock"] == "m"


def test_round_trip_is_bitwise(tmp_path, mixed):
    model, panel = mixed
    model.save(tmp_path / "run")
    loaded = DummyForecaster._load(tmp_path / "run")
    pd.testing.assert_frame_equal(loaded.predict(6, start=-6), model.predict(6, start=-6), check_exact=True)
    pd.testing.assert_frame_equal(loaded.nowcast(), model.nowcast(), check_exact=True)
    pd.testing.assert_frame_equal(loaded.history_, model.history_, check_exact=True)
    assert loaded.index_.freq == model.index_.freq and loaded._slower_ == {"s0": "q"}
    np.testing.assert_array_equal(loaded._transform_, model._transform_)
    pd.testing.assert_frame_equal(loaded.params_.loadings, model.params_.loadings, check_exact=True)


def test_torch_round_trip(tmp_path):
    model = _fit(DummyTorchForecaster, data=make_panel(DummyTorchForecaster), device="cpu")
    model.save(tmp_path / "run")
    assert (tmp_path / "run" / "weights.pt").is_file()
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["runtime"]["device"] == "cpu" and "torch_version" in config["runtime"]
    loaded = DummyTorchForecaster._load(tmp_path / "run", device="cpu")
    np.testing.assert_allclose(loaded.predict(3).to_numpy(), model.predict(3).to_numpy(), rtol=1e-12)


def test_format_1_and_other_runs_are_rejected(tmp_path, mixed):
    model, _ = mixed
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"format_version": 1, "model_type": "DummyForecaster", "params": {}}))
    with pytest.raises(ConfigurationError, match=r"saved by dfm-python 0.8 \(format 1\); load it with that version"):
        DummyForecaster._load(old)
    model.save(tmp_path / "run")
    with pytest.raises(ConfigurationError, match="holds a 'DummyForecaster' run, not DummyAllocator"):
        DummyAllocator._load(tmp_path / "run")
    (tmp_path / "run" / "fitted.json").unlink()
    with pytest.raises(ConfigurationError, match="fitted.json missing"):
        DummyForecaster._load(tmp_path / "run")
    (tmp_path / "single.pkl").write_bytes(b"old")
    with pytest.raises(ConfigurationError, match="release of dfm-python that wrote them"):
        DummyForecaster._load(tmp_path / "single.pkl")


def test_dfm_python_load_dispatches_through_the_private_table(tmp_path, mixed, monkeypatch):
    model, _ = mixed
    model.save(tmp_path / "run")
    with pytest.raises(ConfigurationError, match="'DummyForecaster' run, which is not a model of this release"):
        dfm_python.load(tmp_path / "run")
    monkeypatch.setitem(loading._MODELS, "DummyForecaster", DummyForecaster)
    loaded = dfm_python.load(tmp_path / "run")
    assert type(loaded) is DummyForecaster
    pd.testing.assert_frame_equal(loaded.predict(2), model.predict(2), check_exact=True)
    monkeypatch.setitem(loading._MODELS, "DummyForecaster", f"{DummyForecaster.__module__}:DummyForecaster")
    assert type(dfm_python.load(tmp_path / "run")) is DummyForecaster
    assert set(loading._MODELS) >= {"DFM", "DDFM", "iVDFM", "AFM"}


def test_save_refuses_an_existing_directory_and_leaves_no_debris(tmp_path, mixed):
    model, _ = mixed
    model.save(tmp_path / "run")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        model.save(tmp_path / "run")
    model.save(tmp_path / "run", overwrite=True)

    class Broken(DummyForecaster):
        def _load_fitted(self, arrays, values, *, weights=None, device=None):
            raise RuntimeError("cannot rebuild")

    broken = _fit(Broken, data=make_panel(DummyForecaster))
    with pytest.raises(RuntimeError, match="cannot rebuild"):
        broken.save(tmp_path / "broken")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_save_verifies_the_reloaded_output(tmp_path, mixed):
    class Drifts(DummyForecaster):
        def _load_fitted(self, arrays, values, *, weights=None, device=None):
            super()._load_fitted(arrays, values, weights=weights, device=device)
            self._center_ = self._center_ + 1e-9

    model = _fit(Drifts, data=make_panel(DummyForecaster))
    with pytest.raises(RuntimeError, match="does not reproduce"):
        model.save(tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_update_after_load_equals_update_before_save(tmp_path, mixed):
    model, panel = mixed
    model.save(tmp_path / "run")
    loaded = DummyForecaster._load(tmp_path / "run")
    new = panel._take(slice(47, 48))
    pd.testing.assert_frame_equal(
        loaded.update(new).predict(3, start=-3), copy.deepcopy(model).update(new).predict(3, start=-3),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_verbose_one_shows_progress_and_zero_is_silent(capsys):
    import logging

    class Chatty(DummyForecaster):
        def _estimate(self, panel, log):
            logging.getLogger("dfm_python.test_base").info("estimating %d rows", panel.n_periods)
            return super()._estimate(panel, log)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        Chatty(verbose=1).fit(make_panel(DummyForecaster))
        assert "estimating 48 rows" in capsys.readouterr().out
        Chatty(verbose=0).fit(make_panel(DummyForecaster))
        assert capsys.readouterr().out == ""


def test_fitted_models_pickle(mixed):
    import pickle

    model, _ = mixed
    back = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(back.predict(3, start=-3), model.predict(3, start=-3), check_exact=True)
    pd.testing.assert_frame_equal(back.states_.common, model.states_.common, check_exact=True)
    assert back.__sklearn_is_fitted__() and not DummyForecaster().__sklearn_is_fitted__()

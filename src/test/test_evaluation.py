"""Tests for dfm_python.evaluation: walk-forward backtests and their metrics."""

from __future__ import annotations

import datetime
import functools
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from dfm_python import AFM, DDFM, DFM, PanelData, iVDFM
from dfm_python.base import BaseFactorModel
from dfm_python.evaluation import BacktestResult, backtest
from dfm_python.testing import make_panel
from dfm_python.utils.errors import ConfigurationError, UnsupportedInputError

TINY_DFM = {"max_iter": 5, "tent_weights": {"q:m": [1, 2, 3, 2, 1]}, "scaler": "standard"}
TINY_DDFM = {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 2,
             "random_state": 0, "device": "cpu"}
TINY_IVDFM = {"n_factors": 2, "window": 8, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
              "batch_size": 4, "max_epochs": 2, "random_state": 0, "device": "cpu"}
MODELS = {"DFM": (DFM, TINY_DFM), "DDFM": (DDFM, TINY_DDFM), "iVDFM": (iVDFM, TINY_IVDFM)}

# make_panel: 48 month ends; origins 40, 42, 44, 46 with step 2.
POSITIONS = [40, 42, 44, 46]
NO_FUTR_EXOG = "ignore:iVDFM was fitted with exog"


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The torch models seed the global RNGs; restore them after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _run(estimator, data, **kwargs) -> BacktestResult:
    """``backtest`` from the same global torch RNG state each time (iVDFM's update samples from it)."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return backtest(estimator, data, **kwargs)


def _at(result: BacktestResult, origin) -> pd.DataFrame:
    return result.forecasts[result.forecasts["origin"] == origin]


def _flat(frame: pd.DataFrame) -> np.ndarray:
    """An (h, N) forecast frame in the order of the backtest's rows (horizon, then series)."""
    return frame.to_numpy(dtype=np.float64).reshape(-1)


# -- a tiny estimator that logs what it is fitted and updated on ----------------------

_LOG: list = []


class _LastValue(BaseFactorModel):
    """Forecast the last conditioning row at every step, with intervals of half-width ``width``."""

    _tags = {"capability:missing_values": True, "capability:update": True, "capability:pred_int": True,
             "predict_data": "new_rows"}

    def __init__(self, *, width=1.0):
        self.width = width

    def _condition(self, data: PanelData) -> None:
        self.last_ = data.y.to_numpy()[-1]
        self.conditioning_index_ = data.index

    def _fit(self, data, **fit_kwargs):
        _LOG.append(("fit", data.index[0], data.index[-1]))
        self._condition(data)

    def update(self, data, *, refit=False):
        panel = self._validate_data(data)
        _LOG.append(("update", panel.index[0], panel.index[-1]))
        self._condition(panel)
        return self

    def _point(self, h):
        index = self._forecast_index(self.conditioning_index_, h)
        return pd.DataFrame(np.tile(self.last_, (h, 1)), index=index, columns=list(self.feature_names_in_))

    def predict(self, h, data=None, **predict_kwargs):
        return self._point(h)

    def predict_interval(self, h, coverage=0.9, data=None):
        point = self._point(h)
        columns = pd.MultiIndex.from_product([point.columns, ["lower", "upper"]], names=["series", "bound"])
        values = np.stack([point - self.width, point + self.width], axis=2).reshape(h, -1)
        return pd.DataFrame(values, index=point.index, columns=columns)


class _WindowLastValue(_LastValue):
    _tags = {"predict_data": "window"}


class _NoUpdate(_LastValue):
    _tags = {"capability:update": False, "capability:pred_int": False, "predict_data": None}


class _FutrExog(_LastValue):
    """Logs the ``futr_exog`` that predict and predict_interval get (as iVDFM's, both take it)."""

    _tags = {"capability:exog": True}

    def predict(self, h, data=None, *, futr_exog=None):
        _LOG.append(("predict", futr_exog))
        return super().predict(h)

    def predict_interval(self, h, coverage=0.9, data=None, *, futr_exog=None):
        _LOG.append(("predict_interval", futr_exog))
        return super().predict_interval(h, coverage)


def _panel(T: int = 20, *, seed: int = 0, columns=("a", "b")) -> PanelData:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-31", periods=T, freq="ME", name="date")
    return PanelData(y=pd.DataFrame(rng.standard_normal((T, len(columns))), index=index, columns=list(columns)))


def _perturbed(data: PanelData, after: int, *, exog: bool = True) -> PanelData:
    """``data`` with every value of ``y`` (and of ``exog`` unless False) in the rows after ``after`` changed."""
    y = data.y
    y.iloc[after + 1:] = 7.0 - 3.0 * y.iloc[after + 1:]
    x = data.exog
    if x is not None and exog:
        x.iloc[after + 1:] = x.iloc[after + 1:] + 5.0
    return PanelData(y=y, exog=x, freq=data.freq)


# -- the forecasts frame -------------------------------------------------------------


def test_forecasts_of_a_mixed_frequency_dfm():
    data = make_panel(DFM)  # s0 quarterly (NaN off quarter ends), s1 and s2 monthly, two NaN in s2
    result = backtest(DFM(**TINY_DFM), data, start=40, h=3, step=2, coverage=0.9)
    fc = result.forecasts
    assert list(fc.columns) == ["origin", "time", "horizon", "series", "forecast", "actual", "lower", "upper"]
    assert result.origins.equals(data.index[POSITIONS]) and result.coverage == 0.9
    assert result.fit_seconds.index.equals(result.origins) and (result.fit_seconds > 0).all()
    assert len(fc) == len(POSITIONS) * 3 * 3
    assert fc["horizon"].tolist()[:9] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert fc["series"].tolist()[:3] == ["s0", "s1", "s2"]
    assert fc["origin"].dtype == fc["time"].dtype == data.index.dtype

    # time is the row ``horizon`` steps after the origin; past the end of the data it continues the index.
    clock = pd.date_range(data.index[0], periods=48 + 3, freq="ME")
    rows = data.index.get_indexer(fc["origin"]) + fc["horizon"].to_numpy()
    np.testing.assert_array_equal(fc["time"].to_numpy(), clock[rows].to_numpy())
    lookup = data.y.reindex(clock)
    np.testing.assert_array_equal(fc["actual"], [lookup.at[t, s] for t, s in zip(fc["time"], fc["series"])])
    quarterly = fc[fc["series"] == "s0"]
    assert quarterly.loc[quarterly["time"].dt.month % 3 != 0, "actual"].isna().all()
    assert fc.loc[fc["time"] > data.index[-1], "actual"].isna().all()
    assert np.isfinite(fc[["forecast", "lower", "upper"]].to_numpy()).all()
    assert ((fc["lower"] < fc["forecast"]) & (fc["forecast"] < fc["upper"])).all()

    by_series = result.metrics(by="series")
    assert by_series.index.tolist() == ["s0", "s1", "s2"]
    assert by_series["n"].tolist() == [int(fc.loc[fc["series"] == s, "actual"].notna().sum()) for s in ["s0", "s1", "s2"]]
    assert by_series.loc["s0", "n"] < by_series.loc["s1", "n"]


@pytest.mark.filterwarnings(NO_FUTR_EXOG)
@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize("name", list(MODELS))
def test_the_model_never_sees_the_rows_after_the_origin(name, refit):
    cls, params = MODELS[name]
    data = make_panel(cls)
    kwargs = {"start": 40, "h": 3, "step": 2, "refit": refit}
    base = _run(cls(**params), data, **kwargs)
    # iVDFM gets the exog after the origin as futr_exog (known in advance), so only its y changes.
    changed = _run(cls(**params), _perturbed(data, 42, exog=name != "iVDFM"), **kwargs)
    seen = base.forecasts["origin"] <= data.index[42]
    np.testing.assert_array_equal(changed.forecasts.loc[seen, "forecast"], base.forecasts.loc[seen, "forecast"])
    assert not np.allclose(changed.forecasts.loc[~seen, "forecast"], base.forecasts.loc[~seen, "forecast"])


# -- refit and update ----------------------------------------------------------------


@pytest.mark.parametrize(
    "refit, fitted",
    [(False, [0]), (True, [0, 1, 2, 3, 4]), (1, [0, 1, 2, 3, 4]), (2, [0, 2, 4]), (3, [0, 3]), (np.int64(9), [0])],
)
def test_refit_schedules(refit, fitted):
    data, index = _panel(), _panel().index
    _LOG.clear()
    result = backtest(_LastValue(), data, start=10, h=2, step=2, refit=refit)
    positions = [10, 12, 14, 16, 18]
    assert result.origins.equals(index[positions])
    expected = [
        ("fit", index[0], index[p]) if i in fitted else ("update", index[p - 1], index[p])  # the rows since the last origin
        for i, p in enumerate(positions)
    ]
    assert _LOG == expected
    assert sum(event == "fit" for event, *_ in _LOG) == len(fitted)


def test_window_models_are_updated_with_every_row_up_to_the_origin():
    index = _panel().index
    _LOG.clear()
    backtest(_WindowLastValue(), _panel(), start=10, h=2, step=4)
    assert _LOG == [("fit", index[0], index[10]), ("update", index[0], index[14]), ("update", index[0], index[18])]


def test_a_model_without_update_must_refit_at_every_origin():
    for refit in (False, 2):
        with pytest.raises(ConfigurationError, match="capability:update"):
            backtest(_NoUpdate(), _panel(), start=10, h=2, refit=refit)
    for refit in (True, 1):
        _LOG.clear()
        backtest(_NoUpdate(), _panel(), start=15, h=2, refit=refit)
        assert [event for event, *_ in _LOG] == ["fit"] * 4


@pytest.mark.parametrize("refit", [False, 2])
@pytest.mark.parametrize("name", ["DFM", "DDFM"])
def test_backtest_is_the_manual_fit_and_update_sequence(name, refit):
    cls, params = MODELS[name]
    data, index = make_panel(cls), make_panel(cls).index
    coverage = 0.8 if cls.get_tags()["capability:pred_int"] else None
    result = _run(cls(**params), data, start=40, h=3, step=2, refit=refit, coverage=coverage)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        for i, p in enumerate(POSITIONS):
            if i == 0 or (refit and i % refit == 0):
                model = cls(**params).fit(data.loc(None, index[p]))
            else:
                model.update(data.loc(index[p - 1], index[p]))  # the rows after the previous origin
            rows = _at(result, index[p])
            np.testing.assert_array_equal(rows["forecast"], _flat(model.predict(3)))
            if coverage is not None:
                bands = model.predict_interval(3, coverage)
                np.testing.assert_array_equal(rows["lower"], _flat(bands.xs("lower", axis=1, level="bound")))
                np.testing.assert_array_equal(rows["upper"], _flat(bands.xs("upper", axis=1, level="bound")))


def test_ivdfm_is_updated_with_the_window_and_gets_the_future_exog(monkeypatch):
    data, index = make_panel(iVDFM), make_panel(iVDFM).index
    exog = data.exog.to_numpy()
    passed = []
    predict = iVDFM.predict

    @functools.wraps(predict)
    def spy(self, h, data=None, **kwargs):
        passed.append(kwargs.get("futr_exog"))
        return predict(self, h, data, **kwargs)

    monkeypatch.setattr(iVDFM, "predict", spy)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(iVDFM(**TINY_IVDFM), data, start=40, h=3, step=2)
    monkeypatch.undo()
    # The exog rows after the origin (exog known in advance) wherever the data holds all h of them; at the last
    # origin (row 46) it holds one, so predict runs without futr_exog and warns.
    for p, futr in zip(POSITIONS, passed, strict=True):
        if p + 3 < 48:
            np.testing.assert_array_equal(futr, exog[p + 1:p + 4])
        else:
            assert futr is None
    assert sum("futr_exog" in str(w.message) for w in caught) == 1
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = iVDFM(**TINY_IVDFM).fit(data.loc(None, index[40]))
        for p in POSITIONS:
            if p > 40:
                model.update(data.loc(None, index[p]))  # tag predict_data="window": every row up to the origin
            if p + 3 < 48:
                expected = model.predict(3, futr_exog=exog[p + 1:p + 4])
            else:
                with pytest.warns(UserWarning, match="futr_exog"):
                    expected = model.predict(3)
            np.testing.assert_array_equal(_at(result, index[p])["forecast"], _flat(expected))


def test_predict_and_predict_interval_get_the_same_future_exog():
    y = _panel(T=12).y
    exog = pd.DataFrame(np.arange(24.0).reshape(12, 2), index=y.index, columns=["x0", "x1"])
    exog.iloc[10, 1] = np.nan
    _LOG.clear()
    backtest(_FutrExog(), PanelData(y=y, exog=exog), start=6, h=2, coverage=0.9)
    # Origins 6 to 10: the two rows after origins 8 and 9 include the NaN, and only one row follows origin 10.
    expected = [[[14.0, 15.0], [16.0, 17.0]], [[16.0, 17.0], [18.0, 19.0]], None, None, None]
    for method in ("predict", "predict_interval"):
        passed = [entry[1] for entry in _LOG if entry[0] == method]
        assert [None if futr is None else futr.tolist() for futr in passed] == expected, method


def test_the_estimator_is_cloned_and_not_fitted():
    est = DFM(**TINY_DFM)
    params = est.get_params()
    backtest(est, make_panel(DFM), start=44, h=2)
    assert est.get_params() == params and not hasattr(est, "result_")


# -- inputs --------------------------------------------------------------------------


def test_polars_and_dataframe_input_give_the_panel_result():
    pl = pytest.importorskip("polars")
    df = make_panel(DFM).y[["s1", "s2"]]
    est = DFM(max_iter=5, scaler="standard")
    expected = backtest(est, PanelData(y=df), start=-6, h=2).forecasts
    pd.testing.assert_frame_equal(backtest(est, df, start=-6, h=2).forecasts, expected)
    frame = pl.from_pandas(df.reset_index())  # its Datetime column becomes the index
    pd.testing.assert_frame_equal(backtest(est, frame, start=-6, h=2).forecasts, expected)
    # A Date column gives datetime64[ms] stamps; start can be one of its values (a datetime.date).
    dates = frame.with_columns(pl.col("date").cast(pl.Date))
    fc = backtest(est, dates, start=dates["date"][-6], h=2).forecasts
    assert fc["origin"].dtype == fc["time"].dtype == "datetime64[ms]"
    pd.testing.assert_frame_equal(fc, expected, check_dtype=False)


def test_start_is_a_label_or_a_position():
    data = _panel(T=12)
    for start in (8, np.int64(8), -4, data.index[8], "2020-09-30", datetime.date(2020, 9, 30)):
        assert backtest(_LastValue(), data, start=start, h=1).origins.equals(data.index[[8, 9, 10]]), start
    # The first origin needs two training rows (fit rejects a series observed once).
    assert backtest(_LastValue(), data, start=1, h=1, step=4).origins.equals(data.index[[1, 5, 9]])


def test_h_past_the_end_of_the_data():
    data = _panel(T=12)
    result = backtest(_LastValue(width=0.5), data, start=-3, h=5, coverage=0.5)
    fc = result.forecasts
    assert result.origins.equals(data.index[[9, 10]])
    last = _at(result, data.index[10])
    assert pd.Index(last["time"].unique()).equals(pd.date_range(data.index[10], periods=6, freq="ME")[1:])
    assert last["actual"].isna().tolist() == [False] * 2 + [True] * 8  # only row 11 is in the data
    np.testing.assert_array_equal(last["forecast"], np.tile(data.y.to_numpy()[10], 5))
    m = result.metrics()
    assert m.index.tolist() == [1, 2, 3, 4, 5] and m["n"].tolist() == [4, 2, 0, 0, 0]
    assert m.loc[[3, 4, 5], ["RMSE", "MAE", "bias", "coverage"]].isna().all().all()
    assert len(fc) == 2 * 5 * 2

    ranged = backtest(_LastValue(), pd.DataFrame(np.arange(5.0)[:, None], columns=["a"]), start=3, h=3).forecasts
    assert ranged["time"].tolist() == [4, 5, 6] and ranged["actual"].isna().tolist() == [False, True, True]


# -- metrics -------------------------------------------------------------------------


def test_metrics_match_a_direct_computation():
    y = _panel(T=16, seed=3, columns=("z", "a")).y
    y.iloc[15, 0] = np.nan  # an actual that is not observed (never an origin)
    y.iloc[11, 1] = np.nan
    result = backtest(_LastValue(width=0.8), PanelData(y=y), start=8, h=3, coverage=0.9)
    values = y.to_numpy()
    pairs = []  # (horizon, series, error, inside)
    for p in range(8, 15):
        for k in (1, 2, 3):
            for j, name in enumerate(["z", "a"]):
                if p + k < 16 and np.isfinite(values[p + k, j]) and np.isfinite(values[p, j]):
                    error = values[p, j] - values[p + k, j]
                    pairs.append((k, name, error, abs(error) <= 0.8))

    def expected(keep) -> list:
        e = np.array([pair[2] for pair in pairs if keep(pair)])
        inside = np.array([pair[3] for pair in pairs if keep(pair)])
        return [np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), np.mean(e), len(e), np.mean(inside)]

    columns = ["RMSE", "MAE", "bias", "n", "coverage"]
    m = result.metrics(by="horizon")
    assert list(m.columns) == columns and m.index.tolist() == [1, 2, 3] and m.index.name == "horizon"
    for k in (1, 2, 3):
        np.testing.assert_allclose(m.loc[k].to_numpy(dtype=float), expected(lambda pair: pair[0] == k))
    m = result.metrics(by="series")
    assert m.index.tolist() == ["z", "a"]  # the data's order
    for name in ("z", "a"):
        np.testing.assert_allclose(m.loc[name].to_numpy(dtype=float), expected(lambda pair: pair[1] == name))
    for by in [("horizon", "series"), ("series", "horizon")]:
        m = result.metrics(by=by)
        assert m.index.names == list(by)
        if by[0] == "horizon":
            assert m.index.tolist() == [(k, s) for k in (1, 2, 3) for s in ("z", "a")]
        else:
            assert m.index.tolist() == [(s, k) for s in ("z", "a") for k in (1, 2, 3)]
        for key in m.index:
            k, s = key if by[0] == "horizon" else key[::-1]
            np.testing.assert_allclose(m.loc[key].to_numpy(dtype=float), expected(lambda pair: pair[:2] == (k, s)))
    m = result.metrics(by=None)
    assert m.index.tolist() == ["all"]
    np.testing.assert_allclose(m.loc["all"].to_numpy(dtype=float), expected(lambda pair: True))
    assert m["n"].dtype == np.int64


def test_metrics_without_intervals_have_no_coverage():
    result = backtest(_LastValue(), _panel(T=12), start=8, h=2)
    assert "lower" not in result.forecasts and result.coverage is None
    assert list(result.metrics().columns) == ["RMSE", "MAE", "bias", "n"]


# -- errors --------------------------------------------------------------------------


def test_invalid_arguments():
    data = _panel(T=12)
    est = _LastValue()
    with pytest.raises(TypeError, match="task='forecast'"):
        backtest(AFM(), data, start=8, h=1)
    with pytest.raises(TypeError, match="forecasting estimator"):
        backtest(object(), data, start=8, h=1)
    with pytest.raises(UnsupportedInputError, match="capability:pred_int"):
        backtest(_NoUpdate(), data, start=8, h=1, refit=True, coverage=0.9)
    with pytest.raises(TypeError):
        backtest(est, data, start=8, h=1.0)
    with pytest.raises(ValueError):
        backtest(est, data, start=8, h=0)
    for start in (11, 12, -13, -1, "2030-01-31", "2020-09", pd.Timestamp("2020-02-15"), 1.5):
        with pytest.raises(ConfigurationError, match="start"):
            backtest(est, data, start=start, h=1)
    for step in (0, -1, 1.0, True):
        with pytest.raises(ConfigurationError, match="step"):
            backtest(est, data, start=8, h=1, step=step)
    for refit in (0, -2, 1.5, "yes", None):
        with pytest.raises(ConfigurationError, match="refit"):
            backtest(est, data, start=8, h=1, refit=refit)
    for coverage in (0, 1, 1.5, float("nan"), True, "0.9"):
        with pytest.raises(ConfigurationError, match="coverage"):
            backtest(est, data, start=8, h=1, coverage=coverage)
    result = backtest(est, data, start=8, h=1)
    for by in ("origin", ("horizon", "horizon"), ["horizon"], (), 1):
        with pytest.raises(ConfigurationError, match="by"):
            result.metrics(by=by)

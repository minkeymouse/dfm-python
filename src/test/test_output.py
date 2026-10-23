"""polars frames: ``utils.output.to_polars``, polars input to ``fit``/``predict``/``update`` and polars ``allocate``."""

import numpy as np
import pandas as pd
import pytest

from dfm_python import AFM, DFM
from dfm_python.testing import _assert_bitwise, _result_arrays, make_panel
from dfm_python.utils.output import to_polars

pl = pytest.importorskip("polars")
pytest.importorskip("pyarrow")


def _frame(index):
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, np.nan, 6.0]}, index=index)


# ---------------------------------------------------------------------------
# utils.output.to_polars
# ---------------------------------------------------------------------------


def test_to_polars_named_index_is_first_column():
    index = pd.date_range("2020-01-31", periods=3, freq="ME", name="date")
    out = to_polars(_frame(index))
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["date", "a", "b"]
    assert out.schema["date"] == pl.Datetime("ns")
    np.testing.assert_array_equal(out["date"].to_numpy(), index.to_numpy())
    np.testing.assert_array_equal(out.select("a", "b").to_numpy(), _frame(index).to_numpy())


def test_to_polars_unnamed_index_is_called_time():
    out = to_polars(_frame(pd.RangeIndex(5, 8)))
    assert out.columns == ["time", "a", "b"]
    assert out["time"].to_list() == [5, 6, 7]


def test_to_polars_period_index_becomes_start_timestamps():
    periods = pd.period_range("2001Q1", periods=3, freq="Q", name="quarter")
    out = to_polars(_frame(periods))
    assert out.columns == ["quarter", "a", "b"]
    np.testing.assert_array_equal(out["quarter"].to_numpy(), periods.to_timestamp().to_numpy())


def test_to_polars_joins_multiindex_columns():
    columns = pd.MultiIndex.from_product([["s0", "s1"], ["lower", "upper"]], names=["series", "bound"])
    frame = pd.DataFrame(np.arange(12.0).reshape(3, 4), columns=columns)
    out = to_polars(frame)
    assert out.columns == ["time", "s0__lower", "s0__upper", "s1__lower", "s1__upper"]
    np.testing.assert_array_equal(out.drop("time").to_numpy(), frame.to_numpy())


@pytest.mark.parametrize(("index_name", "column"), [("a", "a"), (None, "time")])
def test_to_polars_rejects_index_name_equal_to_a_column(index_name, column):
    frame = pd.DataFrame({column: [1.0, 2.0]}, index=pd.RangeIndex(2, name=index_name))
    with pytest.raises(ValueError, match="index name"):
        to_polars(frame)


def test_to_polars_series():
    index = pd.date_range("2020-01-31", periods=3, freq="ME", name="date")
    named = to_polars(pd.Series([1.0, 2.0, 3.0], index=index, name="x"))
    assert named.columns == ["date", "x"] and named["x"].to_list() == [1.0, 2.0, 3.0]
    assert to_polars(pd.Series([1.0, 2.0])).columns == ["time", "0"]  # unnamed: the column name as a string


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time_dtype", [pl.Date, pl.Datetime("ns")])
def test_dfm_on_polars_frames_matches_pandas_bitwise(time_dtype):
    frame = make_panel({}).y  # 48 monthly steps, index named "date"
    train, new = frame.iloc[:40], frame.iloc[40:]

    def polars(df):
        return pl.from_pandas(df.reset_index()).with_columns(pl.col("date").cast(time_dtype))

    expected = DFM(max_iter=5).fit(train)
    model = DFM(max_iter=5).fit(polars(train))
    assert model.index_.as_unit("ns").equals(expected.index_)  # a polars Date arrives in milliseconds
    _assert_bitwise(_result_arrays(model), _result_arrays(expected), "fit on a polars frame")
    forecast = expected.predict(3, data=new)
    pd.testing.assert_frame_equal(model.predict(3, data=polars(new)), forecast, check_exact=True)
    pd.testing.assert_frame_equal(expected.predict(3, data=polars(new)), forecast, check_exact=True)
    model.update(polars(new))
    expected.update(new)
    _assert_bitwise(_result_arrays(model), _result_arrays(expected), "update with a polars frame")
    pd.testing.assert_frame_equal(model.predict(3), expected.predict(3), check_exact=True)


def test_afm_set_output_polars_allocate():
    params = AFM.get_test_params()[1]  # PCA factors: no characteristics, so a bare frame is enough
    frame = make_panel(AFM).y
    polars_frame = pl.from_pandas(frame.reset_index()).with_columns(pl.col("date").cast(pl.Date))
    fitted = AFM(**params).fit(frame)
    expected = fitted.allocate()
    model = AFM(**params).fit(polars_frame)
    assert model.allocate(data=polars_frame).to_numpy().tobytes() == expected.to_numpy().tobytes()
    backtest, expected_backtest = model.backtest(polars_frame), fitted.backtest(frame)
    assert backtest.book_returns.tobytes() == expected_backtest.book_returns.tobytes()
    assert backtest.sharpe == expected_backtest.sharpe
    assert model.set_output(output="polars") is model
    out = model.allocate()
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["date", *expected.columns]
    np.testing.assert_array_equal(out["date"].cast(pl.Datetime("ns")).to_numpy(), expected.index.to_numpy())
    assert out.drop("date").to_numpy().tobytes() == expected.to_numpy().tobytes()
    tags = model.get_tags()
    assert tags["task"] == "allocation" and not tags["capability:interpret"] and not tags["capability:pred_int"]

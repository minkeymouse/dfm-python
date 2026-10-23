"""Tests for dfm_python.dataset.panel.PanelData."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import dfm_python
from dfm_python import PanelData
from dfm_python.config.constants import VALID_FREQUENCIES
from dfm_python.dataset.panel import _normalize_freq
from dfm_python.utils.errors import ConfigurationError, DataValidationError, UnsupportedInputError

T, N, M, K = 8, 3, 2, 2
SERIES = ["gdp", "cpi", "ip"]
FEATURES = ["size", "value"]


def _index():
    return pd.date_range("2020-01-31", periods=T, freq="ME", name="date")


def _y(index=None):
    values = np.arange(T * N, dtype=float).reshape(T, N)
    values[2, 1] = np.nan
    return pd.DataFrame(values, index=_index() if index is None else index, columns=SERIES)


def _exog(index=None):
    values = 100.0 + np.arange(T * K, dtype=float).reshape(T, K)
    values[5, 0] = np.nan
    return pd.DataFrame(values, index=_index() if index is None else index, columns=["oil", "fx"])


def _sexog_array():
    arr = 1000.0 + np.arange(T * N * M, dtype=float).reshape(T, N, M)
    arr[3, 2, 1] = np.nan
    return arr


def _sexog(index=None):
    arr = _sexog_array()
    columns = pd.MultiIndex.from_product([FEATURES, SERIES], names=["feature", "series"])
    flat = np.column_stack([arr[:, n, m] for m in range(M) for n in range(N)])
    return pd.DataFrame(flat, index=_index() if index is None else index, columns=columns)


def _full():
    return PanelData(_y(), exog=_exog(), series_exog=_sexog(), freq={"gdp": "q"})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_exported_at_top_level():
    assert dfm_python.PanelData is PanelData
    assert "PanelData" in dfm_python.__all__
    from dfm_python.dataset import PanelData as from_subpackage

    assert from_subpackage is PanelData


def test_constructor_slots_and_accessors():
    data = _full()
    assert data.series_names == tuple(SERIES)
    assert data.exog_names == ("oil", "fx")
    assert data.series_exog_names == tuple(FEATURES)
    assert data.n_timesteps == T
    assert data.index.equals(_index())
    assert data.index.name == "date"
    pd.testing.assert_frame_equal(data.y, _y())
    pd.testing.assert_frame_equal(data.exog, _exog())
    pd.testing.assert_frame_equal(data.series_exog, _sexog())
    assert data.freq == {"gdp": "q"}
    assert data.is_mixed_frequency


def test_constructor_minimal():
    data = PanelData(_y())
    assert data.exog is None and data.series_exog is None and data.freq is None
    assert data.exog_names == () and data.series_exog_names == ()
    assert data.series_exog_array() is None
    assert not data.is_mixed_frequency


def test_values_cast_to_float64_and_nan_kept():
    y = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": pd.array([1, None, 3], dtype="Int64"),
            "c": [True, False, True],
            "d": np.array([1.5, np.nan, 2.5], dtype=np.float32),
        }
    )
    data = PanelData(y)
    assert (data.y.dtypes == np.float64).all()
    assert np.isnan(data.y.loc[1, "b"]) and np.isnan(data.y.loc[1, "d"])
    assert data.y["c"].tolist() == [1.0, 0.0, 1.0]


def test_non_numeric_columns_raise():
    y = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(DataValidationError, match="non-numeric"):
        PanelData(y)


def test_column_order_preserved():
    y = _y()[["ip", "gdp", "cpi"]]
    assert PanelData(y).series_names == ("ip", "gdp", "cpi")


def test_wrong_slot_type_raises_type_error():
    with pytest.raises(TypeError):
        PanelData(_y()["gdp"])
    with pytest.raises(TypeError):
        PanelData(_y(), exog=np.ones((T, 1)))


def test_empty_exog_and_series_exog_become_none():
    empty = pd.DataFrame(index=_index())
    data = PanelData(_y(), exog=empty, series_exog=empty)
    assert data.exog is None and data.series_exog is None


def test_from_wide_with_time_col():
    raw = _y().join(_exog()).reset_index()
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")  # strings, as read_csv gives them
    data = PanelData.from_wide(raw, time_col="date", exog_cols=["oil", "fx"], freq={"q": ["gdp"]})
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.name == "date"
    assert data.series_names == tuple(SERIES)
    assert data.exog_names == ("oil", "fx")
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())
    assert data.freq == {"gdp": "q"}


def test_from_wide_uses_index_and_selects_targets():
    raw = _y().join(_exog())
    data = PanelData.from_wide(raw, target_cols=["cpi", "gdp"])
    assert data.series_names == ("cpi", "gdp")
    assert data.exog is None
    assert data.index.equals(_index())


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_from_wide_errors():
    raw = _y().join(_exog())
    with pytest.raises(DataValidationError, match="not found"):
        PanelData.from_wide(raw, exog_cols=["missing"])
    with pytest.raises(DataValidationError, match="both"):
        PanelData.from_wide(raw, target_cols=["gdp", "oil"], exog_cols=["oil"])
    with pytest.raises(DataValidationError, match="no target"):
        PanelData.from_wide(raw[["oil", "fx"]], exog_cols=["oil", "fx"])
    with pytest.raises(DataValidationError, match="parse"):
        PanelData.from_wide(raw.reset_index().assign(date="not a date"), time_col="date")


def test_from_arrays_defaults():
    y = np.arange(T * N, dtype=float).reshape(T, N)
    data = PanelData.from_arrays(y, exog=np.ones((T, K)), series_exog=_sexog_array())
    assert isinstance(data.index, pd.RangeIndex) and data.index.equals(pd.RangeIndex(T))
    assert data.series_names == ("series_0", "series_1", "series_2")
    assert data.exog_names == ("exog_0", "exog_1")
    assert data.series_exog_names == ("feature_0", "feature_1")
    np.testing.assert_array_equal(data.y.to_numpy(), y)
    np.testing.assert_array_equal(data.series_exog_array(), _sexog_array())


def test_from_arrays_names_and_index():
    data = PanelData.from_arrays(
        _y().to_numpy(),
        series_exog=_sexog_array(),
        index=_index(),
        names=SERIES,
        series_exog_names=FEATURES,
        exog=_exog().to_numpy(),
        exog_names=["oil", "fx"],
        freq={"gdp": "q"},
    )
    assert data.equals(_full())
    pd.testing.assert_frame_equal(data.series_exog, _sexog())


def test_from_arrays_shape_errors():
    y = np.zeros((T, N))
    with pytest.raises(DataValidationError, match=r"\(T, N\)"):
        PanelData.from_arrays(np.zeros(T))
    with pytest.raises(DataValidationError, match="series_exog"):
        PanelData.from_arrays(y, series_exog=np.zeros((T, N + 1, M)))
    with pytest.raises(DataValidationError, match="exog"):
        PanelData.from_arrays(y, exog=np.zeros((T - 1, K)))
    with pytest.raises(DataValidationError, match="names"):
        PanelData.from_arrays(y, names=["a", "b"])
    with pytest.raises(DataValidationError, match="index"):
        PanelData.from_arrays(y, index=pd.RangeIndex(T + 1))


def test_from_long_basic_and_missing_rows():
    long = _full().to_long()
    long = long.drop(index=[0, 5]).sample(frac=1.0, random_state=0)  # drop two rows, shuffle the rest
    data = PanelData.from_long(long, series_exog_cols=FEATURES, exog_cols=["oil", "fx"])
    assert set(data.series_names) == set(SERIES)  # first-appearance order of the shuffled rows
    assert data.index.equals(_index())
    expected = _y()
    expected.iloc[0, 0] = np.nan  # gdp at t=0 was dropped
    expected.iloc[5, 0] = np.nan  # gdp at t=5 was dropped
    np.testing.assert_array_equal(data.y[SERIES].to_numpy(), expected.to_numpy())


def test_from_long_series_follow_first_appearance():
    long = _full().to_long()
    order = ["ip", "gdp", "cpi"]
    long = pd.concat([long[long["unique_id"] == s] for s in order])
    data = PanelData.from_long(long)
    assert data.series_names == tuple(order)


def test_from_long_exog_must_be_constant_across_ids():
    long = _full().to_long()
    long.loc[(long["unique_id"] == "cpi") & (long["ds"] == _index()[3]), "oil"] = -1.0
    with pytest.raises(DataValidationError, match="oil"):
        PanelData.from_long(long, exog_cols=["oil", "fx"])


def test_from_long_exog_nan_in_one_id_is_missing_not_a_conflict():
    long = _full().to_long()
    at = (long["unique_id"] == "cpi") & (long["ds"] == _index()[3])
    long.loc[at, "oil"] = np.nan
    data = PanelData.from_long(long, exog_cols=["oil", "fx"])
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())


def test_from_long_errors():
    long = _full().to_long()
    with pytest.raises(DataValidationError, match="not found"):
        PanelData.from_long(long, exog_cols=["missing"])
    with pytest.raises(DataValidationError, match="duplicated"):
        PanelData.from_long(pd.concat([long, long.iloc[:1]]))
    with pytest.raises(DataValidationError, match="one role"):
        PanelData.from_long(long, exog_cols=["oil"], series_exog_cols=["oil"])


def test_from_long_custom_column_names():
    long = _full().to_long().rename(columns={"unique_id": "ticker", "ds": "when", "y": "ret"})
    data = PanelData.from_long(long, id_col="ticker", time_col="when", target_col="ret")
    assert data.series_names == tuple(SERIES)
    assert data.index.name == "when"


# ---------------------------------------------------------------------------
# polars input
# ---------------------------------------------------------------------------


@pytest.fixture
def pl():
    polars = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    return polars


def test_polars_constructor(pl):
    data = PanelData(pl.from_pandas(_y().reset_index(drop=True)))
    assert isinstance(data.index, pd.RangeIndex)
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())


def test_polars_from_wide(pl):
    frame = pl.from_pandas(_y().join(_exog()).reset_index())
    data = PanelData.from_wide(frame, time_col="date", exog_cols=["oil", "fx"])
    assert isinstance(data.index, pd.DatetimeIndex)
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())


def test_polars_from_long(pl):
    long = _full().to_long()
    data = PanelData.from_long(pl.from_pandas(long), series_exog_cols=FEATURES, exog_cols=["oil", "fx"])
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data.series_exog_array(), _sexog_array())


def test_polars_from_arrays_and_coerce(pl):
    frame = pl.from_pandas(_y().reset_index(drop=True))
    data = PanelData.from_arrays(frame, names=SERIES)
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    coerced = PanelData._coerce(frame)
    assert coerced.series_names == tuple(SERIES)


def test_polars_coerce_uses_date_column_as_index(pl):
    frame = pl.from_pandas(_y().reset_index()).with_columns(pl.col("date").cast(pl.Date))
    data = PanelData._coerce(frame)
    assert isinstance(data.index, pd.DatetimeIndex) and data.index.name == "date"
    assert data.index.as_unit("ns").equals(_index())
    assert data.series_names == tuple(SERIES)
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())


def test_polars_coerce_keeps_datetime_time_zone(pl):
    frame = pl.from_pandas(_y().reset_index()).with_columns(pl.col("date").dt.replace_time_zone("America/New_York"))
    data = PanelData._coerce(frame)
    assert str(data.index.tz) == "America/New_York"
    assert data.index.equals(_index().tz_localize("America/New_York"))
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())


def test_polars_coerce_without_temporal_column_uses_range_index(pl):
    data = PanelData._coerce(pl.from_pandas(_y().reset_index(drop=True)))
    assert isinstance(data.index, pd.RangeIndex) and data.index.equals(pd.RangeIndex(T))
    assert data.series_names == tuple(SERIES)


def test_polars_coerce_rejects_two_temporal_columns(pl):
    frame = pl.from_pandas(_y().reset_index()).with_columns(pl.col("date").cast(pl.Date).alias("released"))
    with pytest.raises(DataValidationError, match=r"\['date', 'released'\].*from_wide\(df, time_col=\.\.\.\)"):
        PanelData._coerce(frame)


# ---------------------------------------------------------------------------
# polars output
# ---------------------------------------------------------------------------


def test_to_polars_long_is_to_long(pl):
    data = _full()
    out = data.to_polars()
    assert isinstance(out, pl.DataFrame)
    pd.testing.assert_frame_equal(out.to_pandas(), data.to_long())
    back = PanelData.from_long(out, series_exog_cols=FEATURES, exog_cols=["oil", "fx"], freq=data.freq)
    assert back.equals(data)


def test_to_polars_wide_layout_and_round_trip(pl):
    data = PanelData(_y(), exog=_exog(), freq={"gdp": "q"})
    wide = data.to_polars(layout="wide")
    assert wide.columns == ["date", *SERIES, "oil", "fx"]
    np.testing.assert_array_equal(wide.select(SERIES).to_numpy(), _y().to_numpy())
    back = PanelData.from_wide(wide, time_col="date", exog_cols=["oil", "fx"], freq=data.freq)
    assert back.equals(data)
    y_only = PanelData(_y())
    assert PanelData._coerce(y_only.to_polars(layout="wide")).equals(y_only)


def test_to_polars_range_and_period_index(pl):
    ranged = PanelData(_y(index=pd.RangeIndex(10, 10 + 3 * T, 3)))
    assert ranged.to_polars(layout="wide").columns[0] == "time"  # unnamed index
    assert PanelData.from_wide(ranged.to_polars(layout="wide"), time_col="time").equals(ranged)
    assert PanelData.from_long(ranged.to_polars()).equals(ranged)
    periods = pd.period_range("2001Q1", periods=T, freq="Q")
    quarterly = PanelData(_y(index=periods))
    # a PeriodIndex is written as its start timestamps
    for back in (
        PanelData.from_long(quarterly.to_polars()),
        PanelData.from_wide(quarterly.to_polars(layout="wide"), time_col="time"),
    ):
        assert back.index.equals(periods.to_timestamp())
        np.testing.assert_array_equal(back.y.to_numpy(), quarterly.y.to_numpy())


def test_to_polars_errors(pl):
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        _full().to_polars(layout="wide")
    with pytest.raises(ConfigurationError, match="layout"):
        _full().to_polars(layout="tall")


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_long_round_trip_full():
    data = _full()
    back = PanelData.from_long(data.to_long(), series_exog_cols=FEATURES, exog_cols=["oil", "fx"], freq=data.freq)
    assert back.equals(data)
    # the long format carries no index freq attribute; values, labels and index type survive
    pd.testing.assert_frame_equal(back.y, data.y, check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(back.exog, data.exog, check_names=False, check_freq=False)


def test_long_round_trip_range_and_period_index():
    for index in (pd.RangeIndex(10, 10 + 3 * T, 3), pd.period_range("2001Q1", periods=T, freq="Q")):
        data = PanelData(_y(index=index), exog=_exog(index=index))
        back = PanelData.from_long(data.to_long(), exog_cols=["oil", "fx"])
        assert type(back.index) is type(index)
        assert back.equals(data)


def test_to_long_layout():
    long = _full().to_long()
    assert list(long.columns) == ["unique_id", "ds", "y", *FEATURES, "oil", "fx"]
    assert len(long) == T * N
    assert long["unique_id"].tolist() == [s for s in SERIES for _ in range(T)]
    assert long["y"].isna().sum() == 1  # NaN targets are kept


def test_to_long_rejects_reserved_names():
    data = PanelData(_y(), exog=_exog().rename(columns={"oil": "y"}))
    with pytest.raises(DataValidationError, match="clash"):
        data.to_long()


def test_series_exog_array_shape_and_order():
    data = _full()
    arr = data.series_exog_array()
    assert arr.shape == (T, N, M) and arr.dtype == np.float64
    np.testing.assert_array_equal(arr, _sexog_array())
    # column order of the stored frame does not matter
    shuffled = _sexog().iloc[:, [5, 0, 3, 1, 4, 2]]
    other = PanelData(_y(), series_exog=shuffled)
    assert other.series_exog_names == ("value", "size")
    expected = _sexog_array()[:, :, ::-1]
    np.testing.assert_array_equal(other.series_exog_array(), expected)


def test_series_exog_column_validation():
    good = _sexog()
    with pytest.raises(DataValidationError, match="MultiIndex"):
        PanelData(_y(), series_exog=_exog())
    with pytest.raises(DataValidationError, match="named"):
        PanelData(_y(), series_exog=good.rename_axis(columns=["series", "feature"]))
    with pytest.raises(DataValidationError, match="series"):
        PanelData(_y(), series_exog=good.drop(columns="ip", level="series"))
    with pytest.raises(DataValidationError, match="every feature"):
        PanelData(_y(), series_exog=good.drop(columns=[("value", "ip")]))
    unnamed = good.rename_axis(columns=[None, None])
    assert PanelData(_y(), series_exog=unnamed).series_exog.columns.names == ["feature", "series"]


# ---------------------------------------------------------------------------
# freq
# ---------------------------------------------------------------------------


def test_freq_forms_normalize_to_per_series():
    per_series = {"ip": "m", "gdp": "q"}
    grouped = {"q": ["gdp"], "m": ("ip",)}
    expected = {"gdp": "q", "ip": "m"}  # in series order
    assert _normalize_freq(per_series, SERIES) == expected
    assert _normalize_freq(grouped, SERIES) == expected
    assert list(_normalize_freq(grouped, SERIES)) == ["gdp", "ip"]
    assert PanelData(_y(), freq=per_series).freq == PanelData(_y(), freq=grouped).freq == expected


def test_freq_none_and_empty():
    assert _normalize_freq(None, SERIES) is None
    assert _normalize_freq({}, SERIES) is None


def test_freq_errors():
    with pytest.raises(DataValidationError, match="unknown series"):
        PanelData(_y(), freq={"nope": "q"})
    with pytest.raises(DataValidationError, match="unknown series"):
        PanelData(_y(), freq={"q": ["nope"]})
    with pytest.raises(DataValidationError, match="frequency code"):
        PanelData(_y(), freq={"gdp": "quarterly"})
    with pytest.raises(DataValidationError, match="frequency code"):
        PanelData(_y(), freq={"Q": ["gdp"]})
    with pytest.raises(DataValidationError, match="two frequencies"):
        PanelData(_y(), freq={"q": ["gdp"], "m": ["gdp"]})
    with pytest.raises(DataValidationError, match="mixes"):
        PanelData(_y(), freq={"gdp": "q", "m": ["ip"]})
    with pytest.raises(TypeError):
        PanelData(_y(), freq=["q"])


def test_freq_codes_come_from_constants():
    for code in VALID_FREQUENCIES:
        assert PanelData(_y(), freq={"gdp": code}).freq == {"gdp": code}


@pytest.mark.parametrize(
    "freq, expected",
    [
        (None, False),
        ({"gdp": "m", "cpi": "m", "ip": "m"}, False),
        ({"q": SERIES}, False),
        ({"gdp": "q", "cpi": "m", "ip": "m"}, True),
        ({"gdp": "q"}, True),
        ({"m": ["gdp"]}, True),
    ],
)
def test_is_mixed_frequency(freq, expected):
    assert PanelData(_y(), freq=freq).is_mixed_frequency is expected


# ---------------------------------------------------------------------------
# Index validation
# ---------------------------------------------------------------------------


def test_index_must_be_unique():
    index = _index()
    with pytest.raises(DataValidationError, match="duplicate"):
        PanelData(_y(index=index[[0, 1, 1, 2, 3, 4, 5, 6]]))


def test_index_must_be_increasing():
    with pytest.raises(DataValidationError, match="increasing"):
        PanelData(_y(index=_index()[::-1]))


def test_index_type():
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        PanelData(_y(index=pd.Index([f"t{i}" for i in range(T)])))
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        PanelData(_y(index=pd.Index(np.arange(T) + 0.5)))
    with pytest.raises(DataValidationError, match="MultiIndex"):
        PanelData(_y(index=pd.MultiIndex.from_arrays([range(T), range(T)])))


def test_index_missing_and_empty():
    index = _index().insert(T, pd.NaT)[1:]
    with pytest.raises(DataValidationError, match="missing"):
        PanelData(_y(index=index))
    with pytest.raises(DataValidationError, match="no rows"):
        PanelData(_y().iloc[:0])
    with pytest.raises(DataValidationError, match="no columns"):
        PanelData(pd.DataFrame(index=_index()))


def test_integer_index_becomes_range_index():
    data = PanelData(_y(index=pd.Index(np.arange(5, 5 + 2 * T, 2), name="t")))
    assert isinstance(data.index, pd.RangeIndex)
    assert data.index.equals(pd.RangeIndex(5, 5 + 2 * T, 2)) and data.index.name == "t"
    with pytest.raises(DataValidationError, match="evenly spaced"):
        PanelData(_y(index=pd.Index([0, 1, 2, 4, 5, 6, 7, 8])))


def test_period_index_accepted():
    index = pd.period_range("2000-01", periods=T, freq="M")
    assert isinstance(PanelData(_y(index=index)).index, pd.PeriodIndex)


def test_slots_must_share_index():
    with pytest.raises(DataValidationError, match="share the index"):
        PanelData(_y(), exog=_exog(index=_index().shift(1)))
    with pytest.raises(DataValidationError, match="share the index"):
        PanelData(_y(), exog=_exog().iloc[:-1])
    with pytest.raises(DataValidationError, match="share the index"):
        PanelData(_y(), series_exog=_sexog(index=pd.RangeIndex(T)))


def test_datetime_resolution_difference_is_not_a_mismatch():
    data = PanelData(_y(), exog=_exog(index=_index().as_unit("us")))
    assert data.exog.index.equals(data.index)


def test_column_names_unique():
    y = _y()
    y.columns = ["gdp", "gdp", "ip"]
    with pytest.raises(DataValidationError, match="unique"):
        PanelData(y)
    with pytest.raises(DataValidationError, match="both"):
        PanelData(_y(), exog=_exog().rename(columns={"oil": "gdp"}))
    with pytest.raises(DataValidationError, match="both"):
        PanelData(_y(), exog=_exog().rename(columns={"oil": "size"}), series_exog=_sexog())


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_attributes_cannot_be_set_or_deleted():
    data = _full()
    for name in ("y", "freq", "_y", "anything"):
        with pytest.raises(AttributeError):
            setattr(data, name, None)
    with pytest.raises(AttributeError):
        del data._y
    assert not hasattr(data, "__dict__")


def test_accessors_return_copies():
    data = _full()
    for frame in (data.y, data.exog, data.series_exog):
        frame.iloc[0, 0] = -999.0
    data.freq["cpi"] = "q"
    data.index.name = "changed"
    data.series_exog_array()[0, 0, 0] = -999.0
    assert data.equals(_full())
    assert data.index.name == "date"
    assert data.freq == {"gdp": "q"}


def test_inputs_are_not_mutated_and_are_copied():
    y, exog, sexog = _y(), _exog(), _sexog()
    freq = {"q": ["gdp"]}
    data = PanelData(y, exog=exog, series_exog=sexog, freq=freq)
    pd.testing.assert_frame_equal(y, _y())
    pd.testing.assert_frame_equal(exog, _exog())
    pd.testing.assert_frame_equal(sexog, _sexog())
    assert freq == {"q": ["gdp"]}
    y.iloc[0, 0] = -1.0
    exog.iloc[0, 0] = -1.0
    sexog.iloc[0, 0] = -1.0
    freq["q"].append("ip")
    assert data.equals(_full())

    arr = _y().to_numpy(copy=True)  # writable under pandas copy-on-write too
    from_arr = PanelData.from_arrays(arr)
    arr[0, 0] = -1.0
    assert from_arr.y.iloc[0, 0] == 0.0

    raw = _y().join(_exog())
    before = raw.copy()
    PanelData.from_wide(raw, exog_cols=["oil", "fx"])
    pd.testing.assert_frame_equal(raw, before)

    long = _full().to_long()
    before = long.copy()
    PanelData.from_long(long, series_exog_cols=FEATURES, exog_cols=["oil", "fx"])
    pd.testing.assert_frame_equal(long, before)


def test_pickle_and_copy():
    data = _full()
    for clone in (pickle.loads(pickle.dumps(data)), copy.copy(data), copy.deepcopy(data)):
        assert isinstance(clone, PanelData)
        assert clone.equals(data)
        with pytest.raises(AttributeError):
            clone._y = None


# ---------------------------------------------------------------------------
# loc
# ---------------------------------------------------------------------------


def test_loc_datetime_labels():
    data = _full()
    index = _index()
    part = data.loc(index[2], index[5])
    assert isinstance(part, PanelData) and part is not data
    assert part.index.equals(index[2:6])  # both ends included
    np.testing.assert_array_equal(part.y.to_numpy(), _y().to_numpy()[2:6])
    np.testing.assert_array_equal(part.exog.to_numpy(), _exog().to_numpy()[2:6])
    np.testing.assert_array_equal(part.series_exog_array(), _sexog_array()[2:6])
    assert part.freq == data.freq
    assert data.n_timesteps == T  # original untouched


def test_loc_open_ends_and_strings():
    data = _full()
    assert data.loc().equals(data)
    assert data.loc(stop="2020-03-31").n_timesteps == 3
    assert data.loc(start="2020-07").n_timesteps == 2


def test_loc_range_index_and_empty():
    data = PanelData(_y(index=pd.RangeIndex(T)))
    part = data.loc(3, 5)
    assert isinstance(part.index, pd.RangeIndex) and part.index.equals(pd.RangeIndex(3, 6))
    with pytest.raises(DataValidationError, match="no rows"):
        data.loc(100, 200)


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------


def test_coerce_panel_is_returned_as_is():
    data = _full()
    assert PanelData._coerce(data) is data


def test_coerce_dataframe_means_y():
    y = _y()
    data = PanelData._coerce(y)
    assert data.equals(PanelData(y))
    assert data.exog is None


def test_coerce_2d_array_means_from_arrays():
    arr = np.arange(12.0).reshape(4, 3)
    data = PanelData._coerce(arr)
    assert data.equals(PanelData.from_arrays(arr))
    assert isinstance(data.index, pd.RangeIndex)
    with pytest.raises(DataValidationError, match="2-D"):
        PanelData._coerce(np.arange(4.0))


@pytest.mark.parametrize("obj", [None, [[1.0, 2.0]], {"a": [1.0]}, pd.Series([1.0, 2.0]), 3.0])
def test_coerce_other_types_raise_type_error(obj):
    with pytest.raises(TypeError):
        PanelData._coerce(obj)


def test_repr():
    text = repr(_full())
    assert text.startswith("PanelData(") and "n_series=3" in text and "mixed_frequency=True" in text

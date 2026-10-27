"""Tests for dfm_python.dataset.panel.PanelData: slots, constructors, between and the merge rule."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import dfm_python
from dfm_python import PanelData
from dfm_python.config.constants import VALID_FREQUENCIES
from dfm_python.dataset.panel import _normalize_freq
from dfm_python.errors import DataValidationError

T, N, M, K = 8, 3, 2, 2
SERIES = ["gdp", "cpi", "ip"]
CHARS = ["size", "value"]


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


def _chars_array():
    arr = 1000.0 + np.arange(T * N * M, dtype=float).reshape(T, N, M)
    arr[3, 2, 1] = np.nan
    return arr


def _chars(index=None):
    arr = _chars_array()
    columns = pd.MultiIndex.from_product([CHARS, SERIES], names=["characteristic", "series"])
    flat = np.column_stack([arr[:, n, m] for m in range(M) for n in range(N)])
    return pd.DataFrame(flat, index=_index() if index is None else index, columns=columns)


def _full():
    return PanelData(_y(), exog=_exog(), characteristics=_chars(), freq={"gdp": "q"})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_exported_at_top_level():
    assert dfm_python.PanelData is PanelData
    assert "PanelData" in dfm_python.__all__
    from dfm_python.dataset import PanelData as from_subpackage

    assert from_subpackage is PanelData


def test_public_members():
    public = sorted(name for name in dir(PanelData) if not name.startswith("_"))
    assert public == sorted([
        "between", "characteristic_names", "characteristics", "exog", "exog_names", "freq", "from_arrays",
        "from_long", "from_wide", "index", "n_periods", "n_series", "series_names", "to_long", "y",
    ])


def test_constructor_slots_and_accessors():
    data = _full()
    assert data.series_names == tuple(SERIES)
    assert data.exog_names == ("oil", "fx")
    assert data.characteristic_names == tuple(CHARS)
    assert data.n_periods == T and data.n_series == N
    assert data.index.equals(_index())
    assert data.index.name == "date"
    pd.testing.assert_frame_equal(data.y, _y())
    pd.testing.assert_frame_equal(data.exog, _exog())
    pd.testing.assert_frame_equal(data.characteristics, _chars())
    assert data.freq == {"gdp": "q"}


def test_constructor_is_keyword_only_after_y():
    with pytest.raises(TypeError):
        PanelData(_y(), _exog())


def test_constructor_minimal():
    data = PanelData(_y())
    assert data.exog is None and data.characteristics is None and data.freq is None
    assert data.exog_names == () and data.characteristic_names == ()
    assert data._characteristic_values() is None
    assert not data._is_mixed_frequency()


def test_characteristics_are_stored_as_a_t_n_m_array():
    data = _full()
    assert data._chars.shape == (T, N, M) and data._chars.dtype == np.float64
    np.testing.assert_array_equal(data._characteristic_values(), _chars_array())
    # the column order of the input frame does not matter; names follow first appearance
    shuffled = _chars().iloc[:, [5, 0, 3, 1, 4, 2]]
    other = PanelData(_y(), characteristics=shuffled)
    assert other.characteristic_names == ("value", "size")
    np.testing.assert_array_equal(other._characteristic_values(), _chars_array()[:, :, ::-1])
    # the property is built on each access, in (characteristic, series) order
    frame = other.characteristics
    assert frame is not other.characteristics
    assert list(frame.columns) == [(c, s) for c in ("value", "size") for s in SERIES]


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
    with pytest.raises(TypeError):
        PanelData(_y(), characteristics=_chars_array())


def test_empty_exog_and_characteristics_become_none():
    empty = pd.DataFrame(index=_index())
    data = PanelData(_y(), exog=empty, characteristics=empty)
    assert data.exog is None and data.characteristics is None


def test_from_wide_with_time_col():
    raw = _y().join(_exog()).reset_index()
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")  # strings, as read_csv gives them
    data = PanelData.from_wide(raw, time_col="date", exog=["oil", "fx"], freq={"gdp": "q"})
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.name == "date"
    assert data.series_names == tuple(SERIES)
    assert data.exog_names == ("oil", "fx")
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())
    assert data.freq == {"gdp": "q"}


def test_from_wide_uses_index_and_selects_targets():
    raw = _y().join(_exog())
    data = PanelData.from_wide(raw, y=["cpi", "gdp"])
    assert data.series_names == ("cpi", "gdp")
    assert data.exog is None
    assert data.index.equals(_index())


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_from_wide_errors():
    raw = _y().join(_exog())
    with pytest.raises(DataValidationError, match="not found"):
        PanelData.from_wide(raw, exog=["missing"])
    with pytest.raises(DataValidationError, match="both"):
        PanelData.from_wide(raw, y=["gdp", "oil"], exog=["oil"])
    with pytest.raises(DataValidationError, match="no target"):
        PanelData.from_wide(raw[["oil", "fx"]], exog=["oil", "fx"])
    with pytest.raises(DataValidationError, match="parse"):
        PanelData.from_wide(raw.reset_index().assign(date="not a date"), time_col="date")
    with pytest.raises(TypeError, match="list of column names"):
        PanelData.from_wide(raw, y="gdp")


def test_from_arrays_defaults():
    y = np.arange(T * N, dtype=float).reshape(T, N)
    data = PanelData.from_arrays(y, exog=np.ones((T, K)), characteristics=_chars_array())
    assert isinstance(data.index, pd.RangeIndex) and data.index.equals(pd.RangeIndex(T))
    assert data.series_names == ("series_0", "series_1", "series_2")
    assert data.exog_names == ("exog_0", "exog_1")
    assert data.characteristic_names == ("characteristic_0", "characteristic_1")
    np.testing.assert_array_equal(data.y.to_numpy(), y)
    np.testing.assert_array_equal(data._characteristic_values(), _chars_array())


def test_from_arrays_names_and_index():
    data = PanelData.from_arrays(
        _y().to_numpy(),
        characteristics=_chars_array(),
        index=_index(),
        series_names=SERIES,
        characteristic_names=CHARS,
        exog=_exog().to_numpy(),
        exog_names=["oil", "fx"],
        freq={"gdp": "q"},
    )
    assert data._equals(_full())
    pd.testing.assert_frame_equal(data.characteristics, _chars())


def test_from_arrays_shape_errors():
    y = np.zeros((T, N))
    with pytest.raises(DataValidationError, match=r"\(T, N\)"):
        PanelData.from_arrays(np.zeros(T))
    with pytest.raises(DataValidationError, match="characteristics"):
        PanelData.from_arrays(y, characteristics=np.zeros((T, N + 1, M)))
    with pytest.raises(DataValidationError, match="exog"):
        PanelData.from_arrays(y, exog=np.zeros((T - 1, K)))
    with pytest.raises(DataValidationError, match="series_names"):
        PanelData.from_arrays(y, series_names=["a", "b"])
    with pytest.raises(DataValidationError, match="unique"):
        PanelData.from_arrays(y, series_names=["a", "b", "a"])
    with pytest.raises(DataValidationError, match="index"):
        PanelData.from_arrays(y, index=pd.RangeIndex(T + 1))


def test_from_arrays_copies_its_inputs():
    arr = _y().to_numpy(copy=True)
    chars = _chars_array()
    data = PanelData.from_arrays(arr, characteristics=chars)
    arr[0, 0] = -1.0
    chars[0, 0, 0] = -1.0
    assert data.y.iloc[0, 0] == 0.0 and data._characteristic_values()[0, 0, 0] == 1000.0


def test_from_long_basic_and_missing_rows():
    long = _full().to_long()
    long = long.drop(index=[0, 5]).sample(frac=1.0, random_state=0)  # drop two rows, shuffle the rest
    data = PanelData.from_long(long, characteristics=CHARS, exog=["oil", "fx"])
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
    with pytest.raises(DataValidationError, match="oil.*belongs in characteristics"):
        PanelData.from_long(long, exog=["oil", "fx"])


def test_from_long_exog_nan_in_one_id_is_missing_not_a_conflict():
    long = _full().to_long()
    at = (long["unique_id"] == "cpi") & (long["ds"] == _index()[3])
    long.loc[at, "oil"] = np.nan
    data = PanelData.from_long(long, exog=["oil", "fx"])
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())


def test_from_long_errors():
    long = _full().to_long()
    with pytest.raises(DataValidationError, match="not found"):
        PanelData.from_long(long, exog=["missing"])
    with pytest.raises(DataValidationError, match="duplicated"):
        PanelData.from_long(pd.concat([long, long.iloc[:1]]))
    with pytest.raises(DataValidationError, match="one role"):
        PanelData.from_long(long, exog=["oil"], characteristics=["oil"])


def test_from_long_custom_column_names():
    long = _full().to_long().rename(columns={"unique_id": "ticker", "ds": "when", "y": "ret"})
    data = PanelData.from_long(long, id_col="ticker", time_col="when", y="ret")
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
    data = PanelData.from_wide(frame, time_col="date", exog=["oil", "fx"])
    assert isinstance(data.index, pd.DatetimeIndex)
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data.exog.to_numpy(), _exog().to_numpy())


def test_polars_from_long(pl):
    long = _full().to_long()
    data = PanelData.from_long(pl.from_pandas(long), characteristics=CHARS, exog=["oil", "fx"])
    np.testing.assert_array_equal(data.y.to_numpy(), _y().to_numpy())
    np.testing.assert_array_equal(data._characteristic_values(), _chars_array())


def test_polars_from_arrays_and_coerce(pl):
    frame = pl.from_pandas(_y().reset_index(drop=True))
    data = PanelData.from_arrays(frame, series_names=SERIES)
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
# Round trips
# ---------------------------------------------------------------------------


def test_long_round_trip_full():
    data = _full()
    back = PanelData.from_long(data.to_long(), characteristics=CHARS, exog=["oil", "fx"], freq=data.freq)
    assert back._equals(data)
    # the long format carries no index freq attribute; values, labels and index type survive
    pd.testing.assert_frame_equal(back.y, data.y, check_names=False, check_freq=False)
    pd.testing.assert_frame_equal(back.exog, data.exog, check_names=False, check_freq=False)


def test_long_round_trip_range_and_period_index():
    for index in (pd.RangeIndex(10, 10 + 3 * T, 3), pd.period_range("2001Q1", periods=T, freq="Q")):
        data = PanelData(_y(index=index), exog=_exog(index=index))
        back = PanelData.from_long(data.to_long(), exog=["oil", "fx"])
        assert type(back.index) is type(index)
        assert back._equals(data)


def test_to_long_layout():
    long = _full().to_long()
    assert list(long.columns) == ["unique_id", "ds", "y", "oil", "fx", *CHARS]
    assert len(long) == T * N
    assert long["unique_id"].tolist() == [s for s in SERIES for _ in range(T)]
    assert long["y"].isna().sum() == 1  # NaN targets are kept
    at = (long["unique_id"] == "ip") & (long["ds"] == _index()[3])
    assert np.isnan(long.loc[at, "value"]).all()


def test_to_long_rejects_reserved_names():
    data = PanelData(_y(), exog=_exog().rename(columns={"oil": "y"}))
    with pytest.raises(DataValidationError, match="clash"):
        data.to_long()


def test_characteristic_column_validation():
    good = _chars()
    with pytest.raises(DataValidationError, match="MultiIndex"):
        PanelData(_y(), characteristics=_exog())
    with pytest.raises(DataValidationError, match="named"):
        PanelData(_y(), characteristics=good.rename_axis(columns=["series", "characteristic"]))
    with pytest.raises(DataValidationError, match="series"):
        PanelData(_y(), characteristics=good.drop(columns="ip", level="series"))
    with pytest.raises(DataValidationError, match="every characteristic"):
        PanelData(_y(), characteristics=good.drop(columns=[("value", "ip")]))
    unnamed = good.rename_axis(columns=[None, None])
    assert PanelData(_y(), characteristics=unnamed).characteristics.columns.names == ["characteristic", "series"]


# ---------------------------------------------------------------------------
# freq
# ---------------------------------------------------------------------------


def test_freq_is_per_series_in_series_order():
    assert _normalize_freq({"ip": "m", "gdp": "q"}, SERIES) == {"gdp": "q", "ip": "m"}
    assert list(_normalize_freq({"ip": "m", "gdp": "q"}, SERIES)) == ["gdp", "ip"]
    assert PanelData(_y(), freq={"ip": "m", "gdp": "q"}).freq == {"gdp": "q", "ip": "m"}


def test_freq_none_and_empty():
    assert _normalize_freq(None, SERIES) is None
    assert _normalize_freq({}, SERIES) is None


def test_freq_errors():
    with pytest.raises(DataValidationError, match="unknown series"):
        PanelData(_y(), freq={"nope": "q"})
    with pytest.raises(DataValidationError, match="frequency code"):
        PanelData(_y(), freq={"gdp": "quarterly"})
    with pytest.raises(DataValidationError, match=r"each series to its frequency code.*\['q'\]"):
        PanelData(_y(), freq={"q": ["gdp"]})  # the grouped form is gone
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
        ({"gdp": "q", "cpi": "m", "ip": "m"}, True),
        ({"gdp": "q"}, True),
    ],
)
def test_private_mixed_frequency(freq, expected):
    assert PanelData(_y(), freq=freq)._is_mixed_frequency() is expected


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
        PanelData(_y(), characteristics=_chars(index=pd.RangeIndex(T)))


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
        PanelData(_y(), exog=_exog().rename(columns={"oil": "size"}), characteristics=_chars())


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
    for frame in (data.y, data.exog, data.characteristics):
        frame.iloc[0, 0] = -999.0
    data.freq["cpi"] = "q"
    data.index.name = "changed"
    data._characteristic_values()[0, 0, 0] = -999.0
    assert data._equals(_full())
    assert data.index.name == "date"
    assert data.freq == {"gdp": "q"}


def test_inputs_are_not_mutated_and_are_copied():
    y, exog, chars = _y(), _exog(), _chars()
    freq = {"gdp": "q"}
    data = PanelData(y, exog=exog, characteristics=chars, freq=freq)
    pd.testing.assert_frame_equal(y, _y())
    pd.testing.assert_frame_equal(exog, _exog())
    pd.testing.assert_frame_equal(chars, _chars())
    y.iloc[0, 0] = -1.0
    exog.iloc[0, 0] = -1.0
    chars.iloc[0, 0] = -1.0
    freq["ip"] = "q"
    assert data._equals(_full())

    raw = _y().join(_exog())
    before = raw.copy()
    PanelData.from_wide(raw, exog=["oil", "fx"])
    pd.testing.assert_frame_equal(raw, before)

    long = _full().to_long()
    before = long.copy()
    PanelData.from_long(long, characteristics=CHARS, exog=["oil", "fx"])
    pd.testing.assert_frame_equal(long, before)


def test_pickle_and_copy():
    data = _full()
    for clone in (pickle.loads(pickle.dumps(data)), copy.copy(data), copy.deepcopy(data)):
        assert isinstance(clone, PanelData)
        assert clone._equals(data)
        with pytest.raises(AttributeError):
            clone._y = None


# ---------------------------------------------------------------------------
# between
# ---------------------------------------------------------------------------


def test_between_datetime_labels():
    data = _full()
    index = _index()
    part = data.between(index[2], index[5])
    assert isinstance(part, PanelData) and part is not data
    assert part.index.equals(index[2:6])  # both ends included
    assert part.index.freq == index.freq
    np.testing.assert_array_equal(part.y.to_numpy(), _y().to_numpy()[2:6])
    np.testing.assert_array_equal(part.exog.to_numpy(), _exog().to_numpy()[2:6])
    np.testing.assert_array_equal(part._characteristic_values(), _chars_array()[2:6])
    assert part.freq == data.freq
    assert data.n_periods == T  # original untouched


def test_between_open_ends_and_periods():
    data = _full()
    assert data.between()._equals(data)
    assert data.between(end="2020-03-31").n_periods == 3
    assert data.between(start="2020-07").n_periods == 2
    # a str or Period covers its whole period: start takes its first stamp, end its last
    part = data.between("2020Q2", "2020Q2")
    assert list(part.index.strftime("%m")) == ["04", "05", "06"]
    assert data.between(start=pd.Period("2020Q3"), end="2020").n_periods == 2
    assert data.between(end=pd.Timestamp("2020-02-15")).n_periods == 1


def test_between_period_index():
    data = PanelData(_y(index=pd.period_range("2020-01", periods=T, freq="M")))
    assert list(data.between("2020Q2", "2020Q2").index.astype(str)) == ["2020-04", "2020-05", "2020-06"]
    assert data.between(pd.Timestamp("2020-03-15")).n_periods == T - 2  # the period holding the stamp


def test_between_range_index_and_errors():
    data = PanelData(_y(index=pd.RangeIndex(T)))
    part = data.between(3, 5)
    assert isinstance(part.index, pd.RangeIndex) and part.index.equals(pd.RangeIndex(3, 6))
    with pytest.raises(DataValidationError, match="no rows"):
        data.between(100, 200)
    with pytest.raises(TypeError, match="int"):
        data.between("2020")
    with pytest.raises(TypeError):
        _full().between(start=1.5)


# ---------------------------------------------------------------------------
# The merge rule (update)
# ---------------------------------------------------------------------------


def _rows(start, values, columns=SERIES, **slots):
    index = pd.date_range(start, periods=len(values), freq="ME", name="date")
    y = pd.DataFrame(np.asarray(values, dtype=float), index=index, columns=columns)
    exog = slots.get("exog")
    return PanelData(y, exog=None if exog is None else pd.DataFrame(exog, index=index, columns=["oil", "fx"]))


def test_merge_appends_new_rows():
    base = PanelData(_y())
    new = _rows("2020-09-30", [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    merged = base._merge(new)
    assert merged.index.equals(pd.date_range("2020-01-31", periods=T + 2, freq="ME", name="date"))
    assert merged.index.freq == "ME"
    np.testing.assert_array_equal(merged.y.to_numpy()[:T], _y().to_numpy())
    np.testing.assert_array_equal(merged.y.to_numpy()[T:], [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])


def test_merge_revisions_and_late_releases_overwrite_cell_by_cell():
    base = PanelData(_y())
    # a revised gdp at t=1, the late release of cpi at t=2 (missing before), NaN elsewhere
    values = np.full((2, N), np.nan)
    values[0, 0] = -1.0
    values[1, 1] = -2.0
    merged = base._merge(_rows("2020-02-29", values))
    expected = _y().to_numpy()
    expected[1, 0] = -1.0
    expected[2, 1] = -2.0
    np.testing.assert_array_equal(merged.y.to_numpy(), expected)
    assert merged.index.equals(base.index)


def test_merge_nan_never_erases_an_observed_value():
    base = PanelData(_y())
    merged = base._merge(_rows("2020-01-31", np.full((T, N), np.nan)))
    assert merged._equals(base)


def test_merge_full_vintage_equals_new_cells_only():
    base = PanelData(_y().iloc[:6])
    full = PanelData(_y())
    only_new = full.between(_index()[6])
    assert base._merge(full)._equals(base._merge(only_new))
    assert base._merge(full)._equals(full)


def test_merge_is_idempotent_and_reorders_columns():
    base = _full()
    assert base._merge(base)._equals(base)
    shuffled = PanelData(
        _y()[["ip", "gdp", "cpi"]], exog=_exog()[["fx", "oil"]], characteristics=_chars().iloc[:, ::-1]
    )
    assert base._merge(shuffled)._equals(base)


def test_merge_exog_and_characteristics_follow_the_same_rule():
    base = _full()
    index = _index()[-2:]
    y = pd.DataFrame(np.nan, index=index, columns=SERIES)
    exog = pd.DataFrame({"oil": [np.nan, -5.0], "fx": [np.nan, np.nan]}, index=index)
    chars = _chars().iloc[-2:] * np.nan
    chars.iloc[0, 0] = -7.0  # (size, gdp) at the second-to-last row
    merged = base._merge(PanelData(y, exog=exog, characteristics=chars))
    assert merged.exog.iloc[-1, 0] == -5.0 and merged.exog.iloc[-2, 0] == _exog().iloc[-2, 0]
    expected = _chars_array()
    expected[-2, 0, 0] = -7.0
    np.testing.assert_array_equal(merged._characteristic_values(), expected)


def test_merge_replace_is_exactly_the_new_data():
    base = PanelData(_y())
    vintage = PanelData(_y().iloc[2:5] * 0.0)
    replaced = base._merge(vintage, replace=True)
    assert replaced._equals(vintage)
    # replace=True may also start before the conditioning data
    earlier = PanelData(_y(index=pd.date_range("2019-01-31", periods=T, freq="ME", name="date")))
    assert base._merge(earlier, replace=True)._equals(earlier)


def test_merge_rejects_data_before_the_conditioning_start_without_replace():
    base = PanelData(_y().iloc[2:])
    with pytest.raises(DataValidationError, match="before the conditioning data.*replace=True"):
        base._merge(PanelData(_y()))


def test_merge_rejects_unknown_stamps_inside_the_conditioning_period():
    base = PanelData(_y())
    odd = PanelData(pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.DatetimeIndex(["2020-03-15"], name="date"), columns=SERIES))
    with pytest.raises(DataValidationError, match="inside the conditioning period"):
        base._merge(odd)


def test_merge_shape_checks():
    base = _full()
    with pytest.raises(DataValidationError, match=r"misses \['ip'\]"):
        base._merge(PanelData(_y()[["gdp", "cpi"]], exog=_exog()))
    with pytest.raises(DataValidationError, match="does not know"):
        PanelData(_y())._merge(_rows("2020-09-30", [[1.0, 2.0, 3.0, 4.0]], columns=[*SERIES, "new"]))
    with pytest.raises(DataValidationError, match="exog"):
        base._merge(PanelData(_y(), characteristics=_chars(), freq={"gdp": "q"}))
    with pytest.raises(DataValidationError, match="characteristics"):
        base._merge(PanelData(_y(), exog=_exog(), freq={"gdp": "q"}))
    with pytest.raises(DataValidationError, match="freq"):
        base._merge(PanelData(_y(), exog=_exog(), characteristics=_chars(), freq={"cpi": "q"}))
    # freq=None inherits the conditioning freq
    assert base._merge(PanelData(_y(), exog=_exog(), characteristics=_chars())).freq == {"gdp": "q"}
    with pytest.raises(DataValidationError, match="RangeIndex"):
        PanelData(_y())._merge(PanelData(_y(index=pd.RangeIndex(T))))
    with pytest.raises(DataValidationError, match="periods of frequency"):
        quarterly = PanelData(_y(index=pd.period_range("2020Q1", periods=T, freq="Q")))
        PanelData(_y(index=pd.period_range("2020-01", periods=T, freq="M")))._merge(quarterly)


def test_merge_range_index_continues_its_grid():
    base = PanelData(_y(index=pd.RangeIndex(0, 2 * T, 2)))
    new = PanelData(pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.RangeIndex(2 * T, 2 * T + 2, 2), columns=SERIES))
    merged = base._merge(new)
    assert isinstance(merged.index, pd.RangeIndex) and merged.index.equals(pd.RangeIndex(0, 2 * T + 2, 2))
    with pytest.raises(DataValidationError, match="off the conditioning grid"):
        base._merge(PanelData(pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.RangeIndex(17, 18), columns=SERIES)))
    with pytest.raises(DataValidationError, match="evenly spaced"):
        base._merge(PanelData(pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.RangeIndex(20, 22, 2), columns=SERIES)))


def test_merge_new_series_for_allocators():
    base = _full()
    index = pd.date_range("2020-09-30", periods=2, freq="ME", name="date")
    y = pd.DataFrame(np.ones((2, N + 1)), index=index, columns=["new", *SERIES])
    chars = pd.DataFrame(
        2.0, index=index, columns=pd.MultiIndex.from_product([CHARS, ["new", *SERIES]], names=["characteristic", "series"])
    )
    exog = pd.DataFrame(3.0, index=index, columns=["oil", "fx"])
    merged = base._merge(PanelData(y, exog=exog, characteristics=chars, freq=None), new_series=True)
    assert merged.series_names == (*SERIES, "new")
    assert np.isnan(merged.y["new"].to_numpy()[:T]).all() and (merged.y["new"].to_numpy()[T:] == 1.0).all()
    values = merged._characteristic_values()
    assert values.shape == (T + 2, N + 1, M)
    assert np.isnan(values[:T, N]).all() and (values[T:] == 2.0).all()
    np.testing.assert_array_equal(values[:T, :N], _chars_array())


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------


def test_coerce_panel_is_returned_as_is():
    data = _full()
    assert PanelData._coerce(data) is data


def test_coerce_dataframe_means_y():
    y = _y()
    data = PanelData._coerce(y)
    assert data._equals(PanelData(y))
    assert data.exog is None


def test_coerce_2d_array_means_from_arrays():
    arr = np.arange(12.0).reshape(4, 3)
    data = PanelData._coerce(arr)
    assert data._equals(PanelData.from_arrays(arr))
    assert isinstance(data.index, pd.RangeIndex)
    with pytest.raises(DataValidationError, match="2-D"):
        PanelData._coerce(np.arange(4.0))


@pytest.mark.parametrize("obj", [None, [[1.0, 2.0]], {"a": [1.0]}, pd.Series([1.0, 2.0]), 3.0])
def test_coerce_other_types_raise_type_error(obj):
    with pytest.raises(TypeError):
        PanelData._coerce(obj)


def test_repr():
    text = repr(_full())
    assert text.startswith("PanelData(") and "n_series=3" in text and "n_periods=8" in text
    assert "n_characteristics=2" in text and "'gdp': 'q'" in text

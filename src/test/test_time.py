"""The time grammar ``dfm_python._time`` (spec sections 4.4, 4.8 and 4.12)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from dfm_python import _time
from dfm_python._time import Grid
from dfm_python.errors import ConfigurationError, DataValidationError

ORIGIN = "2025-10-31"


def _monthly(start="2015-01-31", end=ORIGIN, freq="ME", name="date"):
    return pd.date_range(start, end, freq=freq, name=name)


def _grid(index=None, clock="m"):
    index = _monthly() if index is None else index
    return Grid(index, _time.grid_freq(index), clock)


def _stamps(grid, end, start=None):
    first, last = grid.resolve(end, start)
    return [str(s.date()) for s in grid.stamps(first, last)]


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq, code",
    [
        ("ME", "m"), ("MS", "m"), ("BME", "m"), ("W-FRI", "w"), ("W-SUN", "w"), ("QE-DEC", "q"), ("QS-JAN", "q"),
        ("QE-NOV", "q"), ("B", "d"), ("D", "d"), ("C", "d"), ("6ME", "sa"), ("2QE-OCT", "sa"), ("YE-DEC", "a"),
        ("YS-JAN", "a"), ("h", None), ("2D", None), ("2ME", None), (None, None),
    ],
)
def test_clock_family(freq, code):
    assert _time.clock_family(freq) == code


@pytest.mark.parametrize("freq, code", [("ME", "m"), ("MS", "m"), ("W-FRI", "w"), ("QE-DEC", "q"), ("B", "d")])
def test_clock_inference_from_a_datetime_index(freq, code):
    index = pd.date_range("2020-01-01", periods=30, freq=freq)
    grid = _time.grid_freq(index)
    assert _time.resolve_clock(index, grid) == code
    # without the freq attribute pandas infers it
    bare = pd.DatetimeIndex(list(index))
    assert bare.freq is None
    assert _time.resolve_clock(bare, _time.grid_freq(bare)) == code


@pytest.mark.parametrize("freq, code", [("M", "m"), ("Q", "q"), ("Y", "a"), ("W-FRI", "w"), ("D", "d")])
def test_clock_inference_from_a_period_index(freq, code):
    index = pd.period_range("2020-01-01", periods=12, freq=freq)
    assert _time.resolve_clock(index, _time.grid_freq(index)) == code


def test_explicit_clock_must_match_the_family():
    index = _monthly()
    grid = _time.grid_freq(index)
    assert _time.resolve_clock(index, grid, "m") == "m"
    with pytest.raises(DataValidationError, match=r"clock='q' does not match the index frequency 'ME'"):
        _time.resolve_clock(index, grid, "q")
    with pytest.raises(ConfigurationError, match="clock must be one of"):
        _time.resolve_clock(index, grid, "monthly")
    # a RangeIndex takes the explicit clock as the length of a period
    assert _time.resolve_clock(pd.RangeIndex(10), None, "q") == "q"
    assert _time.resolve_clock(pd.RangeIndex(10), None) is None


def test_slower_series():
    names = ["gdp", "ip", "cpi"]
    assert _time.slower_series(None, names, "m") == {}
    assert _time.slower_series({"gdp": "q", "ip": "m"}, names, "m") == {"gdp": "q"}
    with pytest.raises(DataValidationError, match="faster than the clock"):
        _time.slower_series({"gdp": "w"}, names, "m")
    # without a known clock only a single-frequency panel is allowed
    assert _time.slower_series({n: "q" for n in names}, names, None) == {}
    with pytest.raises(DataValidationError, match="explicit clock"):
        _time.slower_series({"gdp": "q"}, names, None, is_range=True)
    with pytest.raises(DataValidationError, match="not one of the clock frequencies"):
        _time.slower_series({"gdp": "q"}, names, None)


def test_periods_per():
    assert (_time.periods_per("q", "m"), _time.periods_per("m", "w"), _time.periods_per("sa", "m")) == (3, 4, 6)
    assert _time.periods_per("a", "q") == 4 and _time.periods_per("w", "d") == 7


# ---------------------------------------------------------------------------
# Regular grids
# ---------------------------------------------------------------------------


def test_regular_indexes_pass():
    assert _time.grid_freq(pd.RangeIndex(5)) is None
    assert _time.grid_freq(_monthly()) == to_offset("ME")
    assert _time.grid_freq(pd.DatetimeIndex(["2025-03-31", "2025-06-30"])) == to_offset("QE-DEC")
    assert _time.grid_freq(pd.period_range("2020Q1", periods=4, freq="Q")) == to_offset("QE-DEC")
    cbd = pd.offsets.CustomBusinessDay(holidays=["2024-01-15"])
    assert _time.grid_freq(pd.date_range("2024-01-01", periods=20, freq=cbd)) == cbd


def test_gaps_raise_with_the_missing_stamps_and_the_reindex_hint():
    index = _monthly().delete([5, 6, 40])
    with pytest.raises(DataValidationError) as info:
        _time.grid_freq(index, what="DFM.fit")
    text = str(info.value)
    assert text.startswith("DFM.fit: the time index has gaps")
    assert "misses 3 stamp(s), first 2015-06-30, 2015-07-31, 2018-05-31" in text
    assert "df.reindex(pd.date_range('2015-01-31', '2025-10-31', freq='ME'))" in text


def test_business_days_with_holidays_are_gaps():
    index = pd.bdate_range("2024-01-01", "2024-03-01").delete([10, 20])
    with pytest.raises(DataValidationError, match=r"misses 2 stamp\(s\), first 2024-01-15, 2024-01-29.*freq='B'"):
        _time.grid_freq(index)


def test_period_index_gaps():
    index = pd.period_range("2020-01", periods=12, freq="M").delete(4)
    with pytest.raises(DataValidationError, match=r"2020-05.*pd.period_range\('2020-01', '2020-12', freq='M'\)"):
        _time.grid_freq(index)


def test_irregular_index_names_the_closest_grid():
    weekly = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
    mixed = weekly.union(pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"]))
    with pytest.raises(DataValidationError) as info:
        _time.grid_freq(mixed)
    text = str(info.value)
    assert "irregular" in text and "closest grid is 'W-FRI' (30 of 33 stamps on it)" in text
    assert "off it: 2024-01-31, 2024-02-29, 2024-03-31" in text
    assert "freq='W-FRI'" in text


def test_intraday_gaps_are_named_on_the_intraday_grid():
    # an hourly or 5-minute index with gaps is not "irregular on a daily grid"
    hourly = pd.date_range("2025-01-01", periods=10, freq="h").delete(3)
    hourly = pd.DatetimeIndex(list(hourly))  # no freq
    with pytest.raises(DataValidationError, match=r"'h' grid but misses 1 stamp\(s\), first 2025-01-01 03:00:00") as info:
        _time.grid_freq(hourly)
    assert "freq='h'" in str(info.value) and "'D'" not in str(info.value)
    five = pd.DatetimeIndex(list(pd.date_range("2025-01-02 09:30", periods=40, freq="5min").delete([4, 20])))
    with pytest.raises(DataValidationError, match=r"'5min' grid but misses 2 stamp\(s\), first 2025-01-02 09:50:00"):
        _time.grid_freq(five)
    assert _time.grid_freq(pd.DatetimeIndex(["2025-01-01 00:00", "2025-01-01 03:00"])) == to_offset("3h")


def test_bundled_macro_csv_is_irregular_on_a_weekly_clock():
    from pathlib import Path

    csv = Path(__file__).resolve().parents[2] / "data" / "macro.csv"
    index = pd.DatetimeIndex(pd.read_csv(csv, usecols=["date"], parse_dates=["date"])["date"]).sort_values()[-400:]
    with pytest.raises(DataValidationError, match="irregular.*closest grid is 'W-FRI'"):
        _time.grid_freq(index)


def test_single_stamp_without_freq():
    with pytest.raises(DataValidationError, match="single time stamp"):
        _time.grid_freq(pd.DatetimeIndex(["2020-01-31"]))


def test_check_on_grid():
    freq = to_offset("ME")
    _time.check_on_grid(_monthly(), freq)
    with pytest.raises(DataValidationError, match="not on the model's 'ME' grid"):
        _time.check_on_grid(pd.DatetimeIndex(["2025-01-31", "2025-02-15"]), freq)
    with pytest.raises(DataValidationError, match="gaps"):
        _time.check_on_grid(_monthly().delete(3), freq)
    _time.check_on_grid(pd.RangeIndex(3), None)


# ---------------------------------------------------------------------------
# Anchors of slower series
# ---------------------------------------------------------------------------


def _quarterly_values(index, anchor_month_mod=0):
    values = np.full((len(index), 2), np.nan)
    values[:, 1] = 1.0
    at = np.flatnonzero(index.month % 3 == anchor_month_mod)
    values[at, 0] = np.arange(len(at), dtype=float)
    return values


def test_anchors_on_a_month_end_clock():
    index = _monthly("2020-01-31", "2021-12-31")
    grid = _grid(index)
    assert _time.check_anchors(_quarterly_values(index), ["gdp", "ip"], {"gdp": "q"}, grid) == {"gdp": None}
    mask = grid.anchor_mask(0, len(index) - 1, "q")
    assert list(index[mask].strftime("%m")) == ["03", "06", "09", "12"] * 2
    with pytest.raises(DataValidationError, match=r"'gdp'.*last stamp of each quarter.*2020-02-29"):
        _time.check_anchors(_quarterly_values(index, 2), ["gdp", "ip"], {"gdp": "q"}, grid)


def test_anchors_on_a_month_start_clock():
    index = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    grid = _grid(index)
    mask = grid.anchor_mask(0, len(index) - 1, "q")
    assert list(index[mask].strftime("%Y-%m-%d"))[:2] == ["2020-03-01", "2020-06-01"]
    assert _time.check_anchors(_quarterly_values(index), ["gdp", "ip"], {"gdp": "q"}, grid) == {"gdp": None}


def test_anchors_on_a_weekly_clock_are_the_last_week_of_each_month():
    index = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
    grid = _grid(index, "w")
    anchors = index[grid.anchor_mask(0, len(index) - 1, "m")]
    assert [str(s.date()) for s in anchors] == ["2024-01-26", "2024-02-23", "2024-03-29", "2024-04-26"]


def test_anchors_on_a_range_index():
    index = pd.RangeIndex(24)
    grid = Grid(index, None, "m")
    values = np.full((24, 1), np.nan)
    values[2::3, 0] = 1.0  # rows 2, 5, 8, ...: residue 2 mod 3
    assert _time.check_anchors(values, ["gdp"], {"gdp": "q"}, grid) == {"gdp": 2}
    assert list(np.flatnonzero(grid.anchor_mask(0, 23, "q", residue=2))) == list(range(2, 24, 3))
    values[4, 0] = 1.0
    with pytest.raises(DataValidationError, match="different positions"):
        _time.check_anchors(values, ["gdp"], {"gdp": "q"}, grid)
    # with the residue fixed at fit, an update is checked against it
    shifted = np.full((24, 1), np.nan)
    shifted[1::3, 0] = 1.0
    with pytest.raises(DataValidationError, match="value // step % 3 == 2"):
        _time.check_anchors(shifted, ["gdp"], {"gdp": "q"}, grid, residues={"gdp": 2})


# ---------------------------------------------------------------------------
# Targets (spec section 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "end, start, first, last, n",
    [
        (1, None, "2025-11-30", "2025-11-30", 1),
        (6, None, "2025-11-30", "2026-04-30", 6),
        (0, None, "2025-10-31", "2025-10-31", 1),
        (0, -11, "2024-11-30", "2025-10-31", 12),
        ("2026Q1", None, "2025-11-30", "2026-03-31", 5),
        ("2026Q1", "2025Q3", "2025-07-31", "2026-03-31", 9),
        ("2025Q3", None, "2025-09-30", "2025-09-30", 1),
    ],
)
def test_the_spec_targeting_table(end, start, first, last, n):
    stamps = _stamps(_grid(), end, start)
    assert (stamps[0], stamps[-1], len(stamps)) == (first, last, n)


def test_target_stamps_are_named_and_on_the_grid():
    grid = _grid()
    first, last = grid.resolve(3, -2)
    stamps = grid.stamps(first, last)
    assert stamps.name == "date" and stamps.freq == "ME" and len(stamps) == 6
    unnamed = Grid(_monthly(name=None), to_offset("ME"), "m")
    assert unnamed.stamps(*unnamed.resolve(1)).name == "time"


def test_timestamp_and_datetime_targets():
    grid = _grid()
    assert _stamps(grid, pd.Timestamp("2026-01-31")) == ["2025-11-30", "2025-12-31", "2026-01-31"]
    assert _stamps(grid, dt.datetime(2025, 9, 30)) == ["2025-09-30"]
    assert _stamps(grid, dt.date(2026, 1, 31), start=pd.Timestamp("2025-12-31")) == ["2025-12-31", "2026-01-31"]
    with pytest.raises(ConfigurationError, match="not a stamp of the model's 'ME' grid"):
        grid.resolve(pd.Timestamp("2026-01-15"))
    with pytest.raises(ConfigurationError, match="not a stamp"):
        grid.resolve(pd.Timestamp("2025-09-15"))


def test_str_and_period_targets_cover_their_period():
    grid = _grid()
    assert _stamps(grid, pd.Period("2026-02", "M")) == ["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28"]
    assert _stamps(grid, "2026") == _stamps(grid, "2026-12")
    assert len(_stamps(grid, "2026")) == 14
    assert _stamps(grid, "2025-09-30") == ["2025-09-30"]  # a day period holding one grid stamp
    with pytest.raises(ConfigurationError, match="holds no stamp"):
        grid.resolve("2025-09-15")
    with pytest.raises(ConfigurationError, match="not a period label"):
        grid.resolve("not a date")


def test_default_start_rule():
    grid = _grid()
    assert grid.resolve(3) == (grid.n, grid.n + 2)  # after the origin: from the next step
    assert grid.resolve(-2) == (grid.n - 3, grid.n - 3)  # in sample: end itself
    assert grid.resolve(0) == (grid.n - 1, grid.n - 1)


def test_target_errors():
    grid = _grid()
    with pytest.raises(ConfigurationError, match="lies after end"):
        grid.resolve(-1, start=2)
    with pytest.raises(ConfigurationError, match="^start=-500 lies before the first conditioning stamp 2015-01-31"):
        grid.resolve(0, start=-500)
    with pytest.raises(ConfigurationError, match="^end=-500 lies before the first conditioning stamp"):
        grid.resolve(-500)  # without start the first target is end itself
    with pytest.raises(ConfigurationError, match="2010Q1 lies before the first conditioning stamp"):
        grid.resolve("2010Q1", start="2010Q1")
    with pytest.raises(ConfigurationError, match="before the first conditioning stamp"):
        grid.resolve(0, start="2010Q1")
    # A grid stamp before the data lies before it; a stamp off the grid there is still off the grid.
    with pytest.raises(ConfigurationError, match=r"^start=.*2014-12-31.* lies before .* \(1 step\(s\) earlier\)"):
        grid.resolve(0, start=pd.Timestamp("2014-12-31"))
    with pytest.raises(ConfigurationError, match=r"^end=.*2014-11-30.* lies before .* \(2 step\(s\) earlier\)"):
        grid.resolve(pd.Timestamp("2014-11-30"))
    with pytest.raises(ConfigurationError, match="not a stamp of the model's 'ME' grid"):
        grid.resolve(0, start=pd.Timestamp("2014-12-15"))
    hourly = Grid(pd.date_range("2000-01-03 01:00", periods=10, freq="h"), to_offset("h"), None)
    with pytest.raises(ConfigurationError, match=r"lies before .* \(2 step\(s\) earlier\)"):
        hourly.resolve(pd.Timestamp("2000-01-02 23:00"))
    with pytest.raises(ConfigurationError, match="not a stamp of the model's 'h' grid"):
        hourly.resolve(pd.Timestamp("2000-01-02 23:30"))
    for bad in (1.5, None, True, [1], {"a": 1}):
        with pytest.raises(TypeError):
            grid.resolve(bad)


def test_range_index_takes_ints_only():
    grid = Grid(pd.RangeIndex(10, 40, 2), None, None)
    first, last = grid.resolve(2, -1)
    assert list(grid.stamps(first, last)) == [36, 38, 40, 42]
    with pytest.raises(TypeError, match="RangeIndex"):
        grid.resolve("2025Q1")
    with pytest.raises(TypeError, match="RangeIndex"):
        grid.resolve(pd.Timestamp("2025-01-31"))


def test_period_index_targets():
    index = pd.period_range("2024-01", "2025-10", freq="M", name="month")
    grid = Grid(index, None, "m")
    stamps = grid.stamps(*grid.resolve("2026Q1", "2025Q3"))
    assert list(stamps.astype(str)) == [f"2025-{m:02d}" for m in range(7, 13)] + ["2026-01", "2026-02", "2026-03"]
    assert list(grid.stamps(*grid.resolve(pd.Timestamp("2025-11-15"))).astype(str)) == ["2025-11"]


def test_future_stamps_continue_a_custom_business_day_grid():
    cbd = pd.offsets.CustomBusinessDay(holidays=["2024-12-25", "2025-01-01"])
    index = pd.date_range("2024-12-02", "2024-12-24", freq=cbd)
    grid = Grid(index, _time.grid_freq(index), "d")
    future = grid.future(3)
    assert [str(s.date()) for s in future] == ["2024-12-26", "2024-12-27", "2024-12-30"]


# ---------------------------------------------------------------------------
# nowcast periods (spec section 4.8)
# ---------------------------------------------------------------------------


def _q_observed(index):
    """Positions of quarterly values released through 2025Q2."""
    pos = np.flatnonzero((index.month % 3 == 0) & (index <= "2025-06-30"))
    return pos


def test_quarterly_periods_with_defaults():
    index = _monthly()
    grid = _grid(index)
    periods = grid.periods("q", 1, None, observed=_q_observed(index))
    # end=1: the period after the one holding the origin (2025-10-31 lies in 2025Q4)
    assert [str(p.label) for p in periods] == ["2025Q2", "2025Q3", "2025Q4", "2026Q1"]
    origin = grid.n - 1
    observed = set(_q_observed(index))
    status = [p.status(origin, p.last in observed) for p in periods]
    assert status == ["released", "backcast", "nowcast", "forecast"]
    q4 = periods[2]
    assert str(grid.stamps(q4.first, q4.first)[0].date()) == "2025-10-31"
    assert str(grid.stamps(q4.last, q4.last)[0].date()) == "2025-12-31"


def test_the_period_ending_exactly_at_the_origin_is_the_nowcast():
    index = _monthly(end="2025-09-30")
    grid = _grid(index)
    periods = grid.periods("q", 1, -1, observed=_q_observed(index))
    origin = grid.n - 1
    assert [str(p.label) for p in periods] == ["2025Q2", "2025Q3", "2025Q4"]
    assert [p.status(origin, False) for p in periods] == ["backcast", "nowcast", "forecast"]
    assert periods[1].last == origin


def test_period_offsets_and_labels():
    grid = _grid()
    origin_q = grid.periods("q", 0, 0)
    assert [str(p.label) for p in origin_q] == ["2025Q4"]
    assert [str(p.label) for p in grid.periods("q", 2, -2)] == ["2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    assert [str(p.label) for p in grid.periods("q", "2026Q2", "2025-08")] == ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    assert [str(p.label) for p in grid.periods("q", pd.Period("2026", "Y"), "2026")] == [
        "2026Q1", "2026Q2", "2026Q3", "2026Q4"
    ]
    assert [str(p.label) for p in grid.periods("a", 1, 0)] == ["2025", "2026"]
    sa = grid.periods("sa", 1, 0)
    assert [(p.label.month, p.label.year) for p in sa] == [(7, 2025), (1, 2026)]


def test_clock_periods_of_a_clock_series():
    index = _monthly()
    grid = _grid(index)
    periods = grid.periods(None, 2, None, observed=np.array([grid.n - 2]))
    assert [str(p.label) for p in periods] == ["2025-09", "2025-10", "2025-11", "2025-12"]
    assert all(p.first == p.last for p in periods)
    assert [p.status(grid.n - 1, False) for p in periods] == ["backcast", "nowcast", "forecast", "forecast"]
    assert [str(p.label) for p in grid.periods(None, "2025Q4", "2025Q4")] == ["2025-10", "2025-11", "2025-12"]


def test_clock_periods_on_a_weekly_clock():
    index = pd.date_range("2025-09-05", "2025-10-31", freq="W-FRI")
    grid = _grid(index, "w")
    periods = grid.periods(None, 1, 0)
    assert [p.label.freqstr for p in periods] == ["W-FRI", "W-FRI"]
    assert str(periods[0].label.end_time.date()) == "2025-10-31"
    # a mid-week label names the week that holds it
    assert [p.first for p in grid.periods(None, "2025-10-29", "2025-10-29")] == [grid.n - 1]


def test_default_start_without_releases_is_the_first_conditioning_period():
    grid = _grid(_monthly("2025-02-28"))
    periods = grid.periods("q", 0, None, observed=np.array([], dtype=int))
    assert [str(p.label) for p in periods] == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert periods[0].first == 0  # Jan lies before the data; the period starts at index[0]


def test_period_errors():
    grid = _grid(_monthly("2025-01-31"))
    with pytest.raises(ConfigurationError, match="after end"):
        grid.periods("q", -1, 1)
    with pytest.raises(ConfigurationError, match="before the first conditioning period"):
        grid.periods("q", 0, "2024Q4")
    with pytest.raises(TypeError):
        grid.periods("q", 1.5, None)


def test_range_index_periods():
    grid = Grid(pd.RangeIndex(24), None, "m")
    periods = grid.periods("q", 1, -1, residue=2)
    assert [p.label for p in periods] == [20, 23, 26]
    assert [(p.first, p.last) for p in periods] == [(18, 20), (21, 23), (24, 26)]
    assert [p.status(23, False) for p in periods] == ["backcast", "nowcast", "forecast"]
    with pytest.raises(TypeError, match="RangeIndex"):
        grid.periods("q", "2025Q1", None, residue=2)


def test_status_rule():
    period = _time.NowcastPeriod(key=0, label="p", first=3, last=5)
    assert period.status(4, released=True) == "released"
    assert period.status(6, False) == "backcast"
    assert [period.status(t, False) for t in (3, 4, 5)] == ["nowcast"] * 3
    assert period.status(2, False) == "forecast"


# ---------------------------------------------------------------------------
# Allocators' forward stamp and PanelData.between
# ---------------------------------------------------------------------------


def test_forward_stamp():
    cbd = pd.offsets.CustomBusinessDay(holidays=["2024-12-25"])
    index = pd.date_range("2024-12-02", "2024-12-24", freq=cbd)
    assert _time.forward_stamp(index) == pd.Timestamp("2024-12-26")
    assert _time.forward_stamp(pd.bdate_range("2024-01-01", "2024-01-05")) == pd.Timestamp("2024-01-08")
    holidays = pd.bdate_range("2024-01-01", "2024-01-31").delete([3, 10])  # no frequency: median step
    assert _time.forward_stamp(holidays) == holidays[-1] + pd.Timedelta(days=1)
    assert _time.forward_stamp(pd.RangeIndex(0, 10, 2)) == 10
    assert _time.forward_stamp(pd.period_range("2020-01", periods=3, freq="M")) == pd.Period("2020-04", "M")
    with pytest.raises(DataValidationError, match="single time stamp"):
        _time.forward_stamp(pd.DatetimeIndex(["2020-01-31"]))


def test_window():
    index = _monthly("2025-01-31", "2025-12-31")
    assert _time.window(index, "2025Q2", "2025Q2") == slice(3, 6)
    assert _time.window(index, None, pd.Timestamp("2025-02-28")) == slice(0, 2)
    assert _time.window(index, "2026") == slice(12, 12)

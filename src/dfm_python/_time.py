"""The time grammar of the estimators (spec sections 4.4, 4.8 and 4.12).

- Clocks: :func:`clock_family` maps a pandas frequency to its clock code
  (``d`` = D/B, ``w`` = W-*, ``m`` = ME/MS, ``q`` = QE*/QS*, ``sa`` = 6M,
  ``a`` = YE/YS); :func:`resolve_clock` infers the clock from the index or
  checks that an explicit one matches the index family.
- Regular grids: :func:`grid_freq` returns the step of a regular index and
  otherwise raises ``DataValidationError`` naming the first missing (or
  off-grid) stamps and the ``df.reindex(pd.date_range(...))`` fix;
  :func:`check_on_grid` checks an index against a fixed step (``update``).
- Slower series: :func:`slower_series` sorts the ``freq`` codes into the
  series slower than the clock, and :func:`check_anchors` checks that each
  holds values only at the last grid stamp of its periods (on a
  ``RangeIndex``: at one residue modulo the clock periods per period).
- Targets: :class:`Grid` resolves ``end`` / ``start`` to grid positions
  (:meth:`Grid.resolve`), builds the target stamps (:meth:`Grid.stamps`),
  masks slower series off their period ends (:meth:`Grid.anchor_mask`) and
  lists the per-series periods of ``nowcast`` (:meth:`Grid.periods`).
- :func:`forward_stamp`: the stamp after the last one for an allocator
  (``index.freq``, else the inferred frequency, else the median step).
- :func:`window`: the rows of ``PanelData.between``.

Positions count grid stamps from the first conditioning stamp: 0 is
``index[0]``, ``len(index) - 1`` the origin ``T`` and larger positions
continue the grid after ``T``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .config.constants import FREQUENCY_HIERARCHY, PERIODS_PER_YEAR, VALID_FREQUENCIES
from .errors import ConfigurationError, DataValidationError

__all__ = [
    "CLOCK_CODES",
    "Grid",
    "NowcastPeriod",
    "check_anchors",
    "check_on_grid",
    "clock_family",
    "forward_stamp",
    "grid_freq",
    "periods_per",
    "resolve_clock",
    "slower_series",
    "window",
]

#: The clock codes, fastest first.
CLOCK_CODES = ("d", "w", "m", "q", "sa", "a")

_PERIOD_WORDS = {"d": "day", "w": "week", "m": "month", "q": "quarter", "sa": "half-year", "a": "year"}
_MONTHS_PER_STEP = {
    pd.offsets.MonthEnd: 1,
    pd.offsets.MonthBegin: 1,
    pd.offsets.BusinessMonthEnd: 1,
    pd.offsets.BusinessMonthBegin: 1,
    pd.offsets.QuarterEnd: 3,
    pd.offsets.QuarterBegin: 3,
    pd.offsets.BQuarterEnd: 3,
    pd.offsets.BQuarterBegin: 3,
    pd.offsets.YearEnd: 12,
    pd.offsets.YearBegin: 12,
    pd.offsets.BYearEnd: 12,
    pd.offsets.BYearBegin: 12,
}
_MONTHS_TO_CODE = {1: "m", 3: "q", 6: "sa", 12: "a"}
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
# Calendar grids tried when pandas infers no frequency (gaps or an irregular index).
_CANDIDATE_GRIDS = (
    "D", "B", *(f"W-{d}" for d in _WEEKDAYS), "MS", "ME", "BMS", "BME",
    "QS-JAN", "QS-FEB", "QS-MAR", "QE-DEC", "QE-JAN", "QE-FEB", "6MS", "6ME",
    *(f"YS-{m}" for m in _MONTHS), *(f"YE-{m}" for m in _MONTHS),
)
_N_SHOWN = 5
# The longest grid built from an intraday index's smallest step (a guard against a tiny step).
_MAX_STEP_GRID = 1_000_000


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_stamp(stamp: Any) -> str:
    if isinstance(stamp, pd.Timestamp):
        return str(stamp.date()) if stamp == stamp.normalize() and stamp.tz is None else str(stamp)
    return str(stamp)


def _fmt_stamps(stamps: Sequence[Any], n: int = _N_SHOWN) -> str:
    shown = ", ".join(_fmt_stamp(s) for s in list(stamps)[:n])
    more = len(stamps) - n
    return shown + (f" and {more} more" if more > 0 else "")


def _ns(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """``index`` at nanosecond resolution (indexes at other units do not compare equal)."""
    return index.as_unit("ns") if index.unit != "ns" else index


def _wall(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Wall-clock times without the time zone (calendar periods follow the local date)."""
    return times.tz_localize(None) if times.tz is not None else times


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


def clock_family(freq: Any) -> Optional[str]:
    """The clock code of a pandas frequency (an offset or alias), or None outside the families.

    ``d``: D, B (and custom business days); ``w``: W-*; ``m``: ME, MS (and the
    business variants); ``q``: QE*, QS*; ``sa``: 6ME, 6MS, 2QE*; ``a``: YE*, YS*.
    Period frequencies (``M``, ``Q-DEC``, ``Y-DEC``, ``W-SUN``, ``D``) map the same way.
    """
    if freq is None:
        return None
    try:
        offset = to_offset(freq)
    except (TypeError, ValueError):
        return None
    n = offset.n
    if isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay)):
        return "d" if n == 1 else None
    if isinstance(offset, pd.offsets.Week):
        return "w" if n == 1 and offset.weekday is not None else None
    months = _MONTHS_PER_STEP.get(type(offset))
    return None if months is None else _MONTHS_TO_CODE.get(months * n)


def periods_per(code: str, clock: str) -> int:
    """Clock periods per period of ``code``: q on m 3, m on w 4, sa on m 6, a on q 4."""
    return max(1, int(round(PERIODS_PER_YEAR[clock] / PERIODS_PER_YEAR[code])))


def _check_code(code: Any, what: str) -> None:
    if not isinstance(code, str) or code not in VALID_FREQUENCIES:
        raise ConfigurationError(f"{what} must be one of {list(CLOCK_CODES)} or None, got {code!r}.")


def resolve_clock(index: pd.Index, grid: Any, clock: Optional[str] = None, *, what: str = "data") -> Optional[str]:
    """The clock code of the data: inferred from the index, or ``clock`` checked against it.

    Parameters
    ----------
    index : pandas.Index
        The (regular) data index.
    grid : pandas.DateOffset or None
        Its step from :func:`grid_freq` (None for a ``RangeIndex``).
    clock : str, optional
        An explicit clock code. On a ``RangeIndex`` it defines the length of
        a clock period and is returned as is.

    Returns
    -------
    str or None
        None for a ``RangeIndex`` without ``clock``, or an index frequency
        outside the clock families (for example hourly data).

    Raises
    ------
    ConfigurationError
        If ``clock`` is not a clock code.
    DataValidationError
        If ``clock`` does not match the family of the index frequency.
    """
    if clock is not None:
        _check_code(clock, "clock")
    if isinstance(index, pd.RangeIndex):
        return clock
    step = index.freq if isinstance(index, pd.PeriodIndex) else grid
    family = clock_family(step)
    if clock is None or clock == family:
        return family
    freqstr = getattr(step, "freqstr", str(step))
    found = f"clock family {family!r}" if family else "no clock family"
    raise DataValidationError(
        f"{what}: clock={clock!r} does not match the index frequency {freqstr!r} ({found}). Pass clock=None to "
        f"use the index frequency, or put the rows on a {_PERIOD_WORDS[clock]}ly grid first."
    )


def slower_series(
    freq: Optional[Mapping[Any, str]],
    series_names: Sequence[Any],
    clock: Optional[str],
    *,
    is_range: bool = False,
    what: str = "data",
) -> dict:
    """The series sampled more slowly than the clock, ``{series: code}`` in ``series_names`` order.

    A series that ``freq`` does not list, or lists at the clock code, runs at
    the clock. Without a known clock (a ``RangeIndex`` without ``clock``, or
    an index frequency outside the clock families) the panel must be
    single-frequency: every series listed at one code, or none listed.

    Raises
    ------
    DataValidationError
        For a series faster than the clock, or several frequencies without a
        known clock.
    """
    if not freq:
        return {}
    if clock is None:
        codes = {freq.get(name) for name in series_names}
        if len(codes) == 1:
            return {}
        if is_range:
            raise DataValidationError(
                f"{what}: mixed-frequency data on a RangeIndex needs an explicit clock (the frequency of one row), "
                f"for example clock='m'; got freq={dict(freq)}."
            )
        raise DataValidationError(
            f"{what}: the index frequency is not one of the clock frequencies {list(CLOCK_CODES)}, so series cannot "
            f"be sampled at other frequencies; got freq={dict(freq)}."
        )
    rank = FREQUENCY_HIERARCHY[clock]
    faster = [name for name in series_names if name in freq and FREQUENCY_HIERARCHY[freq[name]] < rank]
    if faster:
        raise DataValidationError(
            f"{what}: series {faster} are sampled faster than the clock {clock!r} "
            f"({[freq[n] for n in faster]}); every series must run at the clock or slower."
        )
    return {name: freq[name] for name in series_names if name in freq and FREQUENCY_HIERARCHY[freq[name]] > rank}


# ---------------------------------------------------------------------------
# Regular grids
# ---------------------------------------------------------------------------


def _reindex_hint(index: pd.Index, freqstr: str) -> str:
    first, last = _fmt_stamp(index[0]), _fmt_stamp(index[-1])
    maker = "period_range" if isinstance(index, pd.PeriodIndex) else "date_range"
    return f"df.reindex(pd.{maker}('{first}', '{last}', freq='{freqstr}'))"


def _gaps_error(index: pd.Index, missing: pd.Index, freqstr: str, what: str) -> DataValidationError:
    return DataValidationError(
        f"{what}: the time index has gaps: it lies on a {freqstr!r} grid but misses {len(missing)} stamp(s), "
        f"first {_fmt_stamps(missing)}. Rows must be consecutive clock periods: insert the missing rows as NaN "
        f"with {_reindex_hint(index, freqstr)}."
    )


def _intraday_step(stamps: pd.DatetimeIndex) -> Optional[pd.DateOffset]:
    """The smallest step of an index as a fixed-length grid (``h``, ``5min``), when it is shorter than a day.

    None for daily or coarser steps (the calendar grids cover those), and when
    that grid would exceed ``_MAX_STEP_GRID`` stamps.
    """
    steps = np.diff(stamps.asi8)
    steps = steps[steps > 0]
    if not steps.size:
        return None
    step = int(steps.min())
    day = pd.Timedelta(days=1).value
    if step >= day or (stamps.asi8[-1] - stamps.asi8[0]) // step + 1 > _MAX_STEP_GRID:
        return None
    return to_offset(pd.Timedelta(step))


def _search_grid(index: pd.DatetimeIndex, what: str) -> pd.DateOffset:
    """The coarsest calendar grid holding every stamp; raise naming the gaps, or the irregular stamps.

    The candidates are the calendar grids and, for intraday data, the index's
    smallest step. Gaps explain an index only on a grid at most twice its
    length: a finer grid holds any set of dates (every date is on the daily
    grid), so an index that needs one is reported as irregular, with the
    plausible grid that holds the most stamps. A plausible grid also has at
    least half as many stamps as the index: on a coarser one most rows would
    share a stamp (a daily grid for hourly data).
    """
    stamps = _ns(index)
    first, last = stamps[0], stamps[-1]
    n = len(stamps)
    candidates = [to_offset(alias) for alias in _CANDIDATE_GRIDS]
    step = _intraday_step(stamps)
    if step is not None:
        candidates.append(step)
    containing: Optional[tuple[int, pd.DateOffset]] = None
    closest: Optional[tuple[int, pd.DateOffset, np.ndarray]] = None
    for offset in candidates:
        grid = pd.date_range(first, last, freq=offset)
        on = stamps.isin(grid)
        if on.all() and (containing is None or len(grid) < containing[0]):
            containing = (len(grid), offset)
        if n <= 2 * len(grid) <= 4 * n and (closest is None or on.sum() > closest[2].sum()):
            closest = (len(grid), offset, on)
    if containing is not None and (containing[0] <= 2 * n or closest is None):
        offset = containing[1]
        grid = pd.date_range(first, last, freq=offset)
        if len(grid) == n:
            return offset
        raise _gaps_error(index, grid.difference(stamps), offset.freqstr, what)
    if closest is None:
        raise DataValidationError(
            f"{what}: the time index is irregular: pandas infers no frequency and the stamps lie on no calendar "
            "grid. Rows must be consecutive clock periods: put them on a regular grid (pd.date_range) first."
        )
    _, offset, on = closest
    raise DataValidationError(
        f"{what}: the time index is irregular: pandas infers no frequency and the stamps lie on no single calendar "
        f"grid. The closest grid is {offset.freqstr!r} ({int(on.sum())} of {len(on)} stamps on it); off it: "
        f"{_fmt_stamps(index[~on])}. Rows must be consecutive clock periods: move each value to its grid stamp, "
        f"then {_reindex_hint(index, offset.freqstr)}."
    )


def grid_freq(index: pd.Index, *, what: str = "data") -> Optional[pd.DateOffset]:
    """The step of a regular index: rows must be consecutive clock periods (spec section 4.12).

    - ``RangeIndex``: always regular; returns None.
    - ``PeriodIndex``: the periods must be consecutive; returns ``index.freq``.
    - ``DatetimeIndex``: ``index.freq``, else ``pd.infer_freq``; without
      either, the coarsest calendar grid that holds every stamp, which must
      then equal the index.

    Raises
    ------
    DataValidationError
        For a gap (naming the first missing stamps and the
        ``df.reindex(pd.date_range(first, last, freq=...))`` fix), an
        irregular index (naming the closest grid and the stamps off it), or
        an index of another type.
    """
    if isinstance(index, pd.RangeIndex):
        return None
    if isinstance(index, pd.PeriodIndex):
        expected = pd.period_range(index[0], index[-1], freq=index.freq)
        if len(expected) != len(index):
            raise _gaps_error(index, expected.difference(index), index.freqstr, what)
        return index.freq
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(
            f"{what}: the time index must be a DatetimeIndex, PeriodIndex or RangeIndex, got {type(index).__name__}."
        )
    if index.freq is not None:
        return index.freq
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred is not None:
            return to_offset(inferred)
    if len(index) == 1:
        raise DataValidationError(
            f"{what}: a single time stamp without a frequency has no clock; give the DatetimeIndex a freq "
            "(for example pd.date_range(..., freq='ME')) or pass more rows."
        )
    return _search_grid(index, what)


def check_on_grid(index: pd.Index, freq: Optional[pd.DateOffset], *, what: str = "data") -> None:
    """Check that ``index`` is consecutive stamps of the fixed grid ``freq`` (the fitted clock grid).

    Raises
    ------
    DataValidationError
        For a stamp off the grid or a gap, naming the stamps and the fix.
    """
    if isinstance(index, pd.RangeIndex):
        return
    if isinstance(index, pd.PeriodIndex):
        grid_freq(index, what=what)
        return
    stamps = _ns(index)
    expected = pd.date_range(stamps[0], stamps[-1], freq=freq)
    off = stamps[~stamps.isin(expected)]
    if not freq.is_on_offset(stamps[0]) or len(off):
        off = off if len(off) else stamps[:1]
        raise DataValidationError(
            f"{what}: the stamps {_fmt_stamps(off)} are not on the model's {freq.freqstr!r} grid. Every row must be "
            f"a clock period: move each value to its grid stamp, then {_reindex_hint(index, freq.freqstr)}."
        )
    if len(expected) != len(stamps):
        raise _gaps_error(index, expected.difference(stamps), freq.freqstr, what)


# ---------------------------------------------------------------------------
# Calendar periods
# ---------------------------------------------------------------------------


def _calendar_keys(times: pd.DatetimeIndex, code: str, weekly: str = "W-SUN") -> np.ndarray:
    """Integer period keys (pandas period ordinals; half-years counted from 1970) of wall-clock ``times``."""
    times = _wall(times)
    year, month = times.year.to_numpy(np.int64) - 1970, times.month.to_numpy(np.int64) - 1
    if code == "m":
        return year * 12 + month
    if code == "q":
        return year * 4 + month // 3
    if code == "sa":
        return year * 2 + month // 6
    if code == "a":
        return year
    if code == "w":
        return np.asarray(times.to_period(weekly).asi8, dtype=np.int64)
    return np.asarray(times.to_period("D").asi8, dtype=np.int64)


def _calendar_label(key: int, code: str, weekly: str = "W-SUN") -> pd.Period:
    if code == "sa":
        return pd.Period(year=1970 + key // 2, month=1 + 6 * (key % 2), freq="6M")
    freq = {"d": "D", "w": weekly, "m": "M", "q": "Q-DEC", "a": "Y-DEC"}[code]
    return pd.Period(ordinal=int(key), freq=freq)


def _as_period(value: Any, role: str) -> pd.Period:
    """``value`` (a str or ``pd.Period``) as a period; a str is parsed with ``pd.Period(value)``."""
    if isinstance(value, pd.Period):
        return value
    try:
        return pd.Period(value)
    except (TypeError, ValueError) as err:
        raise ConfigurationError(
            f"{role}={value!r} is not a period label such as '2025Q3', '2025-09' or '2025' ({err})."
        ) from err


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_stamp(value: Any) -> bool:
    return isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64))


def _as_timestamp(value: Any, tz: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if tz is not None and stamp.tz is None:
        return stamp.tz_localize(tz)
    if tz is None and stamp.tz is not None:
        return stamp.tz_convert(None)
    return stamp


# ---------------------------------------------------------------------------
# The conditioning grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NowcastPeriod:
    """One per-series period of ``nowcast``: its label and first and last grid positions."""

    key: int
    label: Any
    first: int
    last: int

    def status(self, origin: int, released: bool) -> str:
        """The ``nowcast`` status (spec section 4.8) with the origin at grid position ``origin``.

        ``"released"`` when the value is observed; otherwise, with ``b <= e`` the
        period's first and last grid stamps, ``"backcast"`` when ``e < T``,
        ``"nowcast"`` when ``b <= T <= e`` (so the period ending exactly at the
        origin is the nowcast) and ``"forecast"`` when ``b > T``.
        """
        if released:
            return "released"
        if self.last < origin:
            return "backcast"
        if self.first <= origin:
            return "nowcast"
        return "forecast"


class Grid:
    """The grid of a forecaster: the conditioning index, its step and the clock.

    Parameters
    ----------
    index : pandas.Index
        The conditioning index (``index_``), regular.
    freq : pandas.DateOffset, optional
        The step of a ``DatetimeIndex`` (:func:`grid_freq`); ignored for a
        ``PeriodIndex`` (its own frequency) and a ``RangeIndex`` (its step).
    clock : str, optional
        The clock code (:func:`resolve_clock`); None when the index frequency
        is outside the clock families or a ``RangeIndex`` has no clock.
    """

    def __init__(self, index: pd.Index, freq: Optional[pd.DateOffset] = None, clock: Optional[str] = None) -> None:
        self.index = index
        self.clock = clock
        self.n = len(index)
        self.name = index.name if index.name is not None else "time"
        if isinstance(index, pd.PeriodIndex):
            self.freq = index.freq
        elif isinstance(index, pd.DatetimeIndex):
            self.freq = index.freq if freq is None else freq
            if self.freq is None:
                raise ValueError("Grid needs the step of a DatetimeIndex (grid_freq).")
        else:
            self.freq = None
        self.is_range = isinstance(index, pd.RangeIndex)

    # Stamps

    def future(self, k: int) -> pd.Index:
        """The ``k`` grid stamps after the origin."""
        last = self.index[-1]
        if self.is_range:
            step = self.index.step
            return pd.RangeIndex(last + step, last + step * (k + 1), step, name=self.name)
        if isinstance(self.index, pd.PeriodIndex):
            return pd.period_range(last + 1, periods=k, freq=self.freq, name=self.name)
        out = pd.date_range(last, periods=k + 1, freq=self.freq, name=self.name)[1:]
        return out.as_unit(self.index.unit)

    def stamps(self, first: int, last: int) -> pd.Index:
        """The grid stamps at positions ``first..last`` (both included), named like the index (or ``"time"``)."""
        if first < 0 or last < first:
            raise ValueError(f"invalid grid positions {first}..{last}")
        parts = []
        if first < self.n:
            parts.append(self.index[first:min(last, self.n - 1) + 1])
        if last >= self.n:
            parts.append(self.future(last - self.n + 1)[max(first, self.n) - self.n:])
        out = parts[0] if len(parts) == 1 else parts[0].append(parts[1])
        if isinstance(out, pd.DatetimeIndex) and len(out) > 1:
            out = pd.DatetimeIndex(out, freq=self.freq)
        return out.rename(self.name)

    def _times(self, first: int, last: int) -> pd.DatetimeIndex:
        """Representative times of positions ``first..last``: the stamp, or a period's end time."""
        stamps = self.stamps(first, last)
        return stamps.end_time if isinstance(stamps, pd.PeriodIndex) else stamps

    # Resolving end / start (spec section 4.4)

    def _last_at_or_before(self, time: pd.Timestamp) -> int:
        """Position of the last grid stamp whose time is <= ``time`` (-1 when there is none)."""
        times = self._times(0, self.n - 1)
        if time <= times[-1]:
            return int(times.searchsorted(time, side="right")) - 1
        k = 1
        while True:
            ahead = self._times(self.n, self.n + k - 1)
            if ahead[-1] > time:
                return self.n + int(ahead.searchsorted(time, side="right")) - 1
            k *= 2

    def _first_at_or_after(self, time: pd.Timestamp) -> int:
        """Position of the first grid stamp whose time is >= ``time``."""
        times = self._times(0, self.n - 1)
        if time <= times[-1]:
            return int(times.searchsorted(time, side="left"))
        k = 1
        while True:
            ahead = self._times(self.n, self.n + k - 1)
            if ahead[-1] >= time:
                return self.n + int(ahead.searchsorted(time, side="left"))
            k *= 2

    def _stamp_position(self, value: Any, role: str) -> int:
        """The position of an exact grid stamp (a ``pd.Timestamp`` or ``datetime``)."""
        if isinstance(self.index, pd.PeriodIndex):
            period = pd.Period(pd.Timestamp(value), freq=self.freq)
            return int(period.ordinal - self.index[0].ordinal)
        stamp = _as_timestamp(value, self.index.tz)
        pos = self._last_at_or_before(stamp)
        if pos < 0:
            # A grid stamp before the first one is a negative position: resolve() names it as such.
            steps = self._steps_before(stamp)
            if steps is not None:
                return -steps
        if pos < 0 or self._times(pos, pos)[0] != stamp:
            raise ConfigurationError(
                f"{role}={_fmt_stamp(stamp)} is not a stamp of the model's {self.freq.freqstr!r} grid "
                f"(the conditioning data runs {_fmt_stamp(self.index[0])} .. {_fmt_stamp(self.index[-1])})."
            )
        return pos

    def _steps_before(self, stamp: pd.Timestamp) -> Optional[int]:
        """Grid steps from ``stamp`` (earlier than the first stamp) to the first stamp; None when it is off the grid."""
        first = self.index[0]
        if isinstance(self.freq, pd.offsets.Tick):
            steps, rest = divmod(first - stamp, pd.Timedelta(self.freq))
            return int(steps) if rest == pd.Timedelta(0) else None
        back = pd.date_range(stamp, first, freq=self.freq)
        return len(back) - 1 if len(back) > 1 and back[0] == stamp and back[-1] == first else None

    def _period_position(self, value: Any, role: str) -> int:
        """``end``: the last grid stamp in the period; ``start``: the first."""
        period = _as_period(value, role)
        tz = getattr(self.index, "tz", None)
        lo, hi = _as_timestamp(period.start_time, tz), _as_timestamp(period.end_time, tz)
        if hi < self._times(0, 0)[0]:
            raise ConfigurationError(
                f"{role}={value!r}: the period {period} lies before the first conditioning stamp "
                f"{_fmt_stamp(self.index[0])}."
            )
        pos = self._last_at_or_before(hi) if role == "end" else self._first_at_or_after(lo)
        time = self._times(pos, pos)[0] if pos >= 0 else None
        if time is None or not lo <= time <= hi:
            raise ConfigurationError(f"{role}={value!r}: the period {period} holds no stamp of the model's grid.")
        return pos

    def position(self, value: Any, role: str) -> int:
        """The grid position ``end`` or ``start`` names (spec section 4.4).

        An int is an offset in grid steps from the origin; a ``pd.Timestamp`` or
        ``datetime`` is that exact grid stamp; a str or ``pd.Period`` names a
        period: as ``end`` its last grid stamp, as ``start`` its first. A
        ``RangeIndex`` takes ints only.

        Raises
        ------
        TypeError
            For any other type (anything but an int on a ``RangeIndex``).
        ConfigurationError
            For a stamp off the grid, or a period that holds no grid stamp.
        """
        if _is_int(value):
            return self.n - 1 + int(value)
        if self.is_range:
            raise TypeError(
                f"{role} must be an int (an offset in rows from the last conditioning row) on a RangeIndex, "
                f"got {type(value).__name__} {value!r}."
            )
        if isinstance(value, (str, pd.Period)):
            return self._period_position(value, role)
        if _is_stamp(value):
            return self._stamp_position(value, role)
        raise TypeError(
            f"{role} must be an int offset, a pd.Timestamp, a str or a pd.Period, got {type(value).__name__} {value!r}."
        )

    def resolve(self, end: Any, start: Any = None) -> tuple[int, int]:
        """The positions ``(first, last)`` of the targets ``start..end`` (spec section 4.4).

        ``start=None`` means the step after the origin when ``end`` lies after
        it, otherwise ``end`` itself.

        Raises
        ------
        ConfigurationError
            When ``start`` lies after ``end`` or before the first conditioning stamp.
        """
        last = self.position(end, "end")
        if start is None:
            first = self.n if last > self.n - 1 else last
        else:
            first = self.position(start, "start")
        if first > last:
            raise ConfigurationError(
                f"start={start!r} lies after end={end!r} ({self._describe(first)} > {self._describe(last)}): "
                "no target stamps."
            )
        if first < 0:
            # Without start, the first target is end itself.
            role, value = ("start", start) if start is not None else ("end", end)
            raise ConfigurationError(
                f"{role}={value!r} lies before the first conditioning stamp {_fmt_stamp(self.index[0])} "
                f"({-first} step(s) earlier)."
            )
        return first, last

    def _describe(self, pos: int) -> str:
        return _fmt_stamp(self.stamps(pos, pos)[0]) if pos >= 0 else f"position {pos}"

    # Periods of a series (masking and nowcast)

    def _absolute(self, first: int, last: int) -> np.ndarray:
        """``RangeIndex`` positions as absolute clock periods (``value // step``)."""
        values = np.asarray(self.stamps(first, last), dtype=np.int64)
        return values // self.index.step

    def keys(self, first: int, last: int, code: Optional[str] = None, residue: Optional[int] = None) -> np.ndarray:
        """Period keys of the positions ``first..last`` at ``code`` (None: the clock periods).

        Keys increase with the position; equal keys share a period. Calendar
        codes give pandas period ordinals. On a ``RangeIndex`` a clock period
        is a row, and a slower period ends at an absolute position ``p`` with
        ``p % k == residue`` (``k`` from :func:`periods_per`).
        """
        if self.is_range:
            pos = self._absolute(first, last)
            if code is None:
                return pos
            k = periods_per(code, self.clock)
            return -((residue - pos) // k)
        if code is None:
            if isinstance(self.index, pd.PeriodIndex):
                return np.asarray(self.stamps(first, last).asi8, dtype=np.int64)
            if self.clock is None:
                return np.arange(first, last + 1, dtype=np.int64)
            return _calendar_keys(self.stamps(first, last), self.clock, self._weekly())
        return _calendar_keys(self._times(first, last), code)

    def _weekly(self) -> str:
        """The weekly period of the clock series on a weekly clock (``W-FRI`` for a ``W-FRI`` grid)."""
        if isinstance(self.freq, pd.offsets.Week) and self.freq.weekday is not None:
            return f"W-{_WEEKDAYS[self.freq.weekday]}"
        return "W-SUN"

    def anchor_mask(self, first: int, last: int, code: str, residue: Optional[int] = None) -> np.ndarray:
        """True at the positions ``first..last`` that end a period of ``code``: the last grid stamp of each period."""
        keys = self.keys(first, last + 1, code, residue)
        return keys[:-1] != keys[1:]

    def _label(self, key: int, code: Optional[str], residue: Optional[int], last: int) -> Any:
        """The ``nowcast`` label of a period: a ``pd.Period``, or on a ``RangeIndex`` the period-end stamp.

        The ``RangeIndex`` label is the stamp value (the row label ``predict``
        gives that stamp), not its position from ``index[0]``: the two agree
        on ``RangeIndex(n)`` and differ on, say, ``RangeIndex(100, 180, 2)``.
        """
        if self.is_range:
            return self.stamps(last, last)[0]
        if code is not None:
            return _calendar_label(key, code)
        if isinstance(self.index, pd.PeriodIndex):
            return pd.Period(ordinal=int(key), freq=self.freq)
        if self.clock is not None:
            return _calendar_label(key, self.clock, self._weekly())
        stamp = self.stamps(last, last)
        try:
            return stamp.to_period(self.freq.freqstr)[0]
        except (TypeError, ValueError):
            return stamp.to_period("D")[0]

    def _label_key(self, value: Any, role: str, code: Optional[str]) -> int:
        """The period key a str or ``pd.Period`` label names, at ``code`` (None: the clock periods)."""
        period = _as_period(value, role)
        time = period.end_time if role == "end" else period.start_time
        if code is not None:
            return int(_calendar_keys(pd.DatetimeIndex([time]), code)[0])
        if isinstance(self.index, pd.PeriodIndex):
            return int(pd.Period(time, freq=self.freq).ordinal)
        if self.clock is not None:
            return int(_calendar_keys(pd.DatetimeIndex([time]), self.clock, self._weekly())[0])
        tz = self.index.tz
        lo, hi = _as_timestamp(period.start_time, tz), _as_timestamp(period.end_time, tz)
        return self._last_at_or_before(hi) if role == "end" else self._first_at_or_after(lo)

    def periods(
        self,
        code: Optional[str],
        end: Any,
        start: Any,
        *,
        residue: Optional[int] = None,
        observed: Optional[np.ndarray] = None,
    ) -> list[NowcastPeriod]:
        """The periods of a series at ``code`` (None: the clock) from ``start`` to ``end`` (spec section 4.8).

        ``end`` / ``start``: an int ``k`` means ``k`` periods after the period
        that holds the origin (0 is that period); a str or ``pd.Period`` names a
        period (as ``end`` the period holding the label's end, as ``start`` the
        one holding its start). ``start=None`` means the latest period with an
        observed value (``observed``: the series' observed in-sample positions),
        or the first conditioning period when there is none. A ``RangeIndex``
        takes ints only.

        Returns
        -------
        list of NowcastPeriod
            One per period, ascending, with its first and last grid positions;
            the first conditioning period may start before ``index[0]`` (its
            ``first`` is then 0).

        Raises
        ------
        TypeError
            For an ``end`` or ``start`` of another type.
        ConfigurationError
            When ``start`` lies after ``end`` or before the first conditioning period.
        """
        origin = int(self.keys(self.n - 1, self.n - 1, code, residue)[0])

        def key_of(value: Any, role: str) -> int:
            if _is_int(value):
                return origin + int(value)
            if self.is_range:
                raise TypeError(
                    f"{role} must be an int (periods after the one holding the origin) on a RangeIndex, "
                    f"got {type(value).__name__} {value!r}."
                )
            if isinstance(value, (str, pd.Period)):
                return self._label_key(value, role, code)
            raise TypeError(f"{role} must be an int, a str or a pd.Period, got {type(value).__name__} {value!r}.")

        last_key = key_of(end, "end")
        if start is None:
            observed = np.asarray([] if observed is None else observed, dtype=np.int64)
            if observed.size:
                first_key = int(self.keys(int(observed.max()), int(observed.max()), code, residue)[0])
            else:
                first_key = int(self.keys(0, 0, code, residue)[0])
        else:
            first_key = key_of(start, "start")
        if first_key > last_key:
            raise ConfigurationError(f"start={start!r} names a period after end={end!r}: no periods.")
        if first_key < int(self.keys(0, 0, code, residue)[0]):
            raise ConfigurationError(
                f"start={start!r} names a period before the first conditioning period (the data starts at "
                f"{_fmt_stamp(self.index[0])})."
            )
        hi = self.n - 1
        keys = self.keys(0, hi, code, residue)
        step = max(1, self.n)
        while keys[-1] <= last_key:
            keys = np.concatenate([keys, self.keys(hi + 1, hi + step, code, residue)])
            hi += step
            step *= 2
        out = []
        for key in range(first_key, last_key + 1):
            where = np.flatnonzero(keys == key)
            if where.size == 0:
                continue
            first, last = int(where[0]), int(where[-1])
            out.append(NowcastPeriod(key=key, label=self._label(key, code, residue, last), first=first, last=last))
        return out


# ---------------------------------------------------------------------------
# Slower-series anchors
# ---------------------------------------------------------------------------


def check_anchors(
    values: np.ndarray,
    series_names: Sequence[Any],
    slower: Mapping[Any, str],
    grid: Grid,
    *,
    residues: Optional[Mapping[Any, int]] = None,
    what: str = "data",
) -> dict:
    """Check that each slower series holds values only at the last grid stamp of its periods.

    Parameters
    ----------
    values : ndarray, shape (T, N)
        The data on ``grid.index`` (NaN where missing).
    series_names : sequence
        The column names of ``values``.
    slower : mapping
        ``{series: code}`` of the series slower than the clock.
    grid : Grid
        The grid of ``values``.
    residues : mapping, optional
        On a ``RangeIndex``: the anchor residue of each slower series fixed at
        fit; None infers them (all observations must share one residue).

    Returns
    -------
    dict
        ``{series: residue}`` on a ``RangeIndex``, else ``{series: None}``.

    Raises
    ------
    DataValidationError
        For a value off its series' anchors, naming the expected anchor stamps.
    """
    out: dict = {}
    position = {name: j for j, name in enumerate(series_names)}
    for name, code in slower.items():
        observed = np.flatnonzero(~np.isnan(values[:, position[name]]))
        word = _PERIOD_WORDS[code]
        if grid.is_range:
            k = periods_per(code, grid.clock)
            absolute = grid._absolute(0, grid.n - 1)[observed]
            found = sorted(set((absolute % k).tolist()))
            residue = None if residues is None else residues.get(name)
            if residue is None and len(found) > 1:
                raise DataValidationError(
                    f"{what}: series {name!r} is sampled at frequency {code!r} ({k} rows per {word}), so its values "
                    f"must sit one {word} apart; its observations fall at {len(found)} different positions within the "
                    f"{word} (rows mod {k}: {found}). Keep each value at its {word}'s last row."
                )
            if residue is None:
                residue = found[0] if found else k - 1
            bad = observed[(absolute % k) != residue]
            if bad.size:
                raise DataValidationError(
                    f"{what}: series {name!r} has values at rows {_fmt_stamps(grid.index[bad])}, but its {word}s end "
                    f"at rows with value // step % {k} == {residue}. Keep each value at its {word}'s last row."
                )
            out[name] = int(residue)
            continue
        anchors = grid.anchor_mask(0, grid.n - 1, code)
        bad = observed[~anchors[observed]]
        if bad.size:
            expected = grid.index[np.flatnonzero(anchors)]
            raise DataValidationError(
                f"{what}: series {name!r} is sampled at frequency {code!r}, so it may hold values only at the last "
                f"stamp of each {word} on this grid (for example {_fmt_stamps(expected[-3:], 3)}); it has values at "
                f"{_fmt_stamps(grid.index[bad])}. Move each value to its {word}'s last stamp."
            )
        out[name] = None
    return out


# ---------------------------------------------------------------------------
# Allocators and PanelData.between
# ---------------------------------------------------------------------------


def forward_stamp(index: pd.Index) -> Any:
    """The stamp after ``index[-1]``: ``index.freq`` (a ``CustomBusinessDay`` included), else the inferred frequency, else the median step.

    Raises
    ------
    DataValidationError
        For a single ``DatetimeIndex`` stamp without a frequency.
    """
    last = index[-1]
    if isinstance(index, pd.RangeIndex):
        return last + index.step
    if isinstance(index, pd.PeriodIndex):
        return last + 1
    freq = index.freq
    if freq is None and len(index) >= 3:
        inferred = pd.infer_freq(index)
        freq = to_offset(inferred) if inferred is not None else None
    if freq is not None:
        return last + freq
    if len(index) < 2:
        raise DataValidationError(
            "Cannot place the next stamp after a single time stamp without a frequency; give the DatetimeIndex a "
            "freq (for example pd.date_range(..., freq='B'))."
        )
    return last + index.to_series().diff().median()


def window(index: pd.Index, start: Any = None, end: Any = None) -> slice:
    """The rows of ``index`` from ``start`` to ``end``, both included (``PanelData.between``).

    A str or ``pd.Period`` covers its whole period: ``start`` takes the first
    stamp in it, ``end`` the last (a ``PeriodIndex`` stamp counts by its end
    time). A ``pd.Timestamp`` or ``datetime`` bound is compared with the
    stamps (on a ``PeriodIndex``: the period holding it). A ``RangeIndex``
    takes ints. None leaves that end open.

    Raises
    ------
    TypeError
        For a bound of another type.
    """
    lo, hi = 0, len(index)
    for role, value in (("start", start), ("end", end)):
        if value is None:
            continue
        if isinstance(index, pd.RangeIndex):
            if not _is_int(value):
                raise TypeError(f"{role} must be an int on a RangeIndex, got {type(value).__name__} {value!r}.")
            values = np.asarray(index)
            if role == "start":
                lo = max(lo, int(np.searchsorted(values, int(value), side="left")))
            else:
                hi = min(hi, int(np.searchsorted(values, int(value), side="right")))
            continue
        if isinstance(value, (str, pd.Period)):
            period = _as_period(value, role)
            bound = period.start_time if role == "start" else period.end_time
            times = index.end_time if isinstance(index, pd.PeriodIndex) else index
        elif _is_stamp(value):
            if isinstance(index, pd.PeriodIndex):
                bound, times = pd.Period(pd.Timestamp(value), freq=index.freq).end_time, index.end_time
            else:
                bound, times = pd.Timestamp(value), index
        else:
            raise TypeError(
                f"{role} must be a pd.Timestamp, a datetime, a str or a pd.Period, got {type(value).__name__} {value!r}."
            )
        bound = _as_timestamp(bound, getattr(times, "tz", None))
        if role == "start":
            lo = max(lo, int(times.searchsorted(bound, side="left")))
        else:
            hi = min(hi, int(times.searchsorted(bound, side="right")))
    return slice(lo, max(lo, hi))

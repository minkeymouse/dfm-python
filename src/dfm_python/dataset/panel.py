"""``PanelData``: the one data container every model's ``fit`` accepts.

A ``PanelData`` holds up to four slots that share one time index:

====================  ========================================================
Slot                  Type and shape
====================  ========================================================
``y``                 DataFrame, T x N: the target series (for AFM, asset
                      excess returns); NaN means missing
``exog``              DataFrame, T x K, or None: per-period inputs that are
                      not targets, known at the end of each period
``characteristics``   T x N x M, or None: per-period, per-series features
                      known at the end of each period (AFM); the property
                      is a frame with ``MultiIndex`` columns
                      ``(characteristic, series)``
``freq``              ``{series: code}``, or None: the sampling frequency of
                      the series slower than the model's clock
====================  ========================================================

The container is frozen and validated. It holds numpy and pandas data only
(polars frames are converted with ``.to_pandas()``), holds no device or
scaler, and never imputes, scales or reorders values: NaN stays NaN and every
value is cast to float64. ``characteristics`` are stored as one (T, N, M)
array; the property builds the frame on each access, so it can be large.

Index rules: unique, strictly increasing, and a ``DatetimeIndex``,
``PeriodIndex`` or ``RangeIndex``. An evenly spaced, increasing integer index is
stored as the equivalent ``RangeIndex``; arrays get a ``RangeIndex``. The
``from_*`` constructors also parse string time stamps with
:func:`pandas.to_datetime`; the plain constructor does not. Index regularity
(consecutive clock periods) is checked by the models, not the container.

``PanelData._merge`` implements the merge rule of ``update``: new rows,
revisions and late releases overwrite the conditioning panel cell by cell,
and a NaN in the new data never erases an observed value.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from .._time import window as _window
from ..config.constants import VALID_FREQUENCIES
from ..utils.errors import DataValidationError

__all__ = ["PanelData"]

_CHAR_LEVELS = ("characteristic", "series")
_LONG_ID, _LONG_TIME, _LONG_TARGET = "unique_id", "ds", "y"
_SLOTS = ("_y", "_exog", "_chars", "_freq", "_series_names", "_exog_names", "_characteristic_names")


# Input coercion helpers


def _to_pandas(obj: Any) -> Any:
    """Convert a polars DataFrame to pandas; return anything else unchanged.

    polars is never imported here: a polars frame can only exist if polars is
    already in ``sys.modules``.
    """
    pl = sys.modules.get("polars")
    if pl is not None and isinstance(obj, pl.DataFrame):
        return obj.to_pandas()
    return obj


def _polars_time_col(obj: Any) -> Optional[str]:
    """The one ``Date`` or ``Datetime`` column of a polars DataFrame.

    None when ``obj`` is not a polars frame or has no such column. Several
    such columns raise :class:`DataValidationError`.
    """
    pl = sys.modules.get("polars")
    if pl is None or not isinstance(obj, pl.DataFrame):
        return None
    temporal = [name for name, dtype in obj.schema.items() if isinstance(dtype, (pl.Date, pl.Datetime))]
    if len(temporal) > 1:
        raise DataValidationError(
            f"The polars frame has several Date/Datetime columns {temporal}; name the time index with "
            "PanelData.from_wide(df, time_col=...)."
        )
    return temporal[0] if temporal else None


def _require_frame(obj: Any, what: str) -> pd.DataFrame:
    obj = _to_pandas(obj)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{what} must be a pandas (or polars) DataFrame, got {type(obj).__name__}.")
    return obj


def _parse_time(values: Any, name: Optional[Hashable]) -> pd.Index:
    """Turn time stamps into an index; strings are parsed with ``pd.to_datetime``.

    Used by the ``from_*`` constructors only. The result still goes through
    :func:`_validate_index`.
    """
    idx = values if isinstance(values, pd.Index) else pd.Index(values)
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)) or pd.api.types.is_integer_dtype(idx.dtype):
        out = idx
    elif idx.dtype == object or pd.api.types.is_string_dtype(idx.dtype):
        try:
            out = pd.DatetimeIndex(pd.to_datetime(idx))
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Could not parse the time values of {name!r} as dates ({exc}). Pass datetimes, periods or "
                "evenly spaced integers."
            ) from exc
    else:
        out = idx
    return out.rename(name)


def _validate_index(idx: pd.Index, slot: str) -> pd.Index:
    """Check the index rules and return the index to store."""
    if isinstance(idx, pd.MultiIndex):
        raise DataValidationError(f"The index of {slot} is a MultiIndex; PanelData needs a single time index.")
    is_int = pd.api.types.is_integer_dtype(idx.dtype) and not isinstance(idx, pd.RangeIndex)
    if not (isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)) or is_int):
        raise DataValidationError(
            f"The index of {slot} must be a DatetimeIndex, PeriodIndex or RangeIndex, got {type(idx).__name__} "
            f"with dtype {idx.dtype}. Convert it (for example with pd.to_datetime), or use PanelData.from_wide, "
            "which parses string time stamps."
        )
    if len(idx) == 0:
        raise DataValidationError(f"{slot} has no rows.")
    if idx.hasnans:
        raise DataValidationError(f"The index of {slot} contains missing time stamps (NaT/NaN).")
    if not idx.is_unique:
        dupes = idx[idx.duplicated()].unique()[:5].tolist()
        raise DataValidationError(f"The index of {slot} has duplicate time stamps, for example {dupes}.")
    if not idx.is_monotonic_increasing:
        raise DataValidationError(f"The index of {slot} must be increasing; sort the rows by time first.")
    if is_int:
        values = idx.to_numpy()
        step = int(values[1] - values[0]) if len(values) > 1 else 1
        if len(values) > 1 and not np.all(np.diff(values) == step):
            raise DataValidationError(
                f"The integer index of {slot} is not evenly spaced; use a RangeIndex, a DatetimeIndex or a "
                "PeriodIndex."
            )
        idx = pd.RangeIndex(int(values[0]), int(values[-1]) + step, step, name=idx.name)
    return idx


def _ns(index: pd.Index) -> pd.Index:
    """A ``DatetimeIndex`` at nanosecond resolution (indexes at other units do not compare equal)."""
    if isinstance(index, pd.DatetimeIndex) and index.unit != "ns":
        return index.as_unit("ns")
    return index


def _same_index(a: pd.Index, b: pd.Index) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, pd.DatetimeIndex) and a.dtype != b.dtype and a.tz == b.tz:
        # Same instants stored at different resolutions (e.g. polars gives microseconds).
        return _ns(a).equals(_ns(b))
    return a.equals(b)


def _validate_flat_columns(columns: pd.Index, slot: str) -> tuple:
    if isinstance(columns, pd.MultiIndex):
        raise DataValidationError(
            f"{slot} has MultiIndex columns; only characteristics use a (characteristic, series) MultiIndex."
        )
    if not columns.is_unique:
        dupes = columns[columns.duplicated()].unique().tolist()
        raise DataValidationError(f"Column names of {slot} must be unique; duplicated: {dupes}.")
    return tuple(columns.tolist())


def _float_values(frame: pd.DataFrame, slot: str) -> np.ndarray:
    """Return a float64 copy of the frame's values; reject non-numeric columns."""
    bad = [
        col for col, dtype in frame.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
    ]
    if bad:
        raise DataValidationError(f"{slot} must hold real numbers only; non-numeric columns: {bad[:10]}.")
    return np.array(frame.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True)


def _normalize_freq(freq: Optional[Mapping], series_names: Sequence[Hashable]) -> Optional[dict]:
    """Check ``freq`` (``{series: code}``) and order it like ``series_names``; an empty mapping gives None.

    Codes are checked against ``dfm_python.config.constants.VALID_FREQUENCIES``.
    """
    if freq is None:
        return None
    if not isinstance(freq, Mapping):
        raise TypeError(f"freq must be a dict {{series: code}} or None, got {type(freq).__name__}.")
    if len(freq) == 0:
        return None
    grouped = [code for code, value in freq.items() if isinstance(value, Iterable) and not isinstance(value, str)]
    if grouped:
        raise DataValidationError(
            f"freq maps each series to its frequency code, for example {{'gdp': 'q'}}; got lists under {grouped}. "
            "List each series with its code instead."
        )
    bad = sorted({repr(c) for c in freq.values() if not isinstance(c, str) or c not in VALID_FREQUENCIES})
    if bad:
        raise DataValidationError(
            f"Unknown frequency code(s) {', '.join(bad)}. Valid codes: {sorted(VALID_FREQUENCIES)}."
        )
    unknown = [name for name in freq if name not in set(series_names)]
    if unknown:
        raise DataValidationError(f"freq names unknown series {unknown}; the series are {list(series_names)}.")
    return {name: freq[name] for name in series_names if name in freq} or None


def _names_list(names: Sequence[Any]) -> list:
    return list(names)


# PanelData


class PanelData:
    """Frozen, validated panel of time series shared by all models.

    Parameters
    ----------
    y : pandas.DataFrame
        Target series, shape (T, N). Its index is the panel's time index.
    exog : pandas.DataFrame, optional
        Per-period inputs that are not targets, shape (T, K), on the same index.
    characteristics : pandas.DataFrame, optional
        Per-period, per-series features, on the same index. The columns are a
        two-level ``MultiIndex`` named ``("characteristic", "series")``
        (unnamed levels are read in that order) whose ``series`` level holds
        exactly the columns of ``y``, every characteristic once per series.
    freq : dict, optional
        ``{series: code}`` with codes ``d, w, m, q, sa, a``: the sampling
        frequency of the series slower than the model's clock. Series that are
        not listed run at the clock.

    Notes
    -----
    An ``exog`` or ``characteristics`` frame with no columns is stored as None.
    Values are copied and cast to float64; NaN is kept. Nothing is imputed,
    scaled or reordered. Attribute assignment raises ``AttributeError``, and
    the ``y``, ``exog``, ``characteristics`` and ``freq`` accessors return copies.
    """

    __slots__ = _SLOTS

    def __init__(
        self,
        y: pd.DataFrame,
        *,
        exog: Optional[pd.DataFrame] = None,
        characteristics: Optional[pd.DataFrame] = None,
        freq: Optional[Mapping] = None,
    ) -> None:
        y = _require_frame(y, "y")
        index = _validate_index(y.index, "y")
        if y.shape[1] == 0:
            raise DataValidationError("y has no columns; PanelData needs at least one target series.")
        series_names = _validate_flat_columns(y.columns, "y")
        y_values = _float_values(y, "y")

        exog_values, exog_names = None, ()
        if exog is not None:
            exog = _require_frame(exog, "exog")
        if exog is not None and exog.shape[1] > 0:
            exog_names = _validate_flat_columns(exog.columns, "exog")
            self._check_shared_index(exog, index, "exog")
            exog_values = _float_values(exog, "exog")

        char_values, char_names = None, ()
        if characteristics is not None:
            characteristics = _require_frame(characteristics, "characteristics")
        if characteristics is not None and characteristics.shape[1] > 0:
            self._check_shared_index(characteristics, index, "characteristics")
            columns, char_names = self._validate_characteristic_columns(characteristics.columns, series_names)
            flat = _float_values(characteristics.set_axis(columns, axis=1), "characteristics")
            full = pd.MultiIndex.from_product([list(char_names), list(series_names)], names=list(_CHAR_LEVELS))
            ordered = flat[:, columns.get_indexer(full)]
            char_values = ordered.reshape(len(index), len(char_names), len(series_names)).transpose(0, 2, 1).copy()

        self._set(index, y_values, series_names, exog_values, exog_names, char_values, char_names, freq)

    def _set(
        self,
        index: pd.Index,
        y: np.ndarray,
        series_names: tuple,
        exog: Optional[np.ndarray],
        exog_names: tuple,
        chars: Optional[np.ndarray],
        char_names: tuple,
        freq: Optional[Mapping],
    ) -> None:
        """Store the validated slots (arrays already float64 copies in the given name order)."""
        overlap = [name for name in exog_names if name in set(series_names)]
        if overlap:
            raise DataValidationError(f"Names used both as targets (y) and as exog: {overlap}.")
        overlap = [name for name in char_names if name in set(exog_names)]
        if overlap:
            raise DataValidationError(f"Names used both as exog and as characteristics: {overlap}.")
        columns = pd.Index(list(series_names), tupleize_cols=False)
        y_frame = pd.DataFrame(y, index=index, columns=columns, copy=False)
        exog_frame = None
        if exog is not None and len(exog_names):
            exog_frame = pd.DataFrame(
                exog, index=index, columns=pd.Index(list(exog_names), tupleize_cols=False), copy=False
            )
        if chars is None or not len(char_names):
            chars, char_names = None, ()
        norm_freq = _normalize_freq(freq, series_names)
        values = (y_frame, exog_frame, chars, norm_freq, tuple(series_names), tuple(exog_names), tuple(char_names))
        for slot, value in zip(_SLOTS, values):
            object.__setattr__(self, slot, value)

    @classmethod
    def _from_arrays_checked(
        cls,
        index: pd.Index,
        y: np.ndarray,
        series_names: Sequence[Any],
        exog: Optional[np.ndarray],
        exog_names: Sequence[Any],
        chars: Optional[np.ndarray],
        char_names: Sequence[Any],
        freq: Optional[Mapping],
    ) -> PanelData:
        """A panel from float64 arrays the caller owns (no copy), with the index validated."""
        out = cls.__new__(cls)
        out._set(
            _validate_index(index, "y"), y, tuple(series_names), exog, tuple(exog_names), chars, tuple(char_names), freq
        )
        return out

    @staticmethod
    def _check_shared_index(frame: pd.DataFrame, index: pd.Index, slot: str) -> None:
        other = _validate_index(frame.index, slot)
        if not _same_index(other, index):
            raise DataValidationError(
                f"{slot} must share the index of y exactly (same type, same time stamps in the same order); "
                f"y has {len(index)} rows, {slot} has {len(other)}."
            )

    @staticmethod
    def _validate_characteristic_columns(columns: pd.Index, series_names: tuple) -> tuple[pd.MultiIndex, tuple]:
        if not isinstance(columns, pd.MultiIndex) or columns.nlevels != 2:
            raise DataValidationError(
                "characteristics need a two-level column MultiIndex named ('characteristic', 'series'); build it "
                "with pd.MultiIndex.from_product([characteristics, series], names=['characteristic', 'series']) or "
                "use PanelData.from_arrays."
            )
        names = tuple(columns.names)
        if names not in ((None, None), _CHAR_LEVELS):
            raise DataValidationError(
                f"characteristics column levels must be named ('characteristic', 'series') in that order, got {names}."
            )
        if not columns.is_unique:
            dupes = columns[columns.duplicated()].unique().tolist()[:5]
            raise DataValidationError(
                f"characteristics have duplicated (characteristic, series) columns, for example {dupes}."
            )
        chars = tuple(pd.unique(columns.get_level_values(0)).tolist())
        level_series = set(columns.get_level_values(1))
        missing = [s for s in series_names if s not in level_series]
        extra = sorted(repr(s) for s in level_series - set(series_names))
        if missing or extra:
            raise DataValidationError(
                f"The 'series' level of characteristics must match the columns of y; missing {missing}, "
                f"unknown {extra}."
            )
        if len(columns) != len(chars) * len(series_names):
            raise DataValidationError(
                f"characteristics must hold every characteristic for every series: {len(chars)} characteristics x "
                f"{len(series_names)} series = {len(chars) * len(series_names)} columns, got {len(columns)}."
            )
        return columns.set_names(list(_CHAR_LEVELS)), chars

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"PanelData is immutable; cannot set {name!r}. Build a new PanelData instead.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"PanelData is immutable; cannot delete {name!r}.")

    def __getstate__(self) -> dict:
        return {slot: getattr(self, slot) for slot in _SLOTS}

    def __setstate__(self, state: dict) -> None:
        for slot in _SLOTS:
            object.__setattr__(self, slot, state[slot])

    # Constructors

    @classmethod
    def from_wide(
        cls,
        df: pd.DataFrame,
        *,
        time_col: Optional[Hashable] = None,
        y: Optional[Sequence[Hashable]] = None,
        exog: Optional[Sequence[Hashable]] = None,
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from one wide frame: one row per period, one column per variable.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
        time_col : column name, optional
            Column holding the time stamps; None uses ``df.index``. String time
            stamps are parsed with :func:`pandas.to_datetime`.
        y : list of column names, optional
            Target series; None means every column that is neither
            ``time_col`` nor in ``exog``.
        exog : list of column names, optional
            Per-period inputs that are not targets.
        freq : dict, optional
            See :class:`PanelData`.
        """
        df = _require_frame(df, "df")
        if not df.columns.is_unique:
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise DataValidationError(f"Column names of df must be unique; duplicated: {dupes}.")
        if time_col is None:
            index = _parse_time(df.index, df.index.name)
            body = df
        else:
            cls._require_columns(df, [time_col], "time_col")
            index = _parse_time(df[time_col], time_col)
            body = df.drop(columns=[time_col])

        exog_cols = cls._as_name_list(exog, "exog")
        cls._require_columns(body, exog_cols, "exog")
        if y is None:
            excluded = set(exog_cols)
            target_cols = [c for c in body.columns if c not in excluded]
        else:
            target_cols = cls._as_name_list(y, "y")
            cls._require_columns(body, target_cols, "y")
            overlap = [c for c in target_cols if c in set(exog_cols)]
            if overlap:
                raise DataValidationError(f"Columns listed in both y and exog: {overlap}.")
        if not target_cols:
            raise DataValidationError("from_wide found no target columns.")

        y_frame = body[target_cols].set_axis(index, axis=0)
        exog_frame = body[exog_cols].set_axis(index, axis=0) if exog_cols else None
        return cls(y_frame, exog=exog_frame, freq=freq)

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        id_col: Hashable = _LONG_ID,
        time_col: Hashable = _LONG_TIME,
        y: Hashable = _LONG_TARGET,
        exog: Sequence[Hashable] = (),
        characteristics: Sequence[Hashable] = (),
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from a long frame with one row per ``(id, time)`` pair.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
        id_col, time_col, y : column names
            The series id, the time stamp and the target value.
        exog : list of column names
            Per-period inputs; they must be constant across ids at each time
            (NaN entries count as missing), otherwise
            :class:`DataValidationError` is raised.
        characteristics : list of column names
            Per-period, per-series features.
        freq : dict, optional
            See :class:`PanelData`.

        Notes
        -----
        Series are ordered by first appearance of their id and time stamps are
        sorted. A missing ``(id, time)`` row becomes NaN. Columns not named in
        any argument are ignored.
        """
        df = _require_frame(df, "df")
        char_cols = cls._as_name_list(characteristics, "characteristics")
        exog_cols = cls._as_name_list(exog, "exog")
        roles = [id_col, time_col, y, *char_cols, *exog_cols]
        repeated = sorted({repr(c) for c in roles if roles.count(c) > 1})
        if repeated:
            raise DataValidationError(f"Each column can have one role only; used more than once: {repeated}.")
        cls._require_columns(df, roles, "from_long")
        if not df.columns.is_unique:
            dupes = df.columns[df.columns.duplicated()].unique().tolist()
            raise DataValidationError(f"Column names of df must be unique; duplicated: {dupes}.")
        if df[id_col].isna().any():
            raise DataValidationError(f"{id_col!r} contains missing ids.")

        long = df[roles].copy()
        long[time_col] = _parse_time(long[time_col], time_col)
        dupes = long.duplicated([id_col, time_col])
        if dupes.any():
            first = long.loc[dupes, [id_col, time_col]].iloc[0].tolist()
            raise DataValidationError(f"from_long found duplicated ({id_col!r}, {time_col!r}) rows, e.g. {first}.")

        ids = pd.Index(pd.unique(long[id_col]), tupleize_cols=False)
        index = _parse_time(pd.Index(pd.unique(long[time_col])).sort_values(), time_col)

        def wide(column: Hashable) -> pd.DataFrame:
            out = long.pivot(index=time_col, columns=id_col, values=column).reindex(index=index, columns=ids)
            out.columns.name = None
            return out

        y_frame = wide(y)
        char_frame = None
        if char_cols:
            char_frame = pd.concat([wide(col) for col in char_cols], axis=1, keys=char_cols, names=list(_CHAR_LEVELS))
        exog_frame = None
        if exog_cols:
            values = pd.DataFrame(_float_values(long[exog_cols], "exog"), index=long.index, columns=pd.Index(exog_cols))
            values[time_col] = long[time_col]
            grouped = values.groupby(time_col, sort=True)
            varying = grouped.nunique(dropna=True) > 1
            if varying.to_numpy().any():
                col = varying.columns[varying.any(axis=0).to_numpy()][0]
                when = varying.index[varying[col].to_numpy()][0]
                raise DataValidationError(
                    f"exog column {col!r} differs across ids at {time_col}={when!r}; a per-series variable "
                    "belongs in characteristics."
                )
            exog_frame = grouped.first().reindex(index)
            exog_frame.columns.name = None
        return cls(y_frame, exog=exog_frame, characteristics=char_frame, freq=freq)

    @classmethod
    def from_arrays(
        cls,
        y: Any,
        *,
        exog: Any = None,
        characteristics: Any = None,
        index: Any = None,
        series_names: Optional[Sequence[Hashable]] = None,
        exog_names: Optional[Sequence[Hashable]] = None,
        characteristic_names: Optional[Sequence[Hashable]] = None,
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from arrays: ``y`` (T, N), ``exog`` (T, K), ``characteristics`` (T, N, M).

        ``index=None`` gives a ``RangeIndex``. Default names are ``series_0..``,
        ``exog_0..`` and ``characteristic_0..``.
        """
        y_arr = cls._as_array(y, 2, "y", "(T, N)")
        n_time, n_series = y_arr.shape
        if index is None:
            time_index = pd.RangeIndex(n_time)
        else:
            index = _to_pandas(index)
            time_index = _parse_time(index, getattr(index, "name", None))
            if len(time_index) != n_time:
                raise DataValidationError(f"index has {len(time_index)} entries but y has {n_time} rows.")
        series = cls._names(series_names, n_series, "series", "series_names")
        if len(set(series)) != len(series):
            raise DataValidationError(f"series_names must be unique, got {list(series)}.")

        x_arr, x_names = None, ()
        if exog is not None:
            x_arr = cls._as_array(exog, 2, "exog", "(T, K)")
            if x_arr.shape[0] != n_time:
                raise DataValidationError(f"exog has {x_arr.shape[0]} rows but y has {n_time}.")
            x_names = tuple(cls._names(exog_names, x_arr.shape[1], "exog", "exog_names"))
            if len(set(x_names)) != len(x_names):
                raise DataValidationError(f"exog_names must be unique, got {list(x_names)}.")

        c_arr, c_names = None, ()
        if characteristics is not None:
            c_arr = cls._as_array(characteristics, 3, "characteristics", "(T, N, M)")
            if c_arr.shape[:2] != (n_time, n_series):
                raise DataValidationError(
                    f"characteristics have shape {c_arr.shape}; expected ({n_time}, {n_series}, M) to match y."
                )
            c_names = tuple(cls._names(characteristic_names, c_arr.shape[2], "characteristic", "characteristic_names"))
            if len(set(c_names)) != len(c_names):
                raise DataValidationError(f"characteristic_names must be unique, got {list(c_names)}.")
        return cls._from_arrays_checked(
            time_index, y_arr.copy(), tuple(series), None if x_arr is None else x_arr.copy(), x_names,
            None if c_arr is None else c_arr.copy(), c_names, freq,
        )

    @classmethod
    def _coerce(cls, obj: Any) -> PanelData:
        """Turn what ``fit`` and ``update`` accept into a ``PanelData``.

        A ``PanelData`` is returned as is, a pandas DataFrame means
        ``PanelData(y=df)`` and a 2-D ndarray means ``from_arrays(arr)``. A
        polars DataFrame with one ``Date`` or ``Datetime`` column means
        ``from_wide(df, time_col=<that column>)``; without one it means
        ``PanelData(y=df)`` (a ``RangeIndex``); with several it raises
        :class:`DataValidationError`. Anything else raises ``TypeError``.
        """
        if isinstance(obj, PanelData):
            return obj
        time_col = _polars_time_col(obj)
        if time_col is not None:
            return cls.from_wide(obj, time_col=time_col)
        obj = _to_pandas(obj)
        if isinstance(obj, pd.DataFrame):
            return cls(obj)
        if isinstance(obj, np.ndarray):
            if obj.ndim != 2:
                raise DataValidationError(
                    f"Expected a 2-D array of shape (T, N), got {obj.ndim}-D with shape {obj.shape}. Reshape a "
                    "single series with arr.reshape(-1, 1)."
                )
            return cls.from_arrays(obj)
        raise TypeError(
            f"Expected PanelData, a DataFrame or a 2-D numpy array, got {type(obj).__name__}. "
            "Wrap other inputs with PanelData(...) or PanelData.from_arrays(...)."
        )

    @staticmethod
    def _as_name_list(cols: Any, what: str) -> list:
        if cols is None:
            return []
        if isinstance(cols, (str, bytes)) or not isinstance(cols, Iterable):
            raise TypeError(f"{what} must be a list of column names, got {type(cols).__name__}.")
        cols = list(cols)
        dupes = sorted({repr(c) for c in cols if cols.count(c) > 1})
        if dupes:
            raise DataValidationError(f"{what} lists columns more than once: {dupes}.")
        return cols

    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: Sequence[Hashable], what: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DataValidationError(f"{what}: columns not found in df: {missing}.")

    @staticmethod
    def _as_array(values: Any, ndim: int, what: str, shape: str) -> np.ndarray:
        values = _to_pandas(values)
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"{what} must be a numeric array of shape {shape} ({exc}).") from exc
        if arr.ndim != ndim:
            raise DataValidationError(f"{what} must have shape {shape}, got {arr.ndim}-D with shape {arr.shape}.")
        return arr

    @staticmethod
    def _names(names: Optional[Sequence[Hashable]], n: int, prefix: str, what: str) -> list:
        if names is None:
            return [f"{prefix}_{i}" for i in range(n)]
        if isinstance(names, (str, bytes)):
            raise TypeError(f"{what} must be a list of names, got a string.")
        names = list(names)
        if len(names) != n:
            raise DataValidationError(f"{what} has {len(names)} entries; expected {n}.")
        return names

    # Accessors

    @property
    def y(self) -> pd.DataFrame:
        """Target series, T x N (a copy)."""
        return self._y.copy()

    @property
    def exog(self) -> Optional[pd.DataFrame]:
        """Per-period inputs, T x K (a copy), or None."""
        return None if self._exog is None else self._exog.copy()

    @property
    def characteristics(self) -> Optional[pd.DataFrame]:
        """Per-series features as a frame with ``(characteristic, series)`` columns (built on each access), or None."""
        if self._chars is None:
            return None
        n_time, n_series, n_char = self._chars.shape
        columns = pd.MultiIndex.from_product(
            [list(self._characteristic_names), list(self._series_names)], names=list(_CHAR_LEVELS)
        )
        flat = self._chars.transpose(0, 2, 1).reshape(n_time, n_char * n_series)
        return pd.DataFrame(flat.copy(), index=self._y.index.copy(), columns=columns)

    @property
    def freq(self) -> Optional[dict]:
        """Per-series frequency codes in ``series_names`` order (a copy), or None."""
        return None if self._freq is None else dict(self._freq)

    @property
    def index(self) -> pd.Index:
        """The shared time index."""
        return self._y.index.copy()

    @property
    def series_names(self) -> tuple:
        """Target names, in column order."""
        return self._series_names

    @property
    def exog_names(self) -> tuple:
        """Exogenous variable names; empty without ``exog``."""
        return self._exog_names

    @property
    def characteristic_names(self) -> tuple:
        """Characteristic names, in order of first appearance; empty without ``characteristics``."""
        return self._characteristic_names

    @property
    def n_series(self) -> int:
        """Number of target series N."""
        return len(self._series_names)

    @property
    def n_periods(self) -> int:
        """Number of periods (rows) T."""
        return len(self._y.index)

    # Methods

    def between(self, start: Any = None, end: Any = None) -> PanelData:
        """The rows from ``start`` to ``end``, both included; None leaves that end open.

        A str or ``pd.Period`` covers its whole period: ``start`` takes the
        first stamp in it, ``end`` the last (``between("2020Q3", "2020Q4")``).
        A ``pd.Timestamp`` or ``datetime`` is compared with the stamps. A
        ``RangeIndex`` takes ints (labels, not positions).

        Raises
        ------
        TypeError
            For a bound of another type.
        DataValidationError
            When no row lies between the bounds.
        """
        rows = _window(self._y.index, start, end)
        if rows.stop <= rows.start:
            raise DataValidationError(f"between({start!r}, {end!r}) selects no rows.")
        return self._take(rows)

    def _take(self, rows: slice) -> PanelData:
        """The panel restricted to the positional slice ``rows``."""
        return type(self)._from_arrays_checked(
            self._y.index[rows],
            self._y.to_numpy()[rows].copy(),
            self._series_names,
            None if self._exog is None else self._exog.to_numpy()[rows].copy(),
            self._exog_names,
            None if self._chars is None else self._chars[rows].copy(),
            self._characteristic_names,
            self._freq,
        )

    def to_long(self) -> pd.DataFrame:
        """Long frame with columns ``unique_id``, ``ds``, ``y``, the exog names, then the characteristics.

        One row per ``(series, time)``, ordered by series then time; NaN
        targets are kept, so ``PanelData.from_long(panel.to_long(), exog=...,
        characteristics=...)`` rebuilds the panel.
        """
        reserved = {_LONG_ID, _LONG_TIME, _LONG_TARGET}
        clash = [c for c in (*self._exog_names, *self._characteristic_names) if c in reserved]
        if clash:
            raise DataValidationError(f"Cannot write a long frame: names {clash} clash with {sorted(reserved)}.")
        n_time, n_series = self.n_periods, self.n_series
        index = self._y.index
        out = pd.DataFrame(
            {
                _LONG_ID: np.repeat(np.asarray(self._series_names, dtype=object), n_time),
                _LONG_TIME: index.take(np.tile(np.arange(n_time), n_series)),
                _LONG_TARGET: self._y.to_numpy().T.reshape(-1),
            }
        )
        if self._exog is not None:
            tiled = np.tile(self._exog.to_numpy(), (n_series, 1))
            for j, name in enumerate(self._exog_names):
                out[name] = tiled[:, j]
        if self._chars is not None:
            arr = self._chars.transpose(1, 0, 2).reshape(n_series * n_time, -1)
            for j, name in enumerate(self._characteristic_names):
                out[name] = arr[:, j]
        return out

    # Private helpers for the estimators

    def _characteristic_values(self) -> Optional[np.ndarray]:
        """``characteristics`` as a float64 array (T, N, M) in ``series_names`` order (a copy), or None."""
        return None if self._chars is None else self._chars.copy()

    def _is_mixed_frequency(self) -> bool:
        """True if ``freq`` lists two or more frequencies, or lists some but not all series."""
        if not self._freq:
            return False
        return len(set(self._freq.values())) > 1 or len(self._freq) < len(self._series_names)

    def _equals(self, other: Any) -> bool:
        """True if ``other`` is a PanelData with the same index, names, freq and values (NaN equals NaN).

        Index names are not compared.
        """
        if not isinstance(other, PanelData):
            return False
        same_meta = (
            _same_index(self._y.index, other._y.index)
            and self._series_names == other._series_names
            and self._exog_names == other._exog_names
            and self._characteristic_names == other._characteristic_names
            and self._freq == other._freq
        )
        if not same_meta:
            return False
        pairs = [(self._y.to_numpy(), other._y.to_numpy())]
        if self._exog is not None:
            pairs.append((self._exog.to_numpy(), other._exog.to_numpy()))
        if self._chars is not None:
            pairs.append((self._chars, other._chars))
        return all(np.array_equal(a, b, equal_nan=True) for a, b in pairs)

    def _merge(self, new: Any, *, replace: bool = False, new_series: bool = False) -> PanelData:
        """This (conditioning) panel merged with ``new`` data: the merge rule of ``update`` (spec section 3).

        Parameters
        ----------
        new : PanelData or what :meth:`_coerce` accepts
            Must hold this panel's series, in any column order, with the same
            exog and characteristic names; ``freq`` must match or be None
            (inherit). Its index has this panel's type.
        replace : bool, default False
            False merges: the result runs from this panel's first stamp to the
            later of the two last stamps; every non-missing cell of ``new``
            overwrites this panel's cell at its stamp and every other cell
            keeps its value (NaN at new stamps), in every slot. A NaN in
            ``new`` never erases an observed value. ``new`` must start at or
            after this panel's first stamp, and its stamps up to this panel's
            last stamp must be stamps of this panel. True returns exactly
            ``new`` (columns in this panel's order).
        new_series : bool, default False
            Allow series this panel does not hold (AFM assets): they are
            appended in first-seen order, with NaN history.

        Raises
        ------
        DataValidationError
            For other series, exog or characteristic names, another ``freq``,
            another index type, or (``replace=False``) data that starts before
            this panel or has stamps inside it that it does not have.
        """
        new = PanelData._coerce(new)
        known = list(self._series_names)
        missing = [s for s in known if s not in set(new._series_names)]
        added = [s for s in new._series_names if s not in set(known)]
        if missing or (added and not new_series):
            parts = []
            if missing:
                parts.append(f"misses {missing}")
            if added and not new_series:
                parts.append(f"has series the model does not know: {added}")
            raise DataValidationError(
                f"The new data {' and '.join(parts)}; it must hold the series {known} (in any order)."
            )
        for what, ours, theirs in (
            ("exog", self._exog_names, new._exog_names),
            ("characteristics", self._characteristic_names, new._characteristic_names),
        ):
            if set(ours) != set(theirs):
                raise DataValidationError(
                    f"The new data has {what} {list(theirs)}, but the conditioning data has {list(ours)}; pass the "
                    "same names (in any order)."
                )
        if new._freq is not None and new._freq != (self._freq or None):
            raise DataValidationError(
                f"The new data has freq={new._freq}, but the conditioning data has freq={self._freq}; pass the same "
                "freq, or freq=None to keep it."
            )
        self._check_merge_index(new._y.index)

        order = known + added
        exog_order = list(self._exog_names)
        char_order = list(self._characteristic_names)
        new_y = new._y[order].to_numpy()
        new_x = None if new._exog is None else new._exog[exog_order].to_numpy()
        new_c = None
        if new._chars is not None:
            series_pos = [new._series_names.index(s) for s in order]
            char_pos = [new._characteristic_names.index(c) for c in char_order]
            new_c = new._chars[np.ix_(np.arange(new.n_periods), series_pos, char_pos)]
        if replace:
            return type(self)._from_arrays_checked(
                new._y.index, new_y.copy(), order, None if new_x is None else new_x.copy(), exog_order,
                None if new_c is None else new_c.copy(), char_order, self._freq,
            )

        index = self._y.index
        new_index = new._y.index
        if new_index[0] < index[0]:
            raise DataValidationError(
                f"The new data starts at {new_index[0]!r}, before the conditioning data ({index[0]!r}). Pass data "
                "from that stamp on, or use update(data, replace=True) to condition on exactly this data."
            )
        inside = new_index[new_index <= index[-1]]
        unknown = _ns(inside).difference(_ns(index))
        if len(unknown):
            raise DataValidationError(
                f"The new data has stamps {list(unknown[:5])} inside the conditioning period that the conditioning "
                "data does not have; stamps up to its last one must be its own (new rows come after it)."
            )
        after = new_index[new_index > index[-1]]
        if isinstance(index, pd.DatetimeIndex):
            after = after.as_unit(index.unit)
        merged = index.append(after) if len(after) else index
        if isinstance(index, pd.DatetimeIndex) and index.freq is not None and len(after):
            try:
                merged = pd.DatetimeIndex(merged, freq=index.freq, name=index.name)
            except ValueError:  # the new stamps do not continue the frequency; the model checks its grid
                merged = pd.DatetimeIndex(merged, name=index.name)
        merged = merged.rename(index.name)
        rows = _ns(merged).get_indexer(_ns(new_index))

        n_old, n_rows = len(index), len(merged)

        def overlay(old: np.ndarray, incoming: np.ndarray) -> np.ndarray:
            shape = (n_rows,) + incoming.shape[1:]
            out = np.full(shape, np.nan)
            out[(slice(0, n_old),) + tuple(slice(0, s) for s in old.shape[1:])] = old
            block = out[rows]
            keep = ~np.isnan(incoming)
            block[keep] = incoming[keep]
            out[rows] = block
            return out

        y = overlay(self._y.to_numpy(), new_y)
        x = None if self._exog is None else overlay(self._exog.to_numpy(), new_x)
        c = None if self._chars is None else overlay(self._chars, new_c)
        return type(self)._from_arrays_checked(merged, y, order, x, exog_order, c, char_order, self._freq)

    def _check_merge_index(self, other: pd.Index) -> None:
        """``other`` has this panel's index type (and period frequency, step grid or time zone)."""
        index = self._y.index
        problem = None
        if type(other) is not type(index):
            problem = f"a {type(other).__name__}, but the conditioning data has a {type(index).__name__}"
        elif isinstance(index, pd.PeriodIndex) and other.freq != index.freq:
            problem = f"periods of frequency {other.freqstr}, but the conditioning data has {index.freqstr}"
        elif isinstance(index, pd.DatetimeIndex) and str(other.tz) != str(index.tz):
            problem = f"time zone {other.tz}, but the conditioning data has {index.tz}"
        elif isinstance(index, pd.RangeIndex) and (
            (len(other) > 1 and other.step != index.step) or (other.start - index.start) % index.step
        ):
            problem = (
                f"rows {other.start}..{other[-1]} (step {other.step}), off the conditioning grid "
                f"{index.start}, {index.start + index.step}, ..."
            )
        if problem is not None:
            raise DataValidationError(f"The new data has {problem}. Pass it on the same kind of index.")

    def __repr__(self) -> str:
        index = self._y.index
        return (
            f"PanelData(n_periods={self.n_periods}, n_series={self.n_series}, "
            f"n_exog={len(self._exog_names)}, n_characteristics={len(self._characteristic_names)}, "
            f"index={type(index).__name__}[{index[0]}..{index[-1]}], freq={self._freq})"
        )

"""``PanelData``: the one data container every model's ``fit`` accepts.

A ``PanelData`` holds up to four slots that share one time index:

==================  ==========================================================
Slot                Type and shape
==================  ==========================================================
``y``               DataFrame, T x N: the observed target series
``exog``            DataFrame, T x K, or None: per-time inputs, not forecast
``series_exog``     DataFrame, T x (M*N), columns ``MultiIndex`` named
                    ``(feature, series)``, or None: per-time, per-series
                    features
``freq``            ``dict`` series -> frequency code, or None
==================  ==========================================================

The container is frozen and validated. It holds numpy and pandas data only
(polars frames are converted with ``.to_pandas()``; :meth:`PanelData.to_polars`
converts back), holds no device or scaler, and never imputes, scales or
reorders values: NaN stays NaN and every value is cast to float64.

Index rules: unique, strictly increasing, and a ``DatetimeIndex``,
``PeriodIndex`` or ``RangeIndex``. An evenly spaced, increasing integer index is
stored as the equivalent ``RangeIndex``; arrays get a ``RangeIndex``. The
``from_*`` constructors also parse string time stamps with
:func:`pandas.to_datetime`; the plain constructor does not.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config.constants import VALID_FREQUENCIES
from ..utils.errors import ConfigurationError, DataValidationError, UnsupportedInputError
from ..utils.output import require_polars, to_polars as _frame_to_polars

__all__ = ["PanelData"]

_SERIES_EXOG_LEVELS = ("feature", "series")
_LONG_ID, _LONG_TIME, _LONG_TARGET = "unique_id", "ds", "y"
_SLOTS = ("_y", "_exog", "_series_exog", "_freq", "_series_names", "_exog_names", "_series_exog_names")


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


def _same_index(a: pd.Index, b: pd.Index) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, pd.DatetimeIndex) and a.dtype != b.dtype and a.tz == b.tz:
        # Same instants stored at different resolutions (e.g. polars gives microseconds).
        return a.as_unit("ns").equals(b.as_unit("ns"))
    return a.equals(b)


def _validate_flat_columns(columns: pd.Index, slot: str) -> tuple:
    if isinstance(columns, pd.MultiIndex):
        raise DataValidationError(f"{slot} has MultiIndex columns; only series_exog uses a (feature, series) MultiIndex.")
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


def _check_codes(codes: Iterable[Any]) -> None:
    bad = sorted({repr(c) for c in codes if not isinstance(c, str) or c not in VALID_FREQUENCIES})
    if bad:
        raise DataValidationError(
            f"Unknown frequency code(s) {', '.join(bad)}. Valid codes: {sorted(VALID_FREQUENCIES)}."
        )


def _normalize_freq(freq: Optional[Mapping], series_names: Sequence[Hashable]) -> Optional[dict]:
    """Normalize ``freq`` to the per-series form ``{series: code}``.

    Accepts the per-series form ``{"gdp": "q"}`` and the grouped form
    ``{"q": ["gdp"]}``. The result is ordered like ``series_names``; an empty
    mapping gives None. Codes are checked against
    ``dfm_python.config.constants.VALID_FREQUENCIES``.
    """
    if freq is None:
        return None
    if not isinstance(freq, Mapping):
        raise TypeError(f"freq must be a dict or None, got {type(freq).__name__}.")
    if len(freq) == 0:
        return None

    def is_group(value: Any) -> bool:
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))

    grouped = [is_group(v) for v in freq.values()]
    known = set(series_names)
    per_series: dict = {}
    if all(grouped):
        _check_codes(freq.keys())
        for code, members in freq.items():
            for name in members:
                if name in per_series:
                    raise DataValidationError(
                        f"Series {name!r} is listed under two frequencies ({per_series[name]!r} and {code!r})."
                    )
                per_series[name] = code
    elif not any(grouped):
        _check_codes(freq.values())
        per_series = dict(freq)
    else:
        raise DataValidationError(
            "freq mixes the per-series form {'gdp': 'q'} with the grouped form {'q': ['gdp']}; use one of them."
        )
    unknown = [name for name in per_series if name not in known]
    if unknown:
        raise DataValidationError(f"freq names unknown series {unknown}; the series are {list(series_names)}.")
    return {name: per_series[name] for name in series_names if name in per_series} or None


# PanelData


class PanelData:
    """Frozen, validated panel of time series shared by all models.

    Parameters
    ----------
    y : pandas.DataFrame
        Target series, shape (T, N). Its index is the panel's time index.
    exog : pandas.DataFrame, optional
        Per-time inputs that are not forecast, shape (T, K), on the same index.
    series_exog : pandas.DataFrame, optional
        Per-time, per-series features, shape (T, M*N), on the same index. The
        columns are a two-level ``MultiIndex`` named ``("feature", "series")``
        (unnamed levels are read in that order) whose ``series`` level holds
        exactly the columns of ``y``, every feature once per series.
    freq : dict, optional
        Sampling frequency per series, either ``{"gdp": "q"}`` or
        ``{"q": ["gdp"]}``. Series that are not listed use the model's clock.

    Notes
    -----
    An ``exog`` or ``series_exog`` frame with no columns is stored as None.
    Values are copied and cast to float64; NaN is kept. Nothing is imputed,
    scaled or reordered. Attribute assignment raises ``AttributeError``, and
    the ``y``, ``exog``, ``series_exog`` and ``freq`` accessors return copies.
    """

    __slots__ = _SLOTS

    def __init__(
        self,
        y: pd.DataFrame,
        exog: Optional[pd.DataFrame] = None,
        series_exog: Optional[pd.DataFrame] = None,
        freq: Optional[Mapping] = None,
    ) -> None:
        y = _require_frame(y, "y")
        index = _validate_index(y.index, "y")
        if y.shape[1] == 0:
            raise DataValidationError("y has no columns; PanelData needs at least one target series.")
        series_names = _validate_flat_columns(y.columns, "y")
        y_frame = pd.DataFrame(_float_values(y, "y"), index=index, columns=pd.Index(series_names, tupleize_cols=False))

        exog_frame, exog_names = None, ()
        if exog is not None:
            exog = _require_frame(exog, "exog")
        if exog is not None and exog.shape[1] > 0:
            exog_names = _validate_flat_columns(exog.columns, "exog")
            self._check_shared_index(exog, index, "exog")
            overlap = [name for name in exog_names if name in set(series_names)]
            if overlap:
                raise DataValidationError(f"Names used both as targets (y) and as exog: {overlap}.")
            exog_frame = pd.DataFrame(
                _float_values(exog, "exog"), index=index, columns=pd.Index(exog_names, tupleize_cols=False)
            )

        sexog_frame, sexog_names = None, ()
        if series_exog is not None:
            series_exog = _require_frame(series_exog, "series_exog")
        if series_exog is not None and series_exog.shape[1] > 0:
            self._check_shared_index(series_exog, index, "series_exog")
            columns, sexog_names = self._validate_series_exog_columns(series_exog.columns, series_names)
            overlap = [name for name in sexog_names if name in set(exog_names)]
            if overlap:
                raise DataValidationError(f"Names used both as exog and as series_exog features: {overlap}.")
            sexog_frame = pd.DataFrame(_float_values(series_exog, "series_exog"), index=index, columns=columns)

        norm_freq = _normalize_freq(freq, series_names)

        for slot, value in zip(
            _SLOTS, (y_frame, exog_frame, sexog_frame, norm_freq, series_names, exog_names, sexog_names)
        ):
            object.__setattr__(self, slot, value)

    @staticmethod
    def _check_shared_index(frame: pd.DataFrame, index: pd.Index, slot: str) -> None:
        other = _validate_index(frame.index, slot)
        if not _same_index(other, index):
            raise DataValidationError(
                f"{slot} must share the index of y exactly (same type, same time stamps in the same order); "
                f"y has {len(index)} rows, {slot} has {len(other)}."
            )

    @staticmethod
    def _validate_series_exog_columns(columns: pd.Index, series_names: tuple) -> tuple[pd.MultiIndex, tuple]:
        if not isinstance(columns, pd.MultiIndex) or columns.nlevels != 2:
            raise DataValidationError(
                "series_exog needs a two-level column MultiIndex named ('feature', 'series'); build it with "
                "pd.MultiIndex.from_product([features, series], names=['feature', 'series']) or use "
                "PanelData.from_arrays."
            )
        names = tuple(columns.names)
        if names not in ((None, None), _SERIES_EXOG_LEVELS):
            raise DataValidationError(
                f"series_exog column levels must be named ('feature', 'series') in that order, got {names}."
            )
        if not columns.is_unique:
            dupes = columns[columns.duplicated()].unique().tolist()[:5]
            raise DataValidationError(f"series_exog has duplicated (feature, series) columns, for example {dupes}.")
        features = tuple(pd.unique(columns.get_level_values(0)).tolist())
        level_series = set(columns.get_level_values(1))
        missing = [s for s in series_names if s not in level_series]
        extra = sorted(repr(s) for s in level_series - set(series_names))
        if missing or extra:
            raise DataValidationError(
                f"The 'series' level of series_exog must match the columns of y; missing {missing}, "
                f"unknown {extra}."
            )
        if len(columns) != len(features) * len(series_names):
            raise DataValidationError(
                f"series_exog must hold every feature for every series: {len(features)} features x "
                f"{len(series_names)} series = {len(features) * len(series_names)} columns, got {len(columns)}."
            )
        return columns.set_names(list(_SERIES_EXOG_LEVELS)), features

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"PanelData is immutable; cannot set {name!r}. Build a new PanelData instead.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"PanelData is immutable; cannot delete {name!r}.")

    def __getstate__(self) -> dict:
        return {slot: getattr(self, slot) for slot in _SLOTS}

    def __setstate__(self, state: dict) -> None:
        for slot in _SLOTS:
            object.__setattr__(self, slot, state[slot])

    @classmethod
    def from_wide(
        cls,
        df: pd.DataFrame,
        *,
        time_col: Optional[Hashable] = None,
        target_cols: Optional[Sequence[Hashable]] = None,
        exog_cols: Optional[Sequence[Hashable]] = None,
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from one wide frame: one row per time, one column per variable.

        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
        time_col : column name, optional
            Column holding the time stamps; None uses ``df.index``. String time
            stamps are parsed with :func:`pandas.to_datetime`.
        target_cols : list of column names, optional
            Target series; None means every column that is neither
            ``time_col`` nor in ``exog_cols``.
        exog_cols : list of column names, optional
            Per-time inputs that are not forecast.
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

        exog_cols = cls._as_name_list(exog_cols, "exog_cols")
        cls._require_columns(body, exog_cols, "exog_cols")
        if target_cols is None:
            excluded = set(exog_cols)
            target_cols = [c for c in body.columns if c not in excluded]
        else:
            target_cols = cls._as_name_list(target_cols, "target_cols")
            cls._require_columns(body, target_cols, "target_cols")
            overlap = [c for c in target_cols if c in set(exog_cols)]
            if overlap:
                raise DataValidationError(f"Columns listed in both target_cols and exog_cols: {overlap}.")
        if not target_cols:
            raise DataValidationError("from_wide found no target columns.")

        y = body[target_cols].set_axis(index, axis=0)
        exog = body[exog_cols].set_axis(index, axis=0) if exog_cols else None
        return cls(y, exog=exog, freq=freq)

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        id_col: Hashable = _LONG_ID,
        time_col: Hashable = _LONG_TIME,
        target_col: Hashable = _LONG_TARGET,
        series_exog_cols: Sequence[Hashable] = (),
        exog_cols: Sequence[Hashable] = (),
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from a long frame with one row per ``(id, time)`` pair.

        Series are ordered by first appearance of their id and time stamps are
        sorted. A missing ``(id, time)`` row becomes NaN. ``exog_cols`` must be
        constant across ids at each time (NaN entries are treated as missing);
        otherwise :class:`DataValidationError` is raised. Columns not named in
        any argument are ignored.
        """
        df = _require_frame(df, "df")
        series_exog_cols = cls._as_name_list(series_exog_cols, "series_exog_cols")
        exog_cols = cls._as_name_list(exog_cols, "exog_cols")
        roles = [id_col, time_col, target_col, *series_exog_cols, *exog_cols]
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

        y = wide(target_col)
        series_exog = None
        if series_exog_cols:
            series_exog = pd.concat(
                [wide(col) for col in series_exog_cols], axis=1, keys=series_exog_cols, names=list(_SERIES_EXOG_LEVELS)
            )
        exog = None
        if exog_cols:
            values = pd.DataFrame(
                _float_values(long[exog_cols], "exog_cols"), index=long.index, columns=pd.Index(exog_cols)
            )
            values[time_col] = long[time_col]
            grouped = values.groupby(time_col, sort=True)
            varying = grouped.nunique(dropna=True) > 1
            if varying.to_numpy().any():
                col = varying.columns[varying.any(axis=0).to_numpy()][0]
                when = varying.index[varying[col].to_numpy()][0]
                raise DataValidationError(
                    f"exog column {col!r} differs across ids at {time_col}={when!r}; a per-series variable "
                    "belongs in series_exog_cols."
                )
            exog = grouped.first().reindex(index)
            exog.columns.name = None
        return cls(y, exog=exog, series_exog=series_exog, freq=freq)

    @classmethod
    def from_arrays(
        cls,
        y: Any,
        *,
        exog: Any = None,
        series_exog: Any = None,
        index: Any = None,
        names: Optional[Sequence[Hashable]] = None,
        exog_names: Optional[Sequence[Hashable]] = None,
        series_exog_names: Optional[Sequence[Hashable]] = None,
        freq: Optional[Mapping] = None,
    ) -> PanelData:
        """Build from arrays: ``y`` (T, N), ``exog`` (T, K), ``series_exog`` (T, N, M).

        ``index=None`` gives a ``RangeIndex``. Default names are ``series_0..``,
        ``exog_0..`` and ``feature_0..``.
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
        series = cls._names(names, n_series, "series", "names")
        y_frame = pd.DataFrame(y_arr, index=time_index, columns=series)

        exog_frame = None
        if exog is not None:
            x_arr = cls._as_array(exog, 2, "exog", "(T, K)")
            if x_arr.shape[0] != n_time:
                raise DataValidationError(f"exog has {x_arr.shape[0]} rows but y has {n_time}.")
            exog_frame = pd.DataFrame(
                x_arr, index=time_index, columns=cls._names(exog_names, x_arr.shape[1], "exog", "exog_names")
            )

        sexog_frame = None
        if series_exog is not None:
            s_arr = cls._as_array(series_exog, 3, "series_exog", "(T, N, M)")
            if s_arr.shape[:2] != (n_time, n_series):
                raise DataValidationError(
                    f"series_exog has shape {s_arr.shape}; expected ({n_time}, {n_series}, M) to match y."
                )
            n_feat = s_arr.shape[2]
            if n_feat == 0:
                return cls(y_frame, exog=exog_frame, freq=freq)
            features = cls._names(series_exog_names, n_feat, "feature", "series_exog_names")
            columns = pd.MultiIndex.from_product([features, series], names=list(_SERIES_EXOG_LEVELS))
            # column m * N + n holds feature m of series n
            flat = s_arr.transpose(0, 2, 1).reshape(n_time, n_feat * n_series)
            sexog_frame = pd.DataFrame(flat, index=time_index, columns=columns)
        return cls(y_frame, exog=exog_frame, series_exog=sexog_frame, freq=freq)

    @classmethod
    def _coerce(cls, obj: Any) -> PanelData:
        """Turn what ``fit``, ``predict`` and ``update`` accept into a ``PanelData``.

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
    def _names(names: Optional[Sequence[Hashable]], n: int, prefix: str, what: str) -> pd.Index:
        if names is None:
            return pd.Index([f"{prefix}_{i}" for i in range(n)])
        if isinstance(names, (str, bytes)):
            raise TypeError(f"{what} must be a list of names, got a string.")
        names = list(names)
        if len(names) != n:
            raise DataValidationError(f"{what} has {len(names)} entries; expected {n}.")
        return pd.Index(names, tupleize_cols=False)

    @property
    def y(self) -> pd.DataFrame:
        """Target series, T x N (a copy)."""
        return self._y.copy()

    @property
    def exog(self) -> Optional[pd.DataFrame]:
        """Per-time inputs, T x K (a copy), or None."""
        return None if self._exog is None else self._exog.copy()

    @property
    def series_exog(self) -> Optional[pd.DataFrame]:
        """Per-series features, T x (M*N) with ``(feature, series)`` columns (a copy), or None."""
        return None if self._series_exog is None else self._series_exog.copy()

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
    def series_exog_names(self) -> tuple:
        """Per-series feature names, in order of first appearance; empty without ``series_exog``."""
        return self._series_exog_names

    @property
    def n_timesteps(self) -> int:
        """Number of time steps T."""
        return len(self._y.index)

    @property
    def is_mixed_frequency(self) -> bool:
        """True if ``freq`` lists two or more frequencies, or lists some but not all series."""
        if not self._freq:
            return False
        return len(set(self._freq.values())) > 1 or len(self._freq) < len(self._series_names)

    def series_exog_array(self) -> Optional[np.ndarray]:
        """``series_exog`` as a float64 array (T, N, M), ordered by ``series_names`` then ``series_exog_names``.

        Returns None without ``series_exog``.
        """
        if self._series_exog is None:
            return None
        n_time, n_series, n_feat = self.n_timesteps, len(self._series_names), len(self._series_exog_names)
        columns = pd.MultiIndex.from_product(
            [list(self._series_exog_names), list(self._series_names)], names=list(_SERIES_EXOG_LEVELS)
        )
        flat = self._series_exog.reindex(columns=columns).to_numpy(dtype=np.float64, copy=True)
        return flat.reshape(n_time, n_feat, n_series).transpose(0, 2, 1).copy()

    def to_long(self) -> pd.DataFrame:
        """Long frame with columns ``unique_id``, ``ds``, ``y``, the series_exog features, then the exog names.

        One row per ``(series, time)``, ordered by series then time; NaN
        targets are kept, so ``PanelData.from_long(panel.to_long(), ...)``
        rebuilds the panel.
        """
        reserved = {_LONG_ID, _LONG_TIME, _LONG_TARGET}
        clash = [c for c in (*self._series_exog_names, *self._exog_names) if c in reserved]
        if clash:
            raise DataValidationError(f"Cannot write a long frame: names {clash} clash with {sorted(reserved)}.")
        n_time, n_series = self.n_timesteps, len(self._series_names)
        index = self._y.index
        out = pd.DataFrame(
            {
                _LONG_ID: np.repeat(np.asarray(self._series_names, dtype=object), n_time),
                _LONG_TIME: index.take(np.tile(np.arange(n_time), n_series)),
                _LONG_TARGET: self._y.to_numpy().T.reshape(-1),
            }
        )
        if self._series_exog is not None:
            arr = self.series_exog_array().transpose(1, 0, 2).reshape(n_series * n_time, -1)
            for j, name in enumerate(self._series_exog_names):
                out[name] = arr[:, j]
        if self._exog is not None:
            tiled = np.tile(self._exog.to_numpy(), (n_series, 1))
            for j, name in enumerate(self._exog_names):
                out[name] = tiled[:, j]
        return out

    def to_polars(self, *, layout: str = "long") -> Any:
        """The panel as a ``polars.DataFrame``.

        ``layout="long"`` is :meth:`to_long` (``PanelData.from_long`` rebuilds
        the panel). ``layout="wide"`` holds the time column (named after the
        index, or ``"time"``), the ``y`` columns, then the ``exog`` columns
        (``PanelData.from_wide(df, time_col=..., exog_cols=...)`` rebuilds the
        panel). A ``PeriodIndex`` becomes its start timestamps and the wide
        column names become strings, as in :func:`dfm_python.utils.output.to_polars`.
        Needs the ``polars`` extra.

        Raises
        ------
        ConfigurationError
            For a ``layout`` other than ``"long"`` or ``"wide"``.
        UnsupportedInputError
            For ``layout="wide"`` with ``series_exog``, which has no wide form.
        DataValidationError
            For ``layout="long"`` when a feature or exog name is ``unique_id``,
            ``ds`` or ``y`` (as in :meth:`to_long`).
        ValueError
            For ``layout="wide"`` when the time column's name is also a column name.
        """
        if layout == "long":
            pl = require_polars()
            long = self.to_long()
            if isinstance(self._y.index, pd.PeriodIndex):
                long[_LONG_TIME] = long[_LONG_TIME].dt.to_timestamp()
            return pl.from_pandas(long)
        if layout == "wide":
            if self._series_exog is not None:
                raise UnsupportedInputError(
                    "series_exog has no wide layout (one column per variable); use to_polars(layout='long')."
                )
            return _frame_to_polars(self._y if self._exog is None else pd.concat([self._y, self._exog], axis=1))
        raise ConfigurationError(f"layout must be 'long' or 'wide', got {layout!r}.")

    def loc(self, start: Any = None, stop: Any = None) -> PanelData:
        """Select rows by label, both ends included (pandas ``.loc`` semantics); None leaves an end open."""
        rows = self._y.loc[start:stop].index
        if len(rows) == 0:
            raise DataValidationError(f"loc({start!r}, {stop!r}) selects no rows.")
        return type(self)(
            self._y.loc[rows],
            exog=None if self._exog is None else self._exog.loc[rows],
            series_exog=None if self._series_exog is None else self._series_exog.loc[rows],
            freq=self._freq,
        )

    def equals(self, other: Any) -> bool:
        """True if ``other`` is a PanelData with the same index, names, freq and values (NaN equals NaN).

        Index names and the column order of ``series_exog`` are not compared.
        """
        if not isinstance(other, PanelData):
            return False
        same_meta = (
            _same_index(self._y.index, other._y.index)
            and self._series_names == other._series_names
            and self._exog_names == other._exog_names
            and self._series_exog_names == other._series_exog_names
            and self._freq == other._freq
        )
        if not same_meta:
            return False
        pairs = [(self._y.to_numpy(), other._y.to_numpy())]
        if self._exog is not None:
            pairs.append((self._exog.to_numpy(), other._exog.to_numpy()))
        if self._series_exog is not None:
            pairs.append((self.series_exog_array(), other.series_exog_array()))
        return all(np.array_equal(a, b, equal_nan=True) for a, b in pairs)

    def __repr__(self) -> str:
        index = self._y.index
        return (
            f"PanelData(n_timesteps={self.n_timesteps}, n_series={len(self._series_names)}, "
            f"n_exog={len(self._exog_names)}, n_series_exog={len(self._series_exog_names)}, "
            f"index={type(index).__name__}[{index[0]}..{index[-1]}], "
            f"mixed_frequency={self.is_mixed_frequency})"
        )

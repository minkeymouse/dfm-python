"""Output frames for :meth:`BaseFactorModel.set_output`: pandas as returned, or polars."""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["OUTPUTS", "require_polars", "to_output", "to_polars"]

#: The frame types :meth:`BaseFactorModel.set_output` accepts.
OUTPUTS = ("pandas", "polars")

#: Joins the levels of MultiIndex columns in a polars frame.
LEVEL_SEPARATOR = "__"


def require_polars() -> Any:
    """Import and return polars, or raise ``ImportError`` naming the ``polars`` extra."""
    try:
        import polars as pl
        import pyarrow  # noqa: F401  (pl.from_pandas needs it)
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            'polars output needs polars and pyarrow: pip install "dfm-python[polars]".'
        ) from err
    return pl


def to_polars(frame: pd.DataFrame | pd.Series) -> Any:
    """A pandas frame (or series) as a ``polars.DataFrame``, its index as the first column.

    The index column is named after the index (``"time"`` when unnamed); a
    ``PeriodIndex`` becomes its start timestamps. MultiIndex columns are
    joined with ``"__"``. Raises ``ValueError`` when the index name equals a
    column name.
    """
    pl = require_polars()
    df = frame.to_frame() if isinstance(frame, pd.Series) else frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [LEVEL_SEPARATOR.join(str(level) for level in col) for col in df.columns]
    else:
        df.columns = [str(col) for col in df.columns]
    name = df.index.name if df.index.name is not None else "time"
    if name in df.columns:
        raise ValueError(f"cannot convert to polars: the index name {name!r} is also a column name.")
    index = df.index.to_timestamp() if isinstance(df.index, pd.PeriodIndex) else df.index
    df.index = pd.Index(index, name=name)
    return pl.from_pandas(df.reset_index())


def to_output(frame: Any, output: str) -> Any:
    """``frame`` unchanged for ``"pandas"``, :func:`to_polars` for ``"polars"``."""
    if output == "pandas" or not isinstance(frame, (pd.DataFrame, pd.Series)):
        return frame
    if output == "polars":
        return to_polars(frame)
    raise ValueError(f"unknown output {output!r}; expected one of {list(OUTPUTS)}.")

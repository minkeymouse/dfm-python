"""Walk-forward (rolling-origin) evaluation of the forecasting estimators.

:func:`backtest` conditions a clone of an estimator on the data up to each of
a sequence of forecast origins, forecasts ``h`` steps from each origin and
returns the forecasts beside the values the data holds for them in a
:class:`BacktestResult`, whose :meth:`~BacktestResult.metrics` gives RMSE,
MAE, bias and interval coverage by horizon, by series or overall::

    >>> from dfm_python import DFM
    >>> from dfm_python.evaluation import backtest
    >>> result = backtest(DFM(n_factors=1, scaler="standard"), data, start=-24, h=6, coverage=0.9)
    >>> result.forecasts            # origin, time, horizon, series, forecast, actual, lower, upper
    >>> result.metrics(by="horizon")  # RMSE, MAE, bias, n, coverage per horizon
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseFactorModel
from .dataset.panel import PanelData
from .utils.errors import ConfigurationError, UnsupportedInputError

__all__ = ["backtest", "BacktestResult"]

_GROUPS = ("horizon", "series")


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """The forecasts of a walk-forward backtest (see :func:`backtest`).

    Attributes
    ----------
    forecasts : pandas.DataFrame
        One row per origin, horizon and series, in that order (the series in
        the data's column order). Columns: ``origin`` (the forecast origin),
        ``time`` (the time stamp of the row ``horizon`` steps after the
        origin; past the end of the data, the stamps continue its index),
        ``horizon`` (1 to ``h``), ``series``, ``forecast``, ``actual`` (the
        data's value at ``time``; NaN where it is missing or past the end of
        the data) and, when the backtest was given a ``coverage``, ``lower``
        and ``upper`` (the prediction interval).
    origins : pandas.Index
        The forecast origins, time stamps of the data's index, in order.
    fit_seconds : pandas.Series
        Wall-clock seconds spent conditioning the model at each origin (in
        ``fit`` where it was fitted, in ``update`` elsewhere), indexed by
        ``origins``.
    coverage : float or None
        The nominal coverage of ``lower`` and ``upper``; None without
        intervals.
    """

    forecasts: pd.DataFrame
    origins: pd.Index
    fit_seconds: pd.Series
    coverage: Optional[float] = None

    def metrics(self, by: Union[str, tuple, None] = "horizon") -> pd.DataFrame:
        """Forecast errors over the forecasts whose actual is observed, per group.

        Parameters
        ----------
        by : {"horizon", "series"}, tuple of both, or None, default "horizon"
            One row per horizon, per series, per pair (a MultiIndex in the
            tuple's order), or for None one row labelled ``"all"``. Horizons
            are in increasing order and series in the data's order.

        Returns
        -------
        pandas.DataFrame
            Columns ``RMSE`` and ``MAE`` (root mean squared and mean absolute
            error), ``bias`` (mean of forecast minus actual), ``n`` (the number
            of forecasts with an observed actual) and, when the backtest gave
            intervals, ``coverage`` (the share of those actuals within
            ``[lower, upper]``). Forecasts whose actual is NaN count nowhere; a
            group without an observed actual has ``n`` 0 and NaN metrics.

        Raises
        ------
        ConfigurationError
            For any other ``by``.
        """
        keys = [by] if isinstance(by, str) else list(by) if isinstance(by, tuple) else []
        if (by is not None and not keys) or any(k not in _GROUPS for k in keys) or len(set(keys)) < len(keys):
            raise ConfigurationError(f"by must be 'horizon', 'series', a tuple of both or None, got {by!r}.")
        frame = self.forecasts
        error = frame["forecast"] - frame["actual"]
        observed = error.notna()
        parts = pd.DataFrame({"sq": error**2, "abs": error.abs(), "bias": error})
        if self.coverage is not None:
            inside = (frame["lower"] <= frame["actual"]) & (frame["actual"] <= frame["upper"])
            parts["coverage"] = inside.astype(np.float64).where(observed)
        groups = [frame[k] for k in keys] if keys else pd.Series("all", index=frame.index)
        means = parts.groupby(groups, sort=False).mean()
        out = pd.DataFrame({
            "RMSE": np.sqrt(means["sq"]),
            "MAE": means["abs"],
            "bias": means["bias"],
            "n": observed.groupby(groups, sort=False).sum().astype(np.int64),
        })
        if self.coverage is not None:
            out["coverage"] = means["coverage"]
        order = pd.Index(pd.unique(frame["series"]))
        return out.sort_index(key=lambda level: pd.Index(order.get_indexer(level)) if level.name == "series" else level)


def _origin_position(index: pd.Index, start: Any) -> int:
    """The position of ``start`` in ``index``: an int is a position (negative from the end), else a label."""
    if isinstance(start, (int, np.integer)) and not isinstance(start, (bool, np.bool_)):
        position = int(start) + (len(index) if start < 0 else 0)
        if not 0 <= position < len(index):
            raise ConfigurationError(f"start={start} is not a position of data, which has {len(index)} rows.")
        return position
    try:
        position = index.get_loc(start)
    except (KeyError, TypeError, pd.errors.InvalidIndexError):
        position = None
    if not isinstance(position, (int, np.integer)):
        raise ConfigurationError(
            f"start={start!r} is not a time stamp of data (it runs from {index[0]!r} to {index[-1]!r}); pass one "
            "of its index labels or an int position."
        )
    return int(position)


def _refit_every(refit: Any) -> int:
    """Refit at every ``n``-th origin; 0 means only at the first."""
    if isinstance(refit, (bool, np.bool_)):
        return int(bool(refit))
    if isinstance(refit, (int, np.integer)) and refit > 0:
        return int(refit)
    raise ConfigurationError(f"refit must be True, False or a positive integer, got {refit!r}.")


def backtest(
    estimator: BaseFactorModel,
    data: Any,
    *,
    start: Any,
    h: int,
    step: int = 1,
    refit: Union[bool, int] = False,
    coverage: Optional[float] = None,
) -> BacktestResult:
    """Walk-forward (rolling-origin) evaluation of a forecasting estimator.

    The origins are the rows ``start``, ``start + step``, ... of ``data`` that
    at least one row follows. At each origin the model is conditioned on the
    rows of ``data`` up to and including the origin, and on nothing later:

    - At the first origin, and wherever ``refit`` says so, a fresh clone of
      ``estimator`` (``sklearn.base.clone``) is fitted on those rows.
    - At the other origins the current model's ``update`` conditions it on
      them with its fitted parameters: it gets the rows after the previous
      origin when its tag ``predict_data`` is ``"new_rows"`` (DFM, DDFM), and
      all the rows up to the origin otherwise (iVDFM, ``"window"``: the
      window is encoded and the forecast starts from its last row).

    Then ``predict(h)``, and with ``coverage`` ``predict_interval(h,
    coverage)``, forecast the ``h`` rows after the origin: horizon ``k`` of
    the origin at row ``p`` is compared with row ``p + k`` of ``data``. A
    model fitted with exog whose ``predict`` takes ``futr_exog`` (iVDFM) gets
    the exog rows ``p + 1`` to ``p + h`` as ``futr_exog`` when ``data`` holds
    them all, finite; this treats exog as known in advance, so its forecasts
    use exog values from after the origin. Otherwise it forecasts without
    them, and the model warns. Other models see no exog after the origin.

    Parameters
    ----------
    estimator : BaseFactorModel
        A forecasting estimator (tag ``task="forecast"``) with its
        hyperparameters; it is cloned, never fitted or changed.
    data : PanelData, DataFrame, polars DataFrame or 2-D array
        Anything the estimator's ``fit`` accepts.
    start : int or index label
        The first origin: an int is a row position (negative counts from the
        end), anything else a time stamp of ``data``'s index.
    h : int
        Forecast horizon: the number of steps forecast at each origin. Steps
        past the end of ``data`` get the time stamps that continue its index
        and a NaN actual.
    step : int, default 1
        Rows from one origin to the next.
    refit : bool or int, default False
        False fits once, at the first origin, and updates afterwards. True
        refits at every origin. A positive int ``k`` refits at every ``k``-th
        origin (the first, the ``k + 1``-th, ...) and updates at the others.
        Updating needs the tag ``capability:update``.
    coverage : float, optional
        Nominal coverage in (0, 1) of the prediction intervals (tag
        ``capability:pred_int``) in the ``lower`` and ``upper`` columns.

    Returns
    -------
    BacktestResult
        The long frame of forecasts and actuals, the origins and the seconds
        spent fitting or updating at each origin.

    Raises
    ------
    TypeError
        For an estimator that is not a forecasting ``BaseFactorModel`` (AFM
        allocates: tag ``task="allocation"``), or an ``h`` that is not an int.
    ValueError
        For ``h < 1``.
    ConfigurationError
        For a ``start`` that is not in ``data`` or leaves no row after it, a
        ``step`` that is not a positive int, a ``refit`` that is neither a
        bool nor a positive int, a ``coverage`` outside (0, 1), and a
        ``refit`` that would update a model without ``capability:update``.
    UnsupportedInputError
        For a ``coverage`` with a model without ``capability:pred_int``.
    """
    tags = estimator.get_tags() if isinstance(estimator, BaseFactorModel) else {}
    name = type(estimator).__name__
    if tags.get("task") != "forecast":
        raise TypeError(
            f"backtest needs a forecasting estimator (a BaseFactorModel with tag task='forecast'), got {name}; "
            "AFM allocates and has its own AFM.backtest(data)."
        )
    h = BaseFactorModel._validate_horizon(h)
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 1:
        raise ConfigurationError(f"step must be a positive integer (rows between origins), got {step!r}.")
    every = _refit_every(refit)
    if every != 1 and not tags["capability:update"]:
        raise ConfigurationError(
            f"{name} cannot update (tag capability:update is False), so backtest must refit it at every origin; "
            f"pass refit=True (got refit={refit!r})."
        )
    if coverage is not None:
        if not tags["capability:pred_int"]:
            raise UnsupportedInputError(
                f"{name} does not give prediction intervals (tag capability:pred_int is False); drop coverage."
            )
        valid = isinstance(coverage, (int, float, np.integer, np.floating)) and not isinstance(coverage, (bool, np.bool_))
        if not (valid and math.isfinite(coverage) and 0 < coverage < 1):
            raise ConfigurationError(f"coverage must be a number in (0, 1), got {coverage!r}.")
        coverage = float(coverage)

    panel = PanelData._coerce(data)
    index = panel.index
    n_rows, n_series = panel.n_timesteps, len(panel.series_names)
    first = _origin_position(index, start)
    if first > n_rows - 2:
        raise ConfigurationError(
            f"start={start!r} is row {first} of {n_rows}: no row of data follows it, so there is nothing to forecast."
        )
    positions = np.arange(first, n_rows - 1, int(step))
    series = list(panel.series_names)
    exog = None if panel.exog is None else panel.exog.to_numpy(dtype=np.float64)

    forecasts, lower, upper, seconds = [], [], [], []
    model: Any = None
    for i, p in enumerate(positions):
        origin = index[p]
        tic = time.perf_counter()
        if i == 0 or (every and i % every == 0):
            model = clone(estimator).fit(panel.loc(None, origin))
        elif tags["predict_data"] == "new_rows":
            model.update(panel.loc(index[positions[i - 1] + 1], origin))
        else:
            model.update(panel.loc(None, origin))
        seconds.append(time.perf_counter() - tic)
        kwargs = {}
        if (
            exog is not None
            and len(model.exog_names_in_)
            and "futr_exog" in inspect.signature(model.predict).parameters
            and p + h < n_rows
            and np.isfinite(exog[p + 1 : p + 1 + h]).all()
        ):
            kwargs["futr_exog"] = exog[p + 1 : p + 1 + h]
        forecasts.append(model.predict(h, **kwargs)[series].to_numpy(dtype=np.float64))
        if coverage is not None:
            bands = model.predict_interval(h, coverage)
            lower.append(bands.xs("lower", axis=1, level="bound")[series].to_numpy(dtype=np.float64))
            upper.append(bands.xs("upper", axis=1, level="bound")[series].to_numpy(dtype=np.float64))

    # Row p + k of the data, extended past its end by h stamps (and NaN values).
    stamps = index.append(BaseFactorModel._forecast_index(index, h))
    values = np.vstack([panel.y.to_numpy(dtype=np.float64), np.full((h, n_series), np.nan)])
    rows = np.repeat((positions[:, None] + np.arange(1, h + 1)).reshape(-1), n_series)
    columns = np.tile(np.arange(n_series), len(positions) * h)
    frame = pd.DataFrame({
        "origin": index.take(np.repeat(positions, h * n_series)),
        "time": stamps.take(rows),
        "horizon": np.tile(np.repeat(np.arange(1, h + 1), n_series), len(positions)),
        "series": np.asarray(series, dtype=object)[columns],
        "forecast": np.concatenate(forecasts).reshape(-1),
        "actual": values[rows, columns],
    })
    if coverage is not None:
        frame["lower"] = np.concatenate(lower).reshape(-1)
        frame["upper"] = np.concatenate(upper).reshape(-1)
    origins = index.take(positions)
    return BacktestResult(
        forecasts=frame,
        origins=origins,
        fit_seconds=pd.Series(seconds, index=origins, name="fit_seconds", dtype=np.float64),
        coverage=coverage,
    )

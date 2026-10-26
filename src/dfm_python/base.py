"""``BaseFactorModel``: the public estimator base class.

The four models (``DFM``, ``DDFM``, ``iVDFM``, ``AFM``) subclass
:class:`BaseFactorModel`, which builds on :class:`sklearn.base.BaseEstimator`:

- ``__init__`` takes keyword-only hyperparameters and only stores them, so
  ``get_params``, ``set_params``, ``clone`` and ``repr`` work unchanged.
- ``fit(data, *, callback=None)`` coerces ``data`` to
  :class:`~dfm_python.PanelData`, rejects inputs the tags exclude, infinite
  values and (forecasting models) target series observed fewer than twice or
  constant, records the data attributes, runs the subclass's ``_fit`` inside
  :func:`~dfm_python.logger.fit_verbosity` (which hands ``callback`` to the
  training loop through :class:`~dfm_python.utils.run_control.RunControl`)
  and returns ``self``.
- Fitted state lives in attributes ending in ``_``; before ``fit`` they raise
  :class:`~dfm_python.utils.errors.NotFittedError` (an ``AttributeError``), so
  ``hasattr(model, "result_")`` is False.
- Tags (``get_tags()``) describe the task, backend and capabilities; the
  default ``predict``, ``predict_factors``, ``update`` and ``allocate`` raise
  ``TypeError`` or ``UnsupportedInputError`` when the tags exclude them.
- ``get_config()`` / ``from_config()`` convert to and from the model's
  :class:`~dfm_python.config.EstimatorConfig` mirror, whose ``json_schema()``
  describes the hyperparameters; ``describe()`` summarizes the model as a
  small JSON-serializable dict.
- ``save(path)`` writes the pickle-free save directory through
  :func:`~dfm_python.io.atomic_save_dir` (temporary sibling, reload to verify,
  rename); ``Model.load(path)`` reads it back without a dataset. The base class
  writes ``config.json`` and ``data_spec.json`` and restores the data
  attributes and ``scaler_``; each model writes and reads the rest in its
  ``_save_state`` / ``_load_state`` hooks with the ``_write_*`` / ``_read_*``
  helpers below.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from .config.estimator import EstimatorConfig, _is_torch_device, _unknown_keys_message
from .dataset.panel import PanelData
from .io.codecs import (
    is_encoded_estimator,
    read_json,
    read_npz,
    read_weights,
    to_jsonable,
    write_json,
    write_npz,
    write_weights,
)
from .io.directory import atomic_save_dir
from .io.layout import (
    CONFIG_FILE,
    DATA_SPEC_FILE,
    FORMAT_VERSION,
    REQUIRED_FILES,
    RESULT_JSON,
    RESULT_NPZ,
    STATE_JSON,
    STATE_NPZ,
    WEIGHTS_FILE,
    index_from_spec,
    index_to_spec,
    read_config,
)
from .io.results import result_from_files, result_to_files
from .io.scaler import SCALER_PREFIX, has_scaler, read_scaler, write_scaler
from .logger.progress import fit_verbosity
from .utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

__all__ = ["BaseFactorModel", "DEFAULT_TAGS"]

#: Tag defaults. Subclasses override any subset in their own ``_tags``.
DEFAULT_TAGS: dict = {
    "task": "forecast",  # "forecast" or "allocation"
    "backend": "numpy",  # "numpy" or "torch"
    "capability:exog": False,
    "capability:series_exog": False,
    "capability:mixed_frequency": False,
    "capability:missing_values": False,
    "capability:update": False,
    "capability:factor_forecast": False,
    "capability:device": False,
    "capability:interpret": False,  # factors_, loadings_, explained_variance_ and decompose()
    "capability:pred_int": False,  # predict_interval()
    "update_kind": None,  # "filter", "encoder_forward_rescale", "encoder_forward" or "reattach"
    "predict_data": None,  # "new_rows", "full_panel" or "window"
    "non_deterministic": False,
}

#: ``describe()`` keys of a fitted model and the attributes they come from; a missing attribute omits its key.
_DESCRIBED_ATTRIBUTES = (
    ("n_features_in", "n_features_in_"),
    ("feature_names", "feature_names_in_"),
    ("index", "index_"),
    ("n_iter", "n_iter_"),
    ("converged", "converged_"),
    ("stop_reason", "stop_reason_"),
    ("factor_names", "factor_names_"),
    ("explained_variance", "explained_variance_"),
)


class _class_or_instance_method:
    """Like ``classmethod``, but also passes the instance (None when called on the class).

    The function is called as ``func(cls, instance, *args, **kwargs)``.
    """

    def __init__(self, func: Any) -> None:
        self.__func__ = func
        functools.update_wrapper(self, func)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        cls = owner if owner is not None else type(instance)
        func = self.__func__

        @functools.wraps(func)
        def bound(*args: Any, **kwargs: Any) -> Any:
            return func(cls, instance, *args, **kwargs)

        return bound


def _is_fitted_name(name: str) -> bool:
    """Public fitted-attribute names: a trailing underscore and no leading one."""
    return name.endswith("_") and not name.startswith("_")


class BaseFactorModel(BaseEstimator):
    """Base class of the public estimators (sklearn conventions, PanelData input).

    Subclasses set ``_tags`` (overrides of :data:`DEFAULT_TAGS`) and
    ``_config_class``, declare keyword-only hyperparameters in ``__init__``
    that are stored unchanged, and implement ``_fit(data, **fit_kwargs)`` plus
    the methods their tags enable.
    """

    _tags: ClassVar[dict] = dict(DEFAULT_TAGS)
    _config_class: ClassVar[Optional[type]] = None
    #: Hyperparameters ``_validate_params`` checks against a fixed set of values: ``{name: values}``,
    #: the ``enum`` of the config's ``json_schema()``.
    _param_choices: ClassVar[dict] = {}
    #: True when the model fills infinite values like missing ones, so ``_validate_data`` lets them through.
    _interpolates_non_finite: ClassVar[bool] = False

    @_class_or_instance_method
    def get_tags(cls, self: Optional[BaseFactorModel] = None) -> dict:
        """Return the tags, merging ``_tags`` along the MRO (base classes first).

        Called on the class (``Model.get_tags()``) it returns the class tags.
        Called on an instance (``model.get_tags()``) it also applies the
        instance's :meth:`_instance_tags`, for tags that depend on
        hyperparameters (for example ``non_deterministic`` when
        ``random_state is None``).

        Raises
        ------
        ConfigurationError
            If a class in the MRO, or ``_instance_tags``, sets a tag name that
            is not in :data:`DEFAULT_TAGS`.
        """
        tags: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_tags")
            if own is None:
                continue
            unknown = [k for k in own if k not in DEFAULT_TAGS]
            if unknown:
                raise ConfigurationError(f"{klass.__name__}._tags: " + _unknown_keys_message(unknown, list(DEFAULT_TAGS), "tag"))
            tags.update(own)
        if self is not None:
            own = self._instance_tags()
            unknown = [k for k in own if k not in DEFAULT_TAGS]
            if unknown:
                raise ConfigurationError(
                    f"{cls.__name__}._instance_tags: " + _unknown_keys_message(unknown, list(DEFAULT_TAGS), "tag")
                )
            tags.update(own)
        return tags

    def _instance_tags(self) -> dict:
        """Tag overrides that depend on this instance's hyperparameters (default: none)."""
        return {}

    def set_params(self, **params: Any) -> BaseFactorModel:
        """Set hyperparameters; an unknown name raises ``ConfigurationError`` naming the closest one."""
        valid = self._get_param_names()
        unknown = [k for k in params if k.split("__", 1)[0] not in valid]
        if unknown:
            raise ConfigurationError(f"{type(self).__name__}.set_params: " + _unknown_keys_message(unknown, valid, "parameter"))
        return super().set_params(**params)

    @classmethod
    def _require_config_class(cls) -> type:
        cfg_cls = cls._config_class
        if cfg_cls is None:
            raise NotImplementedError(f"{cls.__name__} defines no _config_class.")
        return cfg_cls

    def get_config(self) -> EstimatorConfig:
        """Return the config mirror whose fields equal ``get_params(deep=False)``."""
        return self._require_config_class()(**self.get_params(deep=False))

    @classmethod
    def from_config(cls, cfg: EstimatorConfig | Mapping) -> BaseFactorModel:
        """Build the estimator from its config: ``cls(**cfg.to_dict())``.

        A mapping is first validated with ``_config_class.from_dict``, so unknown
        keys raise ``ConfigurationError``.
        """
        cfg_cls = cls._require_config_class()
        if isinstance(cfg, Mapping):
            cfg = cfg_cls.from_dict(cfg)
        elif not isinstance(cfg, cfg_cls):
            raise ConfigurationError(
                f"{cls.__name__}.from_config expects a {cfg_cls.__name__} (or a mapping), got {type(cfg).__name__}."
            )
        return cls(**cfg.to_dict())

    def describe(self) -> dict:
        """A small JSON-serializable summary of the model, for logs, APIs and tool servers.

        Values are only ``str``, ``int``, ``float``, ``bool``, ``None``, lists
        and dicts, and non-finite floats are None, so ``json.dumps`` needs no
        ``default``. The keys:

        - ``model`` (class name), ``version`` (``dfm_python.__version__``),
          ``tags`` (``get_tags()``) and ``fitted`` (bool);
        - ``params``: ``get_params(deep=False)`` with tuples as lists, numpy
          scalars as Python scalars, a ``torch.device`` as its string and any
          other object (a transformer) as its class name.

        A fitted model adds, for each attribute it has: ``n_features_in``,
        ``feature_names``, ``index`` (``start`` and ``end`` as str, int for a
        ``RangeIndex``, ``length``, and ``freq``, inferred for a
        ``DatetimeIndex`` without one), ``n_iter``, ``converged``,
        ``stop_reason``, ``factor_names``, ``explained_variance`` (``{series:
        float or None}``) and the model's own entries (``loglik`` for DFM,
        ``sharpe`` for AFM). A key whose attribute the model lacks is omitted.
        """
        from . import __version__

        out: dict = {
            "model": type(self).__name__,
            "version": str(__version__),
            "params": self.get_params(deep=False),
            "tags": self.get_tags(),
            "fitted": self._is_fitted(),
        }
        if out["fitted"]:
            attrs = vars(self)
            out.update({key: attrs[attr] for key, attr in _DESCRIBED_ATTRIBUTES if attr in attrs})
            if "index" in out:
                out["index"] = _index_summary(out["index"])
            out.update(self._describe_fitted())
        return _json_safe(out)

    def _describe_fitted(self) -> dict:
        """The model's own entries of :meth:`describe` for a fitted model (made JSON-safe there); none by default."""
        return {}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Parameter sets for :func:`dfm_python.testing.check_estimator` (small and fast on CPU)."""
        return [{}]

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. A missing public fitted attribute
        # of an unfitted model raises NotFittedError, which is an AttributeError,
        # so hasattr(model, "result_") is False before fit.
        if _is_fitted_name(name) and not self._is_fitted():
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet, so it has no attribute {name!r}. "
                "Call fit(data) first."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _is_fitted(self) -> bool:
        return any(_is_fitted_name(k) for k in vars(self))

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted()

    def _check_is_fitted(self, method: str) -> None:
        """Raise ``NotFittedError`` naming ``method`` if the model is not fitted."""
        if not self._is_fitted():
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call fit(data) before {method}()."
            )

    def _clear_fitted(self) -> None:
        for name in [k for k in vars(self) if _is_fitted_name(k)]:
            delattr(self, name)

    def _validate_data(self, data: Any) -> PanelData:
        """Coerce ``data`` to ``PanelData`` and reject what the tags exclude.

        Infinite values are rejected in every slot, unless the model fills
        them like missing values (``_interpolates_non_finite``).

        Raises
        ------
        UnsupportedInputError
            For ``exog``, ``series_exog``, a mixed-frequency layout or missing
            values when the matching ``capability:*`` tag is False; the
            message names the tag.
        DataValidationError
            For infinite values, naming the slot and the columns.
        """
        panel = PanelData._coerce(data)
        tags = self.get_tags()
        name = type(self).__name__
        if panel.exog_names and not tags["capability:exog"]:
            raise UnsupportedInputError(
                f"{name} does not accept exog (tag capability:exog is False); got exog columns "
                f"{list(panel.exog_names)}. Pass the data without exog."
            )
        if panel.series_exog_names and not tags["capability:series_exog"]:
            raise UnsupportedInputError(
                f"{name} does not accept series_exog (tag capability:series_exog is False); got features "
                f"{list(panel.series_exog_names)}. Pass the data without series_exog."
            )
        if panel.is_mixed_frequency and not tags["capability:mixed_frequency"]:
            raise UnsupportedInputError(
                f"{name} does not accept mixed-frequency data (tag capability:mixed_frequency is False); got "
                f"freq={panel.freq}. Pass series sampled at one frequency."
            )
        for slot, frame in (("y", panel.y), ("exog", panel.exog), ("series_exog", panel.series_exog)):
            if frame is None:
                continue
            values = frame.to_numpy()
            if not tags["capability:missing_values"] and np.isnan(values).any():
                raise UnsupportedInputError(
                    f"{name} does not accept missing values (tag capability:missing_values is False); "
                    f"{slot} contains NaN. Impute or drop them before fitting."
                )
            infinite = np.isinf(values).any(axis=0)
            if infinite.any() and not self._interpolates_non_finite:
                fix = "finite values or NaN (missing)" if tags["capability:missing_values"] else "finite values"
                raise DataValidationError(
                    f"{name}: {slot} contains infinite values in columns {list(frame.columns[infinite])[:10]}. "
                    f"Replace them with {fix}."
                )
        return panel

    def _check_training_targets(self, panel: PanelData) -> None:
        """Reject training data with target series a forecasting model cannot learn from.

        Each target series of a model with tag ``task="forecast"`` is judged
        on its observed (finite) entries, so a slower series on its
        observation dates only. Allocation models keep their own rules. A
        series is constant when the standard deviation of its ``n`` observed
        values is at most ``n * eps * |mean|`` (``eps`` the float64 machine
        epsilon), the rounding error of computing it, so values equal up to
        rounding count as constant (``StandardScaler`` leaves such a series
        unscaled).

        Raises
        ------
        DataValidationError
            Naming each series with no observed value, a single one, or the
            same value (up to rounding) at every observation.
        """
        if self.get_tags()["task"] != "forecast":
            return
        eps = np.finfo(np.float64).eps
        problems = []
        for series, column in zip(panel.series_names, panel.y.to_numpy().T):
            observed = column[np.isfinite(column)]
            if observed.size == 0:
                problems.append(f"{series!r} has no observed value")
            elif observed.size == 1:
                problems.append(f"{series!r} has a single observed value")
            elif observed.std() <= observed.size * eps * abs(observed.mean()):
                problems.append(
                    f"{series!r} is constant (every observed value is {float(observed[0])!r}, up to rounding)"
                )
        if problems:
            raise DataValidationError(
                f"{type(self).__name__} cannot be fitted on these target series: {'; '.join(problems)}. A target "
                "series needs at least 2 observed values that are not all equal: drop these series from y, or "
                "supply more observations."
            )

    def _set_data_attributes(self, data: PanelData) -> None:
        """Record the fitted data attributes shared by every model."""
        self.n_features_in_ = len(data.series_names)
        self.feature_names_in_ = np.asarray(data.series_names, dtype=object)
        self.exog_names_in_ = np.asarray(data.exog_names, dtype=object)
        if self.get_tags()["capability:series_exog"]:
            self.series_exog_names_in_ = np.asarray(data.series_exog_names, dtype=object)
        self.index_ = data.index

    @staticmethod
    def _validate_horizon(h: Any) -> int:
        """Return ``h`` as an int; a non-integer raises ``TypeError`` and ``h < 1`` raises ``ValueError``."""
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)):
            raise TypeError(f"h must be a positive integer (the forecast horizon), got {type(h).__name__} {h!r}.")
        h = int(h)
        if h < 1:
            raise ValueError(f"h must be a positive integer (the forecast horizon), got {h}.")
        return h

    @staticmethod
    def _forecast_index(index: pd.Index, h: int) -> pd.Index:
        """The ``h`` time stamps after ``index[-1]`` (the forecast index of spec section 6).

        - ``RangeIndex``: continues by position with the same step.
        - ``PeriodIndex``: the next ``h`` periods.
        - ``DatetimeIndex``: the step is ``index.freq``, else ``pd.infer_freq``
          (three or more stamps), else the median step between stamps, for an
          irregular index.

        The name is ``index.name``, or ``"time"`` if it has none.

        Raises
        ------
        DataValidationError
            For a ``DatetimeIndex`` with a single stamp and no ``freq``, or an
            index of another type.
        """
        name = index.name if index.name is not None else "time"
        if isinstance(index, pd.RangeIndex):
            start = index[-1] + index.step
            out: pd.Index = pd.RangeIndex(start, start + index.step * h, index.step)
        elif isinstance(index, pd.PeriodIndex):
            out = pd.period_range(index[-1] + 1, periods=h, freq=index.freq)
        elif isinstance(index, pd.DatetimeIndex):
            freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
            if freq is not None:
                out = pd.date_range(index[-1], periods=h + 1, freq=freq)[1:]
            elif len(index) >= 2:
                step = index.to_series().diff().median()
                out = pd.DatetimeIndex([index[-1] + step * i for i in range(1, h + 1)])
            else:
                raise DataValidationError(
                    "Cannot build a forecast index after a single time stamp without a frequency; give the "
                    "DatetimeIndex a freq (for example pd.date_range(..., freq='ME'))."
                )
        else:
            raise DataValidationError(
                f"Cannot build a forecast index after a {type(index).__name__}; use a DatetimeIndex, PeriodIndex "
                "or RangeIndex."
            )
        return out.rename(name)

    @staticmethod
    def _extend_index(index: pd.Index, new: pd.Index, name: str) -> pd.Index:
        """``index`` followed by ``new`` (later stamps), keeping the training index's type, name and frequency.

        A ``RangeIndex`` stays a ``RangeIndex``: the new rows must continue it
        (start at its stop, same step), since positions count clock periods.
        A ``DatetimeIndex`` keeps its ``freq`` when the new stamps continue it
        at that frequency (otherwise the forecast index infers the step).

        Raises
        ------
        DataValidationError
            For a ``RangeIndex`` that the new rows do not continue.
        """
        if isinstance(index, pd.RangeIndex):
            if new.start != index.stop or (len(new) > 1 and new.step != index.step):
                raise DataValidationError(
                    f"{name}: the new rows are at positions {new.start}..{new[-1]} (step {new.step}), but a RangeIndex "
                    f"must continue the model's index at {index.stop} with step {index.step}: each row is one clock "
                    "period."
                )
            return pd.RangeIndex(index.start, new.start + index.step * len(new), index.step, name=index.name)
        out = index.append(new).rename(index.name)
        if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
            expected = pd.date_range(index[0], periods=len(out), freq=index.freq, tz=index.tz)
            if out.equals(expected):
                out = pd.DatetimeIndex(out, freq=index.freq, name=index.name)
        return out

    def _extend_conditioning_index(self, index: pd.Index, name: str) -> pd.Index:
        """``conditioning_index_`` followed by ``index``, the index of new rows (tag ``predict_data="new_rows"``).

        ``index`` must be of the conditioning index's type and start strictly
        after its last stamp; see :meth:`_extend_index` for how the two join.

        Raises
        ------
        DataValidationError
            For an index of another type, rows that overlap the conditioning
            data, or a ``RangeIndex`` that does not continue the model's.
        """
        last = self.conditioning_index_
        if type(index) is not type(last):
            raise DataValidationError(
                f"{name}: data has a {type(index).__name__}, but the model's time index is a "
                f"{type(last).__name__}. Pass the new rows on the same kind of index."
            )
        if not index[0] > last[-1]:
            raise DataValidationError(
                f"{name}: data starts at {index[0]!r}, which overlaps the conditioning data (it ends at "
                f"{last[-1]!r}). Pass only the rows after {last[-1]!r} (tag predict_data='new_rows')."
            )
        return self._extend_index(last, index, name)

    def fit(self, data: Any, *, callback: Optional[Callable[[dict], Any]] = None) -> BaseFactorModel:
        """Fit on ``data`` (``PanelData``, a DataFrame or a 2-D array) and return ``self``.

        Refitting starts from scratch: earlier fitted attributes are removed
        first, and also when fitting fails (an exception raised by
        ``callback`` included). Progress output follows the ``verbose``
        hyperparameter (0 silent, 1 progress, 2 debug). ``callback(info)`` is
        called after every training iteration with the iteration's info dict
        (see the model's ``fit``); a true return value stops training there.
        When the ``max_time`` budget stopped training (``stop_reason_ ==
        "max_time"``), ``fit`` warns with ``ConvergenceWarning``.

        Raises
        ------
        TypeError
            If ``callback`` is neither None nor callable.
        """
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be None or a callable taking the info dict, got {type(callback).__name__}.")
        panel = self._validate_data(data)
        self._check_training_targets(panel)
        verbose = getattr(self, "verbose", 0)
        self._clear_fitted()
        try:
            with fit_verbosity(verbose):
                self._set_data_attributes(panel)
                self._fit(panel, callback=callback)
            self._warn_if_out_of_time(stacklevel=4)  # _warn_if_out_of_time < BaseFactorModel.fit < Model.fit < caller
        except BaseException:
            self._clear_fitted()
            raise
        return self

    def _fit(self, data: PanelData, *, callback: Optional[Callable[[dict], Any]] = None) -> None:
        """Train on validated data and set the fitted attributes (``result_``, ``stop_reason_`` and the rest)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _fit().")

    def _warn_if_out_of_time(self, stacklevel: int) -> None:
        """Warn with ``ConvergenceWarning`` when the ``max_time`` budget stopped the last training run."""
        if vars(self).get("stop_reason_") == "max_time":
            warnings.warn(
                f"{type(self).__name__}: training stopped after {self.n_iter_} iteration(s) because it ran past "
                f"max_time={self.max_time} seconds; the model keeps the parameters of that iteration and converged_ "
                "is False. Raise max_time, or set it to None, to train longer.",
                ConvergenceWarning,
                stacklevel=stacklevel,
            )

    def predict(self, h: int, data: Any = None, **predict_kwargs: Any) -> pd.DataFrame:
        """Forecast ``h`` steps as a time-indexed DataFrame (``task == "forecast"``)."""
        if self.get_tags()["task"] != "forecast":
            raise TypeError(
                f"{type(self).__name__} is an allocation model (tag task='allocation') and has no predict(h); "
                "use allocate(data=None) for portfolio weights."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement predict().")

    def predict_factors(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (``capability:factor_forecast``)."""
        if not self.get_tags()["capability:factor_forecast"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not forecast factors (tag capability:factor_forecast is False)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement predict_factors().")

    def update(self, data: Any, *, refit: bool = False) -> BaseFactorModel:
        """Condition later predictions on ``data``; parameters change only with ``refit=True`` (``capability:update``)."""
        if not self.get_tags()["capability:update"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not support update (tag capability:update is False); "
                "call fit(data) to refit on the new data."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement update().")

    def predict_interval(self, h: int, coverage: float = 0.9, data: Any = None) -> pd.DataFrame:
        """Central prediction intervals of the ``h`` steps ahead (``capability:pred_int``).

        Returns a DataFrame indexed like :meth:`predict`, with MultiIndex
        columns ``(series, bound)`` where ``bound`` is ``"lower"`` or
        ``"upper"``, in data units; ``attrs["coverage"]`` holds ``coverage``.
        ``data`` conditions on new rows as in :meth:`predict`.
        """
        if not self.get_tags()["capability:pred_int"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} does not give prediction intervals (tag capability:pred_int is False)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement predict_interval().")

    def decompose(self, *, by: str = "component") -> pd.DataFrame:
        """Split the training data into its parts (``capability:interpret``), in data units.

        ``by="component"``: MultiIndex columns ``(component, series)`` with
        ``component`` ``"common"`` (the factors through the loadings, plus the
        series mean) and ``"idiosyncratic"`` (data minus common, NaN where the
        data is missing); ``common + idiosyncratic`` equals the data.
        ``by="factor"``: MultiIndex columns ``(part, series)`` with one ``part``
        per factor name (its contribution), ``"mean"`` and
        ``"idiosyncratic"``; the parts add up to the data. Only models whose
        observations are linear in the factors support ``by="factor"``.
        Indexed by ``index_``.
        """
        if not self.get_tags()["capability:interpret"]:
            raise UnsupportedInputError(
                f"{type(self).__name__} has no factor decomposition (tag capability:interpret is False)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement decompose().")

    def set_output(self, *, output: str = "pandas") -> BaseFactorModel:
        """Choose the frame type the prediction methods return and return ``self``.

        ``output="polars"`` makes :meth:`predict`, :meth:`predict_factors`,
        :meth:`predict_interval`, :meth:`decompose` and :meth:`allocate` return
        ``polars.DataFrame``: the index becomes the first column (named after
        the index, or ``"time"``) and MultiIndex columns are joined with
        ``"__"`` (``"s0__lower"``). ``"pandas"`` (the default) returns pandas.
        Fitted attributes stay pandas. The choice is not a hyperparameter: it
        is not cloned or saved. Needs the ``polars`` extra.
        """
        from .utils.output import OUTPUTS, require_polars

        if output not in OUTPUTS:
            raise ConfigurationError(f"output must be one of {list(OUTPUTS)}, got {output!r}.")
        if output == "polars":
            require_polars()
        self._output = output
        return self

    def _emit(self, frame: Any) -> Any:
        """``frame`` in the type :meth:`set_output` chose (pandas by default)."""
        from .utils.output import to_output

        return to_output(frame, getattr(self, "_output", "pandas"))

    def allocate(self, data: Any = None) -> pd.DataFrame:
        """Return portfolio weights as a time-indexed DataFrame (``task == "allocation"``)."""
        if self.get_tags()["task"] != "allocation":
            raise TypeError(
                f"{type(self).__name__} is a forecasting model (tag task={self.get_tags()['task']!r}) and has no "
                "allocate(); use predict(h)."
            )
        raise NotImplementedError(f"{type(self).__name__} does not implement allocate().")

    def save(self, path: Any, *, overwrite: bool = False) -> None:
        """Save the fitted model as a pickle-free directory (spec section 7).

        The run is written to a temporary sibling directory, reloaded with
        ``type(self).load(tmp, device="cpu")`` to verify it, and then renamed
        to ``path``. The base class writes ``config.json`` (format version,
        model type, library version, ``get_params()``, tags and runtime) and
        ``data_spec.json`` (feature, exog and series-exog names and the training
        index); the model's ``_save_state`` hook writes ``state.npz``,
        ``state.json``, ``result.npz``, ``result.json`` and, for torch models,
        ``weights.pt``.

        Parameters
        ----------
        path : str or os.PathLike
            The directory to create.
        overwrite : bool, default False
            Replace an existing ``path`` (the old run is removed only after the
            new one is in place).

        Raises
        ------
        NotFittedError
            Before ``fit``.
        FileExistsError
            If ``path`` exists and ``overwrite`` is False.

        Warns
        -----
        UserWarning
            When the model has a ``random_state`` parameter set to None: the
            saved config cannot reproduce the run.
        """
        self._check_is_fitted("save")
        params = self.get_params(deep=False)
        if "random_state" in params and params["random_state"] is None:
            warnings.warn(
                f"Saving {type(self).__name__} with random_state=None: config.json cannot reproduce this run. "
                "Set random_state to an integer before fit for reproducible runs.",
                UserWarning,
                stacklevel=2,
            )
        # Converted up front, so a value JSON cannot hold fails before anything is written.
        config = to_jsonable(self._saved_config(params), where=CONFIG_FILE)
        data_spec = to_jsonable(self._saved_data_spec(), where=DATA_SPEC_FILE)
        cls = type(self)

        def write(tmp: Path) -> None:
            write_json(tmp / CONFIG_FILE, config)
            write_json(tmp / DATA_SPEC_FILE, data_spec)
            self._save_state(tmp)

        def verify(tmp: Path) -> None:
            # allow_pickle: the directory was written just now by this process.
            loaded = cls.load(tmp, device="cpu", allow_pickle=True)
            if not loaded._is_fitted() or "result_" not in vars(loaded):
                raise RuntimeError(f"{cls.__name__}.load did not restore result_; the save was not completed.")

        atomic_save_dir(path, write, verify, overwrite=overwrite)

    @classmethod
    def load(cls, path: Any, *, device: Any = None, allow_pickle: bool = False) -> BaseFactorModel:
        """Load a run written by :meth:`save`; no dataset is needed.

        Builds ``cls(**params)`` from ``config.json`` (through ``from_config``
        when the model has a config mirror), restores the data attributes and
        ``scaler_``, and calls the model's ``_load_state`` hook.

        Parameters
        ----------
        path : str or os.PathLike
            The saved-run directory.
        device : str or torch.device, optional
            Device for torch models; None uses the ``device`` parameter
            (``"auto"``: CUDA if available, else CPU). Weights are read with
            ``map_location``, so a GPU-saved run loads on a CPU-only machine:
            a saved CUDA ``device`` parameter falls back to CPU there with a
            ``UserWarning``, and an explicit CUDA ``device`` raises
            ``ConfigurationError``.
        allow_pickle : bool, default False
            Allow reading ``scaler.pkl`` and importing estimator classes outside
            ``sklearn.preprocessing`` named in ``config.json``. Both can run
            arbitrary code; enable only for directories you trust.

        Raises
        ------
        ConfigurationError
            If ``path`` is a single-file checkpoint from an earlier release or
            not a saved run, holds another model type, lacks a file the
            ``backend`` tag requires, or needs ``allow_pickle=True``.
        """
        path = Path(path)
        config = read_config(path, allow_import=allow_pickle)
        model_type = config["model_type"]
        if model_type != cls.__name__:
            raise ConfigurationError(
                f"{str(path)!r} holds a {model_type!r} run, not {cls.__name__}. Load it with "
                f"{model_type}.load(path) or dfm_python.load(path)."
            )
        cls._check_saved_files(path)
        params = dict(config["params"])
        encoded = sorted(name for name, value in params.items() if _contains_encoded_estimator(value))
        if encoded:
            raise ConfigurationError(
                f"config.json in {str(path)!r} stores estimator classes outside sklearn.preprocessing for "
                f"{', '.join(map(repr, encoded))}. Importing them can run arbitrary code, so it is off by default: "
                "pass allow_pickle=True only if you trust the directory."
            )
        if cls._config_class is not None:
            est = cls.from_config(params)
        else:
            try:
                est = cls(**params)
            except TypeError as err:
                raise ConfigurationError(f"config.json in {str(path)!r} does not match {cls.__name__}: {err}") from err
        est._restore_data_attributes(read_json(path / DATA_SPEC_FILE))
        if has_scaler(path):
            est.scaler_ = read_scaler(path, read_npz(path / STATE_NPZ), allow_pickle=allow_pickle)
        cls._load_state(est, path, device, allow_pickle)
        return est

    @classmethod
    def _check_saved_files(cls, path: Path) -> None:
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            raise ConfigurationError(f"{str(path)!r} is not a complete saved run: {', '.join(missing)} missing.")
        has_weights = (path / WEIGHTS_FILE).is_file()
        backend = cls.get_tags()["backend"]
        if backend == "torch" and not has_weights:
            raise ConfigurationError(f"{str(path)!r} has no {WEIGHTS_FILE}, which a torch model ({cls.__name__}) needs.")
        if backend != "torch" and has_weights:
            raise ConfigurationError(
                f"{str(path)!r} has a {WEIGHTS_FILE}, but {cls.__name__} has backend {backend!r} and keeps no torch weights."
            )

    def _saved_config(self, params: dict) -> dict:
        """Content of ``config.json``."""
        import torch

        from . import __version__

        device = vars(self).get("device_")
        return {
            "format_version": FORMAT_VERSION,
            "model_type": type(self).__name__,
            "library_version": str(__version__),
            "params": params,
            "tags": self.get_tags(),
            "runtime": {
                "device": None if device is None else str(device),
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "num_threads": int(torch.get_num_threads()),
            },
        }

    def _saved_data_spec(self) -> dict:
        """Content of ``data_spec.json``: names in the user's order and the training index."""
        spec = {
            "feature_names_in_": list(self.feature_names_in_),
            "exog_names_in_": list(self.exog_names_in_),
            "index_": index_to_spec(self.index_),
        }
        if "series_exog_names_in_" in vars(self):
            spec["series_exog_names_in_"] = list(self.series_exog_names_in_)
        return spec

    def _restore_data_attributes(self, spec: Mapping) -> None:
        """Inverse of :meth:`_saved_data_spec` (the attributes :meth:`_set_data_attributes` sets)."""
        names = list(spec["feature_names_in_"])
        self.n_features_in_ = len(names)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.exog_names_in_ = np.asarray(list(spec["exog_names_in_"]), dtype=object)
        if "series_exog_names_in_" in spec:
            self.series_exog_names_in_ = np.asarray(list(spec["series_exog_names_in_"]), dtype=object)
        self.index_ = index_from_spec(spec["index_"])

    def _save_state(self, path: Path) -> None:
        """Write ``state.npz``, ``state.json``, ``result.npz``, ``result.json`` (and ``weights.pt``) into ``path``.

        Use :meth:`_write_state`, :meth:`_write_result` and :meth:`_write_weights`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _save_state(), so it cannot be saved.")

    @classmethod
    def _load_state(cls, est: BaseFactorModel, path: Path, device: Any, allow_pickle: bool) -> None:
        """Restore the fitted state of ``est`` (already built, with data attributes and ``scaler_``) from ``path``.

        Use :meth:`_read_state`, :meth:`_read_result`, :meth:`_read_weights`
        and :meth:`_load_device`.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement _load_state(), so it cannot be loaded.")

    def _write_state(self, path: Path, arrays: Optional[Mapping] = None, meta: Optional[Mapping] = None) -> None:
        """Write ``state.npz`` (``arrays`` plus ``scaler/<attr>``) and ``state.json`` (``meta``).

        When the model has a ``scaler_`` attribute it is also written:
        ``scaler.json`` for None and the five known scalers, ``scaler.pkl``
        for any other.
        """
        arrays = dict(arrays or {})
        reserved = [key for key in arrays if key.startswith(SCALER_PREFIX)]
        if reserved:
            raise ValueError(f"state.npz keys {reserved} use the reserved prefix {SCALER_PREFIX!r}.")
        if "scaler_" in vars(self):
            arrays.update(write_scaler(path, self.scaler_))
        write_npz(Path(path) / STATE_NPZ, arrays)
        write_json(Path(path) / STATE_JSON, dict(meta or {}))

    @staticmethod
    def _read_state(path: Path) -> tuple[dict, dict]:
        """Return ``(arrays, meta)`` from ``state.npz`` (without the ``scaler/`` keys) and ``state.json``."""
        arrays = {k: v for k, v in read_npz(Path(path) / STATE_NPZ).items() if not k.startswith(SCALER_PREFIX)}
        return arrays, read_json(Path(path) / STATE_JSON)

    def _write_result(self, path: Path, result: Any = None, *, exclude: Iterable[str] = ()) -> None:
        """Write ``result_`` (or ``result``) to ``result.npz`` and ``result.json``; ``exclude`` skips fields."""
        arrays, meta = result_to_files(self.result_ if result is None else result, exclude=exclude)
        write_npz(Path(path) / RESULT_NPZ, arrays)
        write_json(Path(path) / RESULT_JSON, meta)

    @staticmethod
    def _read_result(path: Path, result_cls: type, **extra: Any) -> Any:
        """Read ``result.npz`` and ``result.json`` back into ``result_cls``; ``extra`` gives excluded fields."""
        path = Path(path)
        return result_from_files(result_cls, read_npz(path / RESULT_NPZ), read_json(path / RESULT_JSON), **extra)

    def _write_weights(self, path: Path, module: Any = None) -> None:
        """Write ``weights.pt``: the ``state_dict()`` of ``module`` (default ``network_``)."""
        module = self.network_ if module is None else module
        write_weights(Path(path) / WEIGHTS_FILE, module.state_dict())

    @staticmethod
    def _read_weights(path: Path, device: Any = None) -> dict:
        """Read ``weights.pt`` with ``map_location=device`` and ``weights_only=True``."""
        return read_weights(Path(path) / WEIGHTS_FILE, map_location=device)

    def _load_device(self, device: Any = None) -> Any:
        """Resolve the load device: ``device``, else the ``device`` parameter; ``"auto"`` means CUDA if available.

        A saved CUDA ``device`` parameter falls back to CPU with a
        ``UserWarning`` when CUDA is not available, so a GPU-saved run loads on
        a CPU-only machine. An explicit ``load(device=...)`` naming CUDA there
        raises ``ConfigurationError``.
        """
        import torch

        explicit = device is not None
        if device is None:
            device = self.get_params(deep=False).get("device", "auto")
        if device is None or (isinstance(device, str) and device == "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = type(self).__name__
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError, ValueError) as err:
            source = "load(device=...)" if explicit else "the saved device parameter"
            raise ConfigurationError(f"{name}.load: {source} {device!r} is not a valid torch device ({err}).") from err
        if resolved.type == "cuda" and not torch.cuda.is_available():
            if explicit:
                raise ConfigurationError(
                    f"{name}.load(device={device!r}): CUDA is not available on this machine. "
                    "Pass device='cpu' (or 'auto')."
                )
            warnings.warn(
                f"The {name} run was saved with device={str(device)!r}, but CUDA is not available on this machine; "
                "loading it on CPU. Pass load(path, device='cpu') to choose the device explicitly.",
                UserWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return resolved


def _json_safe(value: Any) -> Any:
    """``value`` as JSON builtins for :meth:`BaseFactorModel.describe`.

    numpy scalars become Python scalars and non-finite floats None; tuples,
    lists and arrays become lists, and mappings and ``pandas.Series`` dicts
    with string keys; a ``torch.device`` becomes its string and any other
    object its class name.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if _is_torch_device(value):
        return str(value)
    return type(value).__name__


def _index_summary(index: pd.Index) -> dict:
    """``{"start", "end", "length", "freq"}`` of a training index: stamps as str (int for a ``RangeIndex``).

    ``freq`` is the index's frequency string; a ``DatetimeIndex`` without one
    gets what ``pd.infer_freq`` finds (three or more stamps), as forecasts do.
    Otherwise it is None.
    """
    stamp = int if isinstance(index, pd.RangeIndex) else str
    freq = getattr(index, "freqstr", None)
    if freq is None and isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        freq = pd.infer_freq(index)
    return {"start": stamp(index[0]), "end": stamp(index[-1]), "length": len(index), "freq": freq}


def _contains_encoded_estimator(value: Any) -> bool:
    if is_encoded_estimator(value):
        return True
    if isinstance(value, dict):
        return any(_contains_encoded_estimator(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_encoded_estimator(v) for v in value)
    return False

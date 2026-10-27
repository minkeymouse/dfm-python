"""The private estimator base classes: ``_FactorModel``, ``_Forecaster`` and ``_Allocator``.

Every model subclasses one of the two leaves, which build on
:class:`sklearn.base.BaseEstimator` (spec sections 1 and 4):

- ``__init__`` takes keyword-only hyperparameters and only stores them, so
  ``get_params``, ``set_params`` (an unknown name raises
  ``ConfigurationError`` naming the closest one) and ``clone`` work. The
  metadata-request dunders keep sklearn from generating
  ``set_fit_request`` and ``set_predict_request``.
- ``fit(data, *, callback=None)`` coerces ``data`` to :class:`PanelData`,
  validates the hyperparameters (``_validate_params``) and the data (the slots
  the class accepts, infinite values, the regular grid, the clock and the
  slower-series anchors of :mod:`dfm_python._time`, and the training
  targets), clears the earlier fitted state, estimates (``_estimate``),
  records the diagnostics and conditions on the training data
  (``_condition``). Any failure leaves the model unfitted.
- ``update(data, *, replace=False)`` merges ``data`` into the conditioning
  panel (``PanelData._merge``), checks the merged grid and conditions again;
  the estimated parameters and the diagnostics never change.
- ``save(path)`` writes the format-2 directory of spec section 10 through
  :func:`~dfm_python.io.directory.atomic_save_dir` and verifies it by
  reloading; :meth:`_FactorModel._load` reads it and conditions again.
- Fitted state lives in attributes ending in ``_`` (public) or starting and
  ending in ``_`` (private); before ``fit`` a public one raises
  :class:`~dfm_python.errors.NotFittedError`, an ``AttributeError``.

:class:`_Forecaster` adds ``predict``, ``predict_interval``,
``predict_factors`` and ``nowcast`` over the hooks ``_moments`` and
``_factor_path`` (spec sections 4.4 to 4.8); :class:`_Allocator` adds
``allocate`` over ``_next_weights``.

The model hooks:

=====================  ========================================================
Hook                   Contract
=====================  ========================================================
``_validate_params``   raise ``ConfigurationError`` for an invalid hyperparameter
``_estimate``          ``(panel, log) -> stop_reason``: estimate on the training
                       panel, call ``log(objective=..., **extras)`` after every
                       iteration (a True return means the callback asked to
                       stop), fix the reporting transform, set ``params_``
``_condition``         ``(panel, previous)``: compute ``states_`` (and
                       ``factors_``) from the conditioning panel with the fixed
                       parameters; ``previous`` is the panel before an
                       ``update`` (None at ``fit`` and ``load``)
``_save_fitted``       ``() -> (arrays, values)``: the model state for
                       ``fitted.npz`` / ``fitted.json``
``_load_fitted``       ``(arrays, values, *, weights, device)``: restore it
``_fitted_weights``    torch models: the ``state_dict`` for ``weights.pt``
``_moments``           forecasters: ``(stamps, exog, coverage) -> (mean, lower,
                       upper)`` in data units, (n_targets, N) in the user's
                       series order; ``lower``/``upper`` None without coverage
``_factor_path``       forecasters: ``(stamps, exog) -> (n_targets, r)``
``_next_weights``      allocators: the next-period weights over every known asset
``_explicit_clock``    forecasters: the clock hyperparameter (DFM), else None
``_test_params``       parameter sets for :func:`dfm_python._testing.check_estimator`
=====================  ========================================================
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import math
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.metadata_routing import UNUSED

from . import _time
from .dataset.panel import PanelData
from .errors import ConfigurationError, DataValidationError, NotFittedError
from .io.codecs import (
    decode_params,
    encode_params,
    read_json_plain,
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
    DATA_JSON,
    DATA_NPZ,
    FITTED_JSON,
    FITTED_NPZ,
    FORMAT_2,
    REQUIRED_FILES_2,
    WEIGHTS_FILE,
    index_from_spec,
    index_to_spec,
    offset_from_spec,
    offset_to_spec,
    read_config_2,
)
from .logger.progress import fit_verbosity, quiet_engine_loggers
from .utils.optional import require_torch

__all__ = ["STOP_REASONS", "TrainingLog", "_Allocator", "_FactorModel", "_Forecaster"]

#: ``stop_reason_`` vocabulary (spec section 4.9).
STOP_REASONS = ("converged", "early_stopping", "max_iter", "callback", "numerical_guard", "not_trainable")
_CONVERGED = frozenset({"converged", "early_stopping", "not_trainable"})
_WARNED = frozenset({"max_iter", "numerical_guard"})
_MODEL_PREFIX = "model/"


def _is_fitted_name(name: str) -> bool:
    """Public fitted-attribute names: a trailing underscore and no leading one."""
    return name.endswith("_") and not name.startswith("_")


def _is_state_name(name: str) -> bool:
    """Every fitted attribute: public ones and private ones like ``_panel_`` (not dunders)."""
    return name.endswith("_") and not name.endswith("__")


def _closest_name_message(unknown: Sequence[str], valid: Sequence[str], what: str) -> str:
    parts = []
    for name in unknown:
        close = difflib.get_close_matches(name, list(valid), n=1)
        parts.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return f"unknown {what}{'s' if len(unknown) > 1 else ''} {', '.join(parts)}; valid {what}s: {sorted(valid)}."


# ---------------------------------------------------------------------------
# Training history and the callback
# ---------------------------------------------------------------------------


class TrainingLog:
    """The per-iteration history rows of a fit, and the user's callback.

    The training loop calls ``log(objective, **extras)`` after every
    iteration. The row is ``{"objective", **extras, "elapsed"}`` (seconds
    since the log was created) and ``callback({"iteration": i, **row})`` runs
    with it; the call returns True when the callback returned a truthy value,
    asking training to stop. An exception from the callback propagates.

    Parameters
    ----------
    callback : callable, optional
    """

    def __init__(self, callback: Optional[Callable[[dict], Any]] = None) -> None:
        self._callback = callback
        self._rows: list[dict] = []
        self._start = time.perf_counter()

    def __call__(self, objective: float, **extras: Any) -> bool:
        row = {"objective": float(objective)}
        for key, value in extras.items():
            row[key] = value.item() if isinstance(value, np.generic) else value
        row["elapsed"] = time.perf_counter() - self._start
        self._rows.append(row)
        if self._callback is None:
            return False
        return bool(self._callback({"iteration": len(self._rows), **row}))

    @property
    def n_iter(self) -> int:
        """Iterations logged so far."""
        return len(self._rows)

    def frame(self) -> pd.DataFrame:
        """``history_``: one row per iteration, indexed by ``iteration`` (1..n), ``objective`` first, ``elapsed`` last."""
        columns = ["objective"]
        for row in self._rows:
            columns += [key for key in row if key not in columns and key != "elapsed"]
        columns.append("elapsed")
        index = pd.RangeIndex(1, len(self._rows) + 1, name="iteration")
        data = {col: [row.get(col, np.nan) for row in self._rows] for col in columns}
        frame = pd.DataFrame(data, index=index, columns=columns)
        for col in ("objective", "elapsed"):
            frame[col] = frame[col].astype(np.float64)
        return frame


# ---------------------------------------------------------------------------
# The data layout of a forecaster (grid, clock, slower series)
# ---------------------------------------------------------------------------


@dataclass
class _Layout:
    """The time layout fixed at fit: the grid step, the clock and the slower series (spec section 4.12)."""

    grid: Optional[pd.DateOffset] = None
    clock: Optional[str] = None
    slower: dict = field(default_factory=dict)
    residues: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# _FactorModel
# ---------------------------------------------------------------------------


class _FactorModel(BaseEstimator):
    """Base of every model: the sklearn shell, fit / update / save / load and the data checks.

    Subclasses set the class attributes below, declare keyword-only
    hyperparameters in ``__init__`` that are stored unchanged, and implement
    the hooks listed in the module docstring.
    """

    #: The model accepts ``exog`` (DDFM, iVDFM, AFM).
    _accepts_exog: ClassVar[bool] = False
    #: The model accepts ``characteristics`` (AFM).
    _accepts_characteristics: ClassVar[bool] = False
    #: The model accepts series slower than the clock (DFM).
    _accepts_mixed_frequency: ClassVar[bool] = False
    #: Rows must be consecutive clock periods (the forecasters); allocators take any increasing index.
    _regular_grid: ClassVar[bool] = True
    #: ``update`` may add series (AFM assets), appended with NaN history.
    _new_series_on_update: ClassVar[bool] = False
    #: The model needs torch (the ``deep`` extra) to fit and load, and saves ``weights.pt``.
    _requires_torch: ClassVar[bool] = False
    #: ``ConvergenceWarning`` when training stops at ``max_iter`` or ``numerical_guard`` (DFM, DDFM).
    _warns_on_nonconvergence: ClassVar[bool] = False
    #: Loggers raised to ERROR inside fit, update and predict when ``verbose=0``.
    _engine_loggers: ClassVar[tuple] = ()

    __metadata_request__fit = {"data": UNUSED, "callback": UNUSED}

    # sklearn shell

    def set_params(self, **params: Any) -> _FactorModel:
        """Set hyperparameters; an unknown name raises ``ConfigurationError`` naming the closest one."""
        valid = self._get_param_names()
        unknown = [k for k in params if k.split("__", 1)[0] not in valid]
        if unknown:
            raise ConfigurationError(f"{type(self).__name__}.set_params: " + _closest_name_message(unknown, valid, "parameter"))
        return super().set_params(**params)

    @classmethod
    def _test_params(cls) -> list[dict]:
        """Parameter sets for :func:`dfm_python._testing.check_estimator` (small and fast on CPU)."""
        return [{}]

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. A missing public fitted
        # attribute of an unfitted model raises NotFittedError, which is an
        # AttributeError, so hasattr(model, "params_") is False before fit.
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
        for name in [k for k in vars(self) if _is_state_name(k)]:
            delattr(self, name)

    @contextlib.contextmanager
    def _verbosity(self) -> Iterator[None]:
        """Logging for the duration of a call: ``verbose`` 0 silent (engine loggers at ERROR), 1 INFO, 2 DEBUG."""
        verbose = getattr(self, "verbose", 0)
        with fit_verbosity(verbose), quiet_engine_loggers(verbose, self._engine_loggers, logging.ERROR):
            yield

    # Hooks

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` for an invalid hyperparameter (called by ``fit``)."""

    def _estimate(self, panel: PanelData, log: TrainingLog) -> str:
        """Estimate on the validated training panel; return ``stop_reason_``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _estimate().")

    def _condition(self, panel: PanelData, previous: Optional[PanelData]) -> None:
        """Compute ``states_`` and ``factors_`` from the conditioning panel with the fitted parameters."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _condition().")

    def _save_fitted(self) -> tuple[dict, dict]:
        """``(arrays, values)``: the model state for ``fitted.npz`` and ``fitted.json``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _save_fitted(), so it cannot be saved.")

    def _load_fitted(self, arrays: dict, values: dict, *, weights: Optional[Mapping] = None, device: Any = None) -> None:
        """Restore the model state written by :meth:`_save_fitted` (and ``weights.pt`` for torch models)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _load_fitted(), so it cannot be loaded.")

    def _fitted_weights(self) -> Optional[Mapping]:
        """Torch models: the ``state_dict`` written to ``weights.pt``."""
        return None

    def _explicit_clock(self) -> Optional[str]:
        """The clock hyperparameter, for a model that has one (DFM); None infers the clock from the index."""
        return None

    def _reload_output(self) -> pd.DataFrame:
        """The output ``save`` compares after reloading (``predict(1)`` or ``allocate()``)."""
        raise NotImplementedError

    def _check_training_data(self, panel: PanelData, what: str) -> None:
        """Extra checks of the training data (forecasters: the targets)."""

    # Data validation

    def _what(self, method: str) -> str:
        return f"{type(self).__name__}.{method}"

    def _check_slots(self, panel: PanelData, what: str) -> None:
        """Reject the slots the model does not accept and infinite values, in plain words."""
        name = type(self).__name__
        if panel.exog_names and not self._accepts_exog:
            raise DataValidationError(
                f"{what}: {name} does not accept exog; got exog columns {list(panel.exog_names)}. Pass the data "
                "without exog."
            )
        if panel.characteristic_names and not self._accepts_characteristics:
            raise DataValidationError(
                f"{what}: {name} does not accept characteristics; got {list(panel.characteristic_names)}. Pass the "
                "data without characteristics."
            )
        slots = (
            ("y", panel._y.to_numpy(), panel.series_names, 0),
            ("exog", None if panel._exog is None else panel._exog.to_numpy(), panel.exog_names, 0),
            ("characteristics", panel._chars, panel.characteristic_names, (0, 1)),
        )
        for slot, values, names, axis in slots:
            if values is None:
                continue
            infinite = np.isinf(values).any(axis=axis)
            if infinite.any():
                bad = [names[j] for j in np.flatnonzero(infinite)][:10]
                raise DataValidationError(
                    f"{what}: {slot} contains infinite values in {bad}. Replace them with finite values or NaN "
                    "(missing)."
                )

    def _layout(self, panel: PanelData, what: str) -> _Layout:
        """The time layout of training data: grid, clock, slower series and their anchors (spec section 4.12)."""
        index = panel.index
        if not self._regular_grid:
            if panel._is_mixed_frequency() and not self._accepts_mixed_frequency:
                raise DataValidationError(
                    f"{what}: {type(self).__name__} does not accept mixed-frequency data; got freq={panel.freq}. "
                    "Pass series sampled at one frequency."
                )
            return _Layout()
        grid = _time.grid_freq(index, what=what)
        clock = _time.resolve_clock(index, grid, self._explicit_clock(), what=what)
        slower = _time.slower_series(
            panel.freq, panel.series_names, clock, is_range=isinstance(index, pd.RangeIndex), what=what
        )
        if slower and not self._accepts_mixed_frequency:
            raise DataValidationError(
                f"{what}: {type(self).__name__} does not accept mixed-frequency data; got series {list(slower)} "
                f"slower than the {clock!r} clock (freq={panel.freq}). Pass series sampled at the clock frequency."
            )
        residues = {}
        if slower:
            found = _time.check_anchors(
                panel._y.to_numpy(), panel.series_names, slower, _time.Grid(index, grid, clock), what=what
            )
            residues = {k: v for k, v in found.items() if v is not None}
        return _Layout(grid=grid, clock=clock, slower=slower, residues=residues)

    def _check_layout(self, panel: PanelData, what: str) -> None:
        """Check a merged conditioning panel against the layout fixed at fit (``update``)."""
        if not self._regular_grid:
            return
        _time.check_on_grid(panel.index, self._grid_, what=what)
        if self._slower_:
            grid = _time.Grid(panel.index, self._grid_, self._clock_)
            _time.check_anchors(
                panel._y.to_numpy(), panel.series_names, self._slower_, grid, residues=self._residues_, what=what
            )

    # fit

    def fit(self, data: Any, *, callback: Optional[Callable[[dict], Any]] = None) -> _FactorModel:
        """Estimate on ``data`` and condition on it; return ``self``.

        Parameters
        ----------
        data : PanelData, pandas or polars DataFrame, or 2-D ndarray
        callback : callable, optional
            ``callback(info)`` runs after every training iteration with
            ``info = {"iteration": i, **history_ row}``; a truthy return stops
            training (``stop_reason_ == "callback"``, no warning). An exception
            it raises propagates and leaves the model unfitted.

        Raises
        ------
        TypeError
            If ``callback`` is neither None nor callable.
        ImportError
            For a torch model when torch (the ``deep`` extra) is not installed.
        ConfigurationError
            For an invalid hyperparameter.
        DataValidationError
            For data the model cannot use.
        """
        if callback is not None and not callable(callback):
            raise TypeError(f"callback must be None or a callable taking the info dict, got {type(callback).__name__}.")
        what = self._what("fit")
        try:
            if self._requires_torch:
                require_torch(what)
            self._validate_params()
            panel = PanelData._coerce(data)
            self._check_slots(panel, what)
            layout = self._layout(panel, what)
            self._check_training_data(panel, what)
            self._clear_fitted()
            with self._verbosity():
                self._set_data_attributes(panel)
                self._grid_, self._clock_ = layout.grid, layout.clock
                self._slower_, self._residues_ = dict(layout.slower), dict(layout.residues)
                self._seed_ = self._draw_seed()
                self._fit_span_ = panel.index[[0, -1]]
                log = TrainingLog(callback)
                reason = self._estimate(panel, log)
                self._set_diagnostics(reason, log)
                self._set_conditioning(panel, None)
            self._warn_convergence(stacklevel=3)  # warn < _warn_convergence < fit < caller
        except BaseException:
            self._clear_fitted()
            raise
        return self

    def _set_data_attributes(self, panel: PanelData) -> None:
        """``n_features_in_``, ``feature_names_in_``, ``exog_names_in_`` and ``characteristic_names_in_``."""
        self.n_features_in_ = panel.n_series
        self.feature_names_in_ = np.asarray(panel.series_names, dtype=object)
        if self._accepts_exog:
            self.exog_names_in_ = tuple(panel.exog_names)
        if self._accepts_characteristics:
            self.characteristic_names_in_ = tuple(panel.characteristic_names)

    def _draw_seed(self) -> Optional[int]:
        """The fit-time seed of the simulations: drawn from ``random_state``, or OS entropy when it is None."""
        params = self.get_params(deep=False)
        if "random_state" not in params:
            return None
        state = params["random_state"]
        try:
            sequence = np.random.SeedSequence(state)
        except (TypeError, ValueError) as err:
            raise ConfigurationError(f"random_state must be None or a non-negative int, got {state!r}.") from err
        return int(sequence.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))

    def _set_diagnostics(self, reason: Any, log: TrainingLog) -> None:
        if reason not in STOP_REASONS:
            raise ValueError(f"{type(self).__name__}._estimate returned stop reason {reason!r}, not one of {STOP_REASONS}.")
        self.n_iter_ = log.n_iter
        self.history_ = log.frame()
        self.stop_reason_ = reason
        self.converged_ = reason in _CONVERGED

    def _warn_convergence(self, stacklevel: int) -> None:
        reason = vars(self).get("stop_reason_")
        if self._warns_on_nonconvergence and reason in _WARNED:
            why = "max_iter was reached" if reason == "max_iter" else "a numerical guard stopped it"
            warnings.warn(
                f"{type(self).__name__}: training stopped after {self.n_iter_} iteration(s) because {why} "
                f"(stop_reason_={reason!r}); converged_ is False.",
                ConvergenceWarning,
                stacklevel=stacklevel,
            )

    def _set_conditioning(self, panel: PanelData, previous: Optional[PanelData]) -> None:
        """Make ``panel`` the conditioning panel and run ``_condition``; on failure the model is left as it was."""
        saved = dict(vars(self))
        try:
            self._panel_ = panel
            self.index_ = panel.index
            self._condition(panel, previous)
        except BaseException:
            vars(self).clear()
            vars(self).update(saved)
            raise

    # update

    def update(self, data: Any, *, replace: bool = False) -> _FactorModel:
        """Condition on new data with the fitted parameters; return ``self``.

        ``data`` is merged into the conditioning panel (``index_``): every
        observed cell overwrites the conditioning value at its stamp, a NaN
        never erases one, and stamps after ``index_[-1]`` extend it. So new
        rows, revisions, late releases and complete vintages all go through
        this call. ``replace=True`` conditions on exactly ``data`` instead
        (withdrawn observations, or another sample). The parameters, the
        reporting transform and the diagnostics never change.

        Raises
        ------
        NotFittedError
            Before ``fit``.
        TypeError
            If ``replace`` is not a bool.
        DataValidationError
            For other series, exog or characteristic names, another index type,
            stamps off the model's grid, data that starts before ``index_[0]``
            without ``replace``, or a slower series off its anchors.
        """
        self._check_is_fitted("update")
        if not isinstance(replace, (bool, np.bool_)):
            raise TypeError(f"replace must be True or False, got {type(replace).__name__} {replace!r}.")
        what = self._what("update")
        new = PanelData._coerce(data)
        self._check_slots(new, what)
        merged = self._panel_._merge(new, replace=bool(replace), new_series=self._new_series_on_update)
        self._check_layout(merged, what)
        with self._verbosity():
            self._set_conditioning(merged, self._panel_)
        return self

    # save / load (format 2)

    def save(self, path: Any, *, overwrite: bool = False) -> None:
        """Save the fitted model as a format-2 directory (spec section 10).

        The run is written to a temporary sibling directory, reloaded and
        checked (``predict(1)`` or ``allocate()``: bitwise for numpy models,
        ``allclose`` for torch), and then renamed to ``path``.

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
        TypeError
            For a hyperparameter value a saved run cannot hold.
        """
        self._check_is_fitted("save")
        cls = type(self)
        # Everything is encoded up front, so a value JSON cannot hold fails before anything is written.
        config = to_jsonable(self._saved_config(), where=CONFIG_FILE)
        data_arrays, data_values = _panel_to_files(self._panel_)
        model_arrays, model_values = self._save_fitted()
        fitted_arrays = self._base_arrays()
        fitted_arrays.update({_MODEL_PREFIX + key: value for key, value in model_arrays.items()})
        fitted_values = to_jsonable({"base": self._base_values(), "model": dict(model_values)}, where=FITTED_JSON)
        weights = self._fitted_weights() if self._requires_torch else None
        if self._requires_torch and weights is None:
            raise NotImplementedError(f"{cls.__name__}._fitted_weights returned None; a torch model saves its weights.")
        expected = self._reload_output()

        def write(tmp: Path) -> None:
            write_json(tmp / CONFIG_FILE, config)
            write_npz(tmp / DATA_NPZ, data_arrays)
            write_json(tmp / DATA_JSON, data_values)
            write_npz(tmp / FITTED_NPZ, fitted_arrays)
            write_json(tmp / FITTED_JSON, fitted_values)
            if weights is not None:
                write_weights(tmp / WEIGHTS_FILE, weights)

        def verify(tmp: Path) -> None:
            loaded = cls._load(tmp, device="cpu")
            _check_reloaded(expected, loaded._reload_output(), exact=not self._requires_torch, name=cls.__name__)

        atomic_save_dir(path, write, verify, overwrite=overwrite)

    def _saved_config(self) -> dict:
        from . import __version__

        runtime: dict = {"numpy_version": str(np.__version__)}
        device = vars(self).get("_device_")
        if self._requires_torch:
            import torch

            runtime.update(device=None if device is None else str(device), torch_version=str(torch.__version__))
        return {
            "format_version": FORMAT_2,
            "model_type": type(self).__name__,
            "library_version": str(__version__),
            "params": encode_params(self.get_params(deep=False)),
            "runtime": runtime,
        }

    def _base_arrays(self) -> dict:
        arrays = {}
        transform = vars(self).get("_transform_")
        if transform is not None:
            arrays["transform"] = np.asarray(transform, dtype=np.float64)
        return arrays

    def _base_values(self) -> dict:
        history = self.history_
        values = {
            "n_features_in_": int(self.n_features_in_),
            "feature_names_in_": list(self.feature_names_in_),
            "n_iter_": int(self.n_iter_),
            "converged_": bool(self.converged_),
            "stop_reason_": self.stop_reason_,
            "history_columns": list(history.columns),
            "history": [history[col].tolist() for col in history.columns],
            "seed": self._seed_,
            "fit_span": index_to_spec(self._fit_span_),
            "grid": offset_to_spec(self._grid_),
            "clock": self._clock_,
            "slower": [[name, code] for name, code in self._slower_.items()],
            "residues": [[name, residue] for name, residue in self._residues_.items()],
        }
        for name in ("exog_names_in_", "characteristic_names_in_"):
            if name in vars(self):
                values[name] = list(vars(self)[name])
        if "loglik_" in vars(self):
            values["loglik_"] = float(self.loglik_)
        return values

    def _restore_base(self, values: Mapping, arrays: Mapping) -> None:
        self.n_features_in_ = int(values["n_features_in_"])
        self.feature_names_in_ = np.asarray(values["feature_names_in_"], dtype=object)
        for name in ("exog_names_in_", "characteristic_names_in_"):
            if name in values:
                setattr(self, name, tuple(values[name]))
        self.n_iter_ = int(values["n_iter_"])
        self.converged_ = bool(values["converged_"])
        self.stop_reason_ = values["stop_reason_"]
        columns = list(values["history_columns"])
        self.history_ = pd.DataFrame(
            dict(zip(columns, values["history"])), index=pd.RangeIndex(1, self.n_iter_ + 1, name="iteration"),
            columns=columns,
        )
        for col in ("objective", "elapsed"):
            if col in self.history_:
                self.history_[col] = self.history_[col].astype(np.float64)
        if "loglik_" in values:
            self.loglik_ = float(values["loglik_"])
        self._seed_ = values["seed"]
        self._fit_span_ = index_from_spec(values["fit_span"])
        self._grid_ = offset_from_spec(values["grid"])
        self._clock_ = values["clock"]
        self._slower_ = {name: code for name, code in values["slower"]}
        self._residues_ = {name: int(residue) for name, residue in values["residues"]}
        if "transform" in arrays:
            self._transform_ = np.asarray(arrays["transform"], dtype=np.float64)

    @classmethod
    def _load(cls, path: Any, *, device: Any = None) -> _FactorModel:
        """Load a format-2 run written by :meth:`save` and condition on its saved panel.

        Raises
        ------
        ImportError
            For a torch model when torch (the ``deep`` extra) is not installed.
        ConfigurationError
            For a format-1 run ("saved by dfm-python 0.8 (format 1)"), a run of
            another model, a directory that lacks a file, or parameters that do
            not match the class.
        """
        path = Path(path)
        name = cls.__name__
        if cls._requires_torch:
            require_torch(f"{name}.load")
        config = read_config_2(path)
        if config["model_type"] != name:
            raise ConfigurationError(
                f"{str(path)!r} holds a {config['model_type']!r} run, not {name}. Load it with dfm_python.load(path)."
            )
        missing = [file for file in REQUIRED_FILES_2 if not (path / file).is_file()]
        if cls._requires_torch and not (path / WEIGHTS_FILE).is_file():
            missing.append(WEIGHTS_FILE)
        if missing:
            raise ConfigurationError(f"{str(path)!r} is not a complete saved run: {', '.join(missing)} missing.")
        params = decode_params(dict(config["params"]), cls)
        try:
            est = cls(**params)
        except TypeError as err:
            raise ConfigurationError(f"config.json in {str(path)!r} does not match {name}: {err}") from err
        panel = _panel_from_files(read_npz(path / DATA_NPZ), read_json_plain(path / DATA_JSON))
        arrays = read_npz(path / FITTED_NPZ)
        values = read_json_plain(path / FITTED_JSON)
        est._restore_base(values["base"], arrays)
        resolved = est._torch_device(device) if cls._requires_torch else None
        weights = read_weights(path / WEIGHTS_FILE, map_location=resolved) if cls._requires_torch else None
        model_arrays = {k[len(_MODEL_PREFIX):]: v for k, v in arrays.items() if k.startswith(_MODEL_PREFIX)}
        with est._verbosity():
            est._load_fitted(model_arrays, values["model"], weights=weights, device=resolved)
            est._set_conditioning(panel, None)
        return est

    def _torch_device(self, device: Any = None) -> Any:
        """The torch device: ``device``, else the ``device`` parameter; ``"auto"`` means CUDA if available.

        A ``device`` parameter naming CUDA falls back to CPU with a
        ``UserWarning`` when CUDA is not available (a GPU-saved run loads on a
        CPU-only machine); an explicit ``device`` naming CUDA there raises
        ``ConfigurationError``.
        """
        torch = require_torch(f"{type(self).__name__}")
        explicit = device is not None
        if device is None:
            device = self.get_params(deep=False).get("device", "auto")
        if device is None or (isinstance(device, str) and device == "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = type(self).__name__
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError, ValueError) as err:
            source = "device" if explicit else "the device parameter"
            raise ConfigurationError(f"{name}: {source} {device!r} is not a valid torch device ({err}).") from err
        if resolved.type == "cuda" and not torch.cuda.is_available():
            if explicit:
                raise ConfigurationError(
                    f"{name}: device={device!r} but CUDA is not available on this machine. Pass device='cpu' (or 'auto')."
                )
            warnings.warn(
                f"{name} has device={str(device)!r}, but CUDA is not available on this machine; using the CPU. "
                "Pass device='cpu' to choose the device explicitly.",
                UserWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return resolved


def _check_reloaded(expected: pd.DataFrame, actual: pd.DataFrame, *, exact: bool, name: str) -> None:
    """Raise when the reloaded model's output differs (bitwise for numpy, ``allclose`` for torch)."""
    same_axes = expected.index.equals(actual.index) and expected.columns.equals(actual.columns)
    a, b = expected.to_numpy(dtype=np.float64), actual.to_numpy(dtype=np.float64)
    if exact:
        same = same_axes and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    else:
        same = same_axes and a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True)
    if not same:
        raise RuntimeError(f"{name}: the reloaded run does not reproduce the model's output; the save was not completed.")


def _panel_to_files(panel: PanelData) -> tuple[dict, dict]:
    """``data.npz`` and ``data.json``: y and exog (float64), characteristics (float32), the index spec, names and freq."""
    arrays = {"y": panel._y.to_numpy(dtype=np.float64)}
    if panel._exog is not None:
        arrays["exog"] = panel._exog.to_numpy(dtype=np.float64)
    if panel._chars is not None:
        arrays["characteristics"] = panel._chars.astype(np.float32)
    values = to_jsonable(
        {
            "index": index_to_spec(panel.index),
            "series_names": list(panel.series_names),
            "exog_names": list(panel.exog_names),
            "characteristic_names": list(panel.characteristic_names),
            "freq": None if panel.freq is None else [[name, code] for name, code in panel.freq.items()],
        },
        where=DATA_JSON,
    )
    return arrays, values


def _panel_from_files(arrays: Mapping, values: Mapping) -> PanelData:
    """The conditioning panel of a saved run (inverse of :func:`_panel_to_files`)."""
    exog = arrays.get("exog")
    chars = arrays.get("characteristics")
    freq = None if values["freq"] is None else {name: code for name, code in values["freq"]}
    return PanelData._from_arrays_checked(
        index_from_spec(values["index"]),
        np.array(arrays["y"], dtype=np.float64),
        values["series_names"],
        None if exog is None else np.array(exog, dtype=np.float64),
        values["exog_names"],
        None if chars is None else np.array(chars, dtype=np.float64),
        values["characteristic_names"],
        freq,
    )


# ---------------------------------------------------------------------------
# _Forecaster
# ---------------------------------------------------------------------------


def _check_coverage(coverage: Any) -> float:
    valid = isinstance(coverage, (int, float, np.integer, np.floating)) and not isinstance(coverage, (bool, np.bool_))
    if not (valid and math.isfinite(coverage) and 0 < coverage < 1):
        raise ConfigurationError(f"coverage must be a number in (0, 1), got {coverage!r}.")
    return float(coverage)


class _Forecaster(_FactorModel):
    """Base of the forecasters (DFM, DDFM, iVDFM): the predict family and ``nowcast`` (spec sections 4.4 to 4.8).

    ``predict*`` evaluate E[. given the conditioning data] at the target
    stamps ``start..end`` of :meth:`dfm_python._time.Grid.resolve`: past,
    present or future. At an observed in-sample cell the data is returned
    (``lower == upper ==`` the data); a slower series is reported only at the
    stamps that end one of its periods.
    """

    #: The forecast uses future exog when the model was fitted with exog (iVDFM).
    _forecast_uses_exog: ClassVar[bool] = False

    __metadata_request__predict = {"end": UNUSED, "start": UNUSED, "exog": UNUSED}

    # Hooks

    def _moments(
        self, stamps: pd.Index, exog: Optional[np.ndarray], coverage: Optional[float]
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """``(mean, lower, upper)`` at the target ``stamps`` in data units, shape (n, N); bounds None without coverage.

        ``exog`` holds the future exog of the stamps after the origin up to the
        last target, (n_future, K), when the forecast needs it (else None).
        :meth:`_target_positions` gives the grid positions of ``stamps``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _moments().")

    def _factor_path(self, stamps: pd.Index, exog: Optional[np.ndarray]) -> np.ndarray:
        """The reported factors at the target ``stamps``, shape (n, r)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _factor_path().")

    def _reload_output(self) -> pd.DataFrame:
        if self._needs_future_exog(self._grid_now().n):
            return self.predict_factors(0)
        return self.predict(1)

    def _check_training_data(self, panel: PanelData, what: str) -> None:
        """Reject target series a forecaster cannot learn from.

        Each series is judged on its observed (finite) entries. A series is
        constant when the standard deviation of its ``n`` observed values is at
        most ``n * eps * |mean|`` (``eps`` the float64 machine epsilon), the
        rounding error of computing it.

        Raises
        ------
        DataValidationError
            Naming each series with no observed value, a single one, or the
            same value (up to rounding) at every observation.
        """
        eps = np.finfo(np.float64).eps
        problems = []
        for series, column in zip(panel.series_names, panel._y.to_numpy().T):
            observed = column[np.isfinite(column)]
            if observed.size == 0:
                problems.append(f"{series!r} has no observed value")
            elif observed.size == 1:
                problems.append(f"{series!r} has a single observed value")
            elif observed.std() <= observed.size * eps * abs(observed.mean()):
                problems.append(f"{series!r} is constant (every observed value is {float(observed[0])!r}, up to rounding)")
        if problems:
            raise DataValidationError(
                f"{what}: {type(self).__name__} cannot be fitted on these target series: {'; '.join(problems)}. A "
                "target series needs at least 2 observed values that are not all equal: drop these series from y, "
                "or supply more observations."
            )

    # Targets

    def _grid_now(self) -> _time.Grid:
        return _time.Grid(self.index_, self._grid_, self._clock_)

    def _target_positions(self, stamps: pd.Index) -> np.ndarray:
        """Grid positions of target stamps (0 is ``index_[0]``; ``len(index_)`` the first stamp after the origin)."""
        index = self.index_
        if isinstance(index, pd.RangeIndex):
            return (np.asarray(stamps, dtype=np.int64) - index[0]) // index.step
        if isinstance(index, pd.PeriodIndex):
            return np.asarray(stamps.asi8, dtype=np.int64) - index[0].ordinal
        grid = self._grid_now()
        first = grid._last_at_or_before(stamps[0])
        return first + np.arange(len(stamps), dtype=np.int64)

    def _targets(self, end: Any, start: Any) -> tuple[int, int, pd.Index]:
        grid = self._grid_now()
        first, last = grid.resolve(end, start)
        return first, last, grid.stamps(first, last)

    def _needs_future_exog(self, last: int) -> bool:
        return bool(self._forecast_uses_exog and vars(self).get("exog_names_in_") and last >= len(self.index_))

    def _future_exog(self, last: int, exog: Any, method: str) -> Optional[np.ndarray]:
        """The future exog the forecast needs (spec section 4.6), validated, or None.

        Needed exactly when the model's forecast uses future context (fitted
        with exog) and a target lies after the origin; it then covers every
        stamp from the one after the origin to the last target, in
        ``exog_names_in_`` order.
        """
        name = type(self).__name__
        n = len(self.index_)
        if not self._needs_future_exog(last):
            if exog is not None:
                if not (self._forecast_uses_exog and vars(self).get("exog_names_in_")):
                    reason = f"{name}'s forecasts do not use future exog"
                    if self._forecast_uses_exog:
                        reason = f"this {name} was fitted without exog"
                else:
                    reason = "every target lies in the conditioning data, whose exog are used"
                raise ConfigurationError(f"{name}.{method} takes no exog here: {reason}. Pass exog=None.")
            return None
        stamps = self._grid_now().stamps(n, last)
        names = list(self.exog_names_in_)
        span = f"{_time._fmt_stamp(stamps[0])}..{_time._fmt_stamp(stamps[-1])}"
        if exog is None:
            raise DataValidationError(
                f"{name}.{method}: exog needed for {span}: the model was fitted with exog {names}. Pass exog=DataFrame "
                "indexed by those stamps, with those columns."
            )
        if not isinstance(exog, pd.DataFrame):
            raise TypeError(f"{name}.{method}: exog must be a pandas DataFrame, got {type(exog).__name__}.")
        missing_cols = [c for c in names if c not in exog.columns]
        if missing_cols:
            raise DataValidationError(f"{name}.{method}: exog lacks the columns {missing_cols} (needed: {names}).")
        index = exog.index
        if isinstance(stamps, pd.DatetimeIndex) and isinstance(index, pd.DatetimeIndex):
            rows = index.as_unit("ns").get_indexer(stamps.as_unit("ns"))
        else:
            rows = index.get_indexer(stamps) if type(index) is type(stamps) else np.full(len(stamps), -1)
        if (rows < 0).any():
            absent = stamps[rows < 0]
            raise DataValidationError(
                f"{name}.{method}: exog lacks the stamps {_time._fmt_stamps(absent)} (needed: every stamp of {span})."
            )
        values = exog[names].to_numpy(dtype=np.float64)[rows]
        if not np.isfinite(values).all():
            raise DataValidationError(f"{name}.{method}: exog has missing or infinite values at the stamps of {span}.")
        return values

    # Output assembly

    def _slower_mask(self, first: int, last: int) -> np.ndarray:
        """``(n, N)`` True where a series is reported: every clock series, slower series at their period ends."""
        mask = np.ones((last - first + 1, self.n_features_in_), dtype=bool)
        if self._slower_:
            grid = self._grid_now()
            names = list(self.feature_names_in_)
            for series, code in self._slower_.items():
                mask[:, names.index(series)] = grid.anchor_mask(first, last, code, self._residues_.get(series))
        return mask

    def _observed(self, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
        """``(values, mask)`` of the conditioning data at the in-sample rows of ``first..last`` (padded with NaN)."""
        n = len(self.index_)
        out = np.full((last - first + 1, self.n_features_in_), np.nan)
        if first < n:
            out[: min(last, n - 1) - first + 1] = self._panel_._y.to_numpy()[first:min(last, n - 1) + 1]
        return out, ~np.isnan(out)

    def _finish(self, values: np.ndarray, first: int, last: int) -> np.ndarray:
        """Data at observed in-sample cells; slower series NaN off their period ends."""
        out = np.array(values, dtype=np.float64, copy=True)
        expected = (last - first + 1, self.n_features_in_)
        if out.shape != expected:
            raise ValueError(f"{type(self).__name__} returned moments of shape {out.shape}, expected {expected}.")
        data, observed = self._observed(first, last)
        out[observed] = data[observed]
        out[~self._slower_mask(first, last)] = np.nan
        return out

    def _series_columns(self) -> pd.Index:
        return pd.Index(list(self.feature_names_in_), name="series")

    # The predict family

    def predict(self, end: Any = 1, *, start: Any = None, exog: Any = None) -> pd.DataFrame:
        """E[y given the conditioning data] at the target stamps ``start..end``, in data units.

        Parameters
        ----------
        end : int, str, pandas.Period, pandas.Timestamp or datetime, default 1
            An int is an offset in grid steps from the origin ``index_[-1]``
            (0 is the origin, 1 the next stamp); a stamp is that grid stamp; a
            str or Period names a period, whose last grid stamp is the end.
        start : same types, optional
            The first target (a period's first grid stamp). None means the step
            after the origin when ``end`` lies after it, otherwise ``end``.
        exog : pandas.DataFrame, optional
            Future exog, only for a model whose forecast uses them (fitted with
            exog) when a target lies after the origin.

        Returns
        -------
        pandas.DataFrame
            Targets x series (axis ``"series"``), float64. Observed in-sample
            cells hold the data; a slower series is NaN off its period ends.
        """
        self._check_is_fitted("predict")
        first, last, stamps = self._targets(end, start)
        future = self._future_exog(last, exog, "predict")
        with self._verbosity():
            mean, _, _ = self._moments(stamps, future, None)
        return pd.DataFrame(self._finish(mean, first, last), index=stamps, columns=self._series_columns())

    def predict_interval(
        self, end: Any = 1, *, start: Any = None, coverage: float = 0.9, exog: Any = None
    ) -> pd.DataFrame:
        """Central intervals at ``coverage`` for the targets of :meth:`predict`.

        Returns a DataFrame with columns ``MultiIndex(series, bound)``, bound in
        ``("lower", "upper")``. At observed in-sample cells ``lower == upper``
        == the data.

        Raises
        ------
        ConfigurationError
            For a ``coverage`` that is not a number in (0, 1).
        """
        self._check_is_fitted("predict_interval")
        coverage = _check_coverage(coverage)
        first, last, stamps = self._targets(end, start)
        future = self._future_exog(last, exog, "predict_interval")
        with self._verbosity():
            _, lower, upper = self._moments(stamps, future, coverage)
        if lower is None or upper is None:
            raise ValueError(f"{type(self).__name__}._moments returned no bounds for coverage={coverage}.")
        lower, upper = self._finish(lower, first, last), self._finish(upper, first, last)
        values = np.stack([lower, upper], axis=2).reshape(len(stamps), -1)
        columns = pd.MultiIndex.from_product(
            [list(self.feature_names_in_), ["lower", "upper"]], names=["series", "bound"]
        )
        return pd.DataFrame(values, index=stamps, columns=columns)

    def predict_factors(self, end: Any = 1, *, start: Any = None, exog: Any = None) -> pd.DataFrame:
        """The reported factors at the targets of :meth:`predict` (columns: the factor names, axis ``"factor"``)."""
        self._check_is_fitted("predict_factors")
        first, last, stamps = self._targets(end, start)
        future = self._future_exog(last, exog, "predict_factors")
        with self._verbosity():
            path = np.asarray(self._factor_path(stamps, future), dtype=np.float64)
        columns = pd.Index(list(self.factors_.columns), name="factor")
        if path.shape != (len(stamps), len(columns)):
            raise ValueError(f"{type(self).__name__}._factor_path returned shape {path.shape}.")
        return pd.DataFrame(path, index=stamps, columns=columns)

    def nowcast(
        self, series: Any = None, *, end: Any = 1, start: Any = None, coverage: float = 0.9, exog: Any = None
    ) -> pd.DataFrame:
        """The per-period table of each series at its own frequency (spec section 4.8).

        A view over :meth:`predict` and :meth:`predict_interval` at each
        period's observation stamp (its last clock stamp).

        Parameters
        ----------
        series : name or list of names, optional
            None means every series slower than the clock, or every series when
            there is none.
        end : int, str or pandas.Period, default 1
            An int ``k`` is ``k`` periods after the period holding the origin;
            a label is that period, parsed at the series' frequency.
        start : int, str or pandas.Period, optional
            None means the latest period released for that series (the first
            conditioning period when none was).
        coverage : float, default 0.9
        exog : pandas.DataFrame, optional
            As in :meth:`predict`.

        Returns
        -------
        pandas.DataFrame
            Rows ``MultiIndex(series, period)``; columns ``estimate``,
            ``lower``, ``upper``, ``actual`` and ``status`` (``"released"``,
            ``"backcast"``, ``"nowcast"`` or ``"forecast"``). Labels are
            ``pd.Period`` at the series' frequency (on a ``RangeIndex``, the
            period-end stamp).

        Raises
        ------
        ConfigurationError
            For an unknown series, an invalid ``coverage``, or periods outside
            the conditioning data.
        """
        self._check_is_fitted("nowcast")
        coverage = _check_coverage(coverage)
        names = self._nowcast_series(series)
        grid = self._grid_now()
        n = grid.n
        y = self._panel_._y.to_numpy()
        features = list(self.feature_names_in_)
        plan = []
        for name in names:
            j = features.index(name)
            periods = grid.periods(
                self._slower_.get(name), end, start, residue=self._residues_.get(name),
                observed=np.flatnonzero(~np.isnan(y[:, j])),
            )
            plan.append((name, j, periods))
        ends = [p.last for _, _, periods in plan for p in periods]
        if not ends:
            raise ConfigurationError(f"nowcast(end={end!r}, start={start!r}) selects no periods.")
        lo, hi = min(ends), max(ends)
        point = self.predict(hi - (n - 1), start=lo - (n - 1), exog=exog).to_numpy()
        bands = self.predict_interval(hi - (n - 1), start=lo - (n - 1), coverage=coverage, exog=exog)
        lower = bands.xs("lower", axis=1, level="bound").to_numpy()
        upper = bands.xs("upper", axis=1, level="bound").to_numpy()
        keys, rows = [], []
        for name, j, periods in plan:
            for period in periods:
                at = period.last - lo
                actual = y[period.last, j] if period.last < n else np.nan
                status = period.status(n - 1, released=not np.isnan(actual))
                keys.append((name, period.label))
                rows.append((point[at, j], lower[at, j], upper[at, j], actual, status))
        index = pd.MultiIndex.from_tuples(keys, names=["series", "period"])
        frame = pd.DataFrame(rows, index=index, columns=["estimate", "lower", "upper", "actual", "status"])
        for col in ("estimate", "lower", "upper", "actual"):
            frame[col] = frame[col].astype(np.float64)
        return frame

    def _nowcast_series(self, series: Any) -> list:
        features = list(self.feature_names_in_)
        if series is None:
            slower = [name for name in features if name in self._slower_]
            return slower or features
        names = list(series) if isinstance(series, (list, tuple, pd.Index, np.ndarray)) else [series]
        try:
            unknown = [s for s in names if s not in set(features)]
        except TypeError as err:
            raise TypeError(f"series must be a name, a list of names or None, got {type(series).__name__}.") from err
        if unknown:
            raise ConfigurationError(f"nowcast: unknown series {unknown}; the series are {features}.")
        return [name for name in features if name in set(names)]


# ---------------------------------------------------------------------------
# _Allocator
# ---------------------------------------------------------------------------


class _Allocator(_FactorModel):
    """Base of the allocators (AFM): rows are decision dates, any increasing index; ``allocate``."""

    _regular_grid: ClassVar[bool] = False

    def _next_weights(self) -> np.ndarray:
        """The weights decided at ``index_[-1]`` for the next period, over every known asset (``_panel_`` order)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement _next_weights().")

    def _reload_output(self) -> pd.DataFrame:
        return self.allocate()

    def allocate(self) -> pd.DataFrame:
        """The next-period weights: one row, indexed by the stamp after ``index_[-1]``, one column per known asset.

        Raises
        ------
        NotFittedError
            Before ``fit``.
        """
        self._check_is_fitted("allocate")
        with self._verbosity():
            weights = np.asarray(self._next_weights(), dtype=np.float64).reshape(1, -1)
        names = list(self._panel_.series_names)
        if weights.shape[1] != len(names):
            raise ValueError(f"{type(self).__name__}._next_weights returned {weights.shape[1]} weights for {len(names)} assets.")
        name = self.index_.name if self.index_.name is not None else "time"
        index = pd.Index([_time.forward_stamp(self.index_)], name=name)
        if isinstance(index, pd.DatetimeIndex):
            index = index.as_unit(self.index_.unit)
        return pd.DataFrame(weights, index=index, columns=pd.Index(names, name="series"))

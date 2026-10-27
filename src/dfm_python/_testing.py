"""Contract checks for the models on :mod:`dfm_python._estimator` (spec sections 1, 4, 5 and 10).

:func:`check_estimator` runs the checks that apply to a model class (a
forecaster or an allocator) and raises ``AssertionError`` naming the first
check that fails. :func:`make_panel` builds the tiny CPU fixture the checks
use, from the slots the class accepts.

==================  ==========================================================
Check               What it asserts
==================  ==========================================================
``parameters``      keyword-only ``__init__`` that stores its arguments
                    unchanged; ``get_params``, ``set_params`` and ``clone``
                    round-trip; an unknown name raises (``set_params``:
                    ``ConfigurationError`` naming the closest name); no
                    ``set_*_request`` methods
``construction``    the constructor leaves the global numpy and torch RNG
                    states untouched
``unfitted``        fitted attributes and verbs raise ``NotFittedError``
                    (also sklearn's), ``hasattr(model, "params_")`` is False
``fit``             ``fit`` returns ``self`` and sets the common fitted
                    attributes (``params_``/``states_`` namespaces,
                    ``factors_ is states_.factors``, diagnostics with
                    ``n_iter_ == len(history_)``); two fits agree
``fit_attributes``  ``fit`` adds only fitted attributes, mutates neither the
                    data nor the parameters; a failing fit leaves the model
                    unfitted
``callback``        the callback gets ``{"iteration", **history row}`` and a
                    true return stops training as ``"callback"``, silently
``namespaces``      every frame field has named axes; numeric values are
                    float64
``outputs``         forecasters: the predict family's index, columns, dtypes
                    and in-sample data at observed cells, ``nowcast`` as a
                    view over ``predict``, argument errors; allocators: one
                    row of weights at the forward stamp
``purity``          the verbs change no fitted state and repeat exactly
``update``          ``update`` returns ``self``; parameters and diagnostics are
                    unchanged; the merged index; idempotence; full vintage
                    equals new rows only; ``replace=True``
``silence``         ``verbose=0`` prints nothing and logs nothing at INFO
``save_load``       format-2 file set, ``config.json``, refusing an existing
                    directory, ``_load`` and ``dfm_python.load`` reproducing
                    the outputs (bitwise for numpy, ``allclose`` for torch),
                    format-1 rejection, ``update`` after ``load``
==================  ==========================================================
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError as _SklearnNotFittedError
from sklearn.utils.validation import check_is_fitted

from ._estimator import STOP_REASONS, _Allocator, _FactorModel, _Forecaster
from ._namespace import Namespace
from .dataset.panel import PanelData
from .errors import ConfigurationError, NotFittedError
from .io import loading as _loading
from .io.layout import CONFIG_FILE, FORMAT_2, REQUIRED_FILES_2, WEIGHTS_FILE

__all__ = ["check_estimator", "make_panel"]

_BOGUS_KEY = "definitely_not_a_parameter"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def make_panel(cls: type, *, seed: int = 0) -> PanelData:
    """The tiny CPU fixture for a model class, from the slots it accepts.

    Forecasters: 48 month-end rows of 3 series (``s0..s2``) driven by one AR(1)
    factor, two missing values in the last series, a quarterly ``s0`` (values
    at quarter ends, ``freq={"s0": "q"}``) when the class accepts mixed
    frequency, and 2 exog columns when it accepts exog. Allocators: 80
    business days of 6 assets with returns of about 1%, characteristics (2)
    and exog (1) when accepted.
    """
    rng = np.random.default_rng(seed)
    if issubclass(cls, _Allocator):
        n_time, n_series = 80, 6
        index = pd.bdate_range("2020-01-01", periods=n_time, name="date")
        y = 0.01 * rng.standard_normal((n_time, n_series))
        chars = rng.standard_normal((n_time, n_series, 2)) if cls._accepts_characteristics else None
        exog = rng.standard_normal((n_time, 1)) if cls._accepts_exog else None
        return PanelData.from_arrays(
            y, exog=exog, characteristics=chars, index=index, series_names=[f"a{i}" for i in range(n_series)],
            exog_names=["x0"] if exog is not None else None,
            characteristic_names=["c0", "c1"] if chars is not None else None,
        )
    n_time, n_series = 48, 3
    index = pd.date_range("2000-01-31", periods=n_time, freq="ME", name="date")
    factor = np.zeros(n_time)
    shocks = rng.standard_normal(n_time)
    for t in range(1, n_time):
        factor[t] = 0.7 * factor[t - 1] + shocks[t]
    y = np.outer(factor, rng.standard_normal(n_series)) + 0.5 * rng.standard_normal((n_time, n_series))
    y[[3, 17], n_series - 1] = np.nan
    freq = None
    if cls._accepts_mixed_frequency:
        y[index.month % 3 != 0, 0] = np.nan
        freq = {"s0": "q"}
    exog = rng.standard_normal((n_time, 2)) if cls._accepts_exog else None
    return PanelData.from_arrays(
        y, exog=exog, index=index, series_names=[f"s{i}" for i in range(n_series)],
        exog_names=["x0", "x1"] if exog is not None else None, freq=freq,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feed(h: Any, obj: Any, seen: set, depth: int = 0) -> None:
    """Feed a stable byte description of ``obj`` (values, not identities) to the hash ``h``."""
    if depth > 12:
        h.update(b"<depth>")
        return
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
        return
    if isinstance(obj, np.generic):
        h.update(f"{obj.dtype}:{obj!r};".encode())
        return
    if isinstance(obj, np.ndarray):
        h.update(f"nd:{obj.dtype}:{obj.shape};".encode())
        if obj.dtype.hasobject:
            for item in obj.ravel().tolist():
                _feed(h, item, seen, depth + 1)
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
        return
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        h.update(f"pt:{t.dtype}:{tuple(t.shape)};".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(obj, type) or inspect.isroutine(obj) or inspect.ismodule(obj):
        h.update(f"<{getattr(obj, '__qualname__', getattr(obj, '__name__', type(obj).__name__))}>;".encode())
        return
    if id(obj) in seen:
        h.update(b"<seen>")
        return
    seen.add(id(obj))
    if torch is not None and isinstance(obj, torch.nn.Module):
        h.update(f"module:{type(obj).__qualname__}:{obj.training};".encode())
        for key, value in obj.state_dict().items():
            h.update(key.encode())
            _feed(h, value, seen, depth + 1)
        return
    if isinstance(obj, pd.Index):
        h.update(f"index:{type(obj).__name__}:{obj.dtype}:{obj.name!r}:{obj.tolist()!r};".encode())
        return
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        _feed(h, obj.index, seen, depth + 1)
        if isinstance(obj, pd.DataFrame):
            _feed(h, obj.columns, seen, depth + 1)
            h.update(repr(list(obj.dtypes)).encode())
        _feed(h, obj.to_numpy(), seen, depth + 1)
        return
    if isinstance(obj, Namespace):
        _feed(h, obj._data, seen, depth + 1)
        return
    if dataclasses.is_dataclass(obj):
        h.update(f"dc:{type(obj).__qualname__};".encode())
        for f in dataclasses.fields(obj):
            h.update(f.name.encode())
            _feed(h, getattr(obj, f.name), seen, depth + 1)
        return
    if isinstance(obj, Mapping):
        h.update(f"map:{type(obj).__qualname__};".encode())
        for key in sorted(obj, key=repr):
            h.update(repr(key).encode())
            _feed(h, obj[key], seen, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        h.update(f"seq:{type(obj).__name__}:{len(obj)};".encode())
        for item in obj:
            _feed(h, item, seen, depth + 1)
        return
    if isinstance(obj, (set, frozenset)):
        h.update(repr(sorted(map(repr, obj))).encode())
        return
    try:
        state = vars(obj)
    except TypeError:
        slots = [s for klass in type(obj).__mro__ for s in getattr(klass, "__slots__", ())]
        if not slots:
            h.update(f"{type(obj).__qualname__}:{obj!r};".encode())
            return
        state = {s: getattr(obj, s, None) for s in slots if s not in ("__dict__", "__weakref__")}
    h.update(f"obj:{type(obj).__qualname__};".encode())
    for key in sorted(state):
        h.update(key.encode())
        _feed(h, state[key], seen, depth + 1)


def _state_hash(est: _FactorModel) -> str:
    """Hash of every instance attribute: parameters, fitted attributes and private state."""
    h = hashlib.sha256()
    _feed(h, dict(vars(est)), set())
    return h.hexdigest()


def _python_scalar(value: Any) -> Any:
    """A numpy scalar as its Python value (``np.str_`` as ``str``); datetimes stay (``.item()`` may give an int)."""
    if isinstance(value, np.generic) and not isinstance(value, (np.datetime64, np.timedelta64)):
        return value.item()
    return value


def _deep_equal(a: Any, b: Any) -> bool:
    """Structural equality: arrays by value (NaN equals NaN), estimators by params, tuple != list.

    A numpy scalar equals its Python value (a JSON round trip turns ``np.str_``
    series names into ``str``).
    """
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.hasobject:
            return all(_deep_equal(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
        return type(a) is type(b) and a.equals(b)
    a, b = _python_scalar(a), _python_scalar(b)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if hasattr(a, "get_params") and hasattr(b, "get_params") and not isinstance(a, type):
        return type(a) is type(b) and _deep_equal(a.get_params(deep=False), b.get_params(deep=False))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return list(a.keys()) == list(b.keys()) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if type(a) is not type(b) and not (isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number))):
        return False
    try:
        result = a == b
    except (TypeError, ValueError):  # values that cannot be compared are not equal
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _diff_keys(a: Mapping, b: Mapping) -> list:
    keys = list(dict.fromkeys([*a.keys(), *b.keys()]))
    return [k for k in keys if k not in a or k not in b or not _deep_equal(a[k], b[k])]


def _expect_raises(exc_type: type, func: Callable[[], Any], what: str) -> BaseException:
    try:
        func()
    except exc_type as err:
        return err
    except Exception as err:  # harness boundary: the wrong exception is the check's failure (chained)
        raise AssertionError(f"{what} raised {type(err).__name__} ({err}), expected {exc_type.__name__}.") from err
    raise AssertionError(f"{what} did not raise {exc_type.__name__}.")


def _same_frame(a: pd.DataFrame, b: Any, what: str, *, exact: bool = True) -> None:
    assert isinstance(b, pd.DataFrame), f"{what} returned {type(b).__name__}, expected a DataFrame."
    assert type(a.index) is type(b.index) and a.index.equals(b.index) and a.index.names == b.index.names, (
        f"{what}: the index differs."
    )
    assert a.columns.equals(b.columns) and a.columns.names == b.columns.names, f"{what}: the columns differ."
    assert list(a.dtypes) == list(b.dtypes), f"{what}: the dtypes differ."
    x, y = a.to_numpy(), b.to_numpy()
    if not exact and x.dtype.kind == "f":
        assert np.allclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=True), f"{what}: values differ beyond allclose."
    elif x.dtype.kind == "f":
        assert np.array_equal(x, y, equal_nan=True), f"{what}: values are not bitwise identical."
    else:
        assert all(_deep_equal(u, v) for u, v in zip(x.ravel().tolist(), y.ravel().tolist())), f"{what}: values differ."


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _split(panel: PanelData) -> tuple[PanelData, PanelData]:
    n = panel.n_periods
    k = max(1, min(6, n // 6))
    if n - k < 2:
        raise ValueError(f"check_estimator needs at least {k + 2} periods, got {n}.")
    return panel._take(slice(0, n - k)), panel._take(slice(n - k, n))


# ---------------------------------------------------------------------------
# Check context
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Context:
    cls: type
    params: dict
    data: PanelData
    train: PanelData
    new: PanelData
    label: str
    fitted: Optional[_FactorModel] = None

    @property
    def forecaster(self) -> bool:
        return issubclass(self.cls, _Forecaster)

    @property
    def exact(self) -> bool:
        """Outputs compare bitwise (numpy models) or with ``allclose`` (torch)."""
        return not self.cls._requires_torch

    def make(self) -> _FactorModel:
        """A fresh estimator from (a deep copy of) the parameters; torch models on CPU."""
        est = self.cls(**copy.deepcopy(self.params))
        if "device" in est.get_params(deep=False):
            est.set_params(device="cpu")
        return est

    def fitted_copy(self) -> _FactorModel:
        assert self.fitted is not None, "no fitted estimator (the fit check did not run)"
        return copy.deepcopy(self.fitted)


def _quiet_fit(est: _FactorModel, data: Any, **kwargs: Any) -> _FactorModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return est.fit(data, **kwargs)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_parameters(ctx: _Context) -> None:
    sig = inspect.signature(ctx.cls.__init__)
    for p in list(sig.parameters.values())[1:]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"__init__ parameter {p.name!r} is {p.kind.description}; hyperparameters must be keyword-only."
        )
    est = ctx.cls(**ctx.params)
    params = est.get_params(deep=False)
    state = vars(est)
    diff = _diff_keys(state, params)
    assert not diff, f"vars(est) != get_params() for {diff}: __init__ must only store its arguments."
    for key, value in ctx.params.items():
        assert state[key] is value, f"__init__ stored a copy or a converted value of {key!r}; store the argument as is."
    cloned = clone(est)
    assert type(cloned) is type(est) and cloned is not est, "clone did not return a new estimator of the same class."
    assert not _diff_keys(cloned.get_params(deep=False), params), "clone changed the parameters."
    assert cloned.set_params(**params) is cloned, "set_params must return self."
    assert not _diff_keys(cloned.get_params(deep=False), params), "set_params(**get_params()) changed the parameters."
    assert isinstance(repr(est), str)
    _expect_raises(TypeError, lambda: ctx.cls(**params, **{_BOGUS_KEY: 1}), "the constructor with an unknown keyword")
    err = _expect_raises(ConfigurationError, lambda: est.set_params(**{_BOGUS_KEY: 1}), "set_params with an unknown name")
    assert _BOGUS_KEY in str(err), f"the set_params error does not name the unknown key: {err}"
    if params:
        target = max(params, key=len)
        typo = target + "x"
        err = _expect_raises(ConfigurationError, lambda: est.set_params(**{typo: 1}), "set_params with a typo")
        assert repr(target) in str(err), f"the set_params error for {typo!r} does not suggest {target!r}: {err}"
    requests = sorted(n for n in dir(est) if n.startswith("set_") and n.endswith("_request"))
    assert not requests, f"sklearn generated metadata-request setters {requests}."
    assert callable(getattr(est, "get_metadata_routing", None)), "get_metadata_routing is missing."


def _check_construction(ctx: _Context) -> None:
    imported = sys.modules.get("torch")
    torch_state = None if imported is None else imported.random.get_rng_state()
    np_state = np.random.get_state()
    ctx.cls(**ctx.params)
    if imported is not None:
        assert imported.equal(torch_state, imported.random.get_rng_state()), "the constructor changed the torch RNG."
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:], (
        "the constructor changed the global numpy RNG state."
    )


def _check_unfitted(ctx: _Context) -> None:
    est = ctx.make()
    for name in ("params_", "states_", "factors_", "index_", "n_iter_", "history_"):
        assert not hasattr(est, name), f"hasattr(est, {name!r}) is True before fit."
        _expect_raises(NotFittedError, lambda name=name: getattr(est, name), f"accessing {name} before fit")
    _expect_raises(_SklearnNotFittedError, lambda: check_is_fitted(est), "sklearn's check_is_fitted before fit")
    verbs: list = [("update", lambda: est.update(ctx.data))]
    if ctx.forecaster:
        verbs += [
            ("predict", lambda: est.predict()),
            ("predict_interval", lambda: est.predict_interval()),
            ("predict_factors", lambda: est.predict_factors()),
            ("nowcast", lambda: est.nowcast()),
        ]
    else:
        verbs.append(("allocate", lambda: est.allocate()))
    for name, call in verbs:
        _expect_raises(NotFittedError, call, f"{name} before fit")
    with tempfile.TemporaryDirectory() as tmp:
        _expect_raises(NotFittedError, lambda: est.save(Path(tmp) / "run"), "save before fit")
        assert not (Path(tmp) / "run").exists(), "save before fit wrote a directory."
    assert not est._is_fitted(), "a call before fit left the estimator looking fitted."


def _check_fit(ctx: _Context) -> None:
    est = ctx.make()
    returned = _quiet_fit(est, ctx.train)
    assert returned is est, f"fit returned {type(returned).__name__}, expected self."
    for name in ("n_features_in_", "feature_names_in_", "index_", "params_", "states_", "factors_", "n_iter_",
                 "converged_", "stop_reason_", "history_"):
        assert name in vars(est) or hasattr(type(est), name), f"fit did not set {name}."
    assert est.n_features_in_ == ctx.train.n_series
    assert list(est.feature_names_in_) == list(ctx.train.series_names), "feature_names_in_ is not in the user's order."
    assert est.index_.equals(ctx.train.index) and type(est.index_) is type(ctx.train.index), "index_ is not the data's index."
    assert isinstance(est.params_, Namespace) and isinstance(est.states_, Namespace), "params_/states_ are not namespaces."
    assert est.factors_ is est.states_.factors, "factors_ is not states_.factors."
    assert isinstance(est.factors_, pd.DataFrame), "factors_ is not a DataFrame."
    if ctx.forecaster:
        assert est.factors_.index.equals(est.index_), "factors_ is not indexed by index_."
    else:
        assert est.factors_.index[: len(est.index_)].equals(est.index_), "factors_ does not start with index_."
    assert est.factors_.columns.name == "factor", "the factor axis is not named 'factor'."
    assert est.stop_reason_ in STOP_REASONS, f"stop_reason_ {est.stop_reason_!r} is not in {STOP_REASONS}."
    assert est.converged_ == (est.stop_reason_ in ("converged", "early_stopping", "not_trainable")), (
        "converged_ does not follow stop_reason_."
    )
    history = est.history_
    assert isinstance(history, pd.DataFrame) and len(history) == est.n_iter_, "len(history_) != n_iter_."
    assert history.index.name == "iteration" and list(history.index) == list(range(1, est.n_iter_ + 1))
    assert list(history.columns[:1]) == ["objective"] and history.columns[-1] == "elapsed", (
        f"history_ columns {list(history.columns)} must start with 'objective' and end with 'elapsed'."
    )
    if est._accepts_exog:
        assert est.exog_names_in_ == tuple(ctx.train.exog_names), "exog_names_in_ is not the data's exog names."
    if est._accepts_characteristics:
        assert est.characteristic_names_in_ == tuple(ctx.train.characteristic_names)
    other = _quiet_fit(ctx.make(), ctx.train)
    _same_frame(est._reload_output(), other._reload_output(), "two fits of fresh estimators", exact=ctx.exact)
    _same_frame(est.factors_, other.factors_, "factors_ of two fits", exact=ctx.exact)
    ctx.fitted = est


def _check_fit_attributes(ctx: _Context) -> None:
    est = ctx.make()
    before = set(vars(est))
    params = copy.deepcopy(est.get_params(deep=False))
    data = copy.deepcopy(ctx.train)
    _quiet_fit(est, ctx.train)
    added = set(vars(est)) - before
    bad = sorted(k for k in added if not k.endswith("_"))
    assert not bad, f"fit added attributes that do not end in '_': {bad}."
    assert not sorted(before - set(vars(est))), "fit removed attributes."
    assert not _diff_keys(est.get_params(deep=False), params), "fit changed the hyperparameters."
    assert ctx.train._equals(data), "fit mutated the input PanelData."

    class Stop(Exception):
        pass

    def raise_(info: dict) -> None:
        raise Stop

    if est.n_iter_ > 0:
        _expect_raises(Stop, lambda: _quiet_fit(est, ctx.train, callback=raise_), "fit with a raising callback")
        assert not est._is_fitted() and not hasattr(est, "params_"), "a failed fit left the model fitted."
    _expect_raises(TypeError, lambda: est.fit(ctx.train, callback="print"), "fit with a callback that is not callable")


def _check_callback(ctx: _Context) -> None:
    est = ctx.make()
    infos: list = []
    _quiet_fit(est, ctx.train, callback=infos.append)
    assert len(infos) == est.n_iter_, f"the callback ran {len(infos)} times for {est.n_iter_} iterations."
    for i, info in enumerate(infos, start=1):
        assert info["iteration"] == i, f"info['iteration'] is {info['iteration']}, expected {i}."
        row = est.history_.loc[i]
        assert set(info) == {"iteration", *row.index}, f"info keys {sorted(info)} != iteration + history_ columns."
    if est.n_iter_ == 0:
        return
    stopped = ctx.make()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        stopped.fit(ctx.train, callback=lambda info: True)
    assert stopped.stop_reason_ == "callback" and stopped.n_iter_ == 1, (
        f"a callback returning True stopped with {stopped.stop_reason_!r} after {stopped.n_iter_} iteration(s)."
    )
    assert not stopped.converged_, "a callback stop reported converged_."


def _check_frame_axes(frame: pd.DataFrame, where: str, axis_labels: Mapping) -> None:
    """Named row axes; columns named when they are series or factors; numeric values float64."""
    names = list(frame.index.names)
    assert all(n is not None for n in names), f"{where}: the index has unnamed levels {names}."
    for level, labels in enumerate(frame.columns.levels if isinstance(frame.columns, pd.MultiIndex) else [frame.columns]):
        for axis, known in axis_labels.items():
            if len(labels) and set(labels) <= set(known):
                assert frame.columns.names[level] is not None, f"{where}: the {axis} columns are unnamed."
    for col, dtype in frame.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            assert dtype == np.float64, f"{where}: column {col!r} is {dtype}, not float64."


def _check_namespaces(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    axis_labels = {"series": list(est._panel_.series_names), "factor": list(est.factors_.columns)}
    for attr in ("params_", "states_", "regimes_"):
        ns = vars(est).get(attr)
        if ns is None:
            continue
        assert isinstance(ns, Namespace), f"{attr} is not a Namespace."
        assert isinstance(repr(ns), str) and set(ns) <= set(dir(ns)), f"dir({attr}) does not list its fields."
        for name in ns:
            value = ns[name]
            if isinstance(value, pd.DataFrame):
                _check_frame_axes(value, f"{attr}.{name}", axis_labels)
            elif isinstance(value, pd.Series):
                assert value.index.name is not None, f"{attr}.{name}: the index is unnamed."
    time_name = est.index_.name if est.index_.name is not None else "time"
    assert est.factors_.index.name == time_name, f"factors_ index name {est.factors_.index.name!r} != {time_name!r}."


def _check_forecaster_outputs(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    names = list(est.feature_names_in_)
    time_name = est.index_.name if est.index_.name is not None else "time"
    one = est.predict()
    assert len(one) == 1 and one.index[0] > est.index_[-1], "predict() is not the step after the origin."
    three = est.predict(3)
    assert isinstance(three, pd.DataFrame) and len(three) == 3, "predict(3) did not return 3 rows."
    assert three.index[:1].equals(one.index) and three.index.name == time_name, "predict(3) has the wrong index."
    assert list(three.columns) == names and three.columns.name == "series", "predict columns are not the series."
    assert all(dtype == np.float64 for dtype in three.dtypes), "predict values are not float64."
    k = min(6, len(est.index_))
    past = est.predict(0, start=-(k - 1))
    assert past.index.equals(est.index_[-k:]), "predict(0, start=-(k-1)) is not the last k conditioning rows."
    data = ctx.train.y.to_numpy()[-k:]
    observed = ~np.isnan(data)
    assert np.array_equal(past.to_numpy()[observed], data[observed]), "predict at observed in-sample cells is not the data."
    bands = est.predict_interval(3, coverage=0.8)
    assert bands.columns.names == ["series", "bound"], "predict_interval columns are not (series, bound)."
    assert list(bands.columns) == [(s, b) for s in names for b in ("lower", "upper")]
    lower, upper = (bands.xs(b, axis=1, level="bound").to_numpy() for b in ("lower", "upper"))
    finite = np.isfinite(lower) & np.isfinite(upper)
    assert np.all(lower[finite] <= upper[finite]), "predict_interval has lower > upper."
    past_bands = est.predict_interval(0, start=-(k - 1))
    for bound in ("lower", "upper"):
        values = past_bands.xs(bound, axis=1, level="bound").to_numpy()
        assert np.array_equal(values[observed], data[observed]), f"the in-sample {bound} bound is not the data."
    factors = est.predict_factors(3)
    assert factors.index.equals(three.index) and factors.columns.equals(est.factors_.columns), (
        "predict_factors has the wrong index or columns."
    )
    assert factors.columns.name == "factor"
    table = est.nowcast()
    assert list(table.columns) == ["estimate", "lower", "upper", "actual", "status"], "nowcast has the wrong columns."
    assert table.index.names == ["series", "period"], "nowcast rows are not (series, period)."
    assert set(table["status"]) <= {"released", "backcast", "nowcast", "forecast"}
    released = table[table["status"] == "released"]
    assert (released["estimate"] == released["actual"]).all() and (released["lower"] == released["actual"]).all()
    clock_series = [s for s in names if s not in est._slower_]
    if clock_series:
        name = clock_series[0]
        rows = est.nowcast(name, end=1, start=0)
        expected = est.predict(1, start=0)[name].to_numpy()
        assert np.array_equal(rows["estimate"].to_numpy(), expected, equal_nan=True), "nowcast is not predict's view."
        assert list(rows["status"]) == ["nowcast", "forecast"] or rows["status"].iloc[0] == "released"
    for bad in (1.5, None, True, [1]):
        _expect_raises(TypeError, lambda bad=bad: est.predict(bad), f"predict({bad!r})")
    for coverage in (0, 1, 1.5, float("nan"), True, "0.9"):
        _expect_raises(ConfigurationError, lambda c=coverage: est.predict_interval(1, coverage=c), f"coverage={coverage!r}")
    _expect_raises(ConfigurationError, lambda: est.predict(-1, start=1), "predict with start after end")
    _expect_raises(ConfigurationError, lambda: est.predict(0, start=-10_000), "predict before the first stamp")
    if not est._needs_future_exog(len(est.index_)):
        future = pd.DataFrame({"x": [0.0]}, index=one.index)
        _expect_raises(ConfigurationError, lambda: est.predict(1, exog=future), "predict with exog it does not use")


def _check_allocator_outputs(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    weights = est.allocate()
    assert isinstance(weights, pd.DataFrame) and len(weights) == 1, "allocate() did not return one row."
    assert weights.index[0] > est.index_[-1], "allocate() is not indexed by the stamp after the origin."
    time_name = est.index_.name if est.index_.name is not None else "time"
    assert weights.index.name == time_name, "allocate() index name differs from index_."
    assert list(weights.columns) == list(est._panel_.series_names) and weights.columns.name == "series"
    assert all(dtype == np.float64 for dtype in weights.dtypes), "allocate() values are not float64."


def _check_outputs(ctx: _Context) -> None:
    if ctx.forecaster:
        _check_forecaster_outputs(ctx)
    else:
        _check_allocator_outputs(ctx)


def _verbs(est: _FactorModel) -> list:
    if isinstance(est, _Forecaster):
        return [lambda: est.predict(3), lambda: est.predict_interval(2, start=-2), lambda: est.predict_factors(2),
                lambda: est.nowcast()]
    return [lambda: est.allocate()]


def _check_purity(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    before = _state_hash(est)
    for verb in _verbs(est):
        first, second = verb(), verb()
        _same_frame(first, second, "two identical calls")
    assert _state_hash(est) == before, "a predict/allocate call changed the fitted state."


def _check_update(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    params, history = est.params_, est.history_.copy()
    diagnostics = (est.n_iter_, est.stop_reason_, est.converged_)
    returned = est.update(ctx.new)
    assert returned is est, "update must return self."
    assert est.params_ is params, "update replaced params_."
    assert (est.n_iter_, est.stop_reason_, est.converged_) == diagnostics and est.history_.equals(history), (
        "update changed the diagnostics."
    )
    assert est.index_.equals(ctx.data.index), "index_ is not the merged index after update."
    assert est.factors_ is est.states_.factors, "factors_ is not states_.factors after update."
    if ctx.forecaster:
        k = min(6, len(est.index_))
        past = est.predict(0, start=-(k - 1)).to_numpy()
        data = ctx.data.y.to_numpy()[-k:]
        observed = ~np.isnan(data)
        assert np.array_equal(past[observed], data[observed]), "predict at observed cells is not the updated data."
    output = est._reload_output()
    factors = est.factors_.copy()
    est.update(ctx.data)  # the current panel again: nothing changes
    _same_frame(output, est._reload_output(), "update(current panel)", exact=ctx.exact)
    _same_frame(factors, est.factors_, "factors_ after update(current panel)", exact=ctx.exact)
    full = ctx.fitted_copy().update(ctx.data)  # the full vintage versus the new rows only
    _same_frame(output, full._reload_output(), "update(full vintage) vs update(new rows)", exact=ctx.exact)
    est.update(ctx.train, replace=True)
    assert est.index_.equals(ctx.train.index), "update(replace=True) did not condition on exactly the data."
    _same_frame(ctx.fitted._reload_output(), est._reload_output(), "update(train, replace=True)", exact=ctx.exact)


def _check_silence(ctx: _Context) -> None:
    est = ctx.make()
    if "verbose" in est.get_params(deep=False):
        est.set_params(verbose=0)
    package_logger = logging.getLogger("dfm_python")
    handler = _ListHandler()
    saved_level = package_logger.level
    package_logger.addHandler(handler)
    if package_logger.getEffectiveLevel() > logging.INFO:
        package_logger.setLevel(logging.INFO)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(ctx.train)
            for verb in _verbs(est):
                verb()
            est.update(ctx.new)
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(saved_level)
    assert not out.getvalue(), f"verbose=0 printed to stdout: {out.getvalue()[:200]!r}"
    assert not err.getvalue(), f"verbose=0 printed to stderr: {err.getvalue()[:200]!r}"
    logged = [f"{r.name}:{r.levelname}:{r.getMessage()}" for r in handler.records if r.levelno >= logging.INFO]
    assert not logged, f"verbose=0 logged at INFO or above: {logged[:3]}"


def _assert_reloaded(est: _FactorModel, loaded: Any, ctx: _Context, what: str) -> None:
    assert type(loaded) is type(est), f"{what} returned {type(loaded).__name__}, expected {type(est).__name__}."
    diff = _diff_keys(loaded.get_params(deep=False), est.get_params(deep=False))
    assert not diff, f"{what} changed the parameters {diff}."
    for name in ("n_features_in_", "feature_names_in_", "exog_names_in_", "characteristic_names_in_", "n_iter_",
                 "converged_", "stop_reason_", "loglik_"):
        if name in vars(est):
            assert name in vars(loaded) and _deep_equal(vars(est)[name], vars(loaded)[name]), f"{what} lost {name}."
    assert loaded.history_.equals(est.history_), f"{what} restored a different history_."
    index, other = est.index_, loaded.index_
    assert type(other) is type(index) and other.equals(index) and other.name == index.name, f"{what}: index_ differs."
    assert getattr(other, "freq", None) == getattr(index, "freq", None), f"{what} lost the frequency of index_."
    _same_frame(est._reload_output(), loaded._reload_output(), f"{what}: the outputs", exact=ctx.exact)
    _same_frame(est.factors_, loaded.factors_, f"{what}: factors_", exact=ctx.exact)


def _check_save_load(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    torch_model = ctx.cls._requires_torch
    before = _state_hash(est)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "run"
        assert est.save(path) is None, "save must return None."
        assert _state_hash(est) == before, "save changed the fitted state."
        assert sorted(p.name for p in root.iterdir()) == ["run"], "save left files next to the run."
        files = {p.name for p in path.iterdir()}
        expected = set(REQUIRED_FILES_2) | ({WEIGHTS_FILE} if torch_model else set())
        assert files == expected, f"save wrote {sorted(files)}, expected {sorted(expected)}."
        config = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
        assert config.get("format_version") == FORMAT_2 and config.get("model_type") == ctx.cls.__name__
        assert set(config.get("params", {})) == set(est.get_params(deep=False)), "config.json params != get_params()."
        assert "library_version" in config and "runtime" in config
        _expect_raises(FileExistsError, lambda: est.save(path), "save to an existing directory")

        loaded = ctx.cls._load(path, device="cpu")
        _assert_reloaded(est, loaded, ctx, f"{ctx.cls.__name__}._load")
        if _loading._MODELS.get(ctx.cls.__name__) in (ctx.cls, f"{ctx.cls.__module__}:{ctx.cls.__name__}"):
            from .io.loading import load

            _assert_reloaded(est, load(path, device="cpu"), ctx, "dfm_python.load")
        updated, reloaded_updated = copy.deepcopy(est).update(ctx.new), loaded.update(ctx.new)
        _same_frame(updated._reload_output(), reloaded_updated._reload_output(), "update after load", exact=ctx.exact)

        est.save(path, overwrite=True)
        assert {p.name for p in path.iterdir()} == files, "save(overwrite=True) wrote another file set."

        format_1 = root / "format_1"
        format_1.mkdir()
        (format_1 / CONFIG_FILE).write_text(
            json.dumps({"format_version": 1, "model_type": ctx.cls.__name__, "params": {}}), encoding="utf-8"
        )
        err = _expect_raises(ConfigurationError, lambda: ctx.cls._load(format_1), "loading a format-1 run")
        assert "format 1" in str(err), f"the format-1 error does not say so: {err}"


def _always(ctx: _Context) -> bool:
    return True


_CHECKS: tuple = (
    ("parameters", _check_parameters, _always),
    ("construction", _check_construction, _always),
    ("unfitted", _check_unfitted, _always),
    ("fit", _check_fit, _always),
    ("fit_attributes", _check_fit_attributes, _always),
    ("callback", _check_callback, _always),
    ("namespaces", _check_namespaces, _always),
    ("outputs", _check_outputs, _always),
    ("purity", _check_purity, _always),
    ("update", _check_update, _always),
    ("silence", _check_silence, _always),
    ("save_load", _check_save_load, _always),
)


def check_estimator(estimator_or_class: Any, *, data: Any = None) -> list[str]:
    """Run the contract checks on a model class (or instance) built on ``_Forecaster`` or ``_Allocator``.

    Parameters
    ----------
    estimator_or_class : class or instance
        A class is checked once per parameter set from its ``_test_params()``;
        an instance is checked with its own parameters.
    data : PanelData, DataFrame or 2-D array, optional
        The fixture; defaults to :func:`make_panel`. The checks fit on all but
        the last few rows and use those rows as new data.

    Returns
    -------
    list of str
        The names of the checks that ran, in order.

    Raises
    ------
    TypeError
        For a class that is not a ``_Forecaster`` or ``_Allocator``.
    AssertionError
        On the first failing check; the message names the check and the
        parameter set.
    """
    if isinstance(estimator_or_class, type):
        cls, params_list = estimator_or_class, None
    else:
        cls, params_list = type(estimator_or_class), [estimator_or_class.get_params(deep=False)]
    if not (issubclass(cls, (_Forecaster, _Allocator))):
        raise TypeError(f"check_estimator expects a _Forecaster or _Allocator subclass, got {cls.__name__}.")
    if params_list is None:
        params_list = cls._test_params()
    panel = make_panel(cls) if data is None else PanelData._coerce(data)
    train, new = _split(panel)
    ran: list[str] = []
    for params in params_list:
        label = f"{cls.__name__}({', '.join(f'{k}={v!r}' for k, v in params.items())})"
        ctx = _Context(cls=cls, params=dict(params), data=panel, train=train, new=new, label=label)
        for name, check, applies in _CHECKS:
            if not applies(ctx):
                continue
            try:
                check(ctx)
            except Exception as err:  # harness boundary: any failure is reported with the check and parameters
                detail = str(err) if isinstance(err, AssertionError) else f"{type(err).__name__}: {err}"
                raise AssertionError(f"check_estimator: check {name!r} failed for {label}: {detail}") from err
            if name not in ran:
                ran.append(name)
    return ran

"""Contract checks for estimators built on :class:`dfm_python.base.BaseFactorModel`.

:func:`check_estimator` runs the checks of the API contract that apply to an
estimator's tags and raises ``AssertionError`` naming the first check that
fails. :func:`make_panel` builds the tiny CPU fixture the checks use.

The checks, in order:

==========================  ================================================
Check                       What it asserts
==========================  ================================================
``tags``                    tag names and values are valid and consistent
``pure_construction``       the constructor leaves the global torch and numpy
                            RNG states (and CUDA) untouched and stores its
                            arguments unchanged: ``vars(est) == get_params()``
``parameters``              keyword-only ``__init__``; ``get_params``,
                            ``set_params`` and ``clone`` round-trip; unknown
                            names raise
``config_mirror``           the config's fields and defaults equal the
                            constructor's; ``from_config(get_config())``
``config_serialization``    dict, JSON and YAML round-trips are lossless
``unknown_keys``            unknown config keys and flat YAML files raise
                            ``ConfigurationError`` naming the closest key
``unfitted``                ``result_`` and the fitted-only methods raise
                            ``NotFittedError``; ``hasattr(est, "result_")``
                            is False
``fit``                     ``fit`` returns ``self``; two fits give
                            bitwise-identical ``result_`` arrays (unless tag
                            ``non_deterministic``)
``fit_attributes``          ``fit`` adds only trailing-underscore attributes
                            and mutates neither the data nor the parameters
``output_shape``            forecast, factor and weight frames have the
                            documented index, columns, dtype and attrs
``purity``                  ``predict`` / ``allocate`` leave the fitted state
                            unchanged, also with ``data=``
``capabilities``            every slot or layout the tags exclude raises
                            ``UnsupportedInputError`` naming the tag
``device``                  ``device="cpu"`` puts every parameter on CPU
``silence``                 with ``verbose=0``, ``fit`` prints nothing and logs
                            nothing at INFO
``update``                  ``update`` returns ``self``, keeps the parameters
                            and moves the forecast origin
``save_load``               ``save`` writes the section 7 file set for the
                            backend (no ``scaler.pkl`` for the known
                            scalers) atomically and refuses an existing
                            directory; ``Model.load`` restores the params,
                            data attributes, ``scaler_``, a bitwise-equal
                            ``result_`` and bitwise-equal ``predict`` /
                            ``allocate`` outputs; ``dfm_python.load``
                            dispatches registered estimators; with tag
                            ``capability:device``, a run saved with a CUDA
                            ``device`` loads on CPU where CUDA is missing
                            (simulated), and a model fitted on CUDA loads
                            with ``device="cpu"`` (skipped without CUDA)
==========================  ================================================

Each check builds its own estimator from the parameter set, so they do not
depend on each other except through the shared fitted estimator that the
``fit`` check produces. Torch estimators run with ``device="cpu"``. The
checks for a parameter set are gated by the tags of an estimator built from
it (``est.get_tags()``), since some tags depend on hyperparameters.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import difflib
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import shutil
import sys
import tempfile
import types
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError as _SklearnNotFittedError
from sklearn.utils.validation import check_is_fitted

from .base import DEFAULT_TAGS, BaseFactorModel
from .config.estimator import EstimatorConfig, load_yaml
from .config.types import is_torch_tensor
from .dataset.panel import PanelData
from .io.layout import CONFIG_FILE, FORMAT_VERSION, REQUIRED_FILES, WEIGHTS_FILE
from .io.loading import load as _load_any
from .io.scaler import SCALER_JSON, SCALER_PKL, is_known_scaler
from .registry import all_estimators
from .utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError
from .utils.optional import require_torch, torch

__all__ = ["check_estimator", "make_panel"]

# Forecast horizon used by the checks.
_H = 3
_BOGUS_KEY = "definitely_not_a_parameter"
_VALID_TAG_VALUES = {
    "task": ("forecast", "allocation"),
    "backend": ("numpy", "torch"),
    "update_kind": (None, "filter", "encoder_forward_rescale", "encoder_forward", "reattach"),
    "predict_data": (None, "new_rows", "full_panel", "window"),
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _month_end_index(n: int) -> pd.DatetimeIndex:
    try:
        return pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    except ValueError:  # pandas < 2.2 spells month end "M"
        return pd.date_range("2000-01-31", periods=n, freq="M", name="date")


def make_panel(tags: Any, *, seed: int = 0) -> PanelData:
    """Build a tiny CPU fixture (48 monthly steps, 3 series) for an estimator's tags.

    Parameters
    ----------
    tags : dict, or an estimator class or instance
        Missing tags take their defaults. The panel holds ``exog`` (2
        columns), ``series_exog`` (2 features) and a mixed-frequency ``freq``
        only when the matching ``capability:*`` tag is True. With
        ``capability:missing_values``, two target values are NaN (and a
        quarterly series is NaN off quarter ends when the panel is mixed).
        Allocation models get return-sized values (about 1%).
    seed : int
        Seed of the local ``numpy.random.default_rng`` (the global RNGs are not used).
    """
    if not isinstance(tags, Mapping):
        tags = tags.get_tags()
    tags = {**DEFAULT_TAGS, **tags}
    rng = np.random.default_rng(seed)
    n_time, n_series = 48, 3
    index = _month_end_index(n_time)
    names = [f"s{i}" for i in range(n_series)]

    if tags["task"] == "allocation":
        y = 0.01 * rng.standard_normal((n_time, n_series))
    else:
        factor = np.zeros(n_time)
        shocks = rng.standard_normal(n_time)
        for t in range(1, n_time):
            factor[t] = 0.7 * factor[t - 1] + shocks[t]
        y = np.outer(factor, rng.standard_normal(n_series)) + 0.5 * rng.standard_normal((n_time, n_series))

    freq = None
    if tags["capability:mixed_frequency"]:
        freq = {names[0]: "q", **{name: "m" for name in names[1:]}}
        if tags["capability:missing_values"]:
            y[index.month % 3 != 0, 0] = np.nan
    if tags["capability:missing_values"]:
        y[[3, 17], n_series - 1] = np.nan

    exog = rng.standard_normal((n_time, 2)) if tags["capability:exog"] else None
    series_exog = rng.standard_normal((n_time, n_series, 2)) if tags["capability:series_exog"] else None
    return PanelData.from_arrays(
        y,
        exog=exog,
        series_exog=series_exog,
        index=index,
        names=names,
        exog_names=["x0", "x1"] if exog is not None else None,
        series_exog_names=["c0", "c1"] if series_exog is not None else None,
        freq=freq,
    )


# ---------------------------------------------------------------------------
# Comparison and hashing helpers
# ---------------------------------------------------------------------------


def _is_torch_module(obj: Any) -> bool:
    # A torch module exists only once torch is imported, so this never imports it.
    imported = sys.modules.get("torch")
    return imported is not None and isinstance(obj, imported.nn.Module)


def _deep_equal(a: Any, b: Any) -> bool:
    """Structural equality: arrays by value (NaN equals NaN), estimators by params, tuple != list."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.hasobject:
            return all(_deep_equal(x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist()))
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if is_torch_tensor(a) or is_torch_tensor(b):
        return is_torch_tensor(a) and is_torch_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (pd.DataFrame, pd.Series, pd.Index)):
        return type(a) is type(b) and a.equals(b)
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
    if is_torch_tensor(obj):
        t = obj.detach().cpu().contiguous()
        h.update(f"pt:{t.dtype}:{tuple(t.shape)};".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(obj, (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
        h.update(f"<{getattr(obj, '__qualname__', getattr(obj, '__name__', type(obj).__name__))}>;".encode())
        return
    if id(obj) in seen:
        h.update(b"<seen>")
        return
    seen.add(id(obj))
    if _is_torch_module(obj):
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
        h.update(repr(obj.attrs).encode())
        _feed(h, obj.to_numpy(), seen, depth + 1)
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


def _state_hash(est: BaseFactorModel) -> str:
    """Hash of every instance attribute: parameters, fitted attributes (``result_``, ``network_``) and private state."""
    h = hashlib.sha256()
    _feed(h, dict(vars(est)), set())
    return h.hexdigest()


def _network_hash(est: BaseFactorModel) -> str:
    """Hash of ``network_.state_dict()``, or ``""`` without a network."""
    network = vars(est).get("network_")
    if network is None:
        return ""
    h = hashlib.sha256()
    _feed(h, network, set())
    return h.hexdigest()


def _collect_arrays(obj: Any, path: str, out: dict, seen: set) -> None:
    if isinstance(obj, np.ndarray):
        out[path] = obj.copy()
    elif is_torch_tensor(obj):
        out[path] = obj.detach().cpu().numpy().copy()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        out[path] = obj.to_numpy(copy=True)
    elif isinstance(obj, (bool, np.bool_)):
        out[path] = np.asarray(obj)
    elif isinstance(obj, (int, float, np.number)):
        out[path] = np.asarray(obj)
    elif id(obj) in seen:
        return
    elif _is_torch_module(obj):
        seen.add(id(obj))
        for key, value in obj.state_dict().items():
            _collect_arrays(value, f"{path}.{key}", out, seen)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        seen.add(id(obj))
        for f in dataclasses.fields(obj):
            _collect_arrays(getattr(obj, f.name), f"{path}.{f.name}", out, seen)
    elif isinstance(obj, Mapping):
        seen.add(id(obj))
        for key, value in obj.items():
            _collect_arrays(value, f"{path}[{key!r}]", out, seen)
    elif isinstance(obj, (list, tuple)):
        seen.add(id(obj))
        for i, value in enumerate(obj):
            _collect_arrays(value, f"{path}[{i}]", out, seen)


def _result_arrays(est: BaseFactorModel) -> dict:
    """Every array and number inside ``result_``, keyed by path."""
    out: dict = {}
    _collect_arrays(est.result_, "result_", out, set())
    return out


def _assert_bitwise(a: dict, b: dict, what: str) -> None:
    assert list(a) == list(b), f"{what}: result_ holds different arrays ({sorted(set(a) ^ set(b))})."
    for key in a:
        x, y = a[key], b[key]
        assert x.dtype == y.dtype and x.shape == y.shape, (
            f"{what}: {key} differs in dtype or shape ({x.dtype}{x.shape} vs {y.dtype}{y.shape})."
        )
        same = _deep_equal(x, y) if x.dtype.hasobject else x.tobytes() == y.tobytes()
        assert same, f"{what}: {key} is not bitwise identical."


def _expect_raises(exc_type: type, func: Callable[[], Any], what: str) -> BaseException:
    try:
        func()
    except exc_type as err:
        return err
    except Exception as err:  # harness boundary: the wrong exception is the check's failure (chained)
        raise AssertionError(f"{what} raised {type(err).__name__} ({err}), expected {exc_type.__name__}.") from err
    raise AssertionError(f"{what} did not raise {exc_type.__name__}.")


def _future_index(index: pd.Index, h: int) -> Optional[pd.Index]:
    """The ``h`` timestamps after ``index[-1]``, or None when the step cannot be inferred."""
    if isinstance(index, pd.RangeIndex):
        start = index[-1] + index.step
        return pd.RangeIndex(start, start + index.step * h, index.step)
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[-1] + 1, periods=h, freq=index.freq)
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
        if freq is None:
            return None
        return pd.date_range(index[-1], periods=h + 1, freq=freq)[1:]
    return None


def _yaml_available() -> bool:
    return importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


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
    tags: dict
    label: str
    fitted: Optional[BaseFactorModel] = None

    def make(self, *, cpu: bool = True) -> BaseFactorModel:
        """A fresh estimator from (a deep copy of) the parameters; torch models on CPU."""
        est = self.cls(**copy.deepcopy(self.params))
        if cpu and self.tags["capability:device"] and "device" in est.get_params(deep=False):
            est.set_params(device="cpu")
        return est

    def fitted_copy(self) -> BaseFactorModel:
        assert self.fitted is not None, "no fitted estimator (the fit check did not run)"
        return copy.deepcopy(self.fitted)

    @property
    def hist_len(self) -> int:
        """The ``hist_len`` hyperparameter of allocation models (rows without weights), else 0."""
        return int(self.make(cpu=False).get_params(deep=False).get("hist_len", 0) or 0)

    def conditioning_data(self) -> Optional[PanelData]:
        """What ``predict(h, data=...)`` receives, following tag ``predict_data``."""
        kind = self.tags["predict_data"]
        if kind is None:
            return None
        return self.new if kind == "new_rows" else self.data

    def update_data(self) -> PanelData:
        if self.tags["predict_data"] == "new_rows" or self.tags["update_kind"] == "filter":
            return self.new
        return self.data


def _split(data: PanelData) -> tuple[PanelData, PanelData]:
    index = data.index
    n = len(index)
    k = max(1, min(6, n // 6))
    if n - k < 2:
        raise ValueError(f"check_estimator needs at least {k + 2} time steps, got {n}.")
    return data.loc(None, index[n - k - 1]), data.loc(index[n - k], None)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_tags(ctx: _Context) -> None:
    ctx.cls.get_tags()  # the class tags must resolve too
    tags = ctx.tags  # the instance's tags (some depend on hyperparameters)
    assert set(tags) == set(DEFAULT_TAGS), f"tags {sorted(set(tags) ^ set(DEFAULT_TAGS))} are missing or unknown."
    for key, allowed in _VALID_TAG_VALUES.items():
        assert tags[key] in allowed, f"tag {key}={tags[key]!r} is not one of {allowed}."
    for key, value in tags.items():
        if key.startswith("capability:") or key == "non_deterministic":
            assert isinstance(value, bool), f"tag {key} must be a bool, got {value!r}."
    assert (tags["update_kind"] is None) == (not tags["capability:update"]), (
        "tag update_kind must be set exactly when capability:update is True."
    )
    if tags["task"] == "allocation":
        assert tags["predict_data"] is None, "allocation models have no predict, so predict_data must be None."
        assert not tags["capability:factor_forecast"], "allocation models cannot have capability:factor_forecast."
    if tags["capability:device"]:
        assert tags["backend"] == "torch", "capability:device requires backend='torch'."


def _check_pure_construction(ctx: _Context) -> None:
    # check_estimator imports torch for a torch model; for a numpy model torch is checked if something imported it.
    imported = sys.modules.get("torch")
    torch_state = None if imported is None else imported.random.get_rng_state()
    np_state = np.random.get_state()
    cuda_was_initialized = imported is not None and imported.cuda.is_initialized()
    est = ctx.cls(**ctx.params)
    if imported is not None:
        assert imported.equal(torch_state, imported.random.get_rng_state()), (
            "the constructor changed the global torch RNG state."
        )
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:], (
        "the constructor changed the global numpy RNG state."
    )
    if imported is not None and not cuda_was_initialized:
        assert not imported.cuda.is_initialized(), "the constructor initialized CUDA."
    params = est.get_params(deep=False)
    state = vars(est)
    diff = _diff_keys(state, params)
    assert not diff, f"vars(est) != get_params() for {diff}: __init__ must only store its arguments."
    for key, value in ctx.params.items():
        assert state[key] is value, f"__init__ stored a copy or a converted value of {key!r}; store the argument as is."


def _check_parameters(ctx: _Context) -> None:
    sig = inspect.signature(ctx.cls.__init__)
    for p in list(sig.parameters.values())[1:]:
        assert p.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"__init__ parameter {p.name!r} is {p.kind.description}; hyperparameters must be keyword-only."
        )
    est = ctx.make(cpu=False)
    params = est.get_params(deep=False)
    cloned = clone(est)
    assert type(cloned) is type(est) and cloned is not est, "clone did not return a new estimator of the same class."
    diff = _diff_keys(cloned.get_params(deep=False), params)
    assert not diff, f"clone changed parameters {diff}."
    returned = cloned.set_params(**params)
    assert returned is cloned, "set_params must return self."
    diff = _diff_keys(cloned.get_params(deep=False), params)
    assert not diff, f"set_params(**get_params()) changed parameters {diff}."
    assert isinstance(repr(est), str)
    _expect_raises(TypeError, lambda: ctx.cls(**params, **{_BOGUS_KEY: 1}), "the constructor with an unknown keyword")
    err = _expect_raises(ConfigurationError, lambda: est.set_params(**{_BOGUS_KEY: 1}), "set_params with an unknown name")
    assert _BOGUS_KEY in str(err), f"the set_params error does not name the unknown key: {err}"


def _config_class(ctx: _Context) -> type:
    cfg_cls = ctx.cls._config_class
    assert cfg_cls is not None, "_config_class is not set; every estimator needs an EstimatorConfig mirror."
    assert isinstance(cfg_cls, type) and issubclass(cfg_cls, EstimatorConfig), (
        f"_config_class {cfg_cls!r} is not an EstimatorConfig subclass."
    )
    assert dataclasses.is_dataclass(cfg_cls), f"{cfg_cls.__name__} is not a dataclass."
    return cfg_cls


def _check_config_mirror(ctx: _Context) -> None:
    cfg_cls = _config_class(ctx)
    sig_defaults = {
        p.name: p.default for p in list(inspect.signature(ctx.cls.__init__).parameters.values())[1:]
    }
    field_defaults = {}
    for f in dataclasses.fields(cfg_cls):
        if f.default is not dataclasses.MISSING:
            field_defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            field_defaults[f.name] = f.default_factory()
        else:
            field_defaults[f.name] = inspect.Parameter.empty
    only_init = sorted(set(sig_defaults) - set(field_defaults))
    only_cfg = sorted(set(field_defaults) - set(sig_defaults))
    assert not only_init and not only_cfg, (
        f"{cfg_cls.__name__} fields differ from the constructor: missing {only_init}, extra {only_cfg}."
    )
    wrong = [k for k in sig_defaults if not _deep_equal(sig_defaults[k], field_defaults[k])]
    assert not wrong, f"{cfg_cls.__name__} defaults differ from the constructor's for {wrong}."

    est = ctx.make(cpu=False)
    params = est.get_params(deep=False)
    cfg = est.get_config()
    assert type(cfg) is cfg_cls, f"get_config returned {type(cfg).__name__}, expected {cfg_cls.__name__}."
    diff = _diff_keys(cfg.to_dict(), params)
    assert not diff, f"get_config().to_dict() != get_params() for {diff}."
    rebuilt = ctx.cls.from_config(cfg)
    assert type(rebuilt) is ctx.cls, "from_config returned another class."
    diff = _diff_keys(rebuilt.get_params(deep=False), params)
    assert not diff, f"from_config(get_config()) changed parameters {diff}."
    diff = _diff_keys(ctx.cls.from_config(cfg_cls.from_dict(cfg.to_dict())).get_params(deep=False), params)
    assert not diff, f"from_config(from_dict(to_dict())) changed parameters {diff}."


def _check_config_serialization(ctx: _Context) -> None:
    cfg_cls = _config_class(ctx)
    est = ctx.make(cpu=False)
    cfg = est.get_config()
    params = est.get_params(deep=False)
    diff = _diff_keys(cfg_cls.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())
    assert not diff, f"from_dict(to_dict()) is not lossless for {diff}."
    plain = cfg.to_plain_dict()
    via_json = cfg_cls.from_dict(json.loads(json.dumps(plain)))
    diff = _diff_keys(via_json.to_dict(), cfg.to_dict())
    assert not diff, f"the JSON round-trip is not lossless for {diff}."
    if not _yaml_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        cfg.to_yaml(path)
        sections = load_yaml(path)
        assert set(sections) == {"model", "data"} and sections["data"] == {}, f"load_yaml returned {sections}."
        diff = _diff_keys(sections["model"], json.loads(json.dumps(plain)))
        assert not diff, f"the YAML model: section differs from to_plain_dict() for {diff}."
        diff = _diff_keys(cfg_cls.from_yaml(path).to_dict(), cfg.to_dict())
        assert not diff, f"the YAML round-trip is not lossless for {diff}."
        diff = _diff_keys(ctx.cls.from_config(cfg_cls.from_yaml(path)).get_params(deep=False), params)
        assert not diff, f"from_config(from_yaml(...)) changed parameters {diff}."


def _check_unknown_keys(ctx: _Context) -> None:
    cfg_cls = _config_class(ctx)
    cfg = ctx.make(cpu=False).get_config()
    values = cfg.to_dict()
    names = list(values)
    err = _expect_raises(
        ConfigurationError, lambda: cfg_cls.from_dict({**values, _BOGUS_KEY: 1}), "from_dict with an unknown key"
    )
    assert _BOGUS_KEY in str(err), f"the from_dict error does not name the unknown key: {err}"
    if names:
        target = max(names, key=len)
        typo = next(c for c in (target + "x", "x" + target, target + "_x") if c not in names)
        expected = difflib.get_close_matches(typo, names, n=1)[0]
        err = _expect_raises(ConfigurationError, lambda: cfg_cls.from_dict({typo: values[target]}), "from_dict with a typo")
        assert repr(expected) in str(err), f"the from_dict error for {typo!r} does not suggest {expected!r}: {err}"
    if _yaml_available():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.yaml"
            path.write_text(f"{names[0] if names else 'n_factors'}: 1\n", encoding="utf-8")
            err = _expect_raises(ConfigurationError, lambda: cfg_cls.from_yaml(path), "from_yaml on a flat file")
            assert "model" in str(err), f"the flat-file error does not mention the 'model:' section: {err}"


def _check_unfitted(ctx: _Context) -> None:
    est = ctx.make()
    tags = ctx.tags
    assert not hasattr(est, "result_"), "hasattr(est, 'result_') is True before fit."
    _expect_raises(NotFittedError, lambda: est.result_, "accessing result_ before fit")
    _expect_raises(_SklearnNotFittedError, lambda: check_is_fitted(est), "sklearn's check_is_fitted before fit")
    if tags["task"] == "forecast":
        _expect_raises(NotFittedError, lambda: est.predict(_H), "predict before fit")
    if tags["capability:factor_forecast"]:
        _expect_raises(NotFittedError, lambda: est.predict_factors(_H), "predict_factors before fit")
    if tags["task"] == "allocation":
        _expect_raises(NotFittedError, lambda: est.allocate(), "allocate before fit")
    if tags["capability:update"]:
        _expect_raises(NotFittedError, lambda: est.update(ctx.data), "update before fit")
    if callable(getattr(est, "save", None)):
        with tempfile.TemporaryDirectory() as tmp:
            _expect_raises(NotFittedError, lambda: est.save(Path(tmp) / "run"), "save before fit")
    assert not hasattr(est, "result_"), "a call before fit left the estimator looking fitted."


def _check_fit(ctx: _Context) -> None:
    est = ctx.make()
    returned = est.fit(ctx.train)
    assert returned is est, f"fit returned {type(returned).__name__}, expected self."
    assert "result_" in vars(est), "fit did not set result_."
    if not ctx.tags["non_deterministic"]:
        first = _result_arrays(est)
        other = ctx.make().fit(ctx.train)
        _assert_bitwise(first, _result_arrays(other), "two fits of fresh estimators")
        est.fit(ctx.train)
        _assert_bitwise(first, _result_arrays(est), "refitting the same estimator")
    ctx.fitted = est


def _check_fit_attributes(ctx: _Context) -> None:
    est = ctx.make()
    before = set(vars(est))
    params = copy.deepcopy(est.get_params(deep=False))
    data = copy.deepcopy(ctx.train)
    est.fit(ctx.train)
    after = set(vars(est))
    removed = sorted(before - after)
    assert not removed, f"fit removed attributes {removed}."
    bad = sorted(k for k in after - before if not k.endswith("_"))
    assert not bad, f"fit added attributes without a trailing underscore: {bad}."
    diff = _diff_keys(est.get_params(deep=False), params)
    assert not diff, f"fit changed hyperparameters {diff}."
    assert ctx.train.equals(data), "fit mutated the input PanelData."

    required = ["result_", "n_features_in_", "feature_names_in_", "exog_names_in_", "index_"]
    if ctx.tags["backend"] == "torch":
        required += ["network_", "device_"]
    missing = [k for k in required if k not in vars(est)]
    assert not missing, f"fit did not set {missing}."
    assert est.n_features_in_ == len(ctx.train.series_names)
    assert list(est.feature_names_in_) == list(ctx.train.series_names), "feature_names_in_ is not in the user's order."
    assert list(est.exog_names_in_) == list(ctx.train.exog_names), "exog_names_in_ does not match the data."
    assert est.index_.equals(ctx.train.index), "index_ does not match the training index."
    if ctx.tags["backend"] == "torch":
        assert isinstance(est.network_, torch.nn.Module), "network_ is not a torch.nn.Module."

    y = ctx.train.y
    if not ctx.train.exog_names and not ctx.train.series_exog_names and ctx.train.freq is None:
        snapshot = y.copy(deep=True)
        ctx.make().fit(y)
        assert y.equals(snapshot) and y.columns.equals(snapshot.columns), "fit mutated an input DataFrame."


def _assert_time_frame(frame: Any, est: BaseFactorModel, origin_index: pd.Index, h: int, kind: str) -> None:
    what = {"forecast": "predict", "factors": "predict_factors"}[kind]
    assert isinstance(frame, pd.DataFrame), f"{what} returned {type(frame).__name__}, expected a DataFrame."
    assert len(frame) == h, f"{what}({h}) returned {len(frame)} rows."
    if kind == "forecast":
        assert list(frame.columns) == list(est.feature_names_in_), (
            f"{what} columns {list(frame.columns)} != feature_names_in_ {list(est.feature_names_in_)}."
        )
    assert all(dtype == np.float64 for dtype in frame.dtypes), f"{what} values are not all float64."
    assert np.isfinite(frame.to_numpy()).all(), f"{what} returned non-finite values."
    expected = _future_index(origin_index, h)
    if expected is not None:
        assert frame.index.equals(expected), f"{what} index {list(frame.index)} != expected {list(expected)}."
    else:
        assert frame.index.is_monotonic_increasing and frame.index[0] > origin_index[-1], (
            f"{what} index does not start after the conditioning data."
        )
    expected_name = est.index_.name if est.index_.name is not None else "time"
    assert frame.index.name == expected_name, f"{what} index name {frame.index.name!r} != {expected_name!r}."
    assert frame.attrs.get("kind") == kind, f"{what} attrs['kind'] is {frame.attrs.get('kind')!r}, expected {kind!r}."
    assert frame.attrs.get("model") == type(est).__name__, f"{what} attrs['model'] is {frame.attrs.get('model')!r}."
    assert frame.attrs.get("origin") == origin_index[-1], (
        f"{what} attrs['origin'] is {frame.attrs.get('origin')!r}, expected {origin_index[-1]!r}."
    )


def _assert_weights_frame(frame: Any, est: BaseFactorModel, data: PanelData, hist_len: int) -> None:
    assert isinstance(frame, pd.DataFrame), f"allocate returned {type(frame).__name__}, expected a DataFrame."
    expected = data.index[hist_len:]
    assert frame.index.equals(expected), f"allocate index has {len(frame)} rows, expected data.index[{hist_len}:]."
    assert list(frame.columns) == list(est.feature_names_in_), "allocate columns != feature_names_in_."
    assert all(dtype == np.float64 for dtype in frame.dtypes), "allocate values are not all float64."
    assert np.isfinite(frame.to_numpy()).all(), "allocate returned non-finite weights."
    assert frame.attrs.get("kind") == "weights", f"allocate attrs['kind'] is {frame.attrs.get('kind')!r}."
    assert frame.attrs.get("model") == type(est).__name__, f"allocate attrs['model'] is {frame.attrs.get('model')!r}."


def _check_output_shape(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    tags = ctx.tags
    if tags["task"] == "forecast":
        _assert_time_frame(est.predict(_H), est, ctx.train.index, _H, "forecast")
        cond = ctx.conditioning_data()
        if cond is not None:
            _assert_time_frame(est.predict(_H, data=cond), est, ctx.data.index, _H, "forecast")
        for bad_h in (1.5, "3", None, True):
            _expect_raises(TypeError, lambda h=bad_h: est.predict(h), f"predict({bad_h!r})")
        if tags["capability:factor_forecast"]:
            _assert_time_frame(est.predict_factors(_H), est, ctx.train.index, _H, "factors")
            if cond is not None:
                _assert_time_frame(est.predict_factors(_H, data=cond), est, ctx.data.index, _H, "factors")
        _expect_raises(TypeError, lambda: est.allocate(), "allocate on a forecasting model")
    else:
        _assert_weights_frame(est.allocate(), est, ctx.train, ctx.hist_len)
        _assert_weights_frame(est.allocate(data=ctx.data), est, ctx.data, ctx.hist_len)
        _expect_raises(TypeError, lambda: est.predict(_H), "predict on an allocation model")
    if not tags["capability:factor_forecast"]:
        _expect_raises(UnsupportedInputError, lambda: est.predict_factors(_H), "predict_factors without the capability")


def _check_purity(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    tags = ctx.tags
    before = _state_hash(est)
    data = copy.deepcopy(ctx.data)
    outputs = []
    if tags["task"] == "forecast":
        cond = ctx.conditioning_data()
        outputs += [est.predict(_H), est.predict(_H)]
        if cond is not None:
            est.predict(_H, data=cond)
        if tags["capability:factor_forecast"]:
            est.predict_factors(_H)
            if cond is not None:
                est.predict_factors(_H, data=cond)
    else:
        outputs += [est.allocate(), est.allocate()]
        est.allocate(data=ctx.data)
    assert _state_hash(est) == before, "predict/allocate changed the fitted state."
    assert ctx.data.equals(data), "predict/allocate mutated the input PanelData."
    if not tags["non_deterministic"]:
        assert outputs[0].equals(outputs[1]), "two identical predict/allocate calls returned different values."


def _with_exog(data: PanelData) -> PanelData:
    exog = pd.DataFrame({"x_unsupported": np.linspace(0.0, 1.0, data.n_timesteps)}, index=data.index)
    return PanelData(data.y, exog=exog, series_exog=data.series_exog, freq=data.freq)


def _with_series_exog(data: PanelData) -> PanelData:
    columns = pd.MultiIndex.from_product([["c_unsupported"], list(data.series_names)], names=["feature", "series"])
    values = np.ones((data.n_timesteps, len(data.series_names)))
    return PanelData(data.y, exog=data.exog, series_exog=pd.DataFrame(values, index=data.index, columns=columns), freq=data.freq)


def _with_mixed_frequency(data: PanelData) -> Optional[PanelData]:
    names = list(data.series_names)
    if len(names) < 2:
        return None
    return PanelData(data.y, exog=data.exog, series_exog=data.series_exog, freq={names[0]: "q", names[1]: "m"})


def _with_missing_values(data: PanelData) -> PanelData:
    y = data.y
    y.iloc[min(1, len(y) - 1), 0] = np.nan
    return PanelData(y, exog=data.exog, series_exog=data.series_exog, freq=data.freq)


_EXCLUDABLE = (
    ("capability:exog", "exog", _with_exog),
    ("capability:series_exog", "series_exog", _with_series_exog),
    ("capability:mixed_frequency", "mixed-frequency data", _with_mixed_frequency),
    ("capability:missing_values", "missing values", _with_missing_values),
)


def _check_capabilities(ctx: _Context) -> None:
    tags = ctx.tags
    for tag, what, build in _EXCLUDABLE:
        if tags[tag]:
            continue
        bad = build(ctx.train)
        if bad is None:
            continue
        est = ctx.make()
        err = _expect_raises(UnsupportedInputError, lambda est=est, bad=bad: est.fit(bad), f"fit with {what} while {tag} is False")
        assert tag in str(err), f"the error for {what} does not name the tag {tag}: {err}"
        assert "result_" not in vars(est), f"fit with {what} left the estimator fitted."

        fitted = ctx.fitted_copy()
        calls = []
        cond = ctx.conditioning_data() if tags["task"] == "forecast" else ctx.data
        if cond is not None:
            bad_cond = build(cond)
            if tags["task"] == "forecast":
                calls.append(("predict", lambda f=fitted, b=bad_cond: f.predict(_H, data=b)))
            else:
                calls.append(("allocate", lambda f=fitted, b=bad_cond: f.allocate(data=b)))
        if tags["capability:update"]:
            bad_update = build(ctx.update_data())
            calls.append(("update", lambda f=fitted, b=bad_update: f.update(b)))
        for method, call in calls:
            err = _expect_raises(UnsupportedInputError, call, f"{method} with {what} while {tag} is False")
            assert tag in str(err), f"the {method} error for {what} does not name the tag {tag}: {err}"

    if tags["capability:update"]:
        est = ctx.fitted_copy()
        try:
            returned = est.update(ctx.update_data(), refit=True)
        except UnsupportedInputError as err:
            assert "refit" in str(err), f"the update(refit=True) error does not mention refit: {err}"
        else:
            assert returned is est, "update(refit=True) must return self."


def _check_device(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    assert "device" in est.get_params(deep=False), "capability:device requires a device parameter."
    assert str(est.device_) == "cpu", f"device='cpu' gave device_={est.device_!r}."
    network = est.network_
    assert isinstance(network, torch.nn.Module), "network_ is not a torch.nn.Module."
    off_cpu = [
        name
        for name, tensor in [*network.named_parameters(), *network.named_buffers()]
        if tensor.device.type != "cpu"
    ]
    assert not off_cpu, f"device='cpu' left {off_cpu} off the CPU."


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
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(saved_level)
    assert not out.getvalue(), f"fit with verbose=0 printed to stdout: {out.getvalue()[:200]!r}"
    assert not err.getvalue(), f"fit with verbose=0 printed to stderr: {err.getvalue()[:200]!r}"
    logged = [f"{r.name}:{r.levelname}:{r.getMessage()}" for r in handler.records if r.levelno >= logging.INFO]
    assert not logged, f"fit with verbose=0 logged at INFO or above: {logged[:3]}"


def _check_update(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    data = ctx.update_data()
    network = _network_hash(est)
    params = copy.deepcopy(est.get_params(deep=False))
    returned = est.update(data)
    assert returned is est, "update must return self."
    assert _network_hash(est) == network, "update(refit=False) changed the network parameters."
    diff = _diff_keys(est.get_params(deep=False), params)
    assert not diff, f"update changed hyperparameters {diff}."
    if ctx.tags["task"] == "forecast":
        _assert_time_frame(est.predict(_H), est, ctx.data.index, _H, "forecast")
    else:
        _assert_weights_frame(est.allocate(), est, data, ctx.hist_len)


def _assert_same_frame(a: pd.DataFrame, b: Any, what: str) -> None:
    assert isinstance(b, pd.DataFrame), f"{what} returned {type(b).__name__}, expected a DataFrame."
    assert type(a.index) is type(b.index) and a.index.equals(b.index) and a.index.name == b.index.name, (
        f"{what}: the index differs from the original model's."
    )
    assert list(a.columns) == list(b.columns), f"{what}: columns {list(b.columns)} != {list(a.columns)}."
    assert list(a.dtypes) == list(b.dtypes), f"{what}: dtypes differ."
    assert a.to_numpy().tobytes() == b.to_numpy().tobytes(), f"{what}: values are not bitwise identical."
    assert _deep_equal(dict(a.attrs), dict(b.attrs)), f"{what}: attrs {b.attrs!r} != {a.attrs!r}."


def _object_hash(obj: Any) -> str:
    h = hashlib.sha256()
    _feed(h, obj, set())
    return h.hexdigest()


def _assert_reloaded(est: BaseFactorModel, loaded: Any, ctx: _Context, what: str) -> None:
    tags = ctx.tags
    assert type(loaded) is type(est), f"{what} returned {type(loaded).__name__}, expected {type(est).__name__}."
    diff = _diff_keys(loaded.get_params(deep=False), est.get_params(deep=False))
    assert not diff, f"{what} changed parameters {diff}."
    for name in ("n_features_in_", "feature_names_in_", "exog_names_in_", "series_exog_names_in_"):
        if name in vars(est):
            assert name in vars(loaded), f"{what} did not restore {name}."
            assert _deep_equal(vars(est)[name], vars(loaded)[name]), f"{what} restored a different {name}."
    index, other = est.index_, loaded.index_
    assert type(other) is type(index) and other.equals(index) and other.name == index.name, (
        f"{what} restored a different index_."
    )
    assert getattr(other, "freq", None) == getattr(index, "freq", None), f"{what} lost the frequency of index_."
    _assert_bitwise(_result_arrays(est), _result_arrays(loaded), f"{what}: result_")
    if "scaler_" in vars(est):
        assert "scaler_" in vars(loaded), f"{what} did not restore scaler_."
        assert _object_hash(est.scaler_) == _object_hash(loaded.scaler_), f"{what} restored a different scaler_."
    if tags["backend"] == "torch":
        assert isinstance(loaded.network_, torch.nn.Module), f"{what}: network_ is not a torch.nn.Module."
        assert _network_hash(loaded) == _network_hash(est), f"{what}: network_ weights are not bitwise identical."
        if tags["capability:device"]:
            assert str(loaded.device_) == "cpu", f"{what}(device='cpu') gave device_={loaded.device_!r}."
            off_cpu = [n for n, t in [*loaded.network_.named_parameters(), *loaded.network_.named_buffers()]
                       if t.device.type != "cpu"]
            assert not off_cpu, f"{what}(device='cpu') left {off_cpu} off the CPU."
    if tags["task"] == "forecast":
        _assert_same_frame(est.predict(_H), loaded.predict(_H), f"predict after {what}")
        cond = ctx.conditioning_data()
        if cond is not None:
            _assert_same_frame(est.predict(_H, data=cond), loaded.predict(_H, data=cond), f"predict(data=) after {what}")
        if tags["capability:factor_forecast"]:
            _assert_same_frame(est.predict_factors(_H), loaded.predict_factors(_H), f"predict_factors after {what}")
    else:
        _assert_same_frame(est.allocate(data=ctx.data), loaded.allocate(data=ctx.data), f"allocate(data=) after {what}")


def _save_quietly(est: BaseFactorModel, path: Path, **kwargs: Any) -> Any:
    """``est.save`` without the random_state=None warning (the checks run with any parameters)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*random_state=None.*", category=UserWarning)
        return est.save(path, **kwargs)


def _check_save_load(ctx: _Context) -> None:
    est = ctx.fitted_copy()
    tags = ctx.tags
    device = "cpu" if tags["capability:device"] else None
    before = _state_hash(est)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "run"
        returned = _save_quietly(est, path)
        assert returned is None, f"save returned {type(returned).__name__}, expected None."
        assert _state_hash(est) == before, "save changed the fitted state."
        debris = sorted(p.name for p in root.iterdir() if p.name != "run")
        assert not debris, f"save left {debris} next to the run."

        files = {p.name for p in path.iterdir()}
        missing = sorted(set(REQUIRED_FILES) - files)
        assert not missing, f"save did not write {missing}."
        if tags["backend"] == "torch":
            assert WEIGHTS_FILE in files, f"a torch model must write {WEIGHTS_FILE}."
        else:
            assert WEIGHTS_FILE not in files, f"a {tags['backend']} model must not write {WEIGHTS_FILE}."
        extra = sorted(files - set(REQUIRED_FILES) - {WEIGHTS_FILE, SCALER_JSON, SCALER_PKL})
        assert not extra, f"save wrote files outside the layout: {extra}."
        if "scaler_" in vars(est):
            assert (SCALER_JSON in files) != (SCALER_PKL in files), "save must write exactly one scaler file."
            if is_known_scaler(est.scaler_):
                assert SCALER_PKL not in files, f"save pickled a {type(est.scaler_).__name__}, a known scaler."
        else:
            assert SCALER_JSON not in files and SCALER_PKL not in files, "save wrote a scaler file without scaler_."
        pickled = SCALER_PKL in files

        config = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
        assert config.get("format_version") == FORMAT_VERSION, f"config.json format_version {config.get('format_version')!r}."
        assert config.get("model_type") == ctx.cls.__name__, f"config.json model_type {config.get('model_type')!r}."
        params = est.get_params(deep=False)
        assert set(config.get("params", {})) == set(params), "config.json params do not match get_params()."
        assert config.get("tags") == json.loads(json.dumps(est.get_tags())), "config.json tags do not match get_tags()."
        for key in ("library_version", "runtime"):
            assert key in config, f"config.json has no {key!r}."

        _expect_raises(FileExistsError, lambda: est.save(path), "save to an existing directory")

        if pickled:
            err = _expect_raises(
                ConfigurationError, lambda: ctx.cls.load(path, device=device), "load of a pickled scaler without allow_pickle"
            )
            assert "allow_pickle" in str(err), f"the pickle error does not name allow_pickle: {err}"
        loaded = ctx.cls.load(path, device=device, allow_pickle=pickled)
        _assert_reloaded(est, loaded, ctx, f"{ctx.cls.__name__}.load")
        assert _state_hash(est) == before, "predict/allocate on the original changed it during the save/load check."

        _expect_raises(ConfigurationError, lambda: BaseFactorModel.load(path), "BaseFactorModel.load on another model's run")
        registered = dict(all_estimators())
        if registered.get(ctx.cls.__name__) is ctx.cls:
            _assert_reloaded(est, _load_any(path, device=device, allow_pickle=pickled), ctx, "dfm_python.load")
        else:
            err = _expect_raises(
                ConfigurationError,
                lambda: _load_any(path, device=device, allow_pickle=pickled),
                "dfm_python.load of an unregistered estimator",
            )
            assert ctx.cls.__name__ in str(err), f"the dfm_python.load error does not name the model type: {err}"

        _save_quietly(est, path, overwrite=True)
        assert {p.name for p in path.iterdir()} == files, "save(overwrite=True) wrote a different file set."
        debris = sorted(p.name for p in root.iterdir() if p.name != "run")
        assert not debris, f"save(overwrite=True) left {debris} next to the run."

        if tags["capability:device"]:
            _check_cuda_saved_loads_on_cpu(est, path, root, ctx)


def _assert_on_cpu(loaded: BaseFactorModel, what: str) -> None:
    assert str(loaded.device_) == "cpu", f"{what} gave device_={loaded.device_!r}."
    off_cpu = [n for n, t in [*loaded.network_.named_parameters(), *loaded.network_.named_buffers()]
               if t.device.type != "cpu"]
    assert not off_cpu, f"{what} left {off_cpu} off the CPU."


def _check_cuda_saved_loads_on_cpu(est: BaseFactorModel, path: Path, root: Path, ctx: _Context) -> None:
    """A CUDA-saved run loads on CPU (spec section 7).

    Always: the saved run with its ``device`` parameter rewritten to ``"cuda"``
    loads where CUDA is unavailable (simulated) on CPU with a ``UserWarning``,
    and an explicit ``load(device="cuda")`` there raises ``ConfigurationError``.
    With CUDA: a model fitted on CUDA loads with ``device="cpu"``.
    """
    cls = ctx.cls
    simulated = root / "cuda_saved"
    shutil.copytree(path, simulated)
    config_path = simulated / CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["params"]["device"] = "cuda"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    pickled = (simulated / SCALER_PKL).is_file()
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = cls.load(simulated, allow_pickle=pickled)
        assert any(issubclass(w.category, UserWarning) and "CUDA is not available" in str(w.message)
                   for w in caught), "loading a CUDA-saved run without CUDA did not warn that it falls back to CPU."
        _assert_on_cpu(loaded, f"{cls.__name__}.load of a CUDA-saved run without CUDA")
        assert _network_hash(loaded) == _network_hash(est), (
            f"{cls.__name__}.load of a CUDA-saved run without CUDA: network_ weights are not bitwise identical."
        )
        err = _expect_raises(ConfigurationError, lambda: cls.load(simulated, device="cuda", allow_pickle=pickled),
                             "load(device='cuda') without CUDA")
        assert "device='cpu'" in str(err), f"the load(device='cuda') error does not suggest device='cpu': {err}"
    shutil.rmtree(simulated)

    if not torch.cuda.is_available():
        return  # the real CUDA round trip is skipped without CUDA
    on_cuda = ctx.make(cpu=False).set_params(device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on_cuda.fit(ctx.train)
    cuda_path = root / "cuda_run"
    _save_quietly(on_cuda, cuda_path)
    loaded = cls.load(cuda_path, device="cpu", allow_pickle=(cuda_path / SCALER_PKL).is_file())
    _assert_on_cpu(loaded, f"{cls.__name__}.load(device='cpu') of a CUDA-fitted run")
    assert _network_hash(loaded) == _network_hash(on_cuda), (
        f"{cls.__name__}.load(device='cpu') of a CUDA-fitted run: network_ weights are not bitwise identical."
    )
    shutil.rmtree(cuda_path)


def _always(tags: dict) -> bool:
    return True


_CHECKS: tuple[tuple[str, Callable[[_Context], None], Callable[[dict], bool]], ...] = (
    ("tags", _check_tags, _always),
    ("pure_construction", _check_pure_construction, _always),
    ("parameters", _check_parameters, _always),
    ("config_mirror", _check_config_mirror, _always),
    ("config_serialization", _check_config_serialization, _always),
    ("unknown_keys", _check_unknown_keys, _always),
    ("unfitted", _check_unfitted, _always),
    ("fit", _check_fit, _always),
    ("fit_attributes", _check_fit_attributes, _always),
    ("output_shape", _check_output_shape, _always),
    ("purity", _check_purity, _always),
    ("capabilities", _check_capabilities, _always),
    ("device", _check_device, lambda tags: tags["capability:device"]),
    ("silence", _check_silence, _always),
    ("update", _check_update, lambda tags: tags["capability:update"]),
    ("save_load", _check_save_load, _always),
)


def _instance_tags(cls: type, params: Mapping, class_tags: dict) -> dict:
    """Tags of an estimator built from ``params`` (some tags depend on hyperparameters)."""
    try:
        return cls(**copy.deepcopy(dict(params))).get_tags()
    except Exception:  # a failing constructor or tag is reported by the checks themselves
        return class_tags


def check_estimator(estimator_or_class: Any, *, data: Any = None) -> list[str]:
    """Run the contract checks that apply to an estimator's tags.

    Parameters
    ----------
    estimator_or_class : BaseFactorModel subclass or instance
        A class is checked once per parameter set from its
        ``get_test_params()``; an instance is checked with its own parameters.
    data : PanelData, DataFrame or 2-D array, optional
        The fixture; defaults to ``make_panel(tags)``. The checks fit on all
        but the last few rows and use those rows as new data.

    Returns
    -------
    list of str
        Names of the checks that ran, in order.

    Raises
    ------
    AssertionError
        On the first failing check; the message names the check and the
        parameter set.
    """
    if isinstance(estimator_or_class, type):
        cls = estimator_or_class
        params_list = None
    else:
        cls = type(estimator_or_class)
        params_list = [estimator_or_class.get_params(deep=False)]
    if not issubclass(cls, BaseFactorModel):
        raise TypeError(f"check_estimator expects a dfm_python.base.BaseFactorModel, got {cls.__name__}.")
    if params_list is None:
        params_list = cls.get_test_params()
    class_tags = cls.get_tags()
    if class_tags["backend"] == "torch":
        require_torch("check_estimator")
    panel = make_panel(class_tags) if data is None else PanelData._coerce(data)
    train, new = _split(panel)

    ran: list[str] = []
    for params in params_list:
        label = f"{cls.__name__}({', '.join(f'{k}={v!r}' for k, v in params.items())})"
        tags = _instance_tags(cls, params, class_tags)
        ctx = _Context(cls=cls, params=dict(params), data=panel, train=train, new=new, tags=tags, label=label)
        for name, check, applies in _CHECKS:
            if not applies(tags):
                continue
            try:
                check(ctx)
            except Exception as err:  # harness boundary: any failure is reported with the check and parameters
                detail = str(err) if isinstance(err, AssertionError) else f"{type(err).__name__}: {err}"
                raise AssertionError(f"check_estimator: check {name!r} failed for {label}: {detail}") from err
            if name not in ran:
                ran.append(name)
    return ran

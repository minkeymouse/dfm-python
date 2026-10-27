"""Pickle-free codecs for the files of a save directory.

- :func:`write_json` / :func:`read_json`: sorted keys; numpy scalars and
  arrays become plain Python values; scikit-learn estimators are written as
  ``{"class": "module.QualName", "params": get_params(deep=False)}`` and
  rebuilt on reading only for the public estimator classes of
  ``sklearn.preprocessing`` (any other class only when the caller explicitly
  allows importing it). This is the format-1 encoding.
- :func:`encode_params` / :func:`decode_params`: the ``params`` of a format-2
  ``config.json``. A component spec (an AFM ``AttentionFactors`` ...) is
  written as ``{"spec": "AttentionFactors", "params": {...}}`` and rebuilt
  through an allow-list: only the classes registered with
  :func:`register_spec`. Any other estimator is refused when saving and any
  other spec name when loading, so loading imports nothing named in a file.
- :func:`write_npz` / :func:`read_npz`: a ``.npz`` archive with namespaced
  keys such as ``"ssm/A"``, written and read without pickle (object arrays are
  refused).
- :func:`write_weights` / :func:`read_weights`: a torch ``state_dict`` saved
  with :func:`torch.save` and read with ``weights_only=True``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np

from ..utils.errors import ConfigurationError

__all__ = [
    "register_spec",
    "encode_params",
    "decode_params",
    "read_json_plain",
    "class_path",
    "to_jsonable",
    "from_jsonable",
    "is_encoded_estimator",
    "write_json",
    "read_json",
    "write_npz",
    "read_npz",
    "write_weights",
    "read_weights",
]

PathLike = Union[str, "os.PathLike[str]"]

#: Estimator classes under this package are rebuilt from JSON without opt-in.
SAFE_ESTIMATOR_PACKAGE = "sklearn.preprocessing"
_ESTIMATOR_KEYS = frozenset({"class", "params"})


# JSON


def class_path(cls: type) -> str:
    """``"module.QualName"`` of a class (the form written to JSON)."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_estimator(obj: Any) -> bool:
    from sklearn.base import BaseEstimator

    return isinstance(obj, BaseEstimator)


def _is_torch_device(obj: Any) -> bool:
    # torch is not imported here: an object can only be a torch.device once torch is loaded.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, torch.device)


def to_jsonable(obj: Any, *, where: str = "value") -> Any:
    """Convert ``obj`` to JSON-safe builtins, or raise ``TypeError`` naming ``where``.

    numpy scalars become Python scalars, arrays become (nested) lists, tuples
    become lists, a ``torch.device`` becomes its string (``"cuda:0"``) and
    scikit-learn estimators become ``{"class", "params"}``. Mapping keys must
    be strings.
    """
    if isinstance(obj, np.generic):  # before the builtins: np.float64 subclasses float
        if obj.dtype.kind in "biuf" or obj.dtype.kind == "U":
            return obj.item()
        raise TypeError(f"{where}: cannot write a numpy {obj.dtype} scalar to JSON.")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biufU":
            return obj.tolist()
        if obj.dtype.hasobject:
            return to_jsonable(obj.tolist(), where=where)
        raise TypeError(f"{where}: cannot write a numpy array of dtype {obj.dtype} to JSON.")
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, where=f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: JSON needs string keys, got {key!r}.")
            out[key] = to_jsonable(value, where=f"{where}[{key!r}]")
        return out
    if _is_torch_device(obj):
        return str(obj)
    if _is_estimator(obj):
        return {
            "class": class_path(type(obj)),
            "params": to_jsonable(obj.get_params(deep=False), where=f"{where}.params"),
        }
    raise TypeError(f"{where}: cannot write a {type(obj).__name__} to JSON.")


def _import_class(name: str) -> Any:
    """Import ``"package.module.Outer.Inner"`` by trying module prefixes from the longest."""
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            break
        return obj
    raise ConfigurationError(f"Cannot import the estimator class {name!r} named in the saved run.")


def _tuple_params(cls: type, params: dict) -> dict:
    """Turn lists back into tuples where the constructor default is a tuple (JSON has no tuples)."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return params
    out = dict(params)
    for name, value in params.items():
        param = signature.parameters.get(name)
        if param is not None and isinstance(param.default, tuple) and isinstance(value, list):
            out[name] = tuple(value)
    return out


def _safe_preprocessing_class(name: str) -> Any:
    """Resolve ``name`` against the public names of ``sklearn.preprocessing`` without importing anything else.

    Accepts ``"sklearn.preprocessing.<Name>"`` and the private-module form
    :func:`class_path` writes (``"sklearn.preprocessing._data.<Name>"``), where
    ``<Name>`` is listed in ``sklearn.preprocessing.__all__``. Any other path
    under the package (for example its ``tests`` subpackage) returns None, so
    that no submodule named in a file is ever imported without opt-in.
    """
    prefix = SAFE_ESTIMATOR_PACKAGE + "."
    if not name.startswith(prefix):
        return None
    *middle, attr = name[len(prefix):].split(".")
    if not attr.isidentifier() or any(not (part.isidentifier() and part.startswith("_")) for part in middle):
        return None
    import sklearn.preprocessing

    if attr not in getattr(sklearn.preprocessing, "__all__", ()):
        return None
    return getattr(sklearn.preprocessing, attr, None)


def _rebuild_estimator(name: str, params: Any, allow_import: bool) -> Any:
    if allow_import:
        cls = _import_class(name)
    else:
        cls = _safe_preprocessing_class(name)
        if cls is None:
            return {"class": name, "params": params}
    from sklearn.base import BaseEstimator

    if not (isinstance(cls, type) and issubclass(cls, BaseEstimator)):
        raise ConfigurationError(f"{name!r} in the saved run is not a scikit-learn estimator class.")
    if not allow_import and not cls.__module__.startswith(SAFE_ESTIMATOR_PACKAGE):
        return {"class": name, "params": params}
    if not isinstance(params, Mapping):
        raise ConfigurationError(f"The saved parameters of {name!r} are not a mapping.")
    return cls(**_tuple_params(cls, dict(params)))


def is_encoded_estimator(obj: Any) -> bool:
    """True for an estimator that :func:`from_jsonable` left as ``{"class", "params"}``."""
    return isinstance(obj, dict) and set(obj) == _ESTIMATOR_KEYS and isinstance(obj["class"], str)


def from_jsonable(obj: Any, *, allow_import: bool = False) -> Any:
    """Inverse of :func:`to_jsonable` for estimators: rebuild ``{"class", "params"}`` dicts.

    The public estimator classes of ``sklearn.preprocessing`` (the names in
    its ``__all__``) are always rebuilt; they are looked up on that package,
    so no module named in the file is imported. Any other class is imported
    and rebuilt only with ``allow_import=True`` (importing a class named in a
    file can run arbitrary code); otherwise the dict is returned as is (see
    :func:`is_encoded_estimator`).
    """
    if isinstance(obj, list):
        return [from_jsonable(v, allow_import=allow_import) for v in obj]
    if isinstance(obj, dict):
        decoded = {k: from_jsonable(v, allow_import=allow_import) for k, v in obj.items()}
        if is_encoded_estimator(decoded):
            return _rebuild_estimator(decoded["class"], decoded["params"], allow_import)
        return decoded
    return obj


def write_json(path: PathLike, obj: Any) -> None:
    """Write ``obj`` as indented JSON with sorted keys (see :func:`to_jsonable`).

    Non-finite floats are written as ``NaN`` / ``Infinity``, which
    :func:`read_json` (Python's ``json``) reads back.
    """
    text = json.dumps(to_jsonable(obj, where=Path(path).name), sort_keys=True, indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def read_json(path: PathLike, *, allow_import: bool = False) -> Any:
    """Read a JSON file written by :func:`write_json` and rebuild estimators (see :func:`from_jsonable`)."""
    with open(path, encoding="utf-8") as fh:
        return from_jsonable(json.load(fh), allow_import=allow_import)


def read_json_plain(path: PathLike) -> Any:
    """Read a JSON file as plain values (no estimator is rebuilt)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# Component specs in format-2 params

_SPEC_KEYS = frozenset({"spec", "params"})
#: The allow-list: spec class name -> class (filled by :func:`register_spec`).
_SPECS: dict = {}


def register_spec(cls: type) -> type:
    """Allow the component spec ``cls`` in saved params; returns ``cls`` (usable as a class decorator).

    A spec is a :class:`sklearn.base.BaseEstimator` whose ``get_params``
    hold only JSON-safe values (or other specs). It is written and read back
    by its class name.

    Raises
    ------
    TypeError
        If ``cls`` is not a ``BaseEstimator`` subclass.
    ValueError
        If another class is registered under the same name.
    """
    from sklearn.base import BaseEstimator

    if not (isinstance(cls, type) and issubclass(cls, BaseEstimator)):
        raise TypeError(f"register_spec expects a sklearn BaseEstimator subclass, got {cls!r}.")
    existing = _SPECS.get(cls.__name__)
    if existing is not None and existing is not cls:
        raise ValueError(f"Another spec class is already registered as {cls.__name__!r}: {class_path(existing)}.")
    _SPECS[cls.__name__] = cls
    return cls


def _encode_value(value: Any, where: str) -> Any:
    spec = _SPECS.get(type(value).__name__)
    if spec is not None and type(value) is spec:
        return {"spec": spec.__name__, "params": encode_params(value.get_params(deep=False), where=f"{where}.params")}
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: JSON needs string keys, got {key!r}.")
            out[key] = _encode_value(item, f"{where}[{key!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if _is_estimator(value):
        raise TypeError(
            f"{where}: {class_path(type(value))} is not an allowed spec class, so it cannot be saved. Saved runs "
            "hold only plain values and the registered component specs."
        )
    return to_jsonable(value, where=where)


def encode_params(params: Mapping[str, Any], *, where: str = "params") -> dict:
    """``get_params(deep=False)`` as JSON-safe values; a registered spec becomes ``{"spec", "params"}``.

    Raises
    ------
    TypeError
        For a value JSON cannot hold, or an estimator that is not a registered spec.
    """
    return {str(key): _encode_value(value, f"{where}[{key!r}]") for key, value in params.items()}


def _is_encoded_spec(obj: Any) -> bool:
    return isinstance(obj, dict) and set(obj) == _SPEC_KEYS and isinstance(obj["spec"], str)


def decode_params(obj: Any, cls: Any = None) -> Any:
    """Inverse of :func:`encode_params`: rebuild the registered specs; lists become tuples where ``cls`` defaults are tuples.

    Parameters
    ----------
    obj : dict
        The ``params`` of a format-2 ``config.json``.
    cls : type, optional
        The class the params belong to; a list is turned back into a tuple
        where its constructor default is a tuple (JSON has no tuples).

    Raises
    ------
    ConfigurationError
        For a spec name that is not registered, or spec params that are not a mapping.
    """
    if isinstance(obj, list):
        return [decode_params(item) for item in obj]
    if not isinstance(obj, dict):
        return obj
    if _is_encoded_spec(obj):
        spec = _SPECS.get(obj["spec"])
        if spec is None:
            allowed = ", ".join(sorted(_SPECS)) or "none"
            raise ConfigurationError(
                f"The saved run names the spec {obj['spec']!r}, which is not an allowed spec class (allowed: {allowed})."
            )
        if not isinstance(obj["params"], dict):
            raise ConfigurationError(f"The saved parameters of the spec {obj['spec']!r} are not a mapping.")
        return spec(**decode_params(obj["params"], spec))
    decoded = {key: decode_params(value) for key, value in obj.items()}
    return decoded if cls is None else _tuple_params(cls, decoded)


# npz


def write_npz(path: PathLike, arrays: Mapping[str, Any]) -> None:
    """Write arrays to an uncompressed ``.npz`` archive without pickle.

    Keys are namespaced names such as ``"ssm/A"`` or ``"scaler/mean_"``. This is
    the archive :func:`numpy.savez` writes (one ``<key>.npy`` member per array),
    built directly so that any key is allowed.

    Raises
    ------
    TypeError
        If a key is not a non-empty string or an array has an object dtype
        (object arrays need pickle).
    """
    checked = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key or key.startswith("/"):
            raise TypeError(f"npz keys must be non-empty relative names such as 'ssm/A', got {key!r}.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"Array {key!r} has dtype object, which cannot be saved without pickle. "
                "Store numbers in a numeric array and names or other values in JSON."
            )
        checked[key] = array
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, array in checked.items():
            with zf.open(key + ".npy", mode="w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, array, allow_pickle=False)


def read_npz(path: PathLike) -> dict:
    """Read every array of a ``.npz`` archive with ``allow_pickle=False``, in the order written."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# torch weights


def write_weights(path: PathLike, state_dict: Mapping[str, Any]) -> None:
    """Save a torch ``state_dict`` (tensors only) with :func:`torch.save`."""
    import torch

    torch.save(dict(state_dict), path)


def read_weights(path: PathLike, map_location: Any = None) -> dict:
    """Load a ``state_dict`` with ``torch.load(path, map_location=..., weights_only=True)``."""
    import torch

    return torch.load(path, map_location=map_location, weights_only=True)

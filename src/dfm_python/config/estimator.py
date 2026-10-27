"""``EstimatorConfig``: the typed, serializable mirror of an estimator's constructor.

Every public model is configured through keyword-only constructor arguments.
Its ``*Config`` dataclass subclasses :class:`EstimatorConfig` and has exactly
the constructor's fields and defaults, in the role of Hugging Face's
``PretrainedConfig``::

    @dataclass
    class MyConfig(EstimatorConfig):
        n_factors: int = 2
        tol: float = 1e-5

    cfg = MyConfig.from_dict({"n_factors": 3})
    cfg.to_yaml("run.yaml")               # writes {"model": {...}}
    MyConfig.from_yaml("run.yaml") == cfg

Unknown keys raise :class:`~dfm_python.utils.errors.ConfigurationError`, which
names the closest field. There is no alias table.

A YAML file has a ``model:`` section (constructor parameters) and an optional
``data:`` section (the arguments of :meth:`dfm_python.PanelData.from_wide`).
:func:`load_yaml` returns both as plain dicts. Reading or writing YAML needs
the ``yaml`` extra (omegaconf) or PyYAML; both are imported lazily.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
import re
import sys
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np

__all__ = ["EstimatorConfig", "load_yaml"]

PathLike = Union[str, "os.PathLike[str]"]

# Top-level sections of a YAML file.
_YAML_SECTIONS = ("model", "data")
_YAML_EXTRA_HINT = "Install the 'yaml' extra: pip install 'dfm-python[yaml]'."


def _configuration_error(message: str) -> Exception:
    # Lazy import: dfm_python.utils imports dfm_python.config.
    from ..utils.errors import ConfigurationError

    return ConfigurationError(message)


def _closest(key: Any, candidates: typing.Iterable[str]) -> str | None:
    matches = difflib.get_close_matches(str(key), list(candidates), n=1)
    return matches[0] if matches else None


def _unknown_keys_message(unknown: list, valid: typing.Sequence[str], what: str) -> str:
    parts = []
    for key in unknown:
        best = _closest(key, valid)
        hint = f" (did you mean {best!r}?)" if best is not None else ""
        parts.append(f"{key!r}{hint}")
    return f"Unknown {what} {', '.join(parts)}. Valid keys: {', '.join(valid)}."


def _is_torch_device(value: Any) -> bool:
    # torch is not imported here: a value can only be a torch.device once torch is loaded.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.device)


def _to_plain(value: Any, key: str) -> Any:
    """Return ``value`` as YAML/JSON-safe builtins, or raise ``ConfigurationError``.

    Tuples become lists (``from_dict`` turns them back for tuple-typed fields),
    numpy scalars become Python scalars and a ``torch.device`` becomes its
    string (``"cuda:0"``).
    """
    if isinstance(value, np.generic):  # before the builtins: np.float64 subclasses float
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_plain(v, key) for v in value]
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise _configuration_error(
                    f"Parameter {key!r} is a mapping with a non-string key {k!r}; YAML and JSON need string keys."
                )
            out[k] = _to_plain(v, key)
        return out
    if _is_torch_device(value):
        return str(value)
    raise _configuration_error(
        f"Parameter {key!r} holds a {type(value).__name__}, which cannot be written to YAML or JSON. "
        "Use a plain value (None, bool, int, float, str, or lists and dicts of these)."
    )


def _is_tuple_hint(hint: Any) -> bool:
    if hint is tuple:
        return True
    origin = typing.get_origin(hint)
    if origin is tuple:
        return True
    if origin is Union or origin is types.UnionType:
        return any(_is_tuple_hint(arg) for arg in typing.get_args(hint))
    return False


@dataclasses.dataclass
class EstimatorConfig:
    """Base class of the ``*Config`` dataclasses that mirror an estimator's constructor.

    Subclasses are dataclasses whose fields are exactly the constructor's
    keyword arguments, with the same defaults. ``to_dict`` returns the values
    as they are; ``to_yaml`` and JSON writers need plain values.
    """

    # -- dict --------------------------------------------------------------

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Field names, in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))

    def to_dict(self) -> dict:
        """Field values keyed by name, in declaration order (values are not copied or converted)."""
        return {name: getattr(self, name) for name in self.field_names()}

    @classmethod
    def from_dict(cls, d: Mapping) -> EstimatorConfig:
        """Build a config from a mapping of field values.

        Raises
        ------
        ConfigurationError
            If ``d`` is not a mapping, has a key that is not a field (the
            message names the closest field), or misses a field without a
            default. Lists given for tuple-typed fields become tuples.
        """
        if not isinstance(d, Mapping):
            raise _configuration_error(
                f"{cls.__name__}.from_dict expects a mapping of parameters, got {type(d).__name__}."
            )
        names = cls.field_names()
        unknown = [k for k in d if k not in names]
        if unknown:
            raise _configuration_error(
                f"{cls.__name__}: " + _unknown_keys_message(unknown, names, "key" if len(unknown) == 1 else "keys")
            )
        missing = [
            f.name
            for f in dataclasses.fields(cls)
            if f.name not in d and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise _configuration_error(f"{cls.__name__}: missing required parameter(s) {', '.join(map(repr, missing))}.")
        tuple_fields = cls._tuple_fields()
        values = {k: (tuple(v) if k in tuple_fields and isinstance(v, list) else v) for k, v in d.items()}
        return cls(**values)

    @classmethod
    def _tuple_fields(cls) -> frozenset:
        """Fields annotated as a tuple (also inside Optional/Union) or with a tuple default."""
        try:
            hints = typing.get_type_hints(cls)
        except (NameError, TypeError):  # unresolvable forward references: fall back to the defaults
            hints = {}
        out = set()
        for f in dataclasses.fields(cls):
            if _is_tuple_hint(hints.get(f.name)) or isinstance(f.default, tuple):
                out.add(f.name)
        return frozenset(out)

    # -- YAML --------------------------------------------------------------

    def to_plain_dict(self) -> dict:
        """``to_dict()`` converted to YAML/JSON-safe builtins (tuples become lists).

        Raises
        ------
        ConfigurationError
            If a value is not a plain value (None, bool, int, float, str, or
            lists and string-keyed dicts of these).
        """
        return {k: _to_plain(v, k) for k, v in self.to_dict().items()}

    def to_yaml(self, path: PathLike) -> None:
        """Write ``{"model": to_plain_dict()}`` to ``path``."""
        _write_yaml({"model": self.to_plain_dict()}, Path(path))

    @classmethod
    def from_yaml(cls, path: PathLike) -> EstimatorConfig:
        """Build a config from the ``model:`` section of a YAML file (see :func:`load_yaml`)."""
        return cls.from_dict(load_yaml(path)["model"])


# ---------------------------------------------------------------------------
# YAML reading and writing
# ---------------------------------------------------------------------------

# YAML 1.1 (PyYAML) reads "1e-5" as a string; omegaconf reads it as a float.
# The PyYAML fallback uses the same float pattern so both backends agree.
_FLOAT_PATTERN = re.compile(
    r"""^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)


def _pyyaml_loader(yaml_module: Any) -> type:
    class _Loader(yaml_module.SafeLoader):
        pass

    _Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_PATTERN, list("-+0123456789."))
    return _Loader


def _read_yaml(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None:
        try:
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except OSError:
            raise
        # Translation boundary: omegaconf fails with its YAML backend's errors, its grammar's and its
        # resolvers'; each becomes a ConfigurationError naming the file, chained to the original error.
        except Exception as err:
            raise _configuration_error(f"Could not read YAML file {str(path)!r}: {err}") from err
    try:
        import yaml
    except ImportError:
        raise ImportError(f"Reading YAML needs omegaconf or PyYAML. {_YAML_EXTRA_HINT}") from None
    with open(path, encoding="utf-8") as fh:
        try:
            return yaml.load(fh, Loader=_pyyaml_loader(yaml))  # a SafeLoader subclass
        except yaml.YAMLError as err:
            raise _configuration_error(f"Could not read YAML file {str(path)!r}: {err}") from err


def _write_yaml(content: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(content, fh, sort_keys=False, default_flow_style=False)
        return
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError(f"Writing YAML needs PyYAML or omegaconf. {_YAML_EXTRA_HINT}") from None
    OmegaConf.save(OmegaConf.create(content), path)


def load_yaml(path: PathLike) -> dict:
    """Read a config file and return ``{"model": dict, "data": dict}``.

    The file must have a top-level ``model:`` section with the constructor
    parameters. An optional ``data:`` section holds the arguments of
    :meth:`dfm_python.PanelData.from_wide` (``time_col``, ``target_cols``,
    ``exog_cols``, ``freq``); it is ``{}`` when absent. Interpolations are
    resolved when omegaconf is installed.

    Raises
    ------
    ImportError
        If neither omegaconf nor PyYAML is installed (the ``yaml`` extra).
    ConfigurationError
        If the file is not a mapping, has no ``model:`` section (a flat file),
        has another top-level key, or a section is not a mapping.
    """
    path = Path(path)
    raw = _read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise _configuration_error(
            f"{str(path)!r} must hold a mapping with a 'model:' section, got a {type(raw).__name__}."
        )
    if "model" not in raw:
        raise _configuration_error(
            f"{str(path)!r} has no top-level 'model:' section. Put the constructor parameters under 'model:' and "
            "the PanelData.from_wide arguments under 'data:'; flat config files are not supported."
        )
    unknown = [k for k in raw if k not in _YAML_SECTIONS]
    if unknown:
        raise _configuration_error(
            f"{str(path)!r}: " + _unknown_keys_message(unknown, _YAML_SECTIONS, "top-level section")
        )
    sections = {}
    for name in _YAML_SECTIONS:
        section = raw.get(name)
        if section is None:
            section = {}
        if not isinstance(section, Mapping):
            raise _configuration_error(
                f"{str(path)!r}: the '{name}:' section must be a mapping, got a {type(section).__name__}."
            )
        sections[name] = dict(section)
    return sections

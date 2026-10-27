"""``Namespace``: the read-only container of a fitted model's interpretable output.

``params_``, ``states_`` and ``regimes_`` are namespaces (spec section 5). A
namespace is a read-only :class:`~collections.abc.Mapping` from field names to
values, with attribute access (``model.states_.factors`` is
``model.states_["factors"]``), ``dir()`` listing the fields (so tab
completion works) and a ``repr`` with one line per field: its name, type and
shape, units and a short meaning.

A namespace holds ``data``: the arrays, labels and small values its fields
are made of. Each :class:`Field` either reads its value from ``data`` (under
its own name) or builds it with a function of ``data``. Built fields are
computed on first access and cached, so a large frame costs nothing until it
is read. Pickling (and ``copy.deepcopy``) keeps ``data`` and the field
descriptions, not the cache: a saved or copied model carries the arrays, not
the frames built from them. A field that does not apply to the fitted
configuration has the value None.

Builder functions must be module-level functions (they are pickled by
reference), taking the ``data`` mapping and returning the value.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = ["Field", "Namespace"]


@dataclass(frozen=True)
class Field:
    """One field of a :class:`Namespace`.

    Parameters
    ----------
    name : str
        The field name (an identifier).
    meaning : str
        A short description for the namespace's ``repr``.
    units : str, optional
        The units of the values (``"data units"``, ``"factor units"``, ...).
    build : callable, optional
        ``build(data) -> value``, computed on first access and cached. Without
        it the value is ``data[name]``.
    shape : callable, optional
        ``shape(data) -> tuple``, the shape shown by ``repr`` for a built field
        that has not been computed yet (so ``repr`` does not build it).
    kind : str, optional
        The type name shown by ``repr`` for a built field that has not been
        computed yet; default ``"DataFrame"``.
    """

    name: str
    meaning: str
    units: str = ""
    build: Optional[Callable[[Mapping[str, Any]], Any]] = None
    shape: Optional[Callable[[Mapping[str, Any]], tuple]] = None
    kind: str = "DataFrame"


_MISSING = object()


class Namespace(Mapping):
    """A read-only mapping of named fields with attribute access and lazy, cached fields.

    Parameters
    ----------
    title : str
        Shown in the ``repr`` header, e.g. ``"DFM.states_"``.
    fields : sequence of Field
        The fields, in display order. Names must be unique identifiers.
    data : mapping
        The arrays, labels and values the fields read or are built from.
        Every field without a ``build`` function needs ``data[field.name]``.

    Raises
    ------
    ValueError
        For duplicate or non-identifier field names, a name that shadows a
        mapping method, or a field without ``build`` whose value is not in
        ``data``.
    """

    __slots__ = ("_title", "_fields", "_data", "_cache")

    def __init__(self, title: str, fields: Sequence[Field], data: Mapping[str, Any]) -> None:
        fields = tuple(fields)
        names = [f.name for f in fields]
        bad = [n for n in names if not (isinstance(n, str) and n.isidentifier() and not n.startswith("_"))]
        if bad:
            raise ValueError(f"Namespace field names must be public identifiers, got {bad}.")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Namespace field names must be unique; duplicated: {dupes}.")
        shadowed = [n for n in names if hasattr(Mapping, n)]
        if shadowed:
            raise ValueError(f"Namespace field names {shadowed} shadow mapping methods.")
        missing = [f.name for f in fields if f.build is None and f.name not in data]
        if missing:
            raise ValueError(f"Namespace data has no value for the fields {missing}.")
        object.__setattr__(self, "_title", str(title))
        object.__setattr__(self, "_fields", {f.name: f for f in fields})
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_cache", {})

    # Mapping protocol

    def __getitem__(self, name: str) -> Any:
        field = self._fields.get(name) if isinstance(name, str) else None
        if field is None:
            raise KeyError(name)
        if field.build is None:
            return self._data[name]
        value = self._cache.get(name, _MISSING)
        if value is _MISSING:
            value = field.build(self._data)
            self._cache[name] = value
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    # Equality would compare frames element-wise; a namespace equals itself only.
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    # Attribute access

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails. Private names (pickling looks
        # up __reduce_ex__ and friends before the slots are set) never reach the fields.
        if name.startswith("_"):
            raise AttributeError(name)
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return self[name]
        close = difflib.get_close_matches(name, list(fields), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise AttributeError(
            f"{object.__getattribute__(self, '_title')} has no field {name!r}; its fields are {list(fields)}.{hint}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{self._title} is read-only; cannot set {name!r}.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{self._title} is read-only; cannot delete {name!r}.")

    def __dir__(self) -> list[str]:
        return sorted({*self._fields, "get", "items", "keys", "values"})

    # Pickling: the data and the field descriptions, never the cache.

    def __getstate__(self) -> dict:
        return {"title": self._title, "fields": tuple(self._fields.values()), "data": self._data}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_title", state["title"])
        object.__setattr__(self, "_fields", {f.name: f for f in state["fields"]})
        object.__setattr__(self, "_data", dict(state["data"]))
        object.__setattr__(self, "_cache", {})

    def __reduce_ex__(self, protocol: int) -> tuple:
        return (_rebuild, (self.__getstate__(),))

    # Display

    def _describe(self, field: Field) -> str:
        """``type[shape]`` of a field's value, without building an uncomputed field."""
        if field.build is None:
            return _type_and_shape(self._data[field.name])
        value = self._cache.get(field.name, _MISSING)
        if value is not _MISSING:
            return _type_and_shape(value)
        if field.shape is not None:
            return f"{field.kind}{_shape_text(field.shape(self._data))}"
        return f"{field.kind} (not built yet)"

    def __repr__(self) -> str:
        rows = [(name, self._describe(field), field.units, field.meaning) for name, field in self._fields.items()]
        widths = [max((len(row[i]) for row in rows), default=0) for i in range(3)]
        lines = [f"{self._title} ({len(rows)} fields)"]
        for name, kind, units, meaning in rows:
            lines.append(f"  {name:<{widths[0]}}  {kind:<{widths[1]}}  {units:<{widths[2]}}  {meaning}".rstrip())
        return "\n".join(lines)


def _rebuild(state: dict) -> Namespace:
    ns = Namespace.__new__(Namespace)
    ns.__setstate__(state)
    return ns


def _shape_text(shape: tuple) -> str:
    return "[" + " x ".join(str(int(n)) for n in shape) + "]"


def _type_and_shape(value: Any) -> str:
    if value is None:
        return "None (does not apply)"
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        return f"{type(value).__name__}{_shape_text(value.shape)}"
    return type(value).__name__

"""torch, the optional dependency of the torch models (the ``deep`` extra).

``DFM`` needs numpy only. ``DDFM``, ``iVDFM`` and ``AFM`` train with torch
(iVDFM also with einops); ``pip install "dfm-python[deep]"`` installs both.
Importing ``dfm_python`` and building any estimator does not import torch.
Fitting or loading a torch model does, and raises ``ImportError`` naming the
extra when it is missing.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable

__all__ = ["DEEP_INSTALL", "lazy_torch_exports", "require_torch", "torch"]

#: The install command the ``ImportError`` names.
DEEP_INSTALL = 'pip install "dfm-python[deep]"'


def require_torch(what: str = "Using DDFM, iVDFM or AFM") -> Any:
    """Import and return torch, or raise ``ImportError`` naming the ``deep`` extra.

    einops, the other package of the extra, is checked too. ``what`` starts
    the message (for example ``"DDFM.fit"``).
    """
    try:
        import einops  # noqa: F401
        import torch
    except ImportError as err:
        raise ImportError(f"{what} needs the deep extra (torch and einops), which is not installed: {DEEP_INSTALL}") from err
    return torch


class _Torch:
    """Stands in for the torch module: each attribute access goes through :func:`require_torch`.

    The estimator modules of the torch models bind ``torch`` to an instance,
    so importing them imports torch only when their code first uses it.
    Dunder names raise ``AttributeError`` without importing torch, so
    introspection (``inspect.unwrap``, doctest collection) never imports it.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(require_torch(), name)

    def __repr__(self) -> str:
        return "<torch, imported on first use>"


#: ``torch``, imported on first attribute access.
torch = _Torch()


def lazy_torch_exports(package: str, exports: Mapping[str, str]) -> Callable[[str], Any]:
    """A module ``__getattr__`` for ``package`` that imports its torch-backed names on first access.

    ``exports`` maps each name to the module that defines it, relative to
    ``package``. The first access calls :func:`require_torch`, imports that
    module and stores the value in the package namespace.
    """

    def __getattr__(name: str) -> Any:
        module = exports.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        require_torch(f"{package}.{name}")
        value = getattr(import_module(module, package), name)
        setattr(sys.modules[package], name, value)
        return value

    return __getattr__

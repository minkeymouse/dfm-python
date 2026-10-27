"""``dfm_python.load``: load any saved run, dispatching on its ``model_type``.

Format-2 runs (spec section 10) are dispatched through the private table
:data:`_MODELS` to the model class's ``_load``. Format-1 runs, written by the
estimators that still sit on :class:`dfm_python.base.BaseFactorModel`, go
through :func:`dfm_python.registry.all_estimators` and ``Model.load``.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..utils.errors import ConfigurationError
from .layout import FORMAT_2, PathLike, read_config, read_config_2, saved_format

__all__ = ["load"]

#: ``model_type`` -> ``"module:Class"`` (or the class itself) of the models that save format-2 runs.
_MODELS: dict = {
    "AFM": "dfm_python.models.afm.estimator:AFM",
    "DDFM": "dfm_python.models.ddfm.estimator:DDFM",
    "DFM": "dfm_python.models.dfm.estimator:DFM",
    "iVDFM": "dfm_python.models.ivdfm.estimator:iVDFM",
}


def _model_class(model_type: Any, path: PathLike) -> type:
    """The class :data:`_MODELS` names for ``model_type``, imported on demand."""
    entry = _MODELS.get(model_type) if isinstance(model_type, str) else None
    if entry is None:
        raise ConfigurationError(
            f"{str(path)!r} holds a {model_type!r} run, which is not a model of this release "
            f"(known: {', '.join(sorted(_MODELS))})."
        )
    if isinstance(entry, type):
        return entry
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)


def load(path: PathLike, *, device: Any = None, allow_pickle: bool = False) -> Any:
    """Load a run saved with ``save(path)``, whatever its model.

    Parameters
    ----------
    path : str or os.PathLike
        The saved-run directory.
    device : str or torch.device, optional
        Device for torch models; None uses the model's ``device`` parameter.
        Ignored by numpy models.
    allow_pickle : bool, default False
        Format-1 runs only: allow reading ``scaler.pkl`` and importing
        estimator classes outside ``sklearn.preprocessing`` named in
        ``config.json``. Both can run arbitrary code; enable only for
        directories you trust. Format-2 runs hold nothing pickled.

    Raises
    ------
    ConfigurationError
        If ``path`` is a single-file checkpoint from an earlier release, is
        not a saved run, or names a ``model_type`` this release does not know.
    """
    if saved_format(path) == FORMAT_2:
        config = read_config_2(path)
        return _model_class(config["model_type"], path)._load(path, device=device)
    from ..registry import all_estimators

    config = read_config(path)
    model_type = config["model_type"]
    registered = dict(all_estimators())
    cls = registered.get(model_type)
    if cls is None:
        names = ", ".join(sorted(registered)) or "none yet"
        raise ConfigurationError(
            f"{str(path)!r} holds a {model_type!r} run, which is not a registered estimator (registered: {names}). "
            f"Load it with {model_type}.load(path) from the class that saved it."
        )
    return cls.load(path, device=device, allow_pickle=allow_pickle)

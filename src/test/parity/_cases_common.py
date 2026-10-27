"""Shared types and helpers of the golden cases (``_cases_<model>.py``).

``CaseRun`` and ``Case`` describe a case; ``jsonable`` / ``normalized`` turn
recipes and hyperparameters into the JSON a manifest holds; the digests,
global-RNG reset, CPU forcing and ``data/macro.csv`` helpers are used by
several models' cases.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import random
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MACRO_CSV = REPO_ROOT / "data" / "macro.csv"

# Global RNGs are reset to this value before every case, so a case never depends
# on what ran before it. The models then seed themselves as they do today.
GLOBAL_RNG_PRESEED = 20251019


# --------------------------------------------------------------------- types
@dataclasses.dataclass
class CaseRun:
    outputs: dict[str, np.ndarray]
    hyperparameters: dict[str, Any]
    data: dict[str, Any]
    raised: dict[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class Case:
    """A golden case.

    ``cross_platform=False`` marks a case too ill-conditioned to compare off
    the capture platform at any useful tolerance: it is compared bitwise on
    the capture platform and skipped elsewhere (``_golden.skip_unless_comparable``).
    No registered case uses it today. ``dfm_macro_weekly_monthly`` and
    ``dfm_macro_monthly_quarterly_update`` did until the DFM numerical-stability
    rewrite; their recaptured fits agree within 1e-6 relative (2.5e-10 at most,
    norm-relative, on parameters and forecasts) between the default AVX-512
    kernels and forced AVX2 (Haswell, Sandybridge) or SSE (Nehalem) kernels.
    """

    name: str
    model: str
    run: Callable[[], CaseRun]
    cross_platform: bool = True

    def __repr__(self) -> str:  # readable pytest ids
        return self.name


# ------------------------------------------------------------------- helpers
def jsonable(obj: Any) -> Any:
    """Convert ``obj`` into plain JSON types (lists, dicts, str, float, int, bool, None)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if _is_torch(obj, "device"):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"class": type(obj).__name__,
                "fields": jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})}
    return {"class": type(obj).__name__}


def normalized(obj: Any) -> Any:
    """``jsonable`` followed by a JSON round trip, for comparing with a manifest."""
    return json.loads(json.dumps(jsonable(obj), sort_keys=True))


def _digest(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, pd.DataFrame):
            h.update(json.dumps([str(c) for c in a.columns]).encode())
            h.update(json.dumps([str(i) for i in a.index]).encode())
            for col in a.columns:
                if np.issubdtype(a[col].dtype, np.number):
                    h.update(np.ascontiguousarray(a[col].to_numpy(np.float64)).tobytes())
                else:
                    h.update(json.dumps([str(v) for v in a[col]]).encode())
        else:
            arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attrs(obj: Any, names: list[str]) -> dict[str, Any]:
    return {n: getattr(obj, n) for n in names}


def _class_name(obj: Any) -> str | None:
    return None if obj is None else type(obj).__name__


def _is_torch(obj: Any, cls: str) -> bool:
    """``isinstance(obj, torch.<cls>)`` without importing torch (the DFM cases run without it)."""
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, getattr(torch, cls))


def _np(t: Any) -> np.ndarray:
    if _is_torch(t, "Tensor"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _reset_global_rngs() -> None:
    random.seed(GLOBAL_RNG_PRESEED)
    np.random.seed(GLOBAL_RNG_PRESEED)
    try:
        import torch
    except ImportError:  # the DFM cases also run without the deep extra
        return
    torch.manual_seed(GLOBAL_RNG_PRESEED)


@contextlib.contextmanager
def force_cpu() -> Iterator[None]:
    """Make today's ``cuda if available else cpu`` device choice resolve to CPU."""
    import torch

    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        yield


def _macro_tail(n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(MACRO_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.iloc[-n_rows:].reset_index(drop=True)


def _macro_recipe(**extra: Any) -> dict[str, Any]:
    return {"source": "data/macro.csv", "source_sha256": _file_digest(MACRO_CSV), **extra}

"""Helpers for the golden (parity) suite.

Golden fixtures pin the numbers each engine produces for a given dataset and a
given set of hyperparameters. They are written by ``make_golden.py`` and read by
the ``test_golden_*`` modules.

Comparison rules (design spec, section 11):

* Every run happens under :func:`pinned`: one BLAS/OpenMP thread, one torch
  thread and deterministic torch algorithms.
* Each fixture's manifest records the :func:`fingerprint` of the platform it
  was captured on: the CPU (architecture, model name and the SIMD level torch
  dispatches to, e.g. ``AVX2`` or ``AVX512``), the Python minor version, the
  numpy and torch versions and the BLAS builds.
* When the current fingerprint equals the recorded one, arrays must be bitwise
  equal (``assert_array_equal``). This is the regression guard.
* On any other platform the comparison falls back to ``assert_allclose`` with a
  tolerance chosen by dtype (float32: ``rtol=1e-2, atol=1e-5``; float64:
  ``rtol=5e-3, atol=1e-6``; see :data:`TOLERANCES`), and a
  :class:`GoldenToleranceWarning` names the fallback in the test output.
  Integer and boolean arrays stay exact everywhere. The tolerances are loose on
  purpose: another CPU runs other SIMD kernels (vector width, FMA, summation
  order), and the iterative fits (EM, gradient training) amplify those
  last-bit differences to about 1.4e-3 relative in float64 and 3e-3 in float32
  on the CI runners; the tolerances leave at least a 3x margin over every
  difference seen there and under forced AVX2 kernels locally. ``atol`` covers
  entries near zero, whose relative error is meaningless. Off the capture
  platform the suite checks that the code reproduces the same fit, not the
  same bits.
* A case registered with ``cross_platform=False`` (see ``_cases_common.Case``) is too
  ill-conditioned to compare across platforms at any useful tolerance. Off the
  capture platform every test that consumes it is skipped
  (:func:`skip_unless_comparable`); on the capture platform it is compared
  bitwise like every other case. No registered case uses this opt-out today.

CI runs on GitHub-hosted runners whose CPUs differ from the capture machine, so
CI takes the tolerance path (look for ``GoldenToleranceWarning`` in its output)
and would skip any ``cross_platform=False`` case. Only the capture machine, the one
whose fingerprint the manifests record, runs the bitwise comparison.

torch is imported by the functions that need it, so :func:`load_golden` also
works without it (the DFM oracle tests run without the ``deep`` extra).
"""

from __future__ import annotations

import contextlib
import json
import platform
import re
import sys
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg  # noqa: F401  (load scipy's BLAS so the fingerprint is stable)
import threadpoolctl

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: Fallback ``(rtol, atol)`` by dtype off the capture platform (see the module docstring).
TOLERANCES: dict[np.dtype, tuple[float, float]] = {
    np.dtype(np.float32): (1e-2, 1e-5),
    np.dtype(np.float64): (5e-3, 1e-6),
}

__all__ = [
    "FIXTURES_DIR",
    "TOLERANCES",
    "GoldenToleranceWarning",
    "assert_golden",
    "fingerprint",
    "load_golden",
    "matches_capture_platform",
    "pinned",
    "skip_unless_comparable",
]


class GoldenToleranceWarning(UserWarning):
    """The golden comparison fell back from bitwise equality to a tolerance."""


@contextlib.contextmanager
def pinned() -> Iterator[None]:
    """Pin threads and determinism for a golden run, restoring all three on exit.

    Sets ``threadpoolctl.threadpool_limits(1)`` (BLAS and OpenMP),
    ``torch.set_num_threads(1)`` and ``torch.use_deterministic_algorithms(True)``.
    """
    import torch

    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with threadpoolctl.threadpool_limits(limits=1):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn_only)
            torch.set_num_threads(prev_threads)


def _blas_info() -> list[dict[str, Any]]:
    keys = ("internal_api", "prefix", "version", "architecture", "threading_layer")
    seen = set()
    out = []
    for lib in threadpoolctl.threadpool_info():
        if lib.get("user_api") != "blas":
            continue
        entry = {k: lib.get(k) for k in keys}
        key = tuple(entry.values())
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return sorted(out, key=lambda e: tuple(str(v) for v in e.values()))


def _cpu_model() -> str:
    """The CPU model name without its clock-speed suffix.

    ``model name`` from ``/proc/cpuinfo`` on Linux, else ``platform.processor()``.
    The ``@ 2.80GHz`` suffix is dropped: on virtual machines it changes between
    hosts of the same CPU model and does not affect the kernels selected.
    """
    name = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                key, sep, value = line.partition(":")
                if sep and key.strip() == "model name":
                    name = value.strip()
                    break
    except OSError:
        pass
    return re.sub(r"\s*@\s*[\d.]+\s*GHz\b", "", name).strip()


def _cpu_capability() -> str | None:
    """The SIMD level torch's CPU kernels dispatch to (e.g. ``AVX2``, ``AVX512``)."""
    import torch

    get = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
    return None if get is None else str(get())


def fingerprint() -> dict[str, Any]:
    """Platform fingerprint that decides between bitwise and tolerance comparison."""
    import torch

    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_capability": _cpu_capability(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "blas": _blas_info(),
    }


def _load_manifest(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


def load_golden(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return ``(manifest, arrays)`` for fixture ``name``."""
    manifest = _load_manifest(name)
    with np.load(FIXTURES_DIR / f"{name}.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return manifest, arrays


def _normalize_fingerprint(fp: Mapping[str, Any]) -> Any:
    # JSON turns tuples into lists; compare through the same encoding.
    return json.loads(json.dumps(fp, sort_keys=True))


def _matches(manifest: Mapping[str, Any]) -> bool:
    return _normalize_fingerprint(fingerprint()) == _normalize_fingerprint(manifest["fingerprint"])


def matches_capture_platform(name: str) -> bool:
    """Whether this platform's fingerprint equals the one fixture ``name`` was captured on."""
    return _matches(_load_manifest(name))


def skip_unless_comparable(name: str) -> None:
    """Skip the calling test when case ``name`` is pinned bitwise only and this is another platform.

    A no-op for names that are not registered cases (e.g. the helper tests'
    fixtures), for cases with ``cross_platform=True`` and on the capture platform.
    """
    import pytest

    from ._cases import CASES

    case = CASES.get(name)
    if case is None or case.cross_platform or matches_capture_platform(name):
        return
    pytest.skip(f"golden '{name}' is pinned bitwise on the capture platform only "
                f"(cross_platform=False: too ill-conditioned to compare across platforms); "
                f"this platform's fingerprint differs from the capture platform's")


def assert_golden(actual: Mapping[str, Any], name: str, *, exclude: Iterable[str] = ()) -> None:
    """Compare ``actual`` arrays with the golden fixture ``name``.

    Bitwise on the fingerprinted platform; otherwise ``assert_allclose`` with
    the dtype's :data:`TOLERANCES` (float32's for any floating dtype other than
    float64) and exact equality for integer and boolean arrays, plus a
    :class:`GoldenToleranceWarning`. Off the capture platform a case registered
    with ``cross_platform=False`` is skipped instead of compared.

    ``exclude`` names fixture outputs that this comparison leaves out; each
    must exist in the fixture and be absent from ``actual``. Every other
    fixture output must be in ``actual``.
    """
    manifest, expected = load_golden(name)
    actual = {k: np.asarray(v) for k, v in actual.items()}
    exclude = set(exclude)
    unknown = sorted(exclude - set(expected))
    assert not unknown, f"golden '{name}': excluded keys {unknown} are not fixture outputs"
    overlap = sorted(exclude & set(actual))
    assert not overlap, f"golden '{name}': keys {overlap} are both excluded and given"
    expected = {k: v for k, v in expected.items() if k not in exclude}

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    assert not missing and not extra, (
        f"golden '{name}': output keys differ (missing={missing}, unexpected={extra})")

    bitwise = _matches(manifest)
    if not bitwise:
        skip_unless_comparable(name)
        tolerances = ", ".join(f"{dt.name} rtol={r:g} atol={a:g}" for dt, (r, a) in TOLERANCES.items())
        warnings.warn(
            f"golden '{name}': platform fingerprint differs from the capture platform; "
            f"falling back to assert_allclose ({tolerances}; integers and booleans exact) "
            f"instead of bitwise assert_array_equal",
            GoldenToleranceWarning, stacklevel=2)

    for key in sorted(expected):
        exp, act = expected[key], actual[key]
        assert act.dtype == exp.dtype, (
            f"golden '{name}' [{key}]: dtype {act.dtype} != expected {exp.dtype}")
        assert act.shape == exp.shape, (
            f"golden '{name}' [{key}]: shape {act.shape} != expected {exp.shape}")
        msg = f"golden '{name}' [{key}]"
        if bitwise or not np.issubdtype(exp.dtype, np.floating):
            np.testing.assert_array_equal(act, exp, err_msg=msg, strict=True)
        else:
            rtol, atol = TOLERANCES.get(exp.dtype, TOLERANCES[np.dtype(np.float32)])
            np.testing.assert_allclose(act, exp, rtol=rtol, atol=atol, err_msg=msg)

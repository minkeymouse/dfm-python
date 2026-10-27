"""Capture the golden fixtures from the current code, or check that it reproduces them.

Run from the repository root::

    uv run python src/test/parity/make_golden.py            # every case
    uv run python src/test/parity/make_golden.py dfm_synthetic ivdfm_macro
    uv run python src/test/parity/make_golden.py --check    # compare, write nothing

Each case runs on CPU under ``pinned()`` and writes
``src/test/parity/fixtures/<case>.npz`` (engine outputs) and ``<case>.json``
(git commit, platform fingerprint, data recipe, constructed hyperparameters,
output names). Fixtures are captured before any engine code moves; regenerate
them only when a numerical change is intended.

``--check`` runs the cases and compares each with its fixture: every array
byte for byte (with its dtype and shape) and the manifest's data recipe,
hyperparameters, raised error and output names. It writes nothing and exits
with status 1 when any case differs, so a refactoring of the cases can show
that they regenerate byte-identical fixtures.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported anywhere

import json
import subprocess
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
# src/ must come first: the editable install appends it after the stdlib, whose
# own ``test`` package would otherwise shadow src/test.
sys.path.insert(0, str(_SRC))

import numpy as np

from test.parity._cases import CASES, GLOBAL_RNG_PRESEED, normalized
from test.parity._golden import FIXTURES_DIR, fingerprint, pinned

REPO_ROOT = _SRC.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _manifest(case, run, fp: dict, commit: str, src_dirty: bool) -> dict:
    return {
        "name": case.name,
        "model": case.model,
        "git_commit": commit,
        "src_dirty": src_dirty,
        "fingerprint": fp,
        "global_rng_preseed": GLOBAL_RNG_PRESEED,
        "data": normalized(run.data),
        "hyperparameters": normalized(run.hyperparameters),
        "raised": normalized(run.raised),
        "outputs": {k: {"dtype": str(np.asarray(v).dtype), "shape": list(np.asarray(v).shape)}
                    for k, v in sorted(run.outputs.items())},
    }


def _differences(case, run, fp: dict) -> list[str]:
    """How a fresh run of ``case`` differs from its fixture (empty when byte-identical)."""
    manifest = json.loads((FIXTURES_DIR / f"{case.name}.json").read_text())
    fresh = _manifest(case, run, fp, manifest["git_commit"], manifest["src_dirty"])
    problems = [f"manifest {key!r} differs" for key in ("name", "model", "global_rng_preseed", "data",
                                                         "hyperparameters", "raised", "outputs")
                if json.loads(json.dumps(fresh[key])) != manifest[key]]
    if json.loads(json.dumps(fp, sort_keys=True)) != manifest["fingerprint"]:
        problems.append("note: this platform's fingerprint differs from the capture platform's")
    with np.load(FIXTURES_DIR / f"{case.name}.npz", allow_pickle=False) as npz:
        expected = {k: npz[k] for k in npz.files}
    actual = {k: np.asarray(v) for k, v in run.outputs.items()}
    if set(expected) != set(actual):
        problems.append(f"output keys differ: missing {sorted(set(expected) - set(actual))}, "
                        f"extra {sorted(set(actual) - set(expected))}")
    for key in sorted(set(expected) & set(actual)):
        exp, act = expected[key], actual[key]
        if exp.dtype != act.dtype or exp.shape != act.shape or exp.tobytes() != act.tobytes():
            problems.append(f"array {key!r} is not byte-identical")
    return problems


def check(selected: list) -> int:
    """Run ``selected`` cases and compare them with their fixtures; 0 when all are byte-identical."""
    failed = 0
    for case in selected:
        with pinned():
            fp = fingerprint()
            run = case.run()
        problems = _differences(case, run, fp)
        real = [p for p in problems if not p.startswith("note:")]
        failed += bool(real)
        status = "byte-identical" if not real else "DIFFERS: " + "; ".join(real)
        notes = [p for p in problems if p.startswith("note:")]
        print(f"{case.name:40s} {status}" + (f" ({notes[0]})" if notes else ""))
    return 1 if failed else 0


def main(names: list[str]) -> int:
    check_mode = "--check" in names
    names = [n for n in names if n != "--check"]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown case(s): {unknown}; known: {sorted(CASES)}", file=sys.stderr)
        return 2
    selected = [CASES[n] for n in names] if names else list(CASES.values())
    if check_mode:
        return check(selected)

    commit = _git("rev-parse", "HEAD")
    src_dirty = bool(_git("status", "--porcelain", "--", "src/dfm_python"))
    if src_dirty:
        print("warning: src/dfm_python has uncommitted changes", file=sys.stderr)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for case in selected:
        start = time.perf_counter()
        with pinned():
            fp = fingerprint()
            run = case.run()
        elapsed = time.perf_counter() - start

        np.savez_compressed(FIXTURES_DIR / f"{case.name}.npz", **run.outputs)
        manifest = _manifest(case, run, fp, commit, src_dirty)
        (FIXTURES_DIR / f"{case.name}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        size = (FIXTURES_DIR / f"{case.name}.npz").stat().st_size
        status = f"raised {run.raised['type']}" if run.raised else f"{len(run.outputs)} outputs"
        print(f"{case.name:40s} {elapsed:6.2f}s  {size / 1024:8.1f} KiB  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

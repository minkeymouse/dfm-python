"""The registry of the golden cases: every model's cases, in a fixed order.

Each ``_cases_<model>.py`` builds its data from a recipe, constructs the
model, trains it and returns a :class:`CaseRun` (see ``_cases_common``) with

* ``outputs``: the engine numbers that the fixture pins,
* ``hyperparameters``: the values the constructed object actually holds, read
  off its attributes right after construction (not the values passed in), and
* ``data``: the recipe plus a digest of the exact input that was fitted.

``make_golden.py`` writes these to ``fixtures/<case>.npz`` and ``<case>.json``
(or, with ``--check``, compares a fresh run with them); the ``test_golden_*``
modules re-run the same case and compare. A model's work item edits only its
own ``_cases_<model>.py``.
"""

from __future__ import annotations

from . import _cases_afm, _cases_ddfm, _cases_dfm, _cases_ivdfm
from ._cases_common import GLOBAL_RNG_PRESEED, Case, CaseRun, jsonable, normalized

__all__ = ["CASES", "GLOBAL_RNG_PRESEED", "Case", "CaseRun", "cases_for", "check_case", "jsonable", "normalized"]

CASES: dict[str, Case] = {
    case.name: case
    for case in [*_cases_afm.CASES, *_cases_dfm.CASES, *_cases_ddfm.CASES, *_cases_ivdfm.CASES]
}


def cases_for(model: str) -> list[Case]:
    return [c for c in CASES.values() if c.model == model]


def check_case(name: str, run: CaseRun) -> None:
    """Compare a fresh run of case ``name`` with its fixture."""
    from ._golden import assert_golden, load_golden

    manifest, _ = load_golden(name)
    assert normalized(run.data) == manifest["data"], f"golden '{name}': input data differs from the fixture"
    assert normalized(run.raised) == manifest["raised"], (
        f"golden '{name}': raised {run.raised}, fixture recorded {manifest['raised']}")
    assert normalized(run.hyperparameters) == manifest["hyperparameters"], (
        f"golden '{name}': constructed hyperparameters differ from the fixture")
    assert_golden(run.outputs, name)

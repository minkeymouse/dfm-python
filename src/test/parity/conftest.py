"""Fixtures for the golden suite."""

import pytest

from ._cases_common import Case
from ._golden import pinned as _pinned
from ._golden import skip_unless_comparable


@pytest.fixture
def pinned():
    """Run the test with one thread and deterministic torch algorithms (restored afterwards)."""
    with _pinned():
        yield


@pytest.fixture(autouse=True)
def _skip_cases_pinned_to_the_capture_platform(request):
    """Skip a test parametrized with a ``cross_platform=False`` case off the capture platform."""
    callspec = getattr(request.node, "callspec", None)
    case = callspec.params.get("case") if callspec is not None else None
    if isinstance(case, Case):
        skip_unless_comparable(case.name)

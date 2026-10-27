"""The API contract (``dfm_python.testing.check_estimator``) for AFM."""

from dfm_python import AFM
from dfm_python.testing import check_estimator


def test_contract():
    ran = check_estimator(AFM)
    # Every check that applies to every estimator ran, save/load included.
    for check in ("pure_construction", "config_serialization", "unfitted", "fit", "purity", "silence", "save_load"):
        assert check in ran, check

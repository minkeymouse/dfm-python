"""Input robustness of AFM: its own target rules and infinite values."""

import numpy as np
import pytest

from dfm_python import AFM, PanelData
from dfm_python.utils.errors import DataValidationError


def _no_training(monkeypatch, cls) -> None:
    def fail(self, *args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(cls, "_fit", fail)


def test_allocation_model_keeps_its_own_rules():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((40, 4)) * 0.01
    returns[:, 1] = 0.0
    panel = PanelData.from_arrays(returns, characteristics=rng.standard_normal((40, 4, 2)))
    AFM()._check_training_targets(panel)


@pytest.mark.parametrize("slot", ["y", "characteristics"])
def test_afm_rejects_infinite_values(monkeypatch, slot):
    _no_training(monkeypatch, AFM)
    rng = np.random.default_rng(0)
    returns, chars = rng.standard_normal((40, 4)) * 0.01, rng.standard_normal((40, 4, 2))
    (returns if slot == "y" else chars)[5, 1] = np.inf
    with pytest.raises(DataValidationError, match=f"AFM: {slot} contains infinite values .* finite values\\.$"):
        AFM(**AFM.get_test_params()[1]).fit(PanelData.from_arrays(returns, characteristics=chars))

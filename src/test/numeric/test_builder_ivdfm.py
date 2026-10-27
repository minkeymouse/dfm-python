"""Tests for the iVDFM functions of numeric.builder."""

import pytest
import torch


class TestOptimizerNameValidation:
    """An unknown optimizer name is an error, not a silent switch to another optimizer."""

    def test_ivdfm_unknown_optimizer_and_scheduler_raise(self):
        from dfm_python.numeric.builder import build_ivdfm_optimizer
        from dfm_python.utils.errors import ConfigurationError
        model = torch.nn.Linear(3, 3)
        with pytest.raises(ConfigurationError, match="optimizer"):
            build_ivdfm_optimizer(model, 0.01, 'adam', max_epochs=3)
        with pytest.raises(ConfigurationError, match="scheduler_type"):
            build_ivdfm_optimizer(model, 0.01, 'Adam', max_epochs=3, scheduler_type='linear')

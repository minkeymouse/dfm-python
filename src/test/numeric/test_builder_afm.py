"""Tests for the AFM functions of numeric.builder."""

import pytest
import torch


class TestOptimizerNameValidation:
    """An unknown optimizer name is an error, not a silent switch to another optimizer."""

    def test_afm_unknown_optimizer_raises(self):
        from dfm_python.numeric.builder import build_afm_optimizer
        from dfm_python.utils.errors import ConfigurationError
        params = list(torch.nn.Linear(3, 3).parameters())
        assert isinstance(build_afm_optimizer(params, 1e-3, "AdamW"), torch.optim.AdamW)
        with pytest.raises(ConfigurationError, match="optimizer"):
            build_afm_optimizer(params, 1e-3, "nadam")

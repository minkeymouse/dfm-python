"""Tests for config schema module."""

from dfm_python.config import (
    DFMConfig,
    DDFMConfig,
)
from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.config.constants import DEFAULT_EM_THRESHOLD, DEFAULT_MAX_ITER


class TestDFMConfig:
    """Test suite for DFMConfig (the estimator mirror) and the private engine config."""

    def test_dfm_config_initialization(self):
        """DFMConfig() holds the DFM constructor defaults."""
        from dfm_python import DFM
        config = DFMConfig()
        assert config.to_dict() == DFM().get_params()
        assert config.tol == 1e-6 and config.max_iter == 5000 and config.blocks is None

    def test_engine_config_initialization(self):
        """The engine config can be initialized with default values."""
        # The engine config requires at least one block
        config = _DFMEngineConfig(blocks={'block1': {'num_factors': 2, 'series': []}})
        assert config is not None
    
    def test_dfm_config_parameters(self):
        """Test the engine config's parameter setting."""
        config = _DFMEngineConfig(
            blocks={'block1': {'num_factors': 2, 'series': []}},
            max_iter=DEFAULT_MAX_ITER,
            threshold=DEFAULT_EM_THRESHOLD
        )
        assert config.max_iter == DEFAULT_MAX_ITER
        assert config.threshold == DEFAULT_EM_THRESHOLD


class TestDDFMConfig:
    """Test suite for DDFMConfig."""
    
    def test_ddfm_config_initialization(self):
        """DDFMConfig() holds the DDFM constructor defaults; it has no clock (DDFM is single frequency)."""
        from dfm_python import DDFM
        config = DDFMConfig()
        assert config is not None
        assert config.to_dict() == DDFM().get_params()
        assert not hasattr(config, "clock")


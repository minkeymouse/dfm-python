"""Tests for config schema module."""

import pytest
from dfm_python.config import (
    DFMConfig,
    DDFMConfig,
    KDFMConfig,
)
from dfm_python.config.constants import DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER


class TestDFMConfig:
    """Test suite for DFMConfig."""
    
    def test_dfm_config_initialization(self):
        """Test DFMConfig can be initialized with default values."""
        # DFMConfig requires at least one block
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}})
        assert config is not None
    
    def test_dfm_config_parameters(self):
        """Test DFMConfig parameter setting."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': []}},
            max_iter=100,
            threshold=1e-5
        )
        assert config.max_iter == 100
        assert config.threshold == 1e-5


class TestDDFMConfig:
    """Test suite for DDFMConfig."""
    
    def test_ddfm_config_initialization(self):
        """Test DDFMConfig can be initialized."""
        config = DDFMConfig()
        assert config is not None
        assert config.clock == 'm'  # Default clock frequency


class TestKDFMConfig:
    """Test suite for KDFMConfig."""
    
    def test_kdfm_config_initialization(self):
        """Test KDFMConfig can be initialized."""
        config = KDFMConfig()
        assert config is not None
        assert config.clock == 'm'  # Default clock frequency
        assert config.ar_order == DEFAULT_KDFM_AR_ORDER  # Default AR order from constants
        assert config.ma_order == DEFAULT_KDFM_MA_ORDER  # Default MA order from constants
        assert config.ar_order == 1
        assert config.ma_order == 0




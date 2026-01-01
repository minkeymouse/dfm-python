"""Tests for datamodule.ddfm_dm module."""

import pytest
from dfm_python.datamodule.ddfm_dm import DDFMDataModule
from dfm_python.config import DFMConfig


class TestDDFMDataModule:
    """Test suite for DDFMDataModule."""
    
    def test_ddfm_datamodule_initialization(self, sample_data, sample_config):
        """Test DDFMDataModule can be initialized."""
        dm = DDFMDataModule(config=sample_config, data=sample_data)
        assert dm is not None
        assert dm.config == sample_config
        assert hasattr(dm, 'window_size')
        assert hasattr(dm, 'batch_size')
        assert hasattr(dm, 'data')
    
    def test_ddfm_datamodule_setup(self, sample_data, sample_config):
        """Test DDFMDataModule setup."""
        dm = DDFMDataModule(config=sample_config, data=sample_data)
        dm.setup()
        assert dm.train_dataset is not None
        # Mx/Wx removed - use target_scaler instead
        assert hasattr(dm, 'target_scaler')


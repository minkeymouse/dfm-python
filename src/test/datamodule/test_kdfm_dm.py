"""Tests for datamodule.kdfm_dm module."""

import pytest
from dfm_python.datamodule.kdfm_dm import KDFMDataModule
from dfm_python.config import DFMConfig


class TestKDFMDataModule:
    """Test suite for KDFMDataModule."""
    
    def test_kdfm_datamodule_initialization(self, sample_data, sample_config):
        """Test KDFMDataModule can be initialized."""
        dm = KDFMDataModule(config=sample_config, data=sample_data)
        assert dm is not None
        assert dm.config == sample_config
        assert hasattr(dm, 'batch_size')
        assert hasattr(dm, 'data')
        assert hasattr(dm, '_dfm_dm')
    
    def test_kdfm_datamodule_setup(self, sample_data, sample_config):
        """Test KDFMDataModule setup."""
        dm = KDFMDataModule(config=sample_config, data=sample_data)
        dm.setup()
        assert dm.train_dataset is not None
        # Mx/Wx removed - use target_scaler instead
        assert hasattr(dm, 'target_scaler')


"""Tests for datamodule.dfm_dm module."""

import pytest
from dfm_python.datamodule.dfm_dm import DFMDataModule
from dfm_python.config import DFMConfig


class TestDFMDataModule:
    """Test suite for DFMDataModule."""
    
    def test_dfm_datamodule_initialization(self, sample_data, sample_config):
        """Test DFMDataModule can be initialized."""
        dm = DFMDataModule(config=sample_config, data=sample_data)
        assert dm is not None
        assert dm.config == sample_config
        assert hasattr(dm, 'data')
        assert hasattr(dm, 'data_processed')
    
    def test_dfm_datamodule_setup(self, sample_data, sample_config):
        """Test DFMDataModule setup."""
        dm = DFMDataModule(config=sample_config, data=sample_data)
        dm.setup()
        assert dm.data_processed is not None
        # Mx/Wx removed - use target_scaler instead
        assert hasattr(dm, 'target_scaler')


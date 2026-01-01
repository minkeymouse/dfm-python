"""Tests for trainer.kdfm module."""

import pytest
from dfm_python.trainer.kdfm import KDFMTrainer
from dfm_python.config.constants import DEFAULT_MAX_EPOCHS


class TestKDFMTrainer:
    """Test suite for KDFMTrainer."""
    
    def test_kdfm_trainer_initialization(self):
        """Test KDFMTrainer can be initialized."""
        trainer = KDFMTrainer(max_epochs=DEFAULT_MAX_EPOCHS)
        assert trainer is not None
    
    def test_kdfm_trainer_default_max_epochs(self):
        """Test KDFMTrainer uses DEFAULT_MAX_EPOCHS as default."""
        trainer = KDFMTrainer()
        # Check that max_epochs is set to DEFAULT_MAX_EPOCHS
        assert trainer.max_epochs == DEFAULT_MAX_EPOCHS
    
    def test_kdfm_trainer_fit(self, sample_data, sample_config):
        """Test KDFMTrainer fit method."""
        # Minimal smoke test - verify fit method exists and can be called
        # Full fit test would require model and datamodule setup
        from dfm_python.models.kdfm import KDFM
        from dfm_python.datamodule.kdfm_dm import KDFMDataModule
        
        model = KDFM(config=sample_config, ar_order=1, ma_order=0)
        dm = KDFMDataModule(config=sample_config, data=sample_data)
        dm.setup()
        
        trainer = KDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
        # Verify fit method exists and is callable
        assert hasattr(trainer, 'fit')
        assert callable(trainer.fit)
        # Note: Full fit test would require proper model initialization and may take time
        # This is a minimal interface test


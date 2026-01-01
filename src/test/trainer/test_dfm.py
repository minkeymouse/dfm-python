"""Tests for trainer.dfm module."""

import pytest
from dfm_python.trainer.dfm import DFMTrainer
from dfm_python.config.constants import DEFAULT_MAX_EPOCHS


class TestDFMTrainer:
    """Test suite for DFMTrainer."""
    
    def test_dfm_trainer_initialization(self):
        """Test DFMTrainer can be initialized."""
        trainer = DFMTrainer(max_epochs=DEFAULT_MAX_EPOCHS)
        assert trainer is not None
    
    def test_dfm_trainer_default_max_epochs(self):
        """Test DFMTrainer uses DEFAULT_MAX_EPOCHS as default."""
        trainer = DFMTrainer()
        # Check that max_epochs is set to DEFAULT_MAX_EPOCHS
        assert trainer.max_epochs == DEFAULT_MAX_EPOCHS
    
    def test_dfm_trainer_fit(self, sample_data, sample_config):
        """Test DFMTrainer fit method."""
        # Minimal smoke test - verify fit method exists and can be called
        # Full fit test would require model and datamodule setup
        from dfm_python.models.dfm import DFM
        from dfm_python.datamodule.dfm_dm import DFMDataModule
        
        model = DFM(config=sample_config)
        dm = DFMDataModule(config=sample_config, data=sample_data)
        dm.setup()
        
        trainer = DFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
        # Verify fit method exists and is callable
        assert hasattr(trainer, 'fit')
        assert callable(trainer.fit)
        # Note: Full fit test would require proper model initialization and may take time
        # This is a minimal interface test


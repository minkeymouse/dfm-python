"""Tests for trainer.ddfm module."""

import pytest
from dfm_python.trainer.ddfm import DDFMTrainer, DDFMDenoisingTrainer
from dfm_python.config.constants import DEFAULT_MAX_EPOCHS


class TestDDFMTrainer:
    """Test suite for DDFMTrainer."""
    
    def test_ddfm_trainer_initialization(self):
        """Test DDFMTrainer can be initialized."""
        trainer = DDFMTrainer(max_epochs=DEFAULT_MAX_EPOCHS)
        assert trainer is not None
    
    def test_ddfm_trainer_default_max_epochs(self):
        """Test DDFMTrainer uses DEFAULT_MAX_EPOCHS as default."""
        trainer = DDFMTrainer()
        # Check that max_epochs is set to DEFAULT_MAX_EPOCHS
        assert trainer.max_epochs == DEFAULT_MAX_EPOCHS


class TestDDFMDenoisingTrainer:
    """Test suite for DDFMDenoisingTrainer."""
    
    def test_denoising_trainer_initialization(self):
        """Test DDFMDenoisingTrainer can be initialized."""
        from dfm_python.models.ddfm import DDFM
        # Create a DDFM model for the trainer
        model = DDFM(encoder_layers=[64, 32], num_factors=2)
        trainer = DDFMDenoisingTrainer(model=model)
        assert trainer is not None
        assert trainer.model is model


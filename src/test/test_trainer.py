"""Tests for PyTorch Lightning trainers.

Tests align with PyTorch Lightning best practices and DFM/DDFM training.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Any

from dfm_python.trainer import (
    DFMTrainer,
    DDFMTrainer,
)
# Import private functions directly (not exported in __all__)
# These are internal functions used for testing
import dfm_python.trainer as trainer_module
_normalize_accelerator = trainer_module._normalize_accel
_normalize_precision = trainer_module._normalize_prec
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.utils.time import TimeIndex, parse_timestamp
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# Import shared test helper functions
from test_helpers import (
    assert_trainer_defaults,
    assert_trainer_callbacks,
    assert_trainer_attribute_value
)


class TestDFMTrainer:
    """Test DFMTrainer for DFM models."""
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test DFM config."""
        return Path(__file__).parent.parent.parent / "config" / "experiment" / "test_dfm.yaml"
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    
    def test_dfm_trainer_initialization(self):
        """Test DFMTrainer initialization."""
        trainer = DFMTrainer(max_epochs=50)
        assert trainer.max_epochs == 50
        assert isinstance(trainer, DFMTrainer)
    
    def test_dfm_trainer_defaults(self):
        """Test DFMTrainer default parameters.
        
        Verifies that default values match documented defaults in DFMTrainer class:
        - max_epochs: 100 (EM iterations)
        - enable_progress_bar: True
        - enable_model_summary: False (DFM modules are simple, usually not needed)
        """
        trainer = DFMTrainer()
        # DFM uses EM algorithm, so defaults should be appropriate
        # Verify actual default values match implementation and documentation
        assert_trainer_defaults(
            trainer,
            expected_max_epochs=100,
            expected_progress_bar=True,
            expected_model_summary=False
        )
    
    def test_dfm_trainer_from_config(self, test_config_path):
        """Test DFMTrainer.from_config() method using test config."""
        if not test_config_path.exists():
            pytest.skip(f"Test config file not found: {test_config_path}")
        
        source = YamlSource(test_config_path)
        config = source.load()
        
        trainer = DFMTrainer.from_config(config)
        assert isinstance(trainer, DFMTrainer)
        # Should extract max_iter from config
        assert trainer.max_epochs == config.max_iter
    
    def test_dfm_trainer_callbacks(self):
        """Test DFMTrainer callback setup.
        
        Verifies that DFMTrainer has expected callbacks:
        - EarlyStopping callback (monitoring 'loglik' metric)
        """
        trainer = DFMTrainer(max_epochs=50)
        # DFMTrainer should have EarlyStopping callback configured
        assert_trainer_callbacks(trainer, expected_callback_types=['EarlyStopping'])
        
        # Verify EarlyStopping callback properties
        early_stopping = next(
            (cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)),
            None
        )
        assert early_stopping is not None, "EarlyStopping callback should be present"
        assert early_stopping.monitor == 'loglik', "EarlyStopping should monitor 'loglik' metric"


class TestDDFMTrainer:
    """Test DDFMTrainer for DDFM models."""
    
    def test_ddfm_trainer_initialization(self):
        """Test DDFMTrainer initialization."""
        trainer = DDFMTrainer(max_epochs=100)
        assert trainer.max_epochs == 100
        assert isinstance(trainer, DDFMTrainer)
    
    def test_ddfm_trainer_defaults(self):
        """Test DDFMTrainer default parameters.
        
        Verifies that default values match documented defaults in DDFMTrainer class:
        - max_epochs: 100 (training epochs)
        - enable_progress_bar: True
        - enable_model_summary: True (useful for debugging DDFM architecture)
        """
        trainer = DDFMTrainer()
        # DDFM uses neural network training, so defaults should be appropriate
        # Verify actual default values match implementation and documentation
        assert_trainer_defaults(
            trainer,
            expected_max_epochs=100,
            expected_progress_bar=True,
            expected_model_summary=True
        )
    
    @pytest.fixture
    def test_ddfm_config_path(self):
        """Path to test DDFM config."""
        return Path(__file__).parent.parent.parent / "config" / "experiment" / "test_ddfm.yaml"
    
    def test_ddfm_trainer_from_config(self, test_ddfm_config_path):
        """Test DDFMTrainer.from_config() method using test config."""
        if not test_ddfm_config_path.exists():
            pytest.skip(f"Test DDFM config file not found: {test_ddfm_config_path}")
        
        source = YamlSource(test_ddfm_config_path)
        config = source.load()
        
        trainer = DDFMTrainer.from_config(config)
        assert isinstance(trainer, DDFMTrainer)
        # Should extract epochs from config (handles multiple config formats)
        # from_config() checks config.epochs, config.ddfm_epochs, or defaults to 100
        if hasattr(config, 'epochs'):
            assert trainer.max_epochs == config.epochs
        elif hasattr(config, 'ddfm_epochs'):
            assert trainer.max_epochs == config.ddfm_epochs
        else:
            # If neither exists, should default to 100
            assert trainer.max_epochs == 100
    
    def test_ddfm_trainer_callbacks(self):
        """Test DDFMTrainer callback setup.
        
        Verifies that DDFMTrainer has expected callbacks:
        - EarlyStopping callback (patience=20, monitoring 'train_loss')
        - LearningRateMonitor callback
        - ModelCheckpoint callback
        """
        trainer = DDFMTrainer(max_epochs=100)
        # DDFMTrainer should have multiple callbacks configured
        assert_trainer_callbacks(
            trainer,
            expected_callback_types=['EarlyStopping', 'LearningRateMonitor', 'ModelCheckpoint']
        )
        
        # Verify EarlyStopping callback properties
        early_stopping = next(
            (cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)),
            None
        )
        assert early_stopping is not None, "EarlyStopping callback should be present"
        assert early_stopping.patience == 20, "EarlyStopping should have patience=20 for DDFM"
        assert early_stopping.monitor == 'train_loss', "EarlyStopping should monitor 'train_loss' for DDFM"
        
        # Verify LearningRateMonitor is present
        lr_monitor = next(
            (cb for cb in trainer.callbacks if isinstance(cb, LearningRateMonitor)),
            None
        )
        assert lr_monitor is not None, "LearningRateMonitor callback should be present"
        
        # Verify ModelCheckpoint is present
        checkpoint = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None
        )
        assert checkpoint is not None, "ModelCheckpoint callback should be present"
    
    def test_ddfm_trainer_gradient_clipping(self):
        """Test DDFMTrainer gradient clipping for stability.
        
        Verifies that gradient_clip_val is properly configured when provided.
        """
        trainer = DDFMTrainer(max_epochs=100, gradient_clip_val=1.0)
        # Should have gradient clipping configured with the specified value
        assert_trainer_attribute_value(trainer, 'gradient_clip_val', 1.0)


class TestTrainerConsistency:
    """Test trainer consistency with PyTorch Lightning."""
    
    def test_trainer_inheritance(self):
        """Test that trainers inherit from pl.Trainer."""
        dfm_trainer = DFMTrainer()
        ddfm_trainer = DDFMTrainer()
        
        import pytorch_lightning as pl
        assert isinstance(dfm_trainer, pl.Trainer)
        assert isinstance(ddfm_trainer, pl.Trainer)
    
    def test_trainer_device_handling(self):
        """Test trainer device configuration."""
        trainer = DFMTrainer(accelerator='cpu', devices=1)
        # Lightning may normalize accelerator values, so use normalization helper for consistent comparison
        assert hasattr(trainer, 'accelerator')
        # Use normalization helper to ensure consistent comparison with trainer implementation
        normalized_accelerator = _normalize_accelerator(trainer.accelerator)
        assert normalized_accelerator == 'cpu'
    
    def test_trainer_precision(self):
        """Test trainer precision configuration."""
        trainer = DDFMTrainer(precision=32)
        # Lightning may normalize precision values (int to string or vice versa)
        # Use normalization helper for consistent comparison with trainer implementation
        assert hasattr(trainer, 'precision')
        # Use normalization helper to ensure consistent comparison
        normalized_precision = _normalize_precision(trainer.precision)
        # Normalized precision should be 32 (int) for simple precision values
        assert normalized_precision == 32


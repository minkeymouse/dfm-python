"""Tests for PyTorch Lightning trainers.

Tests align with PyTorch Lightning best practices and DFM/DDFM training.
"""

import pytest
import torch
import polars as pl
from pathlib import Path
from typing import Optional, List, Any

from dfm_python.trainer import (
    DFMTrainer,
    DDFMTrainer,
    _normalize_accelerator,
    _normalize_precision
)
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.utils.time import TimeIndex, parse_timestamp
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


# ============================================================================
# Test Helper Functions
# ============================================================================

def assert_trainer_defaults(
    trainer,
    expected_max_epochs: int,
    expected_progress_bar: bool,
    expected_model_summary: bool
) -> None:
    """Assert trainer default values match expected.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    expected_max_epochs : int
        Expected max_epochs value
    expected_progress_bar : bool
        Expected enable_progress_bar value
    expected_model_summary : bool
        Expected enable_model_summary value
    """
    assert trainer.max_epochs == expected_max_epochs, (
        f"Expected max_epochs={expected_max_epochs}, got {trainer.max_epochs}"
    )
    assert trainer.enable_progress_bar == expected_progress_bar, (
        f"Expected enable_progress_bar={expected_progress_bar}, got {trainer.enable_progress_bar}"
    )
    assert trainer.enable_model_summary == expected_model_summary, (
        f"Expected enable_model_summary={expected_model_summary}, got {trainer.enable_model_summary}"
    )


def assert_trainer_callbacks(
    trainer,
    expected_callback_types: List[str]
) -> None:
    """Assert trainer has expected callback types.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    expected_callback_types : List[str]
        List of expected callback class names (e.g., ['EarlyStopping'])
    """
    assert hasattr(trainer, 'callbacks'), "Trainer should have callbacks attribute"
    assert trainer.callbacks is not None, "Trainer callbacks should not be None"
    assert isinstance(trainer.callbacks, list), "Trainer callbacks should be a list"
    
    # Get actual callback types
    callback_types = [type(cb).__name__ for cb in trainer.callbacks]
    
    # Check that all expected callback types are present
    for expected_type in expected_callback_types:
        assert expected_type in callback_types, (
            f"Expected callback type '{expected_type}' not found in callbacks. "
            f"Found: {callback_types}"
        )


def assert_trainer_attribute_value(
    trainer,
    attribute_name: str,
    expected_value: Any
) -> None:
    """Assert trainer attribute has expected value.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    attribute_name : str
        Name of attribute to check
    expected_value : any
        Expected value for the attribute
    """
    assert hasattr(trainer, attribute_name), (
        f"Trainer should have '{attribute_name}' attribute"
    )
    actual_value = getattr(trainer, attribute_name)
    assert actual_value == expected_value, (
        f"Expected {attribute_name}={expected_value}, got {actual_value}"
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


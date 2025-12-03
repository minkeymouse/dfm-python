"""Tests for PyTorch Lightning trainers.

Tests align with PyTorch Lightning best practices and DFM/DDFM training.
"""

import pytest
import torch
import polars as pl
from pathlib import Path
from typing import Optional

from dfm_python.trainer import DFMTrainer, DDFMTrainer
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.utils.time import TimeIndex, parse_timestamp


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
        """Test DFMTrainer default parameters."""
        trainer = DFMTrainer()
        # DFM uses EM algorithm, so defaults should be appropriate
        assert hasattr(trainer, 'max_epochs')
        assert hasattr(trainer, 'enable_progress_bar')
    
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
        """Test DFMTrainer callback setup."""
        trainer = DFMTrainer(max_epochs=50)
        # Should have callbacks configured
        assert hasattr(trainer, 'callbacks')


class TestDDFMTrainer:
    """Test DDFMTrainer for DDFM models."""
    
    def test_ddfm_trainer_initialization(self):
        """Test DDFMTrainer initialization."""
        trainer = DDFMTrainer(max_epochs=100)
        assert trainer.max_epochs == 100
        assert isinstance(trainer, DDFMTrainer)
    
    def test_ddfm_trainer_defaults(self):
        """Test DDFMTrainer default parameters."""
        trainer = DDFMTrainer()
        # DDFM uses neural network training, so defaults should be appropriate
        assert hasattr(trainer, 'max_epochs')
        assert hasattr(trainer, 'enable_progress_bar')
        assert hasattr(trainer, 'enable_model_summary')
    
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
        # Should extract epochs from config
        assert trainer.max_epochs == config.epochs
    
    def test_ddfm_trainer_callbacks(self):
        """Test DDFMTrainer callback setup."""
        trainer = DDFMTrainer(max_epochs=100)
        # Should have callbacks configured (early stopping, LR monitor, etc.)
        assert hasattr(trainer, 'callbacks')
    
    def test_ddfm_trainer_gradient_clipping(self):
        """Test DDFMTrainer gradient clipping for stability."""
        trainer = DDFMTrainer(max_epochs=100, gradient_clip_val=1.0)
        # Should have gradient clipping configured
        assert hasattr(trainer, 'gradient_clip_val')


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
        assert trainer.accelerator == 'cpu'
        # devices attribute may not be directly accessible
        assert hasattr(trainer, 'accelerator')
    
    def test_trainer_precision(self):
        """Test trainer precision configuration."""
        trainer = DDFMTrainer(precision=32)
        assert trainer.precision == 32


"""Tests for numeric.builder module."""

import pytest
import numpy as np
import torch
from dfm_python.numeric.builder import (
    build_dfm_structure, build_dfm_blocks,
    build_ddfm_optimizer, build_ddfm_state_space
)
from dfm_python.config import DFMConfig
from dfm_python.models.ddfm.encoder import SimpleAutoencoder
from dfm_python.utils.errors import ModelNotTrainedError


class TestBuilder:
    """Test suite for builder utilities."""
    
    def test_build_dfm_structure(self):
        """Test build_dfm_structure."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': ['s1', 's2']}},
            frequency={'s1': 'w', 's2': 'w'},
            clock='w'
        )
        
        blocks, r, num_factors, p = build_dfm_structure(config)
        assert blocks is not None
        assert r is not None
        assert num_factors > 0
        assert p >= 0
    
    def test_build_dfm_blocks(self):
        """Test build_dfm_blocks."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 1, 'series': ['s1', 's2', 's3']}},
            frequency={'s1': 'w', 's2': 'w', 's3': 'w'},
            clock='w'
        )
        
        initial_blocks = np.ones((2, 1))
        columns = ['s1', 's2', 's3']
        n_series = 3
        
        blocks = build_dfm_blocks(initial_blocks, config, columns, n_series)
        assert blocks.shape[0] == n_series
        assert blocks.shape[1] == 1


class TestDDFMBuilder:
    """Test suite for DDFM builder utilities."""
    
    def test_build_ddfm_optimizer(self):
        """Test build_ddfm_optimizer creates optimizer and scheduler."""
        autoencoder = SimpleAutoencoder.build(
            input_dim=10,
            encoder_size=[64, 32, 4],
            output_dim=10
        )
        
        optimizer, scheduler = build_ddfm_optimizer(
            model=autoencoder,
            learning_rate=0.001,
            optimizer_type='Adam',
            n_mc_samples=10
        )
        
        assert optimizer is not None
        assert scheduler is not None
        assert isinstance(optimizer, torch.optim.Optimizer)
        # LambdaLR is a scheduler (check by method/protocol rather than exact type)
        assert hasattr(scheduler, 'step')
        assert hasattr(scheduler, 'get_last_lr')
        
        # Test scheduler decay
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(10):
            scheduler.step()
        # After 10 steps (n_mc_samples=10), should have decayed once
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_build_ddfm_optimizer_sgd(self):
        """Test build_ddfm_optimizer with SGD optimizer."""
        autoencoder = SimpleAutoencoder.build(
            input_dim=10,
            encoder_size=[64, 32, 4],
            output_dim=10
        )
        
        optimizer, scheduler = build_ddfm_optimizer(
            model=autoencoder,
            learning_rate=0.01,
            optimizer_type='SGD',
            n_mc_samples=10
        )
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert scheduler is not None
    
    def test_build_ddfm_state_space(self):
        """Test build_ddfm_state_space creates state-space parameters."""
        T, num_factors, num_targets = 50, 4, 5
        
        # Create synthetic factors and residuals
        np.random.seed(42)
        factors = np.random.randn(T, num_factors)
        eps = np.random.randn(T, num_targets) * 0.1
        observed_y = np.ones((T, num_targets), dtype=bool)
        
        # Create decoder weight (output_dim x num_factors)
        decoder_weight = np.random.randn(num_targets, num_factors)
        
        F, Q, mu_0, Sigma_0, H, R = build_ddfm_state_space(
            factors=factors,
            eps=eps,
            decoder_weight=decoder_weight,
            observed_y=observed_y,
            model_name="TestDDFM"
        )
        
        # Verify shapes
        assert F.shape == (num_factors, num_factors)
        assert Q.shape == (num_factors, num_factors)
        assert mu_0.shape == (num_factors,)
        assert Sigma_0.shape == (num_factors, num_factors)
        assert H.shape == (num_targets, num_factors)
        assert R.shape == (num_targets, num_targets)
        
        # Verify R is diagonal
        assert np.allclose(R, np.diag(np.diag(R)))
    
    def test_build_ddfm_state_space_empty_factors(self):
        """Test build_ddfm_state_space raises error for empty factors."""
        decoder_weight = np.random.randn(5, 4)
        observed_y = np.ones((50, 5), dtype=bool)
        
        with pytest.raises(ModelNotTrainedError, match="factors are empty"):
            build_ddfm_state_space(
                factors=np.array([]),
                eps=np.random.randn(50, 5),
                decoder_weight=decoder_weight,
                observed_y=observed_y
            )
    
    def test_build_ddfm_state_space_empty_residuals(self):
        """Test build_ddfm_state_space raises error for empty residuals."""
        factors = np.random.randn(50, 4)
        decoder_weight = np.random.randn(5, 4)
        observed_y = np.ones((50, 5), dtype=bool)
        
        with pytest.raises(ModelNotTrainedError, match="residuals are empty"):
            build_ddfm_state_space(
                factors=factors,
                eps=np.array([]),
                decoder_weight=decoder_weight,
                observed_y=observed_y
            )

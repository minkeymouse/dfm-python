"""Tests for Deep Dynamic Factor Model (DDFM).

This module contains tests for the DDFM implementation, including:
- Synthetic DGP tests
- Factor recovery tests
- Comparison with reference DDFM implementation
"""

import sys
from pathlib import Path
import unittest
import numpy as np
from typing import Optional
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False

from dfm_python.engine.synthetic_dgp import SyntheticDGP
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.models.ddfm import DDFM
from dfm_python.models.dfm import DFMLinear


class TestDDFMSyntheticDGP(unittest.TestCase):
    """Test DDFM on synthetic data with known factors."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not _has_torch:
            self.skipTest("PyTorch not available")
        
        self.seed = 42
        self.t_obs = 200
        self.n_series = 10
        self.n_factors = 1
        
        # Create synthetic DGP
        self.dgp = SyntheticDGP(
            seed=self.seed,
            n=self.n_series,
            r=self.n_factors,
            poly_degree=1,  # Linear for now
            sign_features=0,
            rho=0.7,
            alpha=0.2,
            u=0.1,
        )
    
    def test_dgp_simulation(self):
        """Test that synthetic DGP works correctly."""
        X = self.dgp.simulate(self.t_obs, portion_missings=0.0)
        
        self.assertEqual(X.shape, (self.t_obs, self.n_series))
        self.assertEqual(np.sum(np.isnan(X)), 0)
        self.assertIsNotNone(self.dgp.f)
        self.assertEqual(self.dgp.f.shape[0], self.t_obs)
    
    def test_dgp_with_missing_data(self):
        """Test DGP with missing data."""
        X = self.dgp.simulate(self.t_obs, portion_missings=0.2)
        
        self.assertEqual(X.shape, (self.t_obs, self.n_series))
        n_missing = int(self.t_obs * self.n_series * 0.2)
        self.assertEqual(np.sum(np.isnan(X)), n_missing)
    
    def test_dgp_evaluation(self):
        """Test trace R² evaluation metric."""
        _ = self.dgp.simulate(self.t_obs)
        
        # Perfect recovery should give R² = 1
        r2 = self.dgp.evaluate(self.dgp.f)
        self.assertAlmostEqual(r2, 1.0, places=5)
        
        # Random factors should give low R²
        random_f = np.random.randn(self.t_obs, self.n_factors)
        r2_random = self.dgp.evaluate(random_f)
        self.assertLess(r2_random, 0.5)
    
    def test_ddfm_on_linear_dgp(self):
        """Test DDFM on linear synthetic DGP."""
        # Simulate data
        X = self.dgp.simulate(self.t_obs, portion_missings=0.0)
        
        # Create simple config
        config = self._create_simple_config()
        
        # Fit DDFM
        ddfm = DDFM(
            encoder_layers=[32, 16],
            num_factors=self.n_factors,
            epochs=50,  # Reduced for testing
            batch_size=32,
            learning_rate=0.001,
        )
        
        result = ddfm.fit(X, config)
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertEqual(result.Z.shape[0], self.t_obs)
        self.assertEqual(result.Z.shape[1], self.n_factors)
        self.assertEqual(result.C.shape[0], self.n_series)
        self.assertEqual(result.C.shape[1], self.n_factors)
        
        # Check factor recovery (should be reasonable for linear case)
        r2 = self.dgp.evaluate(result.Z)
        self.assertGreater(r2, 0.3, "Factor recovery should be reasonable")
    
    def test_ddfm_vs_linear_dfm(self):
        """Compare DDFM with linear DFM on linear DGP."""
        # Simulate linear data
        X = self.dgp.simulate(self.t_obs, portion_missings=0.0)
        config = self._create_simple_config()
        
        # Fit linear DFM
        dfm_linear = DFMLinear()
        result_dfm = dfm_linear.fit(X, config, max_iter=50)
        
        # Fit DDFM
        ddfm = DDFM(
            encoder_layers=[32],
            num_factors=self.n_factors,
            epochs=50,
            batch_size=32,
        )
        result_ddfm = ddfm.fit(X, config)
        
        # Both should recover factors reasonably well
        r2_dfm = self.dgp.evaluate(result_dfm.Z)
        r2_ddfm = self.dgp.evaluate(result_ddfm.Z)
        
        # Linear DFM should be better on linear data, but DDFM should still work
        self.assertGreater(r2_dfm, 0.5, "Linear DFM should recover factors well")
        self.assertGreater(r2_ddfm, 0.3, "DDFM should recover factors reasonably")
    
    def _create_simple_config(self) -> DFMConfig:
        """Create a simple DFMConfig for testing."""
        series = []
        for i in range(self.n_series):
            series.append(SeriesConfig(
                series_id=f'series_{i+1}',
                frequency='m',
                transformation='lin',
                blocks=[1],  # All series load on first block
            ))
        
        blocks = {
            'Block_Global': BlockConfig(
                factors=1,
                ar_lag=1,
                clock='m',
            )
        }
        
        return DFMConfig(
            series=series,
            blocks=blocks,
            clock='m',
            ar_lag=1,
            threshold=1e-4,
            max_iter=100,
        )


class TestDDFMNonlinearDGP(unittest.TestCase):
    """Test DDFM on nonlinear synthetic DGP."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not _has_torch:
            self.skipTest("PyTorch not available")
        
        self.seed = 42
        self.t_obs = 300
        self.n_series = 15
        self.n_factors = 2
    
    def test_ddfm_on_polynomial_dgp(self):
        """Test DDFM on polynomial DGP (nonlinear factors)."""
        # Create DGP with polynomial factors
        dgp = SyntheticDGP(
            seed=self.seed,
            n=self.n_series,
            r=self.n_factors,
            poly_degree=2,  # Quadratic
            sign_features=0,
            rho=0.7,
            alpha=0.2,
            u=0.1,
        )
        
        X = dgp.simulate(self.t_obs, portion_missings=0.0)
        
        # Create config
        series = []
        for i in range(self.n_series):
            series.append(SeriesConfig(
                series_id=f'series_{i+1}',
                frequency='m',
                transformation='lin',
                blocks=[1],
            ))
        
        config = DFMConfig(
            series=series,
            blocks={'Block_Global': BlockConfig(factors=2, ar_lag=1, clock='m')},
            clock='m',
            ar_lag=1,
            threshold=1e-4,
            max_iter=100,
        )
        
        # Fit DDFM with deeper encoder
        ddfm = DDFM(
            encoder_layers=[64, 32, 16],
            num_factors=2,
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
        )
        
        result = ddfm.fit(X, config)
        
        # Check that DDFM can extract factors
        self.assertIsNotNone(result)
        self.assertEqual(result.Z.shape[1], 2)
        
        # Factor recovery should be reasonable
        # Note: DDFM extracts factors that may not match true factors exactly
        # but should capture the factor space
        r2 = dgp.evaluate(result.Z)
        self.assertGreater(r2, 0.2, "DDFM should recover some factor structure")


class TestDDFMRegression(unittest.TestCase):
    """Regression tests to ensure DDFM behavior is stable."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not _has_torch:
            self.skipTest("PyTorch not available")
    
    def test_ddfm_reproducibility(self):
        """Test that DDFM produces reproducible results with same seed."""
        # Create synthetic data
        dgp = SyntheticDGP(seed=123, n=10, r=1)
        X = dgp.simulate(100)
        
        config = self._create_test_config(10)
        
        # Fit DDFM twice with same parameters
        ddfm1 = DDFM(encoder_layers=[32], num_factors=1, epochs=20)
        ddfm2 = DDFM(encoder_layers=[32], num_factors=1, epochs=20)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        result1 = ddfm1.fit(X, config)
        
        torch.manual_seed(42)
        np.random.seed(42)
        result2 = ddfm2.fit(X, config)
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(
            result1.Z,
            result2.Z,
            rtol=1e-3,
            atol=1e-3,
            err_msg="DDFM should be reproducible with same seed"
        )
    
    def test_ddfm_missing_data(self):
        """Test DDFM with missing data."""
        dgp = SyntheticDGP(seed=42, n=10, r=1)
        X = dgp.simulate(100, portion_missings=0.2)
        
        config = self._create_test_config(10)
        
        ddfm = DDFM(encoder_layers=[32], num_factors=1, epochs=30)
        result = ddfm.fit(X, config)
        
        # Should complete without errors
        self.assertIsNotNone(result)
        self.assertEqual(result.Z.shape[0], 100)
    
    def _create_test_config(self, n_series: int) -> DFMConfig:
        """Create a test config."""
        series = []
        for i in range(n_series):
            series.append(SeriesConfig(
                series_id=f'series_{i+1}',
                frequency='m',
                transformation='lin',
                blocks=[1],
            ))
        
        return DFMConfig(
            series=series,
            blocks={'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')},
            clock='m',
            ar_lag=1,
            threshold=1e-4,
            max_iter=50,
        )


if __name__ == '__main__':
    unittest.main()


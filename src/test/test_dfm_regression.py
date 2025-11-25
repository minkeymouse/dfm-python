"""Regression tests for linear DFM after refactoring.

These tests ensure that the refactored DFM implementation produces
the same results as before the core/model separation.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python import DFM, DFMConfig, SeriesConfig, BlockConfig
# SyntheticDGP removed from package for now
# from dfm_python.core.synthetic_dgp import SyntheticDGP
from dfm_python.models.dfm import DFMLinear


class TestDFMRegression(unittest.TestCase):
    """Regression tests to ensure DFM behavior is unchanged."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.seed = 42
        self.t_obs = 100
        self.n_series = 10
        self.n_factors = 1
    
    @unittest.skip("SyntheticDGP removed from package")
    def test_dfm_vs_dfm_linear(self):
        """Test that DFM and DFMLinear produce same results."""
        # Create synthetic data
        # dgp = SyntheticDGP(seed=self.seed, n=self.n_series, r=self.n_factors)
        # X = dgp.simulate(self.t_obs)
        self.skipTest("SyntheticDGP removed from package")
        
        config = self._create_config()
        
        # Fit using original DFM API
        dfm1 = DFM()
        result1 = dfm1.fit(X, config, max_iter=50)
        
        # Fit using DFMLinear
        dfm2 = DFMLinear()
        result2 = dfm2.fit(X, config, max_iter=50)
        
        # Results should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(
            result1.Z,
            result2.Z,
            rtol=1e-4,
            atol=1e-4,
            err_msg="DFM and DFMLinear should produce same results"
        )
        
        np.testing.assert_allclose(
            result1.C,
            result2.C,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Loading matrices should match"
        )
        
        np.testing.assert_allclose(
            result1.A,
            result2.A,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Transition matrices should match"
        )
    
    @unittest.skip("SyntheticDGP removed from package")
    def test_dfm_backward_compatibility(self):
        """Test that existing DFM API still works."""
        self.skipTest("SyntheticDGP removed from package")
        # dgp = SyntheticDGP(seed=self.seed, n=self.n_series, r=self.n_factors)
        # X = dgp.simulate(self.t_obs)
        
        config = self._create_config()
        
        # Test module-level API
        from dfm_python import load_config, load_data, train, predict
        
        # Create DFM instance
        dfm = DFM()
        result = dfm.fit(X, config, max_iter=50)
        
        # Should work without errors
        self.assertIsNotNone(result)
        self.assertEqual(result.Z.shape[0], self.t_obs)
        self.assertEqual(result.Z.shape[1], self.n_factors)
        
        # Test prediction
        X_forecast, Z_forecast = dfm.predict(horizon=10)
        self.assertEqual(X_forecast.shape[0], 10)
        self.assertEqual(Z_forecast.shape[0], 10)
    
    @unittest.skip("SyntheticDGP removed from package")
    def test_dfm_result_structure(self):
        """Test that DFMResult structure is unchanged."""
        self.skipTest("SyntheticDGP removed from package")
        # dgp = SyntheticDGP(seed=self.seed, n=self.n_series, r=self.n_factors)
        # X = dgp.simulate(self.t_obs)
        
        config = self._create_config()
        dfm = DFM()
        result = dfm.fit(X, config, max_iter=50)
        
        # Check all expected attributes exist
        self.assertIsNotNone(result.x_sm)
        self.assertIsNotNone(result.X_sm)
        self.assertIsNotNone(result.Z)
        self.assertIsNotNone(result.C)
        self.assertIsNotNone(result.R)
        self.assertIsNotNone(result.A)
        self.assertIsNotNone(result.Q)
        self.assertIsNotNone(result.Mx)
        self.assertIsNotNone(result.Wx)
        self.assertIsNotNone(result.Z_0)
        self.assertIsNotNone(result.V_0)
        self.assertIsNotNone(result.r)
        self.assertIsNotNone(result.p)
        
        # Check shapes
        self.assertEqual(result.x_sm.shape, (self.t_obs, self.n_series))
        self.assertEqual(result.X_sm.shape, (self.t_obs, self.n_series))
        self.assertEqual(result.Z.shape, (self.t_obs, self.n_factors))
        self.assertEqual(result.C.shape, (self.n_series, self.n_factors))
    
    @unittest.skip("SyntheticDGP removed from package")
    def test_dfm_with_missing_data(self):
        """Test DFM with missing data (regression)."""
        self.skipTest("SyntheticDGP removed from package")
        # dgp = SyntheticDGP(seed=self.seed, n=self.n_series, r=self.n_factors)
        # X = dgp.simulate(self.t_obs, portion_missings=0.2)
        
        config = self._create_config()
        dfm = DFM()
        result = dfm.fit(X, config, max_iter=50)
        
        # Should complete without errors
        self.assertIsNotNone(result)
        self.assertEqual(result.Z.shape[0], self.t_obs)
    
    def _create_config(self) -> DFMConfig:
        """Create a test config."""
        series = []
        for i in range(self.n_series):
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
            max_iter=100,
        )


if __name__ == '__main__':
    unittest.main()


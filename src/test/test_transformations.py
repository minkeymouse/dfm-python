"""Comprehensive tests for DFM transformations and scaling.

This module tests:
- All 9 transformation types (lin, log, chg, ch1, pch, pc1, pca, cch, cca)
- Frequency-aware transformations
- Polars DataFrame support
- StandardScaler integration
- Error handling
- Edge cases
"""

import pytest
import numpy as np
import polars as pl
from typing import List

from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.transformations import DFMScaler
from dfm_python.transformations.transformers import (
    identity_transform,
    log_transform,
    pch_transform,
    pc1_transform,
    pca_transform,
    cch_transform,
    cca_transform,
    FREQ_TO_LAG_YOY,
    FREQ_TO_LAG_STEP,
)


# Skip all tests if sktime is not available
pytest.importorskip("sktime", reason="sktime is required for transformation tests")


class TestTransformationFunctions:
    """Test individual transformation functions."""
    
    def test_identity_transform(self):
        """Test identity transformation."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = identity_transform(X)
        np.testing.assert_array_equal(result, X)
        
        # Test with 2D
        X2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result2d = identity_transform(X2d)
        np.testing.assert_array_equal(result2d, X2d.flatten())
    
    def test_log_transform(self):
        """Test log transformation."""
        X = np.array([1.0, 2.0, 3.0, 4.0])
        result = log_transform(X)
        expected = np.log(np.abs(X) + 1e-10)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with negative values
        X_neg = np.array([-1.0, -2.0, 3.0])
        result_neg = log_transform(X_neg)
        expected_neg = np.log(np.abs(X_neg) + 1e-10)
        np.testing.assert_array_almost_equal(result_neg, expected_neg)
    
    def test_pch_transform(self):
        """Test percent change transformation."""
        X = np.array([100.0, 110.0, 121.0, 133.1])
        result = pch_transform(X, step=1)
        
        # First value should be NaN
        assert np.isnan(result[0])
        
        # Second value: (110 - 100) / 100 * 100 = 10.0
        assert abs(result[1] - 10.0) < 1e-6
        
        # Third value: (121 - 110) / 110 * 100 ≈ 10.0
        assert abs(result[2] - 10.0) < 1e-6
    
    def test_pc1_transform(self):
        """Test year-over-year percent change transformation."""
        # Monthly data: 12 periods per year
        X = np.array([100.0] * 12 + [110.0] * 12)
        result = pc1_transform(X, year_step=12)
        
        # First 12 values should be NaN
        assert np.all(np.isnan(result[:12]))
        
        # 13th value: (110 - 100) / 100 * 100 = 10.0
        assert abs(result[12] - 10.0) < 1e-6
    
    def test_pca_transform(self):
        """Test percent change annualized transformation."""
        X = np.array([100.0, 110.0])
        result = pca_transform(X, step=1, annual_factor=12.0)
        
        # First value should be NaN
        assert np.isnan(result[0])
        
        # Second value: 12 * (110 - 100) / 100 * 100 = 120.0
        assert abs(result[1] - 120.0) < 1e-6
    
    def test_cch_transform(self):
        """Test continuously compounded rate transformation."""
        X = np.array([100.0, 110.0])
        result = cch_transform(X, step=1)
        
        # First value should be NaN
        assert np.isnan(result[0])
        
        # Second value: 100 * (log(110) - log(100)) ≈ 9.53
        expected = 100.0 * (np.log(110.0) - np.log(100.0))
        assert abs(result[1] - expected) < 1e-6
    
    def test_cca_transform(self):
        """Test continuously compounded annual rate transformation."""
        X = np.array([100.0, 110.0])
        result = cca_transform(X, step=1, annual_factor=12.0)
        
        # First value should be NaN
        assert np.isnan(result[0])
        
        # Second value: 12 * 100 * (log(110) - log(100)) ≈ 114.36
        expected = 12.0 * 100.0 * (np.log(110.0) - np.log(100.0))
        assert abs(result[1] - expected) < 1e-6


class TestDFMScalerBasic:
    """Test basic DFMScaler functionality."""
    
    def create_simple_config(self, transformations: List[str], frequencies: List[str] = None):
        """Create a simple DFMConfig for testing."""
        if frequencies is None:
            frequencies = ['m'] * len(transformations)
        
        series = [
            SeriesConfig(
                series_id=f'series_{i}',
                frequency=freq,
                transformation=trans,
                blocks=['Block_Global']
            )
            for i, (trans, freq) in enumerate(zip(transformations, frequencies))
        ]
        
        blocks = {
            'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')
        }
        
        return DFMConfig(series=series, blocks=blocks)
    
    def test_lin_transformation(self):
        """Test linear (identity) transformation."""
        config = self.create_simple_config(['lin', 'lin'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0],
            'series_1': [10.0, 20.0, 30.0, 40.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Should be Polars DataFrame
        assert isinstance(X_transformed, pl.DataFrame)
        
        # After standardization, mean should be ~0, std ~1
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(X_np.std(axis=0), 1.0, atol=1e-6)
    
    def test_log_transformation(self):
        """Test log transformation."""
        config = self.create_simple_config(['log'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        
        # Check that log was applied (before standardization)
        # Values should be log-transformed and standardized
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6)
    
    def test_chg_transformation(self):
        """Test first difference transformation."""
        config = self.create_simple_config(['chg'], ['m'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [100.0, 110.0, 121.0, 133.1]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        
        # First value should be NaN (after differencing)
        X_np = X_transformed.to_numpy()
        assert np.isnan(X_np[0, 0])
    
    def test_ch1_transformation(self):
        """Test year-over-year difference transformation."""
        config = self.create_simple_config(['ch1'], ['m'])
        scaler = DFMScaler(config)
        
        # Create 13 months of data
        X = pl.DataFrame({
            'series_0': [100.0] * 12 + [110.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        
        # First 12 values should be NaN
        X_np = X_transformed.to_numpy()
        assert np.all(np.isnan(X_np[:12, 0]))
    
    def test_pch_transformation(self):
        """Test percent change transformation."""
        config = self.create_simple_config(['pch'], ['m'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [100.0, 110.0, 121.0, 133.1]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        
        # First value should be NaN
        X_np = X_transformed.to_numpy()
        assert np.isnan(X_np[0, 0])
    
    def test_all_transformations(self):
        """Test all 9 transformation types."""
        transformations = ['lin', 'log', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca']
        frequencies = ['m', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm']
        
        config = self.create_simple_config(transformations, frequencies)
        scaler = DFMScaler(config)
        
        # Create data with enough periods for all transformations
        T = 20
        X = pl.DataFrame({
            f'series_{i}': np.random.randn(T) * 10 + 100
            for i in range(len(transformations))
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        assert X_transformed.shape == (T, len(transformations))
        
        # Check standardization
        X_np = X_transformed.to_numpy()
        # Some values may be NaN due to transformations, so check non-NaN values
        for col in range(X_np.shape[1]):
            col_data = X_np[:, col]
            non_nan = col_data[~np.isnan(col_data)]
            if len(non_nan) > 0:
                assert np.allclose(non_nan.mean(), 0.0, atol=1e-5)
                assert np.allclose(non_nan.std(), 1.0, atol=1e-5)


class TestDFMScalerFrequencyAware:
    """Test frequency-aware transformations."""
    
    def test_monthly_frequency(self):
        """Test monthly frequency transformations."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='ch1',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        # Create 13 months of data
        X = pl.DataFrame({
            'series_0': np.arange(13) * 10 + 100
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
    
    def test_quarterly_frequency(self):
        """Test quarterly frequency transformations."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='q',
                transformation='ch1',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='q')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        # Create 5 quarters of data
        X = pl.DataFrame({
            'series_0': np.arange(5) * 10 + 100
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
    
    def test_mixed_frequencies(self):
        """Test mixed frequency transformations."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='chg',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_1',
                frequency='q',
                transformation='chg',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        # Create data
        X = pl.DataFrame({
            'series_0': np.arange(12) * 10 + 100,  # Monthly
            'series_1': np.arange(4) * 10 + 100,     # Quarterly (only 4 values)
        })
        
        # For quarterly, we need to expand to monthly grid
        # This is a simplified test - actual mixed frequency handling is more complex
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)


class TestDFMScalerPolars:
    """Test Polars DataFrame support."""
    
    def test_polars_input_output(self):
        """Test that Polars DataFrame is preserved."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_1',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0],
            'series_1': [10.0, 20.0, 30.0, 40.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Should be Polars DataFrame
        assert isinstance(X_transformed, pl.DataFrame)
        
        # Column names should be preserved
        assert list(X_transformed.columns) == ['series_0', 'series_1']
    
    def test_column_ordering(self):
        """Test that column ordering is preserved."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_1',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_2',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0],
            'series_1': [10.0, 20.0, 30.0],
            'series_2': [100.0, 200.0, 300.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Column order should match config
        assert list(X_transformed.columns) == ['series_0', 'series_1', 'series_2']
    
    def test_column_reordering(self):
        """Test that columns are reordered to match config if needed."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            ),
            SeriesConfig(
                series_id='series_1',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        # Input with different column order
        X = pl.DataFrame({
            'series_1': [10.0, 20.0, 30.0],
            'series_0': [1.0, 2.0, 3.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Should be reordered to match config
        assert list(X_transformed.columns) == ['series_0', 'series_1']


class TestDFMScalerStandardization:
    """Test StandardScaler integration."""
    
    def test_standardization_applied(self):
        """Test that standardization is applied correctly."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Check standardization parameters
        assert scaler.Mx is not None
        assert scaler.Wx is not None
        
        # Check that transformed data has mean ~0 and std ~1
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(X_np.std(axis=0), 1.0, atol=1e-6)
    
    def test_standardization_parameters(self):
        """Test that Mx and Wx are accessible."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        # Before fitting, should raise error
        with pytest.raises(ValueError, match="Scaler has not been fitted"):
            _ = scaler.Mx
        
        scaler.fit(X)
        
        # After fitting, should work
        assert scaler.Mx is not None
        assert scaler.Wx is not None
        assert len(scaler.Mx) == 1
        assert len(scaler.Wx) == 1


class TestDFMScalerErrorHandling:
    """Test error handling."""
    
    def test_missing_sktime(self):
        """Test ImportError when sktime is not available."""
        # This test is skipped if sktime is not available (via importorskip)
        # But we can test the check_sktime_available function
        from dfm_python.transformations.sktime import check_sktime_available, HAS_SKTIME
        
        if not HAS_SKTIME:
            with pytest.raises(ImportError, match="sktime is required"):
                check_sktime_available()
    
    def test_invalid_transformation(self):
        """Test with invalid transformation code."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        # Invalid transformation should default to identity
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='invalid',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0]
        })
        
        # Should not raise error, defaults to identity
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
    
    def test_mismatched_columns(self):
        """Test with mismatched column names."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        # DataFrame with wrong column name
        X = pl.DataFrame({
            'wrong_name': [1.0, 2.0, 3.0]
        })
        
        # Should raise error or handle gracefully
        with pytest.raises((KeyError, ValueError)):
            scaler.fit_transform(X)


class TestDFMScalerEdgeCases:
    """Test edge cases."""
    
    def test_single_series(self):
        """Test with single series."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
        assert X_transformed.shape[1] == 1
    
    def test_single_time_period(self):
        """Test with single time period."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0]
        })
        
        # Single period may cause issues with some transformations
        # But should not crash
        try:
            X_transformed = scaler.fit_transform(X)
            assert isinstance(X_transformed, pl.DataFrame)
        except (ValueError, RuntimeError):
            # Some transformations may not work with single period
            pass
    
    def test_missing_data(self):
        """Test with missing data (NaN values)."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, np.nan, 3.0, 4.0, np.nan]
        })
        
        # Should handle NaN values
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame)
    
    def test_fit_transform_separate(self):
        """Test fit and transform as separate steps."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X_train = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        X_test = pl.DataFrame({
            'series_0': [6.0, 7.0, 8.0]
        })
        
        # Fit on training data
        scaler.fit(X_train)
        
        # Transform test data
        X_test_transformed = scaler.transform(X_test)
        assert isinstance(X_test_transformed, pl.DataFrame)
        assert X_test_transformed.shape[0] == 3


class TestDFMScalerInverseTransform:
    """Test inverse transform functionality."""
    
    def test_inverse_transform_lin(self):
        """Test inverse transform for linear transformation."""
        from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
        
        series = [
            SeriesConfig(
                series_id='series_0',
                frequency='m',
                transformation='lin',
                blocks=['Block_Global']
            )
        ]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series, blocks=blocks)
        
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_transformed)
        
        assert isinstance(X_inverse, pl.DataFrame)
        # Inverse should approximately recover original (within numerical precision)
        np.testing.assert_array_almost_equal(
            X_inverse.to_numpy(),
            X.to_numpy(),
            decimal=5
        )


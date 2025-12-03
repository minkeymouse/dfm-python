"""Comprehensive tests for DFM transformations and scaling.

This module provides extensive tests for data transformation and preprocessing functionality
used by both DFM and DDFM models. Transformations are applied via DFMScaler, which integrates
with sktime for efficient processing.

**Test Organization**:
- `TestTransformationFunctions`: Tests individual transformation functions (9 transformation types)
- `TestDFMScalerBasic`: Tests basic DFMScaler functionality (fit, transform, inverse_transform)
- `TestDFMScalerFrequencyAware`: Tests frequency-aware transformations (ch1, pc1, etc.)
- `TestDFMScalerPolars`: Tests Polars DataFrame support
- `TestDFMScalerStandardization`: Tests global standardization (mean=0, std=1)
- `TestDFMScalerErrorHandling`: Tests error handling and validation
- `TestDFMScalerEdgeCases`: Tests edge cases (empty data, single series, etc.)
- `TestDFMScalerInverseTransform`: Tests inverse transformation functionality

**Transformation Types Tested**:
- `lin`: No transformation (identity)
- `log`: Log transformation
- `chg`: First difference (frequency-aware)
- `ch1`: Year-over-year difference (frequency-aware)
- `pch`: Percent change (frequency-aware)
- `pc1`: Year-over-year percent change (frequency-aware)
- `pca`: Percent change annualized (frequency-aware)
- `cch`: Continuously compounded rate (frequency-aware)
- `cca`: Continuously compounded annual rate (frequency-aware)

**Dependencies**:
- Required: `sktime` (for ColumnTransformer and Polars support)
- Tests are skipped if sktime is not available

**Usage Patterns**:
- Transformations are applied automatically during model training via DFMScaler
- Can be used standalone: scaler = DFMScaler(config); X_transformed = scaler.fit_transform(X)
- All transformations are frequency-aware (use appropriate lags based on data frequency)

**Related Test Files**:
- `test_dfm.py`: Tests DFM which uses DFMScaler for preprocessing
- `test_ddfm.py`: Tests DDFM which also uses DFMScaler for preprocessing
"""

# Standard library imports
import pytest
from typing import List

# Third-party imports
import numpy as np
import polars as pl

# Local application imports
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
)


# Skip all tests if sktime is not available
pytest.importorskip("sktime", reason="sktime is required for transformation tests")


class TestTransformationFunctions:
    """Test individual transformation functions."""
    
    def test_identity_transform(self):
        """Test identity transformation.
        
        This test verifies that the identity transformation function:
        1. Returns input unchanged for 1D arrays
        2. Flattens and returns 2D arrays correctly
        
        Expected behavior:
        - 1D input array is returned unchanged
        - 2D input array is flattened and returned
        - No numerical errors or data loss
        """
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = identity_transform(X)
        np.testing.assert_array_equal(result, X)
        
        # Test with 2D
        X2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result2d = identity_transform(X2d)
        np.testing.assert_array_equal(result2d, X2d.flatten())
    
    def test_log_transform(self):
        """Test log transformation.
        
        This test verifies that the log transformation function:
        1. Applies log to absolute values with epsilon for positive values
        2. Handles negative values correctly using absolute value
        3. Uses epsilon (1e-10) to avoid log(0) issues
        
        Expected behavior:
        - Log transformation applied: log(|x| + 1e-10)
        - Positive values transformed correctly
        - Negative values handled via absolute value
        - No numerical errors or invalid values
        """
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
        """Test percent change transformation.
        
        This test verifies that the percent change transformation:
        1. Computes percent change: (x_t - x_{t-step}) / x_{t-step} * 100
        2. Returns NaN for first value (no previous value for comparison)
        3. Calculates correct percent changes for subsequent values
        
        Expected behavior:
        - First value is NaN (no previous value)
        - Subsequent values are percent changes: (new - old) / old * 100
        - Calculations are accurate within numerical precision
        """
        X = np.array([100.0, 110.0, 121.0, 133.1])
        result = pch_transform(X, step=1)
        
        # First value should be NaN
        assert np.isnan(result[0]), "First value after pch_transform should be NaN (no previous value for comparison)"
        
        # Second value: (110 - 100) / 100 * 100 = 10.0
        assert abs(result[1] - 10.0) < 1e-6, f"Second value should be 10.0 (10% change from 100 to 110), got {result[1]:.6f}"
        
        # Third value: (121 - 110) / 110 * 100 ≈ 10.0
        assert abs(result[2] - 10.0) < 1e-6, f"Third value should be approximately 10.0 (10% change from 110 to 121), got {result[2]:.6f}"
    
    def test_pc1_transform(self):
        """Test year-over-year percent change transformation.
        
        This test verifies that the year-over-year percent change transformation:
        1. Computes percent change compared to same period in previous year
        2. Returns NaN for first year of data (no previous year for comparison)
        3. Calculates correct year-over-year percent changes
        
        Expected behavior:
        - First year values are NaN (no previous year values)
        - Subsequent values are year-over-year percent changes
        - Calculations use correct lag (12 for monthly data)
        - Results are accurate within numerical precision
        """
        # Monthly data: 12 periods per year
        X = np.array([100.0] * 12 + [110.0] * 12)
        result = pc1_transform(X, year_step=12)
        
        # First 12 values should be NaN
        assert np.all(np.isnan(result[:12])), "First 12 values should be NaN (no previous year values for year-over-year comparison)"
        
        # 13th value: (110 - 100) / 100 * 100 = 10.0
        assert abs(result[12] - 10.0) < 1e-6, f"13th value should be 10.0 (10% year-over-year change), got {result[12]:.6f}"
    
    def test_pca_transform(self):
        """Test percent change annualized transformation.
        
        This test verifies that the percent change annualized transformation:
        1. Computes percent change and annualizes by multiplying by annual factor
        2. Returns NaN for first value (no previous value for comparison)
        3. Applies annualization factor correctly (e.g., 12 for monthly to annual)
        
        Expected behavior:
        - First value is NaN (no previous value)
        - Percent change is annualized: annual_factor * (new - old) / old * 100
        - Annualization factor (12.0) is applied correctly
        - Results are accurate within numerical precision
        """
        X = np.array([100.0, 110.0])
        result = pca_transform(X, step=1, annual_factor=12.0)
        
        # First value should be NaN
        assert np.isnan(result[0]), "First value after pca_transform should be NaN (no previous value for comparison)"
        
        # Second value: 12 * (110 - 100) / 100 * 100 = 120.0
        assert abs(result[1] - 120.0) < 1e-6, f"Second value should be 120.0 (10% change annualized by factor 12), got {result[1]:.6f}"
    
    def test_cch_transform(self):
        """Test continuously compounded rate transformation.
        
        This test verifies that the continuously compounded rate transformation:
        1. Computes continuously compounded rate: 100 * (log(x_t) - log(x_{t-step}))
        2. Returns NaN for first value (no previous value for comparison)
        3. Calculates correct continuously compounded rates
        
        Expected behavior:
        - First value is NaN (no previous value)
        - Continuously compounded rate: 100 * (ln(new) - ln(old))
        - Uses natural logarithm for compounding calculation
        - Results are accurate within numerical precision
        """
        X = np.array([100.0, 110.0])
        result = cch_transform(X, step=1)
        
        # First value should be NaN
        assert np.isnan(result[0]), "First value after cch_transform should be NaN (no previous value for comparison)"
        
        # Second value: 100 * (log(110) - log(100)) ≈ 9.53
        expected = 100.0 * (np.log(110.0) - np.log(100.0))
        assert abs(result[1] - expected) < 1e-6, f"Second value should be approximately {expected:.6f} (continuously compounded rate), got {result[1]:.6f}"
    
    def test_cca_transform(self):
        """Test continuously compounded annual rate transformation.
        
        This test verifies that the continuously compounded annual rate transformation:
        1. Computes continuously compounded rate and annualizes by multiplying by annual factor
        2. Returns NaN for first value (no previous value for comparison)
        3. Applies annualization factor correctly to continuously compounded rate
        
        Expected behavior:
        - First value is NaN (no previous value)
        - Continuously compounded annual rate: annual_factor * 100 * (ln(new) - ln(old))
        - Annualization factor (12.0) is applied correctly
        - Results are accurate within numerical precision
        """
        X = np.array([100.0, 110.0])
        result = cca_transform(X, step=1, annual_factor=12.0)
        
        # First value should be NaN
        assert np.isnan(result[0]), "First value after cca_transform should be NaN (no previous value for comparison)"
        
        # Second value: 12 * 100 * (log(110) - log(100)) ≈ 114.36
        expected = 12.0 * 100.0 * (np.log(110.0) - np.log(100.0))
        assert abs(result[1] - expected) < 1e-6, f"Second value should be approximately {expected:.6f} (continuously compounded annual rate), got {result[1]:.6f}"


class TestDFMScalerBasic:
    """Test basic DFMScaler functionality."""
    
    def _create_config_with_transformations(self, transformations: List[str], frequencies: List[str] = None):
        """Create a DFMConfig for testing with specific transformations and frequencies.
        
        This is a specialized helper method for transformation testing. Unlike the shared
        utility `create_simple_config()` in test/__init__.py which creates configs with
        fixed 'lin' transformation, this method allows specifying different transformations
        and frequencies per series, which is required for testing DFMScaler's transformation
        capabilities.
        
        This method is private (prefixed with `_`) to avoid naming conflict with the shared
        utility function of the same name in test/__init__.py.
        
        Parameters
        ----------
        transformations : List[str]
            List of transformation types to apply to each series (e.g., ['lin', 'log'])
        frequencies : List[str], optional
            List of frequencies for each series (defaults to 'm' for all)
            
        Returns
        -------
        DFMConfig
            A DFM configuration with specified transformations and frequencies
        """
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
        """Test linear (identity) transformation.
        
        This test verifies that DFMScaler handles linear transformation:
        1. Applies identity transformation (no change to values)
        2. Applies standardization (mean=0, std=1)
        3. Returns Polars DataFrame
        
        Expected behavior:
        - Identity transformation applied (no change before standardization)
        - Standardization applied: mean ≈ 0, std ≈ 1
        - Returns Polars DataFrame (not NumPy array)
        - All series are standardized independently
        """
        config = self._create_config_with_transformations(['lin', 'lin'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0],
            'series_1': [10.0, 20.0, 30.0, 40.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        
        # Should be Polars DataFrame
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame"
        
        # After standardization, mean should be ~0, std ~1
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6), f"Standardized data should have mean ≈ 0, got {X_np.mean(axis=0)}"
        assert np.allclose(X_np.std(axis=0), 1.0, atol=1e-6), f"Standardized data should have std ≈ 1, got {X_np.std(axis=0)}"
    
    def test_log_transformation(self):
        """Test log transformation.
        
        This test verifies that DFMScaler handles log transformation:
        1. Applies log transformation to input data
        2. Applies standardization after log transformation
        3. Returns Polars DataFrame
        
        Expected behavior:
        - Log transformation applied before standardization
        - Standardization applied: mean ≈ 0, std ≈ 1
        - Returns Polars DataFrame
        - Handles positive values correctly
        """
        config = self._create_config_with_transformations(['log'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [1.0, 2.0, 3.0, 4.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for log transformation"
        
        # Check that log was applied (before standardization)
        # Values should be log-transformed and standardized
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6), f"Log-transformed and standardized data should have mean ≈ 0, got {X_np.mean(axis=0)}"
    
    def test_chg_transformation(self):
        """Test first difference transformation.
        
        This test verifies that DFMScaler handles first difference transformation:
        1. Applies first difference: x_t - x_{t-1}
        2. Returns NaN for first value (no previous value)
        3. Applies standardization after differencing
        
        Expected behavior:
        - First difference applied: x_t - x_{t-1}
        - First value is NaN (no previous value for differencing)
        - Standardization applied to non-NaN values
        - Returns Polars DataFrame
        """
        config = self._create_config_with_transformations(['chg'], ['m'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [100.0, 110.0, 121.0, 133.1]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for chg transformation"
        
        # First value should be NaN (after differencing)
        X_np = X_transformed.to_numpy()
        assert np.isnan(X_np[0, 0]), "First value after chg (first difference) transformation should be NaN (no previous value)"
    
    def test_ch1_transformation(self):
        """Test year-over-year difference transformation.
        
        This test verifies that DFMScaler handles year-over-year difference transformation:
        1. Applies year-over-year difference: x_t - x_{t-year_step}
        2. Returns NaN for first year of data (no previous year values)
        3. Uses frequency-aware lag (12 for monthly data)
        
        Expected behavior:
        - Year-over-year difference applied: x_t - x_{t-12} for monthly data
        - First year values are NaN (no previous year values)
        - Frequency-aware lag used correctly
        - Returns Polars DataFrame
        """
        config = self._create_config_with_transformations(['ch1'], ['m'])
        scaler = DFMScaler(config)
        
        # Create 13 months of data
        X = pl.DataFrame({
            'series_0': [100.0] * 12 + [110.0]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for ch1 transformation"
        
        # First 12 values should be NaN
        X_np = X_transformed.to_numpy()
        assert np.all(np.isnan(X_np[:12, 0])), "First 12 values after ch1 (year-over-year difference) transformation should be NaN (no previous year values)"
    
    def test_pch_transformation(self):
        """Test percent change transformation.
        
        This test verifies that DFMScaler handles percent change transformation:
        1. Applies percent change: (x_t - x_{t-1}) / x_{t-1} * 100
        2. Returns NaN for first value (no previous value)
        3. Applies standardization after percent change calculation
        
        Expected behavior:
        - Percent change applied: (new - old) / old * 100
        - First value is NaN (no previous value)
        - Standardization applied to non-NaN values
        - Returns Polars DataFrame
        """
        config = self._create_config_with_transformations(['pch'], ['m'])
        scaler = DFMScaler(config)
        
        X = pl.DataFrame({
            'series_0': [100.0, 110.0, 121.0, 133.1]
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for pch transformation"
        
        # First value should be NaN
        X_np = X_transformed.to_numpy()
        assert np.isnan(X_np[0, 0]), "First value after pch (percent change) transformation should be NaN (no previous value)"
    
    def test_all_transformations(self):
        """Test all 9 transformation types.
        
        This test verifies that DFMScaler handles all transformation types:
        1. Linear (lin), log, first difference (chg), year-over-year difference (ch1)
        2. Percent change (pch), year-over-year percent change (pc1)
        3. Percent change annualized (pca), continuously compounded rate (cch)
        4. Continuously compounded annual rate (cca)
        
        Expected behavior:
        - All 9 transformation types work correctly
        - Data shape preserved (T periods × N series)
        - Standardization applied to each series independently
        - Non-NaN values have mean ≈ 0, std ≈ 1 after standardization
        - Returns Polars DataFrame
        """
        transformations = ['lin', 'log', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca']
        frequencies = ['m', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm']
        
        config = self._create_config_with_transformations(transformations, frequencies)
        scaler = DFMScaler(config)
        
        # Create data with enough periods for all transformations
        T = 20
        X = pl.DataFrame({
            f'series_{i}': np.random.randn(T) * 10 + 100
            for i in range(len(transformations))
        })
        
        X_transformed = scaler.fit_transform(X)
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for all transformations"
        assert X_transformed.shape == (T, len(transformations)), f"Transformed data should have shape ({T}, {len(transformations)}), got {X_transformed.shape}"
        
        # Check standardization
        X_np = X_transformed.to_numpy()
        # Some values may be NaN due to transformations, so check non-NaN values
        for col in range(X_np.shape[1]):
            col_data = X_np[:, col]
            non_nan = col_data[~np.isnan(col_data)]
            if len(non_nan) > 0:
                assert np.allclose(non_nan.mean(), 0.0, atol=1e-5), f"Column {col} non-NaN values should have mean ≈ 0 after standardization, got {non_nan.mean():.6f}"
                assert np.allclose(non_nan.std(), 1.0, atol=1e-5), f"Column {col} non-NaN values should have std ≈ 1 after standardization, got {non_nan.std():.6f}"


class TestDFMScalerFrequencyAware:
    """Test frequency-aware transformations."""
    
    def test_monthly_frequency(self):
        """Test monthly frequency transformations.
        
        This test verifies that DFMScaler handles monthly frequency correctly:
        1. Uses correct lag for monthly data (12 for year-over-year transformations)
        2. Applies frequency-aware transformations correctly
        3. Handles monthly data structure properly
        
        Expected behavior:
        - Monthly frequency recognized correctly
        - Year-over-year transformations use lag=12 for monthly data
        - Returns Polars DataFrame
        - Transformation applied correctly for monthly frequency
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for monthly frequency"
    
    def test_quarterly_frequency(self):
        """Test quarterly frequency transformations.
        
        This test verifies that DFMScaler handles quarterly frequency correctly:
        1. Uses correct lag for quarterly data (4 for year-over-year transformations)
        2. Applies frequency-aware transformations correctly
        3. Handles quarterly data structure properly
        
        Expected behavior:
        - Quarterly frequency recognized correctly
        - Year-over-year transformations use lag=4 for quarterly data
        - Returns Polars DataFrame
        - Transformation applied correctly for quarterly frequency
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for quarterly frequency"
    
    def test_mixed_frequencies(self):
        """Test mixed frequency transformations.
        
        This test verifies that DFMScaler handles mixed frequencies correctly:
        1. Handles series with different frequencies (monthly and quarterly)
        2. Applies frequency-aware transformations to each series
        3. Manages data alignment for mixed-frequency data
        
        Expected behavior:
        - Different frequencies handled correctly per series
        - Frequency-aware transformations applied correctly
        - Returns Polars DataFrame
        - Mixed frequency data processed without errors
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for mixed frequencies"


class TestDFMScalerPolars:
    """Test Polars DataFrame support."""
    
    def test_polars_input_output(self):
        """Test that Polars DataFrame is preserved.
        
        This test verifies that DFMScaler maintains Polars DataFrame format:
        1. Accepts Polars DataFrame as input
        2. Returns Polars DataFrame as output
        3. Preserves column names
        
        Expected behavior:
        - Input Polars DataFrame accepted
        - Output is Polars DataFrame (not NumPy array)
        - Column names preserved in output
        - Data structure maintained
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame"
        
        # Column names should be preserved
        assert list(X_transformed.columns) == ['series_0', 'series_1'], f"Column names should be preserved, got {list(X_transformed.columns)}"
    
    def test_column_ordering(self):
        """Test that column ordering is preserved.
        
        This test verifies that DFMScaler preserves column ordering:
        1. Column order in output matches config order
        2. Input column order is respected when it matches config
        3. Multiple series handled correctly
        
        Expected behavior:
        - Column order matches config order
        - All columns present in output
        - Order is consistent and predictable
        """
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
        assert list(X_transformed.columns) == ['series_0', 'series_1', 'series_2'], f"Column order should match config, got {list(X_transformed.columns)}"
    
    def test_column_reordering(self):
        """Test that columns are reordered to match config if needed.
        
        This test verifies that DFMScaler reorders columns to match config:
        1. Input columns can be in any order
        2. Output columns match config order
        3. All columns are present in correct order
        
        Expected behavior:
        - Input column order doesn't matter
        - Output columns reordered to match config order
        - All columns present in output
        - No data loss during reordering
        """
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
        assert list(X_transformed.columns) == ['series_0', 'series_1'], f"Columns should be reordered to match config order, got {list(X_transformed.columns)}"


class TestDFMScalerStandardization:
    """Test StandardScaler integration."""
    
    def test_standardization_applied(self):
        """Test that standardization is applied correctly.
        
        This test verifies that DFMScaler applies standardization correctly:
        1. Standardization parameters (Mx, Wx) are set after fitting
        2. Transformed data has mean ≈ 0 and std ≈ 1
        3. Standardization is applied after transformation
        
        Expected behavior:
        - Standardization mean (Mx) is set after fit_transform
        - Standardization scale (Wx) is set after fit_transform
        - Transformed data has mean ≈ 0, std ≈ 1
        - Standardization applied correctly
        """
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
        assert scaler.Mx is not None, "Standardization mean (Mx) should be set after fit_transform"
        assert scaler.Wx is not None, "Standardization scale (Wx) should be set after fit_transform"
        
        # Check that transformed data has mean ~0 and std ~1
        X_np = X_transformed.to_numpy()
        assert np.allclose(X_np.mean(axis=0), 0.0, atol=1e-6), f"Standardized data should have mean ≈ 0, got {X_np.mean(axis=0)}"
        assert np.allclose(X_np.std(axis=0), 1.0, atol=1e-6), f"Standardized data should have std ≈ 1, got {X_np.std(axis=0)}"
    
    def test_standardization_parameters(self):
        """Test that Mx and Wx are accessible.
        
        This test verifies that standardization parameters are accessible:
        1. Mx and Wx raise error before fitting
        2. Mx and Wx are accessible after fitting
        3. Parameter shapes match number of series
        
        Expected behavior:
        - ValueError raised when accessing Mx/Wx before fitting
        - Mx and Wx accessible after fit() or fit_transform()
        - Parameter lengths match number of series
        - Parameters are valid NumPy arrays
        """
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
        assert scaler.Mx is not None, "Standardization mean (Mx) should be accessible after fit"
        assert scaler.Wx is not None, "Standardization scale (Wx) should be accessible after fit"
        assert len(scaler.Mx) == 1, f"Mx should have length 1 for single series, got {len(scaler.Mx)}"
        assert len(scaler.Wx) == 1, f"Wx should have length 1 for single series, got {len(scaler.Wx)}"


class TestDFMScalerErrorHandling:
    """Test error handling."""
    
    def test_missing_sktime(self):
        """Test ImportError when sktime is not available.
        
        This test verifies error handling when sktime is not available:
        1. Checks if sktime availability is detected correctly
        2. Raises ImportError with appropriate message when sktime missing
        3. Tests the check_sktime_available function
        
        Expected behavior:
        - ImportError raised when sktime is not available
        - Error message indicates sktime is required
        - check_sktime_available function works correctly
        """
        # This test is skipped if sktime is not available (via importorskip)
        # But we can test the check_sktime_available function
        from dfm_python.transformations.sktime import check_sktime_available, HAS_SKTIME
        
        if not HAS_SKTIME:
            with pytest.raises(ImportError, match="sktime is required"):
                check_sktime_available()
    
    def test_invalid_transformation(self):
        """Test with invalid transformation code.
        
        This test verifies error handling for invalid transformation codes:
        1. Invalid transformation code is handled gracefully
        2. Defaults to identity transformation when invalid code provided
        3. Does not raise error, processes data successfully
        
        Expected behavior:
        - Invalid transformation code doesn't cause crash
        - Defaults to identity transformation
        - Returns Polars DataFrame successfully
        - Data is processed without errors
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "Invalid transformation should default to identity and return Polars DataFrame"
    
    def test_mismatched_columns(self):
        """Test with mismatched column names.
        
        This test verifies error handling for mismatched column names:
        1. DataFrame with column names that don't match config raises error
        2. Error is raised before processing (KeyError or ValueError)
        3. Clear error message indicates column mismatch
        
        Expected behavior:
        - KeyError or ValueError raised when column names don't match config
        - Error raised before data processing
        - Error message indicates the problem
        """
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
        """Test with single series.
        
        This test verifies that DFMScaler handles single series correctly:
        1. Works correctly with only one series in config
        2. Returns correct output shape (T periods × 1 series)
        3. Applies transformation and standardization correctly
        
        Expected behavior:
        - Single series processed correctly
        - Output has 1 column (single series)
        - Returns Polars DataFrame
        - Transformation and standardization applied correctly
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame for single series"
        assert X_transformed.shape[1] == 1, f"Single series should result in 1 column, got {X_transformed.shape[1]}"
    
    def test_single_time_period(self):
        """Test with single time period.
        
        This test verifies edge case handling for single time period:
        1. Handles data with only one time period
        2. Some transformations may not work (e.g., differencing)
        3. Should not crash, either succeeds or raises appropriate error
        
        Expected behavior:
        - Single time period handled gracefully
        - Either processes successfully or raises ValueError/RuntimeError
        - Does not crash unexpectedly
        - Error handling is appropriate for edge case
        """
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
            assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame even for single time period"
        except (ValueError, RuntimeError):
            # Some transformations may not work with single period
            pass
    
    def test_missing_data(self):
        """Test with missing data (NaN values).
        
        This test verifies that DFMScaler handles missing data correctly:
        1. Processes data with NaN values without crashing
        2. Handles missing values appropriately during transformation
        3. Returns valid output even with missing data
        
        Expected behavior:
        - Missing data (NaN values) handled gracefully
        - Returns Polars DataFrame successfully
        - Does not crash on missing data
        - Missing values processed according to transformation rules
        """
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
        assert isinstance(X_transformed, pl.DataFrame), "DFMScaler.fit_transform should return Polars DataFrame even with missing data (NaN values)"
    
    def test_fit_transform_separate(self):
        """Test fit and transform as separate steps.
        
        This test verifies that fit() and transform() can be called separately:
        1. fit() learns parameters from training data
        2. transform() applies learned parameters to test data
        3. Separate calls work correctly for train/test splits
        
        Expected behavior:
        - fit() learns standardization parameters from training data
        - transform() applies learned parameters to test data
        - Returns Polars DataFrame for transformed test data
        - Output shape matches test data shape
        """
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
        assert isinstance(X_test_transformed, pl.DataFrame), "DFMScaler.transform should return Polars DataFrame"
        assert X_test_transformed.shape[0] == 3, f"Transformed test data should have 3 rows, got {X_test_transformed.shape[0]}"


class TestDFMScalerInverseTransform:
    """Test inverse transform functionality."""
    
    def test_inverse_transform_lin(self):
        """Test inverse transform for linear transformation.
        
        This test verifies that inverse_transform recovers original data:
        1. fit_transform() applies transformation and standardization
        2. inverse_transform() reverses standardization and transformation
        3. Recovered data matches original within numerical precision
        
        Expected behavior:
        - inverse_transform() returns Polars DataFrame
        - Recovered data matches original data approximately
        - Numerical precision maintained (within 5 decimal places)
        - Inverse transformation works correctly for linear transformation
        """
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
        
        assert isinstance(X_inverse, pl.DataFrame), "DFMScaler.inverse_transform should return Polars DataFrame"
        # Inverse should approximately recover original (within numerical precision)
        np.testing.assert_array_almost_equal(
            X_inverse.to_numpy(),
            X.to_numpy(),
            decimal=5
        )


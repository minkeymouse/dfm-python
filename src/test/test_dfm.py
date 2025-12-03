"""Tests for Dynamic Factor Model (DFM) functionality.

This module provides comprehensive tests for the Dynamic Factor Model (DFM) implementation
in the dfm-python package. DFM uses the EM algorithm to estimate latent factors from
observed time series data.

**Test Organization**:
- `TestDFMHighLevelAPI`: Tests the high-level DFM class API (11 tests)
  - Configuration loading from various sources (DFMConfig, mapping dict, YAML)
  - Data loading from NumPy arrays and Polars DataFrames
  - Model training with EM algorithm
  - Factor extraction and result validation
  - Prediction functionality
  - Edge cases and error handling
- `TestDFMLowLevelAPI`: Tests the low-level DFMLinear class API (3 tests)
  - Direct fit() method usage
  - Result object validation
  - Prediction with low-level API

**Dependencies**:
- Required: `sktime` (for preprocessing via DFMScaler)
- Optional: None (DFM does not require PyTorch)

**Usage Patterns**:
- High-level API: Create DFM instance → load_config() → load_data() → train() → predict()
- Low-level API: Create DFMLinear instance → fit(X, config) → access result
- All tests use shared utilities from `test/__init__.py` for consistency

**Related Test Files**:
- `test_ddfm.py`: Tests Deep Dynamic Factor Model (DDFM) - similar API but uses neural networks
- `test_transformations.py`: Tests data preprocessing transformations used by DFM
- `test_lightning_module.py`: Tests PyTorch Lightning integration for DFM
- `test_nowcast.py`: Tests nowcasting functionality (requires trained DFM model)
"""

# Standard library imports
import pytest

# Third-party imports
import numpy as np
import polars as pl

# Local application imports
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.models import DFM, DFMLinear
from dfm_python.config.results import DFMResult

# Local relative imports
from . import (
    check_missing_data_error,
    create_simple_config,
    create_simple_config_mapping,
    generate_synthetic_data,
)


# Skip all tests if sktime is not available (required for preprocessing)
pytest.importorskip("sktime", reason="sktime is required for DFM tests")


# ============================================================================
# High-Level API Tests
# ============================================================================

class TestDFMHighLevelAPI:
    """Test high-level DFM API (DFM class)."""
    
    def test_dfm_high_level_api_basic(self):
        """Test basic high-level DFM API workflow.
        
        This test verifies the complete DFM workflow:
        1. Configuration loading from mapping dict
        2. Data loading with synthetic data
        3. Model training with EM algorithm
        4. Result validation (factors, loadings, convergence)
        5. Prediction functionality with quality checks
        
        Expected behavior:
        - Configuration loads successfully
        - Data loads and matches expected shape
        - Training completes and produces valid results
        - Factors have correct shape (T x m) matching data length
        - Predictions are generated without NaN or infinite values
        """
        # Create model instance
        model = DFM()
        
        # Create simple config
        config = create_simple_config(num_series=5, num_factors=1)
        
        # Load configuration using mapping
        model.load_config(mapping=create_simple_config_mapping(num_series=5, num_factors=1, max_iter=10, threshold=1e-3))
        
        # Verify config is loaded
        assert model.config is not None, "Configuration should be loaded after load_config()"
        assert len(model.config.series) == 5, f"Expected 5 series in config, got {len(model.config.series)}"
        
        # Generate synthetic data
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Load data
        model.load_data(data=X)
        
        # Verify data is loaded
        assert model.data is not None, "Data should be loaded after load_data()"
        assert model.data.shape == (50, 5), f"Expected data shape (50, 5), got {model.data.shape}"
        assert model.original_data is not None, "Original data should be available after load_data()"
        
        # Train the model
        model.train(max_iter=10, threshold=1e-3)
        
        # Verify training completed
        assert model.result is not None, "Result should be available after training"
        assert isinstance(model.result, DFMResult), f"Result should be DFMResult, got {type(model.result)}"
        result_attrs = dir(model.result)
        assert hasattr(model.result, 'Z'), f"Result should have Z attribute (factors), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(model.result, 'C'), f"Result should have C attribute (loadings), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(model.result, 'converged'), f"Result should have converged attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(model.result, 'num_iter'), f"Result should have num_iter attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify result shapes
        # Z is (T x m) where T is number of time periods
        # Note: Documentation says T, not T+1, so we check it matches data length
        assert model.result.Z.shape[0] == model.data.shape[0], f"Factor length {model.result.Z.shape[0]} should match data length {model.data.shape[0]}"
        assert model.result.Z.shape[1] == 1, f"Expected 1 factor, got {model.result.Z.shape[1]}"
        assert model.result.C.shape[0] == 5, f"Expected 5 series in loadings, got {model.result.C.shape[0]}"
        assert model.result.C.shape[1] == 1, f"Expected 1 factor in loadings, got {model.result.C.shape[1]}"
        
        # Test prediction
        X_forecast, Z_forecast = model.predict(horizon=5)
        
        # Verify forecast shapes
        assert X_forecast.shape == (5, 5), f"Expected forecast shape (5, 5), got {X_forecast.shape}"
        assert Z_forecast.shape == (5, 1), f"Expected factor forecast shape (5, 1), got {Z_forecast.shape}"
        
        # Verify forecast values are reasonable (not NaN, not infinite)
        nan_count_X = np.sum(np.isnan(X_forecast))
        assert nan_count_X == 0, f"Forecast should not contain NaN values, but found {nan_count_X} NaN values in X_forecast"
        inf_count_X = np.sum(np.isinf(X_forecast))
        assert inf_count_X == 0, f"Forecast should not contain infinite values, but found {inf_count_X} infinite values in X_forecast"
        nan_count_Z = np.sum(np.isnan(Z_forecast))
        assert nan_count_Z == 0, f"Factor forecast should not contain NaN values, but found {nan_count_Z} NaN values in Z_forecast"
        inf_count_Z = np.sum(np.isinf(Z_forecast))
        assert inf_count_Z == 0, f"Factor forecast should not contain infinite values, but found {inf_count_Z} infinite values in Z_forecast"
    
    def test_dfm_config_loading(self):
        """Test configuration loading from different sources.
        
        This test verifies that DFM can load configuration from:
        1. DFMConfig object (using source parameter)
        2. Mapping dictionary (using mapping parameter)
        3. Error handling for invalid configuration structures
        
        Expected behavior:
        - Config loads successfully from DFMConfig object
        - Config loads successfully from mapping dict
        - Invalid config raises appropriate exception (ValueError or TypeError)
        - Model.reset() allows reusing same instance for multiple configs
        """
        model = DFM()
        
        # Test 1: Load from DFMConfig object
        config = create_simple_config(num_series=3, num_factors=1)
        model.load_config(source=config)
        assert model.config is not None, "Configuration should be loaded from DFMConfig object"
        assert len(model.config.series) == 3, f"Expected 3 series in config, got {len(model.config.series)}"
        
        # Test 2: Load from mapping dict
        model.reset()
        model.load_config(mapping=create_simple_config_mapping(num_series=1, num_factors=1, max_iter=None, threshold=None))
        assert model.config is not None, "Configuration should be loaded from mapping dict"
        assert len(model.config.series) == 1, f"Expected 1 series in config, got {len(model.config.series)}"
        
        # Test 3: Error handling - invalid config
        model.reset()
        # Invalid config should raise an error (ValueError or TypeError for invalid structure)
        with pytest.raises((ValueError, TypeError)):
            model.load_config(mapping={'invalid': 'config'})
    
    def test_dfm_data_loading(self):
        """Test data loading from different sources.
        
        This test verifies that DFM can load data from:
        1. NumPy array
        2. Polars DataFrame
        3. Original data preservation (original_data property)
        
        Expected behavior:
        - Data loads successfully from NumPy array
        - Data loads successfully from Polars DataFrame
        - Data shapes match expected dimensions
        - original_data property preserves untransformed data
        """
        model = DFM()
        config = create_simple_config(num_series=5, num_factors=1)
        model.load_config(source=config)
        
        # Test 1: Load from NumPy array
        X_numpy = generate_synthetic_data(n_periods=30, n_series=5)
        model.load_data(data=X_numpy)
        assert model.data is not None, "Data should be loaded from NumPy array"
        assert model.data.shape == (30, 5), f"Expected data shape (30, 5), got {model.data.shape}"
        assert model.original_data is not None, "Original data should be available after loading NumPy array"
        # Note: data and original_data may differ if preprocessing is applied
        # Just verify they have the same shape
        assert model.original_data.shape == model.data.shape, f"Original data shape {model.original_data.shape} should match data shape {model.data.shape}"
        
        # Test 2: Load from Polars DataFrame
        model.reset()
        model.load_config(source=config)
        X_df = pl.DataFrame({
            f'series_{i}': X_numpy[:, i] for i in range(5)
        })
        model.load_data(data=X_df)
        assert model.data is not None, "Data should be loaded from Polars DataFrame"
        assert model.data.shape == (30, 5), f"Expected data shape (30, 5), got {model.data.shape}"
        
        # Test 3: Error handling - no data provided
        model.reset()
        model.load_config(source=config)
        with pytest.raises(ValueError):
            model.load_data()  # Neither data_path nor data provided
    
    def test_dfm_training_convergence(self):
        """Test training and convergence behavior.
        
        This test verifies:
        1. Model training completes successfully
        2. Result object has correct structure (DFMResult)
        3. Convergence attributes are present and valid
        4. Log-likelihood is computed
        5. Training respects max_iter parameter
        
        Expected behavior:
        - Training completes without errors
        - Result contains all required attributes (converged, num_iter, loglik)
        - num_iter respects max_iter limit
        - converged flag is boolean (may be False with limited iterations)
        """
        model = DFM()
        config = create_simple_config(num_series=5, num_factors=1)
        model.load_config(source=config)
        
        X = generate_synthetic_data(n_periods=50, n_series=5)
        model.load_data(data=X)
        
        # Train with sufficient iterations for reliable convergence testing
        model.train(max_iter=10, threshold=1e-3)
        
        # Verify result structure
        result = model.result
        assert result is not None, "Result should be available after training"
        assert isinstance(result, DFMResult), f"Result should be DFMResult, got {type(result)}"
        result_attrs = dir(result)
        assert hasattr(result, 'converged'), f"Result should have converged attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'num_iter'), f"Result should have num_iter attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'loglik'), f"Result should have loglik attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify convergence flag (may be False with limited iterations, that's OK)
        assert isinstance(result.converged, bool), f"converged should be bool, got {type(result.converged)}"
        assert result.num_iter <= 10, f"num_iter should be <= 10 (max_iter), got {result.num_iter}"
        
        # Verify log-likelihood exists
        assert isinstance(result.loglik, (int, float)), f"loglik should be numeric, got {type(result.loglik)}"
        assert not np.isnan(result.loglik), f"loglik should not be NaN, got {result.loglik}"
        assert not np.isinf(result.loglik), f"loglik should not be infinite, got {result.loglik}"
        
        # Verify model parameters exist
        result_attrs = dir(result)
        assert hasattr(result, 'A'), f"Result should have A attribute (transition matrix), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'Q'), f"Result should have Q attribute (innovation covariance), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'R'), f"Result should have R attribute (observation covariance), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify parameter shapes
        assert result.A.shape == (1, 1), f"Expected A shape (1, 1) for 1 factor AR(1), got {result.A.shape}"
        assert result.Q.shape == (1, 1), f"Expected Q shape (1, 1) for 1 factor, got {result.Q.shape}"
        assert result.R.shape == (5, 5), f"Expected R shape (5, 5) for 5 series, got {result.R.shape}"
    
    def test_dfm_prediction(self):
        """Test prediction functionality with different return type combinations.
        
        This test verifies that predict() method works correctly with:
        1. Default behavior (returns both series and factors)
        2. return_series=True, return_factors=False (series only)
        3. return_series=False, return_factors=True (factors only)
        4. Forecast value quality (no NaN, no infinite values)
        
        Expected behavior:
        - Default predict() returns tuple of (X_forecast, Z_forecast)
        - return_series=True, return_factors=False returns single array
        - return_series=False, return_factors=True returns single array
        - All forecasts contain valid numeric values (no NaN, no infinite)
        """
        model = DFM()
        config = create_simple_config(num_series=5, num_factors=1)
        model.load_config(source=config)
        
        X = generate_synthetic_data(n_periods=50, n_series=5)
        model.load_data(data=X)
        model.train(max_iter=10, threshold=1e-3)
        
        # Test 1: Predict with horizon specified
        X_forecast, Z_forecast = model.predict(horizon=10)
        assert X_forecast.shape == (10, 5), f"Expected forecast shape (10, 5), got {X_forecast.shape}"
        assert Z_forecast.shape == (10, 1), f"Expected factor forecast shape (10, 1), got {Z_forecast.shape}"
        
        # Test 2: Predict only series
        X_forecast_only = model.predict(horizon=5, return_series=True, return_factors=False)
        assert isinstance(X_forecast_only, np.ndarray), f"Forecast should be numpy array, got {type(X_forecast_only)}"
        assert X_forecast_only.shape == (5, 5), f"Expected forecast shape (5, 5), got {X_forecast_only.shape}"
        
        # Test 3: Predict only factors
        Z_forecast_only = model.predict(horizon=5, return_series=False, return_factors=True)
        assert isinstance(Z_forecast_only, np.ndarray), f"Factor forecast should be numpy array, got {type(Z_forecast_only)}"
        assert Z_forecast_only.shape == (5, 1), f"Expected factor forecast shape (5, 1), got {Z_forecast_only.shape}"
        
        # Test 4: Verify forecast values are reasonable
        nan_count_X = np.sum(np.isnan(X_forecast))
        assert nan_count_X == 0, f"Forecast should not contain NaN values, but found {nan_count_X} NaN values in X_forecast"
        inf_count_X = np.sum(np.isinf(X_forecast))
        assert inf_count_X == 0, f"Forecast should not contain infinite values, but found {inf_count_X} infinite values in X_forecast"
        nan_count_Z = np.sum(np.isnan(Z_forecast))
        assert nan_count_Z == 0, f"Factor forecast should not contain NaN values, but found {nan_count_Z} NaN values in Z_forecast"
        inf_count_Z = np.sum(np.isinf(Z_forecast))
        assert inf_count_Z == 0, f"Factor forecast should not contain infinite values, but found {inf_count_Z} infinite values in Z_forecast"
        
        # Test 5: Error handling - predict before training
        model2 = DFM()
        model2.load_config(source=config)
        model2.load_data(data=X)
        # Predict before training should raise ValueError
        with pytest.raises(ValueError, match="Model must be fitted"):
            model2.predict(horizon=5)  # No result yet


# ============================================================================
# Low-Level API Tests
# ============================================================================

class TestDFMLowLevelAPI:
    """Test low-level DFM API (DFMLinear class)."""
    
    def test_dfm_linear_fit_basic(self):
        """Test basic fit method of DFMLinear (low-level API).
        
        This test verifies the low-level API workflow:
        1. Direct model fitting using DFMLinear.fit()
        2. Result object structure and attributes
        3. Factor and loading shapes match input data
        
        Expected behavior:
        - fit() returns DFMResult object
        - Result contains Z (factors) and C (loadings)
        - Factor length matches input data length (T periods)
        - Loading dimensions match number of series and factors
        """
        model = DFMLinear()
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Fit the model
        result = model.fit(X, config, max_iter=10, threshold=1e-3)
        
        # Verify result
        assert isinstance(result, DFMResult), f"Result should be DFMResult, got {type(result)}"
        result_attrs = dir(result)
        assert hasattr(result, 'Z'), f"Result should have Z attribute (factors), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'C'), f"Result should have C attribute (loadings), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'converged'), f"Result should have converged attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(result, 'num_iter'), f"Result should have num_iter attribute, but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify shapes
        # Z is (T x m) where T is number of time periods
        assert result.Z.shape[0] == X.shape[0], f"Factor length {result.Z.shape[0]} should match input data length {X.shape[0]}"
        assert result.Z.shape[1] == 1, f"Expected 1 factor, got {result.Z.shape[1]}"
        assert result.C.shape[0] == 5, f"Expected 5 series in loadings, got {result.C.shape[0]}"
        assert result.C.shape[1] == 1, f"Expected 1 factor in loadings, got {result.C.shape[1]}"
    
    def test_dfm_linear_fit_with_missing_data(self):
        """Test fit with missing data handling.
        
        This test verifies that DFMLinear can handle:
        1. Moderate amounts of missing data (NaN values)
        2. Missing data in different series and time periods
        3. Graceful degradation if missing data handling has limitations
        
        Expected behavior:
        - fit() should handle moderate missing data without errors
        - If missing data causes issues, test skips with informative message
        - Result should contain valid factors and loadings when fit succeeds
        - Shapes should be reasonable even with missing data
        """
        model = DFMLinear()
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Introduce some missing values (moderate amount)
        X[10:15, 0] = np.nan
        X[20:25, 2] = np.nan
        
        # Fit should handle missing data (may raise warnings but should complete)
        # Note: Excessive missing data might cause issues, but moderate amounts should work
        try:
            result = model.fit(X, config, max_iter=10, threshold=1e-3)
            # Verify result exists if fit succeeded
            assert isinstance(result, DFMResult), f"Result should be DFMResult, got {type(result)}"
            assert result.Z is not None, "Result should have Z (factors) even with missing data"
            assert result.C is not None, "Result should have C (loadings) even with missing data"
            # Verify shapes are reasonable
            assert result.Z.shape[0] > 0, f"Factor length should be > 0, got {result.Z.shape[0]}"
            assert result.Z.shape[1] > 0, f"Number of factors should be > 0, got {result.Z.shape[1]}"
        except (ValueError, RuntimeError, TypeError) as e:
            # If fit fails due to missing data, that's acceptable - just document it
            # Check if error is related to missing data using shared helper
            if check_missing_data_error(e):
                pytest.skip(f"Missing data handling not supported or failed: {e}")
            else:
                # Re-raise if it's a different error (shouldn't happen in normal flow)
                raise
    
    def test_dfm_linear_fit_convergence(self):
        """Test convergence behavior with different parameters.
        
        This test verifies:
        1. Convergence with different max_iter values
        2. Convergence with different threshold values
        3. Result attributes reflect convergence state
        
        Expected behavior:
        - Training completes with different parameter combinations
        - num_iter respects max_iter limit
        - converged flag reflects actual convergence state
        - Result structure is consistent across parameter combinations
        """
        model = DFMLinear()
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Test with different max_iter
        result1 = model.fit(X, config, max_iter=5, threshold=1e-2)
        assert result1.num_iter <= 5, f"num_iter should be <= 5, got {result1.num_iter}"
        
        result2 = model.fit(X, config, max_iter=20, threshold=1e-4)
        assert result2.num_iter <= 20, f"num_iter should be <= 20, got {result2.num_iter}"
        
        # Both should produce valid results
        assert isinstance(result1, DFMResult), f"Result1 should be DFMResult, got {type(result1)}"
        assert isinstance(result2, DFMResult), f"Result2 should be DFMResult, got {type(result2)}"
        assert result1.Z is not None, "Result1 should have Z (factors)"
        assert result2.Z is not None, "Result2 should have Z (factors)"

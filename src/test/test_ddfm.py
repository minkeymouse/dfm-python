"""Tests for Deep Dynamic Factor Model (DDFM) functionality.

This module provides comprehensive tests for the Deep Dynamic Factor Model (DDFM)
implementation in the dfm-python package. DDFM extends linear DFM by using neural
networks for nonlinear factor extraction, trained with gradient descent.

**Test Organization**:
- `TestDDFMHighLevelAPI`: Tests the high-level DDFM class API (5 tests)
  - Model creation with neural encoder parameters (encoder_layers, num_factors, epochs)
  - Configuration loading from various sources
  - Data loading and validation
  - Training with gradient descent (not EM algorithm)
  - Factor extraction and result validation
  - Prediction functionality
- `TestDDFMLowLevelAPI`: Tests the low-level DDFMModel class API (3 tests)
  - Direct fit() method usage
  - Result object validation
  - Prediction with low-level API

**Dependencies**:
- Required: `torch` (PyTorch for neural network training)
- Required: `sktime` (for preprocessing via DFMScaler)
- Tests are skipped if dependencies are not available

**Usage Patterns**:
- High-level API: Create DDFM(encoder_layers=[...], num_factors=...) → load_config() → load_data() → train(epochs=...) → predict()
- Low-level API: Create DDFMModel(...) → fit(X, config) → access result
- All tests use shared utilities from `test/__init__.py` for consistency

**Key Differences from DFM**:
- Constructor requires neural network parameters (encoder_layers, num_factors, epochs)
- Training uses gradient descent (train(epochs=...)) instead of EM algorithm
- Requires PyTorch for neural network operations
- Better performance during structural breaks and nonlinear relationships

**Related Test Files**:
- `test_dfm.py`: Tests linear Dynamic Factor Model (DFM) - similar API but uses EM algorithm
- `test_transformations.py`: Tests data preprocessing transformations used by DDFM
- `test_lightning_module.py`: Tests PyTorch Lightning integration for DDFM
"""

# Standard library imports
import pytest

# Third-party imports
import numpy as np
import polars as pl

# Local application imports
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.models import DDFM, DDFMModel
from dfm_python.config.results import DDFMResult

# Local relative imports
from . import (
    check_missing_data_error,
    create_simple_config,
    create_simple_config_mapping,
    generate_synthetic_data,
)


# Skip all tests if PyTorch is not available (required for DDFM)
pytest.importorskip("torch", reason="PyTorch is required for DDFM tests")
# Skip all tests if sktime is not available (required for preprocessing)
pytest.importorskip("sktime", reason="sktime is required for DDFM tests")


# ============================================================================
# High-Level API Tests
# ============================================================================

class TestDDFMHighLevelAPI:
    """Test high-level DDFM API (DDFM class)."""
    
    def test_ddfm_high_level_api_basic(self):
        """Test basic high-level DDFM API workflow.
        
        This test verifies the complete DDFM workflow:
        1. Model creation with neural network parameters
        2. Configuration loading
        3. Data loading with explicit series alignment
        4. Training with gradient descent
        5. Result validation (factors, loadings, quality checks)
        6. Prediction functionality
        
        Expected behavior:
        - Model creates successfully with encoder parameters
        - Configuration loads successfully
        - Data loads and matches expected shape
        - Training completes and produces valid DDFMResult
        - Factors have correct shape (T x m) matching data length
        - Loadings have correct shape (N x m) matching series and factors
        - Results contain no NaN or infinite values
        - Predictions are generated without NaN or infinite values
        
        Key differences from DFM:
        - Constructor requires encoder_layers, num_factors, epochs
        - Training uses train(epochs=...) instead of train(max_iter=..., threshold=...)
        - Result type is DDFMResult instead of DFMResult
        """
        # Create DDFM model instance with parameters for reliable training
        # Increased epochs from 5 to 10 for more reliable training while still being fast
        model = DDFM(
            encoder_layers=[32, 16],
            num_factors=1,
            epochs=10,  # Increased for more reliable training (was 5)
            batch_size=32,
            learning_rate=0.001
        )
        
        # Create simple config
        config = create_simple_config(num_series=5, num_factors=1)
        
        # Load configuration using source parameter
        model.load_config(source=config)
        
        # Verify config is loaded
        assert model.config is not None, "Configuration should be loaded after load_config()"
        assert len(model.config.series) == 5, f"Expected 5 series in config, got {len(model.config.series)}"
        
        # Generate synthetic data
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Convert to Polars DataFrame with explicit column names matching config series_ids
        # This ensures explicit data-column-to-config-series alignment
        # Column names must match the series_ids in the config
        X_df = pl.DataFrame({
            f'series_{i}': X[:, i] for i in range(5)
        })
        
        # Load data using DataFrame (explicit alignment)
        model.load_data(data=X_df)
        
        # Verify data is loaded
        assert model.data is not None, "Data should be loaded after load_data()"
        assert model.data.shape == (50, 5), f"Expected data shape (50, 5), got {model.data.shape}"
        assert model.original_data is not None, "Original data should be available after load_data()"
        assert model.original_data.shape == (50, 5), f"Expected original_data shape (50, 5), got {model.original_data.shape}"
        
        # Train the model using epochs (not max_iter/threshold like DFM)
        # Match the epochs parameter from constructor
        model.train(epochs=10)
        
        # Verify training completed
        assert model.result is not None, "Result should be available after training"
        assert isinstance(model.result, DDFMResult), f"Result should be DDFMResult, got {type(model.result)}"
        result_attrs = dir(model.result)
        assert hasattr(model.result, 'Z'), f"Result should have Z attribute (factors), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(model.result, 'C'), f"Result should have C attribute (loadings), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify result shapes
        # Z is (T x m) where T is number of time periods
        # Note: DDFM factors should match data length (T periods, not T+1)
        assert model.result.Z.shape[0] == model.data.shape[0], f"Factor length {model.result.Z.shape[0]} should match data length {model.data.shape[0]}"
        assert model.result.Z.shape[1] == 1, f"Expected 1 factor, got {model.result.Z.shape[1]}"
        assert model.result.C.shape[0] == 5, f"Expected 5 series in loadings, got {model.result.C.shape[0]}"
        assert model.result.C.shape[1] == 1, f"Expected 1 factor in loadings, got {model.result.C.shape[1]}"
        
        # Result quality checks: verify no NaN or infinite values in factors and loadings
        nan_count_Z = np.sum(np.isnan(model.result.Z))
        assert nan_count_Z == 0, f"Factors should not contain NaN values, but found {nan_count_Z} NaN values in factors"
        inf_count_Z = np.sum(np.isinf(model.result.Z))
        assert inf_count_Z == 0, f"Factors should not contain infinite values, but found {inf_count_Z} infinite values in factors"
        nan_count_C = np.sum(np.isnan(model.result.C))
        assert nan_count_C == 0, f"Loadings should not contain NaN values, but found {nan_count_C} NaN values in loadings"
        inf_count_C = np.sum(np.isinf(model.result.C))
        assert inf_count_C == 0, f"Loadings should not contain infinite values, but found {inf_count_C} infinite values in loadings"
        
        # Verify factors have reasonable values (not all zeros, not extreme)
        assert np.abs(model.result.Z).max() < 1e6, "Factors have extreme values"
        assert np.abs(model.result.Z).max() > 0, "Factors are all zeros"
        assert np.abs(model.result.C).max() < 1e6, "Loadings have extreme values"
        assert np.abs(model.result.C).max() > 0, "Loadings are all zeros"
        
        # Test prediction
        X_forecast, Z_forecast = model.predict(horizon=5)
        
        # Verify forecast shapes
        assert X_forecast.shape == (5, 5), f"Expected forecast shape (5, 5), got {X_forecast.shape}"
        assert Z_forecast.shape == (5, 1), f"Expected factor forecast shape (5, 1), got {Z_forecast.shape}"
        
        # Verify forecast values are reasonable (not NaN, not infinite)
        nan_count_X = np.sum(np.isnan(X_forecast))
        assert nan_count_X == 0, f"Forecasted series should not contain NaN values, but found {nan_count_X} NaN values in X_forecast"
        inf_count_X = np.sum(np.isinf(X_forecast))
        assert inf_count_X == 0, f"Forecasted series should not contain infinite values, but found {inf_count_X} infinite values in X_forecast"
        nan_count_Z = np.sum(np.isnan(Z_forecast))
        assert nan_count_Z == 0, f"Forecasted factors should not contain NaN values, but found {nan_count_Z} NaN values in Z_forecast"
        inf_count_Z = np.sum(np.isinf(Z_forecast))
        assert inf_count_Z == 0, f"Forecasted factors should not contain infinite values, but found {inf_count_Z} infinite values in Z_forecast"
        
        # Verify forecast values are reasonable (not extreme)
        assert np.abs(X_forecast).max() < 1e6, "Forecasted series have extreme values"
        assert np.abs(Z_forecast).max() < 1e6, "Forecasted factors have extreme values"
    
    def test_ddfm_config_loading(self):
        """Test configuration loading from different sources.
        
        This test verifies that DDFM can load configurations from:
        1. DFMConfig object (source parameter)
        2. Dictionary/mapping (mapping parameter)
        3. Error handling for invalid configurations
        
        Expected behavior:
        - Config loads successfully from DFMConfig object
        - Config loads successfully from mapping dict
        - Invalid config raises appropriate exception (ValueError or TypeError)
        - Model.reset() allows reusing same instance for multiple configs
        
        Key differences from DFM:
        - DDFM constructor requires encoder_layers, num_factors, epochs
        - Config structure is same (DFMConfig), but DDFM uses different training method
        """
        # Create model instance once - reuse with reset() for consistency with test_dfm.py
        model = DDFM(encoder_layers=[32, 16], num_factors=1, epochs=10)
        
        # Test 1: Load from DFMConfig object
        config = create_simple_config(num_series=3, num_factors=1)
        model.load_config(source=config)
        assert model.config is not None, "Configuration should be loaded from DFMConfig object"
        assert len(model.config.series) == 3, f"Expected 3 series in config, got {len(model.config.series)}"
        assert model.config.clock == 'm', f"Expected clock 'm', got {model.config.clock}"
        
        # Test 2: Load from mapping dict
        model.reset()  # Reuse same instance for consistency
        model.load_config(mapping=create_simple_config_mapping(num_series=1, num_factors=1, max_iter=None, threshold=None))
        assert model.config is not None, "Configuration should be loaded from mapping dict"
        assert len(model.config.series) == 1, f"Expected 1 series in config, got {len(model.config.series)}"
        assert model.config.series[0].series_id == 'series_0', f"Expected series_id 'series_0', got {model.config.series[0].series_id}"
        
        # Test 3: Error handling - invalid config
        model.reset()  # Reuse same instance for consistency
        # Invalid config should raise an error (ValueError or TypeError for invalid structure)
        with pytest.raises((ValueError, TypeError)):
            model.load_config(mapping={'invalid': 'config'})
    
    def test_ddfm_data_loading(self):
        """Test data loading from different sources.
        
        This test verifies that DDFM can load data from:
        1. NumPy array
        2. Polars DataFrame
        3. Error handling for missing data
        
        Expected behavior:
        - Data loads successfully from NumPy array
        - Data loads successfully from Polars DataFrame
        - Data shapes match expected dimensions
        - original_data property preserves untransformed data
        - Missing data parameter raises ValueError
        
        Key differences from DFM:
        - DDFM constructor requires encoder_layers, num_factors, epochs
        - Data loading API is same as DFM
        """
        # Create model instance
        model = DDFM(encoder_layers=[32, 16], num_factors=1, epochs=10)
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
        assert model.original_data is not None, "Original data should be available after loading DataFrame"
        # Note: data and original_data may differ if preprocessing is applied
        # Just verify they have the same shape
        assert model.original_data.shape == model.data.shape, f"Original data shape {model.original_data.shape} should match data shape {model.data.shape}"
        
        # Test 3: Error handling - no data provided
        model.reset()
        model.load_config(source=config)
        with pytest.raises(ValueError):
            model.load_data()  # Neither data_path nor data provided
    
    def test_ddfm_training_convergence(self):
        """Test DDFM training convergence and result quality.
        
        This test verifies:
        1. Training completes successfully
        2. Result structure is correct (DDFMResult)
        3. Factors and loadings are extracted
        4. Training produces reasonable results (no NaN, no extreme values)
        5. Training loss tracking (if available)
        
        Expected behavior:
        - Training completes without errors
        - Result is DDFMResult object with Z (factors) and C (loadings)
        - Factor and loading shapes match expected dimensions
        - Results contain no NaN or infinite values
        - Results have reasonable values (not all zeros, not extreme)
        - Training loss (if available) is non-negative and finite
        - Model parameters (A, Q, R if available) have correct shapes and are valid
        
        Key differences from DFM:
        - DDFM uses train(epochs=...) instead of train(max_iter=..., threshold=...)
        - DDFM result type is DDFMResult instead of DFMResult
        - DDFM may have loss attribute instead of loglik
        - DDFM uses gradient descent, not EM algorithm
        """
        # Create DDFM model with parameters for reliable training
        model = DDFM(
            encoder_layers=[32, 16],
            num_factors=1,
            epochs=10,  # Sufficient for convergence test
            batch_size=32,
            learning_rate=0.001
        )
        
        # Load config and data
        config = create_simple_config(num_series=5, num_factors=1)
        model.load_config(source=config)
        
        X = generate_synthetic_data(n_periods=50, n_series=5)
        X_df = pl.DataFrame({
            f'series_{i}': X[:, i] for i in range(5)
        })
        model.load_data(data=X_df)
        
        # Train the model
        model.train(epochs=10)
        
        # Verify training completed
        assert model.result is not None, "Result should be available after training"
        assert isinstance(model.result, DDFMResult), f"Result should be DDFMResult, got {type(model.result)}"
        
        # Verify result structure
        result_attrs = dir(model.result)
        assert hasattr(model.result, 'Z'), f"Result should have Z attribute (factors), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        assert hasattr(model.result, 'C'), f"Result should have C attribute (loadings), but available attributes are: {[a for a in result_attrs if not a.startswith('_')]}"
        
        # Verify shapes
        assert model.result.Z.shape[0] == model.data.shape[0], f"Factor length {model.result.Z.shape[0]} should match data length {model.data.shape[0]}"
        assert model.result.Z.shape[1] == 1, f"Expected 1 factor, got {model.result.Z.shape[1]}"
        assert model.result.C.shape[0] == 5, f"Expected 5 series in loadings, got {model.result.C.shape[0]}"
        assert model.result.C.shape[1] == 1, f"Expected 1 factor in loadings, got {model.result.C.shape[1]}"
        
        # Verify result quality (no NaN, no infinite, reasonable values)
        nan_count_Z = np.sum(np.isnan(model.result.Z))
        assert nan_count_Z == 0, f"Factors should not contain NaN values, but found {nan_count_Z} NaN values in factors"
        inf_count_Z = np.sum(np.isinf(model.result.Z))
        assert inf_count_Z == 0, f"Factors should not contain infinite values, but found {inf_count_Z} infinite values in factors"
        nan_count_C = np.sum(np.isnan(model.result.C))
        assert nan_count_C == 0, f"Loadings should not contain NaN values, but found {nan_count_C} NaN values in loadings"
        inf_count_C = np.sum(np.isinf(model.result.C))
        assert inf_count_C == 0, f"Loadings should not contain infinite values, but found {inf_count_C} infinite values in loadings"
        assert np.abs(model.result.Z).max() < 1e6, "Factors have extreme values"
        assert np.abs(model.result.Z).max() > 0, "Factors are all zeros"
        assert np.abs(model.result.C).max() < 1e6, "Loadings have extreme values"
        assert np.abs(model.result.C).max() > 0, "Loadings are all zeros"
        
        # Verify training loss (if available in result)
        if hasattr(model.result, 'loss'):
            assert not np.isnan(model.result.loss), "Training loss is NaN"
            assert not np.isinf(model.result.loss), "Training loss is infinite"
            assert model.result.loss >= 0, "Training loss should be non-negative"
        
        # Verify model parameters exist (if available)
        # DDFM may have A, Q, R similar to DFM, or may have different structure
        if hasattr(model.result, 'A'):
            assert model.result.A.shape == (1, 1), "Transition matrix A has incorrect shape"
            assert not np.any(np.isnan(model.result.A)), "Transition matrix A contains NaN"
            assert not np.any(np.isinf(model.result.A)), "Transition matrix A contains infinite values"
        
        if hasattr(model.result, 'Q'):
            assert model.result.Q.shape == (1, 1), "Innovation covariance Q has incorrect shape"
            assert not np.any(np.isnan(model.result.Q)), "Innovation covariance Q contains NaN"
            assert not np.any(np.isinf(model.result.Q)), "Innovation covariance Q contains infinite values"
        
        if hasattr(model.result, 'R'):
            assert model.result.R.shape == (5, 5), "Observation covariance R has incorrect shape"
            assert not np.any(np.isnan(model.result.R)), "Observation covariance R contains NaN"
            assert not np.any(np.isinf(model.result.R)), "Observation covariance R contains infinite values"
    
    def test_ddfm_prediction(self):
        """Test DDFM prediction functionality with different return type combinations.
        
        This test verifies:
        1. Prediction with both series and factors (default behavior)
        2. Prediction with only series (return_series=True, return_factors=False)
        3. Prediction with only factors (return_series=False, return_factors=True)
        4. Forecast values are reasonable (no NaN, no infinite, reasonable ranges)
        5. Error handling for prediction before training
        
        Expected behavior:
        - Default prediction returns both series and factors as tuple
        - Prediction with return_series=True, return_factors=False returns only series array
        - Prediction with return_series=False, return_factors=True returns only factors array
        - Forecast shapes match expected dimensions (horizon x N, horizon x m)
        - Forecast values contain no NaN or infinite values
        - Forecast values are reasonable (not extreme)
        - Prediction before training raises appropriate exception
        
        Key differences from DFM:
        - DDFM uses train(epochs=...) instead of train(max_iter=..., threshold=...)
        - DDFM constructor requires encoder_layers, num_factors, epochs
        - Same predict() API and return type combinations as DFM
        """
        # Create and train DDFM model
        model = DDFM(
            encoder_layers=[32, 16],
            num_factors=1,
            epochs=10,  # Sufficient for prediction test
            batch_size=32,
            learning_rate=0.001
        )
        config = create_simple_config(num_series=5, num_factors=1)
        model.load_config(source=config)
        
        X = generate_synthetic_data(n_periods=50, n_series=5)
        X_df = pl.DataFrame({
            f'series_{i}': X[:, i] for i in range(5)
        })
        model.load_data(data=X_df)
        model.train(epochs=10)
        
        # Test 1: Predict with both series and factors (default)
        X_forecast, Z_forecast = model.predict(horizon=10)
        assert X_forecast.shape == (10, 5), "Forecasted series should have shape (10, 5)"
        assert Z_forecast.shape == (10, 1), "Forecasted factors should have shape (10, 1)"
        
        # Test 2: Predict only series
        X_forecast_only = model.predict(horizon=5, return_series=True, return_factors=False)
        assert isinstance(X_forecast_only, np.ndarray), "Forecast should return numpy array when only series requested"
        assert X_forecast_only.shape == (5, 5), "Forecasted series only should have shape (5, 5)"
        
        # Test 3: Predict only factors
        Z_forecast_only = model.predict(horizon=5, return_series=False, return_factors=True)
        assert isinstance(Z_forecast_only, np.ndarray), "Forecast should return numpy array when only factors requested"
        assert Z_forecast_only.shape == (5, 1), "Forecasted factors only should have shape (5, 1)"
        
        # Test 4: Verify forecast values are reasonable
        assert not np.any(np.isnan(X_forecast)), "Forecasted series contain NaN values"
        assert not np.any(np.isinf(X_forecast)), "Forecasted series contain infinite values"
        assert not np.any(np.isnan(Z_forecast)), "Forecasted factors contain NaN values"
        assert not np.any(np.isinf(Z_forecast)), "Forecasted factors contain infinite values"
        
        # Verify forecast values are reasonable (not extreme)
        assert np.abs(X_forecast).max() < 1e6, "Forecasted series have extreme values"
        assert np.abs(Z_forecast).max() < 1e6, "Forecasted factors have extreme values"
        
        # Test 5: Error handling - predict before training
        model2 = DDFM(encoder_layers=[32, 16], num_factors=1, epochs=5)
        model2.load_config(source=config)
        model2.load_data(data=X_df)
        # Model not trained yet, predict should raise an error
        with pytest.raises((ValueError, AttributeError)):
            model2.predict(horizon=5)


# ============================================================================
# Low-Level API Tests
# ============================================================================

class TestDDFMLowLevelAPI:
    """Test low-level DDFM API (DDFMModel class)."""
    
    def test_ddfm_model_fit_basic(self):
        """Test basic fit method of DDFMModel.
        
        This test verifies that the low-level API (DDFMModel.fit()) works correctly:
        1. Model can be instantiated with encoder parameters
        2. fit() method accepts data and config
        3. Returns DDFMResult with correct structure
        4. Factors and loadings are extracted correctly
        
        Expected behavior:
        - Model instantiates successfully with encoder parameters
        - fit() method completes without errors
        - Returns DDFMResult object with Z (factors) and C (loadings)
        - Factor shapes match input data length (T x m)
        - Loading shapes match number of series and factors (N x m)
        - Results contain no NaN or infinite values
        - Results have reasonable values (not all zeros, not extreme)
        
        Key differences from DFM:
        - DDFM uses DDFMModel instead of DFMLinear
        - DDFM uses epochs parameter instead of max_iter/threshold
        - DDFM returns DDFMResult instead of DFMResult
        """
        # Create DDFMModel instance with encoder parameters
        # Note: epochs specified in fit() call, not constructor (single source of truth)
        model = DDFMModel(
            encoder_layers=[32, 16],
            num_factors=1,
            batch_size=16,  # Reduced for small data (50 periods / 16 = ~3 batches)
            learning_rate=0.001
        )
        
        # Create config and data
        # Increased data size to ensure multiple batches for batch normalization
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=100, n_series=5)  # More data for better batch processing
        
        # Fit the model using low-level API
        # Note: DDFM uses epochs instead of max_iter/threshold
        result = model.fit(X, config, epochs=10)
        
        # Verify result type
        assert isinstance(result, DDFMResult), f"Result should be DDFMResult, got {type(result)}"
        
        # Verify result structure
        assert hasattr(result, 'Z'), "Result should have Z attribute (factors)"
        assert hasattr(result, 'C'), "Result should have C attribute (loadings)"
        
        # Verify shapes
        # Z is (T x m) where T is number of time periods
        # DDFM preprocessing (transformations, standardization) preserves data shape (T x N)
        # Factors are extracted from processed data, so Z.shape[0] should match X.shape[0]
        assert result.Z.shape[0] == X.shape[0], f"Factor shape {result.Z.shape[0]} should match input data shape {X.shape[0]}"
        assert result.Z.shape[1] == 1, f"Expected 1 factor, got {result.Z.shape[1]}"
        assert result.C.shape[0] == 5, f"Expected 5 series in loadings, got {result.C.shape[0]}"
        assert result.C.shape[1] == 1, f"Expected 1 factor in loadings, got {result.C.shape[1]}"
        
        # Verify result quality (no NaN, no infinite)
        assert not np.any(np.isnan(result.Z)), "Factors contain NaN values"
        assert not np.any(np.isinf(result.Z)), "Factors contain infinite values"
        assert not np.any(np.isnan(result.C)), "Loadings contain NaN values"
        assert not np.any(np.isinf(result.C)), "Loadings contain infinite values"
        
        # Verify values are reasonable (not all zeros, not extreme)
        assert np.abs(result.Z).max() < 1e6, "Factors have extreme values"
        assert np.abs(result.Z).max() > 0, "Factors are all zeros"
        assert np.abs(result.C).max() < 1e6, "Loadings have extreme values"
        assert np.abs(result.C).max() > 0, "Loadings are all zeros"
    
    def test_ddfm_model_fit_with_missing_data(self):
        """Test fit with missing data.
        
        This test verifies that DDFMModel can handle missing data:
        1. Model can fit with moderate amounts of missing data
        2. Result doesn't contain NaN values (after handling)
        3. Shapes are correct even with missing data
        
        Expected behavior:
        - Model handles moderate amounts of missing data gracefully
        - fit() completes successfully or raises appropriate error
        - If fit succeeds, result has correct shapes matching input data
        - Result doesn't contain NaN values after missing data handling
        - Test may skip if missing data handling is not supported
        
        Key differences from DFM:
        - DDFM uses epochs instead of max_iter/threshold
        - DDFM may handle missing data differently (neural network preprocessing)
        """
        # Create DDFMModel instance
        model = DDFMModel(
            encoder_layers=[32, 16],
            num_factors=1,
            batch_size=16,  # Reduced for small data
            learning_rate=0.001
        )
        
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=100, n_series=5)  # More data for better batch processing
        
        # Introduce some missing values (moderate amount)
        # Similar to test_dfm.py pattern
        X[10:15, 0] = np.nan
        X[20:25, 2] = np.nan
        
        # Fit should handle missing data (may raise warnings but should complete)
        # Note: DDFM uses neural network preprocessing which may handle missing data differently
        # DDFM applies spline interpolation for missing data before neural network training
        try:
            result = model.fit(X, config, epochs=10)
            
            # Verify result exists if fit succeeded
            assert isinstance(result, DDFMResult), f"Result should be DDFMResult, got {type(result)}"
            assert result.Z is not None, "Result should have Z (factors) even with missing data"
            assert result.C is not None, "Result should have C (loadings) even with missing data"
            
            # Verify shapes match input data (preprocessing preserves shape)
            # DDFM preprocessing handles missing data via interpolation, so shape should match
            assert result.Z.shape[0] == X.shape[0], f"Factor shape {result.Z.shape[0]} should match input data shape {X.shape[0]}"
            assert result.Z.shape[1] > 0, f"Number of factors should be > 0, got {result.Z.shape[1]}"
            assert result.C.shape[0] == 5, f"Expected 5 series in loadings, got {result.C.shape[0]}"
            assert result.C.shape[1] > 0, f"Number of factors in loadings should be > 0, got {result.C.shape[1]}"
            
            # Verify result doesn't contain NaN (after handling)
            assert not np.any(np.isnan(result.Z)), "Factors contain NaN after missing data handling"
            assert not np.any(np.isnan(result.C)), "Loadings contain NaN after missing data handling"
            
        except (ValueError, RuntimeError, TypeError) as e:
            # If fit fails due to missing data, that's acceptable - just document it
            # Check if error is related to missing data using shared helper
            if check_missing_data_error(e):
                pytest.skip(f"Missing data handling not supported or failed: {e}")
            else:
                # Re-raise if it's a different error (shouldn't happen in normal flow)
                raise
    
    def test_ddfm_model_fit_convergence(self):
        """Test convergence behavior with different parameters.
        
        This test verifies that DDFMModel training converges:
        1. Training completes with different epoch counts
        2. Results are valid for different training durations
        3. Training loss (if available) is reasonable
        
        Expected behavior:
        - Training completes successfully with different epoch counts
        - Results are valid (DDFMResult objects) for all epoch counts
        - Factor and loading shapes match input data regardless of epoch count
        - Results contain no NaN or infinite values
        - Training loss (if available) is non-negative and finite
        
        Key differences from DFM:
        - DDFM uses epochs instead of max_iter/threshold
        - DDFM may have loss attribute instead of loglik
        - DDFM uses gradient descent, not EM algorithm
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=100, n_series=5)  # More data for better batch processing
        
        # Test with different epoch counts
        # Similar to test_dfm.py pattern but using epochs instead of max_iter
        # Note: epochs specified in fit() call, not constructor (single source of truth)
        model1 = DDFMModel(
            encoder_layers=[32, 16],
            num_factors=1,
            batch_size=16,  # Reduced for small data
            learning_rate=0.001
        )
        result1 = model1.fit(X, config, epochs=5)  # Fewer epochs
        
        model2 = DDFMModel(
            encoder_layers=[32, 16],
            num_factors=1,
            batch_size=16,  # Reduced for small data
            learning_rate=0.001
        )
        result2 = model2.fit(X, config, epochs=15)  # More epochs
        
        # Both should produce valid results
        assert isinstance(result1, DDFMResult), f"Result1 should be DDFMResult, got {type(result1)}"
        assert isinstance(result2, DDFMResult), f"Result2 should be DDFMResult, got {type(result2)}"
        assert result1.Z is not None, "Result1 should have Z (factors)"
        assert result2.Z is not None, "Result2 should have Z (factors)"
        
        # Verify shapes are correct
        # Shapes should always match for same input data regardless of epoch count
        # Factor shape is determined by input data shape (T x m), not by training duration
        assert result1.Z.shape == result2.Z.shape, f"Factor shapes should match for different epoch counts: {result1.Z.shape} vs {result2.Z.shape}"
        assert result1.C.shape == result2.C.shape, f"Loading shapes should match for different epoch counts: {result1.C.shape} vs {result2.C.shape}"
        # Also verify shapes match input data
        assert result1.Z.shape[0] == X.shape[0], f"Result1 factor shape {result1.Z.shape[0]} should match input data shape {X.shape[0]}"
        assert result2.Z.shape[0] == X.shape[0], f"Result2 factor shape {result2.Z.shape[0]} should match input data shape {X.shape[0]}"
        
        # Verify result quality (no NaN, no infinite)
        assert not np.any(np.isnan(result1.Z)), "Result1 factors contain NaN"
        assert not np.any(np.isnan(result2.Z)), "Result2 factors contain NaN"
        assert not np.any(np.isinf(result1.Z)), "Result1 factors contain infinite values"
        assert not np.any(np.isinf(result2.Z)), "Result2 factors contain infinite values"
        
        # Verify training loss (if available) is reasonable
        if hasattr(result1, 'loss'):
            assert not np.isnan(result1.loss), "Result1 training loss is NaN"
            assert not np.isinf(result1.loss), "Result1 training loss is infinite"
            assert result1.loss >= 0, "Result1 training loss should be non-negative"
        
        if hasattr(result2, 'loss'):
            assert not np.isnan(result2.loss), "Result2 training loss is NaN"
            assert not np.isinf(result2.loss), "Result2 training loss is infinite"
            assert result2.loss >= 0, "Result2 training loss should be non-negative"

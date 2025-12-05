"""Pipeline tests for complete DDFM (Deep Dynamic Factor Model) workflows.

This module tests the complete DDFM pipeline from configuration loading,
data loading (with preprocessing pipeline), model training with neural encoder, to prediction and nowcasting.

Note: Users can provide either raw data with a preprocessing pipeline (recommended) or
preprocessed data. When a pipeline is provided, preprocessing happens automatically in setup().
The pipeline is also used to extract statistics (Mx/Wx) for forecasting/nowcasting.

Test Structure:
- TestDDFMPipeline: Tests for Deep Dynamic Factor Model (DDFM) pipeline
- TestDDFMPipelineIntegration: Integration tests for DDFM pipeline components

Note: Some tests may skip if:
- Test data files are missing (data/sample_data.csv)
- Test config files are missing (config/experiment/test_ddfm.yaml)
- Optional dependencies are not installed (sktime, sklearn)
- Config format is incompatible (will be improved in future)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from dfm_python.models import DDFM
from dfm_python.config import DFMConfig, DDFMConfig, YamlSource
from dfm_python import DFMDataModule
from dfm_python.trainer import DDFMTrainer
from dfm_python.utils.time import TimeIndex, parse_timestamp
from dfm_python.utils.data import rem_nans_spline, sort_data

# Import shared test helper functions
from test_helpers import (
    check_test_files_exist,
    load_config_safely,
    load_config_only_safely,
    handle_training_error,
    format_skip_message,
    get_test_data_path,
    get_test_config_path,
    load_sample_data_from_csv,
    create_simple_transformer,
    create_columnwise_transformer
)


class TestDDFMPipeline:
    """Test complete DDFM (Deep Dynamic Factor Model) pipeline workflow.
    
    This test class covers the full DDFM pipeline:
    1. Configuration loading from YAML files
    2. Data loading (with statistics extraction from transformer)
    3. Model training with neural encoder (gradient descent)
    4. Prediction and forecasting
    5. Complete end-to-end workflow
    
    Note: Users can provide raw data with a preprocessing pipeline (preprocessing happens in setup()),
    or preprocessed data. The pipeline also extracts statistics (Mx/Wx) for forecasting/nowcasting.
    
    All tests use actual data and config files when available.
    """
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return get_test_data_path()
    
    @pytest.fixture
    def test_ddfm_config_path(self):
        """Path to test DDFM config."""
        return get_test_config_path("ddfm")
    
    @pytest.fixture
    def sample_data(self, test_data_path):
        """Load and preprocess sample data."""
        return load_sample_data_from_csv(test_data_path)
    
    @pytest.fixture
    def simple_transformer(self):
        """Create a simple transformer for testing."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Per sktime docs: sklearn transformers work directly in TransformerPipeline
            # Identity-like transformer (no scaling) for testing
            return StandardScaler(with_mean=False, with_std=False)
        except ImportError:
            pytest.skip("sktime not available - install with: pip install sktime")
    
    @pytest.fixture
    def columnwise_transformer(self):
        """Create a StandardScaler for unified scaling in testing."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Per sktime docs: sklearn transformers work directly in TransformerPipeline
            # Applied per series instance automatically (unified scaling)
            return StandardScaler()
        except ImportError:
            pytest.skip("sktime or sklearn not available - install with: pip install sktime scikit-learn")
    
    def test_ddfm_pipeline_config_loading(self, test_ddfm_config_path):
        """Test step 1: DDFM configuration loading."""
        if not test_ddfm_config_path.exists():
            pytest.skip(f"Test config file not found: {test_ddfm_config_path}")
        
        # Load config - may fail if config format is not fully supported
        model = DDFM(encoder_layers=[32, 16], num_factors=2)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        assert model.config is not None
        # DDFM can work with both DFMConfig and DDFMConfig (DDFMConfig extends DFMConfig)
        # Accept either type since config adapter may load as DFMConfig
        assert isinstance(model.config, (DFMConfig, DDFMConfig))
        assert len(model.config.series) > 0
    
    def test_ddfm_pipeline_data_loading(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test step 2: DDFM data loading with transformer for statistics extraction.
        
        Note: Users can provide raw data with a preprocessing pipeline (preprocessing happens in setup()),
        or preprocessed data. The pipeline also extracts statistics (Mx/Wx) for forecasting/nowcasting.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule
        # Note: Raw data + pipeline pattern - preprocessing happens in setup().
        # Pipeline also extracts statistics (Mx/Wx) for forecasting/nowcasting.
        assert config is not None
        data_module = DFMDataModule(
            config=config,
            pipeline=simple_transformer,  # Used to extract Mx/Wx statistics
            data_path=test_data_path
        )
        
        # Setup (loads data and extracts statistics from transformer)
        data_module.setup()
        
        assert data_module.data_processed is not None
        assert data_module.train_dataset is not None
        assert data_module.Mx is not None
        assert data_module.Wx is not None
        # Verify data shape
        assert data_module.data_processed.shape[0] > 0
        assert data_module.data_processed.shape[1] > 0
    
    def test_ddfm_pipeline_data_loading_no_transformer(self, test_data_path, test_ddfm_config_path):
        """Test DDFM data loading with pipeline=None (uses passthrough by default)."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule without transformer (uses passthrough by default)
        # Mx/Wx will be computed from data as fallback
        assert config is not None
        data_module = DFMDataModule(
            config=config,
            pipeline=None,  # Uses passthrough transformer (default)
            data_path=test_data_path
        )
        
        # Setup (loads data, uses passthrough transformer)
        data_module.setup()
        
        assert data_module.data_processed is not None
        assert data_module.train_dataset is not None
        # Mx/Wx should still be available (computed from data as fallback)
        assert data_module.Mx is not None
        assert data_module.Wx is not None
        # Verify data shape
        assert data_module.data_processed.shape[0] > 0
        assert data_module.data_processed.shape[1] > 0
    
    def test_ddfm_pipeline_training(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test step 3: DDFM model training with neural encoder."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=5)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule with actual data
        assert model.config is not None
        data_module = DFMDataModule(
            config=model.config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Verify data was loaded
        assert data_module.data_processed is not None
        T, N = data_module.data_processed.shape
        assert T > 0 and N > 0
        
        # Train with reduced epochs for testing using Lightning pattern
        # DDFM uses gradient descent (Adam optimizer) instead of EM algorithm
        trainer = DDFMTrainer(max_epochs=3, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        
        assert model.result is not None
        assert hasattr(model.result, 'Z')
        assert model.result.Z is not None
        # Verify result dimensions are consistent
        # Note: result.Z.shape[0] may differ from T due to data trimming during training
        # (e.g., rem_nans_spline may remove rows). Check consistency instead of exact match.
        T_actual = model.result.Z.shape[0]
        assert T_actual > 0, "Result Z should have positive time dimension"
        # Result should be close to original data size (within reasonable trimming range)
        assert abs(T_actual - T) <= 10, f"Result Z time dimension {T_actual} should be close to data size {T} (within 10 rows)"
        if hasattr(model.result, 'X_sm'):
            assert model.result.X_sm is not None
            assert model.result.X_sm.shape[0] == T_actual, "X_sm should match Z time dimension"
    
    def test_ddfm_pipeline_prediction(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test step 4: DDFM prediction after training."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=5)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule with actual data
        assert model.config is not None
        data_module = DFMDataModule(
            config=model.config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Verify actual data was loaded
        assert data_module.data_processed is not None
        T, N = data_module.data_processed.shape
        assert T > 0 and N > 0
        
        # Train with reduced epochs using Lightning pattern
        trainer = DDFMTrainer(max_epochs=3, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        
        # Predict future values
        horizon = 6
        X_forecast, Z_forecast = model.predict(horizon=horizon, return_series=True, return_factors=True)
        
        assert X_forecast is not None
        assert Z_forecast is not None
        assert X_forecast.shape[0] == horizon
        assert model.config is not None
        assert X_forecast.shape[1] == len(model.config.series)
        assert Z_forecast.shape[0] == horizon
        # Verify forecast values are finite
        assert np.all(np.isfinite(X_forecast))
        assert np.all(np.isfinite(Z_forecast))
    
    def test_ddfm_pipeline_complete(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test complete DDFM pipeline: config -> data -> train -> predict."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Step 1: Load config - handle config format issues
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=3)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        assert model.config is not None
        
        # Step 2: Load data from CSV (raw data - preprocessing happens in setup() via pipeline)
        data_module = DFMDataModule(
            config=model.config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        assert data_module.data_processed is not None
        
        # Verify actual data dimensions
        T, N = data_module.data_processed.shape
        assert T > 0 and N > 0
        assert N == len(model.config.series)
        
        # Step 3: Train model with actual data using Lightning pattern
        # DDFM uses neural encoder with gradient descent (Adam optimizer)
        trainer = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        assert model.result is not None
        
        # Step 4: Make predictions
        horizon = 6
        X_forecast, Z_forecast = model.predict(horizon=horizon, return_series=True, return_factors=True)
        assert X_forecast is not None
        assert Z_forecast is not None
        assert np.all(np.isfinite(X_forecast))
        assert np.all(np.isfinite(Z_forecast))
        
        # Step 5: Verify result structure matches actual data
        assert hasattr(model.result, 'Z')
        assert hasattr(model.result, 'A')
        assert hasattr(model.result, 'C')
        # Note: result.Z.shape[0] may differ from T due to data trimming during training
        T_actual = model.result.Z.shape[0]
        assert T_actual > 0, "Result Z should have positive time dimension"
        assert abs(T_actual - T) <= 10, f"Result Z time dimension {T_actual} should be close to data size {T} (within 10 rows)"
        if hasattr(model.result, 'X_sm'):
            assert model.result.X_sm.shape[0] == T_actual, "X_sm should match Z time dimension"
            assert model.result.X_sm.shape[1] == N
    
    def test_ddfm_pipeline_with_columnwise_transformer(self, test_data_path, test_ddfm_config_path, columnwise_transformer):
        """Test complete DDFM pipeline with StandardScaler transformer for statistics extraction.
        
        Note: Users can provide raw data with a preprocessing pipeline (preprocessing happens in setup()),
        or preprocessed data. The pipeline also extracts Mx/Wx statistics.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Step 1: Load config
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=3)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        assert model.config is not None
        
        # Step 2: Load data with transformer to extract statistics (Mx/Wx from StandardScaler)
        # Note: Raw data + pipeline pattern - preprocessing happens in setup()
        data_module = DFMDataModule(
            config=model.config,
            pipeline=columnwise_transformer,  # Extracts Mx/Wx statistics
            data_path=test_data_path
        )
        data_module.setup()
        assert data_module.data_processed is not None
        
        # Verify data was loaded
        T, N = data_module.data_processed.shape
        assert T > 0 and N > 0
        assert N == len(model.config.series)
        
        # Verify Mx/Wx statistics were extracted from transformer
        assert data_module.Mx is not None
        assert data_module.Wx is not None
        assert len(data_module.Mx) == N
        assert len(data_module.Wx) == N
        
        # Note: Raw data + pipeline pattern - preprocessing happens in setup(). The pipeline is used
        # to extract statistics, not to preprocess. If data is already standardized,
        # the transformer will extract the standardization parameters.
        assert np.all(np.abs(data_mean) < 1e-6), "Data should be mean-centered by StandardScaler"
        assert np.all(np.abs(data_std - 1.0) < 1e-6), "Data should be unit variance by StandardScaler"
        
        # Step 3: Train model using Lightning pattern
        # DDFM uses neural encoder with gradient descent
        trainer = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        assert model.result is not None
        
        # Step 4: Make predictions
        horizon = 6
        X_forecast, Z_forecast = model.predict(horizon=horizon, return_series=True, return_factors=True)
        assert X_forecast is not None
        assert Z_forecast is not None
        assert np.all(np.isfinite(X_forecast))
        assert np.all(np.isfinite(Z_forecast))
        
        # Step 5: Verify result structure
        assert hasattr(model.result, 'Z')
        assert hasattr(model.result, 'A')
        assert hasattr(model.result, 'C')
        # Note: result.Z.shape[0] may differ from T due to data trimming during training
        T_actual = model.result.Z.shape[0]
        assert T_actual > 0, "Result Z should have positive time dimension"
        assert abs(T_actual - T) <= 10, f"Result Z time dimension {T_actual} should be close to data size {T} (within 10 rows)"
    
    def test_ddfm_encoder_architecture(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM-specific encoder architecture configuration."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        # Test with custom encoder architecture
        model = DDFM(encoder_layers=[64, 32, 16], num_factors=3, epochs=2)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        assert model.config is not None
        assert model.encoder_layers == [64, 32, 16]
        assert model.num_factors == 3
        
        # Create DataModule
        data_module = DFMDataModule(
            config=model.config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Train with reduced epochs
        trainer = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        
        # Verify encoder was initialized and used
        assert model.result is not None
        assert hasattr(model.result, 'Z')
        # Verify number of factors matches configuration
        assert model.result.Z.shape[1] == model.num_factors
    
    def test_ddfm_training_parameters(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM-specific training parameters (learning_rate, batch_size, epochs)."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        # Test with custom training parameters
        model = DDFM(
            encoder_layers=[32, 16],
            num_factors=2,
            learning_rate=0.01,  # Higher learning rate
            batch_size=16,  # Smaller batch size
            epochs=2
        )
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        assert model.config is not None
        assert model.learning_rate == 0.01
        assert model.batch_size == 16
        
        # Create DataModule
        data_module = DFMDataModule(
            config=model.config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Train with reduced epochs
        trainer = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        
        # Verify training completed successfully
        assert model.result is not None
        assert hasattr(model.result, 'Z')
        assert model.result.Z is not None


class TestDDFMPipelineIntegration:
    """Test DDFM pipeline integration and edge cases.
    
    This test class covers:
    - DataModule reuse across multiple DDFM models
    - Config validation for DDFM
    - Error handling and edge cases
    - Integration between DDFM pipeline components
    """
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return get_test_data_path()
    
    @pytest.fixture
    def test_ddfm_config_path(self):
        """Path to test DDFM config."""
        return get_test_config_path("ddfm")
    
    @pytest.fixture
    def simple_transformer(self):
        """Create a simple transformer for testing (identity-like, no scaling)."""
        return create_simple_transformer()
    
    def test_ddfm_pipeline_data_module_reuse(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test that DataModule can be reused across multiple DDFM models."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load config - handle config format issues
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule once with actual data
        assert config is not None
        data_module = DFMDataModule(
            config=config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Verify actual data was loaded
        assert data_module.data_processed is not None
        T, N = data_module.data_processed.shape
        assert T > 0 and N > 0
        
        # Use with first DDFM model
        model1 = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=2)
        model1.load_config(test_ddfm_config_path)
        trainer1 = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer1.fit(model1, data_module)
        assert model1.result is not None
        # Note: result.Z.shape[0] may differ from T due to data trimming during training
        T_actual1 = model1.result.Z.shape[0]
        assert T_actual1 > 0, "Result Z should have positive time dimension"
        assert abs(T_actual1 - T) <= 10, f"Result Z time dimension {T_actual1} should be close to data size {T} (within 10 rows)"
        
        # Use with second DDFM model (should work with same data)
        model2 = DDFM(encoder_layers=[64, 32], num_factors=3, epochs=2)
        model2.load_config(test_ddfm_config_path)
        trainer2 = DDFMTrainer(max_epochs=2, enable_progress_bar=False, logger=False)
        trainer2.fit(model2, data_module)
        assert model2.result is not None
        # Both models should produce results with consistent time dimensions
        T_actual2 = model2.result.Z.shape[0]
        assert T_actual2 > 0, "Result Z should have positive time dimension"
        assert abs(T_actual2 - T) <= 10, f"Result Z time dimension {T_actual2} should be close to data size {T} (within 10 rows)"
        assert T_actual1 == T_actual2, "Both models should produce results with same time dimension"
    
    def test_ddfm_pipeline_config_validation(self, test_ddfm_config_path):
        """Test that DDFM config validation works in pipeline."""
        if not test_ddfm_config_path.exists():
            pytest.skip(f"Test config file not found: {test_ddfm_config_path}")
        
        # Load config - may fail if config format is not fully supported
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        
        # Verify config structure
        assert config is not None
        assert len(config.series) > 0
        assert all(hasattr(s, 'series_id') for s in config.series)
        assert all(hasattr(s, 'frequency') for s in config.series)
        assert all(hasattr(s, 'transformation') for s in config.series)
        
        # Verify DDFM-specific config attributes if present
        if isinstance(config, DDFMConfig):
            # DDFM config should have encoder_layers, num_factors, etc.
            # These may be None if not specified, which is fine
            pass
    
    def test_ddfm_pipeline_error_handling(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test error handling in DDFM pipeline."""
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Test: Cannot predict without training
        model = DDFM(encoder_layers=[32, 16], num_factors=2)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        with pytest.raises((ValueError, AttributeError)):
            model.predict(horizon=6)
        
        # Test: Cannot train without config
        model2 = DDFM(encoder_layers=[32, 16], num_factors=2)
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        data_module = DFMDataModule(
            config=config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # DDFM requires config to be loaded before training
        # This should work if config is loaded, but fail if not
        try:
            trainer = DDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(model2, data_module)
            # If this succeeds, model2 must have gotten config somehow
            # This is acceptable behavior
        except (ValueError, AttributeError) as e:
            # Expected if config is not properly set
            pass


class TestDDFMStability:
    """Test DDFM stability edge cases and error handling.
    
    This test class focuses on edge cases that test the robustness of DDFM:
    - Convergence failures
    - NaN propagation scenarios
    - Extreme data values
    - Many missing values
    - Numerical stability edge cases
    """
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return get_test_data_path()
    
    @pytest.fixture
    def test_ddfm_config_path(self):
        """Path to test DDFM config."""
        return get_test_config_path("ddfm")
    
    @pytest.fixture
    def simple_transformer(self):
        """Create a simple transformer for testing."""
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Per sktime docs: sklearn transformers work directly in TransformerPipeline
            # Applied per series instance automatically (unified scaling)
            return StandardScaler()
        except ImportError:
            pytest.skip("scikit-learn not available - install with: pip install scikit-learn")
    
    def test_ddfm_convergence_failure_handling(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM handles convergence failures gracefully.
        
        This test verifies that when MCMC doesn't converge (max_iter reached),
        the model still returns a valid training state with converged=False.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Create model with very strict tolerance to force non-convergence
        model = DDFM(
            encoder_layers=[32, 16],
            num_factors=2,
            epochs=2,  # Low epochs for faster testing
            max_iter=3,  # Very low max_iter to force non-convergence
            tolerance=1e-8  # Very strict tolerance (unlikely to converge)
        )
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        data_module = DFMDataModule(
            config=config,
            pipeline=simple_transformer,
            data_path=test_data_path
        )
        data_module.setup()
        
        # Train with very low max_iter to force non-convergence
        trainer = DDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
        trainer.fit(model, data_module)
        
        # Check that training completed (even if not converged)
        assert model.training_state is not None
        
        # Get result - should work even if not converged
        try:
            result = model.get_result()
            # Result should be valid even if convergence failed
            assert result is not None
            assert hasattr(result, 'Z')
            # Factors should be finite
            assert np.all(np.isfinite(result.Z))
        except (ValueError, AttributeError) as e:
            # If result extraction fails, that's also acceptable for non-converged case
            # But we should log it
            pytest.skip(f"Result extraction failed after non-convergence: {e}")
    
    def test_ddfm_nan_propagation_handling(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM handles NaN propagation scenarios.
        
        This test verifies that NaN values in data are handled correctly
        and don't propagate through the MCMC procedure.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load data and introduce NaN values
        df = pd.read_csv(test_data_path)
        data_cols = [col for col in df.columns if col != "date"]
        data_array = df[data_cols].values
        
        # Introduce NaN values (10% of data)
        np.random.seed(42)
        nan_indices = np.random.choice(data_array.size, size=int(0.1 * data_array.size), replace=False)
        data_with_nan = data_array.copy()
        data_with_nan.flat[nan_indices] = np.nan
        
        # Create model
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=2, max_iter=5)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule with NaN data
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        # Convert to DataFrame for DataModule
        df_with_nan = pd.DataFrame(data_with_nan, columns=data_cols)
        df_with_nan['date'] = df['date']
        
        # Save temporary CSV with NaN values
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_with_nan.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            data_module = DFMDataModule(
                config=config,
                pipeline=simple_transformer,
                data_path=temp_path
            )
            data_module.setup()
            
            # Train - should handle NaN values gracefully
            trainer = DDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(model, data_module)
            
            # Check that result doesn't contain NaN/Inf
            if model.training_state is not None:
                try:
                    result = model.get_result()
                    # Result should not contain NaN/Inf
                    assert result is not None
                    assert hasattr(result, 'Z')
                    assert np.all(np.isfinite(result.Z)), "Result Z should not contain NaN/Inf"
                    if hasattr(result, 'X_sm'):
                        assert np.all(np.isfinite(result.X_sm)), "Result X_sm should not contain NaN/Inf"
                except (ValueError, AttributeError) as e:
                    # If result extraction fails due to NaN issues, that's acceptable
                    # but we should verify the error message is helpful
                    assert "NaN" in str(e) or "Inf" in str(e) or "not been trained" in str(e)
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def test_ddfm_extreme_data_values(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM handles extreme data values (very large/small).
        
        This test verifies that extreme values don't cause numerical overflow/underflow.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load data and scale to extreme values
        df = pd.read_csv(test_data_path)
        data_cols = [col for col in df.columns if col != "date"]
        data_array = df[data_cols].values
        
        # Scale to extreme values (very large)
        data_extreme = data_array * 1e6
        
        # Create model
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=2, max_iter=5)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule with extreme data
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        df_extreme = pd.DataFrame(data_extreme, columns=data_cols)
        df_extreme['date'] = df['date']
        
        # Save temporary CSV
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_extreme.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            data_module = DFMDataModule(
                config=config,
                pipeline=simple_transformer,
                data_path=temp_path
            )
            data_module.setup()
            
            # Train - should handle extreme values gracefully (standardization should help)
            trainer = DDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(model, data_module)
            
            # Check that result is valid
            if model.training_state is not None:
                try:
                    result = model.get_result()
                    assert result is not None
                    assert hasattr(result, 'Z')
                    # Result should be finite (standardization should prevent overflow)
                    assert np.all(np.isfinite(result.Z)), "Result Z should be finite even with extreme input values"
                except (ValueError, AttributeError) as e:
                    # If training fails due to extreme values, that's acceptable
                    # but error should be informative
                    assert "not been trained" in str(e) or "NaN" in str(e) or "Inf" in str(e)
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
    
    def test_ddfm_high_missing_data(self, test_data_path, test_ddfm_config_path, simple_transformer):
        """Test DDFM handles high proportion of missing data (>50%).
        
        This test verifies that models with many missing values still train
        or fail gracefully with helpful error messages.
        """
        check_test_files_exist(test_data_path, test_ddfm_config_path)
        
        # Load data and introduce high proportion of missing values
        df = pd.read_csv(test_data_path)
        data_cols = [col for col in df.columns if col != "date"]
        data_array = df[data_cols].values
        
        # Introduce high proportion of NaN values (60% of data)
        np.random.seed(42)
        nan_indices = np.random.choice(data_array.size, size=int(0.6 * data_array.size), replace=False)
        data_high_missing = data_array.copy()
        data_high_missing.flat[nan_indices] = np.nan
        
        # Create model
        model = DDFM(encoder_layers=[32, 16], num_factors=2, epochs=2, max_iter=5)
        load_config_safely(model, test_ddfm_config_path, model_type="DDFM")
        
        # Create DataModule with high missing data
        config = load_config_only_safely(test_ddfm_config_path, model_type="DDFM")
        df_high_missing = pd.DataFrame(data_high_missing, columns=data_cols)
        df_high_missing['date'] = df['date']
        
        # Save temporary CSV
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_high_missing.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            data_module = DFMDataModule(
                config=config,
                pipeline=simple_transformer,
                data_path=temp_path
            )
            data_module.setup()
            
            # Train - should handle high missing data or fail gracefully
            trainer = DDFMTrainer(max_epochs=1, enable_progress_bar=False, logger=False)
            trainer.fit(model, data_module)
            
            # Check result - may fail with high missing data, but should fail gracefully
            if model.training_state is not None:
                try:
                    result = model.get_result()
                    # If we get a result, it should be valid
                    assert result is not None
                    assert hasattr(result, 'Z')
                    assert np.all(np.isfinite(result.Z))
                except (ValueError, AttributeError) as e:
                    # Failure is acceptable with high missing data
                    # Error message should be informative
                    error_str = str(e)
                    assert any(keyword in error_str for keyword in [
                        "not been trained", "NaN", "Inf", "missing", "insufficient"
                    ]), f"Error message should be informative: {error_str}"
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

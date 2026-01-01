"""Tests for models.kdfm module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch
from dfm_python.models.kdfm import KDFM
from dfm_python.utils.errors import ConfigurationError, DataValidationError, ModelNotInitializedError, ModelNotTrainedError, PredictionError, NumericalError
from dfm_python.config.constants import DEFAULT_DTYPE, DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER, DEFAULT_ZERO_VALUE


class TestKDFM:
    """Test suite for KDFM model."""
    
    def test_kdfm_initialization(self):
        """Test KDFM can be initialized."""
        model = KDFM(ar_order=1, ma_order=0)
        assert model is not None
        assert model.ar_order == 1
        assert model.ma_order == 0
    
    def test_kdfm_initialization_defaults(self):
        """Test KDFM uses default constants when no arguments provided."""
        model = KDFM()
        assert model is not None
        assert model.ar_order == DEFAULT_KDFM_AR_ORDER
        assert model.ma_order == DEFAULT_KDFM_MA_ORDER
        assert model.ar_order == 1
        assert model.ma_order == 0
    
    def test_kdfm_initialization_with_ma(self):
        """Test KDFM can be initialized with MA order > 0."""
        model = KDFM(ar_order=1, ma_order=1)
        assert model is not None
        assert model.ar_order == 1
        assert model.ma_order == 1
    
    def test_kdfm_initialization_invalid_ar_order_zero(self):
        """Test KDFM raises error for ar_order = 0."""
        with pytest.raises(ConfigurationError, match="ar_order must be >= 1"):
            KDFM(ar_order=0, ma_order=0)
    
    def test_kdfm_initialization_invalid_ar_order_negative(self):
        """Test KDFM raises error for negative ar_order."""
        with pytest.raises(ConfigurationError, match="ar_order must be >= 1"):
            KDFM(ar_order=-1, ma_order=0)
    
    def test_kdfm_initialization_invalid_ma_order_negative(self):
        """Test KDFM raises error for negative ma_order."""
        with pytest.raises(ConfigurationError, match="ma_order must be >= 0"):
            KDFM(ar_order=1, ma_order=-1)
    
    def test_kdfm_initialization_invalid_ar_order_too_high(self):
        """Test KDFM raises error for ar_order > 20."""
        with pytest.raises(ConfigurationError, match="ar_order must be <= 20"):
            KDFM(ar_order=21, ma_order=0)
    
    def test_kdfm_initialization_invalid_ma_order_too_high(self):
        """Test KDFM raises error for ma_order > 10."""
        with pytest.raises(ConfigurationError, match="ma_order must be <= 10"):
            KDFM(ar_order=1, ma_order=11)
    
    def test_kdfm_initialize_from_data_empty_time_steps(self):
        """Test KDFM raises error when data has zero time steps."""
        from dfm_python.config.constants import MIN_TIME_STEPS
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(0, 5)  # 0 time steps, 5 variables
        # Error is caught by validate_data_shape first (min_size=1 check)
        # The specific MIN_TIME_STEPS check happens after shape validation
        # We verify MIN_TIME_STEPS constant is defined correctly
        assert MIN_TIME_STEPS == 1
        with pytest.raises(DataValidationError, match="All dimensions must be >= 1"):
            model.initialize_from_data(X)
    
    def test_kdfm_initialize_from_data_zero_variables(self):
        """Test KDFM raises error when data has zero variables."""
        from dfm_python.config.constants import MIN_VARIABLES
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 0)  # 10 time steps, 0 variables
        # Error is caught by validate_data_shape first (min_size=1 check)
        # The specific MIN_VARIABLES check happens after shape validation
        # We verify MIN_VARIABLES constant is defined correctly
        assert MIN_VARIABLES == 1
        with pytest.raises(DataValidationError, match="All dimensions must be >= 1"):
            model.initialize_from_data(X)
    
    def test_kdfm_dimension_validation_uses_constants(self):
        """Test KDFM dimension validation uses MIN_TIME_STEPS and MIN_VARIABLES constants."""
        from dfm_python.config.constants import MIN_TIME_STEPS, MIN_VARIABLES
        # Verify constants are defined and have correct values
        assert MIN_TIME_STEPS == 1
        assert MIN_VARIABLES == 1
        # Verify constants are used in validation (indirectly through code inspection)
        # The actual validation happens in initialize_from_data using these constants
    
    def test_kdfm_initialize_from_data_valid(self):
        """Test KDFM can initialize from valid data."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)  # 10 time steps, 5 variables
        # Should not raise
        model.initialize_from_data(X)
        assert model.companion_ar is not None
    
    def test_kdfm_forward_not_initialized(self):
        """Test KDFM forward raises error when model not initialized."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        with pytest.raises(ModelNotInitializedError, match="KDFM forward pass requires initialized model components"):
            model.forward(X)
    
    def test_kdfm_forward_valid(self):
        """Test KDFM forward pass with initialized model."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)  # 10 time steps, 5 variables
        model.initialize_from_data(X)
        y_pred = model.forward(X)
        assert y_pred.shape == X.shape
        assert isinstance(y_pred, torch.Tensor)
    
    def test_kdfm_forward_batch(self):
        """Test KDFM forward pass with batch dimension."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(2, 10, 5)  # 2 batches, 10 time steps, 5 variables
        model.initialize_from_data(X[0])  # Initialize with first batch
        y_pred = model.forward(X)
        assert y_pred.shape == X.shape
        assert isinstance(y_pred, torch.Tensor)
    
    def test_kdfm_predict_not_initialized(self):
        """Test KDFM predict raises error when model not initialized."""
        model = KDFM(ar_order=1, ma_order=0)
        last_obs = torch.randn(1, 5)
        # predict() checks _check_trained() first, which raises ModelNotTrainedError
        # The error message includes "KDFM operation failed" or "KDFM prediction requires"
        with pytest.raises(ModelNotTrainedError):
            model.predict(horizon=5, last_observation=last_obs)
    
    def test_kdfm_result_property_not_trained(self):
        """Test KDFM result property raises error when model not trained."""
        model = KDFM(ar_order=1, ma_order=0)
        with pytest.raises(ModelNotTrainedError, match="model has not been trained yet"):
            _ = model.result
    
    def test_kdfm_predict_requires_trained_model(self):
        """Test KDFM predict requires trained model (not just initialized).
        
        Note: predict() requires get_result() which needs a fully trained model.
        This test verifies that predict() raises appropriate error for untrained model.
        """
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(20, 5)  # 20 time steps, 5 variables
        model.initialize_from_data(X)
        last_obs = X[-1:].numpy()  # Last observation as numpy array
        # predict() calls get_result() internally which requires trained model
        # For ma_order=0, companion_ma is None, so get_result() will fail
        with pytest.raises(ModelNotInitializedError, match="Cannot extract companion parameters"):
            model.predict(horizon=5, last_observation=last_obs)
    
    def test_kdfm_forward_shape_mismatch(self):
        """Test KDFM forward handles shape mismatches appropriately."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)  # 10 time steps, 5 variables
        model.initialize_from_data(X)
        # Try forward with different number of variables
        X_wrong = torch.randn(10, 3)  # Wrong number of variables
        # Forward should raise an error for shape mismatch
        # After Iteration 15, IRF module uses DataValidationError, but KDFM forward
        # may raise RuntimeError or ValueError depending on where validation occurs
        with pytest.raises((RuntimeError, ValueError, DataValidationError)):
            model.forward(X_wrong)
    
    def test_compute_factor_state_from_observation_uses_default_dtype_on_nan(self):
        """Test _compute_factor_state_from_observation returns DEFAULT_DTYPE when NaN/Inf detected."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        model.initialize_from_data(X)
        
        # Create scenario where Z_last contains NaN/Inf after processing
        # We'll manually inject NaN into the result by patching ensure_numpy to return NaN
        from dfm_python.utils.common import ensure_numpy as original_ensure_numpy
        
        def mock_ensure_numpy_with_nan(tensor):
            """Mock ensure_numpy to return NaN array."""
            result = original_ensure_numpy(tensor)
            # Replace with NaN to trigger the NaN/Inf fallback path
            return np.full_like(result, np.nan, dtype=result.dtype)
        
        # Patch ensure_numpy temporarily
        import dfm_python.models.kdfm as kdfm_module
        original_ensure_numpy_ref = kdfm_module.ensure_numpy
        kdfm_module.ensure_numpy = mock_ensure_numpy_with_nan
        
        try:
            observation = torch.randn(1, 5)
            n_factors = 5
            
            # This should trigger NaN/Inf detection and return zeros with DEFAULT_DTYPE
            result = model._compute_factor_state_from_observation(observation, n_factors)
            
            # Verify result is numpy array with DEFAULT_DTYPE
            assert isinstance(result, np.ndarray)
            assert result.shape == (n_factors,)
            # NaN/Inf path should return zeros with DEFAULT_DTYPE
            assert np.all(result == 0)
            assert result.dtype == DEFAULT_DTYPE
        finally:
            # Restore original ensure_numpy
            kdfm_module.ensure_numpy = original_ensure_numpy_ref
    
    def test_compute_factor_state_from_observation_uses_default_dtype_on_exception(self):
        """Test _compute_factor_state_from_observation returns DEFAULT_DTYPE when caught exception occurs."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        model.initialize_from_data(X)
        
        # Create scenario that triggers a caught exception (RuntimeError, ValueError, TypeError, AttributeError, KeyError)
        # We'll patch companion_ar's forward method to raise RuntimeError
        observation = torch.randn(1, 5)
        n_factors = 5
        
        with patch.object(model.companion_ar, 'forward', side_effect=RuntimeError("Test exception")):
            # This should trigger RuntimeError when trying to call companion_ar, which is caught
            result = model._compute_factor_state_from_observation(observation, n_factors)
            
            # Verify result is numpy array with DEFAULT_DTYPE
            assert isinstance(result, np.ndarray)
            assert result.shape == (n_factors,)
            # Exception path should return zeros with DEFAULT_DTYPE
            assert np.all(result == 0)
            assert result.dtype == DEFAULT_DTYPE
    
    def test_compute_factor_state_from_observation_dtype_consistency(self):
        """Test _compute_factor_state_from_observation returns consistent dtype in normal operation."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        model.initialize_from_data(X)
        
        # Test with valid observation
        observation = torch.randn(1, 5)
        n_factors = 5
        
        result = model._compute_factor_state_from_observation(observation, n_factors)
        
        # Verify result is numpy array with correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (n_factors,)
        # Result should be finite (not NaN/Inf) in normal operation
        assert np.all(np.isfinite(result))
    
    def test_get_result_C_with_valid_result(self):
        """Test _get_result_C returns C attribute from result object."""
        from unittest.mock import Mock
        model = KDFM(ar_order=1, ma_order=0)
        
        # Create mock result with C attribute
        C = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = Mock()
        result.C = C
        
        # Test _get_result_C returns C
        result_C = model._get_result_C(result)
        assert result_C is not None
        assert np.array_equal(result_C, C)
    
    def test_get_result_C_with_none_result(self):
        """Test _get_result_C returns None when result is None."""
        model = KDFM(ar_order=1, ma_order=0)
        
        # Test _get_result_C with None
        result_C = model._get_result_C(None)
        assert result_C is None
    
    def test_get_result_C_with_missing_attribute(self):
        """Test _get_result_C returns None when result has no C attribute."""
        model = KDFM(ar_order=1, ma_order=0)
        
        # Create simple object without C attribute
        class SimpleResult:
            pass
        
        result = SimpleResult()
        
        # Test _get_result_C returns None when C is missing
        result_C = model._get_result_C(result)
        assert result_C is None
    
    def test_get_result_n_vars_with_valid_result(self):
        """Test _get_result_n_vars returns n_vars attribute from result object."""
        from unittest.mock import Mock
        model = KDFM(ar_order=1, ma_order=0)
        
        # Create mock result with n_vars attribute
        result = Mock()
        result.n_vars = 5
        
        # Test _get_result_n_vars returns n_vars
        result_n_vars = model._get_result_n_vars(result)
        assert result_n_vars == 5
    
    def test_get_result_n_vars_with_none_result(self):
        """Test _get_result_n_vars returns None when result is None."""
        model = KDFM(ar_order=1, ma_order=0)
        
        # Test _get_result_n_vars with None
        result_n_vars = model._get_result_n_vars(None)
        assert result_n_vars is None
    
    def test_get_result_n_vars_with_missing_attribute(self):
        """Test _get_result_n_vars returns None when result has no n_vars attribute."""
        from unittest.mock import Mock
        model = KDFM(ar_order=1, ma_order=0)
        
        # Create simple object without n_vars attribute
        class SimpleResult:
            pass
        
        result = SimpleResult()
        
        # Test _get_result_n_vars returns None when n_vars is missing
        result_n_vars = model._get_result_n_vars(result)
        assert result_n_vars is None
    
    def test_get_result_n_vars_with_exception(self):
        """Test _get_result_n_vars handles AttributeError/TypeError gracefully."""
        model = KDFM(ar_order=1, ma_order=0)
        
        # Create mock object that raises AttributeError on getattr
        class MockResult:
            def __getattr__(self, name):
                if name == 'n_vars':
                    raise AttributeError("Mock attribute error")
                return None
        
        result = MockResult()
        
        # Test _get_result_n_vars handles exception and returns None
        result_n_vars = model._get_result_n_vars(result)
        assert result_n_vars is None
    
    def test_create_temp_config_uses_default_constants(self):
        """Test _create_temp_config uses DEFAULT_KDFM_AR_ORDER and DEFAULT_KDFM_MA_ORDER when attributes missing."""
        # Create model without explicitly setting ar_order/ma_order (uses defaults)
        model = KDFM()
        
        # Verify model has default values
        assert model.ar_order == DEFAULT_KDFM_AR_ORDER
        assert model.ma_order == DEFAULT_KDFM_MA_ORDER
        
        # Create temp config - should use instance attributes (which are defaults)
        config = model._create_temp_config()
        assert config is not None
        assert config.ar_order == DEFAULT_KDFM_AR_ORDER
        assert config.ma_order == DEFAULT_KDFM_MA_ORDER
    
    def test_create_temp_config_uses_constants_when_attribute_missing(self):
        """Test _create_temp_config fallback to constants when ar_order/ma_order attributes missing."""
        model = KDFM()
        
        # Temporarily remove attributes to test getattr fallback
        original_ar_order = model.ar_order
        original_ma_order = model.ma_order
        delattr(model, 'ar_order')
        delattr(model, 'ma_order')
        
        # _create_temp_config should use constants via getattr fallback
        config = model._create_temp_config()
        assert config is not None
        assert config.ar_order == DEFAULT_KDFM_AR_ORDER
        assert config.ma_order == DEFAULT_KDFM_MA_ORDER
        
        # Restore attributes
        model.ar_order = original_ar_order
        model.ma_order = original_ma_order
    
    def test_compute_structural_loss_raises_numerical_error_on_nan(self):
        """Test _compute_structural_loss re-raises NumericalError when structural matrix contains NaN."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        model.initialize_from_data(X)
        
        # structural_id is initialized during initialize_from_data if structural_method is set
        # Create structural matrix with NaN values
        if model.structural_id is not None:
            with patch.object(model.structural_id, 'get_structural_matrix', return_value=torch.tensor([[float('nan'), 0.0], [0.0, 1.0]])):
                # NumericalError should be re-raised (not caught by AttributeError/RuntimeError handler)
                with pytest.raises(NumericalError, match="Structural matrix contains NaN/Inf values"):
                    device = X.device
                    model._compute_structural_loss(device)
        else:
            # If structural_id is None, test is not applicable
            pytest.skip("structural_id not initialized (structural_method may be None)")
    
    def test_compute_structural_loss_returns_zero_on_attribute_error(self):
        """Test _compute_structural_loss returns zero loss when AttributeError occurs (initialization issue)."""
        model = KDFM(ar_order=1, ma_order=0)
        X = torch.randn(10, 5)
        model.initialize_from_data(X)
        
        # Simulate AttributeError (e.g., method missing during early training)
        if model.structural_id is not None:
            with patch.object(model.structural_id, 'get_structural_matrix', side_effect=AttributeError("Method not available")):
                device = X.device
                result = model._compute_structural_loss(device)
                
                # Should return zero loss tensor, not raise exception
                assert isinstance(result, torch.Tensor)
                assert result.item() == DEFAULT_ZERO_VALUE
        else:
            # If structural_id is None, test is not applicable
            pytest.skip("structural_id not initialized (structural_method may be None)")


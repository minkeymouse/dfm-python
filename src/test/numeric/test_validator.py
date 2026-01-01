"""Tests for numeric.validator module."""

import pytest
import numpy as np
import torch
from dfm_python.numeric.validator import (
    validate_no_nan_inf,
    validate_data_shape,
    validate_model_components,
    validate_companion_stability,
    validate_learning_rate,
    validate_batch_size,
    validate_horizon,
    validate_irf_horizon,
    validate_ar_order,
    validate_ma_order,
)
from dfm_python.utils.errors import DataValidationError, ModelNotInitializedError, ConfigurationError
from dfm_python.config.constants import DEFAULT_MIN_DELTA


class TestValidator:
    """Test suite for numeric validator utilities."""
    
    def test_validate_no_nan_inf(self):
        """Test NaN/Inf validation."""
        # Test valid data (no NaN/Inf)
        valid_data = np.array([1.0, 2.0, 3.0])
        validate_no_nan_inf(valid_data, name="test_data")  # Should not raise
        
        # Test with NaN
        nan_data = np.array([1.0, np.nan, 3.0])
        with pytest.raises(DataValidationError, match="contains NaN"):
            validate_no_nan_inf(nan_data, name="test_data")
        
        # Test with Inf
        inf_data = np.array([1.0, np.inf, 3.0])
        with pytest.raises(DataValidationError, match="contains Inf"):
            validate_no_nan_inf(inf_data, name="test_data")
        
        # Test with torch tensor
        valid_tensor = torch.tensor([1.0, 2.0, 3.0])
        validate_no_nan_inf(valid_tensor, name="test_tensor")  # Should not raise
        
        # Test torch tensor with NaN
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(DataValidationError, match="contains NaN"):
            validate_no_nan_inf(nan_tensor, name="test_tensor")
    
    def test_validate_data_shape(self):
        """Test data shape validation."""
        # Test valid 2D shape
        data_2d = np.random.randn(10, 5)
        shape = validate_data_shape(data_2d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (10, 5)
        
        # Test valid 3D shape
        data_3d = np.random.randn(2, 10, 5)
        shape = validate_data_shape(data_3d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (2, 10, 5)
        
        # Test invalid: too few dimensions
        data_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(DataValidationError, match="at least 2 dimensions"):
            validate_data_shape(data_1d, min_dims=2, max_dims=3, min_size=1)
        
        # Test invalid: too many dimensions
        data_4d = np.random.randn(2, 3, 10, 5)
        with pytest.raises(DataValidationError, match="at most 3 dimensions"):
            validate_data_shape(data_4d, min_dims=2, max_dims=3, min_size=1)
        
        # Test invalid: dimension too small
        data_small = np.random.randn(10, 0)  # Second dimension is 0
        with pytest.raises(DataValidationError, match="All dimensions must be >= 1"):
            validate_data_shape(data_small, min_dims=2, max_dims=3, min_size=1)
        
        # Test with torch tensor
        tensor_2d = torch.randn(10, 5)
        shape = validate_data_shape(tensor_2d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (10, 5)
        
        # Test invalid type
        with pytest.raises(DataValidationError, match="must be numpy array or torch Tensor"):
            validate_data_shape([1, 2, 3], min_dims=2, max_dims=3, min_size=1)
    
    def test_validate_model_components(self):
        """Test model component validation."""
        # Create mock component objects
        class MockComponent:
            pass
        
        companion_ar = MockComponent()
        structural_id = MockComponent()
        
        # Test valid: all components present
        validate_model_components(
            companion_ar=companion_ar,
            structural_id=structural_id,
            model_name="test_model"
        )  # Should not raise
        
        # Test invalid: companion_ar is None
        with pytest.raises(ModelNotInitializedError, match="requires initialized AR companion matrix"):
            validate_model_components(
                companion_ar=None,
                structural_id=structural_id,
                model_name="test_model"
            )
        
        # Test with optional companion_ma
        companion_ma = MockComponent()
        validate_model_components(
            companion_ar=companion_ar,
            companion_ma=companion_ma,
            structural_id=structural_id,
            model_name="test_model"
        )  # Should not raise
    
    def test_validate_companion_stability_invalid_dimensions(self):
        """Test validate_companion_stability raises DataValidationError for invalid dimensions."""
        # Test with 4D array (should raise DataValidationError)
        matrix_4d = np.random.randn(2, 3, 4, 5)
        with pytest.raises(DataValidationError, match="must be 2D or 3D"):
            validate_companion_stability(matrix_4d, model_name="test_model")
        
        # Test with 1D array (raises NumericalError during eigenvalue computation, not shape validation)
        # The shape validation only checks for > 2 dimensions, 1D arrays fail during eigvals computation
        matrix_1d = np.array([1.0, 2.0, 3.0])
        from dfm_python.utils.errors import NumericalError
        with pytest.raises(NumericalError, match="Cannot compute eigenvalues"):
            validate_companion_stability(matrix_1d, model_name="test_model")
    
    def test_validate_learning_rate_valid(self):
        """Test validate_learning_rate with valid values."""
        # Test valid learning rate
        result = validate_learning_rate(0.001)
        assert result == 0.001
        
        # Test with default min_lr (DEFAULT_MIN_DELTA)
        result = validate_learning_rate(DEFAULT_MIN_DELTA)
        assert result == DEFAULT_MIN_DELTA
        
        # Test with max_lr
        result = validate_learning_rate(0.5, max_lr=1.0)
        assert result == 0.5
    
    def test_validate_learning_rate_invalid_type(self):
        """Test validate_learning_rate raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be a number"):
            validate_learning_rate("0.001")
    
    def test_validate_learning_rate_negative(self):
        """Test validate_learning_rate raises ConfigurationError for negative values."""
        with pytest.raises(ConfigurationError, match="must be > 0"):
            validate_learning_rate(-0.001)
    
    def test_validate_learning_rate_too_small(self):
        """Test validate_learning_rate warns for very small values."""
        # Should warn but not raise
        result = validate_learning_rate(1e-7, min_lr=DEFAULT_MIN_DELTA)
        assert result == 1e-7
    
    def test_validate_learning_rate_too_large(self):
        """Test validate_learning_rate raises ConfigurationError for values exceeding max_lr."""
        with pytest.raises(ConfigurationError, match="is very large"):
            validate_learning_rate(2.0, max_lr=1.0)
    
    def test_validate_batch_size_valid(self):
        """Test validate_batch_size with valid values."""
        result = validate_batch_size(32)
        assert result == 32
        
        result = validate_batch_size(1, min_size=1)
        assert result == 1
    
    def test_validate_batch_size_invalid_type(self):
        """Test validate_batch_size raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_batch_size(32.5)
    
    def test_validate_batch_size_too_small(self):
        """Test validate_batch_size raises ConfigurationError for values below min_size."""
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            validate_batch_size(0, min_size=1)
    
    def test_validate_horizon_valid(self):
        """Test validate_horizon with valid values."""
        result = validate_horizon(10)
        assert result == 10
        
        result = validate_horizon(1, min_horizon=1, max_horizon=100)
        assert result == 1
    
    def test_validate_horizon_invalid_type(self):
        """Test validate_horizon raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_horizon(10.5)
    
    def test_validate_horizon_too_small(self):
        """Test validate_horizon raises ConfigurationError for values below min_horizon."""
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            validate_horizon(0, min_horizon=1)
    
    def test_validate_horizon_too_large(self):
        """Test validate_horizon warns for very large values."""
        # Should warn but not raise
        result = validate_horizon(150, max_horizon=100)
        assert result == 150
    
    def test_validate_irf_horizon_valid(self):
        """Test validate_irf_horizon with valid values."""
        result = validate_irf_horizon(50)
        assert result == 50
        
        result = validate_irf_horizon(1, min_horizon=1, max_horizon=200)
        assert result == 1
    
    def test_validate_irf_horizon_invalid_type(self):
        """Test validate_irf_horizon raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_irf_horizon(50.5)
    
    def test_validate_irf_horizon_too_small(self):
        """Test validate_irf_horizon raises ConfigurationError for values below min_horizon."""
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            validate_irf_horizon(0, min_horizon=1)
    
    def test_validate_irf_horizon_too_large(self):
        """Test validate_irf_horizon warns for very large values."""
        # Should warn but not raise
        result = validate_irf_horizon(250, max_horizon=200)
        assert result == 250
    
    def test_validate_ar_order_valid(self):
        """Test validate_ar_order with valid values."""
        result = validate_ar_order(1)
        assert result == 1
        
        result = validate_ar_order(2, min_order=1, max_order=20)
        assert result == 2
    
    def test_validate_ar_order_invalid_type(self):
        """Test validate_ar_order raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_ar_order(1.5)
    
    def test_validate_ar_order_too_small(self):
        """Test validate_ar_order raises ConfigurationError for values below min_order."""
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            validate_ar_order(0, min_order=1)
    
    def test_validate_ar_order_too_large(self):
        """Test validate_ar_order raises ConfigurationError for values exceeding max_order."""
        with pytest.raises(ConfigurationError, match="must be <= 20"):
            validate_ar_order(25, max_order=20)
    
    def test_validate_ma_order_valid(self):
        """Test validate_ma_order with valid values."""
        result = validate_ma_order(0)
        assert result == 0
        
        result = validate_ma_order(1, min_order=0, max_order=10)
        assert result == 1
    
    def test_validate_ma_order_invalid_type(self):
        """Test validate_ma_order raises ConfigurationError for invalid type."""
        with pytest.raises(ConfigurationError, match="must be an integer"):
            validate_ma_order(1.5)
    
    def test_validate_ma_order_too_small(self):
        """Test validate_ma_order raises ConfigurationError for values below min_order."""
        with pytest.raises(ConfigurationError, match="must be >= 0"):
            validate_ma_order(-1, min_order=0)
    
    def test_validate_ma_order_too_large(self):
        """Test validate_ma_order raises ConfigurationError for values exceeding max_order."""
        with pytest.raises(ConfigurationError, match="must be <= 10"):
            validate_ma_order(15, max_order=10)


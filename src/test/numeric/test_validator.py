"""Tests for numeric.validator module."""

import pytest
import numpy as np
import torch
from dfm_python.numeric.validator import (
    validate_no_nan_inf,
    validate_data_shape,
    validate_horizon,
    validate_parameters_initialized,
    validate_dfm_initialization,
)
from dfm_python.utils.errors import DataValidationError, ModelNotInitializedError, ConfigurationError, NumericalError
from dfm_python.config.constants import DEFAULT_MIN_DELTA


class TestValidator:
    """Test suite for numeric validator utilities."""
    
    def test_validate_no_nan_inf(self):
        """Test NaN/Inf validation."""
        valid_data = np.array([1.0, 2.0, 3.0])
        validate_no_nan_inf(valid_data, name="test_data")
        
        nan_data = np.array([1.0, np.nan, 3.0])
        with pytest.raises(DataValidationError, match="contains NaN"):
            validate_no_nan_inf(nan_data, name="test_data")
        
        inf_data = np.array([1.0, np.inf, 3.0])
        with pytest.raises(DataValidationError, match="contains Inf"):
            validate_no_nan_inf(inf_data, name="test_data")
        
        valid_tensor = torch.tensor([1.0, 2.0, 3.0])
        validate_no_nan_inf(valid_tensor, name="test_tensor")
        
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(DataValidationError, match="contains NaN"):
            validate_no_nan_inf(nan_tensor, name="test_tensor")
    
    def test_validate_data_shape(self):
        """Test data shape validation."""
        data_2d = np.random.randn(10, 5)
        shape = validate_data_shape(data_2d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (10, 5)
        
        data_3d = np.random.randn(2, 10, 5)
        shape = validate_data_shape(data_3d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (2, 10, 5)
        
        data_1d = np.array([1.0, 2.0, 3.0])
        with pytest.raises(DataValidationError, match="at least 2 dimensions"):
            validate_data_shape(data_1d, min_dims=2, max_dims=3, min_size=1)
        
        data_4d = np.random.randn(2, 3, 10, 5)
        with pytest.raises(DataValidationError, match="at most 3 dimensions"):
            validate_data_shape(data_4d, min_dims=2, max_dims=3, min_size=1)
        
        data_small = np.random.randn(10, 0)
        with pytest.raises(DataValidationError, match="All dimensions must be >= 1"):
            validate_data_shape(data_small, min_dims=2, max_dims=3, min_size=1)
        
        tensor_2d = torch.randn(10, 5)
        shape = validate_data_shape(tensor_2d, min_dims=2, max_dims=3, min_size=1)
        assert shape == (10, 5)
        
        with pytest.raises(DataValidationError, match="must be numpy array or torch Tensor"):
            validate_data_shape([1, 2, 3], min_dims=2, max_dims=3, min_size=1)
    
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
        result = validate_horizon(150, max_horizon=100)
        assert result == 150
    
    def test_validate_parameters_initialized(self):
        """Test validate_parameters_initialized."""
        params = {
            'A': np.eye(2),
            'C': np.random.randn(3, 2),
            'Q': np.eye(2),
            'R': np.eye(3),
            'Z_0': np.zeros(2),
            'V_0': np.eye(2)
        }
        validate_parameters_initialized(params, model_name="TestModel")
        
        params_missing = {'A': None, 'C': None, 'Q': None, 'R': None, 'Z_0': None, 'V_0': None}
        with pytest.raises(ModelNotInitializedError):
            validate_parameters_initialized(params_missing, model_name="TestModel")
    
    def test_validate_dfm_initialization(self):
        """Test validate_dfm_initialization."""
        m, n = 3, 5
        A = np.eye(m)
        C = np.random.randn(n, m)
        Q = np.eye(m) * 0.1
        R = np.eye(n) * 0.1
        Z_0 = np.zeros(m)
        V_0 = np.eye(m)
        
        validate_dfm_initialization(A, C, Q, R, Z_0, V_0)
        
        A_nan = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(NumericalError):
            validate_dfm_initialization(A_nan, C, Q, R, Z_0, V_0)

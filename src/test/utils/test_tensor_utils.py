"""Tests for utils.tensor_utils module."""

import pytest
import numpy as np
import torch
from dfm_python.utils.tensor_utils import (
    extract_tensor_value,
    normalize_tensor_shape,
    validate_tensor_device,
    batch_tensor_operation
)
from dfm_python.utils.errors import DataValidationError


class TestExtractTensorValue:
    """Test suite for extract_tensor_value function."""
    
    def test_extract_tensor_value_scalar_tensor(self):
        """Test extract_tensor_value returns float for scalar tensor."""
        tensor = torch.tensor(3.14)
        value = extract_tensor_value(tensor)
        assert isinstance(value, float)
        assert value == pytest.approx(3.14)
    
    def test_extract_tensor_value_array_tensor(self):
        """Test extract_tensor_value returns numpy array for array tensor."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        value = extract_tensor_value(tensor)
        assert isinstance(value, np.ndarray)
        assert np.allclose(value, np.array([1.0, 2.0, 3.0]))
    
    def test_extract_tensor_value_numpy_array(self):
        """Test extract_tensor_value works with numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        value = extract_tensor_value(arr)
        assert isinstance(value, np.ndarray)
        assert np.array_equal(value, arr)
    
    def test_extract_tensor_value_scalar_numpy(self):
        """Test extract_tensor_value returns float for scalar numpy array."""
        arr = np.array(3.14)
        value = extract_tensor_value(arr)
        assert isinstance(value, float)
        assert value == pytest.approx(3.14)
    
    def test_extract_tensor_value_python_scalar(self):
        """Test extract_tensor_value returns scalar unchanged."""
        scalar = 3.14
        value = extract_tensor_value(scalar)
        assert value == scalar
    
    def test_extract_tensor_value_invalid_type(self):
        """Test extract_tensor_value raises error for invalid type."""
        with pytest.raises(DataValidationError, match="Expected Tensor"):
            extract_tensor_value("invalid")


class TestNormalizeTensorShape:
    """Test suite for normalize_tensor_shape function."""
    
    def test_normalize_tensor_shape_no_change(self):
        """Test normalize_tensor_shape returns unchanged tensor when ndim matches."""
        tensor = torch.randn(2, 3, 4)
        result = normalize_tensor_shape(tensor, expected_ndim=3)
        assert result.shape == (2, 3, 4)
        assert result is tensor
    
    def test_normalize_tensor_shape_add_dimensions(self):
        """Test normalize_tensor_shape adds dimensions when needed."""
        tensor = torch.randn(3, 4)
        result = normalize_tensor_shape(tensor, expected_ndim=3)
        assert result.shape == (1, 3, 4)
    
    def test_normalize_tensor_shape_add_multiple_dimensions(self):
        """Test normalize_tensor_shape adds multiple dimensions."""
        tensor = torch.randn(5,)
        result = normalize_tensor_shape(tensor, expected_ndim=3)
        assert result.shape == (1, 1, 5)
    
    def test_normalize_tensor_shape_numpy_array(self):
        """Test normalize_tensor_shape works with numpy array."""
        arr = np.random.randn(3, 4)
        result = normalize_tensor_shape(arr, expected_ndim=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 4)
    
    def test_normalize_tensor_shape_too_many_dimensions(self):
        """Test normalize_tensor_shape raises error for too many dimensions."""
        tensor = torch.randn(2, 3, 4, 5)
        with pytest.raises(DataValidationError, match="at most"):
            normalize_tensor_shape(tensor, expected_ndim=3)
    
    def test_normalize_tensor_shape_with_name(self):
        """Test normalize_tensor_shape uses name in error message."""
        tensor = torch.randn(2, 3, 4, 5)
        with pytest.raises(DataValidationError, match="test_tensor"):
            normalize_tensor_shape(tensor, expected_ndim=3, name="test_tensor")


class TestValidateTensorDevice:
    """Test suite for validate_tensor_device function."""
    
    def test_validate_tensor_device_correct(self):
        """Test validate_tensor_device passes for correct device."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        validate_tensor_device(tensor, expected_device=torch.device('cpu'))
        # Should not raise
    
    def test_validate_tensor_device_wrong_device(self):
        """Test validate_tensor_device raises error for wrong device."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        # Note: This test assumes CPU tensor, GPU device would fail
        # In practice, this would need actual GPU to test properly
        if torch.cuda.is_available():
            with pytest.raises(DataValidationError, match="device"):
                validate_tensor_device(tensor, expected_device=torch.device('cuda:0'))
    
    def test_validate_tensor_device_no_expected(self):
        """Test validate_tensor_device passes when no expected device."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        validate_tensor_device(tensor, expected_device=None)
        # Should not raise
    
    def test_validate_tensor_device_invalid_type(self):
        """Test validate_tensor_device raises error for non-tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(DataValidationError, match="must be a Tensor"):
            validate_tensor_device(arr)
    
    def test_validate_tensor_device_with_name(self):
        """Test validate_tensor_device uses name in error message."""
        arr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(DataValidationError, match="test_tensor"):
            validate_tensor_device(arr, name="test_tensor")


class TestBatchTensorOperation:
    """Test suite for batch_tensor_operation function."""
    
    def test_batch_tensor_operation_basic(self):
        """Test batch_tensor_operation applies operation to each batch item."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 batches, 2 features
        operation = lambda x: x * 2
        result = batch_tensor_operation(tensor, operation, batch_dim=0)
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        assert torch.allclose(result, expected)
    
    def test_batch_tensor_operation_sum(self):
        """Test batch_tensor_operation with sum operation."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        operation = lambda x: x.sum()
        result = batch_tensor_operation(tensor, operation, batch_dim=0)
        expected = torch.tensor([3.0, 7.0])
        assert torch.allclose(result, expected)
    
    def test_batch_tensor_operation_3d(self):
        """Test batch_tensor_operation with 3D tensor."""
        tensor = torch.randn(2, 3, 4)  # 2 batches, 3 time steps, 4 features
        operation = lambda x: x.mean(dim=0)  # Average over time
        result = batch_tensor_operation(tensor, operation, batch_dim=0)
        assert result.shape == (2, 4)  # 2 batches, 4 features
    
    def test_batch_tensor_operation_custom_batch_dim(self):
        """Test batch_tensor_operation with custom batch dimension.
        
        Note: batch_tensor_operation stacks results along batch_dim, so the result
        will have the same shape as the input except the batch dimension is preserved
        based on the operation result shape.
        """
        tensor = torch.randn(3, 2, 4)  # 3 features, 2 batches, 4 time steps
        operation = lambda x: x  # Return unchanged (preserves shape)
        result = batch_tensor_operation(tensor, operation, batch_dim=1)
        # Result should preserve original shape since operation returns unchanged
        assert result.shape == (3, 2, 4)


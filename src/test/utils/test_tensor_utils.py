"""Tests for utils.tensor_utils module.

NOTE: extract_tensor_value function was removed with utils.common module (2026-01-06).
These tests are kept for reference but marked as skipped.
"""

import pytest
import numpy as np
import torch
from dfm_python.utils.errors import DataValidationError

# Skip all tests - extract_tensor_value was removed with utils.common
pytestmark = pytest.mark.skip(reason="extract_tensor_value was removed with utils.common (2026-01-06)")


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


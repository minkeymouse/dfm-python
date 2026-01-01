"""Tests for utils.validation module."""

import pytest
import numpy as np
import torch
from dfm_python.utils.validation import (
    check_condition,
    check_not_none,
    check_has_attr,
    has_shape_with_min_dims
)
from dfm_python.utils.errors import NumericalError, ConfigurationError


class TestCheckCondition:
    """Test suite for check_condition."""
    
    def test_check_condition_passes(self):
        """Test check_condition passes when condition is True."""
        # Should not raise
        check_condition(True, ValueError, "Should not raise")
    
    def test_check_condition_raises(self):
        """Test check_condition raises when condition is False."""
        with pytest.raises(ValueError, match="Test error"):
            check_condition(False, ValueError, "Test error")
    
    def test_check_condition_with_details(self):
        """Test check_condition with details parameter."""
        with pytest.raises(NumericalError) as exc_info:
            check_condition(False, NumericalError, "Main error", details="Additional details")
        assert "Main error" in str(exc_info.value)
        assert "Additional details" in str(exc_info.value)


class TestCheckNotNone:
    """Test suite for check_not_none."""
    
    def test_check_not_none_passes(self):
        """Test check_not_none passes when value is not None."""
        check_not_none(42, "value")
        check_not_none([1, 2, 3], "list")
        check_not_none("string", "str")
    
    def test_check_not_none_raises_default(self):
        """Test check_not_none raises ValueError by default."""
        with pytest.raises(ValueError, match="value must not be None"):
            check_not_none(None, "value")
    
    def test_check_not_none_raises_custom(self):
        """Test check_not_none raises custom error class."""
        with pytest.raises(ConfigurationError, match="config must not be None"):
            check_not_none(None, "config", error_class=ConfigurationError)


class TestCheckHasAttr:
    """Test suite for check_has_attr."""
    
    def test_check_has_attr_passes(self):
        """Test check_has_attr passes when object has attribute."""
        class TestObj:
            def __init__(self):
                self.attr = 42
        
        obj = TestObj()
        check_has_attr(obj, "attr", "TestObj")
    
    def test_check_has_attr_raises_default(self):
        """Test check_has_attr raises AttributeError by default."""
        class TestObj:
            pass
        
        obj = TestObj()
        with pytest.raises(AttributeError, match="TestObj must have attribute 'missing_attr'"):
            check_has_attr(obj, "missing_attr", "TestObj")
    
    def test_check_has_attr_raises_custom(self):
        """Test check_has_attr raises custom error class."""
        class TestObj:
            pass
        
        obj = TestObj()
        with pytest.raises(ConfigurationError, match="Config must have attribute 'missing'"):
            check_has_attr(obj, "missing", "Config", error_class=ConfigurationError)


class TestHasShapeWithMinDims:
    """Test suite for has_shape_with_min_dims."""
    
    def test_has_shape_with_min_dims_numpy_1d_min_dims_1(self):
        """Test has_shape_with_min_dims returns True for 1D numpy array with min_dims=1."""
        arr = np.array([1, 2, 3])
        assert has_shape_with_min_dims(arr, min_dims=1) is True
    
    def test_has_shape_with_min_dims_numpy_1d_min_dims_2(self):
        """Test has_shape_with_min_dims returns False for 1D numpy array with min_dims=2."""
        arr = np.array([1, 2, 3])
        assert has_shape_with_min_dims(arr, min_dims=2) is False
    
    def test_has_shape_with_min_dims_numpy_2d_min_dims_2(self):
        """Test has_shape_with_min_dims returns True for 2D numpy array with min_dims=2."""
        arr = np.array([[1, 2], [3, 4]])
        assert has_shape_with_min_dims(arr, min_dims=2) is True
    
    def test_has_shape_with_min_dims_torch_1d_min_dims_1(self):
        """Test has_shape_with_min_dims returns True for 1D torch tensor with min_dims=1."""
        tensor = torch.tensor([1, 2, 3])
        assert has_shape_with_min_dims(tensor, min_dims=1) is True
    
    def test_has_shape_with_min_dims_torch_1d_min_dims_2(self):
        """Test has_shape_with_min_dims returns False for 1D torch tensor with min_dims=2."""
        tensor = torch.tensor([1, 2, 3])
        assert has_shape_with_min_dims(tensor, min_dims=2) is False
    
    def test_has_shape_with_min_dims_torch_2d_min_dims_2(self):
        """Test has_shape_with_min_dims returns True for 2D torch tensor with min_dims=2."""
        tensor = torch.tensor([[1, 2], [3, 4]])
        assert has_shape_with_min_dims(tensor, min_dims=2) is True
    
    def test_has_shape_with_min_dims_none(self):
        """Test has_shape_with_min_dims returns False for None."""
        assert has_shape_with_min_dims(None, min_dims=1) is False
        assert has_shape_with_min_dims(None, min_dims=2) is False
    
    def test_has_shape_with_min_dims_no_shape_attribute(self):
        """Test has_shape_with_min_dims returns False for object without shape attribute."""
        obj = object()
        assert has_shape_with_min_dims(obj, min_dims=1) is False
    
    def test_has_shape_with_min_dims_3d_min_dims_1(self):
        """Test has_shape_with_min_dims returns True for 3D array with min_dims=1."""
        arr = np.array([[[1, 2]]])
        assert has_shape_with_min_dims(arr, min_dims=1) is True
        assert has_shape_with_min_dims(arr, min_dims=2) is True
        assert has_shape_with_min_dims(arr, min_dims=3) is True
        assert has_shape_with_min_dims(arr, min_dims=4) is False


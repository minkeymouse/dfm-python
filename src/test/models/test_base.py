"""Tests for models.base module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from dfm_python.models.base import BaseFactorModel
from dfm_python.models.dfm import DFM
from dfm_python.models.ddfm import DDFM
from dfm_python.dataset.ddfm_dataset import DDFMDataset
from dfm_python.utils.errors import ConfigurationError, DataError, DataValidationError, ModelNotInitializedError, NumericalError
from dfm_python.config import DFMConfig
from dfm_python.config.constants import DEFAULT_DTYPE, DEFAULT_ENCODER_LAYERS


class TestBaseFactorModel:
    """Test suite for BaseFactorModel."""
    
    def _create_test_ddfm(self):
        """Helper to create DDFM instance for testing."""
        test_data = pd.DataFrame(np.random.randn(10, 5), columns=[f'series_{i}' for i in range(5)])
        ddfm_dataset = DDFMDataset(data=test_data, time_idx='index', target_series=list(test_data.columns), target_scaler=None)
        return DDFM(dataset=ddfm_dataset, encoder_size=tuple(DEFAULT_ENCODER_LAYERS))
    
    def test_base_factor_model_is_abstract(self):
        """Test BaseFactorModel cannot be instantiated directly."""
        # BaseFactorModel is abstract, so direct instantiation should fail
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseFactorModel()
    
    def test_base_factor_model_interface(self):
        """Test BaseFactorModel defines required interface via concrete implementations."""
        # Test that concrete implementations have required methods
        dfm = DFM()
        # DDFM requires dataset - create minimal test dataset
        ddfm = self._create_test_ddfm()
        
        # All should have get_result method (abstract, must be implemented)
        assert hasattr(dfm, 'get_result')
        assert hasattr(ddfm, 'get_result')
        assert callable(dfm.get_result)
        assert callable(ddfm.get_result)
        
        # DFM has result property for convenience
        # Check result property exists for models that implement it (DFM)
        assert isinstance(getattr(type(dfm), 'result', None), property)
        # DDFM only has get_result() method, not result property
        assert not isinstance(getattr(type(ddfm), 'result', None), property)
        
        # All should have reset method (concrete in base)
        assert hasattr(dfm, 'reset')
        assert hasattr(ddfm, 'reset')
        assert callable(dfm.reset)
        assert callable(ddfm.reset)
    
    def test_config_property_raises_when_not_set(self):
        """Test config property raises ConfigurationError when config not set."""
        # DFM auto-initializes config, so we need to reset it first
        dfm = DFM()
        dfm.reset()  # Clear the auto-initialized config
        # Config not set, should raise ConfigurationError
        with pytest.raises(ConfigurationError, match="config access failed"):
            _ = dfm.config
    
    def test_config_property_returns_config_when_set(self):
        """Test config property returns config when set."""
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}}, frequency={'m': 'm'})
        dfm = DFM(config=config)
        assert dfm.config is not None
        assert dfm.config == config
    
    def test_reset_method(self):
        """Test reset method clears model state."""
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}}, frequency={'m': 'm'})
        dfm = DFM(config=config)
        # Verify config is set
        assert dfm.config is not None
        
        # Reset should clear config and return self
        result = dfm.reset()
        assert result is dfm
        # Config should be cleared (accessing should raise error)
        with pytest.raises(ConfigurationError):
            _ = dfm.config
    
    def test_predict_interface(self):
        """Test predict method interface exists in concrete implementations."""
        # Verify predict methods exist (signatures differ between models)
        dfm = DFM()
        # DDFM requires dataset - create minimal test dataset
        ddfm = self._create_test_ddfm()
        
        assert hasattr(dfm, 'predict')
        assert hasattr(ddfm, 'predict')
        assert callable(dfm.predict)
        assert callable(ddfm.predict)
    
    # Tests for removed legacy methods (_forecast_var_factors, _compute_default_horizon, _resolve_target_series)
    # These methods were removed during refactoring. Tests removed to reduce clutter.
    
    def test_validate_ndarray_with_valid_1d_array(self):
        """Test validate_ndarray_ndim accepts valid 1D numpy array."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        validate_ndarray_ndim(arr, "test_array", 1)
    
    def test_validate_ndarray_with_valid_2d_array(self):
        """Test validate_ndarray_ndim accepts valid 2D numpy array."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Should not raise
        validate_ndarray_ndim(arr, "test_array", 2)
    
    def test_validate_ndarray_with_non_numpy_array(self):
        """Test validate_ndarray_ndim raises DataValidationError for non-numpy array."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = [1.0, 2.0, 3.0]  # Python list, not numpy array
        
        with pytest.raises(DataValidationError, match="test_array must be 1D numpy array"):
            validate_ndarray_ndim(arr, "test_array", 1)
    
    def test_validate_ndarray_with_wrong_ndim(self):
        """Test validate_ndarray_ndim raises DataValidationError for wrong number of dimensions."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = np.array([1.0, 2.0, 3.0])  # 1D array
        
        with pytest.raises(DataValidationError, match="test_array must be 2D numpy array"):
            validate_ndarray_ndim(arr, "test_array", 2)
    
    def test_validate_ndarray_with_none(self):
        """Test validate_ndarray_ndim raises DataValidationError for None."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        
        with pytest.raises(DataValidationError, match="test_array must be 1D numpy array"):
            validate_ndarray_ndim(None, "test_array", 1)
    
    def test_validate_ndarray_with_scalar(self):
        """Test validate_ndarray_ndim raises DataValidationError for scalar."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        scalar = 5.0
        
        with pytest.raises(DataValidationError, match="test_array must be 1D numpy array"):
            validate_ndarray_ndim(scalar, "test_array", 1)
    
    def test_validate_ndarray_with_tuple(self):
        """Test validate_ndarray_ndim raises DataValidationError for tuple."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = (1.0, 2.0, 3.0)
        
        with pytest.raises(DataValidationError, match="test_array must be 1D numpy array"):
            validate_ndarray_ndim(arr, "test_array", 1)
    
    def test_validate_ndarray_with_3d_array_when_2d_expected(self):
        """Test validate_ndarray_ndim raises DataValidationError for 3D array when 2D expected."""
        from dfm_python.numeric.validator import validate_ndarray_ndim
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # 3D array
        
        with pytest.raises(DataValidationError, match="test_array must be 2D numpy array"):
            validate_ndarray_ndim(arr, "test_array", 2)


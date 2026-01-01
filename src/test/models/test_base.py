"""Tests for models.base module."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dfm_python.models.base import BaseFactorModel
from dfm_python.models.dfm import DFM
from dfm_python.models.ddfm import DDFM
from dfm_python.models.kdfm import KDFM
from dfm_python.utils.errors import ConfigurationError, DataError, DataValidationError, ModelNotInitializedError, NumericalError
from dfm_python.config import DFMConfig
from dfm_python.config.constants import DEFAULT_DTYPE


class TestBaseFactorModel:
    """Test suite for BaseFactorModel."""
    
    def test_base_factor_model_is_abstract(self):
        """Test BaseFactorModel cannot be instantiated directly."""
        # BaseFactorModel is abstract, so direct instantiation should fail
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseFactorModel()
    
    def test_base_factor_model_interface(self):
        """Test BaseFactorModel defines required interface via concrete implementations."""
        # Test that concrete implementations have required methods
        dfm = DFM()
        ddfm = DDFM(num_factors=1)
        kdfm = KDFM(ar_order=1, ma_order=0)
        
        # All should have get_result method (abstract, must be implemented)
        assert hasattr(dfm, 'get_result')
        assert hasattr(ddfm, 'get_result')
        assert hasattr(kdfm, 'get_result')
        assert callable(dfm.get_result)
        assert callable(ddfm.get_result)
        assert callable(kdfm.get_result)
        
        # All should have result property (preferred way to access results)
        # Check property exists without accessing it (accessing raises if not trained)
        assert 'result' in dir(dfm)
        assert 'result' in dir(ddfm)
        assert 'result' in dir(kdfm)
        # result is a property, not directly callable, but accessible via attribute access
        
        # All should have reset method (concrete in base)
        assert hasattr(dfm, 'reset')
        assert hasattr(ddfm, 'reset')
        assert hasattr(kdfm, 'reset')
        assert callable(dfm.reset)
        assert callable(ddfm.reset)
        assert callable(kdfm.reset)
    
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
        ddfm = DDFM(num_factors=1)
        kdfm = KDFM(ar_order=1, ma_order=0)
        
        assert hasattr(dfm, 'predict')
        assert hasattr(ddfm, 'predict')
        assert hasattr(kdfm, 'predict')
        assert callable(dfm.predict)
        assert callable(ddfm.predict)
        assert callable(kdfm.predict)
    
    def test_forecast_var_factors_uses_default_dtype(self):
        """Test _forecast_var_factors uses DEFAULT_DTYPE for output arrays."""
        dfm = DFM()
        # Test VAR(1) forecast
        Z_last = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        A = np.eye(3, dtype=np.float64) * 0.9
        horizon = 5
        
        Z_forecast = dfm._forecast_var_factors(Z_last, A, p=1, horizon=horizon)
        
        # Verify output uses DEFAULT_DTYPE (np.float32)
        assert Z_forecast.dtype == DEFAULT_DTYPE
        assert Z_forecast.shape == (horizon, 3)
    
    # VAR(2) tests removed - factors now always use AR(1) dynamics (simplified)
    
    # Tests for removed legacy methods (_transform_factors_to_observations, _standardize_data, _clean_mx_wx)
    # These methods were removed as part of refactoring to use sklearn scalers directly.
    # Models now use target_scaler.inverse_transform() for unstandardization.
    
    def test_compute_default_horizon_with_none_default(self):
        """Test _compute_default_horizon uses DEFAULT_FORECAST_HORIZON when default is None."""
        from dfm_python.config.constants import DEFAULT_FORECAST_HORIZON
        dfm = DFM()
        # Reset to clear config
        dfm.reset()
        # When default is None and config is not available, should use DEFAULT_FORECAST_HORIZON
        horizon = dfm._compute_default_horizon(default=None)
        assert horizon == DEFAULT_FORECAST_HORIZON
    
    def test_compute_default_horizon_with_custom_default(self):
        """Test _compute_default_horizon uses provided default value."""
        dfm = DFM()
        # Reset to clear config
        dfm.reset()
        # When default is provided, should use it
        custom_default = 10
        horizon = dfm._compute_default_horizon(default=custom_default)
        assert horizon == custom_default
    
    def test_resolve_target_series_from_datamodule(self):
        """Test _resolve_target_series successfully resolves from DataModule."""
        dfm = DFM()
        # Mock DataModule with target_series
        mock_datamodule = Mock()
        mock_datamodule.target_series = ['series1', 'series2']
        dfm._data_module = mock_datamodule
        
        # Provide series_ids for validation
        series_ids = ['series1', 'series2', 'series3', 'series4']
        
        target_series, target_indices = dfm._resolve_target_series(series_ids, None)
        
        # Should resolve successfully
        assert target_series == ['series1', 'series2']
        assert target_indices == [0, 1]
    
    def test_resolve_target_series_from_result(self):
        """Test _resolve_target_series falls back to result.series_ids."""
        dfm = DFM()
        dfm.reset()  # Clear config and datamodule
        
        # Mock result with series_ids
        mock_result = Mock()
        mock_result.series_ids = ['series1', 'series2', 'series3']
        
        # No DataModule, should use result
        target_series, target_indices = dfm._resolve_target_series(None, mock_result)
        
        # Should return None for target_series (no DataModule), None for indices (no target_series)
        assert target_series is None
        assert target_indices is None
    
    def test_resolve_target_series_from_config(self):
        """Test _resolve_target_series falls back to config.get_series_ids()."""
        dfm = DFM()
        # Set config with series_ids
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': ['series1', 'series2', 'series3']}},
            frequency={'m': 'm'}
        )
        dfm._config = config
        
        # No DataModule, no result, should use config
        target_series, target_indices = dfm._resolve_target_series(None, None)
        
        # Should return None for target_series (no DataModule), None for indices (no target_series)
        assert target_series is None
        assert target_indices is None
    
    def test_resolve_target_series_with_empty_target_list(self):
        """Test _resolve_target_series raises DataError when target_series is empty list."""
        dfm = DFM()
        # Mock DataModule with empty target_series (empty list)
        mock_datamodule = Mock()
        mock_datamodule.target_series = []
        dfm._data_module = mock_datamodule
        
        series_ids = ['series1', 'series2']
        
        # Empty list is not None, so it enters resolution block but has no items
        # This results in empty target_indices, which raises DataError
        with pytest.raises(DataError, match="none of the specified target series found"):
            dfm._resolve_target_series(series_ids, None)
    
    def test_resolve_target_series_with_missing_series(self):
        """Test _resolve_target_series raises DataError when all target series are missing."""
        dfm = DFM()
        # Mock DataModule with target_series not in series_ids
        mock_datamodule = Mock()
        mock_datamodule.target_series = ['missing_series']
        dfm._data_module = mock_datamodule
        
        series_ids = ['series1', 'series2', 'series3']
        
        # Should raise DataError when no target series found (all missing)
        with pytest.raises(DataError, match="none of the specified target series found"):
            dfm._resolve_target_series(series_ids, None)
    
    def test_resolve_target_series_all_missing_raises_error(self):
        """Test _resolve_target_series raises DataError when all target series are missing."""
        dfm = DFM()
        # Mock DataModule with target_series not in series_ids
        mock_datamodule = Mock()
        mock_datamodule.target_series = ['missing1', 'missing2']
        dfm._data_module = mock_datamodule
        
        series_ids = ['series1', 'series2', 'series3']
        
        # Should raise DataError when no target series found
        with pytest.raises(DataError, match="none of the specified target series found"):
            dfm._resolve_target_series(series_ids, None)
    
    def test_resolve_target_series_partial_match(self):
        """Test _resolve_target_series handles partial matches (some found, some missing)."""
        dfm = DFM()
        # Mock DataModule with mixed target_series
        mock_datamodule = Mock()
        mock_datamodule.target_series = ['series1', 'missing_series', 'series3']
        dfm._data_module = mock_datamodule
        
        series_ids = ['series1', 'series2', 'series3', 'series4']
        
        # Should resolve found series, log warning for missing, return partial indices
        target_series, target_indices = dfm._resolve_target_series(series_ids, None)
        
        # Should return found series indices only
        assert target_series == ['series1', 'missing_series', 'series3']
        assert target_indices == [0, 2]  # series1 and series3 found, missing_series skipped
    
    # Invalid order test removed - factors now always use AR(1) dynamics (simplified)
    # The method now only supports p=1 (AR(1)), and p parameter is kept for backward compatibility
    
    def test_forecast_var_factors_invalid_a_shape_var1(self):
        """Test _forecast_var_factors raises DataValidationError for invalid A shape in VAR(1)."""
        dfm = DFM()
        Z_last = np.array([1.0, 2.0], dtype=np.float64)
        A = np.eye(3, dtype=np.float64)  # Wrong shape: (3, 3) instead of (2, 2)
        
        with pytest.raises(DataValidationError, match="must have shape"):
            dfm._forecast_var_factors(Z_last, A, p=1, horizon=5)
    
    # VAR(2) tests removed - factors now always use AR(1) dynamics (simplified)
    
    def test_forecast_var_factors_invalid_z_prev_shape(self):
        """Test _forecast_var_factors raises DataValidationError for invalid Z_prev shape."""
        dfm = DFM()
        Z_last = np.array([1.0, 2.0], dtype=np.float64)
        A = np.hstack([np.eye(2) * 0.7, np.eye(2) * 0.2]).astype(np.float64)  # VAR(2) shape
        Z_prev = np.array([0.5, 1.5, 2.5])  # Wrong shape: (3,) instead of (2,)
        
        with pytest.raises(DataValidationError, match="must have shape"):
            dfm._forecast_var_factors(Z_last, A, p=2, horizon=5, Z_prev=Z_prev)
    
    # VAR(2) tests removed - factors now always use AR(1) dynamics (simplified)
    
    def test_compute_default_horizon_with_config_exception(self):
        """Test _compute_default_horizon falls back to default when config access raises exception."""
        from dfm_python.config.constants import DEFAULT_FORECAST_HORIZON
        from unittest.mock import patch
        dfm = DFM()
        # Set a config
        config = DFMConfig(blocks={'block1': {'num_factors': 2, 'series': []}}, frequency={'m': 'm'})
        dfm._config = config
        # Mock get_clock_frequency to raise ValueError to trigger exception path
        # Patch where it's imported (inside the method)
        with patch('dfm_python.utils.misc.get_clock_frequency', side_effect=ValueError("Test exception")):
            horizon = dfm._compute_default_horizon(default=None)
            # Should fall back to DEFAULT_FORECAST_HORIZON due to exception handling
            assert horizon == DEFAULT_FORECAST_HORIZON
    
    def test_forecast_var_factors_error_wrapping(self):
        """Test _forecast_var_factors wraps RuntimeError in NumericalError."""
        from unittest.mock import patch
        dfm = DFM()
        Z_last = np.array([1.0, 2.0], dtype=np.float64)
        A = np.eye(2, dtype=np.float64) * 0.9
        
        # Mock np.zeros to raise RuntimeError during computation
        with patch('numpy.zeros', side_effect=RuntimeError("Test runtime error")):
            with pytest.raises(NumericalError, match="Forecast computation failed"):
                dfm._forecast_var_factors(Z_last, A, p=1, horizon=5)
    
    def test_forecast_var_factors_horizon_zero(self):
        """Test _forecast_var_factors raises DataValidationError for horizon=0."""
        dfm = DFM()
        Z_last = np.array([1.0, 2.0], dtype=np.float64)
        A = np.eye(2, dtype=np.float64) * 0.9
        
        with pytest.raises(DataValidationError, match="horizon must be >= 1"):
            dfm._forecast_var_factors(Z_last, A, p=1, horizon=0)
    
    def test_forecast_var_factors_horizon_one(self):
        """Test _forecast_var_factors with horizon=1 returns single-step forecast."""
        dfm = DFM()
        Z_last = np.array([1.0, 2.0], dtype=np.float64)
        A = np.eye(2, dtype=np.float64) * 0.9
        
        Z_forecast = dfm._forecast_var_factors(Z_last, A, p=1, horizon=1)
        assert Z_forecast.shape == (1, 2)
        assert Z_forecast.dtype == DEFAULT_DTYPE
        # Should be A @ Z_last
        expected = A @ Z_last
        np.testing.assert_array_almost_equal(Z_forecast[0, :], expected)
    
    # VAR(2) tests removed - factors now always use AR(1) dynamics (simplified)
    
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


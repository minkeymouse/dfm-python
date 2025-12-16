"""Tests for update() and predict() implementations.

Tests verify that update() and predict() work correctly with dfm-python package only.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from dfm_python.models import DFM
from dfm_python.config import DFMConfig, SeriesConfig
from dfm_python import DFMDataModule, DFMTrainer
from dfm_python.utils.time import TimeIndex, parse_timestamp


class TestUpdateImplementation:
    """Test update() implementation."""
    
    @pytest.fixture
    def simple_config(self):
        """Create a simple config for testing."""
        return DFMConfig(
            series=[
                SeriesConfig(series_id='series1', frequency='m', transformation='lin', blocks=[1]),
                SeriesConfig(series_id='series2', frequency='m', transformation='lin', blocks=[1]),
            ],
            blocks={'block1': {'factors': 1, 'ar_lag': 1, 'clock': 'm'}}
        )
    
    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data."""
        np.random.seed(42)
        T, N = 50, 2
        X = np.random.randn(T, N).cumsum(axis=0)
        dates = pd.date_range('2020-01-01', periods=T, freq='ME')
        time_index = TimeIndex([parse_timestamp(d.strftime('%Y-%m-%d')) for d in dates])
        df = pd.DataFrame(X, columns=['series1', 'series2'])
        return df, time_index
    
    def test_nowcast_basic(self, simple_config, simple_data):
        """Test basic nowcast functionality."""
        model = DFM()
        model._config = simple_config
        
        # Create DataModule
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        # Train model - config에 max_iter, threshold 설정
        simple_config.max_iter = 2
        simple_config.threshold = 1e-2
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Test update method works
        X_std = np.random.randn(10, 2)
        updated_model = model.update(X_std)
        assert updated_model is model  # Should return self for chaining
        
        # After update, predict should work
        forecast = model.predict(horizon=1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 1
    
    def test_update_with_different_data_shapes(self, simple_config, simple_data):
        """Test update with different data shapes."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Test with different time periods
        X_std1 = np.random.randn(5, 2)
        model.update(X_std1)
        forecast1 = model.predict(horizon=1, return_series=True, return_factors=False)
        assert np.isfinite(forecast1).all()
        
        X_std2 = np.random.randn(10, 2)
        model.update(X_std2)
        forecast2 = model.predict(horizon=1, return_series=True, return_factors=False)
        assert np.isfinite(forecast2).all()
    
    def test_update_invalid_shape(self, simple_config, simple_data):
        """Test update raises error for invalid data shape."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # 1D array should raise error
        with pytest.raises(ValueError, match="X_std must be 2D array"):
            model.update(np.random.randn(10))
    
    def test_update_with_history(self, simple_config, simple_data):
        """Test update with history parameter filters to recent periods."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        simple_config.max_iter = 2
        simple_config.threshold = 1e-2
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Create data with more periods than history
        X_std = np.random.randn(20, 2)
        
        # Update with history=5 should use only last 5 periods
        updated_model = model.update(X_std, history=5)
        assert updated_model is model  # Should return self for chaining
        
        # After update, predict should work
        forecast = model.predict(horizon=1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 1
        assert np.isfinite(forecast).all()
    
    def test_update_with_history_larger_than_data(self, simple_config, simple_data):
        """Test update with history larger than data uses all data."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        simple_config.max_iter = 2
        simple_config.threshold = 1e-2
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Create data with fewer periods than history
        X_std = np.random.randn(5, 2)
        
        # Update with history=20 should use all 5 periods (data is smaller)
        updated_model = model.update(X_std, history=20)
        assert updated_model is model
        
        # After update, predict should work
        forecast = model.predict(horizon=1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 1
        assert np.isfinite(forecast).all()
    
    def test_update_with_history_none(self, simple_config, simple_data):
        """Test update with history=None uses all data (default behavior)."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        simple_config.max_iter = 2
        simple_config.threshold = 1e-2
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        X_std = np.random.randn(10, 2)
        
        # Update with history=None should use all data
        updated_model = model.update(X_std, history=None)
        assert updated_model is model
        
        # Should work the same as update without history parameter
        forecast1 = model.predict(horizon=1, return_series=True, return_factors=False)
        
        # Reset and update without history
        model._result.Z[-1, :] = model._result.Z[-2, :]  # Reset to previous state
        model.update(X_std)
        forecast2 = model.predict(horizon=1, return_series=True, return_factors=False)
        
        # Both should produce valid forecasts
        assert np.isfinite(forecast1).all()
        assert np.isfinite(forecast2).all()
    
    def test_predict_basic(self, simple_config, simple_data):
        """Test basic predict functionality."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Test predict returns array
        forecast = model.predict(horizon=3, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 3  # horizon
        assert forecast.shape[1] == 2  # number of series
        assert np.all(np.isfinite(forecast))
    
    def test_predict_with_factors(self, simple_config, simple_data):
        """Test predict returns both series and factors."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Test predict returns tuple
        X_forecast, Z_forecast = model.predict(horizon=3, return_series=True, return_factors=True)
        assert isinstance(X_forecast, np.ndarray)
        assert isinstance(Z_forecast, np.ndarray)
        assert X_forecast.shape[0] == 3
        assert Z_forecast.shape[0] == 3
        assert np.all(np.isfinite(X_forecast))
        assert np.all(np.isfinite(Z_forecast))
    
    
    def test_nowcast_all_missing_data(self, simple_config, simple_data):
        """Test nowcast with all missing data (should handle gracefully)."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        # Make all data NaN
        df_missing = df.copy()
        df_missing.iloc[:, :] = np.nan
        
        data_module = DFMDataModule(config=simple_config, data=df_missing, time_index=time_index)
        data_module.setup()
        
        simple_config.max_iter = 2
        simple_config.threshold = 1e-2
        trainer = DFMTrainer()
        
        # Training may fail with all NaN data, which is expected
        # All NaN data cannot be used for training, so we expect an error
        with pytest.raises((ValueError, RuntimeError, TypeError, IndexError)):
            trainer.fit(model, data_module)
    
    def test_predict_horizon_zero(self, simple_config, simple_data):
        """Test predict with horizon=0 raises error."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=0)
    
    def test_predict_horizon_negative(self, simple_config, simple_data):
        """Test predict with negative horizon raises error."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        with pytest.raises(ValueError, match="horizon must be positive"):
            model.predict(horizon=-1)
    
    def test_predict_history_zero(self, simple_config, simple_data):
        """Test predict with history=0 (should use full history)."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # history=0 should be treated as None (use full history)
        forecast = model.predict(horizon=3, history=0, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 3
    
    def test_predict_history_negative(self, simple_config, simple_data):
        """Test predict with negative history (should use full history)."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Negative history should be treated as None (use full history)
        forecast = model.predict(horizon=3, history=-1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 3
    
    def test_predict_history_larger_than_data(self, simple_config, simple_data):
        """Test predict with history larger than data length."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # history > data length should use all available data
        forecast = model.predict(horizon=3, history=1000, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape[0] == 3
    
    def test_predict_both_returns_false(self, simple_config, simple_data):
        """Test predict with both return_series and return_factors False."""
        model = DFM()
        model._config = simple_config
        
        df, time_index = simple_data
        data_module = DFMDataModule(config=simple_config, data=df, time_index=time_index)
        data_module.setup()
        
        trainer = DFMTrainer()
        trainer.fit(model, data_module)
        
        # Should return factors when both are False (default behavior)
        result = model.predict(horizon=3, return_series=False, return_factors=False)
        # Implementation may return factors or raise error - check it's valid
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3


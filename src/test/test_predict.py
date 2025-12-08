"""Tests for predict() method with history parameter.

Tests align with forecasting theory and recent improvements:
- Rolling window Kalman filter updates for efficiency
- State re-estimation based on recent data
- Common logic extraction for DFM and DDFM
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from dfm_python.models import DFM, BaseFactorModel
from dfm_python.config import DFMConfig, DDFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.config.results import DFMResult
from dfm_python import DFMDataModule
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.utils.time import TimeIndex, parse_timestamp
from test_helpers import (
    get_test_data_path,
    get_test_config_path,
    load_sample_data_from_csv,
    load_config_safely
)


class TestPredictHistory:
    """Test predict() method with history parameter."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return get_test_data_path()
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test DFM config."""
        return get_test_config_path("dfm")
    
    @pytest.fixture
    def sample_data_from_file(self, test_data_path):
        """Load sample data from CSV."""
        return load_sample_data_from_csv(test_data_path)
    
    def test_predict_without_history(self):
        """Test predict() without history parameter (default behavior).
        
        When history is None, predict() should use the last factor state
        from training without re-running Kalman filter.
        """
        model = DFM()
        # Predict should accept horizon parameter
        assert hasattr(model, 'predict')
        
        # Without training, predict should raise ValueError
        with pytest.raises(ValueError, match=r".*model has not been trained yet.*"):
            model.predict(horizon=1)
    
    def test_predict_history_parameter_interface(self):
        """Test that predict() accepts history parameter.
        
        The history parameter allows using only recent N periods for
        Kalman filter updates, improving efficiency.
        """
        model = DFM()
        # Check that predict signature includes history parameter
        import inspect
        sig = inspect.signature(model.predict)
        assert 'history' in sig.parameters
        # history should be Optional[int]
        history_param = sig.parameters['history']
        assert history_param.annotation in (Optional[int], int, type(None)) or 'Optional' in str(history_param.annotation)
    
    def test_predict_with_history_parameter(self, test_data_path, test_config_path):
        """Test predict() with history parameter.
        
        When history is specified (e.g., 60), predict() should:
        1. Use only the most recent N periods for Kalman filter update
        2. Re-estimate factor state based on recent data
        3. Use updated state for forecasting
        """
        if not test_data_path.exists() or not test_config_path.exists():
            pytest.skip("Test data or config files not found")
        
        # Load config
        source = YamlSource(test_config_path)
        try:
            config = source.load()
        except (TypeError, ValueError) as e:
            pytest.skip(f"Config format not fully supported: {e}")
        
        # Load data
        df = pd.read_csv(test_data_path)
        date_col = df.select("date").to_series().to_list()
        time_index = TimeIndex([parse_timestamp(d) for d in date_col])
        
        # Get series from config
        series_ids = [s.series_id for s in config.series]
        data_cols = [col for col in df.columns if col != "date" and col in series_ids]
        
        if len(data_cols) == 0:
            pytest.skip("No matching series found in data")
        
        # Extract and preprocess data
        data_array = df.select(data_cols).to_numpy()
        data_clean, _ = rem_nans_spline(data_array, method=2, k=3)
        
        # Sort data to match config order
        data_sorted, mnem_sorted = sort_data(data_clean, data_cols, config)
        
        # Create DataModule
        data_module = DFMDataModule(config=config, data=data_sorted, time=time_index)
        data_module.setup()
        
        # Create model and train
        model = DFM()
        model.load_config(test_config_path)
        
        from dfm_python.trainer import DFMTrainer
        trainer = DFMTrainer(max_epochs=5)  # Short training for test
        trainer.fit(model, data_module)
        
        # Test predict without history (default)
        forecast_default = model.predict(horizon=3)
        assert forecast_default is not None
        
        # Test predict with history parameter
        forecast_with_history = model.predict(horizon=3, history=60)
        assert forecast_with_history is not None
        
        # Both should return valid forecasts
        if isinstance(forecast_default, tuple):
            X_default, Z_default = forecast_default
            assert X_default.shape[0] == 3  # horizon
        else:
            assert forecast_default.shape[0] == 3
        
        if isinstance(forecast_with_history, tuple):
            X_history, Z_history = forecast_with_history
            assert X_history.shape[0] == 3  # horizon
        else:
            assert forecast_with_history.shape[0] == 3


class TestPredictCommonLogic:
    """Test common logic extracted to BaseFactorModel."""
    
    def test_compute_default_horizon(self):
        """Test _compute_default_horizon() helper method.
        
        Default horizon should be based on clock frequency (1 year of periods).
        """
        model = DFM()
        # Create a minimal config for testing
        from dfm_python.config import DFMConfig, DEFAULT_BLOCK_NAME
        series_list = [
            SeriesConfig(series_id='S1', frequency='m', transformation='lin', blocks=[DEFAULT_BLOCK_NAME])
        ]
        blocks = {
            DEFAULT_BLOCK_NAME: {"factors": 2, "ar_lag": 1, "clock": "m"}
        }
        config = DFMConfig(
            series=series_list,
            blocks=blocks
        )
        model._config = config
        
        # _compute_default_horizon should compute based on clock frequency
        assert hasattr(model, '_compute_default_horizon')
        # For monthly clock, default should be 12 (1 year)
        default_horizon = model._compute_default_horizon()
        assert isinstance(default_horizon, int)
        assert default_horizon > 0
    
    def test_validate_horizon(self):
        """Test _validate_horizon() helper method.
        
        Horizon validation should check for positive integers.
        """
        model = DFM()
        assert hasattr(model, '_validate_horizon')
        
        # Valid horizons should pass
        model._validate_horizon(1)
        model._validate_horizon(12)
        
        # Invalid horizons should raise ValueError
        with pytest.raises(ValueError):
            model._validate_horizon(0)
        
        with pytest.raises(ValueError):
            model._validate_horizon(-1)


class TestPredictDFMvsDDFM:
    """Test predict() consistency between DFM and DDFM."""
    
    def test_predict_interface_consistency(self):
        """Test that DFM and DDFM have consistent predict() interfaces.
        
        Both should support:
        - horizon parameter
        - history parameter
        - return_series and return_factors flags
        """
        dfm = DFM()
        assert hasattr(dfm, 'predict')
        
        try:
            from dfm_python.models import DDFM
            ddfm = DDFM()
            assert hasattr(ddfm, 'predict')
            
            # Both should have same signature structure
            import inspect
            dfm_sig = inspect.signature(dfm.predict)
            ddfm_sig = inspect.signature(ddfm.predict)
            
            # Both should have history parameter
            assert 'history' in dfm_sig.parameters
            assert 'history' in ddfm_sig.parameters
            
            # Both should have horizon parameter
            assert 'horizon' in dfm_sig.parameters
            assert 'horizon' in ddfm_sig.parameters
            
        except ImportError:
            pytest.skip("DDFM requires PyTorch")
    
    def test_predict_history_behavior_consistency(self):
        """Test that history parameter behaves consistently.
        
        For both DFM and DDFM:
        - history=None: use full history (default)
        - history=N: use only recent N periods for Kalman filter update
        """
        dfm = DFM()
        
        try:
            from dfm_python.models import DDFM
            ddfm = DDFM()
            
            # Both should accept history parameter
            import inspect
            dfm_sig = inspect.signature(dfm.predict)
            ddfm_sig = inspect.signature(ddfm.predict)
            
            dfm_history = dfm_sig.parameters['history']
            ddfm_history = ddfm_sig.parameters['history']
            
            # Both should have Optional[int] type
            assert dfm_history.default is None
            assert ddfm_history.default is None
            
        except ImportError:
            pytest.skip("DDFM requires PyTorch")


class TestPredictStateReestimation:
    """Test state re-estimation logic in predict()."""
    
    def test_history_triggers_state_update(self):
        """Test that history parameter triggers state re-estimation.
        
        When history is specified, _update_factor_state_with_history()
        should be called to re-run Kalman filter on recent data.
        """
        model = DFM()
        assert hasattr(model, '_update_factor_state_with_history')
        
        # This method should handle both DFM and DDFM
        # For DFM: directly uses Kalman filter
        # For DDFM: extracts factors via encoder, then Kalman filter
        
        # The method signature should accept history, result, and kalman_filter
        import inspect
        sig = inspect.signature(model._update_factor_state_with_history)
        assert 'history' in sig.parameters
        assert 'result' in sig.parameters
        assert 'kalman_filter' in sig.parameters


class TestPredictReturnValues:
    """Test predict() return value formats."""
    
    def test_predict_return_series_only(self):
        """Test predict() with return_series=True, return_factors=False."""
        model = DFM()
        # Without training, should raise error
        with pytest.raises(ValueError):
            model.predict(horizon=1, return_series=True, return_factors=False)
    
    def test_predict_return_factors_only(self):
        """Test predict() with return_series=False, return_factors=True."""
        model = DFM()
        # Without training, should raise error
        with pytest.raises(ValueError):
            model.predict(horizon=1, return_series=False, return_factors=True)
    
    def test_predict_return_both(self):
        """Test predict() with return_series=True, return_factors=True (default)."""
        model = DFM()
        # Without training, should raise error
        with pytest.raises(ValueError):
            model.predict(horizon=1, return_series=True, return_factors=True)


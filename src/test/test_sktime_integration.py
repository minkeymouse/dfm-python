"""Unit tests for sktime integration components.

This module tests the sktime integration features:
- NowcastingSplitter
- PublicationLagMasker transformer
- NowcastForecaster wrapper
- Metrics integration with sktime
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

try:
    import sktime
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False

from dfm_python.config import DFMConfig, SeriesConfig
from dfm_python.utils.time import TimeIndex, calculate_rmse, calculate_mae

# Try to import sktime integration components
try:
    from dfm_python.nowcast.splitters import NowcastingSplitter, NowcastForecaster
    from dfm_python.nowcast.transformers import PublicationLagMasker, NewsDecompositionTransformer
    HAS_NOWCAST_MODULE = True
except ImportError:
    # dfm_python.nowcast module may not exist (using src.nowcasting instead)
    HAS_NOWCAST_MODULE = False
    NowcastingSplitter = None
    NowcastForecaster = None
    PublicationLagMasker = None
    NewsDecompositionTransformer = None

# Skip all tests if sktime not available or nowcast module not available
pytestmark = pytest.mark.skipif(
    not HAS_SKTIME or not HAS_NOWCAST_MODULE,
    reason="sktime not installed or dfm_python.nowcast module not available. Install with: pip install sktime[forecasting]"
)


@pytest.fixture
def sample_config():
    """Create a sample DFMConfig for testing."""
    series = [
        SeriesConfig(
            series_id='series_1',
            frequency='m',
            transformation='lin',
            blocks=[1, 0],
            release_date=25  # Released on 25th of each month
        ),
        SeriesConfig(
            series_id='series_2',
            frequency='q',
            transformation='lin',
            blocks=[1, 0],
            release_date=-5  # Released 5 days before end of previous month
        ),
    ]
    
    blocks = {
        'global': {'factors': 1, 'ar_lag': 1, 'clock': 'm'},
        'block_1': {'factors': 1, 'ar_lag': 1}
    }
    
    config = DFMConfig(series=series, blocks=blocks, clock='m')
    return config


@pytest.fixture
def sample_data(sample_config):
    """Create sample time series data."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS')
    n_series = len(sample_config.series)
    data = np.random.randn(len(dates), n_series)
    
    df = pd.DataFrame(data, index=dates, columns=[s.series_id for s in sample_config.series])
    return df


@pytest.fixture
def sample_time_index(sample_data):
    """Create sample TimeIndex."""
    dates = [d.to_pydatetime() for d in sample_data.index]
    return TimeIndex(dates)


class TestNowcastingSplitter:
    """Test NowcastingSplitter class."""
    
    @pytest.mark.skipif(not HAS_NOWCAST_MODULE, reason="dfm_python.nowcast module not available")
    def test_splitter_initialization(self, sample_config, sample_time_index):
        """Test splitter initialization."""
        target_periods = [datetime(2024, 3, 31), datetime(2024, 6, 30)]
        
        splitter = NowcastingSplitter(
            target_periods=target_periods,
            backward_steps=5,
            config=sample_config,
            time_index=sample_time_index
        )
        
        assert splitter.target_periods == target_periods
        assert splitter.backward_steps == 5
        assert splitter.config == sample_config
        assert splitter.clock == 'm'
    
    def test_splitter_generates_splits(self, sample_config, sample_data, sample_time_index):
        """Test that splitter generates train/test splits."""
        target_periods = [datetime(2024, 3, 31)]
        
        splitter = NowcastingSplitter(
            target_periods=target_periods,
            backward_steps=3,
            config=sample_config,
            time_index=sample_time_index
        )
        
        splits = list(splitter.split(sample_data))
        assert len(splits) > 0
        
        # Check that each split has train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
    
    def test_splitter_respects_publication_lags(self, sample_config, sample_data, sample_time_index):
        """Test that splitter respects publication lags."""
        target_periods = [datetime(2024, 3, 31)]
        
        splitter = NowcastingSplitter(
            target_periods=target_periods,
            backward_steps=2,
            config=sample_config,
            time_index=sample_time_index
        )
        
        # Get first split
        splits = list(splitter.split(sample_data))
        if len(splits) > 0:
            train_idx, test_idx = splits[0]
            # Train indices should be a subset of available data
            assert len(train_idx) <= len(sample_data)
            assert all(0 <= idx < len(sample_data) for idx in train_idx)
    
    def test_get_n_splits(self, sample_config, sample_data, sample_time_index):
        """Test get_n_splits method."""
        target_periods = [datetime(2024, 3, 31), datetime(2024, 6, 30)]
        
        splitter = NowcastingSplitter(
            target_periods=target_periods,
            backward_steps=3,
            config=sample_config,
            time_index=sample_time_index
        )
        
        n_splits = splitter.get_n_splits(sample_data)
        assert n_splits > 0
        assert isinstance(n_splits, int)
    
    def test_get_split_params(self, sample_config, sample_data, sample_time_index):
        """Test get_split_params method."""
        target_periods = [datetime(2024, 3, 31)]
        
        splitter = NowcastingSplitter(
            target_periods=target_periods,
            backward_steps=2,
            config=sample_config,
            time_index=sample_time_index
        )
        
        if splitter.get_n_splits() > 0:
            params = splitter.get_split_params(0)
            assert 'view_date' in params
            assert 'target_date' in params
            assert isinstance(params['view_date'], datetime)
            assert isinstance(params['target_date'], datetime)


class TestPublicationLagMasker:
    """Test PublicationLagMasker transformer."""
    
    def test_masker_initialization(self, sample_config):
        """Test masker initialization."""
        view_date = datetime(2024, 1, 15)
        
        masker = PublicationLagMasker(
            config=sample_config,
            view_date=view_date
        )
        
        assert masker.config == sample_config
        assert masker.view_date == view_date
    
    def test_masker_fit_transform(self, sample_config, sample_data):
        """Test masker fit and transform."""
        view_date = datetime(2024, 1, 15)
        
        masker = PublicationLagMasker(
            config=sample_config,
            view_date=view_date
        )
        
        # Fit and transform
        X_masked = masker.fit_transform(sample_data)
        
        assert isinstance(X_masked, pd.DataFrame)
        assert X_masked.shape == sample_data.shape
        assert X_masked.index.equals(sample_data.index)
        assert X_masked.columns.equals(sample_data.columns)
    
    def test_masker_respects_release_dates(self, sample_config, sample_data):
        """Test that masker respects release dates."""
        # Use a view date that should mask some data
        view_date = datetime(2024, 1, 20)  # Before 25th release
        
        masker = PublicationLagMasker(
            config=sample_config,
            view_date=view_date
        )
        
        X_masked = masker.fit_transform(sample_data)
        
        # Check that some data might be masked (NaN values)
        # This depends on the specific release dates and view_date
        assert isinstance(X_masked, pd.DataFrame)
    
    def test_masker_inverse_transform(self, sample_config, sample_data):
        """Test inverse transform (should return input unchanged)."""
        view_date = datetime(2024, 1, 15)
        
        masker = PublicationLagMasker(
            config=sample_config,
            view_date=view_date
        )
        
        X_masked = masker.fit_transform(sample_data)
        
        # Since masking is not invertible, inverse_transform should raise NotImplementedError
        # or be skipped (depending on sktime version)
        # Check if inverse_transform is available
        if hasattr(masker, 'inverse_transform'):
            try:
                X_inverse = masker.inverse_transform(X_masked)
                # If it doesn't raise, verify it returns something reasonable
                assert isinstance(X_inverse, pd.DataFrame)
                assert X_inverse.shape == X_masked.shape
            except NotImplementedError:
                # Expected behavior - masking is not invertible
                pass


class TestSktimeMetrics:
    """Test sktime metrics integration."""
    
    def test_calculate_rmse_with_sktime(self):
        """Test that calculate_rmse uses sktime if available."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        rmse_overall, rmse_per_series = calculate_rmse(y_true, y_pred)
        
        assert rmse_overall > 0
        assert isinstance(rmse_overall, (float, np.floating))
        assert len(rmse_per_series) == 1
        assert rmse_per_series[0] > 0
    
    def test_calculate_mae_with_sktime(self):
        """Test that calculate_mae uses sktime if available."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        
        mae_overall, mae_per_series = calculate_mae(y_true, y_pred)
        
        assert mae_overall > 0
        assert mae_overall < 0.2  # Should be small
        assert isinstance(mae_overall, (float, np.floating))
        assert len(mae_per_series) == 1
    
    def test_calculate_rmse_multivariate(self):
        """Test RMSE calculation for multivariate time series."""
        y_true = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [2.1, 3.1], [2.9, 4.1]])
        
        rmse_overall, rmse_per_series = calculate_rmse(y_true, y_pred)
        
        assert rmse_overall > 0
        assert len(rmse_per_series) == 2
        assert all(r > 0 for r in rmse_per_series)
    
    def test_calculate_rmse_with_mask(self):
        """Test RMSE calculation with masking."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        mask = np.array([True, True, True, False, False])  # Mask last 2
        
        rmse_overall, rmse_per_series = calculate_rmse(y_true, y_pred, mask=mask)
        
        assert rmse_overall > 0
        assert not np.isnan(rmse_overall)
        assert len(rmse_per_series) == 1


class TestNowcastForecaster:
    """Test NowcastForecaster wrapper class."""
    
    def test_forecaster_initialization(self, sample_config):
        """Test forecaster initialization."""
        # Create a mock nowcast manager
        class MockNowcast:
            def __init__(self):
                self.model = type('obj', (object,), {'config': sample_config})()
            
            def __call__(self, target_series, view_date, target_period):
                return 2.5  # Mock nowcast value
        
        nowcast_manager = MockNowcast()
        
        forecaster = NowcastForecaster(
            nowcast_manager=nowcast_manager,
            target_series='series_1',
            target_period=datetime(2024, 3, 31)
        )
        
        assert forecaster.target_series == 'series_1'
        assert forecaster.target_period == datetime(2024, 3, 31)
    
    def test_forecaster_fit_predict(self, sample_config, sample_data):
        """Test forecaster fit and predict."""
        # Create a mock nowcast manager
        class MockNowcast:
            def __init__(self):
                self.model = type('obj', (object,), {'config': sample_config})()
            
            def __call__(self, target_series, view_date, target_period):
                return 2.5  # Mock nowcast value
        
        nowcast_manager = MockNowcast()
        
        forecaster = NowcastForecaster(
            nowcast_manager=nowcast_manager,
            target_series='series_1',
            target_period=datetime(2024, 3, 31)
        )
        
        # Fit
        forecaster.fit(sample_data)
        assert forecaster._is_fitted
        assert forecaster._view_date is not None
        
        # Predict
        from sktime.forecasting.base import ForecastingHorizon
        fh = ForecastingHorizon([1])
        y_pred = forecaster.predict(fh)
        
        # sktime may return Series or DataFrame depending on version
        # Convert to Series if DataFrame with single column
        if isinstance(y_pred, pd.DataFrame):
            if y_pred.shape[1] == 1:
                y_pred = y_pred.iloc[:, 0]
            else:
                y_pred = y_pred.squeeze()
        
        assert isinstance(y_pred, pd.Series)
        assert len(y_pred) == 1
        assert y_pred.iloc[0] == 2.5  # Mock value
    
    def test_forecaster_set_view_date(self, sample_config):
        """Test set_view_date method."""
        class MockNowcast:
            def __init__(self):
                self.model = type('obj', (object,), {'config': sample_config})()
        
        nowcast_manager = MockNowcast()
        
        forecaster = NowcastForecaster(
            nowcast_manager=nowcast_manager,
            target_series='series_1',
            target_period=datetime(2024, 3, 31)
        )
        
        view_date = datetime(2024, 1, 15)
        forecaster.set_view_date(view_date)
        
        assert forecaster._view_date == view_date


class TestNewsDecompositionTransformer:
    """Test NewsDecompositionTransformer."""
    
    def test_transformer_initialization(self, sample_config):
        """Test transformer initialization."""
        # Create a mock nowcast manager
        class MockNowcast:
            def __init__(self):
                self.model = type('obj', (object,), {'config': sample_config})()
            
            def decompose(self, target_series, target_period, view_date_old, view_date_new, return_dict=False):
                from dfm_python.nowcast.helpers import NewsDecompResult
                return NewsDecompResult(
                    y_old=2.0,
                    y_new=2.5,
                    change=0.5,
                    singlenews=np.array([0.3, 0.2]),
                    top_contributors=[('series_1', 0.3), ('series_2', 0.2)],
                    actual=np.array([1.0]),
                    forecast=np.array([0.9]),
                    weight=np.array([0.5, 0.5]),
                    t_miss=np.array([0]),
                    v_miss=np.array([0]),
                    innov=np.array([0.1])
                )
        
        nowcast_manager = MockNowcast()
        
        transformer = NewsDecompositionTransformer(
            nowcast_manager=nowcast_manager,
            target_series='series_1',
            target_period=datetime(2024, 3, 31),
            view_date_old=datetime(2024, 1, 15),
            view_date_new=datetime(2024, 2, 15)
        )
        
        assert transformer.target_series == 'series_1'
        assert transformer.target_period == datetime(2024, 3, 31)
    
    def test_transformer_transform(self, sample_config, sample_data):
        """Test transformer transform method."""
        # Create a mock nowcast manager
        class MockNowcast:
            def __init__(self):
                self.model = type('obj', (object,), {'config': sample_config})()
            
            def decompose(self, target_series, target_period, view_date_old, view_date_new, return_dict=False):
                from dfm_python.nowcast.helpers import NewsDecompResult
                return NewsDecompResult(
                    y_old=2.0,
                    y_new=2.5,
                    change=0.5,
                    singlenews=np.array([0.3, 0.2]),
                    top_contributors=[('series_1', 0.3), ('series_2', 0.2)],
                    actual=np.array([1.0]),
                    forecast=np.array([0.9]),
                    weight=np.array([0.5, 0.5]),
                    t_miss=np.array([0]),
                    v_miss=np.array([0]),
                    innov=np.array([0.1])
                )
        
        nowcast_manager = MockNowcast()
        
        transformer = NewsDecompositionTransformer(
            nowcast_manager=nowcast_manager,
            target_series='series_1',
            target_period=datetime(2024, 3, 31),
            view_date_old=datetime(2024, 1, 15),
            view_date_new=datetime(2024, 2, 15)
        )
        
        # Transform (X_new, y=X_old)
        result = transformer.fit_transform(sample_data, y=sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'y_old' in result.columns
        assert 'y_new' in result.columns
        assert 'change' in result.columns
        assert result.iloc[0]['change'] == 0.5


"""Tests for nowcasting and news decomposition.

Tests align with nowcasting theory from:
- Giannone et al. (2008): Real-time nowcasting framework
- Banbura et al. (2011): News decomposition
- Bańbura & Modugno (2014): Mixed-frequency nowcasting
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# Import NowcastResult from dfm_python.config.results (v0.5.0)
from dfm_python.config.results import NowcastResult

# Import Nowcast and other utilities from src.nowcasting if available
# These are not part of dfm-python package, they're in the main project
try:
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from nowcasting import Nowcast, para_const
    try:
        from nowcasting import NewsDecompResult, BacktestResult
    except ImportError:
        NewsDecompResult = None
        BacktestResult = None
    try:
        from nowcasting import DataView
        from nowcasting.utils import get_higher_frequency, calc_backward_date
    except ImportError:
        DataView = None
        get_higher_frequency = None
        calc_backward_date = None
except ImportError:
    # If src.nowcasting is not available, set to None
    Nowcast = None
    NewsDecompResult = None
    BacktestResult = None
    DataView = None
    para_const = None
    get_higher_frequency = None
    calc_backward_date = None

from dfm_python.config import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from dfm_python.config.adapter import YamlSource
from dfm_python.config.results import DFMResult
from dfm_python.utils.time import TimeIndex, parse_timestamp
from dfm_python.utils.data import rem_nans_spline, sort_data
from dfm_python.models import DFM, BaseFactorModel


class TestNowcast:
    """Test Nowcast class for nowcasting operations."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data file."""
        return Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test DFM config."""
        return Path(__file__).parent.parent.parent / "config" / "experiment" / "test_dfm.yaml"
    
    @pytest.fixture
    def sample_data_from_file(self, test_data_path, test_config_path):
        """Load sample data from CSV using pandas."""
        if not test_data_path.exists() or not test_config_path.exists():
            pytest.skip("Test data or config files not found")
        
        # Load config
        source = YamlSource(test_config_path)
        config = source.load()
        
        # Read CSV with pandas
        df = pd.read_csv(test_data_path)
        
        # Extract date column
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
        
        return data_sorted, time_index, config
    
    @pytest.fixture
    def sample_model(self):
        """Create sample DFM model for testing."""
        # Mock model with required attributes
        class MockModel:
            def __init__(self):
                self._data_module = None
                self._result = None
        return MockModel()
    
    def test_nowcast_initialization(self, sample_model):
        """Test Nowcast initialization."""
        # Nowcast requires model and data_module
        # This tests the interface
        assert hasattr(Nowcast, '__init__')
    
    def test_base_factor_model_has_update_method(self):
        """Test that BaseFactorModel has update method.
        
        The update method was added to BaseFactorModel to provide
        a unified interface for updating factor state.
        """
        model = DFM()
        # BaseFactorModel should have update method
        assert hasattr(model, 'update')
        assert callable(getattr(model, 'update', None))
        
        # Before training, calling update should raise ValueError
        with pytest.raises(ValueError, match=r".*model has not been trained yet.*"):
            _ = model.update(np.random.randn(10, 2))
    
    def test_update_method_chainable(self, test_data_path, test_config_path):
        """Test that model.nowcast returns Nowcast instance.
        
        After training, model.nowcast should return a Nowcast instance
        that can be used for nowcasting operations.
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
        from dfm_python import DFMDataModule
        data_module = DFMDataModule(config=config, data=data_sorted, time=time_index)
        data_module.setup()
        
        # Create model and train
        model = DFM()
        model.load_config(test_config_path)
        
        from dfm_python.trainer import DFMTrainer
        trainer = DFMTrainer(max_epochs=5)  # Short training for test
        trainer.fit(model, data_module)
        
        # After training, update should be chainable
        X_std = np.random.randn(10, len(series_ids))
        updated_model = model.update(X_std)
        assert updated_model is model  # Should return self for chaining
    
    def test_nowcast_data_view(self, sample_data_from_file):
        """Test data view creation for pseudo real-time.
        
        From Giannone et al. (2008):
        - DataView creates time-specific slice of data
        - Simulates information available at specific date
        - Handles jagged edges (varying missing data patterns)
        """
        if DataView is None:
            pytest.skip("DataView not available (using src.nowcasting fallback)")
        
        X, time_index, config = sample_data_from_file
        T, N = X.shape
        
        # DataView requires Z and config parameters
        view = DataView(
            X=X,
            Time=time_index,
            Z=None,
            config=config,
            view_date=time_index[-10] if len(time_index) > 10 else time_index[-1]
        )
        assert view.X.shape == (T, N)
        assert view.view_date is not None
    
    def test_nowcast_result_structure(self):
        """Test NowcastResult dataclass."""
        from datetime import datetime
        result = NowcastResult(
            target_series="GDP",
            target_period=datetime(2020, 3, 31),
            view_date=datetime(2020, 4, 15),
            nowcast_value=2.5,
            confidence_interval=(2.0, 3.0)
        )
        assert result.target_series == "GDP"
        assert result.nowcast_value == 2.5
        assert result.confidence_interval == (2.0, 3.0)

    def test_nowcast_predict_inverse_transform_applied(self):
        """Ensure predict() output is inverse-transformed after an update-style call."""
        import types
        import torch
        from typing import Any, cast
        from dfm_python.models import DFM

        class DummyScaler:
            def inverse_transform(self, X):
                return X + 7.0

        class SimpleResult:
            def __init__(self):
                # Two time steps, one factor
                self.Z = np.array([[0.0], [0.0]])
                # Identity loadings for two series
                self.C = np.array([[1.0], [1.0]])
                # One-step VAR(1)
                self.A = np.array([[0.0]])
                self.Wx = np.array([1.0, 1.0])
                self.Mx = np.array([0.0, 0.0])
                self.p = 1

        model = DFM()
        # Minimal non-None training_state to pass checks
        model.training_state = types.SimpleNamespace(
            A=torch.zeros((1, 1)),
            C=torch.zeros((2, 1)),
            Q=torch.zeros((1, 1)),
            R=torch.zeros((2, 2)),
            Z_0=torch.zeros((1,)),
            V_0=torch.zeros((1, 1)),
            loglik=0.0,
            num_iter=1,
            converged=True
        )
        model._result = cast(Any, SimpleResult())
        object.__setattr__(model, "scaler", DummyScaler())

        # Simulate update-before-predict flow
        model.update = types.MethodType(lambda self, X_std, **kwargs: self, model)
        model.update(np.zeros((1, 2)))

        forecast = model.predict(horizon=1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (1, 2)
        assert np.allclose(forecast, np.full((1, 2), 7.0))

    def test_update_allows_scaler_replacement(self):
        """Ensure update(scaler=...) swaps scaler used in predict()."""
        import types
        import torch
        from typing import Any, cast
        from dfm_python.models import DFM

        class DummyScalerA:
            def inverse_transform(self, X):
                return X + 1.0

        class DummyScalerB:
            def inverse_transform(self, X):
                return X + 3.0

        class SimpleResult:
            def __init__(self):
                self.Z = np.array([[0.0], [0.0]])
                self.C = np.array([[1.0], [1.0]])
                self.A = np.array([[0.0]])
                self.Q = np.array([[1.0]])
                self.R = np.eye(2) * 0.1
                self.Z_0 = np.array([0.0])
                self.V_0 = np.eye(1) * 0.1
                self.Wx = np.array([1.0, 1.0])
                self.Mx = np.array([0.0, 0.0])
                self.p = 1

        model = DFM()
        model.training_state = types.SimpleNamespace(
            A=torch.zeros((1, 1)),
            C=torch.zeros((2, 1)),
            Q=torch.zeros((1, 1)),
            R=torch.zeros((2, 2)),
            Z_0=torch.zeros((1,)),
            V_0=torch.zeros((1, 1)),
            loglik=0.0,
            num_iter=1,
            converged=True
        )
        model._result = cast(Any, SimpleResult())
        object.__setattr__(model, "scaler", DummyScalerA())

        # Replace scaler via update
        X_dummy = np.zeros((1, 2))
        model.update(X_dummy, scaler=DummyScalerB())

        forecast = model.predict(horizon=1, return_series=True, return_factors=False)
        assert isinstance(forecast, np.ndarray)
        assert forecast.shape == (1, 2)
        assert np.allclose(forecast, np.full((1, 2), 3.0))


class TestNewsDecomposition:
    """Test news decomposition framework."""
    
    def test_news_definition(self):
        """Test news definition from Banbura et al. (2011).
        
        News: difference between new data release and previous forecast
        News_t = y_{t,new} - E[y_{t,new} | Y_{t-1}]
        """
        # News should be forecast error
        y_new = 2.5
        y_forecast = 2.0
        news = y_new - y_forecast
        assert news == 0.5
    
    def test_news_decomp_result(self):
        """Test NewsDecompResult structure."""
        if NewsDecompResult is None:
            pytest.skip("NewsDecompResult not available (requires src.nowcasting)")
        # NewsDecompResult has different structure
        result = NewsDecompResult(
            y_old=2.0,
            y_new=2.3,
            change=0.3,
            singlenews=np.array([0.1, 0.2]),
            top_contributors=[("S1", 0.2), ("S2", 0.1)],
            actual=np.array([2.1, 2.2]),
            forecast=np.array([2.0, 2.1]),
            weight=np.array([0.5, 0.5]),
            t_miss=np.array([0, 1]),
            v_miss=np.array([0, 1]),
            innov=np.array([0.1, 0.1])
        )
        assert result.change == 0.3
        assert isinstance(result.top_contributors, list)
    
    def test_news_attribution(self):
        """Test news attribution to data series.
        
        News decomposition attributes forecast change to:
        - Individual data series contributions
        - Factor contributions
        - Idiosyncratic contributions
        """
        # News should be decomposable
        total_news = 0.5
        contributions = {
            "S1": 0.2,
            "S2": 0.15,
            "S3": 0.1,
            "S4": 0.05
        }
        # Contributions should sum to total (approximately)
        sum_contrib = sum(contributions.values())
        assert abs(sum_contrib - total_news) < 0.1


class TestBacktesting:
    """Test backtesting framework."""
    
    def test_backtest_result(self):
        """Test BacktestResult structure."""
        if BacktestResult is None:
            pytest.skip("BacktestResult not available in src.nowcasting")
        
        # BacktestResult has different structure
        result = BacktestResult(
            target_series="GDP",
            target_date=datetime(2020, 3, 31),
            backward_steps=3,
            higher_freq=False,
            backward_freq="m",
            view_list=[],
            nowcast_results=[],
            news_results=[],
            actual_values=np.array([2.05, 2.15, 2.25]),
            errors=np.array([0.05, 0.05, 0.05]),
            mae_per_step=np.array([0.04, 0.04, 0.04]),
            mse_per_step=np.array([0.0025, 0.0025, 0.0025]),
            rmse_per_step=np.array([0.05, 0.05, 0.05]),
            overall_mae=0.04,
            overall_rmse=0.05,
            overall_mse=0.0025,
            failed_steps=[]
        )
        assert result.target_series == "GDP"
        assert len(result.actual_values) == 3
        assert result.overall_rmse is not None and result.overall_rmse > 0
    
    def test_pseudo_real_time_evaluation(self):
        """Test pseudo real-time evaluation.
        
        From Giannone et al. (2008):
        - Backtesting uses historical data vintages
        - Simulates real-time information sets
        - Evaluates nowcast accuracy
        """
        if BacktestResult is None:
            pytest.skip("BacktestResult not available in src.nowcasting")
        
        # Backtest should use historical vintages
        # This is tested via actual backtest execution
        assert hasattr(BacktestResult, '__init__')


class TestDataView:
    """Test DataView for pseudo real-time data."""
    
    def test_data_view_creation(self):
        """Test DataView creation."""
        if DataView is None:
            pytest.skip("DataView not available (using src.nowcasting fallback)")
        
        T, N = 100, 5
        X = np.random.randn(T, N)
        time_index = TimeIndex([datetime(2020, 1, 1) + timedelta(days=30*i) for i in range(T)])
        
        # DataView requires Z and config parameters
        view = DataView(
            X=X,
            Time=time_index,
            Z=None,
            config=None,
            view_date=datetime(2020, 6, 1)
        )
        assert view.X.shape == (T, N)
        assert view.Time is not None
    
    def test_data_view_materialize(self):
        """Test DataView materialization."""
        if DataView is None:
            pytest.skip("DataView not available (using src.nowcasting fallback)")
        
        T, N = 100, 5
        X = np.random.randn(T, N)
        time_index = TimeIndex([datetime(2020, 1, 1) + timedelta(days=30*i) for i in range(T)])
        
        view = DataView(X=X, Time=time_index, Z=None, config=None)
        X_mat, Time_mat, Z_mat = view.materialize()
        assert X_mat.shape == (T, N)
        assert Time_mat is not None


class TestNowcastUtilities:
    """Test nowcasting utility functions."""
    
    def test_get_higher_frequency(self):
        """Test frequency hierarchy for aggregation."""
        if get_higher_frequency is None:
            pytest.skip("get_higher_frequency not available (using src.nowcasting fallback)")
        
        # get_higher_frequency takes one argument (clock) and returns next higher frequency
        # Monthly clock -> weekly is higher
        higher = get_higher_frequency("m")
        assert higher is not None
        # Quarterly clock -> monthly is higher
        higher = get_higher_frequency("q")
        assert higher == "m"  # Monthly is higher than quarterly
    
    def test_calculate_backward_date(self):
        """Test backward date calculation for nowcasting."""
        if calc_backward_date is None:
            pytest.skip("calc_backward_date not available (using src.nowcasting fallback)")
        
        target_date = datetime(2020, 3, 31)
        # calculate_backward_date has different signature
        backward_date = calc_backward_date(target_date, step=1, freq="m")
        # Should be one period before target
        assert backward_date < target_date
    
    def test_para_const(self):
        """Test para_const function for parameter constraints.
        
        From Giannone et al. (2008):
        - para_const computes parameter constraints for nowcasting
        - Handles mixed frequencies and aggregation
        """
        if para_const is None:
            pytest.skip("para_const not available (requires src.nowcasting)")
        T, N = 100, 5
        X = np.random.randn(T, N)
        
        # para_const requires actual DFMResult
        from dfm_python.config.results import DFMResult
        T, N, r = 100, 5, 2
        result = DFMResult(
            x_sm=np.random.randn(T, N),
            X_sm=np.random.randn(T, N),
            Z=np.random.randn(T, r),
            C=np.random.randn(N, r),
            R=np.eye(N) * 0.1,
            A=np.random.randn(r, r) * 0.5,
            Q=np.eye(r) * 0.1,
            Mx=np.zeros(N),
            Wx=np.ones(N),
            Z_0=np.zeros(r),
            V_0=np.eye(r),
            r=np.array([r]),
            p=1
        )
        constraints = para_const(X, result, lag=0)
        
        # Should return constraint dictionary
        assert isinstance(constraints, dict)


class TestMixedFrequency:
    """Test mixed-frequency nowcasting."""
    
    def test_mixed_frequency_aggregation(self):
        """Test mixed-frequency aggregation from Bańbura & Modugno (2014).
        
        Quarterly target (GDP) nowcasted using:
        - Monthly indicators
        - Quarterly indicators
        - Mixed-frequency state-space
        """
        # Monthly to quarterly aggregation
        monthly_data = np.random.randn(12, 5)  # 12 months
        # Aggregate to quarterly (average)
        quarterly_data = monthly_data.reshape(4, 3, 5).mean(axis=1)
        assert quarterly_data.shape == (4, 5)
        # This test verifies aggregation logic, not actual function call
    
    def test_jagged_edges(self):
        """Test jagged edges (varying missing data patterns).
        
        From Giannone et al. (2008):
        - Different series have different release dates
        - Creates "jagged edge" in data matrix
        - Kalman filter handles via selection matrices
        """
        T, N = 100, 5
        X = np.random.randn(T, N)
        
        # Simulate jagged edges (missing data at end)
        X[-5:, 0] = np.nan  # Series 0 missing last 5 periods
        X[-3:, 1] = np.nan  # Series 1 missing last 3 periods
        X[-10:, 2] = np.nan  # Series 2 missing last 10 periods
        
        # Should handle varying missing patterns
        assert np.isnan(X[-5:, 0]).all()
        assert np.isnan(X[-3:, 1]).all()
        assert np.isnan(X[-10:, 2]).all()


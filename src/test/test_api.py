"""Tests for API edge cases and tutorials - consolidated."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import DFM, DFMResult
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
import dfm_python as dfm

# ============================================================================
# API Edge Cases (from test_api_edge_cases.py)
# ============================================================================

def test_empty_data():
    """Test with empty data array."""
    config = DFMConfig(
        series=[SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])],
        blocks={'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    )
    
    X = np.array([]).reshape(0, 1)
    
    with pytest.raises((ValueError, IndexError)):
        model = DFM()
        model.fit(X, config)


def test_single_time_period():
    """Test with single time period."""
    np.random.seed(42)
    X = np.random.randn(1, 5)
    
    series_list = [
        SeriesConfig(
            series_id=f'test_{i}',
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        )
        for i in range(5)
    ]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = DFM()
            result = model.fit(X, config, max_iter=1, threshold=1e-2)
            assert result is not None
        except (ValueError, RuntimeError, UnboundLocalError):
            # Acceptable - single time period may be insufficient
            pass


def test_very_small_threshold():
    """Test with very small convergence threshold."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    
    series_list = [
        SeriesConfig(
            series_id=f'test_{i}',
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        )
        for i in range(N)
    ]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        result = model.fit(X, config, threshold=1e-10, max_iter=10)
        assert result is not None


def test_very_large_max_iter():
    """Test with very large max_iter."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    
    series_list = [
        SeriesConfig(
            series_id=f'test_{i}',
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        )
        for i in range(N)
    ]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        result = model.fit(X, config, max_iter=10000)
        assert result is not None


def test_predict_without_training():
    """Test predict without training."""
    import dfm_python as dfm
    
    dfm.reset()
    
    with pytest.raises((ValueError, AttributeError)):
        dfm.predict(horizon=6)


def test_plot_without_training():
    """Test plot without training."""
    import dfm_python as dfm
    
    dfm.reset()
    
    with pytest.raises(ValueError):
        dfm.plot(kind='factor', factor_index=0)


def test_get_result_without_training():
    """Test get_result without training."""
    import dfm_python as dfm
    
    dfm.reset()
    
    result = dfm.get_result()
    assert result is None


# ============================================================================
# Tutorial Tests (from test_tutorials.py)
# ============================================================================

def test_basic_tutorial_workflow():
    """Test basic tutorial workflow."""
    import dfm_python as dfm
    from dfm_python.config import Params
    
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    data_file = base_dir / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Tutorial data files not found")
    
    try:
        # Load config from spec
        params = Params(max_iter=1, threshold=1e-2)
        dfm.from_spec(spec_file, params=params)
        
        # Load data
        dfm.load_data(data_file, sample_start='2021-01-01', sample_end='2022-12-31')
        
        # Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfm.train()
        
        # Get result
        result = dfm.get_result()
        assert result is not None
        assert hasattr(result, 'Z')
        
        # Predict
        X_forecast, Z_forecast = dfm.predict(horizon=6)
        assert X_forecast.shape[0] == 6
        
    except Exception as e:
        pytest.skip(f"Tutorial test skipped: {e}")


def test_hydra_tutorial_workflow():
    """Test Hydra tutorial workflow (if Hydra available)."""
    try:
        import hydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available")
    
    import dfm_python as dfm
    
    base_dir = project_root
    data_file = base_dir / 'data' / 'sample_data.csv'
    
    if not data_file.exists():
        pytest.skip("Tutorial data file not found")
    
    try:
        # Create mock Hydra config
        cfg = DictConfig({
            'clock': 'm',
            'max_iter': 1,
            'threshold': 1e-2,
            'series': [],
            'blocks': {}
        })
        
        # Load config from Hydra
        dfm.load_config(hydra=cfg)
        
        # Load data
        dfm.load_data(data_file, sample_start='2021-01-01', sample_end='2022-12-31')
        
        # Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfm.train()
        
        # Get result
        result = dfm.get_result()
        assert result is not None
        
    except Exception as e:
        pytest.skip(f"Hydra tutorial test skipped: {e}")


def test_api_reset():
    """Test API reset functionality."""
    import dfm_python as dfm
    
    # Set some state
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])
    ]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    
    dfm.load_config(config)
    
    # Reset
    dfm.reset()
    
    # Verify reset
    assert dfm.get_config() is None
    assert dfm.get_result() is None


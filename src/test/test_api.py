"""Tests for API edge cases and tutorials - consolidated."""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.api import DFM
from dfm_python.dfm import DFMResult
from dfm_python.config import (
    DFMConfig, SeriesConfig, BlockConfig,
    YamlSource, DictSource, MergedConfigSource,
    make_config_source
)
from dfm_python import load_config, load_data
from dfm_python.nowcast import Nowcast
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
    """Test with single time period - should handle gracefully with warnings."""
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
        model = DFM()
        result = model.fit(X, config, max_iter=1, threshold=1e-2)
        # Single time period may not converge, but should not crash
        assert result is not None
        # Should not converge with insufficient data
        assert not result.converged


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

def test_tutorial_smoke_test():
    """Smoke test for tutorial structure (no data files required).
    
    This test validates that tutorial modules can be imported and basic
    structure is correct, without requiring data files.
    """
    # Test that tutorial modules can be imported
    import dfm_python as dfm
    from dfm_python.config import Params
    
    # Verify basic API structure
    assert hasattr(dfm, 'from_spec')
    assert hasattr(dfm, 'load_data')
    assert hasattr(dfm, 'train')
    assert hasattr(dfm, 'predict')
    assert hasattr(dfm, 'get_result')
    
    # Verify Params can be instantiated
    params = Params(max_iter=10, threshold=1e-4)
    assert params.max_iter == 10
    assert params.threshold == 1e-4


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


# ============================================================================
# Config Source Tests
# ============================================================================

def test_series_config_basic():
    """Test basic SeriesConfig creation."""
    series = SeriesConfig(
        series_id='test_series',
        series_name='Test Series',
        frequency='m',
        transformation='lin',
        blocks=['Block_Global']
    )
    
    assert series.series_id == 'test_series'
    assert series.series_name == 'Test Series'
    assert series.frequency == 'm'
    assert series.transformation == 'lin'
    assert series.blocks == ['Block_Global']
    assert series.units is None
    assert series.release_date is None


def test_series_config_with_release():
    """Test SeriesConfig with release_date."""
    series = SeriesConfig(
        frequency='m',
        transformation='pch',
        blocks=['Block_Global'],
        release_date=25
    )
    
    assert series.release_date == 25


def test_block_config_basic():
    """Test basic BlockConfig creation."""
    block = BlockConfig(factors=2, ar_lag=1, clock='m')
    
    assert block.factors == 2
    assert block.ar_lag == 1
    assert block.clock == 'm'
    assert block.notes is None


def test_dfm_config_basic():
    """Test basic DFMConfig creation."""
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])
    ]
    blocks_dict = {
        'Block_Global': BlockConfig(factors=1, clock='m')
    }
    
    config = DFMConfig(series=series_list, blocks=blocks_dict)
    
    assert len(config.series) == 1
    assert len(config.blocks) == 1
    assert 'Block_Global' in config.blocks
    assert config.threshold == 1e-5
    assert config.max_iter == 5000
    assert config.clock == 'm'


def test_yaml_source():
    """Test YAML config source."""
    yaml_file = project_root / 'config' / 'default.yaml'
    
    if not yaml_file.exists():
        pytest.skip(f"YAML config file not found: {yaml_file}")
    
    source = YamlSource(yaml_file)
    config = source.load()
    
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0
    assert len(config.blocks) > 0


def test_dict_source():
    """Test dictionary config source."""
    config_dict = {
        'series': [
            {
                'series_id': 'test_1',
                'series_name': 'Test Series 1',
                'frequency': 'm',
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
        ],
        'blocks': {
            'Block_Global': {
                'factors': 1,
                'clock': 'm'
            }
        },
        'clock': 'm',
        'max_iter': 100,
        'threshold': 1e-5
    }
    
    source = DictSource(config_dict)
    config = source.load()
    
    assert isinstance(config, DFMConfig)
    assert len(config.series) == 1
    assert config.series[0].series_id == 'test_1'
    assert config.max_iter == 100


def test_merged_config_source():
    """Test merged config source."""
    base_dict = {
        'series': [
            {
                'series_id': 'base_1',
                'series_name': 'Base Series 1',
                'frequency': 'm',
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
        ],
        'blocks': {
            'Block_Global': {
                'factors': 1,
                'clock': 'm'
            }
        }
    }
    
    override_dict = {
        'max_iter': 200,
        'threshold': 1e-5
    }
    
    base_source = DictSource(base_dict)
    override_source = DictSource(override_dict)
    merged_source = MergedConfigSource(base_source, override_source)
    
    config = merged_source.load()
    
    assert config.max_iter == 200
    assert config.threshold == 1e-5
    assert len(config.series) > 0


def test_make_config_source():
    """Test config source factory function."""
    yaml_file = project_root / 'config' / 'default.yaml'
    
    if yaml_file.exists():
        source = make_config_source(yaml_file)
        assert isinstance(source, YamlSource)
    
    config_dict = {
        'series': [
            {
                'series_id': 'test_1',
                'frequency': 'm',
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
        ],
        'blocks': {
            'Block_Global': {
                'factors': 1,
                'clock': 'm'
            }
        }
    }
    source = make_config_source(config_dict)
    assert isinstance(source, DictSource)
    
    spec_file = project_root / 'data' / 'sample_spec.csv'
    if spec_file.exists():
        with pytest.raises(ValueError):
            make_config_source(spec=spec_file)
    
    series_list = [
        SeriesConfig(
            series_id='test_1',
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        )
    ]
    config = DFMConfig(series=series_list, blocks={'Block_Global': BlockConfig(factors=1, clock='m')})
    source = make_config_source(config)
    assert hasattr(source, 'load')
    assert source.load() is config


def test_from_dict():
    """Test from_dict convenience constructor."""
    config_dict = {
        'series': [
            {
                'series_id': 'test_1',
                'frequency': 'm',
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
        ],
        'blocks': {
            'Block_Global': {
                'factors': 1,
                'clock': 'm'
            }
        }
    }
    
    config = DFMConfig.from_dict(config_dict)
    assert isinstance(config, DFMConfig)
    assert len(config.series) == 1


def test_from_spec_conversion():
    """Test from_spec() utility function for CSV to YAML conversion."""
    from dfm_python.config_sources import from_spec
    
    csv_file = project_root / 'data' / 'sample_spec.csv'
    if not csv_file.exists():
        pytest.skip(f"Sample spec CSV not found: {csv_file}")
    
    series_path, blocks_path = from_spec(csv_file)
    
    assert series_path.exists()
    assert blocks_path.exists()
    assert series_path.name == 'sample_spec.yaml'
    assert blocks_path.name == 'sample_spec.yaml'


def test_hydra_compose_api():
    """Test Hydra compose API for loading configs programmatically."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available. Install with: pip install hydra-core")
    
    GlobalHydra.instance().clear()
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        pytest.skip(f"Config directory not found: {config_dir}")
    
    initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
    
    try:
        cfg = compose(config_name="default")
        assert isinstance(cfg, DictConfig)
        
        from dfm_python.config_sources import HydraSource
        config = HydraSource(cfg).load()
        
        assert isinstance(config, DFMConfig)
        assert len(config.series) > 0
        assert len(config.blocks) > 0
    finally:
        GlobalHydra.instance().clear()


# ============================================================================
# News Decomposition Tests
# ============================================================================


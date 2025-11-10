"""Tests for config sources and convenience API - consolidated."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.config import (
    DFMConfig, SeriesConfig, BlockConfig,
    YamlSource, DictSource, SpecCSVSource, HydraSource, MergedConfigSource,
    make_config_source
)
from dfm_python.api import DFM
import dfm_python as dfm

# ============================================================================
# Config Source Tests (from test_config_sources.py)
# ============================================================================

def test_yaml_source():
    """Test YAML config source."""
    base_dir = project_root
    yaml_file = base_dir / 'config' / 'default.yaml'
    
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


def test_spec_csv_source():
    """Test spec CSV config source."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    
    if not spec_file.exists():
        pytest.skip(f"Spec CSV file not found: {spec_file}")
    
    source = SpecCSVSource(spec_file)
    config = source.load()
    
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0
    assert len(config.blocks) > 0


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
        # No series or blocks - partial config (will use base's series and blocks)
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
    # Test with YAML path
    base_dir = project_root
    yaml_file = base_dir / 'config' / 'default.yaml'
    
    if yaml_file.exists():
        source = make_config_source(yaml_file)
        assert isinstance(source, YamlSource)
    
    # Test with dict
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
    
    # Test with DFMConfig
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
    # make_config_source returns a DFMConfigAdapter wrapper, not the config itself
    assert hasattr(source, 'load')
    assert source.load() is config


# ============================================================================
# Convenience API Tests (from test_convenience_api.py)
# ============================================================================

def test_from_yaml():
    """Test from_yaml convenience constructor."""
    base_dir = project_root
    yaml_file = base_dir / 'config' / 'default.yaml'
    
    if not yaml_file.exists():
        pytest.skip(f"YAML config file not found: {yaml_file}")
    
    # Use YamlSource directly (DFMConfig.from_yaml doesn't exist)
    source = YamlSource(yaml_file)
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0


def test_from_spec():
    """Test from_spec convenience constructor."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    
    if not spec_file.exists():
        pytest.skip(f"Spec CSV file not found: {spec_file}")
    
    # Use SpecCSVSource directly (DFMConfig.from_spec doesn't exist)
    source = SpecCSVSource(spec_file)
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0


def test_from_spec_df():
    """Test from_spec_df convenience constructor."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    
    if not spec_file.exists():
        pytest.skip(f"Spec CSV file not found: {spec_file}")
    
    # Use SpecCSVSource with DataFrame (DFMConfig.from_spec_df doesn't exist)
    df = pd.read_csv(spec_file)
    # Convert DataFrame to dict and use DictSource
    from dfm_python.data_loader import _load_config_from_dataframe
    config = _load_config_from_dataframe(df)
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0


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


def test_module_level_api():
    """Test module-level convenience API."""
    # Test load_config
    base_dir = project_root
    yaml_file = base_dir / 'config' / 'default.yaml'
    
    if yaml_file.exists():
        dfm.load_config(yaml_file)
        config = dfm.get_config()
        assert isinstance(config, DFMConfig)


def test_train_predict_workflow():
    """Test complete train-predict workflow with config sources."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    
    # Create config using DictSource
    config_dict = {
        'series': [
            {
                'series_id': f'test_{i}',
                'frequency': 'm',
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
            for i in range(N)
        ],
        'blocks': {
            'Block_Global': {
                'factors': 1,
                'clock': 'm'
            }
        }
    }
    
    config = DFMConfig.from_dict(config_dict)
    
    # Train model
    model = DFM()
    result = model.fit(X, config, max_iter=5, threshold=1e-2)
    
    assert result is not None
    assert result.x_sm.shape == (T, N)
    
    # Predict
    Xf, Zf = model.predict(horizon=6)
    assert Xf.shape[0] == 6
    assert Zf.shape[0] == 6

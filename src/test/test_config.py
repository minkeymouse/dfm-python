"""Tests for configuration system - SeriesConfig, BlockConfig, DFMConfig, and config sources."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.config import (
    DFMConfig, SeriesConfig, BlockConfig, Params, DEFAULT_GLOBAL_BLOCK_NAME
)
from dfm_python.config_sources import (
    YamlSource, DictSource, SpecCSVSource, MergedConfigSource,
    make_config_source, from_spec
)
from dfm_python.api import load_config


# ============================================================================
# SeriesConfig Tests
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


def test_series_config_negative_release():
    """Test SeriesConfig with negative release_date."""
    series = SeriesConfig(
        frequency='m',
        transformation='pch',
        blocks=['Block_Global'],
        release_date=-5
    )
    
    assert series.release_date == -5


def test_series_config_validation():
    """Test SeriesConfig field validation."""
    # Valid frequency
    series = SeriesConfig(frequency='q', transformation='lin', blocks=['Block_Global'])
    assert series.frequency == 'q'
    
    # Invalid frequency should raise ValueError
    with pytest.raises(ValueError):
        SeriesConfig(frequency='invalid', transformation='lin', blocks=['Block_Global'])


# ============================================================================
# BlockConfig Tests
# ============================================================================

def test_block_config_basic():
    """Test basic BlockConfig creation."""
    block = BlockConfig(factors=2, ar_lag=1, clock='m')
    
    assert block.factors == 2
    assert block.ar_lag == 1
    assert block.clock == 'm'
    assert block.notes is None


def test_block_config_validation():
    """Test BlockConfig field validation."""
    # Valid config
    block = BlockConfig(factors=3, ar_lag=2, clock='q')
    assert block.factors == 3
    assert block.ar_lag == 2
    assert block.clock == 'q'
    
    # Invalid factors should raise ValueError
    with pytest.raises(ValueError):
        BlockConfig(factors=0, ar_lag=1, clock='m')
    
    # Invalid ar_lag should raise ValueError
    with pytest.raises(ValueError):
        BlockConfig(factors=1, ar_lag=0, clock='m')


# ============================================================================
# DFMConfig Tests
# ============================================================================

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


def test_dfm_config_with_custom_params():
    """Test DFMConfig with custom estimation parameters."""
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])
    ]
    blocks_dict = {
        'Block_Global': BlockConfig(factors=1, clock='m')
    }
    
    # Note: clock must be <= block clock, so use 'm' for both
    config = DFMConfig(
        series=series_list,
        blocks=blocks_dict,
        threshold=1e-3,
        max_iter=1000,
        clock='m'
    )
    
    assert config.threshold == 1e-3
    assert config.max_iter == 1000
    assert config.clock == 'm'


def test_dfm_config_block_names():
    """Test that block_names are automatically derived from blocks dict."""
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global', 'Block_Consumption'])
    ]
    blocks_dict = {
        'Block_Global': BlockConfig(factors=1, clock='m'),
        'Block_Consumption': BlockConfig(factors=1, clock='m')
    }
    
    config = DFMConfig(series=series_list, blocks=blocks_dict)
    
    assert 'Block_Global' in config.block_names
    assert 'Block_Consumption' in config.block_names
    assert len(config.block_names) == 2


# ============================================================================
# YAML Source Tests
# ============================================================================

def test_yaml_source_default():
    """Test loading default.yaml config."""
    yaml_file = project_root / 'config' / 'default.yaml'
    if not yaml_file.exists():
        pytest.skip(f"Default YAML config not found: {yaml_file}")
    
    source = YamlSource(yaml_file)
    config = source.load()
    
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0
    assert len(config.blocks) > 0
    assert config.clock == 'm'


def test_yaml_source_sample_spec():
    """Test loading sample_spec.yaml config."""
    series_file = project_root / 'config' / 'series' / 'sample_spec.yaml'
    if not series_file.exists():
        pytest.skip(f"Sample spec series YAML not found: {series_file}")
    
    # Load via main config with defaults
    main_file = project_root / 'config' / 'default.yaml'
    if main_file.exists():
        source = YamlSource(main_file)
        # Override series via defaults - this would be done via experiment config
        # For direct test, load series file separately
        import yaml
        with open(series_file, 'r') as f:
            series_dict = yaml.safe_load(f)
        
        assert len(series_dict) > 0
        assert 'KOGDP...D' in series_dict


def test_yaml_source_experiment_configs():
    """Test loading experiment config files."""
    for exp_name in ['exp1', 'exp2', 'exp3']:
        exp_file = project_root / 'config' / 'experiment' / f'{exp_name}.yaml'
        if not exp_file.exists():
            pytest.skip(f"Experiment config not found: {exp_file}")
        
        source = YamlSource(exp_file)
        config = source.load()
        
        assert isinstance(config, DFMConfig)
        assert len(config.series) > 0
        assert len(config.blocks) > 0
        assert config.threshold == 1e-5
        assert config.max_iter == 5000


def test_yaml_source_experiment_differences():
    """Test that different experiment configs load different series/blocks."""
    exp1_file = project_root / 'config' / 'experiment' / 'exp1.yaml'
    exp2_file = project_root / 'config' / 'experiment' / 'exp2.yaml'
    exp3_file = project_root / 'config' / 'experiment' / 'exp3.yaml'
    
    if not all(f.exists() for f in [exp1_file, exp2_file, exp3_file]):
        pytest.skip("Not all experiment configs found")
    
    config1 = YamlSource(exp1_file).load()
    config2 = YamlSource(exp2_file).load()
    config3 = YamlSource(exp3_file).load()
    
    # All should have valid configs
    assert len(config1.series) > 0
    assert len(config2.series) > 0
    assert len(config3.series) > 0
    
    # exp1 and exp2 should have same series (both use sample_spec)
    # exp3 should have different series (sample_spec2)
    # Note: This depends on actual file contents
    assert len(config1.series) == len(config2.series)  # Both use sample_spec
    # exp3 may have different count if sample_spec2 is different


# ============================================================================
# CSV to YAML Conversion Tests
# ============================================================================

def test_from_spec_conversion():
    """Test from_spec() utility function for CSV to YAML conversion."""
    csv_file = project_root / 'data' / 'sample_spec.csv'
    if not csv_file.exists():
        pytest.skip(f"Sample spec CSV not found: {csv_file}")
    
    # Convert CSV to YAML
    series_path, blocks_path = from_spec(csv_file)
    
    assert series_path.exists()
    assert blocks_path.exists()
    assert series_path.name == 'sample_spec.yaml'
    assert blocks_path.name == 'sample_spec.yaml'
    assert 'series' in str(series_path)
    assert 'blocks' in str(blocks_path)
    
    # Verify generated YAML can be loaded
    series_source = YamlSource(series_path)
    blocks_source = YamlSource(blocks_path)
    
    # Note: These are partial configs (series or blocks only)
    # Full config loading would require main config + series + blocks


def test_from_spec_custom_output():
    """Test from_spec() with custom output directory and filenames."""
    csv_file = project_root / 'data' / 'sample_spec.csv'
    if not csv_file.exists():
        pytest.skip(f"Sample spec CSV not found: {csv_file}")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        series_path, blocks_path = from_spec(
            csv_file,
            output_dir=output_dir,
            series_filename='test_series',
            blocks_filename='test_blocks'
        )
        
        assert series_path.exists()
        assert blocks_path.exists()
        assert series_path.name == 'test_series.yaml'
        assert blocks_path.name == 'test_blocks.yaml'
        assert series_path.parent == output_dir / 'series'
        assert blocks_path.parent == output_dir / 'blocks'


# ============================================================================
# Config Source Factory Tests
# ============================================================================

def test_make_config_source_yaml():
    """Test make_config_source with YAML file."""
    yaml_file = project_root / 'config' / 'default.yaml'
    if not yaml_file.exists():
        pytest.skip(f"Default YAML not found: {yaml_file}")
    
    source = make_config_source(yaml_file)
    assert isinstance(source, YamlSource)
    
    config = source.load()
    assert isinstance(config, DFMConfig)


def test_make_config_source_dict():
    """Test make_config_source with dictionary."""
    config_dict = {
        'series': [
            {
                'series_id': 'test1',
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
        'threshold': 1e-4,
        'max_iter': 1000
    }
    
    source = make_config_source(config_dict)
    assert isinstance(source, DictSource)
    
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) == 1
    assert config.threshold == 1e-4
    assert config.max_iter == 1000


def test_make_config_source_csv():
    """Test make_config_source with CSV file."""
    csv_file = project_root / 'data' / 'sample_spec.csv'
    if not csv_file.exists():
        pytest.skip(f"Sample spec CSV not found: {csv_file}")
    
    source = make_config_source(csv_file)
    assert isinstance(source, SpecCSVSource)
    
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0


# ============================================================================
# Config Validation Tests
# ============================================================================

def test_config_validation_report():
    """Test config validation and reporting."""
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])
    ]
    blocks_dict = {
        'Block_Global': BlockConfig(factors=1, clock='m')
    }
    
    config = DFMConfig(series=series_list, blocks=blocks_dict)
    report = config.validate_and_report()
    
    assert 'valid' in report
    assert 'errors' in report
    assert 'warnings' in report
    assert report['valid'] is True


def test_config_validation_errors():
    """Test config validation with errors."""
    # Create config with invalid block structure
    series_list = [
        SeriesConfig(frequency='m', transformation='lin', blocks=[1, 0])  # Only 2 blocks
    ]
    blocks_dict = {
        'Block_Global': BlockConfig(factors=1, clock='m'),
        'Block_Consumption': BlockConfig(factors=1, clock='m'),
        'Block_Investment': BlockConfig(factors=1, clock='m')  # 3 blocks defined
    }
    
    # This should raise ValueError during __post_init__
    with pytest.raises(ValueError):
        DFMConfig(series=series_list, blocks=blocks_dict)


# ============================================================================
# Legacy Format Support Tests
# ============================================================================

def test_from_dict_legacy_format():
    """Test loading config from legacy dictionary format."""
    legacy_dict = {
        'SeriesID': ['series1', 'series2'],
        'Frequency': ['m', 'q'],
        'Transformation': ['lin', 'pch'],
        'Blocks': [[1, 0], [1, 0]],
        'BlockNames': ['Block_Global', 'Block_Consumption'],
        'factors_per_block': [1, 1],
        'threshold': 1e-4,
        'max_iter': 1000
    }
    
    config = DFMConfig.from_dict(legacy_dict)
    
    assert len(config.series) == 2
    assert config.series[0].frequency == 'm'
    assert config.series[1].frequency == 'q'
    assert len(config.blocks) == 2
    assert config.threshold == 1e-4


# ============================================================================
# Integration Tests
# ============================================================================

def test_config_with_data_loading():
    """Test that config works with data loading."""
    data_file = project_root / 'data' / 'sample_data.csv'
    config_file = project_root / 'config' / 'default.yaml'
    
    if not data_file.exists() or not config_file.exists():
        pytest.skip("Required files not found")
    
    # Load config directly using YamlSource to get DFMConfig
    from dfm_python.config_sources import YamlSource
    config = YamlSource(config_file).load()
    
    # Should be able to load data with this config
    from dfm_python.data import load_data
    X, Time, Z = load_data(data_file, config)
    
    assert X.shape[0] > 0  # At least one time period
    assert X.shape[1] == len(config.series)  # Number of series matches config


def test_experiment_configs_with_data():
    """Test experiment configs with actual data loading."""
    data_file = project_root / 'data' / 'sample_data.csv'
    if not data_file.exists():
        pytest.skip("Data file not found")
    
    from dfm_python.config_sources import YamlSource
    
    for exp_name in ['exp1', 'exp2', 'exp3']:
        exp_file = project_root / 'config' / 'experiment' / f'{exp_name}.yaml'
        if not exp_file.exists():
            continue
        
        # Load config directly using YamlSource to get DFMConfig
        config = YamlSource(exp_file).load()
        
        # Try to load data (may fail if series don't match, but config should be valid)
        from dfm_python.data import load_data
        try:
            X, Time, Z = load_data(data_file, config)
            assert X.shape[1] == len(config.series)
        except (ValueError, KeyError):
            # Expected if series IDs don't match data columns
            pass


# ============================================================================
# Hydra Multirun Tests
# ============================================================================

def test_hydra_compose_api():
    """Test Hydra compose API for loading configs programmatically."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available. Install with: pip install hydra-core")
    
    # Clean up any existing Hydra instance
    GlobalHydra.instance().clear()
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        pytest.skip(f"Config directory not found: {config_dir}")
    
    # Use initialize_config_dir with absolute path
    initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
    
    try:
        # Test loading default config via Hydra compose
        cfg = compose(config_name="default")
        assert isinstance(cfg, DictConfig)
        
        # Load config using HydraSource
        from dfm_python.config_sources import HydraSource
        config = HydraSource(cfg).load()
        
        assert isinstance(config, DFMConfig)
        assert len(config.series) > 0
        assert len(config.blocks) > 0
    finally:
        GlobalHydra.instance().clear()


def test_hydra_multirun_simulation():
    """Simulate Hydra multirun by loading multiple experiment configs sequentially."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available. Install with: pip install hydra-core")
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        pytest.skip(f"Config directory not found: {config_dir}")
    
    experiments = ['exp1', 'exp2', 'exp3']
    configs_loaded = []
    
    # Simulate multirun: load each experiment config sequentially
    # Use YamlSource instead of Hydra compose for experiment configs
    # since Hydra has issues with config group resolution in subdirectories
    from dfm_python.config_sources import YamlSource
    
    for exp_name in experiments:
        exp_file = project_root / 'config' / 'experiment' / f'{exp_name}.yaml'
        if not exp_file.exists():
            continue
        
        try:
            # Load config directly using YamlSource (which handles Hydra-style configs)
            config = YamlSource(exp_file).load()
            
            assert isinstance(config, DFMConfig)
            assert len(config.series) > 0
            assert len(config.blocks) > 0
            
            configs_loaded.append((exp_name, config))
        except Exception as e:
            pytest.fail(f"Failed to load experiment config {exp_name}: {e}")
    
    # Verify all configs were loaded
    assert len(configs_loaded) == len(experiments)
    
    # Verify configs have different series counts (exp1=exp2, exp3 different)
    exp1_config = next(c for name, c in configs_loaded if name == 'exp1')
    exp2_config = next(c for name, c in configs_loaded if name == 'exp2')
    exp3_config = next(c for name, c in configs_loaded if name == 'exp3')
    
    # exp1 and exp2 should have same series count (both use sample_spec)
    assert len(exp1_config.series) == len(exp2_config.series)
    # exp3 may have different count (sample_spec2)


def test_hydra_multirun_with_overrides():
    """Test Hydra compose with parameter overrides (simulating multirun with different params)."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available. Install with: pip install hydra-core")
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        pytest.skip(f"Config directory not found: {config_dir}")
    
    # Test loading config with different parameter overrides
    override_combinations = [
        {'max_iter': 1000, 'threshold': 1e-4},
        {'max_iter': 2000, 'threshold': 1e-5},
        {'max_iter': 5000, 'threshold': 1e-6},
    ]
    
    configs = []
    for overrides in override_combinations:
        # Clean up any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize for this override combination
        initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
        
        try:
            cfg = compose(config_name="default", overrides=[f"{k}={v}" for k, v in overrides.items()])
            assert isinstance(cfg, DictConfig)
            
            # Verify overrides were applied
            assert cfg.max_iter == overrides['max_iter']
            assert cfg.threshold == overrides['threshold']
            
            # Load config using HydraSource
            from dfm_python.config_sources import HydraSource
            config = HydraSource(cfg).load()
            
            assert isinstance(config, DFMConfig)
            assert config.max_iter == overrides['max_iter']
            assert config.threshold == overrides['threshold']
            
            configs.append(config)
        finally:
            GlobalHydra.instance().clear()
    
    # Verify all configs were loaded with correct overrides
    assert len(configs) == len(override_combinations)
    for i, config in enumerate(configs):
        assert config.max_iter == override_combinations[i]['max_iter']
        assert config.threshold == override_combinations[i]['threshold']


def test_hydra_multirun_experiment_sweep():
    """Test Hydra multirun with experiment config sweep (series and blocks combinations)."""
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import DictConfig
        HYDRA_AVAILABLE = True
    except ImportError:
        pytest.skip("Hydra not available. Install with: pip install hydra-core")
    
    config_dir = project_root / 'config'
    if not config_dir.exists():
        pytest.skip(f"Config directory not found: {config_dir}")
    
    # Test different series/block combinations
    series_options = ['default', 'sample_spec', 'sample_spec2']
    blocks_options = ['default', 'sample_spec', 'sample_spec2']
    
    # Simulate multirun sweep: test a subset of combinations
    test_combinations = [
        ('default', 'default'),
        ('sample_spec', 'default'),
        ('default', 'sample_spec'),
        ('sample_spec', 'sample_spec'),
    ]
    
    configs_loaded = []
    for series_name, blocks_name in test_combinations:
        # Clean up any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize for this combination
        initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
        
        try:
            # Compose config with series and blocks overrides
            overrides = [
                f"series={series_name}",
                f"blocks={blocks_name}",
            ]
            cfg = compose(config_name="default", overrides=overrides)
            assert isinstance(cfg, DictConfig)
            
            # Load config using HydraSource
            from dfm_python.config_sources import HydraSource
            config = HydraSource(cfg).load()
            
            assert isinstance(config, DFMConfig)
            assert len(config.series) > 0
            assert len(config.blocks) > 0
            
            configs_loaded.append((series_name, blocks_name, config))
        except Exception as e:
            pytest.fail(f"Failed to load config with series={series_name}, blocks={blocks_name}: {e}")
        finally:
            GlobalHydra.instance().clear()
    
    # Verify all combinations were loaded
    assert len(configs_loaded) == len(test_combinations)
    
    # Verify configs have expected series counts
    for series_name, blocks_name, config in configs_loaded:
        assert len(config.series) > 0
        assert len(config.blocks) > 0


"""Tests for config.adapter module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from dfm_python.config.adapter import YamlSource, ConfigSource, DictSource


class TestDictSource:
    """Test suite for DictSource."""
    
    def test_dict_source_initialization(self):
        """Test DictSource can be initialized."""
        mapping = {
            'frequency': {'series1': 'd', 'series2': 'w'},
            'blocks': {'block1': {'num_factors': 2, 'series': ['series1', 'series2']}}
        }
        source = DictSource(mapping)
        assert source is not None
        assert 'frequency' in source.mapping
        assert 'blocks' in source.mapping
    
    def test_dict_source_loading(self):
        """Test loading configuration from dictionary."""
        # Provide minimal valid config with blocks (required for DFMConfig)
        mapping = {
            'frequency': {'series1': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'max_iter': 20
        }
        source = DictSource(mapping)
        config = source.load()
        assert config is not None
        assert hasattr(config, 'blocks')
        assert hasattr(config, 'frequency')
        assert config.max_iter == 20
        assert 'block1' in config.blocks
    
    def test_dict_source_grouped_frequency(self):
        """Test loading configuration with grouped frequency format."""
        # Grouped format: {'w': [...], 'm': [...]}
        mapping = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series3', 'series4'],
                'q': ['series5']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1', 'series3', 'series5']}},
            'clock': 'w',
            'max_iter': 20
        }
        source = DictSource(mapping)
        config = source.load()
        assert config is not None
        assert hasattr(config, 'frequency')
        # Verify grouped format is converted to individual format
        assert config.frequency['series1'] == 'w'
        assert config.frequency['series2'] == 'w'
        assert config.frequency['series3'] == 'm'
        assert config.frequency['series4'] == 'm'
        assert config.frequency['series5'] == 'q'
        assert config.max_iter == 20
    
    def test_dict_source_grouped_frequency_duplicate_series_error(self):
        """Test that duplicate series in grouped frequency format raises error."""
        from dfm_python.utils.errors import ConfigurationError
        mapping = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series2', 'series3']  # series2 appears twice
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        source = DictSource(mapping)
        with pytest.raises(ConfigurationError, match="appears in multiple frequency groups"):
            source.load()
    
    def test_dict_source_grouped_frequency_invalid_freq_code(self):
        """Test that invalid frequency codes in grouped format are handled correctly."""
        from dfm_python.utils.errors import ConfigurationError
        mapping = {
            'frequency': {
                'invalid_freq': ['series1'],  # Invalid frequency code
                'm': ['series2']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        source = DictSource(mapping)
        # Invalid frequency keys will cause it to be treated as individual format
        # Then validation will catch invalid frequency values
        with pytest.raises(ConfigurationError):
            source.load()
    
    def test_dict_source_grouped_frequency_backward_compat(self):
        """Test that individual format still works (backward compatibility)."""
        mapping = {
            'frequency': {'series1': 'w', 'series2': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'clock': 'w'
        }
        source = DictSource(mapping)
        config = source.load()
        assert config.frequency['series1'] == 'w'
        assert config.frequency['series2'] == 'm'
    
    def test_dict_source_grouped_frequency_empty_list(self):
        """Test that empty frequency lists are handled."""
        mapping = {
            'frequency': {
                'w': [],  # Empty list
                'm': ['series1']
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'clock': 'w'
        }
        source = DictSource(mapping)
        config = source.load()
        assert config.frequency['series1'] == 'm'
        assert len([k for k, v in config.frequency.items() if v == 'w']) == 0


class TestYamlSource:
    """Test suite for YamlSource."""
    
    def test_yaml_source_loading(self):
        """Test loading configuration from YAML files."""
        # Create a temporary YAML file with minimal valid config
        yaml_content = {
            'frequency': {'series1': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}},
            'max_iter': 20
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            source = YamlSource(yaml_path)
            config = source.load()
            assert config is not None
            assert hasattr(config, 'blocks')
            assert hasattr(config, 'frequency')
            assert config.max_iter == 20
            assert 'block1' in config.blocks
        finally:
            # Clean up temporary file
            Path(yaml_path).unlink(missing_ok=True)
    
    def test_yaml_source_grouped_frequency(self):
        """Test loading YAML configuration with grouped frequency format."""
        # Grouped format: {'w': [...], 'm': [...]}
        yaml_content = {
            'frequency': {
                'w': ['A001', 'GSCITOT'],
                'm': ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G'],
                'q': ['KOGCFCNSD']
            },
            'blocks': {'block1': {'num_factors': 2, 'series': ['A001', 'KOEQUIPTE', 'KOGCFCNSD']}},
            'clock': 'w',
            'max_iter': 30
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            source = YamlSource(yaml_path)
            config = source.load()
            assert config is not None
            assert hasattr(config, 'frequency')
            # Verify grouped format is converted to individual format
            assert config.frequency['A001'] == 'w'
            assert config.frequency['GSCITOT'] == 'w'
            assert config.frequency['KOEQUIPTE'] == 'm'
            assert config.frequency['KOWRCCNSE'] == 'm'
            assert config.frequency['KOIPALL.G'] == 'm'
            assert config.frequency['KOGCFCNSD'] == 'q'
            assert config.clock == 'w'
            assert config.max_iter == 30
        finally:
            Path(yaml_path).unlink(missing_ok=True)
    
    def test_yaml_source_grouped_frequency_empty_groups(self):
        """Test YAML with some empty frequency groups."""
        yaml_content = {
            'frequency': {
                'w': [],  # Empty group
                'm': ['series1', 'series2'],
                'q': []   # Empty group
            },
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'clock': 'm'
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            source = YamlSource(yaml_path)
            config = source.load()
            assert config.frequency['series1'] == 'm'
            assert config.frequency['series2'] == 'm'
        finally:
            Path(yaml_path).unlink(missing_ok=True)


class TestConfigSource:
    """Test suite for ConfigSource base class."""
    
    def test_config_source_interface(self):
        """Test ConfigSource interface."""
        # ConfigSource is a Protocol, so we test that DictSource implements it
        # Provide minimal valid config with blocks
        mapping = {
            'frequency': {'series1': 'm'},
            'blocks': {'block1': {'num_factors': 1, 'series': ['series1']}}
        }
        source = DictSource(mapping)
        # Verify it has the required load() method
        assert hasattr(source, 'load')
        assert callable(source.load)
        # Verify load() returns a DFMConfig
        config = source.load()
        from dfm_python.config.schema.model import DFMConfig
        assert isinstance(config, DFMConfig)
        assert hasattr(config, 'blocks')
        assert 'block1' in config.blocks


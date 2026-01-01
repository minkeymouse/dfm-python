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


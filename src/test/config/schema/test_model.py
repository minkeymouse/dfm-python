"""Tests for config.schema.model module."""

import pytest


class TestModelSchema:
    """Test suite for model schema."""
    
    def test_model_schema_validation(self):
        """Test model schema validation."""
        from dfm_python.config.schema.model import DFMConfig
        # Test that DFMConfig validates blocks structure
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': ['series1', 'series2']}},
            frequency={'series1': 'm', 'series2': 'm'}
        )
        assert config is not None
        assert 'block1' in config.blocks
        assert config.blocks['block1']['num_factors'] == 2
        assert len(config.block_names) == 1
        assert config.block_names[0] == 'block1'
    
    def test_ddfm_config_with_grouped_frequency(self):
        """Test DDFMConfig from_dict with grouped frequency format."""
        from dfm_python.config.schema.model import DDFMConfig
        config_dict = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series3', 'series4']
            },
            'clock': 'w',
            'num_factors': 2,
            'encoder_layers': [64, 32]
        }
        config = DDFMConfig.from_dict(config_dict)
        assert config is not None
        assert config.frequency['series1'] == 'w'
        assert config.frequency['series2'] == 'w'
        assert config.frequency['series3'] == 'm'
        assert config.frequency['series4'] == 'm'
        assert config.clock == 'w'
    
    def test_dfm_config_from_dict_grouped_frequency(self):
        """Test DFMConfig.from_dict with grouped frequency format."""
        from dfm_python.config.schema.model import DFMConfig
        config_dict = {
            'frequency': {
                'w': ['series1', 'series2'],
                'm': ['series3']
            },
            'clock': 'w',
            'blocks': {
                'block1': {'num_factors': 1, 'series': ['series1', 'series3']}
            }
        }
        config = DFMConfig.from_dict(config_dict)
        assert config is not None
        assert config.frequency['series1'] == 'w'
        assert config.frequency['series2'] == 'w'
        assert config.frequency['series3'] == 'm'


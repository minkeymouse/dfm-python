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


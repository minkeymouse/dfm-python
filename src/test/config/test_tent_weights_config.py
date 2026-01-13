"""Tests for tent weights parameterization in DFMConfig."""

import pytest
import numpy as np
import yaml
from pathlib import Path
from dfm_python.config import DFMConfig
from dfm_python.config.adapter import YamlSource
from dfm_python.dataset import DFMDataset
from dfm_python.numeric.tent import get_agg_structure


class TestTentWeightsConfig:
    """Test suite for tent weights configuration."""
    
    def test_tent_weights_in_config_dict(self):
        """Test that tent_weights parameter can be set in config dict."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m:w': [1, 2, 1]},  # Custom tent weights
        }
        
        config = DFMConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm:w' in config.tent_weights
        assert config.tent_weights['m:w'] == [1, 2, 1]
    
    def test_tent_weights_override_lookup(self):
        """Test that config tent_weights are used correctly (no lookup table anymore)."""
        # Custom tent weights different from lookup
        custom_weights = [1, 2, 3, 2, 1]  # 5 periods
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m:w': custom_weights},
        }
        
        config = DFMConfig.from_dict(config_dict)
        
        # Verify config has custom weights
        assert config.tent_weights['m:w'] == custom_weights
        
        # Verify get_agg_structure uses config weights
        agg_structure = get_agg_structure(config, clock='w')
        assert 'm' in agg_structure['tent_weights']
        assert np.array_equal(agg_structure['tent_weights']['m'], np.array(custom_weights))
    
    def test_tent_weights_required_for_mixed_freq(self):
        """Test that tent_weights are required for mixed-frequency data (no fallback)."""
        from dfm_python.utils.errors import DataValidationError
        
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            # No tent_weights specified - should raise error for mixed-frequency data
        }
        
        config = DFMConfig.from_dict(config_dict)
        assert config.tent_weights is None
        
        # Should raise error when trying to get aggregation structure for mixed-frequency data
        with pytest.raises(DataValidationError, match="tent_weights"):
            get_agg_structure(config, clock='w')
    
    def test_tent_weights_frequency_shorthand(self):
        """Test that tent_weights can use frequency-only key (shorthand)."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'w': ['series2']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2']}},
            'tent_weights': {'m': [1, 2, 1]},  # Shorthand: just frequency, uses clock
        }
        
        config = DFMConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm' in config.tent_weights
        assert config.tent_weights['m'] == [1, 2, 1]
    
    def test_tent_weights_multiple_pairs(self):
        """Test tent_weights with multiple frequency pairs."""
        config_dict = {
            'clock': 'w',
            'frequency': {'m': ['series1'], 'q': ['series2'], 'w': ['series3']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['series1', 'series2', 'series3']}},
            'tent_weights': {
                'm:w': [1, 2, 1],  # Monthly -> Weekly
                'q:w': [1, 2, 3, 4, 5, 4, 3, 2, 1],  # Quarterly -> Weekly
            },
        }
        
        config = DFMConfig.from_dict(config_dict)
        assert config.tent_weights is not None
        assert 'm:w' in config.tent_weights
        assert 'q:w' in config.tent_weights
        assert config.tent_weights['m:w'] == [1, 2, 1]
        assert len(config.tent_weights['q:w']) == 9


class TestInvestmentProductionConfigs:
    """Test suite for actual investment and production config files."""
    
    def test_investment_config_tent_weights(self):
        """Test that investment config has correct tent weights."""
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent / 'config' / 'model' / 'dfm' / 'investment.yaml',
            Path(__file__).parent.parent.parent.parent / 'config' / 'model' / 'dfm' / 'investment.yaml',
            Path.cwd() / 'config' / 'model' / 'dfm' / 'investment.yaml',
        ]
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            pytest.skip(f"Investment config not found at any of: {possible_paths}")
        
        source = YamlSource(str(config_path))
        config = source.load()
        
        # Verify tent weights are set
        assert config.tent_weights is not None, "Investment config should have tent_weights"
        assert 'm:w' in config.tent_weights, "Investment config should have m:w tent weights"
        
        # Verify investment uses 5-period tent kernel
        expected_weights = [1, 2, 3, 2, 1]
        actual_weights = config.tent_weights['m:w']
        assert actual_weights == expected_weights, \
            f"Investment config should use {expected_weights}, got {actual_weights}"
    
    def test_production_config_tent_weights(self):
        """Test that production config has correct tent weights."""
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent.parent.parent.parent / 'config' / 'model' / 'dfm' / 'production.yaml',
            Path(__file__).parent.parent.parent.parent / 'config' / 'model' / 'dfm' / 'production.yaml',
            Path.cwd() / 'config' / 'model' / 'dfm' / 'production.yaml',
        ]
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            pytest.skip(f"Production config not found at any of: {possible_paths}")
        
        source = YamlSource(str(config_path))
        config = source.load()
        
        # Verify tent weights are set
        assert config.tent_weights is not None, "Production config should have tent_weights"
        assert 'm:w' in config.tent_weights, "Production config should have m:w tent weights"
        
        # Verify production uses 3-period tent kernel
        expected_weights = [1, 2, 1]
        actual_weights = config.tent_weights['m:w']
        assert actual_weights == expected_weights, \
            f"Production config should use {expected_weights}, got {actual_weights}"
    
    def test_config_tent_weights_different(self):
        """Test that investment and production configs use different tent weights."""
        # Try multiple possible paths
        possible_bases = [
            Path(__file__).parent.parent.parent.parent.parent / 'config' / 'model' / 'dfm',
            Path(__file__).parent.parent.parent.parent / 'config' / 'model' / 'dfm',
            Path.cwd() / 'config' / 'model' / 'dfm',
        ]
        
        investment_path = None
        production_path = None
        for config_base in possible_bases:
            inv_path = config_base / 'investment.yaml'
            prod_path = config_base / 'production.yaml'
            if inv_path.exists() and prod_path.exists():
                investment_path = inv_path
                production_path = prod_path
                break
        
        if investment_path is None or production_path is None:
            pytest.skip(f"Config files not found. Tried: {possible_bases}")
        
        investment_source = YamlSource(str(investment_path))
        production_source = YamlSource(str(production_path))
        
        investment_config = investment_source.load()
        production_config = production_source.load()
        
        # Verify they have different tent weights
        investment_weights = investment_config.tent_weights.get('m:w') if investment_config.tent_weights else None
        production_weights = production_config.tent_weights.get('m:w') if production_config.tent_weights else None
        
        assert investment_weights is not None, "Investment config should have tent weights"
        assert production_weights is not None, "Production config should have tent weights"
        assert investment_weights != production_weights, \
            f"Investment and production should have different tent weights. " \
            f"Investment: {investment_weights}, Production: {production_weights}"


class TestDatasetTentWeightsIntegration:
    """Test suite for dataset integration with tent weights."""
    
    def test_dataset_uses_config_tent_weights(self):
        """Test that DFMDataset uses tent_weights from config."""
        import pandas as pd
        
        # Create minimal data
        dates = pd.date_range('2020-01-01', periods=20, freq='W')
        data = pd.DataFrame({
            'weekly_series': np.random.randn(20),
            'monthly_series': [np.nan] * 19 + [1.0],  # Only last value (monthly observation)
        }, index=dates)
        
        # Config with custom tent weights
        config_dict = {
            'clock': 'w',
            'frequency': {'w': ['weekly_series'], 'm': ['monthly_series']},
            'blocks': {'Block_Global': {'num_factors': 1, 'series': ['weekly_series', 'monthly_series']}},
            'tent_weights': {'m:w': [1, 2, 1]},  # 3-period tent kernel
        }
        
        config = DFMConfig.from_dict(config_dict)
        
        # Create dataset
        dataset = DFMDataset(config=config, data=data)
        
        # Verify dataset has tent weights
        init_params = dataset.get_initialization_params()
        assert 'tent_weights_dict' in init_params
        assert init_params['tent_weights_dict'] is not None
        
        # Verify tent weights match config
        tent_weights_dict = init_params['tent_weights_dict']
        assert 'm' in tent_weights_dict
        assert np.array_equal(tent_weights_dict['m'], np.array([1, 2, 1], dtype=np.float32))

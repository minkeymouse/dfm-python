"""Tests for numeric.builder module."""

import pytest
import numpy as np
from dfm_python.numeric.builder import build_dfm_structure, build_dfm_blocks
from dfm_python.config import DFMConfig


class TestBuilder:
    """Test suite for builder utilities."""
    
    def test_build_dfm_structure(self):
        """Test build_dfm_structure."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 2, 'series': ['s1', 's2']}},
            frequency={'s1': 'w', 's2': 'w'},
            clock='w'
        )
        
        blocks, r, num_factors, p = build_dfm_structure(config)
        assert blocks is not None
        assert r is not None
        assert num_factors > 0
        assert p >= 0
    
    def test_build_dfm_blocks(self):
        """Test build_dfm_blocks."""
        config = DFMConfig(
            blocks={'block1': {'num_factors': 1, 'series': ['s1', 's2', 's3']}},
            frequency={'s1': 'w', 's2': 'w', 's3': 'w'},
            clock='w'
        )
        
        initial_blocks = np.ones((2, 1))
        columns = ['s1', 's2', 's3']
        n_series = 3
        
        blocks = build_dfm_blocks(initial_blocks, config, columns, n_series)
        assert blocks.shape[0] == n_series
        assert blocks.shape[1] == 1

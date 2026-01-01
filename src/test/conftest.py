"""Pytest configuration and shared fixtures for dfm_python tests."""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from dfm_python.config import DFMConfig


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_series = 5
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='ME')  # 'ME' replaces deprecated 'M'
    data = np.random.randn(n_samples, n_series)
    df = pd.DataFrame(data, index=dates, columns=[f'series_{i}' for i in range(n_series)])
    
    return df


@pytest.fixture
def sample_config():
    """Create a basic DFMConfig for testing."""
    # DFMConfig requires blocks with at least one block
    # Also needs frequency dict or columns for validation
    config = DFMConfig(
        max_iter=10,
        threshold=1e-4,
        blocks={'block1': {'num_factors': 2, 'series': ['series_0', 'series_1']}},
        frequency={'series_0': 'm', 'series_1': 'm'}
    )
    return config


@pytest.fixture
def sample_frequency_dict():
    """Create a sample frequency dictionary for testing."""
    return {
        'series_0': 'M',
        'series_1': 'M',
        'series_2': 'Q',
        'series_3': 'Q',
        'series_4': 'A'
    }


@pytest.fixture
def sample_block_structure():
    """Create a sample block structure for testing."""
    return {
        'block1': ['series_0', 'series_1'],
        'block2': ['series_2', 'series_3'],
        'block3': ['series_4']
    }


"""Pytest configuration and shared fixtures for dfm_python tests."""

import pytest
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from dfm_python.config import DFMConfig
from dfm_python.config.constants import DEFAULT_CONVERGENCE_THRESHOLD

# Note: PyTorch Lightning warnings removed - package now uses plain PyTorch

# Test-specific constants (smaller values for faster test execution)
TEST_N_SAMPLES = 100  # Number of time steps for test data
TEST_N_SERIES = 5  # Number of time series variables for test data
TEST_MAX_ITER = 10  # Maximum iterations for test config (smaller than DEFAULT_MAX_ITER for faster tests)


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_samples = TEST_N_SAMPLES
    n_series = TEST_N_SERIES
    
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
        max_iter=TEST_MAX_ITER,
        threshold=DEFAULT_CONVERGENCE_THRESHOLD,
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


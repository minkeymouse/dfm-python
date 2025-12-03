"""Shared test utilities for dfm-python tests.

This module provides common utility functions used across all test files
to avoid code duplication and ensure consistency.
"""

import numpy as np
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.models import DFM


def create_simple_config(num_series: int = 5, num_factors: int = 1) -> DFMConfig:
    """Create a simple DFMConfig for testing.
    
    Shared utility function used by all test files.
    
    Parameters
    ----------
    num_series : int, default 5
        Number of time series to include in the configuration
    num_factors : int, default 1
        Number of factors in the Block_Global block
        
    Returns
    -------
    DFMConfig
        A simple DFM configuration suitable for testing
    """
    series = [
        SeriesConfig(
            series_id=f'series_{i}',
            frequency='m',
            transformation='lin',
            blocks=['Block_Global']
        )
        for i in range(num_series)
    ]
    
    blocks = {
        'Block_Global': BlockConfig(factors=num_factors, ar_lag=1, clock='m')
    }
    
    return DFMConfig(
        series=series,
        blocks=blocks,
        max_iter=10,  # Small for fast testing
        threshold=1e-3,  # Relaxed for fast convergence
        clock='m'
    )


def generate_synthetic_data(n_periods: int = 50, n_series: int = 5) -> np.ndarray:
    """Generate synthetic time series data for testing.
    
    Shared utility function used by all test files.
    Uses deterministic random seed for reproducibility.
    
    Parameters
    ----------
    n_periods : int, default 50
        Number of time periods to generate
    n_series : int, default 5
        Number of time series to generate
        
    Returns
    -------
    np.ndarray
        Synthetic data matrix of shape (n_periods, n_series) with common factor structure
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate data with some common factor structure
    # Create a common factor
    common_factor = np.cumsum(np.random.randn(n_periods))
    
    # Each series is a combination of common factor + idiosyncratic noise
    X = np.zeros((n_periods, n_series))
    for i in range(n_series):
        loading = np.random.randn() * 0.5 + 1.0  # Random loading around 1.0
        X[:, i] = loading * common_factor + np.random.randn(n_periods) * 0.3
    
    return X


def create_simple_config_mapping(
    num_series: int = 5,
    num_factors: int = 1,
    max_iter: int = None,
    threshold: float = None,
    clock: str = 'm'
) -> dict:
    """Create a simple config mapping dictionary for testing.
    
    Shared utility function used by test files to create mapping dictionaries
    for load_config(mapping=...). This reduces code duplication across test files.
    
    Parameters
    ----------
    num_series : int, default 5
        Number of time series to include in the configuration
    num_factors : int, default 1
        Number of factors in the Block_Global block
    max_iter : int, optional
        Maximum iterations for EM algorithm (only included if provided)
    threshold : float, optional
        Convergence threshold (only included if provided)
    clock : str, default 'm'
        Clock frequency for all factors
        
    Returns
    -------
    dict
        A dictionary suitable for load_config(mapping=...) with series, blocks, and optional parameters
    """
    mapping = {
        'series': [
            {
                'series_id': f'series_{i}',
                'frequency': clock,
                'transformation': 'lin',
                'blocks': ['Block_Global']
            }
            for i in range(num_series)
        ],
        'blocks': {
            'Block_Global': {
                'factors': num_factors,
                'ar_lag': 1,
                'clock': clock
            }
        },
        'clock': clock
    }
    
    # Add optional parameters only if explicitly provided
    # This allows tests to omit them when not needed
    if max_iter is not None:
        mapping['max_iter'] = max_iter
    if threshold is not None:
        mapping['threshold'] = threshold
    
    return mapping


def check_missing_data_error(exception: Exception) -> bool:
    """Check if an exception is related to missing data handling.
    
    Shared utility function used by test files to check if an exception
    is related to missing data (NaN, missing, null values). This standardizes
    error checking patterns across test files.
    
    Parameters
    ----------
    exception : Exception
        The exception to check
        
    Returns
    -------
    bool
        True if the exception is related to missing data, False otherwise
    """
    error_msg = str(exception).lower()
    return "nan" in error_msg or "missing" in error_msg or "null" in error_msg


def create_trained_dfm_model(
    num_series: int = 5,
    num_factors: int = 1,
    n_periods: int = 50,
    max_iter: int = 10,
    threshold: float = 1e-3
) -> DFM:
    """Create and train a DFM model for testing.
    
    Helper function to create a trained DFM model that can be used for nowcast tests
    and other tests requiring a trained model. This function handles the full workflow:
    config creation, data generation, loading, and training.
    
    Shared utility function used by test files to create trained DFM models.
    This reduces code duplication across test files.
    
    Parameters
    ----------
    num_series : int, default 5
        Number of time series
    num_factors : int, default 1
        Number of factors
    n_periods : int, default 50
        Number of time periods
    max_iter : int, default 10
        Maximum EM iterations for training
    threshold : float, default 1e-3
        Convergence threshold for training
        
    Returns
    -------
    DFM
        Trained DFM model instance with result set
    """
    # Create model instance
    model = DFM()
    
    # Create config
    config = create_simple_config(num_series=num_series, num_factors=num_factors)
    model.load_config(source=config)
    
    # Generate and load data
    X = generate_synthetic_data(n_periods=n_periods, n_series=num_series)
    model.load_data(data=X)
    
    # Train the model
    model.train(max_iter=max_iter, threshold=threshold)
    
    # Verify training completed
    assert model.result is not None, "Model training failed - result is None"
    
    return model


__all__ = [
    'create_simple_config',
    'generate_synthetic_data',
    'create_simple_config_mapping',
    'check_missing_data_error',
    'create_trained_dfm_model',
]

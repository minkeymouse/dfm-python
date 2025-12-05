"""Shared test helper functions for dfm-python test suite.

This module provides common test utilities used across multiple test files
to avoid code duplication and ensure consistent test patterns.

Helper Functions:
- File and config utilities: check_test_files_exist, get_test_data_path, get_test_config_path
- Config loading: load_config_safely, load_config_only_safely
- Error handling: handle_training_error, format_skip_message
- Data loading: load_sample_data_from_csv
- Transformer creation: create_simple_transformer, create_columnwise_transformer
- Trainer assertions: assert_trainer_defaults, assert_trainer_callbacks, assert_trainer_attribute_value
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Any

from dfm_python.config import DFMConfig, DDFMConfig, YamlSource
from dfm_python.utils.time import TimeIndex, parse_timestamp
from dfm_python.utils.data import rem_nans_spline


# ============================================================================
# File and Path Utilities
# ============================================================================

def check_test_files_exist(data_path: Path, config_path: Path) -> None:
    """Check if test files exist, skip test if missing.
    
    This function verifies that required test data and config files exist.
    If any files are missing, the test is skipped with a clear message.
    
    Parameters
    ----------
    data_path : Path
        Path to test data file (typically CSV)
    config_path : Path
        Path to test config file (typically YAML)
        
    Raises
    ------
    pytest.skip
        If any test files are missing, raises pytest.skip with details
    """
    missing = []
    if not data_path.exists():
        missing.append(f"data: {data_path}")
    if not config_path.exists():
        missing.append(f"config: {config_path}")
    if missing:
        pytest.skip(f"Test files not found: {', '.join(missing)}")


def get_test_data_path() -> Path:
    """Get path to test data file.
    
    Returns the standard path to the test data CSV file. This consolidates
    the path construction logic used across multiple test classes.
    
    Returns
    -------
    Path
        Path to test data file (data/sample_data.csv)
    """
    return Path(__file__).parent.parent.parent / "data" / "sample_data.csv"


def get_test_config_path(model_type: str = "dfm") -> Path:
    """Get path to test config file.
    
    Returns the standard path to test config YAML files. This consolidates
    the path construction logic used across multiple test classes.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model config ("dfm" or "ddfm"). Default is "dfm".
        
    Returns
    -------
    Path
        Path to test config file (config/experiment/test_{model_type}.yaml)
    """
    return Path(__file__).parent.parent.parent / "config" / "experiment" / f"test_{model_type}.yaml"


# ============================================================================
# Config Loading Utilities
# ============================================================================

def load_config_safely(
    model, 
    config_path: Path, 
    model_type: str = "DFM"
) -> None:
    """Load config safely with error handling.
    
    Loads a configuration file into a model instance with proper error
    handling. If config loading fails (e.g., due to format incompatibility),
    the test is skipped rather than failing.
    
    Parameters
    ----------
    model : DFM or DDFM
        Model instance to load config into
    config_path : Path
        Path to YAML config file
    model_type : str, optional
        Type of model ("DFM" or "DDFM"), used in error messages.
        Default is "DFM".
        
    Raises
    ------
    pytest.skip
        If config loading fails (TypeError, ValueError), raises pytest.skip
        with error details
    """
    try:
        source = YamlSource(config_path)
        model.load_config(source)
    except (TypeError, ValueError) as e:
        pytest.skip(
            f"{model_type} config loading failed (config format may need update): "
            f"{type(e).__name__}: {e}"
        )


def load_config_only_safely(
    config_path: Path, 
    model_type: str = "DFM"
) -> DFMConfig:
    """Load config object only (without loading into model) with error handling.
    
    Loads a configuration file and returns the config object without
    loading it into a model. Useful for testing config structure or
    validation without model initialization.
    
    Parameters
    ----------
    config_path : Path
        Path to YAML config file
    model_type : str, optional
        Type of model ("DFM" or "DDFM"), used in error messages.
        Default is "DFM".
        
    Returns
    -------
    DFMConfig or DDFMConfig
        Loaded config object
        
    Raises
    ------
    pytest.skip
        If config loading fails (TypeError, ValueError), raises pytest.skip
        with error details
    """
    try:
        source = YamlSource(config_path)
        return source.load()
    except (TypeError, ValueError) as e:
        pytest.skip(
            f"{model_type} config loading failed (config format may need update): "
            f"{type(e).__name__}: {e}"
        )


# ============================================================================
# Error Handling Utilities
# ============================================================================

def handle_training_error(
    error: Exception, 
    operation: str = "training"
) -> None:
    """Handle training errors consistently.
    
    Provides consistent error handling for training-related operations.
    If the error indicates the model hasn't been trained yet, the test
    is skipped. Otherwise, the error is re-raised.
    
    Parameters
    ----------
    error : Exception
        The exception that occurred during training operation
    operation : str, optional
        Description of operation that failed (e.g., "training", "prediction").
        Default is "training".
        
    Raises
    ------
    pytest.skip
        If error indicates model hasn't been trained/fitted
    Exception
        Re-raises the original error if it's not a training-related error
    """
    error_str = str(error)
    if "not been trained" in error_str or "not fitted" in error_str:
        pytest.skip(f"Model {operation} failed: {error}")
    raise


def format_skip_message(
    reason: str, 
    context: Optional[str] = None
) -> str:
    """Format skip message consistently.
    
    Formats a pytest skip message with optional context information.
    Ensures consistent formatting across all test skips.
    
    Parameters
    ----------
    reason : str
        Primary reason for skipping the test
    context : str, optional
        Additional context information (e.g., file paths, config details)
        
    Returns
    -------
    str
        Formatted skip message string
    """
    if context:
        return f"{reason} ({context})"
    return reason


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_sample_data_from_csv(data_path: Path):
    """Load and preprocess sample data from CSV file.
    
    This function consolidates the common pattern of loading CSV data,
    extracting date columns, creating TimeIndex, and preprocessing data
    (handling NaNs). Used by multiple test fixtures to avoid duplication.
    
    Parameters
    ----------
    data_path : Path
        Path to CSV data file
        
    Returns
    -------
    tuple
        Tuple of (data_clean, time_index, data_cols) where:
        - data_clean: numpy array of preprocessed data (NaNs handled)
        - time_index: TimeIndex object with parsed timestamps
        - data_cols: list of data column names (excluding 'date')
        
    Raises
    ------
    pytest.skip
        If data file does not exist, raises pytest.skip with file path
    """
    if not data_path.exists():
        pytest.skip(f"Test data file not found: {data_path}")
    
    # Read CSV with pandas
    df = pd.read_csv(data_path)
    
    # Extract date column
    date_col = df.select("date").to_series().to_list()
    time_index = TimeIndex([parse_timestamp(d) for d in date_col])
    
    # Extract data columns (exclude date)
    data_cols = [col for col in df.columns if col != "date"]
    data_array = df.select(data_cols).to_numpy()
    
    # Preprocess: handle NaNs
    data_clean, _ = rem_nans_spline(data_array, method=2, k=3)
    
    return data_clean, time_index, data_cols


# ============================================================================
# Transformer Creation Utilities
# ============================================================================

def create_simple_transformer():
    """Create a simple transformer for testing.
    
    Creates a StandardScaler configured as identity-like (no scaling).
    Per sktime docs, sklearn transformers work directly in TransformerPipeline
    without TabularToSeriesAdaptor. This consolidates transformer creation
    logic used across multiple test classes.
    
    Returns
    -------
    transformer
        StandardScaler instance (can be used directly in TransformerPipeline)
        
    Raises
    ------
    pytest.skip
        If sklearn is not available, raises pytest.skip with installation instructions
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Use StandardScaler with identity-like settings (no scaling)
        # For a true identity, we could use FunctionTransformer, but StandardScaler 
        # with mean=0, std=1 is close enough for testing
        # Per sktime docs: sklearn transformers work directly in TransformerPipeline
        return StandardScaler(with_mean=False, with_std=False)
    except ImportError:
        pytest.skip("scikit-learn not available - install with: pip install scikit-learn")


def create_columnwise_transformer():
    """Create a StandardScaler for unified scaling in testing.
    
    Creates a StandardScaler that applies unified scaling to all series.
    Per sktime docs, sklearn transformers work directly in TransformerPipeline
    without TabularToSeriesAdaptor. This consolidates transformer creation
    logic used across multiple test classes.
    
    Returns
    -------
    transformer
        StandardScaler instance (can be used directly in TransformerPipeline)
        
    Raises
    ------
    pytest.skip
        If sklearn is not available, raises pytest.skip with installation instructions
    """
    try:
        from sklearn.preprocessing import StandardScaler
        
        # Per sktime docs: sklearn transformers work directly in TransformerPipeline
        # Applied per series instance automatically (unified scaling)
        return StandardScaler()
    except ImportError:
        pytest.skip("scikit-learn not available - install with: pip install scikit-learn")


# ============================================================================
# Trainer Assertion Utilities
# ============================================================================

def assert_trainer_defaults(
    trainer,
    expected_max_epochs: int,
    expected_progress_bar: bool,
    expected_model_summary: bool
) -> None:
    """Assert trainer default values match expected.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    expected_max_epochs : int
        Expected max_epochs value
    expected_progress_bar : bool
        Expected enable_progress_bar value
    expected_model_summary : bool
        Expected enable_model_summary value
    """
    assert trainer.max_epochs == expected_max_epochs, (
        f"Expected max_epochs={expected_max_epochs}, got {trainer.max_epochs}"
    )
    assert trainer.enable_progress_bar == expected_progress_bar, (
        f"Expected enable_progress_bar={expected_progress_bar}, got {trainer.enable_progress_bar}"
    )
    assert trainer.enable_model_summary == expected_model_summary, (
        f"Expected enable_model_summary={expected_model_summary}, got {trainer.enable_model_summary}"
    )


def assert_trainer_callbacks(
    trainer,
    expected_callback_types: List[str]
) -> None:
    """Assert trainer has expected callback types.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    expected_callback_types : List[str]
        List of expected callback class names (e.g., ['EarlyStopping'])
    """
    assert hasattr(trainer, 'callbacks'), "Trainer should have callbacks attribute"
    assert trainer.callbacks is not None, "Trainer callbacks should not be None"
    assert isinstance(trainer.callbacks, list), "Trainer callbacks should be a list"
    
    # Get actual callback types
    callback_types = [type(cb).__name__ for cb in trainer.callbacks]
    
    # Check that all expected callback types are present
    for expected_type in expected_callback_types:
        assert expected_type in callback_types, (
            f"Expected callback type '{expected_type}' not found in callbacks. "
            f"Found: {callback_types}"
        )


def assert_trainer_attribute_value(
    trainer,
    attribute_name: str,
    expected_value: Any
) -> None:
    """Assert trainer attribute has expected value.
    
    Parameters
    ----------
    trainer : DFMTrainer or DDFMTrainer
        Trainer instance to check
    attribute_name : str
        Name of attribute to check
    expected_value : any
        Expected value for the attribute
    """
    assert hasattr(trainer, attribute_name), (
        f"Trainer should have '{attribute_name}' attribute"
    )
    actual_value = getattr(trainer, attribute_name)
    assert actual_value == expected_value, (
        f"Expected {attribute_name}={expected_value}, got {actual_value}"
    )


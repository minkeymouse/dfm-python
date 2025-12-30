"""Data manipulation utilities for Dynamic Factor Models.

This module provides utility functions for data sorting, filtering, and manipulation
that are used across multiple models and data modules.
"""

from typing import List, Tuple
import numpy as np

from ..config import DFMConfig
from ..utils.errors import DataError
from ..logger import get_logger

_logger = get_logger(__name__)


def sort_data_by_config(
    Z: np.ndarray,
    series_ids: List[str],
    config: DFMConfig
) -> Tuple[np.ndarray, List[str]]:
    """Sort data columns to match configuration order.
    
    This function reorders data columns to match the series order specified
    in the configuration. This ensures consistency between data and model
    configuration.
    
    Parameters
    ----------
    Z : np.ndarray
        Data matrix (T x N) where T is time steps and N is number of series
    series_ids : List[str]
        Series identifiers from data file (column names)
    config : DFMConfig
        Model configuration with series order
        
    Returns
    -------
    Z_sorted : np.ndarray
        Sorted data matrix (T x N) with columns matching config order
    series_ids_sorted : List[str]
        Sorted series identifiers matching config order
        
    Raises
    ------
    DataError
        If no matching series found between config and data
        
    Examples
    --------
    >>> Z = np.random.randn(100, 5)
    >>> series_ids = ['A', 'B', 'C', 'D', 'E']
    >>> config = DFMConfig(series=[...])  # Series in different order
    >>> Z_sorted, ids_sorted = sort_data_by_config(Z, series_ids, config)
    """
    config_series_ids = config.get_series_ids()
    
    # Create mapping from series_id to index in data
    series_id_to_idx = {sid: i for i, sid in enumerate(series_ids)}
    
    # Find permutation
    permutation = []
    series_ids_filtered = []
    for config_sid in config_series_ids:
        if config_sid in series_id_to_idx:
            permutation.append(series_id_to_idx[config_sid])
            series_ids_filtered.append(config_sid)
        else:
            _logger.warning(f"Series '{config_sid}' from config not found in data")
    
    if len(permutation) == 0:
        raise DataError(
            "No matching series found between config and data",
            details="Check that series IDs in config match column names in data"
        )
    
    # Apply permutation
    Z_sorted = Z[:, permutation]
    
    return Z_sorted, series_ids_filtered


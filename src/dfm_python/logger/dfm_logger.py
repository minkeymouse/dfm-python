"""DFM-specific logging utilities.

This module provides DFM-specific logging functions for diagnostics and debugging.
"""

import numpy as np
from typing import Optional, List

from .logger import get_logger

_logger = get_logger(__name__)


def log_blocks_diagnostics(
    blocks: np.ndarray,
    columns: Optional[List[str]],
    N_actual: int
) -> None:
    """Log diagnostics about blocks array.
    
    Parameters
    ----------
    blocks : np.ndarray
        Blocks array
    columns : Optional[List[str]]
        Column names if available
    N_actual : int
        Number of series
    """
    n_in_block = np.sum(blocks[:, 0] > 0) if blocks.shape[1] > 0 else 0
    _logger.info(f"Blocks array: shape={blocks.shape}, series in Block_Global: {n_in_block}/{N_actual}")
    
    if n_in_block < N_actual:
        missing_indices = np.where(blocks[:, 0] == 0)[0]
        if columns and len(columns) > 0:
            missing_series = [columns[i] for i in missing_indices[:10]]
            _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} ({missing_series[:5]}...)")
        else:
            _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} (indices: {missing_indices[:10].tolist()}...)")

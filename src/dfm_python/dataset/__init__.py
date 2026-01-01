"""PyTorch Dataset and DataLoader utilities for Dynamic Factor Models.

This module provides PyTorch-compatible Dataset and DataLoader implementations
for PyTorch-based DFM models (e.g., DDFM, KDFM) that use gradient descent training.

This module provides:
- Dataset classes: DDFMDataset (windowed sequences), KDFMDataset (full sequences)
- DataLoader factories: create_ddfm_dataloader, create_kdfm_dataloader
- Data reading: read_data, load_data

Note: Transformation functions are application-specific and should be provided
by users in their preprocessing pipelines. The package does not include
transformation utilities to remain generic.
"""

from .dataset import DDFMDataset, KDFMDataset
from .dataloader import create_ddfm_dataloader, create_kdfm_dataloader
# Data reading/loading moved to datamodule.base
# Re-export for backward compatibility (lazy import to avoid circular dependencies)
def read_data(datafile):
    """Re-export of BaseDataModule.read_data for backward compatibility."""
    from ..datamodule.base import BaseDataModule
    return BaseDataModule.read_data(datafile)

def load_data(datafile, config, sample_start=None, sample_end=None):
    """Re-export of BaseDataModule.load_data for backward compatibility."""
    from ..datamodule.base import BaseDataModule
    # Call load_data as a static method (it's actually an instance method, but we need config)
    # Create minimal instance just for the method call
    dm = BaseDataModule(config=config)
    return dm.load_data(datafile, sample_start=sample_start, sample_end=sample_end)
from .process import (
    TimeIndex,
    _get_scaler,
)
# sort_data_by_config moved to datamodule.base._sort_data_by_config
# Re-export for backward compatibility
def sort_data_by_config(Z, series_ids, config):
    """Re-export of BaseDataModule._sort_data_by_config for backward compatibility."""
    from ..datamodule.base import BaseDataModule
    # Create minimal instance just for the method call
    dm = BaseDataModule(config=config)
    return dm._sort_data_by_config(Z, series_ids)

__all__ = [
    # Datasets
    'DDFMDataset',
    'KDFMDataset',
    # Dataloaders
    'create_ddfm_dataloader',
    'create_kdfm_dataloader',
    # Data reading
    'read_data',
    'load_data',
    # Preprocessing
    'TimeIndex',
    '_get_scaler',
    # Data utilities
    'sort_data_by_config',
]

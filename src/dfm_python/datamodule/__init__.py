"""DataModule classes for Dynamic Factor Models.

This package provides DataModule implementations for DFM, DDFM, and KDFM.
Includes both PyTorch Lightning DataModules and custom implementations.
"""

# Note: KalmanFilter and KalmanFilterState were removed as DFM now uses pykalman.
# If DDFM/KDFM need PyTorch-based Kalman filter, they should import it directly.
# For now, these are not exported to avoid import errors.

from .base import BaseDataModule
from .dfm_dm import DFMDataModule
from .ddfm_dm import DDFMDataModule
from .kdfm_dm import KDFMDataModule
from ..dataset.dataset import DDFMDataset, KDFMDataset

# DFMTrainingState is defined in models.dfm and exported here for convenience
from ..models.dfm import DFMTrainingState

__all__ = [
    # Base class
    'BaseDataModule',
    # Data handling
    'DFMDataModule',
    'DDFMDataModule',
    'KDFMDataModule',
    'DDFMDataset',
    'KDFMDataset',
    # Training state (defined in models.dfm)
    'DFMTrainingState',
]

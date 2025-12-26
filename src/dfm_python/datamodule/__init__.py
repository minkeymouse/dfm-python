"""DataModule classes for Dynamic Factor Models.

This package provides DataModule implementations for DFM, DDFM, and KDFM.
Includes both PyTorch Lightning DataModules and custom implementations.
"""

# Note: KalmanFilter and KalmanFilterState were removed as DFM now uses pykalman.
# If DDFM/KDFM need PyTorch-based Kalman filter, they should import it directly.
# For now, these are not exported to avoid import errors.

from .dfm_dm import DFMDataModule
from .ddfm_dm import DDFMDataModule
from .kdfm_dm import KDFMDataModule
from ..data.dataset import DFMDataset, DDFMDataset

from .utils import (
    create_scaling_transformer_from_config,
    create_uniform_scaling_transformer,
    create_preprocessing_pipeline_with_scaling,
    ScalingStrategy,
    DefaultScalingStrategy,
    NoScalingStrategy,
)

# DFMTrainingState is defined in models.dfm and exported here for convenience
from ..models.dfm import DFMTrainingState

__all__ = [
    # Data handling
    'DFMDataModule',
    'DDFMDataModule',
    'KDFMDataModule',
    'DFMDataset',
    'DDFMDataset',
    # Scaling utilities
    'create_scaling_transformer_from_config',
    'create_uniform_scaling_transformer',
    'create_preprocessing_pipeline_with_scaling',
    'ScalingStrategy',
    'DefaultScalingStrategy',
    'NoScalingStrategy',
    # Training state (defined in models.dfm)
    'DFMTrainingState',
]

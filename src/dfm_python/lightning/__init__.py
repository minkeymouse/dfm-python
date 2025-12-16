"""PyTorch Lightning modules for Dynamic Factor Models.

This package provides PyTorch Lightning implementations of DFM and DDFM.
"""

from ..ssm.kalman import (
    KalmanFilter,  # Module class
    KalmanFilterState,  # Dataclass
)

from ..ssm.em import (
    EMAlgorithm,  # Module class
    EMStepParams,  # Dataclass
)

from .dfm_dm import DFMDataModule
from .ddfm_dm import DDFMDataModule
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
    # Kalman filter
    'KalmanFilter',
    'KalmanFilterState',
    # EM algorithm
    'EMAlgorithm',
    'EMStepParams',
    # Data handling
    'DFMDataModule',
    'DDFMDataModule',
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

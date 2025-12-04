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

from .data_module import (
    DFMDataModule,
    DFMDataset,
)

# DFMLightningModule is an internal implementation used by DFMLinear and estimation.py.
# DDFMLightningModule is defined but not currently used - DDFM class uses DDFMModel directly.
# These are not part of the public API.
from .dfm_module import (
    DFMLightningModule,  # Internal: used by DFMLinear._train_with_lightning()
    DFMTrainingState,
)

from .ddfm_module import (
    DDFMLightningModule,  # Internal: defined but not currently used (DDFM uses DDFMModel)
    DDFMTrainingState,
)

__all__ = [
    # Kalman filter
    'KalmanFilter',  # New module class
    'KalmanFilterState',
    # EM algorithm
    'EMAlgorithm',  # New module class
    'EMStepParams',
    # Data handling
    'DFMDataModule',
    'DFMDataset',
    # Lightning modules (internal implementations)
    'DFMLightningModule',  # Internal: used by DFMLinear._train_with_lightning()
    'DFMTrainingState',
    'DDFMLightningModule',  # Internal: defined but not currently used
    'DDFMTrainingState',
]


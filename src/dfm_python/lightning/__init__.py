"""PyTorch Lightning modules for Dynamic Factor Models.

This package provides PyTorch Lightning implementations of DFM and DDFM,
replacing the legacy NumPy-based implementations.
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

from .dfm_module import (
    DFMLightningModule,
    DFMTrainingState,
)

from .ddfm_module import (
    DDFMLightningModule,
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
    # Lightning modules
    'DFMLightningModule',
    'DFMTrainingState',
    'DDFMLightningModule',
    'DDFMTrainingState',
]


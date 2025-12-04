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

# Note: DFMLightningModule and DDFMLightningModule consolidated into DFM and DDFM classes.
# These imports are kept for backward compatibility but are deprecated.
from .dfm_module import (
    DFMLightningModule,  # Deprecated: use DFM from models.dfm instead
    DFMTrainingState,
)

from .ddfm_module import (
    DDFMLightningModule,  # Deprecated: use DDFM from models.ddfm instead
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
    # Lightning modules (deprecated - use DFM and DDFM from models instead)
    'DFMLightningModule',  # Deprecated
    'DFMTrainingState',
    'DDFMLightningModule',  # Deprecated
    'DDFMTrainingState',
]


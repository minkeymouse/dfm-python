"""State-space model (SSM) modules.

This package provides:
- DFMKalmanFilter: Kalman filtering for DFM using pykalman
- CompanionSSM: Companion form state-space models
"""

from .kalman import DFMKalmanFilter
from .companion import CompanionSSM, MACompanionSSM, CompanionSSMBase

__all__ = [
    # Main modules
    'DFMKalmanFilter',
    # Companion SSM modules
    'CompanionSSM',
    'MACompanionSSM',
    'CompanionSSMBase',
]


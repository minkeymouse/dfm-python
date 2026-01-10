"""State-space model (SSM) modules.

This package provides:
- DFMKalmanFilter: Kalman filtering for DFM using pykalman
- CompanionSSM: Companion form state-space models (optional, for DDFM)
"""

from .kalman import DFMKalmanFilter

# Companion SSM modules are optional (used by DDFM, not DFM)
try:
    from .companion import CompanionSSM, MACompanionSSM, CompanionSSMBase
    _HAS_COMPANION = True
except ImportError:
    # Companion module not available - make these None or create stubs
    CompanionSSM = None
    MACompanionSSM = None
    CompanionSSMBase = None
    _HAS_COMPANION = False

__all__ = [
    # Main modules
    'DFMKalmanFilter',
    # Companion SSM modules (may be None if module not available)
    'CompanionSSM',
    'MACompanionSSM',
    'CompanionSSMBase',
]


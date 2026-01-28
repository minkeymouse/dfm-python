"""State-space model (SSM) modules.

This package provides:
- DFMKalmanFilter: Kalman filtering for DFM using pykalman
- BaseSSM: Base class for SSMs in iVDFM (innovations → factors)
- iVDFMCompanionSSM: Companion form SSM for iVDFM (block-diagonal, supports p=1 and p>1)
- CompanionSSM: Companion form state-space models (optional, for DDFM)
"""

from .kalman import DFMKalmanFilter
from .base import BaseSSM
from .companion import iVDFMCompanionSSM

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
    'BaseSSM',
    'iVDFMCompanionSSM',
    # Companion SSM modules (may be None if module not available)
    'CompanionSSM',
    'MACompanionSSM',
    'CompanionSSMBase',
]


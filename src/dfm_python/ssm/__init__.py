"""State-space model (SSM) modules.

This package provides:
- DFMKalmanFilter: Kalman filtering for DFM using pykalman
- Utilities: Numerical stability functions for SSM operations
"""

from .kalman import DFMKalmanFilter
from .utils import (
    check_finite,
    ensure_real,
    ensure_symmetric,
    ensure_real_and_symmetric,
    ensure_positive_definite,
    ensure_covariance_stable,
    clean_matrix,
    safe_inverse,
    safe_determinant,
    DEFAULT_MIN_EIGENVAL,
    DEFAULT_MIN_DIAGONAL_VARIANCE,
    DEFAULT_INV_REGULARIZATION,
)

from .companion import CompanionSSM, MACompanionSSM, CompanionSSMBase

__all__ = [
    # Main modules
    'DFMKalmanFilter',
    # Companion SSM modules
    'CompanionSSM',
    'MACompanionSSM',
    'CompanionSSMBase',
    # Utilities
    'check_finite',
    'ensure_real',
    'ensure_symmetric',
    'ensure_real_and_symmetric',
    'ensure_positive_definite',
    'ensure_covariance_stable',
    'clean_matrix',
    'safe_inverse',
    'safe_determinant',
    # Constants
    'DEFAULT_MIN_EIGENVAL',
    'DEFAULT_MIN_DIAGONAL_VARIANCE',
    'DEFAULT_INV_REGULARIZATION',
]


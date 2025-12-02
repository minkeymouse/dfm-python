"""State-space model (SSM) modules for PyTorch.

This package provides PyTorch implementations of:
- KalmanFilter: Kalman filtering and smoothing
- EMAlgorithm: Expectation-Maximization algorithm for DFM
- Utilities: Numerical stability functions for SSM operations
"""

from .kalman import KalmanFilter, KalmanFilterState
from .em import EMAlgorithm, EMStepParams
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

__all__ = [
    # Main modules
    'KalmanFilter',
    'KalmanFilterState',
    'EMAlgorithm',
    'EMStepParams',
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


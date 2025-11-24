"""Kalman filter module (backward compatibility re-export).

This module re-exports Kalman filter components for backward compatibility.
New code should use dfm_python.engine.kalman.* directly.
"""

from .engine.kalman import (
    run_kf,
    skf,
    fis,
    miss_data,
    KalmanFilterState,
)

__all__ = [
    'run_kf',
    'skf',
    'fis',
    'miss_data',
    'KalmanFilterState',
]

"""Expectation-Maximization core routines (initialization and one EM iteration).

This package contains:
- convergence: EM convergence checking
- initialization: Initial parameter estimation via PCA and OLS
- iteration: Single EM iteration (E-step via Kalman, M-step updates)
"""

from .convergence import em_converged
from .initialization import init_conditions, NaNHandlingOptions
from .iteration import em_step, EMStepParams

# Re-export constants for backward compatibility
from .initialization import (
    DEFAULT_AR_COEFFICIENT,
    DEFAULT_INNOVATION_VARIANCE,
    DEFAULT_IDIO_AR,
    DEFAULT_IDIO_VAR,
    DEFAULT_IDIO_COV,
    DEFAULT_OBSERVATION_VARIANCE,
    MIN_INNOVATION_VARIANCE,
    MIN_OBSERVATION_VARIANCE,
    MIN_EIGENVALUE_THRESHOLD,
    MIN_VARIANCE_THRESHOLD,
    MIN_DATA_COVERAGE_RATIO,
    MIN_EIGENVALUE_ABSOLUTE,
    MIN_EIGENVALUE_RELATIVE,
    MIN_LOADING,
    MAX_LOADING,
    MIN_AR,
    FALLBACK_AR,
    FALLBACK_SCALE,
)

from .convergence import (
    MIN_LOG_LIKELIHOOD_DELTA,
    DAMPING,
    MAX_LOADING_REPLACE,
)

__all__ = [
    'em_converged',
    'init_conditions',
    'em_step',
    'EMStepParams',
    'NaNHandlingOptions',
    # Constants
    'DEFAULT_AR_COEFFICIENT',
    'DEFAULT_INNOVATION_VARIANCE',
    'DEFAULT_IDIO_AR',
    'DEFAULT_IDIO_VAR',
    'DEFAULT_IDIO_COV',
    'DEFAULT_OBSERVATION_VARIANCE',
    'MIN_INNOVATION_VARIANCE',
    'MIN_OBSERVATION_VARIANCE',
    'MIN_EIGENVALUE_THRESHOLD',
    'MIN_VARIANCE_THRESHOLD',
    'MIN_DATA_COVERAGE_RATIO',
    'MIN_EIGENVALUE_ABSOLUTE',
    'MIN_EIGENVALUE_RELATIVE',
    'MIN_LOADING',
    'MAX_LOADING',
    'MIN_AR',
    'FALLBACK_AR',
    'FALLBACK_SCALE',
    'MIN_LOG_LIKELIHOOD_DELTA',
    'DAMPING',
    'MAX_LOADING_REPLACE',
]

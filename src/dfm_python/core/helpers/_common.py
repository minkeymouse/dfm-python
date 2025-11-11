"""Common constants and utilities for helper modules."""

import numpy as np

# Common exception types for numerical operations
NUMERICAL_EXCEPTIONS = (
    np.linalg.LinAlgError,
    ValueError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
)


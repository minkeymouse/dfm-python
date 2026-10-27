"""Internal exception classes, built on the public ones in :mod:`dfm_python.errors`.

The public classes are re-exported here unchanged, so ``utils.errors.ConfigurationError
is errors.ConfigurationError``. The internal classes derive from them:

============================  ==========================================
Exception                     Bases
============================  ==========================================
``ConfigValidationError``     ``ConfigurationError``
``UnsupportedInputError``     ``DataValidationError``
``ModelNotTrainedError``      alias of ``NotFittedError``
``ModelNotInitializedError``  ``DFMError``
``DataError``                 ``DFMError``
``PredictionError``           ``DFMError``
``NumericalStabilityError``   ``NumericalError``
============================  ==========================================
"""

from ..errors import ConfigurationError, DataValidationError, DFMError, NotFittedError, NumericalError

__all__ = [
    "DFMError",
    "ConfigurationError",
    "DataValidationError",
    "NotFittedError",
    "NumericalError",
    "ConfigValidationError",
    "UnsupportedInputError",
    "ModelNotTrainedError",
    "ModelNotInitializedError",
    "DataError",
    "PredictionError",
    "NumericalStabilityError",
]


class ModelNotInitializedError(DFMError):
    """Raised when model operations are attempted before initialization."""


# Engines raise ``ModelNotTrainedError``; the alias keeps their raise sites unchanged.
ModelNotTrainedError = NotFittedError


class DataError(DFMError):
    """Raised when data validation fails."""


class PredictionError(DFMError):
    """Raised when prediction fails."""


class UnsupportedInputError(DataValidationError):
    """Raised when an input slot, frequency layout or option is not supported by the model."""


class NumericalStabilityError(NumericalError):
    """Raised when numerical stability issues are detected (e.g., near-singular matrices)."""


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

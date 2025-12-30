"""Custom exception classes for dfm-python.

This module provides specialized exception classes for different error types,
enabling better error handling and more informative error messages.
"""

from typing import Optional


class DFMError(Exception):
    """Base exception class for all dfm-python errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        """Initialize error with message and optional details.
        
        Parameters
        ----------
        message : str
            Main error message
        details : str, optional
            Additional details about the error
        """
        self.message = message
        self.details = details
        if details:
            super().__init__(f"{message}\nDetails: {details}")
        else:
            super().__init__(message)


class ModelNotInitializedError(DFMError):
    """Raised when model operations are attempted before initialization."""
    pass


class ModelNotTrainedError(DFMError):
    """Raised when prediction or result extraction is attempted before training."""
    pass


class ConfigurationError(DFMError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(DFMError):
    """Raised when data validation fails."""
    pass


class NumericalError(DFMError):
    """Raised when numerical stability issues are detected."""
    pass


class PredictionError(DFMError):
    """Raised when prediction fails."""
    pass


class DataValidationError(DFMError):
    """Raised when data validation fails (e.g., scale mismatches)."""
    pass


class NumericalStabilityError(NumericalError):
    """Raised when numerical stability issues are detected (e.g., near-singular matrices)."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass

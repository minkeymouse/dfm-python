"""DDFM-specific logging utilities.

This module provides DDFM-specific logging classes and functions.
"""

from .train_logger import BaseTrainLogger
from .inference_logger import BaseInferenceLogger


class DDFMTrainLogger(BaseTrainLogger):
    """Training logger for DDFM models (gradient descent)."""
    
    def __init__(self, verbose: bool = True):
        """Initialize DDFM training logger."""
        super().__init__(model_name="DDFM", verbose=verbose)


class DDFMInferenceLogger(BaseInferenceLogger):
    """Inference logger for DDFM models."""
    
    def __init__(self, verbose: bool = True):
        """Initialize DDFM inference logger."""
        super().__init__(model_name="DDFM", verbose=verbose)

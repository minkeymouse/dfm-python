"""Custom logging utilities for dfm-python.

This package provides a custom logging system built on Python's standard logging module.
The package provides:
- Basic logging configuration (get_logger, setup_logging, configure_logging)
- Training process tracking (BaseTrainLogger and model-specific loggers: DFMTrainLogger, DDFMTrainLogger)
- Inference process tracking (BaseInferenceLogger and model-specific loggers)
- Convenience functions for common logging tasks (log_em_iteration, log_convergence, etc.)
"""

from .logger import (
    get_logger,
    setup_logging,
    configure_logging,
)

from .train_logger import (
    BaseTrainLogger,
    DFMTrainLogger,
    DDFMTrainLogger,
    log_training_start,
    log_training_step,
    log_training_end,
    log_em_iteration,
    log_training_epoch,
    log_convergence,
)

from .inference_logger import (
    BaseInferenceLogger,
    DFMInferenceLogger,
    DDFMInferenceLogger,
    log_inference_start,
    log_inference_step,
    log_inference_end,
    log_prediction,
)

__all__ = [
    # Basic logging
    'get_logger',
    'setup_logging',
    'configure_logging',
    # Training tracking - base and model-specific
    'BaseTrainLogger',
    'DFMTrainLogger',
    'DDFMTrainLogger',
    'log_training_start',
    'log_training_step',
    'log_training_end',
    'log_em_iteration',
    'log_training_epoch',
    'log_convergence',
    # Inference tracking - base and model-specific
    'BaseInferenceLogger',
    'DFMInferenceLogger',
    'DDFMInferenceLogger',
    'log_inference_start',
    'log_inference_step',
    'log_inference_end',
    'log_prediction',
]


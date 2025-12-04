"""Logging utilities for dfm-python.

This package provides:
- Basic logging configuration (get_logger, setup_logging)
- Training process tracking (TrainLogger)
- Inference process tracking (InferenceLogger)
"""

from .logger import (
    get_logger,
    setup_logging,
    configure_logging,
)

from .train_logger import (
    TrainLogger,
    log_training_start,
    log_training_step,
    log_training_end,
    log_em_iteration,
    log_training_epoch,
    log_convergence,
)

from .inference_logger import (
    InferenceLogger,
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
    # Training tracking
    'TrainLogger',
    'log_training_start',
    'log_training_step',
    'log_training_end',
    'log_em_iteration',
    'log_training_epoch',
    'log_convergence',
    # Inference tracking
    'InferenceLogger',
    'log_inference_start',
    'log_inference_step',
    'log_inference_end',
    'log_prediction',
]


"""Logging utilities for dfm-python.

This package provides:
- Basic logging configuration (get_logger, setup_logging)
- Training process tracking (BaseTrainLogger and model-specific loggers)
- Inference process tracking (BaseInferenceLogger and model-specific loggers)
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
    KDFMTrainLogger,
    TrainLogger,  # Backward compatibility alias
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
    KDFMInferenceLogger,
    InferenceLogger,  # Backward compatibility alias
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
    'KDFMTrainLogger',
    'TrainLogger',  # Backward compatibility
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
    'KDFMInferenceLogger',
    'InferenceLogger',  # Backward compatibility
    'log_inference_start',
    'log_inference_step',
    'log_inference_end',
    'log_prediction',
]


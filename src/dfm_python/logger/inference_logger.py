"""Inference process logging utilities.

This module provides specialized logging for inference/prediction processes,
including prediction steps, nowcasting, and forecast generation.
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np
from datetime import datetime

from .logger import get_logger

_logger = get_logger(__name__)


class InferenceLogger:
    """Logger for tracking inference/prediction process.
    
    This class provides structured logging for inference processes including:
    - Inference start/end
    - Prediction steps
    - Nowcasting updates
    - Forecast generation
    """
    
    def __init__(self, model_name: str = "DFM", verbose: bool = True):
        """Initialize inference logger.
        
        Parameters
        ----------
        model_name : str, default "DFM"
            Name of the model being used for inference
        verbose : bool, default True
            Whether to log detailed information
        """
        self.model_name = model_name
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.num_predictions: int = 0
        
    def start(self, task: str = "inference", **kwargs) -> None:
        """Log inference start.
        
        Parameters
        ----------
        task : str, default "inference"
            Type of inference task (e.g., "prediction", "nowcasting", "forecast")
        **kwargs
            Additional context to log
        """
        self.start_time = datetime.now()
        self.num_predictions = 0
        
        _logger.info(f"{'='*70}")
        _logger.info(f"Starting {self.model_name} {task}")
        _logger.info(f"{'='*70}")
        
        if kwargs and self.verbose:
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
        _logger.info("")
    
    def log_step(
        self,
        step: int,
        description: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log inference step.
        
        Parameters
        ----------
        step : int
            Step number
        description : str, optional
            Description of the step
        **kwargs
            Additional metrics to log
        """
        if self.verbose:
            msg = f"Step {step:4d}"
            if description:
                msg += f" | {description}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                elif isinstance(value, np.ndarray):
                    msg += f" | {key}: shape {value.shape}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
    
    def log_prediction(
        self,
        prediction_type: str = "prediction",
        horizon: Optional[int] = None,
        **kwargs
    ) -> None:
        """Log prediction generation.
        
        Parameters
        ----------
        prediction_type : str, default "prediction"
            Type of prediction (e.g., "point", "interval", "nowcast")
        horizon : int, optional
            Prediction horizon
        **kwargs
            Additional prediction information
        """
        self.num_predictions += 1
        
        if self.verbose:
            msg = f"Generated {prediction_type}"
            if horizon is not None:
                msg += f" (horizon={horizon})"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                elif isinstance(value, np.ndarray):
                    msg += f" | {key}: shape {value.shape}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
    
    def end(self, success: bool = True, **kwargs) -> None:
        """Log inference end.
        
        Parameters
        ----------
        success : bool, default True
            Whether inference completed successfully
        **kwargs
            Additional information to log (e.g., metrics, summary)
        """
        self.end_time = datetime.now()
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            _logger.info("")
            _logger.info(f"{'='*70}")
            if success:
                _logger.info(f"Inference completed successfully")
            else:
                _logger.error(f"Inference failed")
            
            _logger.info(f"  Duration: {duration:.2f} seconds")
            _logger.info(f"  Predictions generated: {self.num_predictions}")
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
            
            _logger.info(f"{'='*70}")
            _logger.info("")


# Convenience functions for backward compatibility and simpler usage

def log_inference_start(
    model_name: str = "DFM",
    task: str = "inference",
    **kwargs
) -> InferenceLogger:
    """Create and start an inference logger.
    
    Parameters
    ----------
    model_name : str, default "DFM"
        Name of the model being used
    task : str, default "inference"
        Type of inference task
    **kwargs
        Additional context to log
        
    Returns
    -------
    InferenceLogger
        Logger instance
    """
    logger = InferenceLogger(model_name=model_name)
    logger.start(task=task, **kwargs)
    return logger


def log_inference_step(
    logger: InferenceLogger,
    step: int,
    description: Optional[str] = None,
    **kwargs
) -> None:
    """Log an inference step.
    
    Parameters
    ----------
    logger : InferenceLogger
        Inference logger instance
    step : int
        Step number
    description : str, optional
        Description of the step
    **kwargs
        Additional metrics to log
    """
    logger.log_step(step, description=description, **kwargs)


def log_inference_end(
    logger: InferenceLogger,
    success: bool = True,
    **kwargs
) -> None:
    """Log inference end.
    
    Parameters
    ----------
    logger : InferenceLogger
        Inference logger instance
    success : bool, default True
        Whether inference completed successfully
    **kwargs
        Additional information to log
    """
    logger.end(success=success, **kwargs)


def log_prediction(
    prediction_type: str = "prediction",
    horizon: Optional[int] = None,
    **kwargs
) -> None:
    """Log prediction generation (convenience function).
    
    Parameters
    ----------
    prediction_type : str, default "prediction"
        Type of prediction
    horizon : int, optional
        Prediction horizon
    **kwargs
        Additional prediction information
    """
    msg = f"Generated {prediction_type}"
    if horizon is not None:
        msg += f" (horizon={horizon})"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            msg += f" | {key}: {value:.6f}"
        elif isinstance(value, np.ndarray):
            msg += f" | {key}: shape {value.shape}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


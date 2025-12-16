"""Inference process logging utilities.

This module provides specialized logging for inference/prediction processes,
including prediction steps, nowcasting, and forecast generation for both DFM and DDFM.
"""

import logging
from typing import Optional, Dict, Any, Union, List
import numpy as np
from datetime import datetime

from .logger import get_logger

_logger = get_logger(__name__)


class InferenceLogger:
    """Logger for tracking inference/prediction process for both DFM and DDFM.
    
    This class provides structured logging for inference processes including:
    - Inference start/end
    - Prediction steps
    - Nowcasting updates
    - Forecast generation
    - Prediction metrics
    """
    
    def __init__(
        self, 
        model_name: str = "DFM", 
        model_type: str = "dfm",
        verbose: bool = True
    ):
        """Initialize inference logger.
        
        Parameters
        ----------
        model_name : str, default "DFM"
            Name of the model being used for inference (e.g., "DFM", "DDFM")
        model_type : str, default "dfm"
            Type of model: "dfm" or "ddfm"
        verbose : bool, default True
            Whether to log detailed information
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.num_predictions: int = 0
        self.prediction_history: List[Dict[str, Any]] = []
        
    def start(self, task: str = "inference", **kwargs) -> None:
        """Log inference start.
        
        Parameters
        ----------
        task : str, default "inference"
            Type of inference task (e.g., "prediction", "nowcasting", "forecast")
        **kwargs
            Additional context to log (e.g., horizon, target_series, view_date)
        """
        self.start_time = datetime.now()
        self.num_predictions = 0
        self.prediction_history = []
        
        _logger.info(f"{'='*70}")
        _logger.info(f"Starting {self.model_name} {task} ({self.model_type.upper()})")
        _logger.info(f"{'='*70}")
        
        if kwargs and self.verbose:
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                elif isinstance(value, (list, tuple)):
                    _logger.info(f"  {key}: {value}")
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
            Type of prediction (e.g., "point", "interval", "nowcast", "forecast")
        horizon : int, optional
            Prediction horizon (number of periods ahead)
        **kwargs
            Additional prediction information (e.g., shape, metrics, confidence_intervals)
        """
        self.num_predictions += 1
        
        # Store prediction in history
        pred_info = {
            "type": prediction_type,
            "horizon": horizon,
            "timestamp": datetime.now()
        }
        pred_info.update(kwargs)
        self.prediction_history.append(pred_info)
        
        if self.verbose:
            msg = f"Generated {prediction_type}"
            if horizon is not None:
                msg += f" (horizon={horizon})"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                elif isinstance(value, np.ndarray):
                    msg += f" | {key}: shape {value.shape}"
                elif isinstance(value, (list, tuple)):
                    msg += f" | {key}: {value}"
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
            Additional information to log (e.g., metrics, summary, forecast_stats)
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
            
            _logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            _logger.info(f"  Predictions generated: {self.num_predictions}")
            
            # Log summary of prediction history
            if self.prediction_history and self.verbose:
                prediction_types = [p.get("type", "unknown") for p in self.prediction_history]
                unique_types = set(prediction_types)
                _logger.info(f"  Prediction types: {', '.join(sorted(unique_types))}")
                
                horizons = [p.get("horizon") for p in self.prediction_history if p.get("horizon") is not None]
                if horizons and len(horizons) > 0:
                    horizon_values = [h for h in horizons if h is not None]
                    if horizon_values:
                        _logger.info(f"  Horizons: {min(horizon_values)} to {max(horizon_values)} periods")
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                elif isinstance(value, (list, tuple)):
                    _logger.info(f"  {key}: {value}")
                else:
                    _logger.info(f"  {key}: {value}")
            
            _logger.info(f"{'='*70}")
            _logger.info("")


# Convenience functions for simpler usage

def log_inference_start(
    model_name: str = "DFM",
    model_type: str = "dfm",
    task: str = "inference",
    **kwargs
) -> InferenceLogger:
    """Create and start an inference logger.
    
    Parameters
    ----------
    model_name : str, default "DFM"
        Name of the model being used (e.g., "DFM", "DDFM")
    model_type : str, default "dfm"
        Type of model: "dfm" or "ddfm"
    task : str, default "inference"
        Type of inference task
    **kwargs
        Additional context to log
        
    Returns
    -------
    InferenceLogger
        Logger instance
    """
    logger = InferenceLogger(model_name=model_name, model_type=model_type)
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
        Type of prediction (e.g., "point", "interval", "nowcast", "forecast")
    horizon : int, optional
        Prediction horizon (number of periods ahead)
    **kwargs
        Additional prediction information (e.g., shape, metrics, confidence_intervals)
    """
    msg = f"Generated {prediction_type}"
    if horizon is not None:
        msg += f" (horizon={horizon})"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            msg += f" | {key}: {value:.6f}"
        elif isinstance(value, np.ndarray):
            msg += f" | {key}: shape {value.shape}"
        elif isinstance(value, (list, tuple)):
            msg += f" | {key}: {value}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


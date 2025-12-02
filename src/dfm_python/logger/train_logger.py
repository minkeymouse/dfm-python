"""Training process logging utilities.

This module provides specialized logging for training/inference processes,
including EM iterations, convergence tracking, and training metrics.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

from .logger import get_logger

_logger = get_logger(__name__)


class TrainLogger:
    """Logger for tracking training process.
    
    This class provides structured logging for training processes including:
    - Training start/end
    - EM iterations
    - Convergence status
    - Training metrics (loss, log-likelihood, etc.)
    """
    
    def __init__(self, model_name: str = "DFM", verbose: bool = True):
        """Initialize training logger.
        
        Parameters
        ----------
        model_name : str, default "DFM"
            Name of the model being trained
        verbose : bool, default True
            Whether to log detailed information
        """
        self.model_name = model_name
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.iterations: int = 0
        self.converged: bool = False
        
    def start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Log training start.
        
        Parameters
        ----------
        config : dict, optional
            Training configuration to log
        """
        self.start_time = datetime.now()
        self.iterations = 0
        self.converged = False
        
        _logger.info(f"{'='*70}")
        _logger.info(f"Starting {self.model_name} training")
        _logger.info(f"{'='*70}")
        
        if config and self.verbose:
            _logger.info("Training configuration:")
            for key, value in config.items():
                _logger.info(f"  {key}: {value}")
        _logger.info("")
    
    def log_iteration(
        self,
        iteration: int,
        loglik: Optional[float] = None,
        delta: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log EM iteration information.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        loglik : float, optional
            Log-likelihood value
        delta : float, optional
            Convergence delta (change in log-likelihood)
        **kwargs
            Additional metrics to log
        """
        self.iterations = iteration
        
        if self.verbose:
            msg = f"Iteration {iteration:4d}"
            if loglik is not None:
                msg += f" | Log-likelihood: {loglik:12.6f}"
            if delta is not None:
                msg += f" | Delta: {delta:10.6e}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
        else:
            # Less verbose: log every 10 iterations
            if iteration % 10 == 0 or iteration == 1:
                msg = f"Iteration {iteration:4d}"
                if loglik is not None:
                    msg += f" | Log-likelihood: {loglik:12.6f}"
                _logger.info(msg)
    
    def log_convergence(
        self,
        converged: bool,
        num_iter: int,
        final_loglik: Optional[float] = None,
        reason: Optional[str] = None
    ) -> None:
        """Log convergence status.
        
        Parameters
        ----------
        converged : bool
            Whether training converged
        num_iter : int
            Number of iterations completed
        final_loglik : float, optional
            Final log-likelihood value
        reason : str, optional
            Reason for stopping (e.g., "converged", "max_iterations")
        """
        self.converged = converged
        self.iterations = num_iter
        
        _logger.info("")
        if converged:
            _logger.info(f"✓ Training converged after {num_iter} iterations")
        else:
            _logger.warning(f"⚠ Training did not converge after {num_iter} iterations")
            if reason:
                _logger.warning(f"  Reason: {reason}")
        
        if final_loglik is not None:
            _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")
    
    def end(self, success: bool = True, **kwargs) -> None:
        """Log training end.
        
        Parameters
        ----------
        success : bool, default True
            Whether training completed successfully
        **kwargs
            Additional information to log (e.g., metrics, warnings)
        """
        self.end_time = datetime.now()
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            _logger.info("")
            _logger.info(f"{'='*70}")
            if success:
                _logger.info(f"Training completed successfully")
            else:
                _logger.error(f"Training failed")
            
            _logger.info(f"  Duration: {duration:.2f} seconds")
            _logger.info(f"  Iterations: {self.iterations}")
            _logger.info(f"  Converged: {self.converged}")
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                else:
                    _logger.info(f"  {key}: {value}")
            
            _logger.info(f"{'='*70}")
            _logger.info("")


# Convenience functions for backward compatibility and simpler usage

def log_training_start(model_name: str = "DFM", config: Optional[Dict[str, Any]] = None) -> TrainLogger:
    """Create and start a training logger.
    
    Parameters
    ----------
    model_name : str, default "DFM"
        Name of the model being trained
    config : dict, optional
        Training configuration to log
        
    Returns
    -------
    TrainLogger
        Logger instance
    """
    logger = TrainLogger(model_name=model_name)
    logger.start(config=config)
    return logger


def log_training_step(
    logger: TrainLogger,
    iteration: int,
    loglik: Optional[float] = None,
    delta: Optional[float] = None,
    **kwargs
) -> None:
    """Log a training step.
    
    Parameters
    ----------
    logger : TrainLogger
        Training logger instance
    iteration : int
        Current iteration number
    loglik : float, optional
        Log-likelihood value
    delta : float, optional
        Convergence delta
    **kwargs
        Additional metrics to log
    """
    logger.log_iteration(iteration, loglik=loglik, delta=delta, **kwargs)


def log_training_end(
    logger: TrainLogger,
    success: bool = True,
    **kwargs
) -> None:
    """Log training end.
    
    Parameters
    ----------
    logger : TrainLogger
        Training logger instance
    success : bool, default True
        Whether training completed successfully
    **kwargs
        Additional information to log
    """
    logger.end(success=success, **kwargs)


def log_em_iteration(
    iteration: int,
    loglik: Optional[float] = None,
    delta: Optional[float] = None,
    **kwargs
) -> None:
    """Log EM algorithm iteration (convenience function).
    
    Parameters
    ----------
    iteration : int
        Current iteration number
    loglik : float, optional
        Log-likelihood value
    delta : float, optional
        Convergence delta
    **kwargs
        Additional metrics to log
    """
    if loglik is not None:
        msg = f"EM iteration {iteration:4d} | Log-likelihood: {loglik:12.6f}"
    else:
        msg = f"EM iteration {iteration:4d}"
    
    if delta is not None:
        msg += f" | Delta: {delta:10.6e}"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            msg += f" | {key}: {value:.6f}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


def log_convergence(
    converged: bool,
    num_iter: int,
    final_loglik: Optional[float] = None,
    reason: Optional[str] = None
) -> None:
    """Log convergence status (convenience function).
    
    Parameters
    ----------
    converged : bool
        Whether training converged
    num_iter : int
        Number of iterations completed
    final_loglik : float, optional
        Final log-likelihood value
    reason : str, optional
        Reason for stopping
    """
    if converged:
        _logger.info(f"✓ EM algorithm converged after {num_iter} iterations")
    else:
        _logger.warning(f"⚠ EM algorithm did not converge after {num_iter} iterations")
        if reason:
            _logger.warning(f"  Reason: {reason}")
    
    if final_loglik is not None:
        _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")


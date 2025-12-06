"""Training process logging utilities.

This module provides specialized logging for training processes,
including EM iterations (DFM), epochs (DDFM), convergence tracking, and training metrics.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import numpy as np
from datetime import datetime

from .logger import get_logger

_logger = get_logger(__name__)


class TrainLogger:
    """Logger for tracking training process for both DFM and DDFM.
    
    This class provides structured logging for training processes including:
    - Training start/end
    - EM iterations (DFM) or epochs (DDFM)
    - Convergence status
    - Training metrics (loss, log-likelihood, etc.)
    - Validation metrics (for DDFM)
    - Model-specific information
    
    Supports both DFM (EM algorithm) and DDFM (gradient descent) workflows.
    """
    
    def __init__(
        self, 
        model_name: str = "DFM", 
        model_type: str = "dfm",
        verbose: bool = True
    ):
        """Initialize training logger.
        
        Parameters
        ----------
        model_name : str, default "DFM"
            Name of the model being trained (e.g., "DFM", "DDFM")
        model_type : str, default "dfm"
            Type of model: "dfm" (EM algorithm) or "ddfm" (gradient descent)
        verbose : bool, default True
            Whether to log detailed information
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.verbose = verbose
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.iterations: int = 0
        self.epochs: int = 0
        self.converged: bool = False
        self.metrics_history: List[Dict[str, Any]] = []
        
    def start(
        self, 
        config: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log training start.
        
        Parameters
        ----------
        config : dict, optional
            Training configuration to log (e.g., max_iter, threshold, epochs, learning_rate)
        data_info : dict, optional
            Data information to log (e.g., shape, num_series, num_timepoints)
        """
        self.start_time = datetime.now()
        self.iterations = 0
        self.epochs = 0
        self.converged = False
        self.metrics_history = []
        
        _logger.info(f"{'='*70}")
        _logger.info(f"Starting {self.model_name} training ({self.model_type.upper()})")
        _logger.info(f"{'='*70}")
        
        if config and self.verbose:
            _logger.info("Training configuration:")
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, (list, tuple)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
        
        if data_info and self.verbose:
            _logger.info("Data information:")
            for key, value in data_info.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
        
        _logger.info("")
    
    def log_iteration(
        self,
        iteration: int,
        loglik: Optional[float] = None,
        delta: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log EM iteration information (for DFM).
        
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
        
        # Store metrics in history
        metrics = {"iteration": iteration, "loglik": loglik, "delta": delta}
        metrics.update(kwargs)
        self.metrics_history.append(metrics)
        
        if self.verbose:
            msg = f"EM Iteration {iteration:4d}"
            if loglik is not None:
                msg += f" | Log-likelihood: {loglik:12.6f}"
            if delta is not None:
                msg += f" | Delta: {delta:10.6e}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                elif isinstance(value, np.ndarray):
                    msg += f" | {key}: shape {value.shape}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
        else:
            # Less verbose: log every 10 iterations
            if iteration % 10 == 0 or iteration == 1:
                msg = f"EM Iteration {iteration:4d}"
                if loglik is not None:
                    msg += f" | Log-likelihood: {loglik:12.6f}"
                _logger.info(msg)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log training epoch information (for DDFM).
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        train_loss : float, optional
            Training loss value
        val_loss : float, optional
            Validation loss value
        learning_rate : float, optional
            Current learning rate
        **kwargs
            Additional metrics to log
        """
        self.epochs = epoch
        
        # Store metrics in history
        metrics = {
            "epoch": epoch, 
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "learning_rate": learning_rate
        }
        metrics.update(kwargs)
        self.metrics_history.append(metrics)
        
        if self.verbose:
            msg = f"Epoch {epoch:4d}"
            if train_loss is not None:
                msg += f" | Train Loss: {train_loss:10.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:10.6f}"
            if learning_rate is not None:
                msg += f" | LR: {learning_rate:.6e}"
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    msg += f" | {key}: {value:.6f}"
                elif isinstance(value, np.ndarray):
                    msg += f" | {key}: shape {value.shape}"
                else:
                    msg += f" | {key}: {value}"
            
            _logger.info(msg)
        else:
            # Less verbose: log every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                msg = f"Epoch {epoch:4d}"
                if train_loss is not None:
                    msg += f" | Train Loss: {train_loss:10.6f}"
                _logger.info(msg)
    
    def log_convergence(
        self,
        converged: bool,
        num_iter: Optional[int] = None,
        num_epochs: Optional[int] = None,
        final_loglik: Optional[float] = None,
        final_loss: Optional[float] = None,
        reason: Optional[str] = None
    ) -> None:
        """Log convergence status.
        
        Parameters
        ----------
        converged : bool
            Whether training converged
        num_iter : int, optional
            Number of EM iterations completed (for DFM)
        num_epochs : int, optional
            Number of epochs completed (for DDFM)
        final_loglik : float, optional
            Final log-likelihood value (for DFM)
        final_loss : float, optional
            Final training loss (for DDFM)
        reason : str, optional
            Reason for stopping (e.g., "converged", "max_iterations", "max_epochs", "early_stopping")
        """
        self.converged = converged
        if num_iter is not None:
            self.iterations = num_iter
        if num_epochs is not None:
            self.epochs = num_epochs
        
        _logger.info("")
        if self.model_type == "dfm":
            if converged:
                _logger.info(f"✓ Training converged after {num_iter or self.iterations} EM iterations")
            else:
                _logger.warning(f"⚠ Training did not converge after {num_iter or self.iterations} EM iterations")
        else:  # ddfm
            if converged:
                _logger.info(f"✓ Training converged after {num_epochs or self.epochs} epochs")
            else:
                _logger.warning(f"⚠ Training did not converge after {num_epochs or self.epochs} epochs")
        
        if reason:
            _logger.info(f"  Reason: {reason}")
        
        if final_loglik is not None:
            _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")
        if final_loss is not None:
            _logger.info(f"  Final training loss: {final_loss:.6f}")
    
    def end(self, success: bool = True, **kwargs) -> None:
        """Log training end.
        
        Parameters
        ----------
        success : bool, default True
            Whether training completed successfully
        **kwargs
            Additional information to log (e.g., metrics, warnings, model info)
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
            
            _logger.info(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            if self.model_type == "dfm":
                _logger.info(f"  EM Iterations: {self.iterations}")
            else:  # ddfm
                _logger.info(f"  Epochs: {self.epochs}")
            
            _logger.info(f"  Converged: {self.converged}")
            
            # Log summary metrics from history
            if self.metrics_history:
                if self.model_type == "dfm":
                    logliks = [m.get("loglik") for m in self.metrics_history if m.get("loglik") is not None]
                    if logliks:
                        _logger.info(f"  Initial log-likelihood: {logliks[0]:.6f}")
                        _logger.info(f"  Final log-likelihood: {logliks[-1]:.6f}")
                        if len(logliks) > 1:
                            improvement = logliks[-1] - logliks[0]
                            _logger.info(f"  Improvement: {improvement:.6f}")
                else:  # ddfm
                    losses = [m.get("train_loss") for m in self.metrics_history if m.get("train_loss") is not None]
                    if losses:
                        _logger.info(f"  Initial train loss: {losses[0]:.6f}")
                        _logger.info(f"  Final train loss: {losses[-1]:.6f}")
                        if len(losses) > 1:
                            improvement = losses[0] - losses[-1]  # Loss decreases
                            _logger.info(f"  Improvement: {improvement:.6f}")
            
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    _logger.info(f"  {key}: {value:.6f}")
                elif isinstance(value, np.ndarray):
                    _logger.info(f"  {key}: shape {value.shape}")
                else:
                    _logger.info(f"  {key}: {value}")
            
            _logger.info(f"{'='*70}")
            _logger.info("")


# Convenience functions for simpler usage

def log_training_start(
    model_name: str = "DFM",
    model_type: str = "dfm",
    config: Optional[Dict[str, Any]] = None,
    data_info: Optional[Dict[str, Any]] = None
) -> TrainLogger:
    """Create and start a training logger.
    
    Parameters
    ----------
    model_name : str, default "DFM"
        Name of the model being trained (e.g., "DFM", "DDFM")
    model_type : str, default "dfm"
        Type of model: "dfm" (EM algorithm) or "ddfm" (gradient descent)
    config : dict, optional
        Training configuration to log
    data_info : dict, optional
        Data information to log
        
    Returns
    -------
    TrainLogger
        Logger instance
    """
    logger = TrainLogger(model_name=model_name, model_type=model_type)
    logger.start(config=config, data_info=data_info)
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
    """Log EM algorithm iteration (convenience function for DFM).
    
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
        elif isinstance(value, np.ndarray):
            msg += f" | {key}: shape {value.shape}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


def log_training_epoch(
    epoch: int,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    learning_rate: Optional[float] = None,
    **kwargs
) -> None:
    """Log training epoch (convenience function for DDFM).
    
    Parameters
    ----------
    epoch : int
        Current epoch number
    train_loss : float, optional
        Training loss value
    val_loss : float, optional
        Validation loss value
    learning_rate : float, optional
        Current learning rate
    **kwargs
        Additional metrics to log
    """
    msg = f"Epoch {epoch:4d}"
    if train_loss is not None:
        msg += f" | Train Loss: {train_loss:10.6f}"
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:10.6f}"
    if learning_rate is not None:
        msg += f" | LR: {learning_rate:.6e}"
    
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            msg += f" | {key}: {value:.6f}"
        elif isinstance(value, np.ndarray):
            msg += f" | {key}: shape {value.shape}"
        else:
            msg += f" | {key}: {value}"
    
    _logger.info(msg)


def log_convergence(
    converged: bool,
    num_iter: Optional[int] = None,
    num_epochs: Optional[int] = None,
    final_loglik: Optional[float] = None,
    final_loss: Optional[float] = None,
    reason: Optional[str] = None,
    model_type: str = "dfm"
) -> None:
    """Log convergence status (convenience function).
    
    Parameters
    ----------
    converged : bool
        Whether training converged
    num_iter : int, optional
        Number of EM iterations completed (for DFM)
    num_epochs : int, optional
        Number of epochs completed (for DDFM)
    final_loglik : float, optional
        Final log-likelihood value (for DFM)
    final_loss : float, optional
        Final training loss (for DDFM)
    reason : str, optional
        Reason for stopping
    model_type : str, default "dfm"
        Type of model: "dfm" or "ddfm"
    """
    model_type = model_type.lower()
    
    if model_type == "dfm":
        if converged:
            _logger.info(f"✓ EM algorithm converged after {num_iter} iterations")
        else:
            _logger.warning(f"⚠ EM algorithm did not converge after {num_iter} iterations")
    else:  # ddfm
        if converged:
            _logger.info(f"✓ Training converged after {num_epochs} epochs")
        else:
            _logger.warning(f"⚠ Training did not converge after {num_epochs} epochs")
    
    if reason:
        _logger.info(f"  Reason: {reason}")
    
    if final_loglik is not None:
        _logger.info(f"  Final log-likelihood: {final_loglik:.6f}")
    if final_loss is not None:
        _logger.info(f"  Final training loss: {final_loss:.6f}")


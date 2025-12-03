"""PyTorch Lightning Trainer for Deep Dynamic Factor Model (DDFM).

This module provides DDFMTrainer, a specialized Trainer class for DDFM models
with sensible defaults for neural network training.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing import Optional, Dict, Any, List
from ..logger import get_logger

_logger = get_logger(__name__)


class DDFMTrainer(pl.Trainer):
    """Specialized PyTorch Lightning Trainer for DDFM models.
    
    This trainer provides sensible defaults for training DDFM models using
    neural networks (autoencoders). It includes appropriate callbacks and
    logging for deep learning training.
    
    Parameters
    ----------
    max_epochs : int, default 100
        Maximum number of training epochs
    enable_progress_bar : bool, default True
        Whether to show progress bar during training
    enable_model_summary : bool, default True
        Whether to print model summary (useful for debugging DDFM architecture)
    logger : bool or Logger, default True
        Whether to use a logger. Can be False, True (uses TensorBoardLogger), or a Logger instance
    callbacks : List[Callback], optional
        Additional callbacks beyond defaults
    accelerator : str, default 'auto'
        Accelerator type ('cpu', 'gpu', 'auto', etc.)
    devices : int or List[int], default 'auto'
        Device configuration
    precision : str or int, default 32
        Training precision (16, 32, 'bf16', etc.)
    gradient_clip_val : float, optional
        Gradient clipping value (useful for training stability)
    accumulate_grad_batches : int, default 1
        Number of batches to accumulate gradients before optimizer step
    **kwargs
        Additional arguments passed to pl.Trainer
    
    Examples
    --------
    >>> from dfm_python.trainer import DDFMTrainer
    >>> from dfm_python.lightning import DDFMLightningModule
    >>> 
    >>> trainer = DDFMTrainer(max_epochs=100, enable_progress_bar=True)
    >>> trainer.fit(module, dataloader)
    """
    
    def __init__(
            self,
            max_epochs: int = 100,
            enable_progress_bar: bool = True,
            enable_model_summary: bool = True,
            logger: Optional[Any] = True,
            callbacks: Optional[List[Any]] = None,
            accelerator: str = 'auto',
            devices: Any = 'auto',
            precision: Any = 32,
            gradient_clip_val: Optional[float] = None,
            accumulate_grad_batches: int = 1,
            **kwargs
    ):
        # Build callbacks list
        trainer_callbacks = callbacks if callbacks is not None else []
        
        # Add early stopping if max_epochs is set and not already in callbacks
        if max_epochs > 0 and not any(isinstance(cb, EarlyStopping) for cb in trainer_callbacks):
            early_stopping = EarlyStopping(
                monitor='train_loss',
                patience=20,  # More patience for neural network training
                mode='min',
                verbose=True,
                min_delta=1e-6
            )
            trainer_callbacks.append(early_stopping)
        
        # Add learning rate monitor for neural network training
        if not any(isinstance(cb, LearningRateMonitor) for cb in trainer_callbacks):
            lr_monitor = LearningRateMonitor(logging_interval='step')
            trainer_callbacks.append(lr_monitor)
        
        # Add model checkpoint callback
        if not any(isinstance(cb, ModelCheckpoint) for cb in trainer_callbacks):
            checkpoint = ModelCheckpoint(
                monitor='train_loss',
                mode='min',
                save_top_k=1,
                filename='ddfm-{epoch:02d}-{train_loss:.4f}'
            )
            trainer_callbacks.append(checkpoint)
        
        # Setup logger
        if logger is True:
            # Use TensorBoardLogger as default for DDFM (better for neural networks)
            try:
                logger = TensorBoardLogger(save_dir='lightning_logs', name='ddfm')
            except Exception:
                # Fallback to CSVLogger if TensorBoard not available
                _logger.warning("TensorBoard not available, using CSVLogger")
                logger = CSVLogger(save_dir='lightning_logs', name='ddfm')
        elif logger is False:
            logger = None
        
        # Call parent constructor with DDFM-specific defaults
        super().__init__(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            logger=logger,
            callbacks=trainer_callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            **kwargs
        )
    
    @classmethod
    def from_config(
        cls,
        config: Any,  # DDFMConfig or DFMConfig with DDFM params
        **kwargs
    ) -> 'DDFMTrainer':
        """Create DDFMTrainer from DDFMConfig or DFMConfig.
        
        Extracts training parameters from config and creates trainer with
        appropriate settings for neural network training.
        
        Parameters
        ----------
        config : DDFMConfig or DFMConfig
            Configuration object (can be DDFMConfig or DFMConfig with DDFM parameters)
        **kwargs
            Additional arguments to override config values
            
        Returns
        -------
        DDFMTrainer
            Configured trainer instance
        """
        # Extract training parameters from config
        # Handle both DDFMConfig and DFMConfig with ddfm_* parameters
        if hasattr(config, 'epochs'):
            max_epochs = kwargs.pop('max_epochs', config.epochs)
        elif hasattr(config, 'ddfm_epochs'):
            max_epochs = kwargs.pop('max_epochs', config.ddfm_epochs)
        else:
            max_epochs = kwargs.pop('max_epochs', 100)
        
        enable_progress_bar = kwargs.pop('enable_progress_bar', True)
        enable_model_summary = kwargs.pop('enable_model_summary', True)
        
        # Extract gradient clipping if available
        gradient_clip_val = kwargs.pop('gradient_clip_val', None)
        if gradient_clip_val is None and hasattr(config, 'gradient_clip_val'):
            gradient_clip_val = config.gradient_clip_val
        
        return cls(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            gradient_clip_val=gradient_clip_val,
            **kwargs
        )


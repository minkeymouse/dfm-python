"""PyTorch Lightning Trainer for Kernelized Dynamic Factor Model (KDFM).

This module provides KDFMTrainer, a specialized Trainer class for KDFM models
with sensible defaults for gradient descent training.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing import Optional, Dict, Any, List, Union
from ..logger import get_logger
from . import (
    _create_base,
    _extract_train_params,
    _validate_config,
    DDFM_TRAINER_DEFAULTS
)

_logger = get_logger(__name__)


class KDFMTrainer(pl.Trainer):
    """Specialized PyTorch Lightning Trainer for KDFM models.
    
    This trainer provides sensible defaults for training KDFM models using
    gradient descent. KDFM training uses standard gradient-based optimization
    similar to DDFM.
    
    Default Values:
        - max_epochs: 100 (training epochs)
        - enable_progress_bar: True
        - enable_model_summary: True (useful for debugging KDFM architecture)
        - logger: True (uses TensorBoardLogger with CSVLogger fallback)
        - accelerator: 'auto'
        - devices: 'auto'
        - precision: 32
        - gradient_clip_val: 1.0 (default, for numerical stability)
        - accumulate_grad_batches: 1
    
    These defaults are optimized for KDFM neural network training. The trainer
    automatically sets up early stopping (patience=20), learning rate monitor,
    and model checkpoint callbacks.
    
    Parameters
    ----------
    max_epochs : int, default 100
        Maximum number of training epochs
    enable_progress_bar : bool, default True
        Whether to show progress bar during training
    enable_model_summary : bool, default True
        Whether to print model summary (useful for debugging KDFM architecture)
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
    gradient_clip_val : float, optional, default 1.0
        Gradient clipping value for numerical stability. Default 1.0 helps prevent
        gradient explosion that can cause NaN values during training.
    accumulate_grad_batches : int, default 1
        Number of batches to accumulate gradients before optimizer step
    **kwargs
        Additional arguments passed to pl.Trainer
    
    Examples
    --------
    >>> from dfm_python.trainer import KDFMTrainer
    >>> from dfm_python import KDFM, KDFMDataModule
    >>> 
    >>> model = KDFM(ar_order=1, ma_order=0)
    >>> dm = KDFMDataModule(config_path='config.yaml', data=df)
    >>> trainer = KDFMTrainer(max_epochs=100, enable_progress_bar=True)
    >>> trainer.fit(model, dm)
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
            gradient_clip_val: Optional[float] = 1.0,  # Default: 1.0 for numerical stability
            accumulate_grad_batches: int = 1,
            **kwargs
    ):
        # Setup KDFM-specific callbacks (learning rate monitor and checkpoint)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            filename='kdfm-{epoch:02d}-{train_loss:.4f}'
        )
        
        # Use common trainer base setup with KDFM-specific parameters
        # KDFM logs 'train_loss' metric, uses TensorBoard logger, and has patience=20
        trainer_config = _create_base(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            logger=logger,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            early_stopping_patience=20,
            early_stopping_min_delta=None,
            early_stopping_monitor='train_loss',  # KDFM uses 'train_loss' metric
            logger_type='tensorboard',  # Use TensorBoard logger like DDFM
            additional_callbacks=[lr_monitor, checkpoint]
        )
        
        # Add gradient clipping if specified
        if gradient_clip_val is not None and gradient_clip_val > 0:
            trainer_config['gradient_clip_val'] = gradient_clip_val
        
        # Add gradient accumulation
        trainer_config['accumulate_grad_batches'] = accumulate_grad_batches
        
        # Store attributes for testing/verification
        # Note: These are stored as instance attributes to allow tests to verify
        # default values. The parent Trainer class also stores these, but storing
        # them here ensures they're accessible even if parent implementation changes.
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        
        super().__init__(**trainer_config, **kwargs)


"""PyTorch Lightning Trainer for Deep Dynamic Factor Model (DDFM).

This module provides:
- DDFMTrainer: A specialized PyTorch Lightning Trainer class for DDFM models

DDFM training uses PyTorch Lightning's standard training loop with vectorized MC sampling
for efficient training. The recommended pattern is:

    from dfm_python import DDFM, DDFMDataModule, DDFMTrainer
    
    model = DDFM(encoder_layers=[64, 32], num_factors=2)
    model.load_config(config)
    datamodule = DDFMDataModule(config=config, data=data)
    datamodule.setup()
    trainer = DDFMTrainer(max_epochs=100)
    trainer.fit(model, datamodule)

The legacy DDFMDenoisingTrainer has been removed as its functionality is now integrated
into the DDFM LightningModule.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np
import torch

from ..logger import get_logger
from ..config import DFMConfig, DDFMConfig
from ..config.constants import DEFAULT_GRAD_CLIP_VAL, DEFAULT_MIN_DELTA, DEFAULT_MAX_EPOCHS
from . import (
    _create_base,
    _extract_train_params,
    _validate_config,
    DDFM_TRAINER_DEFAULTS
)

if TYPE_CHECKING:
    from ..models.ddfm import DDFM, DDFMTrainingState

_logger = get_logger(__name__)


class DDFMTrainer(pl.Trainer):
    """Specialized PyTorch Lightning Trainer for DDFM models.
    
    This trainer provides sensible defaults for training DDFM models using
    neural networks (autoencoders). It includes appropriate callbacks and
    logging for deep learning training.
    
    Default Values:
        - max_epochs: 100 (training epochs)
        - enable_progress_bar: True
        - enable_model_summary: True (useful for debugging DDFM architecture)
        - logger: True (uses TensorBoardLogger with CSVLogger fallback)
        - accelerator: 'auto'
        - devices: 'auto'
        - precision: 32
        - gradient_clip_val: Not used (DDFM handles gradient clipping manually in training_step)
        - accumulate_grad_batches: 1
    
    These defaults are optimized for DDFM neural network training. The trainer
    automatically sets up early stopping (patience=20), learning rate monitor,
    and model checkpoint callbacks.
    
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
    gradient_clip_val : float, optional, default None
        Not used - DDFM uses manual optimization and handles gradient clipping in training_step().
        Gradient clipping is controlled by model.grad_clip_val parameter.
    accumulate_grad_batches : int, default 1
        Number of batches to accumulate gradients before optimizer step
    **kwargs
        Additional arguments passed to pl.Trainer
    
    Examples
    --------
    >>> from dfm_python.trainer import DDFMTrainer
    >>> from dfm_python import DDFM, DDFMDataModule
    >>> 
    >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> dm = DDFMDataModule(config_path='config.yaml', data=df)
    >>> trainer = DDFMTrainer(max_epochs=100, enable_progress_bar=True)  # max_epochs=DEFAULT_MAX_EPOCHS
    >>> trainer.fit(model, dm)
    """
    
    def __init__(
            self,
            max_epochs: int = DEFAULT_MAX_EPOCHS,
            enable_progress_bar: bool = True,
            enable_model_summary: bool = True,
            logger: Optional[Any] = True,
            callbacks: Optional[List[Any]] = None,
            accelerator: str = 'auto',
            devices: Any = 'auto',
            precision: Any = 32,
            gradient_clip_val: Optional[float] = None,  # Not used - DDFM handles gradient clipping manually in training_step
            accumulate_grad_batches: int = 1,
            **kwargs
    ):
        # Setup DDFM-specific callbacks (learning rate monitor and checkpoint)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint = ModelCheckpoint(
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            filename='ddfm-{epoch:02d}-{train_loss:.4f}'
        )
        
        # Use common trainer base setup with DDFM-specific parameters
        # DDFM uses 'train_loss' metric, TensorBoard logger, patience=20, and additional callbacks
        # Note: gradient_clip_val is NOT passed to trainer - DDFM uses manual optimization
        # and handles gradient clipping in training_step() manually
        trainer_config = _create_base(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            logger=logger,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            early_stopping_patience=20,  # More patience for neural network training
            early_stopping_min_delta=DEFAULT_MIN_DELTA,  # Minimum change for improvement
            early_stopping_monitor='train_loss',  # DDFM uses 'train_loss' metric
            logger_type='tensorboard',  # DDFM uses TensorBoard logger
            logger_name='ddfm',
            additional_callbacks=[lr_monitor, checkpoint],  # DDFM-specific callbacks
            # gradient_clip_val is NOT passed - DDFM handles it manually in training_step
            accumulate_grad_batches=accumulate_grad_batches,  # DDFM-specific parameter
            **kwargs
        )
        
        # Store attributes for testing/verification
        # Note: These are stored as instance attributes to allow tests to verify
        # default values. The parent Trainer class also stores these, but storing
        # them here ensures they're accessible even if parent implementation changes.
        self.enable_progress_bar = enable_progress_bar
        self.enable_model_summary = enable_model_summary
        
        # Call parent constructor with configured parameters
        super().__init__(**trainer_config)
    
    @classmethod
    def from_config(
        cls,
        config: Union[DFMConfig, DDFMConfig],
        **kwargs
    ) -> 'DDFMTrainer':
        """Create DDFMTrainer from DDFMConfig or DFMConfig.
        
        Extracts training parameters from config and creates trainer with
        appropriate settings for neural network training. Parameters can be overridden via kwargs.
        
        Parameters
        ----------
        config : Union[DFMConfig, DDFMConfig]
            Configuration object (can be DDFMConfig or DDFMConfig with DDFM parameters)
        **kwargs
            Additional arguments to override config values.
            Supported parameters: max_epochs, enable_progress_bar, enable_model_summary.
            Note: gradient_clip_val is not used - DDFM handles gradient clipping manually.
            For additional Trainer parameters, use __init__() directly.
            
        Returns
        -------
        DDFMTrainer
            Configured trainer instance
        """
        # Validate config before processing
        _validate_config(config, trainer_name="DDFMTrainer")
        
        # Extract training parameters from config and kwargs
        # Handle both DDFMConfig and DFMConfig with ddfm_* parameters
        # Note: Don't use max_iter for DDFM (only epochs/ddfm_epochs)
        # Use constants from trainer/__init__.py to ensure single source of truth
        # These defaults match __init__() defaults for consistency
        # Note: _extract_train_params() modifies kwargs by popping extracted keys
        # After extraction, only extracted parameters are used (kwargs are consumed)
        params = _extract_train_params(config, kwargs, DDFM_TRAINER_DEFAULTS, use_max_iter=False)
        
        # Create trainer with extracted parameters
        # All relevant parameters are extracted, so kwargs are not passed through
        # If additional Trainer parameters are needed, use __init__() directly
        return cls(**params)


# DDFMDenoisingTrainer removed - functionality now integrated into DDFM LightningModule
# The legacy sequential Python for-loop implementation has been replaced with vectorized MC sampling
# in DDFMMCDataset and Lightning's training_step/on_train_epoch_end hooks for better performance.


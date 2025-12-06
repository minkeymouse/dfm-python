"""PyTorch Lightning Trainer for Linear Dynamic Factor Model (DFM).

This module provides DFMTrainer, a specialized Trainer class for DFM models
with sensible defaults for EM algorithm training.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from typing import Optional, Dict, Any, List, Union
from ..logger import get_logger
from ..config import DFMConfig, DDFMConfig
from . import (
    _create_base,
    _extract_train_params,
    _validate_config,
    DFM_TRAINER_DEFAULTS
)

_logger = get_logger(__name__)


class DFMTrainer(pl.Trainer):
    """Specialized PyTorch Lightning Trainer for DFM models.
    
    This trainer provides sensible defaults for training DFM models using
    the EM algorithm. DFM training typically doesn't use standard gradient-based
    optimization, but this trainer can be used for consistency with Lightning
    workflows or for any gradient-based components.
    
    Default Values:
        - max_epochs: 100 (EM iterations)
        - enable_progress_bar: True
        - enable_model_summary: False (DFM modules are simple, usually not needed)
        - logger: True (uses CSVLogger, creates lightning_logs/dfm/ folder)
        - accelerator: 'auto'
        - devices: 'auto'
        - precision: 32
    
    These defaults are optimized for DFM training with EM algorithm. The trainer
    automatically sets up early stopping callback monitoring 'loglik' metric.
    
    Parameters
    ----------
    max_epochs : int, default 100
        Maximum number of EM iterations/epochs
    enable_progress_bar : bool, default True
        Whether to show progress bar during training
    enable_model_summary : bool, default False
        Whether to print model summary (DFM modules are simple, usually not needed)
    logger : bool or Logger, default True
        Whether to use a logger. Can be False, True (uses CSVLogger), or a Logger instance
    callbacks : List[Callback], optional
        Additional callbacks beyond defaults
    accelerator : str, default 'auto'
        Accelerator type ('cpu', 'gpu', 'auto', etc.)
    devices : int or List[int], default 'auto'
        Device configuration
    precision : str or int, default 32
        Training precision (16, 32, 'bf16', etc.)
    **kwargs
        Additional arguments passed to pl.Trainer
    
    Examples
    --------
    >>> from dfm_python.trainer import DFMTrainer
    >>> from dfm_python import DFM, DFMDataModule
    >>> 
    >>> model = DFM()
    >>> dm = DFMDataModule(config_path='config.yaml', data=df)
    >>> trainer = DFMTrainer(max_epochs=50, enable_progress_bar=True)
    >>> trainer.fit(model, dm)
    """
    
    def __init__(
            self,
            max_epochs: int = 100,
            enable_progress_bar: bool = True,
            enable_model_summary: bool = False,
            logger: Optional[Any] = True,
            callbacks: Optional[List[Any]] = None,
            accelerator: str = 'auto',
            devices: Any = 'auto',
            precision: Any = 32,
            **kwargs
    ):
        # Use common trainer base setup with DFM-specific parameters
        # DFM logs 'loglik' metric, uses CSV logger, and has patience=10
        trainer_config = _create_base(
            max_epochs=max_epochs,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
            logger=logger,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            early_stopping_patience=10,
            early_stopping_min_delta=None,
            early_stopping_monitor='loglik',  # DFM uses 'loglik' metric
            logger_type='csv',  # DFM uses CSV logger
            logger_name='dfm',
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
    ) -> 'DFMTrainer':
        """Create DFMTrainer from DFMConfig or DDFMConfig.
        
        Extracts training parameters from config and creates trainer with
        appropriate settings. Parameters can be overridden via kwargs.
        
        Parameters
        ----------
        config : Union[DFMConfig, DDFMConfig]
            DFM or DDFM configuration object
        **kwargs
            Additional arguments to override config values.
            Supported parameters: max_epochs, enable_progress_bar, enable_model_summary.
            For additional Trainer parameters, use __init__() directly.
            
        Returns
        -------
        DFMTrainer
            Configured trainer instance
        """
        # Validate config before processing
        _validate_config(config, trainer_name="DFMTrainer")
        
        # Extract training parameters from config and kwargs
        # Use constants from trainer/__init__.py to ensure single source of truth
        # These defaults match __init__() defaults for consistency
        # Note: _extract_train_params() modifies kwargs by popping extracted keys
        # After extraction, only extracted parameters are used (kwargs are consumed)
        params = _extract_train_params(config, kwargs, DFM_TRAINER_DEFAULTS, use_max_iter=True)
        
        # Create trainer with extracted parameters
        # All relevant parameters are extracted, so kwargs are not passed through
        # If additional Trainer parameters are needed, use __init__() directly
        return cls(**params)


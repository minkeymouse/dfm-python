"""PyTorch Lightning Trainer for Linear Dynamic Factor Model (DFM).

This module provides DFMTrainer, a specialized Trainer class for DFM models
with sensible defaults for EM algorithm training.
"""

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    _has_lightning = True
except ImportError:
    _has_lightning = False
    pl = None
    EarlyStopping = None
    ModelCheckpoint = None
    CSVLogger = None

from typing import Optional, Dict, Any, List
from ..logger import get_logger

_logger = get_logger(__name__)


if _has_lightning:
    class DFMTrainer(pl.Trainer):
        """Specialized PyTorch Lightning Trainer for DFM models.
        
        This trainer provides sensible defaults for training DFM models using
        the EM algorithm. DFM training typically doesn't use standard gradient-based
        optimization, but this trainer can be used for consistency with Lightning
        workflows or for any gradient-based components.
        
        Parameters
        ----------
        max_epochs : int, default 100
            Maximum number of EM iterations/epochs
        enable_progress_bar : bool, default True
            Whether to show progress bar during training
        enable_model_summary : bool, default False
            Whether to print model summary (DFM modules are simple, usually not needed)
        logger : bool or Logger, default False
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
        >>> from dfm_python.lightning import DFMLightningModule
        >>> 
        >>> trainer = DFMTrainer(max_epochs=50, enable_progress_bar=True)
        >>> trainer.fit(module, dataloader)
        """
        
        def __init__(
            self,
            max_epochs: int = 100,
            enable_progress_bar: bool = True,
            enable_model_summary: bool = False,
            logger: Optional[Any] = False,
            callbacks: Optional[List[Any]] = None,
            accelerator: str = 'auto',
            devices: Any = 'auto',
            precision: Any = 32,
            **kwargs
        ):
            # Build callbacks list
            trainer_callbacks = callbacks if callbacks is not None else []
            
            # Add early stopping if max_epochs is set and not already in callbacks
            if max_epochs > 0 and not any(isinstance(cb, EarlyStopping) for cb in trainer_callbacks):
                early_stopping = EarlyStopping(
                    monitor='train_loss',
                    patience=10,
                    mode='min',
                    verbose=True
                )
                trainer_callbacks.append(early_stopping)
            
            # Setup logger
            if logger is True:
                # Use CSVLogger as default
                logger = CSVLogger(save_dir='lightning_logs', name='dfm')
            elif logger is False:
                logger = None
            
            # Call parent constructor with DFM-specific defaults
            super().__init__(
                max_epochs=max_epochs,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=enable_model_summary,
                logger=logger,
                callbacks=trainer_callbacks,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                **kwargs
            )
        
        @classmethod
        def from_config(
            cls,
            config: Any,  # DFMConfig
            **kwargs
        ) -> 'DFMTrainer':
            """Create DFMTrainer from DFMConfig.
            
            Extracts training parameters from config and creates trainer with
            appropriate settings.
            
            Parameters
            ----------
            config : DFMConfig
                DFM configuration object
            **kwargs
                Additional arguments to override config values
                
            Returns
            -------
            DFMTrainer
                Configured trainer instance
            """
            # Extract training parameters from config
            max_epochs = kwargs.pop('max_epochs', getattr(config, 'max_iter', 100))
            enable_progress_bar = kwargs.pop('enable_progress_bar', True)
            enable_model_summary = kwargs.pop('enable_model_summary', False)
            
            return cls(
                max_epochs=max_epochs,
                enable_progress_bar=enable_progress_bar,
                enable_model_summary=enable_model_summary,
                **kwargs
            )
else:
    # Placeholder class when Lightning is not available
    class DFMTrainer:
        """Placeholder for DFMTrainer when PyTorch Lightning is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DFMTrainer requires PyTorch Lightning. "
                "Install with: pip install pytorch-lightning"
            )


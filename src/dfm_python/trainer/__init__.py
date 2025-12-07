"""Trainer classes for Dynamic Factor Models.

This package provides specialized PyTorch Lightning Trainer classes
for DFM and DDFM models with model-specific defaults and configurations.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from typing import Optional, List, Any, Dict, Union, Tuple
from ..logger import get_logger

_logger = get_logger(__name__)


# ============================================================================
# Trainer Default Values (Single Source of Truth)
# ============================================================================

# DFM Trainer Defaults
DFM_TRAINER_DEFAULTS = {
    'max_epochs': 100,
    'enable_progress_bar': True,
    'enable_model_summary': False
}

# DDFM Trainer Defaults
DDFM_TRAINER_DEFAULTS = {
    'max_epochs': 100,
    'enable_progress_bar': True,
    'enable_model_summary': True
}


# ============================================================================
# Helper Functions for Shared Trainer Logic
# ============================================================================

def _setup_early_stopping(
    max_epochs: int,
    patience: int = 10,
    min_delta: Optional[float] = None,
    monitor: str = 'train_loss',
    mode: str = 'min'
) -> Optional[EarlyStopping]:
    """Create EarlyStopping callback if max_epochs > 0.
    
    Parameters
    ----------
    max_epochs : int
        Maximum number of epochs. If <= 0, returns None.
    patience : int, default 10
        Number of epochs to wait before stopping (10 for DFM, 20 for DDFM)
    min_delta : float, optional
        Minimum change to qualify as improvement (None for DFM, 1e-6 for DDFM)
    monitor : str, default 'train_loss'
        Metric to monitor for early stopping. Use 'loglik' for DFM, 'train_loss' for DDFM.
    mode : str, default 'min'
        Whether to minimize ('min') or maximize ('max') the monitored metric.
        Use 'max' for log-likelihood (DFM), 'min' for loss (DDFM).
        
    Returns
    -------
    Optional[EarlyStopping]
        EarlyStopping callback if max_epochs > 0, None otherwise
    """
    if max_epochs <= 0:
        return None
    
    kwargs = {
        'monitor': monitor,
        'patience': patience,
        'mode': mode,
        'verbose': True
    }
    if min_delta is not None:
        kwargs['min_delta'] = min_delta
    
    return EarlyStopping(**kwargs)


def _setup_logger(
    logger: Any,
    logger_type: str = 'csv',
    name: str = 'trainer'
) -> Optional[pl.loggers.Logger]:
    """Setup logger based on logger parameter and type.
    
    Parameters
    ----------
    logger : bool, Logger, or None
        - True: Create logger of specified type
        - False: Return None
        - Logger instance: Return as-is
    logger_type : str, default 'csv'
        Logger type: 'csv' for CSVLogger, 'tensorboard' for TensorBoardLogger
    name : str, default 'trainer'
        Logger name (used in save_dir path)
        
    Returns
    -------
    Optional[Logger]
        Configured logger instance or None
    """
    if logger is False:
        return None
    elif logger is True:
        if logger_type == 'tensorboard':
            try:
                return TensorBoardLogger(save_dir='lightning_logs', name=name)
            except Exception:
                _logger.warning(
                    "Trainer logger setup failed: TensorBoard not available, using CSVLogger. "
                    "Please install tensorboard if you need TensorBoard logging."
                )
                return CSVLogger(save_dir='lightning_logs', name=name)
        else:  # csv
            return CSVLogger(save_dir='lightning_logs', name=name)
    else:
        # logger is already a Logger instance
        return logger


def _build_callbacks(
    callbacks: Optional[List[Any]],
    early_stopping: Optional[EarlyStopping] = None,
    lr_monitor: Optional[LearningRateMonitor] = None,
    checkpoint: Optional[ModelCheckpoint] = None
) -> List[Any]:
    """Build callback list from provided callbacks and optional defaults.
    
    Checks for duplicates before adding default callbacks to avoid conflicts.
    
    Parameters
    ----------
    callbacks : Optional[List[Callback]]
        Existing callbacks list (None becomes empty list)
    early_stopping : Optional[EarlyStopping], default None
        EarlyStopping callback to add if not already present
    lr_monitor : Optional[LearningRateMonitor], default None
        LearningRateMonitor callback to add if not already present
    checkpoint : Optional[ModelCheckpoint], default None
        ModelCheckpoint callback to add if not already present
        
    Returns
    -------
    List[Callback]
        Combined callback list with defaults added (if not duplicates)
    """
    trainer_callbacks = callbacks if callbacks is not None else []
    
    # Add early stopping if provided and not already in callbacks
    if early_stopping is not None:
        if not any(isinstance(cb, EarlyStopping) for cb in trainer_callbacks):
            trainer_callbacks.append(early_stopping)
    
    # Add learning rate monitor if provided and not already in callbacks
    if lr_monitor is not None:
        if not any(isinstance(cb, LearningRateMonitor) for cb in trainer_callbacks):
            trainer_callbacks.append(lr_monitor)
    
    # Add checkpoint if provided and not already in callbacks
    if checkpoint is not None:
        if not any(isinstance(cb, ModelCheckpoint) for cb in trainer_callbacks):
            trainer_callbacks.append(checkpoint)
    
    return trainer_callbacks


# ============================================================================
# Device/Precision Normalization Helper Functions
# ============================================================================

def _normalize_accel(accelerator: Any) -> str:
    """Normalize accelerator value for consistent handling.
    
    PyTorch Lightning may normalize accelerator values (e.g., 'cpu' to 'CPU').
    This function normalizes common accelerator values to lowercase strings
    for consistent comparison and validation.
    
    Handles both input values (before passing to Lightning) and Lightning-normalized
    values (after Lightning has processed them). This ensures consistent comparison
    regardless of when normalization occurs.
    
    Parameters
    ----------
    accelerator : Any
        Accelerator value (str, int, or accelerator object)
        Can be input value or Lightning-normalized value
        Common values: 'cpu', 'gpu', 'cuda', 'auto', 'mps', 'CPU', 'GPU', etc.
        Can also be accelerator objects like CPUAccelerator, GPUAccelerator, etc.
        
    Returns
    -------
    str
        Normalized accelerator value (lowercase string)
        
    Examples
    --------
    >>> _normalize_accel('cpu')
    'cpu'
    >>> _normalize_accelerator('CPU')
    'cpu'
    >>> _normalize_accelerator('auto')
    'auto'
    >>> # Handles Lightning-normalized values
    >>> _normalize_accelerator('CPU')  # Lightning might store as uppercase
    'cpu'
    >>> # Handles accelerator objects
    >>> _normalize_accelerator(CPUAccelerator())  # Lightning accelerator object
    'cpu'
    """
    if accelerator is None:
        return 'auto'
    
    # Handle accelerator objects (PyTorch Lightning converts strings to objects)
    if hasattr(accelerator, '__class__'):
        class_name = accelerator.__class__.__name__.lower()
        # Extract accelerator type from class name (e.g., 'CPUAccelerator' -> 'cpu')
        if 'cpu' in class_name:
            return 'cpu'
        elif 'gpu' in class_name or 'cuda' in class_name:
            return 'gpu' if 'gpu' in class_name else 'cuda'
        elif 'mps' in class_name:
            return 'mps'
        elif 'tpu' in class_name:
            return 'tpu'
        # If it's an accelerator object but we can't identify type, try str() method
        if hasattr(accelerator, '__str__'):
            accelerator_str = str(accelerator).lower().strip()
            if accelerator_str in ('cpu', 'gpu', 'cuda', 'mps', 'auto', 'tpu'):
                return accelerator_str
    
    # Convert to string and lowercase for normalization
    # This handles both input values and Lightning-normalized values
    accelerator_str = str(accelerator).lower().strip()
    
    # Handle common variations (case-insensitive)
    if accelerator_str in ('cpu', 'gpu', 'cuda', 'mps', 'auto', 'tpu'):
        return accelerator_str
    
    # Return normalized string (Lightning will handle validation)
    return accelerator_str


def _normalize_prec(precision: Any) -> Union[str, int]:
    """Normalize precision value for consistent handling.
    
    PyTorch Lightning may normalize precision values (e.g., 32 to '32' or '32-true').
    This function normalizes precision values to a consistent format for
    comparison and validation.
    
    Handles both input values (before passing to Lightning) and Lightning-normalized
    values (after Lightning has processed them). For simple numeric precisions (16, 32),
    returns int for consistency. For complex formats, returns string.
    
    Parameters
    ----------
    precision : Any
        Precision value (int, str, or other type)
        Can be input value or Lightning-normalized value
        Common values: 16, 32, '16', '32', 'bf16', '16-mixed', '32-true', etc.
        
    Returns
    -------
    Union[str, int]
        Normalized precision value (int for simple values like 16, 32; str for complex)
        
    Examples
    --------
    >>> _normalize_prec(32)
    32
    >>> _normalize_precision('32')
    32
    >>> _normalize_precision('32-true')  # Lightning might store as '32-true'
    32  # Extracts numeric part for simple comparison
    >>> _normalize_precision('bf16')
    'bf16'
    >>> _normalize_precision('16-mixed')
    '16-mixed'
    """
    if precision is None:
        return 32  # Default precision
    
    # Handle integer precision
    if isinstance(precision, int):
        if precision in (16, 32):
            return precision
        else:
            return str(precision)
    
    # Handle string precision
    if isinstance(precision, str):
        precision_str = precision.strip().lower()
        
        # Handle simple numeric strings
        if precision_str == '16':
            return 16
        elif precision_str == '32':
            return 32
        # Handle Lightning formats like '32-true', '32-false', etc.
        # Extract numeric part for simple comparison (32-true -> 32)
        elif precision_str.startswith('32-'):
            return 32
        elif precision_str.startswith('16-'):
            # Check if it's a simple format (16-true, 16-false) or complex (16-mixed)
            if 'mixed' in precision_str:
                # Keep '16-mixed' as string (complex format)
                return precision
            else:
                # Simple format like '16-true' -> 16
                return 16
        elif precision_str in ('bf16', 'bfloat16'):
            return 'bf16'
        elif precision_str in ('fp16', 'float16'):
            return 16
        elif precision_str in ('fp32', 'float32'):
            return 32
        else:
            # Return as-is for other complex formats
            return precision
    
    # For other types, convert to string
    return str(precision)


def _validate_device(
    accelerator: str,
    devices: Any
) -> Tuple[str, Any]:
    """Validate and normalize device configuration.
    
    Validates accelerator and devices configuration for compatibility.
    Provides clear error messages for invalid configurations.
    
    Parameters
    ----------
    accelerator : str
        Accelerator type (normalized, e.g., 'cpu', 'gpu', 'cuda', 'auto')
    devices : Any
        Device configuration (int, list, str, 'auto', etc.)
        
    Returns
    -------
    tuple[str, Any]
        Normalized (accelerator, devices) tuple
        
    Raises
    ------
    ValueError
        If accelerator and devices are incompatible
        
    Examples
    --------
    >>> _validate_device('cpu', 1)
    ('cpu', 1)
    >>> _validate_device_config('gpu', [0, 1])
    ('gpu', [0, 1])
    >>> _validate_device_config('auto', 'auto')
    ('auto', 'auto')
    """
    # Normalize accelerator
    accelerator = _normalize_accel(accelerator)
    
    # Validate accelerator-device compatibility
    if accelerator == 'cpu':
        # CPU doesn't need specific device numbers, but Lightning accepts them
        if devices not in ('auto', 1, [1], None):
            _logger.warning(
                f"Trainer device configuration warning: CPU accelerator with devices={devices} may be normalized by Lightning. "
                f"Please consider using devices=1 or devices='auto' for CPU to avoid potential issues."
            )
    elif accelerator in ('gpu', 'cuda'):
        # GPU/CUDA should have valid device configuration
        if devices == 'auto':
            # 'auto' is valid, Lightning will detect available GPUs
            pass
        elif isinstance(devices, (int, list)):
            # Valid device specification
            pass
        else:
            _logger.warning(
                f"Trainer device configuration warning: GPU accelerator with devices={devices} may be normalized by Lightning. "
                f"Please consider using devices='auto' or devices=[0] for single GPU to avoid potential issues."
            )
    
    return accelerator, devices


# ============================================================================
# Config Validation Helper Functions
# ============================================================================

def _validate_config(
    config: Any,
    trainer_name: str = "trainer"
) -> None:
    """Validate config object for trainer initialization.
    
    Checks that config is not None and provides clear error messages
    for invalid configurations. This helps catch configuration errors early.
    
    Parameters
    ----------
    config : Any
        Configuration object to validate
    trainer_name : str, default "trainer"
        Name of trainer (for error messages)
        
    Raises
    ------
    ValueError
        If config is None or invalid
    TypeError
        If config is not the expected type
    """
    if config is None:
        raise ValueError(
            f"{trainer_name}.from_config() requires a valid config object, "
            f"but received None. Please provide a DFMConfig or DDFMConfig instance."
        )
    
    # Log warning if config doesn't have expected attributes
    # This is a warning, not an error, since defaults will be used
    if not hasattr(config, 'max_iter') and not hasattr(config, 'epochs') and not hasattr(config, 'ddfm_epochs'):
        _logger.warning(
            f"{trainer_name}.from_config() received config without max_iter/epochs attributes. "
            f"Using default max_epochs=100. If this is unexpected, please check your config."
        )


# ============================================================================
# Parameter Extraction Helper Functions
# ============================================================================

def _extract_max_epochs(
    config: Any,
    kwargs: Dict[str, Any],
    defaults: Dict[str, Any],
    use_max_iter: bool = False
) -> int:
    """Extract max_epochs with fallback chain.
    
    Fallback order (highest to lowest priority):
    1. kwargs['max_epochs'] (explicit override)
    2. config.epochs (DDFM config attribute)
    3. config.ddfm_epochs (alternative DDFM config attribute)
    4. config.max_iter (DFM config attribute, only if use_max_iter=True)
    5. defaults['max_epochs'] (default value, typically 100)
    
    Parameters
    ----------
    config : Any
        Configuration object (DFMConfig, DDFMConfig, or any object with attributes)
    kwargs : Dict[str, Any]
        Keyword arguments that may contain max_epochs
    defaults : Dict[str, Any]
        Default values dictionary
    use_max_iter : bool, default False
        If True, also check for max_iter attribute (for DFM configs)
        If False, only check for epochs/ddfm_epochs (for DDFM configs)
        
    Returns
    -------
    int
        Extracted max_epochs value
    """
    if 'max_epochs' in kwargs:
        return kwargs.pop('max_epochs')
    elif hasattr(config, 'epochs'):
        return getattr(config, 'epochs')
    elif hasattr(config, 'ddfm_epochs'):
        return getattr(config, 'ddfm_epochs')
    elif use_max_iter and hasattr(config, 'max_iter'):
        # Only use max_iter for DFM configs (not DDFM)
        return getattr(config, 'max_iter')
    else:
        return defaults.get('max_epochs', 100)


def _extract_bool(
    param_name: str,
    config: Any,
    kwargs: Dict[str, Any],
    defaults: Dict[str, Any]
) -> bool:
    """Extract boolean parameter with fallback chain.
    
    Fallback order (highest to lowest priority):
    1. kwargs[param_name] (explicit override)
    2. config.param_name (config object attribute)
    3. defaults[param_name] (default value)
    
    Parameters
    ----------
    param_name : str
        Name of the boolean parameter to extract
    config : Any
        Configuration object (DFMConfig, DDFMConfig, or any object with attributes)
    kwargs : Dict[str, Any]
        Keyword arguments that may contain the parameter
    defaults : Dict[str, Any]
        Default values dictionary
        
    Returns
    -------
    bool
        Extracted boolean parameter value
    """
    if param_name in kwargs:
        return kwargs.pop(param_name)
    elif hasattr(config, param_name):
        return getattr(config, param_name)
    else:
        return defaults.get(param_name, False)


def _extract_opt(
    param_name: str,
    config: Any,
    kwargs: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Extract optional parameter with fallback chain.
    
    Fallback order (highest to lowest priority):
    1. kwargs[param_name] (explicit override)
    2. config.param_name (config object attribute)
    3. defaults[param_name] (if defaults provided)
    4. None (if not found)
    
    Parameters
    ----------
    param_name : str
        Name of the optional parameter to extract
    config : Any
        Configuration object (DFMConfig, DDFMConfig, or any object with attributes)
    kwargs : Dict[str, Any]
        Keyword arguments that may contain the parameter
    defaults : Optional[Dict[str, Any]], default None
        Default values dictionary (optional)
        
    Returns
    -------
    Optional[Any]
        Extracted parameter value, or None if not found
    """
    if param_name in kwargs:
        return kwargs.pop(param_name)
    elif hasattr(config, param_name):
        return getattr(config, param_name)
    elif defaults and param_name in defaults:
        return defaults[param_name]
    else:
        return None


def _extract_train_params(
    config: Any,
    kwargs: Dict[str, Any],
    defaults: Dict[str, Any],
    use_max_iter: bool = False
) -> Dict[str, Any]:
    """Extract training parameters from config and kwargs with fallbacks.
    
    This function uses helper functions for cleaner organization and better
    maintainability. Each parameter type (max_epochs, boolean, optional) has
    its own extraction function with a clear fallback chain.
    
    Parameters
    ----------
    config : Any
        Configuration object (DFMConfig, DDFMConfig, or any object with attributes)
    kwargs : Dict[str, Any]
        Keyword arguments that may override config values
    defaults : Dict[str, Any]
        Default values to use if not in config or kwargs
    use_max_iter : bool, default False
        If True, also check for max_iter attribute (for DFM configs)
        If False, only check for epochs/ddfm_epochs (for DDFM configs)
        
    Returns
    -------
    Dict[str, Any]
        Extracted parameters dictionary
    """
    params = {}
    
    # Extract max_epochs using helper function
    params['max_epochs'] = _extract_max_epochs(config, kwargs, defaults, use_max_iter)
    
    # Extract boolean parameters using helper function
    params['enable_progress_bar'] = _extract_bool(
        'enable_progress_bar', config, kwargs, defaults
    )
    params['enable_model_summary'] = _extract_bool(
        'enable_model_summary', config, kwargs, defaults
    )
    
    # Extract optional parameters using helper function
    gradient_clip_val = _extract_opt('gradient_clip_val', config, kwargs, defaults)
    if gradient_clip_val is not None:
        params['gradient_clip_val'] = gradient_clip_val
    
    return params


# ============================================================================
# Common Trainer Initialization Helper
# ============================================================================

def _create_base(
    max_epochs: int,
    enable_progress_bar: bool,
    enable_model_summary: bool,
    logger: Any,
    callbacks: Optional[List[Any]],
    accelerator: str,
    devices: Any,
    precision: Any,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: Optional[float] = None,
    early_stopping_monitor: str = 'train_loss',
    logger_type: str = 'csv',
    logger_name: str = 'trainer',
    additional_callbacks: Optional[List[Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create base trainer configuration with common setup logic.
    
    This helper function consolidates the common initialization logic shared
    between DFMTrainer and DDFMTrainer. It handles:
    - Early stopping callback setup
    - Callback list building
    - Logger setup
    - Device/precision normalization and validation
    - Attribute storage preparation
    
    Parameters
    ----------
    max_epochs : int
        Maximum number of epochs
    enable_progress_bar : bool
        Whether to show progress bar
    enable_model_summary : bool
        Whether to print model summary
    logger : Any
        Logger configuration (bool, Logger instance, or None)
    callbacks : Optional[List[Any]]
        Existing callbacks list
    accelerator : str
        Accelerator type ('cpu', 'gpu', 'auto', etc.)
    devices : Any
        Device configuration
    precision : Any
        Training precision (16, 32, 'bf16', etc.)
    early_stopping_patience : int, default 10
        Patience for early stopping callback
    early_stopping_min_delta : Optional[float], default None
        Minimum change for early stopping
    early_stopping_monitor : str, default 'train_loss'
        Metric to monitor for early stopping
    logger_type : str, default 'csv'
        Logger type ('csv' or 'tensorboard')
    logger_name : str, default 'trainer'
        Logger name for save_dir path
    additional_callbacks : Optional[List[Any]], default None
        Additional callbacks to add (e.g., LearningRateMonitor, ModelCheckpoint)
    **kwargs
        Additional arguments to pass to Trainer constructor
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with trainer configuration:
        - 'callbacks': List of callbacks
        - 'logger': Logger instance or None
        - 'accelerator': Normalized accelerator
        - 'devices': Validated devices
        - 'precision': Normalized precision
        - 'enable_progress_bar': bool
        - 'enable_model_summary': bool
        - 'max_epochs': int
        - 'kwargs': Additional kwargs for Trainer
    """
    # Setup early stopping callback
    # Determine mode: 'max' for loglik (DFM), 'min' for loss (DDFM)
    early_stopping_mode = 'max' if early_stopping_monitor == 'loglik' else 'min'
    early_stopping = _setup_early_stopping(
        max_epochs=max_epochs,
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        monitor=early_stopping_monitor,
        mode=early_stopping_mode
    )
    
    # Build callbacks list with early stopping and any additional callbacks
    trainer_callbacks = _build_callbacks(
        callbacks=callbacks,
        early_stopping=early_stopping
    )
    
    # Add additional callbacks if provided
    if additional_callbacks:
        for callback in additional_callbacks:
            if callback is not None:
                trainer_callbacks.append(callback)
    
    # Setup logger
    configured_logger = _setup_logger(logger, logger_type=logger_type, name=logger_name)
    
    # Normalize and validate device/precision configuration
    normalized_accelerator = _normalize_accel(accelerator)
    normalized_precision = _normalize_prec(precision)
    validated_accelerator, validated_devices = _validate_device(
        normalized_accelerator, devices
    )
    
    # Return configuration dictionary for Trainer constructor
    return {
        'max_epochs': max_epochs,
        'enable_progress_bar': enable_progress_bar,
        'enable_model_summary': enable_model_summary,
        'logger': configured_logger,
        'callbacks': trainer_callbacks,
        'accelerator': validated_accelerator,
        'devices': validated_devices,
        'precision': normalized_precision,
        **kwargs
    }


# ============================================================================
# Trainer Class Exports
# ============================================================================

from .dfm import DFMTrainer
from .ddfm import DDFMTrainer

__all__ = [
    'DFMTrainer',
    'DDFMTrainer',
    'DFM_TRAINER_DEFAULTS',
    'DDFM_TRAINER_DEFAULTS',
]


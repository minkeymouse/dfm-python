"""Custom logging configuration for dfm-python.

This module provides standard Python logging setup and configuration utilities.
All logging uses Python's standard logging module with hierarchical configuration.
"""

import logging
import sys
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime


class CoreDebugFilter(logging.Filter):
    """Filter to capture only core debugging information for log_dfm files.
    
    Captures:
    - EM iterations and convergence
    - Training start/end
    - Initialization summaries
    - Warnings and errors
    - Key parameter updates
    - Skips: DEBUG messages, verbose timing logs, detailed breakdowns
    """
    
    # Keywords that indicate important/core messages
    CORE_KEYWORDS = [
        'training', 'iteration', 'converged', 'convergence',
        'initialization', 'initialized', 'initializing',
        'failed', 'error', 'warning', 'exception',
        'max_iter', 'threshold', 'loglik', 'delta',
        'completed', 'starting', 'block structure',
        'mixed-frequency', 'dataset ready', 'model saved',
        'eigenval', 'max |Q|', 'max |C|', 'max |A|',
        'e-step:', 'm-step:', 'completed in'
    ]
    
    # Logger names that should always be included (even DEBUG)
    CORE_LOGGERS = [
        'dfm_python.models.dfm',
        'dfm_python.ssm.kalman',
        'dfm_python.logger.train_logger'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if record should be logged.
        
        Returns True if:
        - Level is WARNING or higher
        - Level is INFO and message contains core keywords
        - Logger is in CORE_LOGGERS and level is INFO or higher
        - Message starts with important patterns
        """
        # Always include warnings and errors
        if record.levelno >= logging.WARNING:
            return True
        
        # Skip DEBUG messages unless from core loggers
        if record.levelno == logging.DEBUG:
            # Allow DEBUG from core loggers if they're critical
            logger_name = record.name
            if any(logger_name.startswith(core) for core in self.CORE_LOGGERS):
                # Only allow DEBUG for critical initialization/error messages
                msg_lower = record.getMessage().lower()
                if any(kw in msg_lower for kw in ['error', 'failed', 'exception', 'initialization']):
                    return True
            return False
        
        # For INFO messages, check if they contain core keywords
        if record.levelno == logging.INFO:
            msg_lower = record.getMessage().lower()
            
            # Check if from core logger
            logger_name = record.name
            if any(logger_name.startswith(core) for core in self.CORE_LOGGERS):
                # Include INFO from core loggers, but filter verbose timing
                # Skip very detailed breakdown logs
                if any(skip in msg_lower for skip in [
                    'detailed breakdown', 'timing breakdown',
                    'smooth_pair', 'filter timing', 'smooth timing'
                ]):
                    return False
                # Include everything else from core loggers
                return True
            
            # For other loggers, check keywords
            if any(kw in msg_lower for kw in self.CORE_KEYWORDS):
                return True
        
        return False


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    This is the standard way to get a logger in the DFM package.
    All modules should use: _logger = get_logger(__name__)
    
    The logger uses hierarchical configuration:
    - Child loggers (e.g., 'dfm_python.models.dfm') inherit from parent logger ('dfm_python')
    - Parent logger ('dfm_python') is configured once with handlers
    - Child loggers propagate to parent (default behavior)
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Logger instance configured for the package
    """
    logger = logging.getLogger(name)
    
    # Ensure package-level logger is configured (only once)
    package_logger = logging.getLogger('dfm_python')
    
    # Configure package logger if not already configured
    # Use a flag to avoid re-configuring if already done
    if not hasattr(package_logger, '_dfm_configured'):
        # CRITICAL: Check if handlers already exist to prevent duplicates
        # This can happen if configure_logging() was called before get_logger()
        if not package_logger.handlers:
            # Configure handler for dfm_python package logger
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            package_logger.addHandler(handler)
        
        package_logger.setLevel(logging.INFO)
        # CRITICAL: Disable propagation to root logger to prevent duplicate messages
        # Root logger may have handlers from other code (e.g., basicConfig)
        # By setting propagate=False, we ensure messages only go through our handler
        package_logger.propagate = False
        # Mark as configured to avoid duplicate handlers
        package_logger._dfm_configured = True
    
    # Child logger should propagate to package logger (not root)
    # Since package logger has propagate=False, messages won't go to root
    logger.propagate = True
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """Setup package-wide logging configuration.
    
    Alias for configure_logging().
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
    configure_logging(level=level, format_string=format_string)


def configure_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None,
    log_dfm_dir: Optional[Path] = None,
    enable_core_log: bool = True,
    log_file_prefix: Optional[str] = None
) -> None:
    """Configure package-wide logging.
    
    Parameters
    ----------
    level : int, default logging.INFO
        Logging level for the package
    format_string : str, optional
        Custom format string. If None, uses default format.
    log_file : str, optional
        Optional file path to write logs to. If provided, logs will be
        written to both console and file.
    module_levels : dict, optional
        Dictionary mapping module names to specific log levels.
        Example: {'dfm_python.models': logging.DEBUG, 'dfm_python.trainer': logging.WARNING}
    log_dfm_dir : Path, optional
        Directory for core debugging logs. If provided and enable_core_log=True,
        creates a timestamped log file with filtered core debugging information.
    enable_core_log : bool, default True
        Whether to enable core debugging log file in log_dfm_dir.
    log_file_prefix : str, optional
        Prefix for log file name. If None, defaults to "dfm_core" or infers from directory name.
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure package logger
    logger = logging.getLogger('dfm_python')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    # Clear the configuration flag so it can be reconfigured
    if hasattr(logger, '_dfm_configured'):
        delattr(logger, '_dfm_configured')
    
    # CRITICAL: Disable propagation to root logger to prevent duplicate messages
    # If root logger also has handlers, messages would be logged twice
    logger.propagate = False
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified (unfiltered)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add core debugging log file (filtered) in log_dfm_dir
    if enable_core_log and log_dfm_dir is not None:
        log_dfm_dir = Path(log_dfm_dir)
        log_dfm_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log file prefix
        if log_file_prefix is None:
            # Infer from directory name (log_dfm -> dfm_core, log_ddfm -> ddfm_core)
            dir_name = log_dfm_dir.name
            if dir_name == "log_ddfm":
                log_file_prefix = "ddfm_core"
            elif dir_name == "log_dfm":
                log_file_prefix = "dfm_core"
            else:
                log_file_prefix = "dfm_core"  # Default fallback
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        core_log_file = log_dfm_dir / f"{log_file_prefix}_{timestamp}.log"
        
        core_file_handler = logging.FileHandler(core_log_file)
        core_file_handler.setLevel(logging.DEBUG)  # Set to DEBUG so filter can decide
        core_file_handler.setFormatter(formatter)
        
        # Add filter to capture only core debugging info
        core_filter = CoreDebugFilter()
        core_file_handler.addFilter(core_filter)
        
        logger.addHandler(core_file_handler)
        
        # Log that core logging is enabled
        logger.info(f"Core debugging logs enabled: {core_log_file}")
    
    # Set module-specific log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)


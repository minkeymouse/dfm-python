"""Basic logging configuration for dfm-python.

This module provides standard logging setup and configuration utilities.
"""

import logging
import sys
from typing import Optional, Dict


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
    module_levels: Optional[Dict[str, int]] = None
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
    
    # Add file handler if specified
    if log_file:
        from pathlib import Path
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set module-specific log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)


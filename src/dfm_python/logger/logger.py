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
    
    # Only configure if not already configured
    if not logger.handlers:
        # Use package-level logger configuration
        package_logger = logging.getLogger('dfm_python')
        if not package_logger.handlers:
            # Configure root logger for dfm_python package
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            package_logger.addHandler(handler)
            package_logger.setLevel(logging.INFO)
    
    return logger


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """Setup package-wide logging configuration.
    
    This is an alias for configure_logging() for backward compatibility.
    
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
    
    # Remove existing handlers
    logger.handlers.clear()
    
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


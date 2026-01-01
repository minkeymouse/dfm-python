"""Tests for logger.logger module."""

import pytest
from dfm_python.logger.logger import get_logger


class TestLogger:
    """Test suite for logger."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger(__name__)
        assert logger is not None
    
    def test_logger_levels(self):
        """Test logger level configuration."""
        import logging
        from dfm_python.logger.logger import configure_logging
        
        # Test setting different log levels
        configure_logging(level=logging.DEBUG)
        logger = get_logger(__name__)
        assert logger.level <= logging.DEBUG
        
        configure_logging(level=logging.WARNING)
        logger2 = get_logger(__name__)
        assert logger2.level <= logging.WARNING


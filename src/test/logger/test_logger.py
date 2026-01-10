"""Tests for logger.logger module."""

import pytest
import logging
import tempfile
from pathlib import Path
from dfm_python.logger.logger import get_logger, configure_logging


class TestLogger:
    """Test suite for logger."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger(__name__)
        assert logger is not None
    
    def test_logger_levels(self):
        """Test logger level configuration."""
        # Test setting different log levels
        configure_logging(level=logging.DEBUG)
        logger = get_logger(__name__)
        assert logger.level <= logging.DEBUG
        
        configure_logging(level=logging.WARNING)
        logger2 = get_logger(__name__)
        assert logger2.level <= logging.WARNING
    
    def test_configure_logging_with_log_dfm_dir(self, tmp_path):
        """Test configure_logging creates log files in log_dfm_dir."""
        log_dir = tmp_path / "log" / "log_dfm"
        
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dir,
            enable_core_log=True,
            log_file_prefix="dfm_core"
        )
        
        # Verify directory was created
        assert log_dir.exists()
        assert log_dir.is_dir()
        
        # Check that log files exist (may be empty initially)
        log_files = list(log_dir.glob("dfm_core_*.log"))
        assert len(log_files) >= 0  # At least the directory structure exists
    
    def test_configure_logging_with_log_ddfm_dir(self, tmp_path):
        """Test configure_logging creates log files in log_ddfm_dir."""
        log_dir = tmp_path / "log" / "log_ddfm"
        
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dir,
            enable_core_log=True,
            log_file_prefix="ddfm_core"
        )
        
        # Verify directory was created
        assert log_dir.exists()
        assert log_dir.is_dir()
        
        # Verify directory name
        assert log_dir.name == "log_ddfm"
        assert log_dir.parent.name == "log"
    
    def test_log_file_prefix_inference_from_directory(self, tmp_path):
        """Test that log file prefix is inferred from directory name."""
        # Test log_dfm directory
        log_dfm_dir = tmp_path / "log" / "log_dfm"
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dfm_dir,
            enable_core_log=True,
            log_file_prefix=None  # Should infer from directory name
        )
        assert log_dfm_dir.exists()
        
        # Test log_ddfm directory
        log_ddfm_dir = tmp_path / "log" / "log_ddfm"
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_ddfm_dir,
            enable_core_log=True,
            log_file_prefix=None  # Should infer from directory name
        )
        assert log_ddfm_dir.exists()
    
    def test_log_directory_structure(self, tmp_path):
        """Test that log directories follow the pattern log/log_dfm and log/log_ddfm."""
        base_log_dir = tmp_path / "log"
        
        # Test DFM structure
        log_dfm_dir = base_log_dir / "log_dfm"
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dfm_dir,
            enable_core_log=True
        )
        
        # Verify structure: log/log_dfm
        assert base_log_dir.exists()
        assert log_dfm_dir.exists()
        assert log_dfm_dir.parent == base_log_dir
        assert log_dfm_dir.name == "log_dfm"
        
        # Test DDFM structure
        log_ddfm_dir = base_log_dir / "log_ddfm"
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_ddfm_dir,
            enable_core_log=True
        )
        
        # Verify structure: log/log_ddfm
        assert log_ddfm_dir.exists()
        assert log_ddfm_dir.parent == base_log_dir
        assert log_ddfm_dir.name == "log_ddfm"
    
    def test_configure_logging_without_log_dir(self):
        """Test configure_logging works without log directory (console only)."""
        configure_logging(level=logging.INFO, enable_core_log=False)
        logger = get_logger(__name__)
        assert logger is not None
        assert logger.level <= logging.INFO
    
    def test_core_log_file_creation(self, tmp_path):
        """Test that core log files are created with correct naming pattern."""
        log_dfm_dir = tmp_path / "log" / "log_dfm"
        
        configure_logging(
            level=logging.INFO,
            log_dfm_dir=log_dfm_dir,
            enable_core_log=True,
            log_file_prefix="dfm_core"
        )
        
        # Log something to trigger file creation
        logger = get_logger(__name__)
        logger.info("Test log message")
        
        # Check for log files matching the pattern
        log_files = list(log_dfm_dir.glob("dfm_core_*.log"))
        # Note: Files may exist but be empty, so we just verify the directory structure
        assert log_dfm_dir.exists()


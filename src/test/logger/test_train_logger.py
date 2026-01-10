"""Tests for logger.train_logger module."""

import pytest


class TestTrainLogger:
    """Test suite for BaseTrainLogger."""
    
    def test_train_logger_initialization(self):
        """Test BaseTrainLogger can be initialized."""
        from dfm_python.logger.train_logger import BaseTrainLogger
        
        logger = BaseTrainLogger(model_name="TestModel", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.verbose is True
        assert hasattr(logger, 'iterations')
        assert logger.iterations == 0  # Initialized to 0


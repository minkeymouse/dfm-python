"""Tests for logger.train_logger module."""

import pytest


class TestTrainLogger:
    """Test suite for TrainLogger."""
    
    def test_train_logger_initialization(self):
        """Test TrainLogger can be initialized."""
        from dfm_python.logger.train_logger import TrainLogger
        
        logger = TrainLogger(model_name="TestModel", model_type="dfm", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.model_type == "dfm"
        assert logger.verbose is True
        assert hasattr(logger, 'iterations') or hasattr(logger, 'num_iterations')


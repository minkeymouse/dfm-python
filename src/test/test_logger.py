"""Tests for logging utilities.

Tests logging configuration, training logging, and inference logging.
"""

import pytest
import logging
from datetime import datetime
from typing import Dict, Any

from dfm_python.logger import (
    get_logger, setup_logging, configure_logging,
    TrainLogger, InferenceLogger,
    log_training_start, log_training_step, log_training_end,
    log_em_iteration, log_convergence,
    log_inference_start, log_inference_step, log_inference_end,
    log_prediction,
)


class TestBasicLogging:
    """Test basic logging configuration."""
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        setup_logging(level=logging.DEBUG)
        logger = get_logger("test")
        assert logger.level <= logging.DEBUG
    
    def test_configure_logging(self):
        """Test configure_logging function."""
        configure_logging(level=logging.INFO)
        logger = get_logger("test")
        assert logger.level <= logging.INFO
    
    def test_configure_logging_with_file(self, tmp_path):
        """Test configure_logging with file output."""
        log_file = tmp_path / "test.log"
        configure_logging(level=logging.INFO, log_file=str(log_file))
        
        logger = get_logger("test")
        logger.info("Test message")
        
        # File should be created
        assert log_file.exists()


class TestTrainLogger:
    """Test TrainLogger for training process tracking."""
    
    def test_train_logger_initialization(self):
        """Test TrainLogger initialization."""
        logger = TrainLogger(model_name="DFM", verbose=True)
        assert logger.model_name == "DFM"
        assert logger.verbose is True
        assert logger.iterations == 0
        assert logger.converged is False
    
    def test_train_logger_start(self):
        """Test training start logging."""
        logger = TrainLogger(model_name="DFM")
        logger.start(config={"max_iter": 100, "threshold": 1e-5})
        
        assert logger.start_time is not None
        assert logger.iterations == 0
        assert logger.converged is False
    
    def test_train_logger_iteration(self):
        """Test training iteration logging."""
        logger = TrainLogger(model_name="DFM")
        logger.start()
        
        logger.log_iteration(
            iteration=1,
            loglik=-100.0,
            delta=0.01
        )
        
        assert logger.iterations == 1
    
    def test_train_logger_end(self):
        """Test training end logging."""
        logger = TrainLogger(model_name="DFM")
        logger.start()
        logger.log_iteration(iteration=1, loglik=-100.0)
        logger.log_convergence(converged=True, num_iter=1)
        logger.end()
        
        assert logger.end_time is not None
        assert logger.converged is True
    
    def test_log_em_iteration(self):
        """Test EM iteration logging."""
        # log_em_iteration requires logger parameter
        logger = TrainLogger()
        log_em_iteration(logger=logger, iteration=1, loglik=-100.0, delta=0.01)
        # Function should execute without error
    
    def test_log_convergence(self):
        """Test convergence logging."""
        log_convergence(converged=True, num_iter=50)
        # Function should execute without error


class TestInferenceLogger:
    """Test InferenceLogger for inference process tracking."""
    
    def test_inference_logger_initialization(self):
        """Test InferenceLogger initialization."""
        logger = InferenceLogger(model_name="DFM", verbose=True)
        assert logger.model_name == "DFM"
        assert logger.verbose is True
        assert logger.num_predictions == 0
    
    def test_inference_logger_start(self):
        """Test inference start logging."""
        logger = InferenceLogger(model_name="DFM")
        logger.start(task="nowcasting")
        
        assert logger.start_time is not None
    
    def test_inference_logger_step(self):
        """Test inference step logging."""
        logger = InferenceLogger(model_name="DFM")
        logger.start()
        
        logger.log_step(step=1, description="Computing nowcast")
        assert logger.num_predictions == 0  # Step doesn't increment predictions
    
    def test_inference_logger_end(self):
        """Test inference end logging."""
        logger = InferenceLogger(model_name="DFM")
        logger.start()
        logger.end()
        
        assert logger.end_time is not None
    
    def test_log_inference_start(self):
        """Test inference start logging function."""
        logger = InferenceLogger()
        log_inference_start(logger=logger, task="nowcasting")
        # Function should execute without error
    
    def test_log_inference_step(self):
        """Test inference step logging function."""
        logger = InferenceLogger()
        log_inference_step(logger=logger, step=1, description="Computing nowcast")
        # Function should execute without error
    
    def test_log_inference_end(self):
        """Test inference end logging function."""
        logger = InferenceLogger()
        log_inference_end(logger=logger, num_predictions=10)
        # Function should execute without error
    
    def test_log_prediction(self):
        """Test prediction logging function."""
        log_prediction(
            target="GDP",
            period=datetime(2020, 3, 31),
            value=2.5
        )
        # Function should execute without error


class TestLoggingConsistency:
    """Test logging consistency across modules."""
    
    def test_logger_namespace(self):
        """Test that loggers use correct namespace."""
        logger1 = get_logger("dfm_python.models")
        logger2 = get_logger("dfm_python.ssm")
        
        assert logger1.name == "dfm_python.models"
        assert logger2.name == "dfm_python.ssm"
    
    def test_logging_levels(self):
        """Test logging level configuration."""
        configure_logging(level=logging.WARNING)
        logger = get_logger("test")
        
        # INFO messages should not be logged
        assert logger.level <= logging.WARNING


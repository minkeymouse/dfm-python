"""Tests for logger.inference_logger module."""

import pytest


class TestInferenceLogger:
    """Test suite for BaseInferenceLogger."""
    
    def test_inference_logger_initialization(self):
        """Test BaseInferenceLogger can be initialized."""
        from dfm_python.logger.inference_logger import BaseInferenceLogger
        
        logger = BaseInferenceLogger(model_name="TestModel", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.verbose is True
        assert logger.num_predictions == 0


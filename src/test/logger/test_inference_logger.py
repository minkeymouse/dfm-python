"""Tests for logger.inference_logger module."""

import pytest


class TestInferenceLogger:
    """Test suite for InferenceLogger."""
    
    def test_inference_logger_initialization(self):
        """Test InferenceLogger can be initialized."""
        from dfm_python.logger.inference_logger import InferenceLogger
        
        logger = InferenceLogger(model_name="TestModel", model_type="dfm", verbose=True)
        assert logger.model_name == "TestModel"
        assert logger.model_type == "dfm"
        assert logger.verbose is True
        assert logger.num_predictions == 0


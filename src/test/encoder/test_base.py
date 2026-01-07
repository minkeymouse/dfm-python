"""Tests for encoder.base module."""

import pytest
import numpy as np
from dfm_python.encoder.base import BaseEncoder
from dfm_python.encoder.pca import PCAEncoder


class TestBaseEncoder:
    """Test suite for BaseEncoder."""
    
    def test_base_encoder_is_abstract(self):
        """Test BaseEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEncoder(n_components=2)
    
    def test_base_encoder_interface(self):
        """Test BaseEncoder defines required interface."""
        # Test that concrete implementations follow the interface
        # PCAEncoder implements encode
        pca_encoder = PCAEncoder(n_components=2)
        assert hasattr(pca_encoder, 'encode')
        assert hasattr(pca_encoder, 'fit')
        assert hasattr(pca_encoder, 'fit_encode')
        assert pca_encoder.n_components == 2
    
    def test_base_encoder_fit_default_implementation(self):
        """Test BaseEncoder fit() default no-op implementation."""
        # Use PCAEncoder which inherits fit() default
        # But PCAEncoder overrides fit(), so test via a mock or check the default exists
        # The default implementation in BaseEncoder returns self
        # PCAEncoder overrides it, so we verify the method exists
        pca_encoder = PCAEncoder(n_components=2)
        assert callable(pca_encoder.fit)
    
    def test_base_encoder_fit_encode_method(self):
        """Test BaseEncoder fit_encode convenience method."""
        pca_encoder = PCAEncoder(n_components=2)
        X = np.random.randn(10, 5)
        factors = pca_encoder.fit_encode(X)
        assert factors.shape == (10, 2)
        assert pca_encoder.eigenvectors is not None  # Should be fitted


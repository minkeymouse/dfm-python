"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- Encoder: DDFM-specific nonlinear encoder (simple_autoencoder)
"""

from .base import BaseEncoder

from .pca import (
    PCAEncoder,
    compute_principal_components,
)

from .simple_autoencoder import (
    Encoder,
    SimpleAutoencoder,
    extract_decoder_params,
)

from ..decoder import LinearDecoder, MLPDecoder

__all__ = [
    # Base
    'BaseEncoder',
    # PCA
    'PCAEncoder',
    'compute_principal_components',
    # DDFM Encoder
    'Encoder',
    'SimpleAutoencoder',
    'extract_decoder_params',
    # Decoder
    'LinearDecoder',
    'MLPDecoder',
]


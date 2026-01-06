"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- Encoder: DDFM-specific nonlinear encoder (simple_autoencoder)
- AutoencoderEncoder: Generic autoencoder wrapper (variational_encoder)
- VariationalEncoder: VAE encoder (variational_encoder, placeholder)
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

# Variational autoencoder imports (optional - file may be empty)
try:
    from .variational_autoencoder import (
        AutoencoderEncoder,
        VariationalEncoder,
    )
except ImportError:
    # Variational autoencoder not implemented yet
    AutoencoderEncoder = None
    VariationalEncoder = None

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
    # Autoencoder/VAE
    'AutoencoderEncoder',
    'VariationalEncoder',
    # Decoder
    'LinearDecoder',
    'MLPDecoder',
]


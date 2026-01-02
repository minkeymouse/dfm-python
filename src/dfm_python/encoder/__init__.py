"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- Encoder: DDFM-specific nonlinear encoder (simple_encoder)
- AutoencoderEncoder: Generic autoencoder wrapper (variational_encoder)
- VariationalEncoder: VAE encoder (variational_encoder, placeholder)
"""

from .base import BaseEncoder

from .pca import (
    PCAEncoder,
    compute_principal_components,
)

from .simple_encoder import (
    Encoder,
    extract_decoder_params,
    convert_decoder_to_numpy,
)

from .variational_encoder import (
    AutoencoderEncoder,
    VariationalEncoder,
)

from ..decoder.linear import Decoder

__all__ = [
    # Base
    'BaseEncoder',
    # PCA
    'PCAEncoder',
    'compute_principal_components',
    # DDFM Encoder
    'Encoder',
    'extract_decoder_params',
    'convert_decoder_to_numpy',
    # Autoencoder/VAE
    'AutoencoderEncoder',
    'VariationalEncoder',
    # Decoder
    'Decoder',
]


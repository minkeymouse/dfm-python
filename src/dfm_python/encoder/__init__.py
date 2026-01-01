"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- Autoencoder: Nonlinear deep learning encoder/decoder for DDFM
"""

from .base import BaseEncoder

from .pca import (
    PCAEncoder,
    compute_principal_components,
)

from .simple_encoder import (
    Encoder,
    AutoencoderEncoder,
    extract_decoder_params,
    convert_decoder_to_numpy,
)

from ..decoder.linear import Decoder

__all__ = [
    # Base
    'BaseEncoder',
    # PCA
    'PCAEncoder',
    'compute_principal_components',
    # Autoencoder
    'Encoder',
    'AutoencoderEncoder',
    'Decoder',
    'extract_decoder_params',
    'convert_decoder_to_numpy',
]


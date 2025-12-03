"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- VAE: Variational Autoencoder (nonlinear deep learning encoder/decoder)
"""

from .base import BaseEncoder

from .pca import (
    PCAEncoder,
    compute_principal_components,
    compute_principal_components_torch,
)

from .vae import (
    Encoder,
    VAEEncoder,
    Decoder,
    extract_decoder_params,
    convert_decoder_to_numpy,
)

__all__ = [
    # Base
    'BaseEncoder',
    # PCA
    'PCAEncoder',
    'compute_principal_components',
    'compute_principal_components_torch',
    # VAE
    'Encoder',
    'VAEEncoder',
    'Decoder',
    'extract_decoder_params',
    'convert_decoder_to_numpy',
]


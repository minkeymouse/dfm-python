"""Encoder modules for factor extraction.

This package provides implementations of various encoding methods for
extracting latent factors from observed time series data:
- PCA: Principal Component Analysis (linear dimension reduction)
- VAE: Variational Autoencoder (nonlinear deep learning encoder/decoder)
"""

from .pca import (
    compute_principal_components,
    compute_principal_components_torch,
    _compute_principal_components,  # Backward compatibility alias
)

from .vae import (
    Encoder,
    Decoder,
    extract_decoder_params,
    convert_decoder_to_numpy,
)

__all__ = [
    # PCA
    'compute_principal_components',
    'compute_principal_components_torch',
    '_compute_principal_components',  # Backward compatibility
    # VAE
    'Encoder',
    'Decoder',
    'extract_decoder_params',
    'convert_decoder_to_numpy',
]


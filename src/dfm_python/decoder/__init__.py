"""Decoder modules for DDFM.

This package provides decoder implementations for reconstructing observations
from latent factors in the Deep Dynamic Factor Model (DDFM).
"""

from .linear import Decoder
from .mlp import MLPDecoder

__all__ = [
    'Decoder',
    'MLPDecoder',
]


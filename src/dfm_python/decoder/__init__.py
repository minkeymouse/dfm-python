"""Decoder modules for DDFM.

This package provides decoder implementations for reconstructing observations
from latent factors in the Deep Dynamic Factor Model (DDFM).
"""

from .base import Decoder
from .linear import LinearDecoder
from .mlp import MLPDecoder

__all__ = [
    'Decoder',
    'LinearDecoder',
    'MLPDecoder',
]


"""Decoder modules for DDFM.

This package provides decoder implementations for reconstructing observations
from latent factors in the Deep Dynamic Factor Model (DDFM).
"""

from .base import BaseDecoder, build_decoder
from .linear import Decoder
from .mlp import MLPDecoder

__all__ = [
    'BaseDecoder',
    'build_decoder',
    'Decoder',
    'MLPDecoder',
]


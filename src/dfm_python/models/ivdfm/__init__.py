"""Identifiable Variational Dynamic Factor Model (iVDFM).

This module provides the iVDFM model and its components.
"""

from .ivdfm import iVDFM
from .encoder import iVDFMInnovationEncoder
from .decoder import iVDFMDecoder
from .prior import iVDFMPriorNetwork

__all__ = [
    'iVDFM',
    'iVDFMInnovationEncoder',
    'iVDFMDecoder',
    'iVDFMPriorNetwork',
]

"""Deep Dynamic Factor Model (DDFM) implementation.

This package contains the PyTorch-based DDFM implementation with
nonlinear encoder and linear decoder.
"""

try:
    from .model import DDFM
    __all__ = ['DDFM']
except ImportError:
    DDFM = None  # type: ignore
    __all__ = []


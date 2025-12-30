"""Functional modules for KDFM.

This package provides functional utilities for KDFM:
- Krylov FFT computation
- Structural identification
- IRF computation
"""

from .krylov import krylov, krylov_sequential
from .irf import compute_irf

__all__ = ['krylov', 'krylov_sequential', 'compute_irf']


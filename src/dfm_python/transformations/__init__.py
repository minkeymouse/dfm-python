"""Transformations module for DFM.

This module provides transformation utilities and data reading functions.
Users should provide their own sktime transformers to DFMDataModule.
"""

from .transformers import (
    identity_transform,
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
)
# Import get_periods_per_year and get_annual_factor from config.structure (canonical location)
from ..config.structure import get_periods_per_year, get_annual_factor
from .utils import (
    read_data,
    load_data,
)

__all__ = [
    # Frequency utilities (from config.structure)
    'get_periods_per_year',
    'get_annual_factor',
    # Transformation functions
    'identity_transform',
    'make_pch_transformer',
    'make_pc1_transformer',
    'make_pca_transformer',
    'make_cch_transformer',
    'make_cca_transformer',
    # Data reading
    'read_data',
    'load_data',
]


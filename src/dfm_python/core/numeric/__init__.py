"""Numerical utilities for DFM estimation.

This package contains:
- matrix: Matrix operations (symmetric, real, square)
- covariance: Covariance/variance computation
- regularization: Regularization, PSD enforcement, AR clipping, and general utilities
"""

# Matrix operations
from .matrix import (
    _ensure_square_matrix,
    _ensure_symmetric,
    _ensure_real,
    _ensure_real_and_symmetric,
    _ensure_covariance_stable,
    _compute_principal_components,
    _clean_matrix,
    MIN_EIGENVAL_CLEAN,
    MIN_DIAGONAL_VARIANCE,
)

# Covariance operations
from .covariance import (
    _compute_covariance_safe,
    _compute_variance_safe,
    DEFAULT_VARIANCE_FALLBACK,
    MIN_VARIANCE_COVARIANCE,
)

# Regularization, clipping, and utilities
from .regularization import (
    _ensure_innovation_variance_minimum,
    _ensure_positive_definite,
    _compute_regularization_param,
    _cap_max_eigenvalue,
    _estimate_ar_coefficient,
    _check_finite,
    _safe_divide,
    _clip_ar_coefficients,
    _apply_ar_clipping,
)

__all__ = [
    # Matrix operations
    '_ensure_square_matrix',
    '_ensure_symmetric',
    '_ensure_real',
    '_ensure_real_and_symmetric',
    '_ensure_covariance_stable',
    '_compute_principal_components',
    '_clean_matrix',
    'MIN_EIGENVAL_CLEAN',
    'MIN_DIAGONAL_VARIANCE',
    # Covariance
    '_compute_covariance_safe',
    '_compute_variance_safe',
    'DEFAULT_VARIANCE_FALLBACK',
    'MIN_VARIANCE_COVARIANCE',
    # Regularization
    '_ensure_innovation_variance_minimum',
    '_ensure_positive_definite',
    '_compute_regularization_param',
    '_cap_max_eigenvalue',
    '_estimate_ar_coefficient',
    # Clipping
    '_clip_ar_coefficients',
    '_apply_ar_clipping',
    # Utilities
    '_check_finite',
    '_safe_divide',
]

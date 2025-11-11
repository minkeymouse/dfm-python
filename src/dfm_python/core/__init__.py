"""Core utilities for DFM estimation.

This package contains core functionality organized by domain:
- em: EM algorithm core package (init_conditions, em_step, em_converged)
- numeric: Numerical utilities package (matrix operations, regularization, clipping)
- diagnostics: Diagnostic functions and output formatting
- helpers: Common helper functions for code patterns
"""

from .em import init_conditions, em_step, em_converged
from .numeric import (
    _ensure_symmetric,
    _ensure_real,
    _ensure_real_and_symmetric,
    _ensure_covariance_stable,
    _ensure_innovation_variance_minimum,
    _compute_covariance_safe,
    _compute_variance_safe,
    _compute_principal_components,
    _clean_matrix,
    _ensure_positive_definite,
    _compute_regularization_param,
    _clip_ar_coefficients,
    _apply_ar_clipping,
    _cap_max_eigenvalue,
    _estimate_ar_coefficient,
    _safe_divide,
    _check_finite,
    _ensure_square_matrix,
)
from .diagnostics import (
    calculate_rmse,
    _display_dfm_tables,
    diagnose_series,
    print_series_diagnosis,
)
from .helpers import safe_get_method, safe_get_attr
from ..utils import group_series_by_frequency

__all__ = [
    'init_conditions',
    'em_step',
    'em_converged',
    'calculate_rmse',
    'group_series_by_frequency',
    '_ensure_symmetric',
    '_ensure_real',
    '_ensure_real_and_symmetric',
    '_ensure_covariance_stable',
    '_ensure_innovation_variance_minimum',
    '_compute_covariance_safe',
    '_compute_principal_components',
    '_clean_matrix',
    '_ensure_positive_definite',
    '_compute_regularization_param',
    '_clip_ar_coefficients',
    '_apply_ar_clipping',
    '_cap_max_eigenvalue',
    '_estimate_ar_coefficient',
    '_compute_variance_safe',
    '_safe_divide',
    '_check_finite',
    '_ensure_square_matrix',
    '_display_dfm_tables',
    'diagnose_series',
    'print_series_diagnosis',
    'safe_get_method',
    'safe_get_attr',
]

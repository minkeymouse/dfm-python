"""Helper functions organized by domain.

This package provides utility functions for common code patterns, organized
into domain-specific modules for better maintainability.
"""

# Import from domain-specific modules
from .config import safe_get_method, safe_get_attr
from .estimation import (
    estimate_ar_coefficients_ols,
    compute_innovation_covariance,
    compute_sufficient_stats,
    safe_mean_std,
)
from .validation import validate_params, stabilize_cov
from .matrix import (
    reg_inv,
    update_loadings,
    extract_3d_matrix_slice,
    compute_obs_cov,
    clean_variance_array,
)
from .block import (
    update_block_diag,
    get_block_indices,
    compute_block_slice_indices,
    extract_block_matrix,
    update_block_in_matrix,
)
from .frequency import get_tent_weights, infer_nQ
from .array import (
    safe_time_index,
    safe_array_operation,
    get_matrix_shape,
    has_valid_data,
    ensure_minimum_size,
)
from .utils import (
    append_or_initialize,
    create_empty_matrix,
    reshape_to_column_vector,
    reshape_to_row_vector,
    pad_matrix_to_shape,
    safe_numerical_operation,
    resolve_param,
)

__all__ = [
    # Config
    'safe_get_method',
    'safe_get_attr',
    # Estimation
    'estimate_ar_coefficients_ols',
    'compute_innovation_covariance',
    'compute_sufficient_stats',
    'safe_mean_std',
    # Validation
    'validate_params',
    'stabilize_cov',
    # Matrix
    'reg_inv',
    'update_loadings',
    'extract_3d_matrix_slice',
    'compute_obs_cov',
    'clean_variance_array',
    # Block
    'update_block_diag',
    'get_block_indices',
    'compute_block_slice_indices',
    'extract_block_matrix',
    'update_block_in_matrix',
    # Frequency
    'get_tent_weights',
    'infer_nQ',
    # Array
    'safe_time_index',
    'safe_array_operation',
    'get_matrix_shape',
    'has_valid_data',
    'ensure_minimum_size',
    # Utils
    'append_or_initialize',
    'create_empty_matrix',
    'reshape_to_column_vector',
    'reshape_to_row_vector',
    'pad_matrix_to_shape',
    'safe_numerical_operation',
    'resolve_param',
]


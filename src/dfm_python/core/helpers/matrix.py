"""Matrix operation helpers for loading updates, covariance computation, and matrix slicing."""

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import inv
from ...config import DFMConfig
from .config import safe_get_attr
from ._common import NUMERICAL_EXCEPTIONS
from .array import safe_time_index
from ..numeric import _compute_regularization_param


def reg_inv(
    denom: np.ndarray,
    nom: np.ndarray,
    config: Optional[DFMConfig],
    default_scale: float = 1e-5
) -> Tuple[np.ndarray, bool]:
    """Compute regularized matrix inverse for loading updates.
    
    Parameters
    ----------
    denom : np.ndarray
        Denominator matrix (typically sum of Z @ Z' terms)
    nom : np.ndarray
        Numerator matrix or vector (typically sum of y @ Z' terms)
    config : DFMConfig, optional
        Configuration object for regularization parameters
    default_scale : float, default 1e-5
        Default regularization scale if not in config
        
    Returns
    -------
    result : np.ndarray
        Computed result: inv(denom_reg) @ nom
    success : bool
        True if computation succeeded, False if exception occurred
        
    Notes
    -----
    - Aligns with MATLAB: vec_C = inv(denom)*nom(:)
    - Uses regularization for numerical stability (MATLAB uses direct inversion)
    - Handles both matrix and vector numerators
    """
    try:
        scale_factor = safe_get_attr(config, "regularization_scale", default_scale)
        warn_reg = safe_get_attr(config, "warn_on_regularization", True)
        reg_param, _ = _compute_regularization_param(denom, scale_factor, warn_reg)
        denom_reg = denom + np.eye(denom.shape[0]) * reg_param
        
        # Align with MATLAB: vec_C = inv(denom)*nom(:)
        # MATLAB uses direct inversion, we use regularized for numerical stability
        if nom.ndim == 1:
            result = inv(denom_reg) @ nom
        elif nom.ndim == 2 and nom.shape[0] == 1:
            result = inv(denom_reg) @ nom.T
        else:
            result = inv(denom_reg) @ nom.flatten()
        return result, True
    except NUMERICAL_EXCEPTIONS:
        return np.zeros(denom.shape[0]), False


def update_loadings(
    C_new: np.ndarray,
    C_update: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray
) -> None:
    """Update loading matrix from computed values.
    
    Parameters
    ----------
    C_new : np.ndarray
        Loading matrix to update (modified in-place)
    C_update : np.ndarray
        Computed loading values (n_rows x n_cols)
    row_indices : np.ndarray
        Row indices in C_new to update
    col_indices : np.ndarray
        Column indices in C_new to update
        
    Notes
    -----
    - Updates C_new in-place
    - Assumes C_update.shape == (len(row_indices), len(col_indices))
    - More efficient than nested loops for large matrices
    """
    if len(row_indices) == 0 or len(col_indices) == 0:
        return
    
    # Use vectorized assignment for efficiency
    if C_update.shape == (len(row_indices), len(col_indices)):
        C_new[np.ix_(row_indices, col_indices)] = C_update
    else:
        # Fallback to element-wise assignment if shapes don't match
        for ii, row_idx in enumerate(row_indices):
            for jj, col_idx in enumerate(col_indices):
                if ii < C_update.shape[0] and jj < C_update.shape[1]:
                    C_new[row_idx, col_idx] = C_update[ii, jj]


def extract_3d_matrix_slice(
    matrix_3d: np.ndarray,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    time_idx: int
) -> np.ndarray:
    """Extract 2D slice from 3D matrix with safe indexing.
    
    Parameters
    ----------
    matrix_3d : np.ndarray
        3D matrix (m x m x T) to extract from
    row_indices : np.ndarray
        Row indices to extract
    col_indices : np.ndarray
        Column indices to extract
    time_idx : int
        Time index (third dimension)
        
    Returns
    -------
    slice_2d : np.ndarray
        2D slice (len(row_indices) x len(col_indices))
        Returns zeros if indices are invalid
        
    Notes
    -----
    - Handles dimension reduction (3D -> 2D)
    - Returns zeros if time_idx is out of bounds
    - Validates that extracted slice has expected shape
    """
    if time_idx >= matrix_3d.shape[2]:
        return np.zeros((len(row_indices), len(col_indices)))
    
    try:
        slice_3d = matrix_3d[np.ix_(row_indices, col_indices, [time_idx])]
        slice_2d = slice_3d[:, :, 0] if slice_3d.ndim == 3 else slice_3d
        # Ensure correct shape
        expected_shape = (len(row_indices), len(col_indices))
        if slice_2d.shape != expected_shape:
            return np.zeros(expected_shape)
        return slice_2d
    except (IndexError, ValueError):
        return np.zeros((len(row_indices), len(col_indices)))


def compute_obs_cov(
    y: np.ndarray,
    C: np.ndarray,
    Zsmooth: np.ndarray,
    vsmooth: np.ndarray,
    default_variance: float = 1e-4,
    min_variance: float = 1e-8,
    min_diagonal_variance_ratio: float = 1e-6
) -> np.ndarray:
    """Compute observation covariance diagonal (R_diag) from residuals and factor uncertainty.
    
    Computes: R[i,i] = mean_t((y[i,t] - C[i,:] @ Z[t])^2 + C[i,:] @ V[t] @ C[i,:]')
    
    Parameters
    ----------
    y : np.ndarray
        Observation matrix (n x T), may contain NaN
    C : np.ndarray
        Loading matrix (n x m)
    Zsmooth : np.ndarray
        Smoothed factor estimates (m x (T+1))
    vsmooth : np.ndarray
        Smoothed factor covariances (m x m x (T+1))
    default_variance : float, default 1e-4
        Default variance if computation fails
    min_variance : float, default 1e-8
        Minimum variance to enforce
    min_diagonal_variance_ratio : float, default 1e-6
        Minimum variance as ratio of mean variance
        
    Returns
    -------
    R_diag : np.ndarray
        Diagonal elements of R (n,)
        
    Notes
    -----
    - Handles missing data (NaN) by skipping those time points
    - Computes both residual variance and factor uncertainty contribution
    - Enforces minimum variance for numerical stability
    - Uses median imputation for invalid values
    """
    n, T = y.shape
    R_diag = np.zeros(n)
    n_obs_per_series = np.zeros(n, dtype=int)
    
    for t in range(T):
        if not safe_time_index(t, Zsmooth.shape[0], offset=1):
            continue
        Z_t = Zsmooth[t + 1, :].reshape(-1, 1)
        vsmooth_t = vsmooth[:, :, t + 1]
        y_pred = (C @ Z_t).flatten()
        
        for i in range(n):
            if np.isnan(y[i, t]):
                continue
            n_obs_per_series[i] += 1
            resid_sq = (y[i, t] - y_pred[i]) ** 2
            C_i = C[i, :].reshape(1, -1)
            var_factor = (C_i @ vsmooth_t @ C_i.T)[0, 0]
            R_diag[i] += resid_sq + var_factor
    
    # Normalize by number of observations per series
    n_obs_per_series = np.maximum(n_obs_per_series, 1)
    R_diag = R_diag / n_obs_per_series
    
    # Enforce minimum variance
    mean_var = np.mean(R_diag[R_diag > 0]) if np.any(R_diag > 0) else default_variance
    min_var = np.maximum(mean_var * min_diagonal_variance_ratio, min_variance)
    R_diag = np.maximum(R_diag, min_var)
    
    # Handle invalid values (NaN/Inf/negative)
    valid_mask = np.isfinite(R_diag) & (R_diag > 0)
    if np.any(valid_mask):
        median_var = np.median(R_diag[valid_mask])
        R_diag = np.where(valid_mask, R_diag, median_var)
    else:
        R_diag.fill(default_variance)
    
    return R_diag


def clean_variance_array(
    variance_array: np.ndarray,
    default_value: float = 1e-4,
    min_value: Optional[float] = None,
    replace_nan: bool = True,
    replace_inf: bool = True,
    replace_negative: bool = True
) -> np.ndarray:
    """Clean variance array by replacing invalid values.
    
    Parameters
    ----------
    variance_array : np.ndarray
        Array of variance values to clean
    default_value : float, default 1e-4
        Default value to use for invalid entries
    min_value : float, optional
        Minimum value to enforce. If None, uses default_value.
    replace_nan : bool, default True
        Replace NaN values
    replace_inf : bool, default True
        Replace Inf values
    replace_negative : bool, default True
        Replace negative values
        
    Returns
    -------
    cleaned_array : np.ndarray
        Cleaned variance array with all invalid values replaced
        
    Notes
    -----
    - Uses median imputation if any valid values exist
    - Falls back to default_value if no valid values
    - Enforces minimum value after cleaning
    """
    cleaned = variance_array.copy()
    
    # Build mask for invalid values
    invalid_mask = np.zeros(cleaned.shape, dtype=bool)
    if replace_nan:
        invalid_mask |= np.isnan(cleaned)
    if replace_inf:
        invalid_mask |= np.isinf(cleaned)
    if replace_negative:
        invalid_mask |= (cleaned < 0)
    
    # Replace invalid values
    if np.any(invalid_mask):
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            # Use median of valid values
            median_val = np.median(cleaned[valid_mask])
            cleaned = np.where(invalid_mask, median_val, cleaned)
        else:
            # All invalid - use default
            cleaned[invalid_mask] = default_value
    
    # Enforce minimum value
    min_val = min_value if min_value is not None else default_value
    cleaned = np.maximum(cleaned, min_val)
    
    return cleaned


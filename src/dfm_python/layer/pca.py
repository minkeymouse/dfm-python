"""Principal Component Analysis (PCA) utilities.

This module provides pure PCA functions for factor extraction.
Similar to MLP, this is a utility layer without class inheritance.
"""

import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ..logger import get_logger
from ..numeric.stability import create_scaled_identity
from ..config.constants import DEFAULT_IDENTITY_SCALE
from ..config.types import to_numpy

_logger = get_logger(__name__)


def compute_principal_components(
    cov_matrix: Union[np.ndarray, "torch.Tensor"],
    n_components: int,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition (NumPy-based).
    
    Accepts both NumPy arrays and PyTorch tensors, but performs all
    computations in NumPy for consistency with the refactored codebase.
    
    Parameters
    ----------
    cov_matrix : np.ndarray or torch.Tensor
        Covariance matrix (N x N)
    n_components : int
        Number of principal components to extract
    block_idx : int, optional
        Block index for error messages
        
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (n_components,)
    eigenvectors : np.ndarray
        Eigenvectors (N x n_components)
    """
    # Convert to NumPy if needed
    cov_matrix = to_numpy(cov_matrix)
    
    if cov_matrix.size == 1:
        eigenvector = np.array([[DEFAULT_IDENTITY_SCALE]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else DEFAULT_IDENTITY_SCALE
        return np.array([eigenvalue]), eigenvector
    
    n_series = cov_matrix.shape[0]
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by absolute value, descending
        sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
        eigenvalues_sorted = eigenvalues[sort_idx]
        eigenvectors_sorted = eigenvectors[:, sort_idx]
        return np.real(eigenvalues_sorted), np.real(eigenvectors_sorted)
    except (ValueError, np.linalg.LinAlgError) as e:
        if block_idx is not None:
            _logger.warning(
                f"PCA: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        else:
            _logger.warning(
                f"PCA: Eigendecomposition failed, using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = create_scaled_identity(n_series, DEFAULT_IDENTITY_SCALE)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def fit_pca(
    X: Union[np.ndarray, "torch.Tensor"],
    n_components: int,
    cov_matrix: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit PCA by computing principal components.
    
    If cov_matrix is provided, uses eigendecomposition.
    If X is provided, uses NumPy's SVD for efficiency.
    All computations are performed in NumPy.
    
    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Training data (T x N). If cov_matrix is provided, this is ignored.
    n_components : int
        Number of principal components to extract
    cov_matrix : np.ndarray or torch.Tensor, optional
        Precomputed covariance matrix (N x N). If None, computed from X.
    block_idx : int, optional
        Block index for error messages
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        (eigenvalues, eigenvectors, mean, cov_matrix)
    """
    if cov_matrix is not None:
        # Use eigendecomposition for covariance matrix
        eigenvalues, eigenvectors = compute_principal_components(
            cov_matrix, n_components, block_idx=block_idx
        )
        # Store as NumPy array
        cov_matrix_np = to_numpy(cov_matrix)
        return eigenvalues, eigenvectors, None, cov_matrix_np
    else:
        # Convert to NumPy if needed
        X = to_numpy(X)
        
        # Center the data
        mean_ = np.mean(X, axis=0, keepdims=True)
        X_centered = X - mean_
        
        # Use SVD for efficient low-rank PCA (NumPy equivalent of torch.pca_lowrank)
        try:
            # Check for NaN/Inf values before SVD
            if np.any(~np.isfinite(X_centered)):
                raise ValueError("Input contains non-finite values")
            
            # Check for zero variance columns (can cause SVD issues)
            col_stds = np.std(X_centered, axis=0)
            if np.any(col_stds < 1e-10):
                _logger.warning(
                    f"PCA: Some columns have near-zero variance (min_std={col_stds.min():.2e}). "
                    "This may cause SVD convergence issues. Consider removing constant columns."
                )
            
            # Use truncated SVD for efficiency
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False, hermitian=False)
            
            # Validate SVD results
            if not np.all(np.isfinite(S)):
                raise ValueError("SVD produced non-finite singular values")
            
            # Take top n_components
            n_components_actual = min(n_components, len(S))
            if n_components_actual < n_components:
                _logger.warning(
                    f"PCA: Requested {n_components} components but only {len(S)} available. "
                    f"Using {n_components_actual} components."
                )
            
            U = U[:, :n_components_actual]
            S = S[:n_components_actual]
            Vt = Vt[:n_components_actual, :]
            
            # Vt contains the principal components (eigenvectors) as rows
            # Transpose to get (N x n_components)
            eigenvectors = Vt.T
            # Convert singular values to eigenvalues
            eigenvalues = S ** 2
            
            return eigenvalues, eigenvectors, mean_, None
            
        except (ValueError, np.linalg.LinAlgError) as e:
            # Fallback: compute covariance and use eigendecomposition
            error_msg = str(e) if str(e) else type(e).__name__
            _logger.warning(
                f"PCA SVD failed ({error_msg}), falling back to eigendecomposition. "
                "This may be due to numerical instability or ill-conditioned data."
            )
            
            # Clean data for covariance computation
            X_centered_clean = np.nan_to_num(X_centered, nan=0.0, posinf=0.0, neginf=0.0)
            T = X_centered_clean.shape[0]
            
            # Compute covariance with regularization for stability
            cov_raw = (X_centered_clean.T @ X_centered_clean) / max(T - 1, 1)
            
            # Add small regularization to diagonal for numerical stability
            reg = 1e-8 * np.eye(cov_raw.shape[0], dtype=cov_raw.dtype)
            cov_matrix_np = cov_raw + reg
            
            try:
                eigenvalues, eigenvectors = compute_principal_components(
                    cov_matrix_np, n_components, block_idx=block_idx
                )
                return eigenvalues, eigenvectors, mean_, cov_matrix_np
            except (ValueError, np.linalg.LinAlgError) as e2:
                # If eigendecomposition also fails, raise with helpful message
                raise RuntimeError(
                    f"PCA failed: Both SVD and eigendecomposition failed. "
                    f"Original error: {error_msg}, Eigendecomposition error: {str(e2)}. "
                    "This may indicate severely ill-conditioned data. "
                    "Consider: (1) Removing constant/near-constant columns, "
                    "(2) Checking for outliers, (3) Using fewer components."
                ) from e2


def encode_pca(
    X: Union[np.ndarray, "torch.Tensor"],
    eigenvectors: np.ndarray,
    mean: Optional[np.ndarray] = None
) -> np.ndarray:
    """Extract factors using PCA eigenvectors.
    
    All computations are performed in NumPy. Returns NumPy array.
    
    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Observed data (T x N)
    eigenvectors : np.ndarray
        PCA eigenvectors (N x n_components) from fit_pca
    mean : Optional[np.ndarray]
        Mean vector (1 x N) from fit_pca. If None, centers using X's mean.
        
    Returns
    -------
    factors : np.ndarray
        Extracted factors (T x n_components)
    """
    # Convert to NumPy if needed
    X = to_numpy(X)
    
    # Center the data
    if mean is not None:
        X_centered = X - mean
    else:
        X_centered = X - np.mean(X, axis=0, keepdims=True)
    
    # Project: X @ eigenvectors
    factors = X_centered @ eigenvectors
    
    return factors


# Backward compatibility wrapper class (does not inherit from BaseEncoder)
class PCAEncoder:
    """Principal Component Analysis encoder for factor extraction (backward compatibility wrapper).
    
    This is a wrapper around pure PCA functions for backward compatibility.
    The actual implementation uses pure functions in this module (layer.pca).
    
    Parameters
    ----------
    n_components : int
        Number of factors to extract
    block_idx : int, optional
        Block index for error messages
    """
    
    def __init__(
        self,
        n_components: int,
        block_idx: Optional[int] = None
    ):
        self.n_components = n_components
        self.block_idx = block_idx
        
        # Will be set in fit()
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        cov_matrix: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        **kwargs
    ) -> "PCAEncoder":
        """Fit PCA encoder by computing principal components."""
        eigenvalues, eigenvectors, mean, cov_matrix_np = fit_pca(
            X, self.n_components, cov_matrix=cov_matrix, block_idx=self.block_idx
        )
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.mean_ = mean
        self.cov_matrix = cov_matrix_np
        return self
    
    def encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> np.ndarray:
        """Extract factors using fitted PCA encoder."""
        if self.eigenvectors is None:
            from ..utils.errors import ModelNotTrainedError
            raise ModelNotTrainedError(
                "PCAEncoder must be fitted before encoding. Call fit() first.",
                details="The encoder has not been fitted with training data yet."
            )
        return encode_pca(X, self.eigenvectors, mean=self.mean_)
    
    def fit_encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> np.ndarray:
        """Fit encoder and extract factors in one step."""
        self.fit(X, **kwargs)
        return self.encode(X, **kwargs)

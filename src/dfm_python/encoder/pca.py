"""Principal Component Analysis (PCA) for factor extraction.

This module provides NumPy-based PCA implementations for efficient
low-rank PCA computation using NumPy's SVD and eigendecomposition.
"""

import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING
from .base import BaseEncoder
from ..logger import get_logger
from ..numeric.stability import create_scaled_identity
import torch
from ..utils.errors import ModelNotTrainedError
from ..config.constants import DEFAULT_IDENTITY_SCALE
from ..config.types import to_numpy

if TYPE_CHECKING:
    import torch

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




class PCAEncoder(BaseEncoder):
    """Principal Component Analysis encoder for factor extraction.
    
    This encoder extracts factors using NumPy-based SVD for efficient computation
    when working with data matrices, or eigendecomposition when working with
    covariance matrices. All computations are performed in NumPy for consistency
    with the refactored codebase.
    
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
        super().__init__(n_components)
        self.block_idx = block_idx
        
        # Will be set in fit() - all stored as NumPy arrays
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
        """Fit PCA encoder by computing principal components.
        
        If cov_matrix is provided, uses eigendecomposition.
        If X is provided, uses NumPy's SVD for efficiency (similar to torch.pca_lowrank).
        All computations are performed in NumPy.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Training data (T x N). If cov_matrix is provided, this is ignored.
        cov_matrix : np.ndarray or torch.Tensor, optional
            Precomputed covariance matrix (N x N). If None, computed from X.
        **kwargs
            Additional parameters (ignored)
            
        Returns
        -------
        self : PCAEncoder
            Returns self for method chaining
        """
        if cov_matrix is not None:
            # Use eigendecomposition for covariance matrix
            self.eigenvalues, self.eigenvectors = compute_principal_components(
                cov_matrix, self.n_components, block_idx=self.block_idx
            )
            # Store as NumPy array
            self.cov_matrix = to_numpy(cov_matrix)
        else:
            # Convert to NumPy if needed
            X = to_numpy(X)
            
            # Center the data
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            X_centered = X - self.mean_
            
            # Use SVD for efficient low-rank PCA (NumPy equivalent of torch.pca_lowrank)
            try:
                # Use truncated SVD for efficiency
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                # Take top n_components
                U = U[:, :self.n_components]
                S = S[:self.n_components]
                Vt = Vt[:self.n_components, :]
                
                # Vt contains the principal components (eigenvectors) as rows
                # Transpose to get (N x n_components)
                self.eigenvectors = Vt.T
                # Convert singular values to eigenvalues
                self.eigenvalues = S ** 2
            except (ValueError, np.linalg.LinAlgError) as e:
                # Fallback: compute covariance and use eigendecomposition
                _logger.warning(f"PCA SVD failed, falling back to eigendecomposition. Error: {type(e).__name__}")
                T = X_centered.shape[0]
                self.cov_matrix = (X_centered.T @ X_centered) / (T - 1)
                self.eigenvalues, self.eigenvectors = compute_principal_components(
                    self.cov_matrix, self.n_components, block_idx=self.block_idx
                )
        
        return self
    
    def encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> np.ndarray:
        """Extract factors using fitted PCA encoder.
        
        All computations are performed in NumPy. Returns NumPy array.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Observed data (T x N)
        **kwargs
            Additional parameters (ignored)
            
        Returns
        -------
        factors : np.ndarray
            Extracted factors (T x n_components)
        """
        if self.eigenvectors is None:
            raise ModelNotTrainedError(
                "PCAEncoder must be fitted before encoding. Call fit() first.",
                details="The encoder has not been fitted with training data yet."
            )
        
        # Convert to NumPy if needed
        X = to_numpy(X)
        
        # Center the data
        if self.mean_ is not None:
            X_centered = X - self.mean_
        else:
            X_centered = X - np.mean(X, axis=0, keepdims=True)
        
        # Project: X @ eigenvectors
        factors = X_centered @ self.eigenvectors
        
        return factors

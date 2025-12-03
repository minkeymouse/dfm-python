"""Principal Component Analysis (PCA) for factor extraction.

This module provides both NumPy and PyTorch implementations of PCA for
initializing factor models via eigendecomposition.
"""

import numpy as np
from typing import Tuple, Optional, Union, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    import torch
else:
    torch = None

try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None

try:
    from scipy.sparse.linalg import eigs
    from scipy.sparse import csc_matrix
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    eigs = None
    csc_matrix = None

from .base import BaseEncoder
from ..logger import get_logger

_logger = get_logger(__name__)

# Default fallback values
DEFAULT_VARIANCE_FALLBACK = 1.0


def compute_principal_components(
    cov_matrix: np.ndarray,
    n_components: int,
    block_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition with fallbacks.
    
    NumPy implementation for factor initialization in DFM.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
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
    if cov_matrix.size == 1:
        eigenvector = np.array([[1.0]])
        eigenvalue = cov_matrix[0, 0] if np.isfinite(cov_matrix[0, 0]) else DEFAULT_VARIANCE_FALLBACK
        return np.array([eigenvalue]), eigenvector
    
    n_series = cov_matrix.shape[0]
    
    # Strategy 1: Sparse eigs when feasible
    if n_components < n_series - 1 and SCIPY_SPARSE_AVAILABLE:
        try:
            cov_sparse = csc_matrix(cov_matrix)
            eigenvalues, eigenvectors = eigs(cov_sparse, k=n_components, which='LM')
            eigenvectors = eigenvectors.real
            if np.any(~np.isfinite(eigenvalues)) or np.any(~np.isfinite(eigenvectors)):
                raise ValueError("Invalid eigenvalue results")
            return eigenvalues.real, eigenvectors
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
            if block_idx is not None:
                _logger.warning(
                    f"PCA: Sparse eigendecomposition failed for block {block_idx+1}, "
                    f"falling back to np.linalg.eig. Error: {type(e).__name__}"
                )
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            sort_idx = np.argsort(np.abs(eigenvalues))[::-1][:n_components]
            return eigenvalues[sort_idx].real, eigenvectors[:, sort_idx].real
    
    # Strategy 2: Full eig
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        valid_mask = np.isfinite(eigenvalues)
        if np.sum(valid_mask) < n_components:
            raise ValueError("Not enough valid eigenvalues")
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask]
        sort_idx = np.argsort(np.abs(valid_eigenvalues))[::-1][:n_components]
        return valid_eigenvalues[sort_idx].real, valid_eigenvectors[:, sort_idx].real
    except (IndexError, ValueError, np.linalg.LinAlgError) as e:
        if block_idx is not None:
            _logger.warning(
                f"PCA: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = np.eye(n_series)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def compute_principal_components_torch(
    cov_matrix: "torch.Tensor",
    n_components: int
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Compute top principal components via eigendecomposition (PyTorch).
    
    PyTorch implementation for factor initialization in DDFM.
    
    Parameters
    ----------
    cov_matrix : torch.Tensor
        Covariance matrix (N x N)
    n_components : int
        Number of principal components to extract
        
    Returns
    -------
    eigenvalues : torch.Tensor
        Eigenvalues (n_components,)
    eigenvectors : torch.Tensor
        Eigenvectors (N x n_components)
    """
    if not _has_torch:
        raise ImportError("PyTorch is required for compute_principal_components_torch")
    
    if cov_matrix.numel() == 1:
        eigenvector = torch.tensor([[1.0]], device=cov_matrix.device, dtype=cov_matrix.dtype)
        eigenvalue = cov_matrix[0, 0] if torch.isfinite(cov_matrix[0, 0]) else 1.0
        return torch.tensor([eigenvalue], device=cov_matrix.device, dtype=cov_matrix.dtype), eigenvector
    
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        # Sort by absolute value, descending
        sort_idx = torch.argsort(torch.abs(eigenvalues), descending=True)[:n_components]
        return eigenvalues[sort_idx].real, eigenvectors[:, sort_idx].real
    except RuntimeError:
        # Fallback: use identity matrix
        n_series = cov_matrix.shape[0]
        eigenvectors = torch.eye(n_series, device=cov_matrix.device, dtype=cov_matrix.dtype)[:, :n_components]
        eigenvalues = torch.ones(n_components, device=cov_matrix.device, dtype=cov_matrix.dtype)
        return eigenvalues, eigenvectors


class PCAEncoder(BaseEncoder):
    """Principal Component Analysis encoder for factor extraction.
    
    This encoder extracts factors via eigendecomposition of the covariance matrix.
    It supports both NumPy and PyTorch backends.
    
    Parameters
    ----------
    n_components : int
        Number of factors to extract
    use_torch : bool, default False
        Whether to use PyTorch backend (True) or NumPy backend (False)
    block_idx : int, optional
        Block index for error messages
    """
    
    def __init__(
        self,
        n_components: int,
        use_torch: bool = False,
        block_idx: Optional[int] = None
    ):
        super().__init__(n_components)
        self.use_torch = use_torch
        self.block_idx = block_idx
        
        if use_torch and not _has_torch:
            raise ImportError("PyTorch is required for use_torch=True")
        
        # Will be set in fit()
        self.eigenvectors: Optional[Union[np.ndarray, "torch.Tensor"]] = None
        self.eigenvalues: Optional[Union[np.ndarray, "torch.Tensor"]] = None
        self.cov_matrix: Optional[Union[np.ndarray, "torch.Tensor"]] = None
    
    def fit(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        cov_matrix: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
        **kwargs
    ) -> "PCAEncoder":
        """Fit PCA encoder by computing covariance matrix and eigendecomposition.
        
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
            self.cov_matrix = cov_matrix
        else:
            # Compute covariance matrix
            if self.use_torch:
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                # Center the data
                X_centered = X - X.mean(dim=0, keepdim=True)
                # Compute covariance: (1/(T-1)) * X^T @ X
                T = X_centered.shape[0]
                self.cov_matrix = (X_centered.T @ X_centered) / (T - 1)
            else:
                if isinstance(X, torch.Tensor):
                    X = X.cpu().numpy()
                # Center the data
                X_mean = np.mean(X, axis=0, keepdims=True)
                X_centered = X - X_mean
                # Compute covariance
                self.cov_matrix = np.cov(X_centered.T)
        
        # Compute principal components
        if self.use_torch:
            self.eigenvalues, self.eigenvectors = compute_principal_components_torch(
                self.cov_matrix,
                self.n_components
            )
        else:
            if isinstance(self.cov_matrix, torch.Tensor):
                self.cov_matrix = self.cov_matrix.cpu().numpy()
            self.eigenvalues, self.eigenvectors = compute_principal_components(
                self.cov_matrix,
                self.n_components,
                block_idx=self.block_idx
            )
        
        return self
    
    def encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Extract factors using fitted PCA encoder.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Observed data (T x N)
        **kwargs
            Additional parameters (ignored)
            
        Returns
        -------
        factors : np.ndarray or torch.Tensor
            Extracted factors (T x n_components)
        """
        if self.eigenvectors is None:
            raise RuntimeError("PCAEncoder must be fitted before encoding. Call fit() first.")
        
        # Project data onto principal components
        if self.use_torch:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(self.eigenvectors, torch.Tensor):
                self.eigenvectors = torch.tensor(self.eigenvectors, dtype=X.dtype, device=X.device)
            # Center the data
            X_centered = X - X.mean(dim=0, keepdim=True)
            # Project: X @ eigenvectors
            factors = X_centered @ self.eigenvectors
        else:
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            if isinstance(self.eigenvectors, torch.Tensor):
                self.eigenvectors = self.eigenvectors.cpu().numpy()
            # Center the data
            X_mean = np.mean(X, axis=0, keepdims=True)
            X_centered = X - X_mean
            # Project: X @ eigenvectors
            factors = X_centered @ self.eigenvectors
        
        return factors




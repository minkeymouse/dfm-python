"""Matrix operations for ensuring matrix properties (symmetric, real, square)."""

from typing import Optional, Tuple
import logging
import numpy as np

_logger = logging.getLogger(__name__)

# Numerical stability constants
MIN_EIGENVAL_CLEAN = 1e-8  # Minimum eigenvalue for matrix cleaning operations
MIN_DIAGONAL_VARIANCE = 1e-6  # Minimum diagonal variance for diagonal matrix cleaning


def _ensure_square_matrix(M: np.ndarray, method: str = 'diag') -> np.ndarray:
    """Ensure matrix is square by extracting diagonal if needed.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to ensure is square
    method : str
        Method to use if matrix is not square:
        - 'diag': Extract diagonal (default)
        - 'eye': Return identity matrix of appropriate size
        
    Returns
    -------
    np.ndarray
        Square matrix (n x n)
        
    Notes
    -----
    - Used when a square matrix is required but input may be non-square
    - 'diag' method preserves diagonal elements, discards off-diagonal
    - 'eye' method creates identity matrix of size max(rows, cols)
    - Empty matrices are returned unchanged
    """
    if M.size == 0:
        return M
    if M.shape[0] != M.shape[1]:
        if method == 'diag':
            return np.diag(np.diag(M))
        elif method == 'eye':
            size = max(M.shape[0], M.shape[1])
            return np.eye(size)
    return M


def _ensure_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric by averaging with its transpose.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to symmetrize
        
    Returns
    -------
    np.ndarray
        Symmetric matrix (M + M.T) / 2
        
    Notes
    -----
    - Used for covariance matrices that should be symmetric but may have
      numerical asymmetry due to floating-point errors
    - Formula: (M + M.T) / 2 ensures symmetry while preserving diagonal elements
    - Commonly used in Kalman filtering and EM algorithm for covariance matrices
    """
    return 0.5 * (M + M.T)


def _ensure_real(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real by extracting real part if complex.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix that may be complex due to numerical errors
        
    Returns
    -------
    np.ndarray
        Real matrix (extracts real part if complex)
        
    Notes
    -----
    - Used when numerical errors introduce tiny imaginary parts (typically < 1e-15)
    - Covariance matrices should be real; this function ensures they remain real
    - Only extracts real part if matrix is actually complex; otherwise returns unchanged
    - Commonly used in Kalman filtering and eigenvalue computations
    """
    if np.iscomplexobj(M):
        return np.real(M)
    return M


def _ensure_real_and_symmetric(M: np.ndarray) -> np.ndarray:
    """Ensure matrix is real and symmetric.
    
    This is a common operation in Kalman filtering where covariance matrices
    should be real and symmetric, but numerical errors can introduce complex
    values or asymmetry. This function applies both transformations in sequence.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to process (may be complex or asymmetric due to numerical errors)
        
    Returns
    -------
    np.ndarray
        Real, symmetric matrix (M_real + M_real.T) / 2
        
    Notes
    -----
    - First extracts real part if matrix is complex
    - Then symmetrizes by averaging with transpose
    - This is a convenience function combining _ensure_real and _ensure_symmetric
    - Used extensively in Kalman filter/smoother for covariance stability
    """
    M = _ensure_real(M)
    M = _ensure_symmetric(M)
    return M


def _ensure_covariance_stable(M: np.ndarray, min_eigenval: float = 1e-8,
                               ensure_real: bool = True) -> np.ndarray:
    """Ensure covariance matrix is real, symmetric, and positive semi-definite.
    
    This is a comprehensive function for stabilizing covariance matrices in
    numerical computations, commonly used in Kalman filtering. It consolidates
    real extraction, symmetrization, and PSD regularization into one operation.
    
    Parameters
    ----------
    M : np.ndarray
        Covariance matrix to stabilize
    min_eigenval : float
        Minimum eigenvalue threshold for positive semi-definiteness
    ensure_real : bool
        If True, extract real part if matrix is complex
        
    Returns
    -------
    np.ndarray
        Stabilized covariance matrix (real, symmetric, PSD)
        
    Notes
    -----
    - Uses `_ensure_positive_definite` internally for PSD regularization
    - Suppresses warnings from `_ensure_positive_definite` to avoid noise
    - This function is used in Kalman filter/smoother for covariance stability
    """
    if M.size == 0 or M.shape[0] == 0:
        return M
    
    # Step 1: Ensure real (if needed)
    if ensure_real:
        M = _ensure_real(M)
    
    # Step 2: Ensure symmetric and positive semi-definite
    # Import here to avoid circular dependency
    from .regularization import _ensure_positive_definite
    M, _ = _ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
    
    return M


def _compute_principal_components(cov_matrix: np.ndarray, n_components: int,
                                   block_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top principal components via eigendecomposition with fallbacks.
    
    This function computes the top n_components principal components from a covariance
    matrix using eigendecomposition. It uses multiple strategies for robustness:
    1. Sparse eigendecomposition (scipy.sparse.linalg.eigs) when feasible
    2. Full eigendecomposition (np.linalg.eig) as fallback
    3. Identity matrix fallback if all strategies fail
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (n x n) to decompose
    n_components : int
        Number of principal components to extract (top eigenvalues/eigenvectors)
    block_idx : int, optional
        Block index for logging purposes. If provided, warnings include block number.
        
    Returns
    -------
    eigenvalues : np.ndarray
        Top n_components eigenvalues, sorted by absolute value (descending)
        Shape: (n_components,)
    eigenvectors : np.ndarray
        Corresponding eigenvectors, shape (n x n_components)
        Each column is an eigenvector corresponding to the eigenvalue at same index
        
    Notes
    -----
    - Eigenvalues are sorted by absolute value in descending order
    - All outputs are guaranteed to be real (complex parts discarded)
    - Used in init_conditions for PCA-based factor initialization
    - Falls back to identity matrix if eigendecomposition fails completely
    - For single-element matrices, returns the value as eigenvalue with [1.0] eigenvector
    """
    try:
        from scipy.sparse.linalg import eigs
        from scipy.sparse import csc_matrix
        SCIPY_SPARSE_AVAILABLE = True
    except ImportError:
        SCIPY_SPARSE_AVAILABLE = False
        eigs = None
        csc_matrix = None
    
    # Import here to avoid circular dependency
    try:
        from .covariance import DEFAULT_VARIANCE_FALLBACK
    except ImportError:
        # Fallback if covariance module not yet imported
        DEFAULT_VARIANCE_FALLBACK = 1.0
    
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
                    f"init_conditions: Sparse eigendecomposition failed for block {block_idx+1}, "
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
                f"init_conditions: Eigendecomposition failed for block {block_idx+1}, "
                f"using identity matrix as fallback. Error: {type(e).__name__}"
            )
        eigenvectors = np.eye(n_series)[:, :n_components]
        eigenvalues = np.ones(n_components)
        return eigenvalues, eigenvectors


def _clean_matrix(M: np.ndarray, matrix_type: str = 'general', 
                  default_nan: float = 0.0, default_inf: Optional[float] = None) -> np.ndarray:
    """Clean matrix by removing NaN/Inf values and ensuring numerical stability.
    
    This function handles different matrix types with appropriate cleaning strategies:
    - 'covariance': Ensures symmetry and positive semi-definiteness
    - 'diagonal': Cleans diagonal elements and enforces minimum variance
    - 'loading': Removes non-finite values from loading matrices
    - 'general': Basic cleaning for any matrix type
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to clean (may contain NaN/Inf values)
    matrix_type : str, default 'general'
        Type of matrix, determines cleaning strategy:
        - 'covariance': Apply PSD regularization
        - 'diagonal': Clean diagonal and enforce minimum variance
        - 'loading': Replace Inf with bounded values
        - 'general': Basic NaN/Inf replacement
    default_nan : float, default 0.0
        Value to replace NaN with
    default_inf : float, optional
        Value to replace Inf with. If None, uses type-specific defaults:
        - 'covariance': 1e6 for posinf, -1e6 for neginf
        - 'diagonal': 1e4 for posinf, default_nan for neginf
        - 'loading': 1.0 for posinf, -1.0 for neginf
        - 'general': 1e6 for posinf, -1e6 for neginf
        
    Returns
    -------
    np.ndarray
        Cleaned matrix with all non-finite values replaced and stability enforced
        
    Notes
    -----
    - For covariance matrices, also ensures symmetry and minimum eigenvalue
    - For diagonal matrices, enforces MIN_DIAGONAL_VARIANCE on diagonal elements
    - For loading matrices, clips Inf values to prevent numerical overflow
    - Used extensively in EM algorithm for parameter cleaning
    """
    from typing import Optional
    
    if matrix_type == 'covariance':
        M = np.nan_to_num(M, nan=default_nan, posinf=1e6, neginf=-1e6)
        M = _ensure_symmetric(M)
        try:
            eigenvals = np.linalg.eigvals(M)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < MIN_EIGENVAL_CLEAN:
                M = M + np.eye(M.shape[0]) * (MIN_EIGENVAL_CLEAN - min_eigenval)
                M = _ensure_symmetric(M)
        except (np.linalg.LinAlgError, ValueError):
            M = M + np.eye(M.shape[0]) * MIN_EIGENVAL_CLEAN
            M = _ensure_symmetric(M)
    elif matrix_type == 'diagonal':
        diag = np.diag(M)
        diag = np.nan_to_num(diag, nan=default_nan, 
                            posinf=default_inf if default_inf is not None else 1e4,
                            neginf=default_nan)
        diag = np.maximum(diag, MIN_DIAGONAL_VARIANCE)
        M = np.diag(diag)
    elif matrix_type == 'loading':
        M = np.nan_to_num(M, nan=default_nan, posinf=1.0, neginf=-1.0)
    else:
        default_inf_val = default_inf if default_inf is not None else 1e6
        M = np.nan_to_num(M, nan=default_nan, posinf=default_inf_val, neginf=-default_inf_val)
    return M

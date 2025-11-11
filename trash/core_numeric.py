from typing import Optional, Tuple, Dict, Any
import logging
import warnings
import numpy as np

try:
    from scipy.sparse.linalg import eigs
    from scipy.sparse import csc_matrix
    SCIPY_SPARSE_AVAILABLE = True
except ImportError:
    SCIPY_SPARSE_AVAILABLE = False
    eigs = None
    csc_matrix = None

_logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency (helpers imports are safe)
def _get_helpers():
    """Lazy import helper functions."""
    from .helpers import safe_get_attr
    return safe_get_attr

# Numerical stability constants for numeric utilities
DEFAULT_VARIANCE_FALLBACK = 1.0  # Default variance when computation fails or result is invalid
MIN_VARIANCE_COVARIANCE = 1e-10  # Minimum variance threshold for covariance matrix diagonal
MIN_EIGENVAL_CLEAN = 1e-8  # Minimum eigenvalue for matrix cleaning operations
MIN_DIAGONAL_VARIANCE = 1e-6  # Minimum diagonal variance for diagonal matrix cleaning


def _check_finite(array: np.ndarray, name: str = "array", raise_on_invalid: bool = False) -> bool:
    """Check if array contains only finite values.
    
    Parameters
    ----------
    array : np.ndarray
        Array to check
    name : str
        Name of array for logging
    raise_on_invalid : bool
        If True, raise ValueError on invalid values. If False, only log warning.
        
    Returns
    -------
    bool
        True if all values are finite, False otherwise
        
    Raises
    ------
    ValueError
        If raise_on_invalid=True and array contains non-finite values
        
    Notes
    -----
    - Checks for both NaN and Inf values
    - Provides detailed error messages with counts of NaN/Inf values
    - Used throughout the package for input validation and debugging
    - When raise_on_invalid=False, logs warnings but allows execution to continue
    """
    has_nan = np.any(np.isnan(array))
    has_inf = np.any(np.isinf(array))
    
    if has_nan or has_inf:
        msg = f"{name} contains "
        issues = []
        if has_nan:
            issues.append(f"{np.sum(np.isnan(array))} NaN values")
        if has_inf:
            issues.append(f"{np.sum(np.isinf(array))} Inf values")
        msg += " and ".join(issues)
        
        if raise_on_invalid:
            raise ValueError(msg)
        else:
            _logger.warning(msg)
        return False
    return True


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
    # Use _ensure_positive_definite which handles both symmetrization and PSD
    M, _ = _ensure_positive_definite(M, min_eigenval=min_eigenval, warn=False)
    
    return M


def _compute_covariance_safe(data: np.ndarray, rowvar: bool = True, 
                              pairwise_complete: bool = False,
                              min_eigenval: float = 1e-8,
                              fallback_to_identity: bool = True) -> np.ndarray:
    """Compute covariance matrix safely with robust error handling.
    
    This function computes covariance matrices with automatic handling of:
    - Missing data (NaN values)
    - Edge cases (empty data, single series, etc.)
    - Numerical instability (negative eigenvalues)
    - Fallback strategies for failed computations
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix (T x N) where T is time periods and N is number of series.
        If rowvar=True, each row is a variable (standard). If rowvar=False, each column is a variable.
    rowvar : bool
        If True (default), each row represents a variable, each column an observation.
        If False, each column represents a variable, each row an observation.
    pairwise_complete : bool
        If True, use pairwise complete observations (more robust to missing data).
        If False, use listwise deletion (all variables must be observed simultaneously).
    min_eigenval : float
        Minimum eigenvalue threshold for positive semi-definiteness.
    fallback_to_identity : bool
        If True, return identity matrix if covariance computation fails.
        If False, raise an exception.
        
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N) where N is number of variables.
        Guaranteed to be positive semi-definite.
        
    Notes
    -----
    - For single series, returns variance as 1x1 matrix
    - For empty data, returns identity matrix if fallback_to_identity=True
    - Automatically regularizes if negative eigenvalues are found
    - Uses pairwise complete observations when pairwise_complete=True for robustness
    
    Examples
    --------
    >>> data = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 4.0]])
    >>> cov = _compute_covariance_safe(data, pairwise_complete=True)
    >>> assert cov.shape == (2, 2)
    >>> assert np.all(np.linalg.eigvalsh(cov) >= 0)  # PSD
    """
    if data.size == 0:
        if fallback_to_identity:
            return np.eye(1) if data.ndim == 1 else np.eye(data.shape[1] if rowvar else data.shape[0])
        raise ValueError("Cannot compute covariance: data is empty")
    
    # Handle 1D case
    if data.ndim == 1:
        var_val = _compute_variance_safe(data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE, 
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Determine number of variables
    n_vars = data.shape[1] if rowvar else data.shape[0]
    
    # Handle single variable case
    if n_vars == 1:
        series_data = data.flatten()
        var_val = _compute_variance_safe(series_data, ddof=0, min_variance=MIN_VARIANCE_COVARIANCE,
                                         default_variance=DEFAULT_VARIANCE_FALLBACK)
        return np.array([[var_val]])
    
    # Compute covariance
    try:
        if pairwise_complete:
            # Pairwise complete covariance: compute covariance for each pair separately
            # This is more robust when data has missing values
            if rowvar:
                # data is (T, N): each row is time, each column is series
                # We want (N, N) covariance matrix
                data_for_cov = data.T  # Transpose to (N, T) for np.cov
            else:
                # data is (N, T): each row is series, each column is time
                # We want (N, N) covariance matrix
                data_for_cov = data
            
            # Compute pairwise complete covariance manually
            # For each pair of variables, compute covariance using only complete observations
            cov = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(i, n_vars):
                    # Extract data for variables i and j
                    var_i = data_for_cov[i, :]
                    var_j = data_for_cov[j, :]
                    # Find complete observations (both non-NaN)
                    complete_mask = np.isfinite(var_i) & np.isfinite(var_j)
                    if np.sum(complete_mask) < 2:
                        # Not enough data, use default variance
                        if i == j:
                            cov[i, j] = DEFAULT_VARIANCE_FALLBACK
                        else:
                            cov[i, j] = 0.0
                    else:
                        # Compute covariance for this pair
                        var_i_complete = var_i[complete_mask]
                        var_j_complete = var_j[complete_mask]
                        if i == j:
                            # Variance
                            cov[i, j] = np.var(var_i_complete, ddof=0)
                        else:
                            # Covariance
                            mean_i = np.mean(var_i_complete)
                            mean_j = np.mean(var_j_complete)
                            cov[i, j] = np.mean((var_i_complete - mean_i) * (var_j_complete - mean_j))
                            cov[j, i] = cov[i, j]  # Symmetric
            
            # Ensure minimum variance
            np.fill_diagonal(cov, np.maximum(np.diag(cov), MIN_VARIANCE_COVARIANCE))
        else:
            # Standard covariance (listwise deletion)
            # Remove rows/columns with any NaN
            if rowvar:
                # data is (T, N): remove rows (time periods) with any NaN
                complete_rows = np.all(np.isfinite(data), axis=1)
                if np.sum(complete_rows) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[complete_rows, :]
                # np.cov with rowvar=True expects each row to be a variable
                # But our data is (T, N) where each row is time, each column is series
                # So we need to transpose: (N, T) where each row is series, each column is time
                data_for_cov = data_clean.T  # (N, T)
                cov = np.cov(data_for_cov, rowvar=True)  # Returns (N, N)
            else:
                # data is (N, T): remove columns (time periods) with any NaN
                complete_cols = np.all(np.isfinite(data), axis=0)
                if np.sum(complete_cols) < 2:
                    raise ValueError("Insufficient complete observations for covariance")
                data_clean = data[:, complete_cols]
                # np.cov with rowvar=False expects each column to be a variable
                # Our data is (N, T) where each row is series, each column is time
                # So we need: (N, T) where each column is series, each row is time
                # Actually, np.cov with rowvar=False treats each column as variable
                # So we need to transpose: (T, N) where each column is series
                data_for_cov = data_clean.T  # (T, N)
                cov = np.cov(data_for_cov, rowvar=False)  # Returns (N, N)
            
            # np.cov can sometimes return unexpected shapes, so verify
            if cov.ndim == 0:
                # Single value case
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                # 1D case - convert to 2D
                if len(cov) == n_vars:
                    cov = np.diag(cov)
                else:
                    raise ValueError(f"np.cov returned unexpected 1D shape: {cov.shape}, expected ({n_vars}, {n_vars})")
        
        # Ensure correct shape - critical check to prevent broadcasting errors
        if cov.shape != (n_vars, n_vars):
            raise ValueError(
                f"Covariance shape mismatch: expected ({n_vars}, {n_vars}), got {cov.shape}. "
                f"Data shape was {data.shape}, rowvar={rowvar}, pairwise_complete={pairwise_complete}"
            )
        
        # Ensure positive semi-definite
        if np.any(~np.isfinite(cov)):
            raise ValueError("Covariance contains non-finite values")
        
        # Ensure positive semi-definite
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < 0):
            # Regularize if needed
            reg_amount = abs(np.min(eigenvals)) + min_eigenval
            # Double-check shape before broadcasting to prevent errors
            if cov.shape != (n_vars, n_vars):
                raise ValueError(
                    f"Cannot regularize: cov shape {cov.shape} != expected ({n_vars}, {n_vars})"
                )
            eye_matrix = np.eye(n_vars)
            if eye_matrix.shape != cov.shape:
                raise ValueError(
                    f"Cannot regularize: eye shape {eye_matrix.shape} != cov shape {cov.shape}"
                )
            cov = cov + eye_matrix * reg_amount
            # Re-verify shape after regularization
            if cov.shape != (n_vars, n_vars):
                raise ValueError(
                    f"Shape changed after regularization: {cov.shape} != expected ({n_vars}, {n_vars})"
                )
        
        return cov
    except (ValueError, np.linalg.LinAlgError) as e:
        if fallback_to_identity:
            _logger.warning(
                f"Covariance computation failed ({type(e).__name__}), "
                f"falling back to identity matrix. Error: {str(e)[:100]}"
            )
            return np.eye(n_vars)
        raise


def _ensure_innovation_variance_minimum(Q: np.ndarray, min_variance: float = 1e-8) -> np.ndarray:
    """Ensure innovation covariance matrix Q has minimum diagonal values.
    
    This is critical for factor evolution: if Q[i,i] = 0, factor i cannot evolve
    (innovation variance is zero). This function enforces a minimum variance
    threshold on the diagonal elements while preserving off-diagonal structure.
    
    Parameters
    ----------
    Q : np.ndarray
        Innovation covariance matrix (m x m) where m is the state dimension.
        Can be any square matrix representing innovation variances.
    min_variance : float
        Minimum allowed variance for each diagonal element. Default is 1e-8.
        This ensures factors can evolve even with very small innovations.
        
    Returns
    -------
    np.ndarray
        Q matrix with guaranteed minimum diagonal values. Off-diagonal elements
        are preserved unchanged.
        
    Notes
    -----
    - Only modifies diagonal elements, preserving correlation structure
    - If Q is empty or non-square, returns Q unchanged
    - This is a common operation in DFM initialization and EM steps
    
    Examples
    --------
    >>> Q = np.array([[0.0, 0.1], [0.1, 0.0]])  # Zero diagonal
    >>> Q_safe = _ensure_innovation_variance_minimum(Q, min_variance=1e-8)
    >>> assert np.all(np.diag(Q_safe) >= 1e-8)  # All diagonals >= 1e-8
    """
    if Q.size == 0 or Q.shape[0] == 0 or Q.shape[0] != Q.shape[1]:
        return Q
    
    Q_diag = np.diag(Q)
    Q_diag = np.maximum(Q_diag, min_variance)
    # Preserve off-diagonal elements: Q_new = diag(Q_diag) + (Q - diag(Q))
    Q = np.diag(Q_diag) + Q - np.diag(np.diag(Q))
    return Q


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


def _ensure_positive_definite(M: np.ndarray, min_eigenval: float = 1e-8, 
                              warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ensure matrix is positive semi-definite by adding regularization if needed.
    
    This function ensures a matrix is positive semi-definite (PSD) by adding
    a diagonal regularization term if the minimum eigenvalue is below the threshold.
    The matrix is first symmetrized, then regularized if necessary.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to regularize (any shape, will be symmetrized first)
    min_eigenval : float, default 1e-8
        Minimum eigenvalue threshold. If minimum eigenvalue < min_eigenval,
        regularization is applied by adding (min_eigenval - min_eig) * I to diagonal.
    warn : bool, default True
        If True, log warning when regularization is applied
        
    Returns
    -------
    M_regularized : np.ndarray
        Positive semi-definite matrix (guaranteed min eigenvalue >= min_eigenval)
    stats : dict
        Statistics dictionary with:
        - 'regularized': bool, whether regularization was applied
        - 'min_eigenval_before': float, minimum eigenvalue before regularization
        - 'reg_amount': float, amount of regularization added to diagonal
        - 'min_eigenval_after': float, minimum eigenvalue after regularization
        
    Notes
    -----
    - Matrix is first symmetrized via _ensure_symmetric()
    - Regularization adds a diagonal term: M + reg_amount * I
    - Used extensively in covariance matrix stabilization (Kalman filter, EM algorithm)
    - Regularization biases the matrix but ensures numerical stability
    - If eigendecomposition fails, applies conservative regularization (min_eigenval * I)
    """
    M = _ensure_symmetric(M)
    stats = {
        'regularized': False,
        'min_eigenval_before': None,
        'reg_amount': 0.0,
        'min_eigenval_after': None
    }
    if M.size == 0 or M.shape[0] == 0:
        return M, stats
    try:
        eigenvals = np.linalg.eigvalsh(M)
        min_eig = float(np.min(eigenvals))
        stats['min_eigenval_before'] = float(min_eig)
        if min_eig < min_eigenval:
            reg_amount = min_eigenval - min_eig
            M = M + np.eye(M.shape[0]) * reg_amount
            M = _ensure_symmetric(M)
            stats['regularized'] = True
            stats['reg_amount'] = float(reg_amount)
            eigenvals_after = np.linalg.eigvalsh(M)
            stats['min_eigenval_after'] = float(np.min(eigenvals_after))
            if warn:
                _logger.warning(
                    f"Matrix regularization applied: min eigenvalue {min_eig:.2e} < {min_eigenval:.2e}, "
                    f"added {reg_amount:.2e} to diagonal. This biases the covariance matrix."
                )
        else:
            stats['min_eigenval_after'] = float(min_eig)
    except (np.linalg.LinAlgError, ValueError) as e:
        M = M + np.eye(M.shape[0]) * min_eigenval
        M = _ensure_symmetric(M)
        stats['regularized'] = True
        stats['reg_amount'] = float(min_eigenval)
        if warn:
            _logger.warning(
                f"Matrix regularization applied (eigendecomposition failed: {e}). "
                f"Added {min_eigenval:.2e} to diagonal. This biases the covariance matrix."
            )
    return M, stats


def _compute_regularization_param(matrix: np.ndarray, scale_factor: float = 1e-5, 
                                  warn: bool = True) -> Tuple[float, Dict[str, Any]]:
    """Compute regularization parameter based on matrix scale.
    
    Computes a regularization parameter as a fraction of the matrix trace.
    This is used to stabilize matrix inversions by adding a small diagonal term.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix for which to compute regularization parameter
    scale_factor : float, default 1e-5
        Scaling factor applied to matrix trace (typically 1e-5 to 1e-3)
    warn : bool, default True
        If True, log info message when regularization is applied
        
    Returns
    -------
    reg_param : float
        Regularization parameter (max(trace * scale_factor, 1e-8))
        Minimum value of 1e-8 ensures numerical stability
    stats : dict
        Dictionary with 'trace', 'scale_factor', and 'reg_param' values
        
    Notes
    -----
    - Used in EM step for loading matrix (C) updates to prevent singular matrices
    - Minimum value of 1e-8 ensures regularization even for very small traces
    - Formula: reg_param = max(trace(matrix) * scale_factor, 1e-8)
    """
    trace = np.trace(matrix)
    reg_param = max(trace * scale_factor, 1e-8)
    stats = {'trace': float(trace), 'scale_factor': float(scale_factor), 'reg_param': float(reg_param)}
    if warn and reg_param > 1e-8:
        _logger.info(
            f"Regularization parameter computed: {reg_param:.2e} "
            f"(trace={trace:.2e}, scale={scale_factor:.2e})."
        )
    return reg_param, stats


def _clip_ar_coefficients(A: np.ndarray, min_val: float = -0.99, max_val: float = 0.99, 
                         warn: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip AR coefficients to stability bounds.
    
    Clips transition matrix (AR) coefficients to ensure stability of the
    factor dynamics. Coefficients outside [min_val, max_val] are clipped
    to the nearest bound.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix containing AR coefficients (any shape)
    min_val : float, default -0.99
        Minimum allowed AR coefficient (lower bound for clipping)
    max_val : float, default 0.99
        Maximum allowed AR coefficient (upper bound for clipping)
    warn : bool, default True
        If True, log warning when clipping is applied
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped transition matrix with all values in [min_val, max_val]
    stats : dict
        Statistics dictionary with:
        - 'n_clipped': Number of coefficients that were clipped
        - 'n_total': Total number of coefficients
        - 'clipped_indices': List of flattened indices that were clipped
        - 'min_violations': Number of values below min_val
        - 'max_violations': Number of values above max_val
        
    Notes
    -----
    - Default bounds [-0.99, 0.99] ensure factor dynamics remain stable
    - Used in EM step to prevent explosive or oscillatory factor behavior
    - Clipping preserves matrix structure (only values are modified)
    """
    A_flat = A.flatten()
    n_total = len(A_flat)
    below_min = A_flat < min_val
    above_max = A_flat > max_val
    needs_clip = below_min | above_max
    n_clipped = np.sum(needs_clip)
    A_clipped = np.clip(A, min_val, max_val)
    stats = {
        'n_clipped': int(n_clipped),
        'n_total': int(n_total),
        'clipped_indices': np.where(needs_clip)[0].tolist() if n_clipped > 0 else [],
        'min_violations': int(np.sum(below_min)),
        'max_violations': int(np.sum(above_max))
    }
    if warn and n_clipped > 0:
        pct_clipped = 100.0 * n_clipped / n_total if n_total > 0 else 0.0
        _logger.warning(
            f"AR coefficient clipping applied: {n_clipped}/{n_total} ({pct_clipped:.1f}%) "
            f"coefficients clipped to [{min_val}, {max_val}]."
        )
    return A_clipped, stats


def _apply_ar_clipping(A: np.ndarray, config: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply AR coefficient clipping based on configuration.
    
    This is a convenience wrapper around `_clip_ar_coefficients()` that reads
    clipping parameters from a configuration object. If clipping is disabled
    in config, returns the matrix unchanged.
    
    Parameters
    ----------
    A : np.ndarray
        Transition matrix to clip (any shape)
    config : object, optional
        Configuration object with clipping parameters. If None, uses defaults.
        Expected attributes:
        - clip_ar_coefficients: bool, whether clipping is enabled (default: True)
        - ar_clip_min: float, minimum AR coefficient (default: -0.99)
        - ar_clip_max: float, maximum AR coefficient (default: 0.99)
        - warn_on_ar_clip: bool, whether to log warnings (default: True)
        
    Returns
    -------
    A_clipped : np.ndarray
        Clipped transition matrix (unchanged if clipping disabled)
    stats : dict
        Statistics about clipping operation (same format as _clip_ar_coefficients)
        
    Notes
    -----
    - Wrapper function that delegates to `_clip_ar_coefficients()` after reading config
    - If config is None, uses default bounds [-0.99, 0.99]
    - If clipping is disabled in config, returns A unchanged with empty stats
    - Used in EM step to apply configurable AR coefficient constraints
    - Default bounds ensure factor dynamics remain stable (prevent explosive behavior)
    """
    if config is None:
        return _clip_ar_coefficients(A, -0.99, 0.99, True)
    
    from .helpers import safe_get_attr
    
    clip_enabled = safe_get_attr(config, 'clip_ar_coefficients', True)
    if not clip_enabled:
        return A, {'n_clipped': 0, 'n_total': A.size, 'clipped_indices': []}
    
    min_val = safe_get_attr(config, 'ar_clip_min', -0.99)
    max_val = safe_get_attr(config, 'ar_clip_max', 0.99)
    warn = safe_get_attr(config, 'warn_on_ar_clip', True)
    return _clip_ar_coefficients(A, min_val, max_val, warn)


def _cap_max_eigenvalue(M: np.ndarray, max_eigenval: float = 1e6) -> np.ndarray:
    """Cap maximum eigenvalue of a matrix to prevent numerical explosion.
    
    If the maximum eigenvalue exceeds max_eigenval, the entire matrix is scaled
    down proportionally to bring the maximum eigenvalue to max_eigenval.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to cap (any shape)
    max_eigenval : float, default 1e6
        Maximum allowed eigenvalue
        
    Returns
    -------
    np.ndarray
        Matrix with capped eigenvalues (scaled if necessary)
        
    Notes
    -----
    - If eigendecomposition fails, falls back to diagonal capping
    - Preserves matrix structure (scales entire matrix, not just eigenvalues)
    """
    try:
        eigenvals = np.linalg.eigvals(M)
        max_eig = np.max(eigenvals)
        if max_eig > max_eigenval:
            scale = max_eigenval / max_eig
            return M * scale
    except (np.linalg.LinAlgError, ValueError):
        M_diag = np.diag(M)
        M_diag = np.maximum(M_diag, MIN_EIGENVAL_CLEAN)
        M_diag = np.minimum(M_diag, max_eigenval)
        M_capped = np.diag(M_diag)
        return _ensure_symmetric(M_capped)
    return M


def _estimate_ar_coefficient(EZZ_FB: np.ndarray, EZZ_BB: np.ndarray, 
                             vsmooth_sum: Optional[np.ndarray] = None,
                             T: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate AR coefficients and innovation variances from expectations.
    
    Computes AR(1) coefficients from smoothed factor expectations:
    A = E[Z_t Z_{t-1}'] / E[Z_{t-1} Z_{t-1}']
    
    Parameters
    ----------
    EZZ_FB : np.ndarray
        Expected cross-covariance E[Z_t Z_{t-1}'] (diagonal extracted)
    EZZ_BB : np.ndarray
        Expected lag covariance E[Z_{t-1} Z_{t-1}'] (diagonal extracted)
    vsmooth_sum : np.ndarray, optional
        Additional variance terms from smoothing (added to EZZ_BB)
    T : int, optional
        Number of time periods. Currently unused but reserved for future
        computation of innovation variances Q_diag.
        
    Returns
    -------
    A_diag : np.ndarray
        AR coefficients (diagonal elements)
    Q_diag : np.ndarray
        Innovation variances. Currently always None, but reserved for future
        implementation when T parameter is utilized.
        
    Notes
    -----
    - Handles scalar, 1D, and 2D input arrays
    - Enforces minimum denominator threshold to prevent division by zero
    - Cleans non-finite values in numerator before division
    - Future: Q_diag computation will use T parameter for proper scaling
    """
    if np.isscalar(EZZ_FB):
        EZZ_FB = np.array([EZZ_FB])
        EZZ_BB = np.array([EZZ_BB])
    if EZZ_FB.ndim > 1:
        EZZ_FB_diag = np.diag(EZZ_FB).copy()
        EZZ_BB_diag = np.diag(EZZ_BB).copy()
    else:
        EZZ_FB_diag = EZZ_FB.copy()
        EZZ_BB_diag = EZZ_BB.copy()
    if vsmooth_sum is not None:
        if vsmooth_sum.ndim > 1:
            vsmooth_diag = np.diag(vsmooth_sum)
        else:
            vsmooth_diag = vsmooth_sum
        EZZ_BB_diag = EZZ_BB_diag + vsmooth_diag
    min_denom = np.maximum(np.abs(EZZ_BB_diag) * MIN_DIAGONAL_VARIANCE, MIN_VARIANCE_COVARIANCE)
    EZZ_BB_diag = np.where(
        (np.isnan(EZZ_BB_diag) | np.isinf(EZZ_BB_diag) | (np.abs(EZZ_BB_diag) < min_denom)),
        min_denom, EZZ_BB_diag
    )
    # Use _clean_matrix for consistency (handles both scalar and array cases)
    if EZZ_FB_diag.ndim == 0:
        # Scalar case: convert to array, clean, then extract scalar
        EZZ_FB_diag_clean = _clean_matrix(np.array([EZZ_FB_diag]), 'general', default_nan=0.0, default_inf=1e6)
        EZZ_FB_diag = EZZ_FB_diag_clean[0] if EZZ_FB_diag_clean.size > 0 else 0.0
    else:
        EZZ_FB_diag = _clean_matrix(EZZ_FB_diag, 'general', default_nan=0.0, default_inf=1e6)
    A_diag = EZZ_FB_diag / EZZ_BB_diag
    # Q_diag computation reserved for future implementation when T parameter is utilized
    Q_diag = None
    return A_diag, Q_diag


def _compute_variance_safe(data: np.ndarray, ddof: int = 0, 
                           min_variance: float = MIN_VARIANCE_COVARIANCE,
                           default_variance: float = DEFAULT_VARIANCE_FALLBACK) -> float:
    """Compute variance safely with robust error handling.
    
    This function computes variance with automatic handling of:
    - Missing data (NaN values)
    - Edge cases (empty data, insufficient samples)
    - Numerical instability (non-finite results)
    - Minimum variance threshold enforcement
    
    Parameters
    ----------
    data : np.ndarray
        Data array (1D or 2D). If 2D, variance is computed over all elements.
    ddof : int
        Delta degrees of freedom. Default is 0 (population variance).
    min_variance : float
        Minimum allowed variance threshold. Default is MIN_VARIANCE_COVARIANCE (1e-10).
        Values below this are replaced with default_variance.
    default_variance : float
        Default variance value to use when computation fails or result is invalid.
        Default is DEFAULT_VARIANCE_FALLBACK (1.0).
        
    Returns
    -------
    float
        Variance value, guaranteed to be finite and >= min_variance.
        
    Notes
    -----
    - Uses np.nanvar for automatic NaN handling
    - Returns default_variance if result is NaN, Inf, or < min_variance
    - Flattens 2D arrays before computation
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, np.nan, 4.0])
    >>> var = _compute_variance_safe(data)
    >>> assert var >= 1e-10  # Minimum threshold
    >>> assert np.isfinite(var)  # Always finite
    """
    if data.size == 0:
        return default_variance
    
    # Flatten if 2D
    if data.ndim > 1:
        data = data.flatten()
    
    # Compute variance with NaN handling
    # Suppress warning when ddof >= number of non-NaN values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        var_val = np.nanvar(data, ddof=ddof)
    
    # Validate and enforce minimum
    if np.isnan(var_val) or np.isinf(var_val) or var_val < min_variance:
        return default_variance
    
    return float(var_val)


def _safe_divide(numerator: np.ndarray, denominator: float, default: float = 0.0) -> np.ndarray:
    """Safely divide numerator by denominator, handling zero and invalid values.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array
    denominator : float
        Denominator value
    default : float, default 0.0
        Default value to use if denominator is zero, NaN, or Inf
        
    Returns
    -------
    np.ndarray
        Result of division, with invalid results replaced by default value
        
    Notes
    -----
    - Replaces non-finite results (NaN, Inf) with default value
    - Useful for avoiding division by zero errors in numerical computations
    - Preserves array shape and dtype of numerator
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return np.full_like(numerator, default)
    result = numerator / denominator
    result = np.where(np.isfinite(result), result, default)
    return result



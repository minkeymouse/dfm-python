"""Block structure helpers for extracting and updating block matrices."""

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import block_diag


def update_block_diag(
    A: Optional[np.ndarray],
    Q: Optional[np.ndarray],
    V_0: Optional[np.ndarray],
    A_block: np.ndarray,
    Q_block: np.ndarray,
    V_0_block: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update block diagonal matrices by appending new blocks.
    
    Parameters
    ----------
    A : np.ndarray, optional
        Existing transition matrix (m x m) or None if first block
    Q : np.ndarray, optional
        Existing innovation covariance (m x m) or None if first block
    V_0 : np.ndarray, optional
        Existing initial covariance (m x m) or None if first block
    A_block : np.ndarray
        New transition matrix block to append (k x k)
    Q_block : np.ndarray
        New innovation covariance block to append (k x k)
    V_0_block : np.ndarray
        New initial covariance block to append (k x k)
        
    Returns
    -------
    A_new : np.ndarray
        Updated transition matrix with new block appended
    Q_new : np.ndarray
        Updated innovation covariance with new block appended
    V_0_new : np.ndarray
        Updated initial covariance with new block appended
    """
    if A is None:
        return A_block, Q_block, V_0_block
    else:
        return (
            block_diag(A, A_block),
            block_diag(Q, Q_block),
            block_diag(V_0, V_0_block)
        )


def get_block_indices(
    blocks: np.ndarray,
    block_idx: int
) -> np.ndarray:
    """Get series indices for a specific block.
    
    Parameters
    ----------
    blocks : np.ndarray
        Block structure matrix (n x n_blocks)
    block_idx : int
        Index of the block (0-based)
        
    Returns
    -------
    indices : np.ndarray
        Array of series indices belonging to the block
    """
    return np.where(blocks[:, block_idx] == 1)[0]


def compute_block_slice_indices(
    r: np.ndarray,
    block_idx: int,
    ppC: int
) -> Tuple[int, int]:
    """Compute start and end indices for a block in state space.
    
    Parameters
    ----------
    r : np.ndarray
        Number of factors per block (n_blocks,)
    block_idx : int
        Index of current block (0-based)
    ppC : int
        Maximum of p (AR lag) and pC (tent length)
        
    Returns
    -------
    t_start : int
        Start index for block in state space
    t_end : int
        End index for block in state space (exclusive)
    """
    factor_start_idx = int(np.sum(r[:block_idx]) * ppC)
    r_i_int = int(r[block_idx])
    t_start = factor_start_idx
    t_end = factor_start_idx + r_i_int * ppC
    return t_start, t_end


def extract_block_matrix(
    matrix: np.ndarray,
    t_start: int,
    t_end: int,
    copy: bool = True
) -> np.ndarray:
    """Extract block submatrix from a larger matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Full matrix to extract from (m x m)
    t_start : int
        Start index for block
    t_end : int
        End index for block (exclusive)
    copy : bool, default True
        Whether to copy the submatrix (recommended to avoid aliasing)
        
    Returns
    -------
    block_matrix : np.ndarray
        Extracted block submatrix ((t_end-t_start) x (t_end-t_start))
    """
    block = matrix[t_start:t_end, t_start:t_end]
    return block.copy() if copy else block


def update_block_in_matrix(
    matrix: np.ndarray,
    block_matrix: np.ndarray,
    t_start: int,
    t_end: int
) -> None:
    """Update a block submatrix in a larger matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to update (modified in-place)
    block_matrix : np.ndarray
        Block matrix to insert
    t_start : int
        Start index for block
    t_end : int
        End index for block (exclusive)
        
    Notes
    -----
    - Updates matrix in-place
    - Assumes block_matrix.shape == (t_end-t_start, t_end-t_start)
    """
    matrix[t_start:t_end, t_start:t_end] = block_matrix


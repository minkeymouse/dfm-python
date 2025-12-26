"""IRF computation for KDFM.

Computes reduced-form and structural impulse response functions (IRFs)
from companion matrices and structural identification matrix.
"""

from typing import Tuple
import numpy as np
import torch


def compute_irf(
    A_ar: torch.Tensor,
    A_ma: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_prime: torch.Tensor,
    C_prime: torch.Tensor,
    S: torch.Tensor,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reduced-form and structural IRFs.
    
    Computes:
    - Reduced-form IRF: K_h = C' (A^MA)^h B' C (A^AR)^h B for h = 0, ..., horizon-1
    - Structural IRF: K_h^struct = K_h S for h = 0, ..., horizon-1
    
    Parameters
    ----------
    A_ar : torch.Tensor
        AR companion matrix of shape (p*K, p*K)
    A_ma : torch.Tensor
        MA companion matrix of shape (q*K, q*K)
    B : torch.Tensor
        AR input matrix of shape (p*K, K)
    C : torch.Tensor
        AR output matrix of shape (K, p*K)
    B_prime : torch.Tensor
        MA input matrix of shape (q*K, K)
    C_prime : torch.Tensor
        MA output matrix of shape (K, q*K)
    S : torch.Tensor
        Structural identification matrix of shape (K, K)
    horizon : int
        Number of horizons to compute
        
    Returns
    -------
    irf_reduced : np.ndarray
        Reduced-form IRFs of shape (horizon, K, K)
    irf_structural : np.ndarray
        Structural IRFs of shape (horizon, K, K)
    """
    device = A_ar.device
    dtype = A_ar.dtype
    K = C.shape[0]
    
    # Initialize IRF arrays
    irf_reduced = np.zeros((horizon, K, K))
    irf_structural = np.zeros((horizon, K, K))
    
    # Compute IRF for each horizon
    A_ar_power = torch.eye(A_ar.shape[0], device=device, dtype=dtype)
    A_ma_power = torch.eye(A_ma.shape[0], device=device, dtype=dtype)
    
    for h in range(horizon):
        # Compute K_h = C' (A^MA)^h B' C (A^AR)^h B
        # First compute (A^AR)^h B
        ar_term = A_ar_power @ B  # (p*K, K)
        
        # Then compute C (A^AR)^h B
        ar_output = C @ ar_term  # (K, K)
        
        # Compute (A^MA)^h B'
        ma_term = A_ma_power @ B_prime  # (q*K, K)
        
        # Compute C' (A^MA)^h B'
        ma_output = C_prime @ ma_term  # (K, K)
        
        # Combine: C' (A^MA)^h B' C (A^AR)^h B
        K_h = ma_output @ ar_output  # (K, K)
        
        irf_reduced[h] = K_h.detach().cpu().numpy()
        
        # Structural IRF: K_h^struct = K_h S
        K_h_struct = K_h @ S
        irf_structural[h] = K_h_struct.detach().cpu().numpy()
        
        # Update powers for next iteration
        if h < horizon - 1:
            A_ar_power = A_ar_power @ A_ar
            A_ma_power = A_ma_power @ A_ma
    
    return irf_reduced, irf_structural


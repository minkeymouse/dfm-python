"""IRF computation for KDFM.

This module provides functions for computing impulse response functions (IRFs)
from KDFM companion matrices. IRFs are the primary object of estimation in KDFM,
computed directly from companion matrix powers without requiring coefficient-to-IRF
conversion.

Key Concepts:
- **Reduced-form IRF**: IRF computed from companion matrices without structural
  identification. Represents how reduced-form residuals propagate through the system.
- **Structural IRF**: IRF computed with structural identification matrix S.
  Represents how orthogonal structural shocks propagate through the system.
  Structural IRF = Reduced-form IRF @ S.

Direct IRF Computation:
KDFM computes IRFs directly as matrix powers: K_h = C' (A^MA)^h B' C (A^AR)^h B.
This direct computation avoids numerical error accumulation that occurs in
traditional VAR methods when converting coefficients to moving average representation.

References:
- Kilian & Lütkepohl (2017): Structural Vector Autoregressive Analysis
- Sims (1980): Macroeconomics and Reality
"""

from typing import Tuple, Optional
import numpy as np
import torch
from ..logger import get_logger

_logger = get_logger(__name__)


def compute_irf(
    A_ar: torch.Tensor,
    A_ma: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_prime: torch.Tensor,
    C_prime: torch.Tensor,
    S: torch.Tensor,
    horizon: int,
    structural: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute reduced-form and structural IRFs from KDFM companion matrices.
    
    This function computes IRFs directly from companion matrix powers, making
    IRF estimation the primary object rather than a derived quantity. This
    direct approach avoids numerical error accumulation that occurs in traditional
    VAR methods.
    
    Mathematical Formulation:
    - Reduced-form IRF: K_h = C' (A^MA)^h B' C (A^AR)^h B for h = 0, ..., horizon-1
    - Structural IRF: K_h^struct = K_h S for h = 0, ..., horizon-1
    
    When q = 0 (no MA stage), this reduces to:
    - Reduced-form IRF: K_h = C (A^AR)^h B
    - Structural IRF: K_h^struct = K_h S
    
    Parameters
    ----------
    A_ar : torch.Tensor
        AR companion matrix of shape (p*K, p*K) where p is AR order and K is
        number of variables. Encodes VAR coefficients in companion form.
    A_ma : torch.Tensor
        MA companion matrix of shape (q*K, q*K) where q is MA order. Set to
        identity if q = 0 (pure VAR model).
    B : torch.Tensor
        AR input matrix of shape (p*K, K). Maps structural shocks to AR stage
        latent state.
    C : torch.Tensor
        AR output matrix of shape (K, p*K). Maps AR stage latent state to
        observations (factor loading matrix).
    B_prime : torch.Tensor
        MA input matrix of shape (q*K, K). Maps AR stage output to MA stage
        latent state. Set to identity if q = 0.
    C_prime : torch.Tensor
        MA output matrix of shape (K, q*K). Maps MA stage latent state to
        final observations. Set to identity if q = 0.
    S : torch.Tensor
        Structural identification matrix of shape (K, K). Transforms
        reduced-form residuals to orthogonal structural shocks.
        Must satisfy: E[ε_t ε_t^T] = I where ε_t = S^{-1} e_t.
    horizon : int
        Number of horizons to compute (h = 0, ..., horizon-1).
        Must be positive.
    structural : bool, optional
        Whether to compute structural IRF. If False, only reduced-form IRF
        is computed. Default: True.
        
    Returns
    -------
    irf_reduced : np.ndarray
        Reduced-form IRFs of shape (horizon, K, K).
        irf_reduced[h, i, j] = response of variable i to shock in variable j
        at horizon h.
    irf_structural : np.ndarray or None
        Structural IRFs of shape (horizon, K, K) if structural=True, else None.
        irf_structural[h, i, j] = response of variable i to structural shock j
        at horizon h.
        
    Raises
    ------
    ValueError
        If horizon <= 0 or tensor shapes are incompatible.
    RuntimeError
        If IRF computation fails (e.g., numerical instability).
        
    Notes
    -----
    - IRFs are computed directly as matrix powers, avoiding recursive computation
      of moving average coefficients that can accumulate errors in traditional VAR.
    - This direct computation is more numerically stable, especially for systems
      with near-unit-root eigenvalues.
    - Structural IRF enables causal interpretation: each structural shock has
      clear economic/scientific meaning (orthogonal, unit variance).
      
    Examples
    --------
    >>> # Compute IRFs for a VAR(1) model (q=0)
    >>> A_ar = torch.randn(3, 3)  # p=1, K=3
    >>> A_ma = torch.eye(3)  # Identity for q=0
    >>> B = torch.randn(3, 3)
    >>> C = torch.randn(3, 3)
    >>> B_prime = torch.eye(3)  # Identity for q=0
    >>> C_prime = torch.eye(3)  # Identity for q=0
    >>> S = torch.randn(3, 3)
    >>> irf_reduced, irf_structural = compute_irf(
    ...     A_ar, A_ma, B, C, B_prime, C_prime, S, horizon=20
    ... )
    >>> print(irf_reduced.shape)  # (20, 3, 3)
    >>> print(irf_structural.shape)  # (20, 3, 3)
    """
    # Import validation utilities
    from ..numeric.validator import validate_irf_horizon, validate_no_nan_inf, validate_eigenvalue_bounds
    
    # Validate horizon
    horizon = validate_irf_horizon(horizon)
    
    device = A_ar.device
    dtype = A_ar.dtype
    K = C.shape[0]
    
    # Validate tensor shapes with detailed error messages
    if A_ar.shape[0] != A_ar.shape[1]:
        raise ValueError(
            f"A_ar (AR companion matrix) must be square, got shape {A_ar.shape}. "
            f"Expected shape: (p*K, p*K) where p is AR order and K is number of variables."
        )
    if A_ma.shape[0] != A_ma.shape[1]:
        raise ValueError(
            f"A_ma (MA companion matrix) must be square, got shape {A_ma.shape}. "
            f"Expected shape: (q*K, q*K) where q is MA order and K is number of variables."
        )
    if B.shape[0] != A_ar.shape[0] or B.shape[1] != K:
        raise ValueError(
            f"B (AR input matrix) shape {B.shape} incompatible with A_ar {A_ar.shape} and K={K}. "
            f"Expected shape: ({A_ar.shape[0]}, {K})"
        )
    if C.shape[0] != K or C.shape[1] != A_ar.shape[0]:
        raise ValueError(
            f"C (AR output matrix) shape {C.shape} incompatible with A_ar {A_ar.shape} and K={K}. "
            f"Expected shape: ({K}, {A_ar.shape[0]})"
        )
    if S.shape != (K, K):
        raise ValueError(
            f"S (structural identification matrix) must be square of size K={K}, got shape {S.shape}. "
            f"Expected shape: ({K}, {K})"
        )
    
    # Validate for NaN/Inf
    validate_no_nan_inf(A_ar, name="A_ar")
    validate_no_nan_inf(A_ma, name="A_ma")
    validate_no_nan_inf(B, name="B")
    validate_no_nan_inf(C, name="C")
    validate_no_nan_inf(S, name="S")
    
    # Validate eigenvalue stability (warn if near-unstable)
    try:
        A_ar_eigenvalues = torch.linalg.eigvals(A_ar).detach().cpu().numpy()
        validate_eigenvalue_bounds(A_ar_eigenvalues, max_magnitude=1.0, warn_threshold=0.99)
    except Exception:
        pass  # Skip eigenvalue check if computation fails
    
    # Initialize IRF arrays
    irf_reduced = np.zeros((horizon, K, K), dtype=np.float64)
    irf_structural = np.zeros((horizon, K, K), dtype=np.float64) if structural else None
    
    # Compute IRF for each horizon using direct matrix powers
    # This avoids recursive computation that can accumulate errors
    # Direct computation: K_h = C' (A^MA)^h B' C (A^AR)^h B
    A_ar_power = torch.eye(A_ar.shape[0], device=device, dtype=dtype)
    A_ma_power = torch.eye(A_ma.shape[0], device=device, dtype=dtype)
    
    try:
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
            
            # Validate for NaN/Inf before storing
            if torch.any(torch.isnan(K_h)) or torch.any(torch.isinf(K_h)):
                from ..utils.errors import NumericalError
                raise NumericalError(
                    f"IRF computation produced NaN/Inf at horizon {h}. "
                    f"This indicates numerical instability.",
                    details=(
                        f"Horizon: {h}, K_h shape: {K_h.shape}. "
                        f"Consider: (1) Regularization, (2) Checking companion matrix stability, "
                        f"(3) Lower initialization scale, (4) Gradient clipping."
                    )
                )
            
            # Convert to numpy and store
            irf_reduced[h] = K_h.detach().cpu().numpy()
            
            # Structural IRF: K_h^struct = K_h S
            if structural:
                K_h_struct = K_h @ S
                
                # Validate structural IRF
                if torch.any(torch.isnan(K_h_struct)) or torch.any(torch.isinf(K_h_struct)):
                    from ..utils.errors import NumericalError
                    raise NumericalError(
                        f"Structural IRF computation produced NaN/Inf at horizon {h}. "
                        f"This may indicate issues with structural identification matrix S.",
                        details=(
                            f"Horizon: {h}, K_h_struct shape: {K_h_struct.shape}, S shape: {S.shape}. "
                            f"Check structural identification matrix S for numerical issues."
                        )
                    )
                
                irf_structural[h] = K_h_struct.detach().cpu().numpy()
            
            # Update powers for next iteration
            if h < horizon - 1:
                A_ar_power = A_ar_power @ A_ar
                A_ma_power = A_ma_power @ A_ma
                
    except RuntimeError as e:
        from ..utils.errors import NumericalError
        raise NumericalError(
            f"IRF computation failed at horizon {h}: {e}",
            details=(
                f"A_ar shape: {A_ar.shape}, A_ma shape: {A_ma.shape}, "
                f"B shape: {B.shape}, C shape: {C.shape}, "
                f"S shape: {S.shape}, horizon: {horizon}. "
                f"Consider: (1) Regularization, (2) Checking companion matrix stability, "
                f"(3) Lower initialization scale, (4) Gradient clipping."
            )
        ) from e
    
    # Final validation: Check for NaN/Inf in output
    from ..utils.errors import NumericalError
    from ..numeric.validator import validate_no_nan_inf
    
    try:
        validate_no_nan_inf(irf_reduced, name="reduced-form IRF")
    except Exception as e:
        raise NumericalError(
            "IRF computation produced NaN/Inf values in reduced-form IRF. "
            "This indicates numerical instability.",
            details=(
                f"NaN count: {np.sum(np.isnan(irf_reduced))}, "
                f"Inf count: {np.sum(np.isinf(irf_reduced))}. "
                f"Consider: (1) Regularization, (2) Checking companion matrix stability, "
                f"(3) Lower initialization scale."
            )
        ) from e
    
    if structural and irf_structural is not None:
        try:
            validate_no_nan_inf(irf_structural, name="structural IRF")
        except Exception as e:
            raise NumericalError(
                "IRF computation produced NaN/Inf values in structural IRF. "
                "This may indicate issues with structural identification matrix S.",
                details=(
                    f"NaN count: {np.sum(np.isnan(irf_structural))}, "
                    f"Inf count: {np.sum(np.isinf(irf_structural))}. "
                    f"Check structural identification matrix S for numerical issues."
                )
            ) from e
    
    return irf_reduced, irf_structural


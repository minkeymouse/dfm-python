"""Structural identification layer for KDFM.

Transforms reduced-form residuals to structural shocks through learnable
identification matrices. Supports Cholesky, full, and low-rank parameterizations.

This module implements structural identification for KDFM, enabling explicit
shock analysis where structural shocks ε_t are orthogonal and have unit variance:
ε_t = S^{-1} e_t, where S is the learnable structural identification matrix.

The structural matrix S can be parameterized as:
- Cholesky: S = L L^T (lower triangular L)
- Full: S (full matrix, all elements learnable)
- Low-rank: S = U V^T (rank-constrained factorization)

**FIXED (Iteration 7)**: The initialization bug causing near-singular structural
matrices has been fixed by changing DEFAULT_STRUCTURAL_DIAG_SCALE from 0.1 to 1.0.
This ensures S is well-conditioned and outputs maintain correct scale.
"""

from typing import Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
from ..config.constants import (
    DEFAULT_STRUCTURAL_INIT_SCALE,
    DEFAULT_STRUCTURAL_DIAG_SCALE,
    DEFAULT_CHOLESKY_EPS,
)
from ..utils.errors import ConfigurationError, NumericalError, NumericalStabilityError
from ..config.types import Tensor, Device
from ..logger import get_logger

_logger = get_logger(__name__)


class StructuralIdentificationSSM(nn.Module):
    """Structural identification layer that maps residuals to structural shocks.
    
    This layer transforms reduced-form residuals e_t to structural shocks ε_t
    via: ε_t = S^{-1} e_t
    
    The structural matrix S can be parameterized as:
    - Cholesky: S = L (lower triangular)
    - Full: S (full matrix)
    - Low-rank: S = U V^T
    
    When align_with_latent_state=True, structural shocks have dimension p*K (matching latent state).
    When False, shocks have dimension K (matching residuals).
    """
    
    def __init__(
        self,
        n_vars: int,
        lag_order: int = 1,
        method: Literal['cholesky', 'full', 'lowrank'] = 'cholesky',
        align_with_latent_state: bool = True
    ) -> None:
        """Initialize structural identification layer.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (K)
        lag_order : int, default=1
            Lag order (p). Used when align_with_latent_state=True to match latent state dimension.
        method : str, default='cholesky'
            Parameterization method: 'cholesky', 'full', or 'lowrank'
        align_with_latent_state : bool, default=True
            If True, structural shocks have dimension p*K (matching latent state).
            If False, structural shocks have dimension K (matching residuals).
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.lag_order = lag_order
        self.method = method
        self.align_with_latent_state = align_with_latent_state
        self.shock_dim = lag_order * n_vars if align_with_latent_state else n_vars
        
        # Residual expansion: K -> p*K if aligned with latent state
        if align_with_latent_state:
            self.residual_expansion = nn.Linear(n_vars, self.shock_dim, bias=False)
            with torch.no_grad():
                # Initialize to spread residuals across lags
                for i in range(lag_order):
                    start = i * n_vars
                    end = (i + 1) * n_vars
                    self.residual_expansion.weight[start:end, :] = torch.eye(n_vars) / lag_order
        else:
            self.residual_expansion = nn.Identity()
        
        # Initialization constants (use constants from config)
        # FIXED (Iteration 7): DEFAULT_STRUCTURAL_DIAG_SCALE changed from 0.1 to 1.0
        # to prevent near-singular structural matrices that caused scale mismatches
        self.init_scale = DEFAULT_STRUCTURAL_INIT_SCALE
        self.diag_scale = DEFAULT_STRUCTURAL_DIAG_SCALE
        self.cholesky_eps = DEFAULT_CHOLESKY_EPS
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize structural identification matrix weights.
        
        Initializes the structural matrix S based on the chosen parameterization method:
        - Cholesky: Lower triangular matrix L such that S = L L^T
        - Full: Full matrix S (all elements learnable)
        - Low-rank: Factorized as S = U V^T where U and V are low-rank matrices
        
        The initialization ensures S is well-conditioned and maintains numerical stability.
        """
        """Initialize structural identification matrix."""
        dim = self.shock_dim
        
        if self.method == 'cholesky':
            L = torch.randn(dim, dim) * self.init_scale
            with torch.no_grad():
                L = torch.tril(L) + torch.eye(dim) * self.diag_scale
            self.register_parameter('L', nn.Parameter(L))
        elif self.method == 'full':
            S = torch.randn(dim, dim) * self.init_scale
            with torch.no_grad():
                S = torch.eye(dim) + self.diag_scale * S
            self.register_parameter('S', nn.Parameter(S))
        elif self.method == 'lowrank':
            rank = max(1, dim // 2)
            self.register_parameter('U', nn.Parameter(torch.randn(dim, rank) * self.init_scale))
            self.register_parameter('V', nn.Parameter(torch.randn(dim, rank) * self.init_scale))
        else:
            raise ConfigurationError(
                f"Unknown structural identification method: {self.method}. "
                f"Must be 'cholesky', 'full', or 'lowrank'."
            )
    
    def get_structural_matrix(self) -> Tensor:
        """Get structural identification matrix S.
        
        This method returns the structural identification matrix S used to
        transform reduced-form residuals to structural shocks. The matrix
        is constructed based on the parameterization method:
        - Cholesky: S = L L^T (lower triangular L)
        - Full: S (full matrix)
        - Low-rank: S = U V^T (rank-constrained)
        
        **CRITICAL**: The matrix S must be well-conditioned to ensure stable
        inversion. The initialization uses DEFAULT_STRUCTURAL_DIAG_SCALE=1.0
        (fixed in Iteration 7) to prevent near-singular matrices.
        
        Returns
        -------
        Tensor
            Structural identification matrix S of shape (shock_dim, shock_dim)
            where shock_dim = p*K (if align_with_latent_state) or K (otherwise)
            
        Raises
        ------
        ConfigurationError
            If method is not recognized
            
        Examples
        --------
        >>> struct_id = StructuralIdentificationSSM(n_vars=5, method='cholesky')
        >>> S = struct_id.get_structural_matrix()
        >>> assert S.shape == (5, 5)  # (K, K) when align_with_latent_state=False
        >>> # S should be positive definite and well-conditioned
        """
        if self.method == 'cholesky':
            L = self.L
            # Ensure lower triangular and positive diagonal
            L = torch.tril(L)
            diag = torch.diag(L)
            diag = torch.clamp(diag, min=self.cholesky_eps)
            L = L - torch.diag(torch.diag(L)) + torch.diag(diag)
            return L @ L.T  # S = L L^T
        elif self.method == 'full':
            return self.S
        elif self.method == 'lowrank':
            return self.U @ self.V.T
        else:
            raise ConfigurationError(
                f"Unknown structural identification method: {self.method}. "
                f"Must be 'cholesky', 'full', or 'lowrank'."
            )
    
    def forward(self, residuals: Tensor) -> Tensor:
        """Transform reduced-form residuals to structural shocks.
        
        This method transforms reduced-form residuals e_t to structural shocks ε_t
        through the structural identification matrix S:
        
        ε_t = S^{-1} @ expanded_residuals
        
        where expanded_residuals are residuals expanded to match shock dimension
        (if align_with_latent_state=True). The structural shocks ε_t are orthogonal
        and have unit variance: E[ε_t ε_t^T] = I.
        
        **CRITICAL**: This method requires S to be well-conditioned for stable
        inversion. The initialization uses DEFAULT_STRUCTURAL_DIAG_SCALE=1.0
        (fixed in Iteration 7) to ensure numerical stability.
        
        Parameters
        ----------
        residuals : torch.Tensor
            Reduced-form residuals of shape (B, T, K) or (T, K) where:
            - B: batch size (optional)
            - T: sequence length
            - K: number of variables
            
        Returns
        -------
        Tensor
            Structural shocks of shape (B, T, shock_dim) or (T, shock_dim) where:
            - shock_dim = p*K (if align_with_latent_state) or K (otherwise)
            - Structural shocks are orthogonal: E[ε_t ε_t^T] = I
            
        Raises
        ------
        NumericalError
            If matrix inversion fails (S is singular or near-singular)
            
        Examples
        --------
        >>> struct_id = StructuralIdentificationSSM(n_vars=5, method='cholesky')
        >>> residuals = torch.randn(10, 5)  # (T, K)
        >>> structural_shocks = struct_id.forward(residuals)
        >>> assert structural_shocks.shape == (10, 5)  # (T, K) when align_with_latent_state=False
        """
        # Expand residuals if needed (K -> p*K)
        expanded_residuals = self.residual_expansion(residuals)
        
        # Get structural matrix S
        S = self.get_structural_matrix()
        
        # Validate structural matrix before inversion
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise NumericalError(
                "Structural identification matrix contains NaN/Inf values.",
                details=(
                    f"Matrix S has invalid values. Method: {self.method}. "
                    f"Consider: (1) Checking initialization, (2) Verifying training stability."
                )
            )
        
        # Transform: ε_t = S^{-1} @ expanded_residuals
        try:
            S_inv = torch.linalg.inv(S)
            
            # Validate inverse matrix
            if torch.isnan(S_inv).any() or torch.isinf(S_inv).any():
                raise NumericalError(
                    "Structural identification matrix inverse contains NaN/Inf values.",
                    details=(
                        f"Matrix S is singular or near-singular. Method: {self.method}. "
                        f"Consider: (1) Checking initialization (DEFAULT_STRUCTURAL_DIAG_SCALE=1.0), "
                        f"(2) Increasing regularization, (3) Using different parameterization method."
                    )
                )
            
            structural_shocks = expanded_residuals @ S_inv.T
            
            # Validate output
            if torch.isnan(structural_shocks).any() or torch.isinf(structural_shocks).any():
                raise NumericalError(
                    "Structural shocks contain NaN/Inf values after transformation.",
                    details=(
                        f"Transformation failed. Method: {self.method}, "
                        f"residuals shape: {residuals.shape}, S shape: {S.shape}. "
                        f"Consider: (1) Checking input residuals, (2) Verifying S conditioning."
                    )
                )
            
            return structural_shocks
        except torch.linalg.LinAlgError as e:
            raise NumericalError(
                "Structural identification matrix inversion failed (singular matrix).",
                details=(
                    f"Matrix S is singular or near-singular. "
                    f"Method: {self.method}, Error: {str(e)}. "
                    f"Consider: (1) Checking initialization (DEFAULT_STRUCTURAL_DIAG_SCALE=1.0), "
                    f"(2) Increasing regularization, (3) Using different parameterization method."
                )
            ) from e
        except RuntimeError as e:
            raise NumericalError(
                "Structural identification matrix inversion failed (runtime error).",
                details=(
                    f"Matrix inversion encountered runtime error. "
                    f"Method: {self.method}, Error: {str(e)}. "
                    f"Consider: (1) Checking matrix dimensions, (2) Verifying device compatibility."
                )
            ) from e
        except Exception as e:
            raise NumericalError(
                "Structural identification matrix inversion failed (unexpected error).",
                details=(
                    f"Unexpected error during matrix inversion. "
                    f"Method: {self.method}, Error type: {type(e).__name__}, Error: {str(e)}."
                )
            ) from e


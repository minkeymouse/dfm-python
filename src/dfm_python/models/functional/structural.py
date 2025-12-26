"""Structural identification layer for KDFM.

Transforms reduced-form residuals to structural shocks through learnable
identification matrices. Supports Cholesky, full, and low-rank parameterizations.
"""

import torch
import torch.nn as nn
from ...config.constants import (
    DEFAULT_STRUCTURAL_INIT_SCALE,
    DEFAULT_STRUCTURAL_DIAG_SCALE,
    DEFAULT_CHOLESKY_EPS,
)


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
        method: str = 'cholesky',
        align_with_latent_state: bool = True
    ):
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
        self.init_scale = DEFAULT_STRUCTURAL_INIT_SCALE
        self.diag_scale = DEFAULT_STRUCTURAL_DIAG_SCALE
        self.cholesky_eps = DEFAULT_CHOLESKY_EPS
        
        self._init_weights()
    
    def _init_weights(self) -> None:
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
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_structural_matrix(self) -> torch.Tensor:
        """Get structural identification matrix S."""
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
            raise ValueError(f"Unknown method: {self.method}")
    
    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """Transform reduced-form residuals to structural shocks.
        
        Parameters
        ----------
        residuals : torch.Tensor
            Reduced-form residuals of shape (..., K) or (..., T, K)
            
        Returns
        -------
        structural_shocks : torch.Tensor
            Structural shocks ε_t of shape (..., shock_dim) or (..., T, shock_dim)
        """
        # Expand residuals if needed
        expanded = self.residual_expansion(residuals)
        
        # Get structural matrix
        S = self.get_structural_matrix()
        
        # Compute S^{-1} @ expanded
        # Handle batched case
        if expanded.dim() == 2:
            # (T, shock_dim)
            S_inv = torch.linalg.solve(S, torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            return expanded @ S_inv.T
        elif expanded.dim() == 3:
            # (B, T, shock_dim)
            S_inv = torch.linalg.solve(S, torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            return torch.einsum('bti,ij->btj', expanded, S_inv.T)
        else:
            # (..., shock_dim)
            S_inv = torch.linalg.solve(S, torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            return torch.einsum('...i,ij->...j', expanded, S_inv.T)


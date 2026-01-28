"""Companion form State Space Model for iVDFM.

Implements block-diagonal companion form SSM for mapping innovations to factors.
Supports both first-order (p=1) with optimized direct computation and
higher-order (p>1) with companion form + Krylov methods.
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

from .base import BaseSSM
from ..functional.krylov import krylov
from ..numeric.builder import ivdfm_companion_from_p, build_ivdfm_diagonal_companion


class iVDFMCompanionSSM(BaseSSM):
    """Companion form SSM for iVDFM with block-diagonal structure.
    
    Maps innovations η_t to factors f_t via deterministic linear dynamics.
    Supports both first-order (p=1) and higher-order (p>1) AR dynamics.
    
    For p=1: Uses optimized direct computation (f_{t+1} = A*f_t + B*η_t)
    For p>1: Uses companion form + Krylov methods for efficiency
    
    Key features:
    - Block-diagonal structure (one block per factor) preserves identifiability
    - Efficient Krylov convolution for AR(p) > 1
    - Optimized path for p=1 (no companion form overhead)
    """
    
    def __init__(
        self,
        latent_dim: int,
        factor_order: int = 1,
        norm_order: int = 1,
        device: Optional[torch.device] = None,
        init_scale: float = 0.1,
    ):
        """Initialize iVDFMCompanionSSM.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of latent factors (r in paper)
        factor_order : int, default 1
            AR order for factor dynamics (p in AR(p))
        norm_order : int, default 1
            Order for normalization (0 = no normalization, 1 = L1 norm, etc.)
        device : Optional[torch.device]
            Device for computation
        init_scale : float, default 0.1
            Scale for parameter initialization
        """
        super().__init__(latent_dim, factor_order, device)
        self.norm_order = norm_order
        
        # Initialize parameters
        # Per paper: A and B must be DIAGONAL for identifiability
        if self.factor_order == 1:
            # First-order: diagonal A and B
            # A: (r,) diagonal elements
            self.A = nn.Parameter(torch.randn(latent_dim) * init_scale)
            # B: (r,) diagonal elements (not full matrix!)
            self.B = nn.Parameter(torch.randn(latent_dim) * init_scale)
        else:
            # AR(p): AR coefficients per factor
            # ar_coeffs: (r, p) - one AR(p) per factor
            self.ar_coeffs = nn.Parameter(
                torch.randn(latent_dim, factor_order) * init_scale
            )
            # B: (r,) diagonal elements for block-diagonal companion form
            self.B = nn.Parameter(torch.randn(latent_dim) * init_scale)
        
        # C matrix: extract factors from augmented state (for p>1)
        # For p=1, C is identity
        if self.factor_order > 1:
            # C: (r, r*p) - extracts first component from each block
            self.C = nn.Parameter(torch.zeros(latent_dim, latent_dim * factor_order))
            # Initialize C to extract first component: C[i, i*p] = 1
            for i in range(latent_dim):
                self.C.data[i, i * factor_order] = 1.0
        else:
            self.C = None
        
        # Initial state f_0
        self.f0 = nn.Parameter(torch.zeros(latent_dim))
        
        self.to(self.device)
    
    def norm(self, x: torch.Tensor, ord: int = None) -> torch.Tensor:
        """Normalize tensor for stability.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor to normalize
        ord : int, optional
            Norm order (uses self.norm_order if None)
            
        Returns
        -------
        torch.Tensor
            Normalized tensor
        """
        if ord is None:
            ord = self.norm_order
        
        if ord == 0:
            return x
        
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        # Only normalize if norm is not too small (avoid division by zero)
        if torch.abs(x_norm).mean().item() > 1e-4:
            x = x / x_norm
        return x
    
    def get_impulse_response(self, length: int) -> torch.Tensor:
        """Compute impulse response kernel H_k = C A^k B.
        
        Parameters
        ----------
        length : int
            Length of impulse response sequence (T)
            
        Returns
        -------
        torch.Tensor
            Impulse response kernel, shape (latent_dim, length)
            Each row corresponds to one factor's impulse response
        """
        if self.factor_order == 1:
            # First-order: H_k = A^k * B (element-wise for diagonal A and B)
            # For diagonal A and B: H_k[i] = A[i]^k * B[i]
            A_diag = torch.sigmoid(self.A)  # Ensure stability: A in (0, 1)
            B_diag = self.norm(self.B, ord=self.norm_order) if self.norm_order > 0 else self.B

            # Vectorized: H[i, k] = B[i] * A[i]^k
            k = torch.arange(length, device=A_diag.device, dtype=A_diag.dtype)  # (length,)
            A_powers = A_diag.unsqueeze(-1) ** k  # (r, length)
            H = B_diag.unsqueeze(-1) * A_powers  # (r, length)
            return H
        else:
            # AR(p): Use companion form + Krylov
            # Build block-diagonal companion matrix
            ar_coeffs_norm = self.norm(self.ar_coeffs, ord=self.norm_order) if self.norm_order > 0 else self.ar_coeffs
            A_block = build_ivdfm_diagonal_companion(ar_coeffs_norm)  # (r*p, r*p)
            
            # B_block: (r*p, r) - maps innovations to augmented state
            # For each factor i, innovations go to first component of augmented state
            # B is diagonal (r,), so B[i] affects only factor i
            B_block = torch.zeros(self.latent_dim * self.factor_order, self.latent_dim, 
                                 device=self.B.device, dtype=self.B.dtype)
            B_diag = self.norm(self.B, ord=self.norm_order) if self.norm_order > 0 else self.B
            for i in range(self.latent_dim):
                B_block[i * self.factor_order, i] = B_diag[i]  # Only diagonal element
            
            # C: (r, r*p) - extracts factors from augmented state
            C_norm = self.C if self.C is not None else torch.eye(self.latent_dim)
            
            # Compute impulse response using Krylov
            # H_k = C A^k B for k = 0, ..., length-1
            # Use Krylov to compute [B, AB, A^2B, ..., A^{length-1}B]
            # Then multiply by C: H = C * [B, AB, A^2B, ...]
            
            # Initialize: compute Krylov sequence for each factor independently
            # Since B is diagonal, each innovation component affects only its corresponding factor
            H_list = []
            for i in range(self.latent_dim):
                b_i = B_block[:, i]  # (r*p,) - only factor i's block is non-zero
                # Compute [b_i, A*b_i, A^2*b_i, ..., A^{length-1}*b_i]
                krylov_seq = krylov(length, A_block, b_i, c=None)  # (r*p, length)
                # Extract factor i: C[i, :] * krylov_seq
                H_i = torch.einsum('j, jl -> l', C_norm[i, :], krylov_seq)  # (length,)
                H_list.append(H_i)
            
            # Stack: (r, length)
            H = torch.stack(H_list, dim=0)  # (r, length)
            
            return H
    
    def forward(
        self,
        eta_sequence: torch.Tensor,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute factors from innovation sequence.
        
        Implements: f_t = A^t f_0 + Σ_{k=0}^{t-1} H_k η_{t-1-k}
        
        Parameters
        ----------
        eta_sequence : torch.Tensor
            Innovation sequence, shape (batch, T, latent_dim)
        f0 : Optional[torch.Tensor]
            Initial state, shape (batch, latent_dim) or (latent_dim,)
            
        Returns
        -------
        torch.Tensor
            Factor sequence, shape (batch, T, latent_dim)
        """
        batch_size, T, r = eta_sequence.shape
        assert r == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {r}"
        
        # Handle initial state
        if f0 is None:
            f0 = self.f0.unsqueeze(0).expand(batch_size, -1)
        elif f0.dim() == 1:
            f0 = f0.unsqueeze(0).expand(batch_size, -1)
        
        if self.factor_order == 1:
            # Optimized first-order path
            return self._forward_first_order(eta_sequence, f0)
        else:
            # Companion form path
            return self._forward_companion(eta_sequence, f0)
    
    def _forward_first_order(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """Optimized forward pass for first-order (p=1) dynamics.
        
        Implements: f_{t+1} = A * f_t + B * η_t
        where A and B are both diagonal (component-wise independence).
        """
        batch_size, T, r = eta_sequence.shape
        A_diag = torch.sigmoid(self.A)  # Ensure stability: A in (0, 1)
        B_diag = self.norm(self.B, ord=self.norm_order) if self.norm_order > 0 else self.B
        
        # Stable O(T) scan (T=sequence_length is typically ~100), avoids a^{-t} blowups.
        # Still fully vectorized across batch and factors.
        factors = torch.empty(batch_size, T, r, device=eta_sequence.device, dtype=eta_sequence.dtype)
        f_t = f0  # (batch, r)
        for t in range(T):
            f_t = A_diag * f_t + B_diag * eta_sequence[:, t, :]
            factors[:, t, :] = f_t
        return factors
    
    def _forward_companion(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for AR(p) dynamics using companion form + Krylov.
        
        Uses efficient convolution: f_t = Σ_{k=0}^{t-1} H_k η_{t-1-k}
        """
        batch_size, T, r = eta_sequence.shape
        
        # Get impulse response kernel
        H = self.get_impulse_response(T)  # (r, T)
        
        # Convolve innovations with impulse response
        # For each factor i: f_i[t] = Σ_{k=0}^{t-1} H[i, k] * η[i, t-1-k]
        # Use FFT convolution for efficiency
        
        # Rearrange: (batch, T, r) -> (batch, r, T)
        eta_T = rearrange(eta_sequence, 'b t r -> b r t')

        # Vectorized FFT convolution across all factors:
        # eta_T: (batch, r, T), H: (r, T) -> (batch, r, T)
        factors = self.fft_conv(eta_T, H)  # (batch, r, T)
        factors = rearrange(factors, 'b r t -> b t r')
        
        # Add initial state contribution: f_t += A^t * f_0
        # For companion form, need to compute A^t and extract first component
        ar_coeffs_norm = self.norm(self.ar_coeffs, ord=self.norm_order) if self.norm_order > 0 else self.ar_coeffs
        A_block = build_ivdfm_diagonal_companion(ar_coeffs_norm)
        
        # Augment initial state: s_0 = [f_0, 0, ..., 0]^T
        s0 = torch.zeros(batch_size, r * self.factor_order, device=f0.device, dtype=f0.dtype)
        s0[:, ::self.factor_order] = f0  # First component of each block
        
        # Compute A^t * s_0 for each batch element.
        # Use Krylov to get [s_0, A*s_0, A^2*s_0, ..., A^{T-1}*s_0]
        # Broadcast A across batch so each s0[b] is handled correctly.
        A_block_b = A_block.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, r*p, r*p)
        s_seq = krylov(T, A_block_b, s0, c=None)  # (batch, r*p, T)
        
        # Extract factors: C * s_seq
        C_norm = self.C if self.C is not None else torch.eye(r)
        f_init = torch.einsum('ij, bjt -> bit', C_norm, s_seq)  # (batch, r, T)
        f_init = rearrange(f_init, 'b r t -> b t r')
        
        # Add initial state contribution
        factors = factors + f_init
        
        return factors

"""Base State Space Model (SSM) for iVDFM.

This module provides the base SSM class that maps innovations to factors
using deterministic linear dynamics. The SSM implements the convolution:
    f_t = Σ_{k=0}^{t-1} H_k η_{t-1-k}
where H_k = C A^k B is the impulse response kernel.

Key differences from standard SSMs:
- Input: innovations η_t (not observations)
- Output: factors f_t (not processed observations)
- Block-diagonal structure for identifiability (one block per factor)
- Deterministic mapping (no stochasticity in SSM itself)
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from einops import rearrange


class BaseSSM(nn.Module, ABC):
    """Abstract base class for State Space Models in iVDFM.
    
    Maps innovations η_t to factors f_t via deterministic linear dynamics:
        f_t = Σ_{k=0}^{t-1} H_k η_{t-1-k}
    where H_k = C A^k B is the impulse response kernel.
    
    Subclasses should implement:
    - get_impulse_response(): Compute impulse response kernel H_k
    - forward(): Compute factors from innovation sequence
    """
    
    def __init__(
        self,
        latent_dim: int,
        factor_order: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Initialize BaseSSM.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of latent factors (r in paper)
        factor_order : int, default 1
            AR order for factor dynamics (p in AR(p))
        device : Optional[torch.device]
            Device for computation
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.factor_order = factor_order
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fft_conv(
        self,
        u_input: torch.Tensor,
        v_kernel: torch.Tensor
    ) -> torch.Tensor:
        """Convolve input with kernel using FFT (O(n log n) complexity).
        
        Parameters
        ----------
        u_input : torch.Tensor
            Input sequence, shape (batch, dim, length) or (batch, length, dim)
        v_kernel : torch.Tensor
            Convolution kernel, shape (..., length)
            
        Returns
        -------
        torch.Tensor
            Convolved output, same shape as u_input
        """
        # Ensure input is (batch, dim, length).
        # Avoid heuristic based on shape[-1] != shape[1] (can be true for correct (b, d, L) when d != L).
        if u_input.dim() == 3:
            # If kernel is provided as (dim, L), use it to disambiguate.
            if v_kernel.dim() == 2:
                k_dim = v_kernel.shape[0]
                if u_input.shape[1] == k_dim:
                    # Already (batch, dim, length)
                    pass
                elif u_input.shape[-1] == k_dim:
                    # (batch, length, dim) -> (batch, dim, length)
                    u_input = rearrange(u_input, 'b l d -> b d l')
            else:
                # Fall back: assume already (batch, dim, length)
                pass
        
        L = u_input.shape[-1]  # Sequence length
        batch_size, dim = u_input.shape[:2]
        
        # FFT convolution
        u_f = torch.fft.rfft(u_input, n=2*L, dim=-1)  # (batch, dim, L//2+1)
        
        # Handle kernel shape: could be (dim, L) or (L,) or (..., L)
        if v_kernel.dim() == 1:
            # Broadcast to (dim, L)
            v_kernel = v_kernel.unsqueeze(0).expand(dim, -1)
        elif v_kernel.dim() == 2 and v_kernel.shape[0] == dim:
            # Already (dim, L)
            pass
        else:
            # Try to broadcast: (..., L) -> (dim, L)
            if v_kernel.shape[-1] == L:
                v_kernel = v_kernel[..., :L]
                if v_kernel.dim() == 1:
                    v_kernel = v_kernel.unsqueeze(0).expand(dim, -1)
        
        v_f = torch.fft.rfft(v_kernel[..., :L], n=2*L, dim=-1)  # (dim, L//2+1)
        
        # Element-wise multiplication and inverse FFT
        y_f = u_f * v_f.unsqueeze(0)  # (batch, dim, L//2+1)
        y = torch.fft.irfft(y_f, n=2*L, dim=-1)[..., :L]  # (batch, dim, L)
        
        return y
    
    @abstractmethod
    def get_impulse_response(
        self,
        length: int
    ) -> torch.Tensor:
        """Compute impulse response kernel H_k = C A^k B.
        
        Parameters
        ----------
        length : int
            Length of impulse response sequence (T)
            
        Returns
        -------
        torch.Tensor
            Impulse response kernel, shape (latent_dim, length) or (latent_dim, latent_dim, length)
            Each row/block corresponds to one factor's impulse response
        """
        raise NotImplementedError
    
    @abstractmethod
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
        raise NotImplementedError
    
    def forward_closed_loop(
        self,
        f_current: torch.Tensor,
        eta_future: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """Rollout factors for forecasting (closed-loop).
        
        Computes: f_{t+h} = A^h f_t + Σ_{k=0}^{h-1} A^k B η_{t+h-k}
        
        Parameters
        ----------
        f_current : torch.Tensor
            Current factor state, shape (batch, latent_dim)
        eta_future : torch.Tensor
            Future innovations, shape (batch, horizon, latent_dim)
        horizon : int
            Forecast horizon
            
        Returns
        -------
        torch.Tensor
            Forecasted factors, shape (batch, horizon, latent_dim)
        """
        # Default implementation: use forward with zero-padded sequence
        # Subclasses can override for efficiency
        batch_size = f_current.shape[0]
        T = eta_future.shape[1]
        
        # Pad with zeros to get full sequence
        eta_padded = torch.cat([
            torch.zeros(batch_size, 1, self.latent_dim, device=eta_future.device),
            eta_future
        ], dim=1)
        
        # Use forward with initial state
        factors = self.forward(eta_padded, f0=f_current)
        
        # Return only future horizon
        return factors[:, 1:, :]  # Skip initial state
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = device
        return self

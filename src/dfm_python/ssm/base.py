"""Base state-space model of iVDFM: innovations to factors.

The SSM is the deterministic convolution ``f_t = sum_{k=0}^{t-1} H_k eta_{t-1-k}``
with the impulse response kernel ``H_k = C A^k B``. Its input is the
innovations ``eta_t`` (not observations), its output the factors ``f_t``, and
it is block-diagonal (one block per factor) for identifiability.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from einops import rearrange


class BaseSSM(nn.Module, ABC):
    """Abstract SSM mapping innovations ``eta_t`` to factors ``f_t``.

    Subclasses implement ``get_impulse_response`` and ``forward``.

    Parameters
    ----------
    latent_dim : int
        Number of factors (r).
    factor_order : int, default 1
        AR order p of the factor dynamics.
    device : torch.device, optional
        Defaults to CUDA when available, else CPU.
    """

    def __init__(
        self,
        latent_dim: int,
        factor_order: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.factor_order = factor_order
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fft_conv(
        self,
        u_input: torch.Tensor,
        v_kernel: torch.Tensor
    ) -> torch.Tensor:
        """Causal FFT convolution of ``u_input`` with ``v_kernel``, (batch, dim, length).

        ``u_input`` is (batch, dim, length); a (batch, length, dim) input is
        recognized (and transposed) only against a (dim, length) kernel. A 1-D
        kernel is used for every channel.
        """
        if (u_input.dim() == 3 and v_kernel.dim() == 2
                and u_input.shape[1] != v_kernel.shape[0] and u_input.shape[-1] == v_kernel.shape[0]):
            u_input = rearrange(u_input, 'b l d -> b d l')

        L = u_input.shape[-1]
        dim = u_input.shape[1]
        # Zero-padding to 2L makes the circular FFT convolution a linear one.
        u_f = torch.fft.rfft(u_input, n=2*L, dim=-1)  # (batch, dim, L+1)
        if v_kernel.dim() == 1:
            v_kernel = v_kernel.unsqueeze(0).expand(dim, -1)
        v_f = torch.fft.rfft(v_kernel[..., :L], n=2*L, dim=-1)  # (dim, L+1)

        y_f = u_f * v_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=2*L, dim=-1)[..., :L]  # (batch, dim, L)

        return y

    @abstractmethod
    def get_impulse_response(
        self,
        length: int
    ) -> torch.Tensor:
        """Impulse response ``H_k = C A^k B`` for ``k < length``, one row per factor.

        Shape (latent_dim, length), or (latent_dim, latent_dim, length).
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        eta_sequence: torch.Tensor,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Factors ``f_t = A^t f_0 + sum_{k<t} H_k eta_{t-1-k}`` of the innovations.

        ``eta_sequence`` is (batch, T, latent_dim); ``f0`` is (batch, latent_dim)
        or (latent_dim,). Returns (batch, T, latent_dim).
        """
        raise NotImplementedError

    def forward_closed_loop(
        self,
        f_current: torch.Tensor,
        eta_future: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """Forecast rollout ``f_{t+h} = A^h f_t + sum_{k=0}^{h-1} A^k B eta_{t+h-k}``.

        ``f_current`` is (batch, latent_dim) and ``eta_future`` (batch, horizon,
        latent_dim); returns (batch, horizon, latent_dim). The horizon is
        ``eta_future.shape[1]``; ``horizon`` is not read.
        """
        # forward() from f_current on a sequence with a leading zero innovation,
        # dropping that first step: forward() treats f0 as the state at its
        # first row. A model whose forward() treats f0 as the state before the
        # first row (the AR(1) scan of iVDFMCompanionSSM) overrides this.
        batch_size = f_current.shape[0]
        eta_padded = torch.cat([
            torch.zeros(batch_size, 1, self.latent_dim, device=eta_future.device),
            eta_future
        ], dim=1)
        factors = self.forward(eta_padded, f0=f_current)
        return factors[:, 1:, :]

    def to(self, device):
        """Move the module to ``device`` and record it in ``self.device``."""
        super().to(device)
        self.device = device
        return self

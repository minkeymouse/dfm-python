"""LongConv trading filter for the Attention Factor Model.

Implements the residual-history sequence model of Epstein et al. (2025), which
extracts the arbitrage portfolio weight from the past ``s`` residuals of each
asset:

    omega_port_{i,t-1} = LongConv_theta( eps_{i, (t-s, t-1)} ).

LongConv (Fu et al., 2023) is a long causal convolution with a learned kernel of
the full history length, computed in O(L log L) via the convolution theorem
K * u = IFFT( FFT(u) . FFT(K) ). Faithful to the paper's Appendix, this includes
the two ingredients it specifies from Fu et al. (2023):

  * the **Squash** kernel regularizer applied every forward pass,
        Kbar = sign(K) . max(|K| - lambda_squash, 0),
    a proximal step for an L1 penalty that sparsifies the kernel; and
  * the **geometric-decay initialization**,
        K^(h)_t = x . exp(-(t/T) . (d/2)^(h/d)),  x ~ N(0, 1),
    giving filters that act on both short and long time-scales.

Each asset's scalar residual history is one channel; ``n_kernels`` learned long
kernels produce features that a linear head reads out into the scalar portfolio
weight at the most recent step, with a per-kernel skip on the latest residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LongConv1d(nn.Module):
    """Long causal convolution mapping a length-``seq_len`` history to a scalar.

    Parameters
    ----------
    seq_len : int
        Residual-history length ``s`` consumed per asset.
    n_kernels : int
        Number of distinct long convolutions (32 in the tuned model of the
        paper). Each captures a different time-series pattern.
    squash_lambda : float
        Strength of the Squash (soft-threshold) kernel regularizer.
    """

    def __init__(self, seq_len: int, n_kernels: int = 32,
                 squash_lambda: float = 1e-3) -> None:
        super().__init__()
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        self.seq_len = int(seq_len)
        self.n_kernels = int(n_kernels)
        self.squash_lambda = float(squash_lambda)
        self.kernel = nn.Parameter(self._geometric_init(n_kernels, seq_len))
        # Per-kernel skip on the most recent residual (LongConv residual term).
        self.skip = nn.Parameter(torch.zeros(n_kernels))
        # Read out the per-kernel features at the final step into one weight.
        self.readout = nn.Linear(n_kernels, 1)

    @staticmethod
    def _geometric_init(d: int, T: int) -> torch.Tensor:
        """Geometric decay across sequence and hidden dims (Fu et al., 2023)."""
        h = torch.arange(1, d + 1, dtype=torch.float32).unsqueeze(1)     # (d, 1)
        t = torch.arange(1, T + 1, dtype=torch.float32).unsqueeze(0)     # (1, T)
        decay = torch.exp(-(t / T) * (d / 2.0) ** (h / d))              # (d, T)
        return torch.randn(d, T) * decay

    def _squash(self) -> torch.Tensor:
        """Soft-threshold the kernel: sign(K) * max(|K| - lambda, 0)."""
        k = self.kernel
        return torch.sign(k) * torch.clamp(k.abs() - self.squash_lambda, min=0.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map residual histories to portfolio weights.

        Parameters
        ----------
        u : torch.Tensor
            Residual history, shape ``(..., seq_len)`` (oldest first, latest last).

        Returns
        -------
        torch.Tensor
            Scalar weight per leading-axis element, shape ``(...)``.
        """
        if u.shape[-1] != self.seq_len:
            raise ValueError(
                f"history length {u.shape[-1]} != seq_len {self.seq_len}")
        kernel = self._squash()                              # regularized kernel
        n_fft = 2 * self.seq_len  # zero-pad for linear (non-circular) convolution
        uf = torch.fft.rfft(u, n=n_fft)                      # (..., F)
        kf = torch.fft.rfft(kernel, n=n_fft)                 # (n_kernels, F)
        yf = uf.unsqueeze(-2) * kf                           # (..., n_kernels, F)
        y = torch.fft.irfft(yf, n=n_fft)[..., : self.seq_len]  # (..., n_kernels, L)
        # Causal output at the final step (uses the whole history) + skip.
        y_last = y[..., -1] + self.skip * u[..., -1:]        # (..., n_kernels)
        return self.readout(y_last).squeeze(-1)              # (...)

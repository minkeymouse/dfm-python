"""Factor construction for the Attention Factor Model.

Two ways to build the conditional factor portfolios whose residuals are traded
(Epstein et al., 2025):

* :class:`AttentionFactors` -- learned. Firm characteristics are embedded and
  cross-sectional attention produces the factor-portfolio weight matrix
  ``omega_F = softmax(Q Xtilde^T / sqrt(d))``; loadings follow in closed form via
  a ridge pseudo-inverse (no regression).
* :class:`PCAFactors` -- the paper's benchmark. Factor weights are the top-K
  eigenvectors of the trailing return covariance; the same trading filter is
  applied downstream so the comparison isolates the value of learned factors.

Both yield ``omega_F`` (K x N) and the loadings ``beta^T`` (N x K); the residual
projection ``omega_eps = I_N - beta^T omega_F`` makes the residuals traded
portfolios, ``eps_t = omega_eps R_t = R_t - beta^T F_t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def ridge_loadings(omega_f: torch.Tensor, ridge: float) -> torch.Tensor:
    """Closed-form loadings from factor weights (paper Eq. for beta).

    ``beta^T = omega_F^T (omega_F omega_F^T + ridge I_K)^{-1}``.

    Parameters
    ----------
    omega_f : torch.Tensor
        Factor-portfolio weights, shape ``(..., K, N)``.
    ridge : float
        Ridge penalty for stability.

    Returns
    -------
    torch.Tensor
        Loadings ``beta^T``, shape ``(..., N, K)``.
    """
    K = omega_f.shape[-2]
    gram = omega_f @ omega_f.transpose(-1, -2)                       # (..., K, K)
    eye = torch.eye(K, dtype=omega_f.dtype, device=omega_f.device)
    inv = torch.linalg.inv(gram + ridge * eye)                      # (..., K, K)
    return omega_f.transpose(-1, -2) @ inv                          # (..., N, K)


def residuals(omega_f: torch.Tensor, beta_t: torch.Tensor,
              returns: torch.Tensor) -> torch.Tensor:
    """Residual portfolios ``eps_t = R_t - beta^T (omega_F R_t)``.

    Parameters
    ----------
    omega_f : torch.Tensor
        Factor weights ``(..., K, N)``.
    beta_t : torch.Tensor
        Loadings ``beta^T`` ``(..., N, K)``.
    returns : torch.Tensor
        Asset returns ``(..., N)``.

    Returns
    -------
    torch.Tensor
        Residuals ``(..., N)``.
    """
    factors = (omega_f @ returns.unsqueeze(-1)).squeeze(-1)          # (..., K)
    fitted = (beta_t @ factors.unsqueeze(-1)).squeeze(-1)            # (..., N)
    return returns - fitted


class AttentionFactors(nn.Module):
    """Learned conditional factors via cross-sectional attention.

    Parameters
    ----------
    n_char : int
        Number of firm characteristics ``M``.
    n_factors : int
        Number of latent factors ``K``.
    embed_dim : int
        Characteristic embedding dimension ``d``.
    ridge : float
        Ridge penalty for the closed-form loadings.
    """

    def __init__(self, n_char: int, n_factors: int, embed_dim: int = 32,
                 ridge: float = 1e-2) -> None:
        super().__init__()
        self.n_char = int(n_char)
        self.n_factors = int(n_factors)
        self.embed_dim = int(embed_dim)
        self.ridge = float(ridge)
        self.embed = nn.Parameter(
            torch.randn(n_char, embed_dim) / math.sqrt(n_char)
        )  # W^K
        self.query = nn.Parameter(
            torch.randn(n_factors, embed_dim) / math.sqrt(embed_dim)
        )  # Q

    def forward(self, characteristics: torch.Tensor):
        """Compute factor weights and loadings from characteristics.

        Parameters
        ----------
        characteristics : torch.Tensor
            Firm characteristics ``X_{t-1}``, shape ``(..., N, M)``.

        Returns
        -------
        tuple of torch.Tensor
            ``omega_F`` ``(..., K, N)`` and ``beta^T`` ``(..., N, K)``.
        """
        embedded = characteristics @ self.embed                     # (..., N, d)
        scores = self.query @ embedded.transpose(-1, -2)            # (..., K, N)
        scores = scores / math.sqrt(self.embed_dim)
        omega_f = torch.softmax(scores, dim=-1)                     # normalize over assets
        beta_t = ridge_loadings(omega_f, self.ridge)
        return omega_f, beta_t


class PCAFactors:
    """PCA benchmark factors (non-learned): top-K eigenvectors of return cov."""

    def __init__(self, n_factors: int, ridge: float = 1e-2) -> None:
        self.n_factors = int(n_factors)
        self.ridge = float(ridge)

    def __call__(self, return_window: torch.Tensor):
        """Factor weights/loadings from a trailing return window.

        Parameters
        ----------
        return_window : torch.Tensor
            Trailing returns ``(..., L, N)`` used to estimate the covariance.

        Returns
        -------
        tuple of torch.Tensor
            ``omega_F`` ``(..., K, N)`` and ``beta^T`` ``(..., N, K)``.
        """
        x = return_window - return_window.mean(dim=-2, keepdim=True)
        L = x.shape[-2]
        cov = x.transpose(-1, -2) @ x / max(L - 1, 1)               # (..., N, N)
        # eigh returns ascending eigenvalues; take the top-K eigenvectors.
        _, vecs = torch.linalg.eigh(cov)
        omega_f = vecs[..., -self.n_factors:].transpose(-1, -2)     # (..., K, N)
        beta_t = ridge_loadings(omega_f, self.ridge)
        return omega_f, beta_t

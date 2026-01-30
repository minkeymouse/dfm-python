"""Regime probability network.

Outputs π_t = softmax(h(u_t) / τ) from observed context only.
Temperature τ < 1 sharpens π toward one-hot (local commitment); τ = 1 is standard softmax.
No latent inputs; no gradients into η through π.
"""

from typing import Optional
import torch
import torch.nn as nn

from .mlp import MLP


class RegimeNet(nn.Module):
    """Maps context u to regime probabilities π over K regimes. Supports temperature for sharper assignments."""

    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        hidden_dim: int = 32,
        n_hidden_layers: int = 1,
        activation: str = "lrelu",
        slope: float = 0.1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        if n_hidden_layers < 1:
            self.net = nn.Linear(input_dim, num_regimes)
        else:
            self.net = MLP(
                input_dim=input_dim,
                output_dim=num_regimes,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation=activation,
                slope=slope,
                device=device,
                seed=seed,
            )
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return regime probabilities (batch, ..., K). τ < 1 sharpens π (local commitment)."""
        logits = self.net(x)
        if temperature != 1.0:
            logits = logits / temperature
        return torch.softmax(logits, dim=-1)

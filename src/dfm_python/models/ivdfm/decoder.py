"""Decoder for iVDFM.

Maps latent factors f_t to observations y_t. Three modes:
- linear: y_t = W f_t + b
- residual: y_t = W f_t + b + h(f_t), h small and non-temporal
- mlp: y_t = g(f_t)
"""

from typing import Union, List, Optional
import torch
import torch.nn as nn

from ..base import BaseDecoder
from ...layer.mlp import MLP
from ...logger import get_logger

_logger = get_logger(__name__)


class iVDFMDecoder(BaseDecoder):
    """Decoder for iVDFM with modes: linear, residual, mlp."""

    def __init__(
        self,
        latent_dim: int,
        data_dim: int,
        decoder_type: str = "linear",
        hidden_dim: Union[int, List[int]] = 32,
        n_hidden_layers: int = 1,
        activation: str = "lrelu",
        slope: float = 0.1,
        use_layer_norm: bool = False,
        decoder_var: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if decoder_type not in ("linear", "residual", "mlp"):
            raise ValueError(f"decoder_type must be 'linear', 'residual', or 'mlp', got {decoder_type!r}")

        self.decoder_type = decoder_type
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        if decoder_type in ("linear", "residual"):
            self.linear = nn.Linear(latent_dim, data_dim)
        else:
            self.linear = None

        if decoder_type in ("residual", "mlp"):
            self.mlp = MLP(
                input_dim=latent_dim,
                output_dim=data_dim,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation=activation,
                slope=slope,
                use_layer_norm=use_layer_norm,
                device=device,
                seed=seed,
            )
        else:
            self.mlp = None

        self.register_buffer(
            "decoder_variance",
            torch.tensor(decoder_var, dtype=torch.float32),
        )

        if device is not None:
            self.to(device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.decoder_type == "linear":
            return self.linear(f)
        if self.decoder_type == "residual":
            return self.linear(f) + self.mlp(f)
        return self.mlp(f)

    def forward_with_variance(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward(f)
        var = self.decoder_variance
        if var.dim() < mean.dim():
            var = var.view(*([1] * (mean.dim() - var.dim())), *var.shape).expand_as(mean)
        return mean, var

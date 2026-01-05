"""Linear decoder network for DDFM."""

import torch
import torch.nn as nn
from typing import Optional

from ..config.constants import DEFAULT_ZERO_VALUE


class Decoder(nn.Module):
    """Linear decoder network for DDFM."""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_normal_(self.decoder.weight)
        if self.decoder.bias is not None:
            nn.init.constant_(self.decoder.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)

"""Linear decoder network for DDFM."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..config.constants import DEFAULT_ZERO_VALUE, DEFAULT_XAVIER_GAIN
from ..logger import get_logger

_logger = get_logger(__name__)


class LinearDecoder(nn.Module):
    """Linear decoder network for DDFM."""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        if seed is not None:
            torch.manual_seed(seed)
        # Use explicit gain for consistency with encoder initialization
        nn.init.xavier_normal_(self.decoder.weight, gain=DEFAULT_XAVIER_GAIN)
        if self.decoder.bias is not None:
            nn.init.constant_(self.decoder.bias, DEFAULT_ZERO_VALUE)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)
    
    def extract_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract observation matrix C and bias from decoder.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (weight, bias) where weight is (output_dim, input_dim) and bias is (output_dim,)
        """
        weight = self.decoder.weight.data.cpu().numpy()
        bias = self.decoder.bias.data.cpu().numpy() if self.decoder.bias is not None else np.zeros(weight.shape[0])
        
        if np.any(np.isnan(weight)):
            _logger.warning("LinearDecoder: C matrix contains NaN values. Replacing with zeros.")
            weight = np.nan_to_num(weight, nan=0.0)
        
        return weight, bias

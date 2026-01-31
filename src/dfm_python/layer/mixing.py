"""Post-factor mixing layer for iVDFM.

Global linear map f -> M f (r -> r) before decode when mixing=True. Identity init; no bias.
"""

from typing import Optional
import torch
import torch.nn as nn


class FactorMixer(nn.Module):
    """Linear mixing of factors before decoder: mixed = M @ f. Identity init by default."""

    def __init__(self, r: int, init: str = "identity", device: Optional[torch.device] = None):
        super().__init__()
        self.r = r
        self.M = nn.Linear(r, r, bias=False)
        if init == "identity":
            nn.init.eye_(self.M.weight)
        if device is not None:
            self.to(device)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply mixing: (..., r) -> (..., r)."""
        return self.M(f)


def build_factor_mixing(
    r: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> FactorMixer:
    """Build global mixing module (single M, identity init). Used when config mixing=True."""
    module = FactorMixer(r=r, init="identity", device=device)
    if dtype is not None:
        module = module.to(dtype=dtype)
    return module

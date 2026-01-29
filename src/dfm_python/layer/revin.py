"""Reversible Instance Normalization (RevIN) for time series.

RevIN normalizes each instance (time series) by subtracting its mean and dividing
by its standard deviation. This is reversible - we can denormalize predictions
by multiplying by std and adding mean back.

This helps with distribution shift in time series data, as used in models like
PatchTST and iTransformer.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RevIN(nn.Module):
    """Reversible Instance Normalization.
    
    Normalizes each instance along the time dimension:
    - Normalize: x_norm = (x - mean) / std
    - Denormalize: x = x_norm * std + mean
    
    This is reversible and helps with distribution shift.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """Initialize RevIN.
        
        Parameters
        ----------
        num_features : int
            Number of features (variables) in the time series
        eps : float, default 1e-5
            Small epsilon for numerical stability
        affine : bool, default True
            If True, learnable affine transformation (scale and shift parameters)
            If False, simple normalization without learnable parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            # Learnable affine transformation parameters
            self._init_params()
    
    def _init_params(self):
        """Initialize learnable parameters."""
        # Initialize scale and shift to identity (no transformation initially)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def _get_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and standard deviation along time dimension.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, T, num_features)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (mean, std) where:
            - mean: shape (batch, 1, num_features)
            - std: shape (batch, 1, num_features)
        """
        # Compute statistics along time dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, num_features)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (batch, 1, num_features)
        std = torch.sqrt(var + self.eps)  # (batch, 1, num_features)
        return mean, std
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """Forward pass: normalize or denormalize.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, T, num_features)
        mode : str, default 'norm'
            'norm': normalize (subtract mean, divide by std)
            'denorm': denormalize (multiply by std, add mean)
            
        Returns
        -------
        torch.Tensor
            Normalized or denormalized tensor, same shape as input
        """
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            return self.denormalize(x)
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")
    
    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize input: x_norm = (x - mean) / std.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, T, num_features)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (x_norm, mean, std) where:
            - x_norm: normalized tensor, shape (batch, T, num_features)
            - mean: mean values, shape (batch, 1, num_features)
            - std: std values, shape (batch, 1, num_features)
        """
        mean, std = self._get_statistics(x)
        x_norm = (x - mean) / std
        
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight.unsqueeze(0).unsqueeze(0) + \
                     self.affine_bias.unsqueeze(0).unsqueeze(0)
        
        return x_norm, mean, std
    
    def denormalize(self, x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize input: x = x_norm * std + mean.
        
        Parameters
        ----------
        x_norm : torch.Tensor
            Normalized tensor, shape (batch, T, num_features)
        mean : torch.Tensor
            Mean values, shape (batch, 1, num_features)
        std : torch.Tensor
            Standard deviation values, shape (batch, 1, num_features)
            
        Returns
        -------
        torch.Tensor
            Denormalized tensor, shape (batch, T, num_features)
        """
        # Reverse affine transformation if enabled
        if self.affine:
            x_norm = (x_norm - self.affine_bias.unsqueeze(0).unsqueeze(0)) / \
                     (self.affine_weight.unsqueeze(0).unsqueeze(0) + self.eps)
        
        # Denormalize: x = x_norm * std + mean
        x = x_norm * std + mean
        return x

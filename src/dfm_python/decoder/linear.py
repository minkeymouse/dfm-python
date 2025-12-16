"""Linear decoder network for DDFM.

Maps latent factors f_t back to observed variables X_t using a linear transformation.
This preserves interpretability and allows Kalman filtering.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    torch = None
    nn = None

try:
    import torch
    import torch.nn as nn
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None


if _has_torch:
    class Decoder(nn.Module):
        """Linear decoder network for DDFM.
        
        Maps latent factors f_t back to observed variables X_t using a linear transformation.
        This preserves interpretability and allows Kalman filtering.
        """
        
        def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
            """Initialize linear decoder.
            
            Parameters
            ----------
            input_dim : int
                Number of factors (input dimension)
            output_dim : int
                Number of series (output dimension)
            use_bias : bool
                Whether to use bias term
            """
            super().__init__()
            self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        def forward(self, f: "torch.Tensor") -> "torch.Tensor":
            """Forward pass through decoder.
            
            Parameters
            ----------
            f : torch.Tensor
                Factors (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Reconstructed observations (batch_size x output_dim)
            """
            return self.decoder(f)
else:
    # Placeholder class when PyTorch is not available
    class Decoder:
        """Placeholder Decoder class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM. Install with: pip install dfm-python[deep]")

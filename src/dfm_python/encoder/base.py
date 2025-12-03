"""Base encoder class for factor extraction.

This module provides the abstract base class for all encoder implementations.
Encoders extract latent factors from observed time series data.
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import torch

from ..logger import get_logger

_logger = get_logger(__name__)


class BaseEncoder(ABC):
    """Abstract base class for factor encoders.
    
    This class defines the common interface for all encoder implementations,
    including PCA, VAE, and future encoders (e.g., diffusion-based).
    
    Encoders extract latent factors f_t from observed variables X_t:
        f_t = encode(X_t)
    
    Subclasses should implement:
    - encode(): Extract factors from data
    - fit() (optional): Train/initialize the encoder
    """
    
    def __init__(self, n_components: int):
        """Initialize encoder.
        
        Parameters
        ----------
        n_components : int
            Number of factors to extract
        """
        self.n_components = n_components
    
    @abstractmethod
    def encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Extract factors from observed data.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Observed data (T x N) where T is time periods and N is number of series
        **kwargs
            Additional encoder-specific parameters
            
        Returns
        -------
        factors : np.ndarray or torch.Tensor
            Extracted factors (T x n_components)
        """
        pass
    
    def fit(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> "BaseEncoder":
        """Fit/train the encoder on data.
        
        For some encoders (e.g., PCA), this computes statistics.
        For others (e.g., VAE), this trains the model.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Training data (T x N)
        **kwargs
            Additional training parameters
            
        Returns
        -------
        self : BaseEncoder
            Returns self for method chaining
        """
        # Default implementation: no-op (for encoders that don't need training)
        return self
    
    def fit_encode(
        self,
        X: Union[np.ndarray, "torch.Tensor"],
        **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Fit encoder and extract factors in one step.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Observed data (T x N)
        **kwargs
            Additional parameters
            
        Returns
        -------
        factors : np.ndarray or torch.Tensor
            Extracted factors (T x n_components)
        """
        self.fit(X, **kwargs)
        return self.encode(X, **kwargs)


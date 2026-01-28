"""Abstract base class for factor models and model components."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ..config import BaseResult
from ..utils.errors import ModelNotTrainedError

class BaseFactorModel(ABC):
    """Abstract base class for all factor models."""
    
    def __init__(self):
        """Initialize factor model instance."""
        self._config: Optional[Any] = None
        self._result: Optional[BaseResult] = None
        self.training_state: Optional[Any] = None
        self._dataset: Optional[Any] = None
        self.data_processed: Optional[np.ndarray] = None
    
    def _ensure_result(self) -> BaseResult:
        """Ensure result exists, computing it if necessary."""
        if self._result is None:
            if self.training_state is None:
                raise ModelNotTrainedError("Model has not been trained yet")
            self._result = self.get_result()
        return self._result
    
    def reset(self) -> 'BaseFactorModel':
        """Reset model state."""
        self._config = None
        self._result = None
        self.training_state = None
        self._dataset = None
        return self
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """Fit the model."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Predict future values."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    @abstractmethod
    def update(self, data: Union[np.ndarray, Any], *args, **kwargs) -> None:
        """Update model state with new observations."""
        raise NotImplementedError("Subclasses must implement update()")
    
    @abstractmethod
    def get_result(self) -> BaseResult:
        """Extract result from trained model.
        
        Returns
        -------
        BaseResult
            Model-specific result object
        """
        raise NotImplementedError("Subclasses must implement get_result()")
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        raise NotImplementedError("Subclasses must implement save()")
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path], *args, **kwargs) -> 'BaseFactorModel':
        """Load model from file."""
        raise NotImplementedError("Subclasses must implement load()")


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for model encoders.
    
    Model-specific encoder base class. Encoders extract latent factors
    from observed data. This is a PyTorch module for neural encoders.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        
        Returns
        -------
        torch.Tensor
            Encoded representation (factors)
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for model decoders.
    
    Model-specific decoder base class. Decoders map latent factors
    to observed data. This is a PyTorch module for neural decoders.
    """
    
    @abstractmethod
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Parameters
        ----------
        f : torch.Tensor
            Latent factors
        
        Returns
        -------
        torch.Tensor
            Reconstructed observations
        """
        pass

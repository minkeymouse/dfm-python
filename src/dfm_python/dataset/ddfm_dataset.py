"""PyTorch Dataset classes for Deep Dynamic Factor Model (DDFM).

This module provides dataset implementations for DDFM training:
- DDFMDataset: Windowed sequences for batch training
- DDFMMCDataset: Monte Carlo samples for denoising training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Union, Callable, TYPE_CHECKING
from ..logger import get_logger
from ..config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_EPSILON
from .base import DeepFactorModelDataset

if TYPE_CHECKING:
    from ..models.ddfm import DDFM

_logger = get_logger(__name__)


class DDFMDataset(DeepFactorModelDataset):
    """PyTorch Dataset for windowed time series data.
    
    This dataset handles windowed sequences for PyTorch-based models that require
    batch training on overlapping windows (e.g., neural network-based DFM models).
    It creates overlapping windows from the time series for batch training.
    
    Parameters
    ----------
    data : torch.Tensor or np.ndarray
        Data tensor/array (T x N) where T is time periods and N is number of series
    window_size : int
        Window size for creating sequences
    stride : int, default 1
        Stride for windowing. Default 1 means overlapping windows.
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        window_size: int,
        stride: int = 1
    ):
        """Initialize DDFM dataset with windowing.
        
        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Data tensor/array (T x N) where T is time periods and N is number of series
        window_size : int
            Window size for creating sequences
        stride : int, default 1
            Stride for windowing. Default 1 means overlapping windows.
        """
        super().__init__(data)
        
        self.window_size = window_size
        self.stride = stride
        
        if self.window_size > self.T:
            _logger.warning(
                f"window_size ({self.window_size}) > sequence length ({self.T}). "
                f"Using full sequence as single window."
            )
            self.window_size = self.T
        
        # Compute number of samples
        if self.window_size >= self.T:
            self.n_samples = 1
        else:
            self.n_samples = (self.T - self.window_size) // stride + 1
    
    def __len__(self) -> int:
        """Return number of windowed samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a windowed data sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        x : torch.Tensor
            Input data window (window_size x N)
        target : torch.Tensor
            Target data (same as x for autoencoder/reconstruction)
        """
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for {self.n_samples} samples")
        
        if self.window_size >= self.T:
            # Return full sequence
            x = self.data
        else:
            # Return window
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size
            x = self.data[start_idx:end_idx, :]
        
        # For autoencoder/reconstruction tasks, target is same as input
        target = x.clone()
        
        return x, target


class DDFMMCDataset(Dataset):
    """PyTorch Dataset for DDFM Monte Carlo sampling.
    
    This dataset generates MC samples for denoising training. It:
    - Takes filtered data (after subtracting AR-idio mean)
    - Generates MC noise samples from idiosyncratic distribution
    - Creates corrupted inputs by subtracting noise
    - Returns all MC samples as a single batch for vectorized processing
    
    Parameters
    ----------
    data_mod : torch.Tensor
        Filtered data (T x N), after subtracting AR-idio mean
    data_mod_only_miss : torch.Tensor
        Original data with missing values (T x N)
    missing_mask : np.ndarray
        Missing data mask (T x N), True where data is missing
    n_mc_samples : int
        Number of MC samples to generate
    get_phi_fn : Callable[[], torch.Tensor]
        Function to get current Phi (AR coefficients) from model buffers
    get_sigma_eps_fn : Callable[[], torch.Tensor]
        Function to get current Sigma_eps (idiosyncratic std) from model buffers
    rng : np.random.RandomState
        Random number generator for MC sampling
    """
    
    def __init__(
        self,
        data_mod: torch.Tensor,
        data_mod_only_miss: torch.Tensor,
        missing_mask: np.ndarray,
        n_mc_samples: int,
        get_phi_fn: Callable[[], torch.Tensor],
        get_sigma_eps_fn: Callable[[], torch.Tensor],
        rng: np.random.RandomState
    ):
        """Initialize DDFM MC dataset.
        
        Parameters
        ----------
        data_mod : torch.Tensor
            Filtered data (T x N), after subtracting AR-idio mean
        data_mod_only_miss : torch.Tensor
            Original data with missing values (T x N)
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
        n_mc_samples : int
            Number of MC samples to generate
        get_phi_fn : Callable[[], torch.Tensor]
            Function to get current Phi (AR coefficients) from model buffers
        get_sigma_eps_fn : Callable[[], torch.Tensor]
            Function to get current Sigma_eps (idiosyncratic std) from model buffers
        rng : np.random.RandomState
            Random number generator for MC sampling
        """
        if isinstance(data_mod, np.ndarray):
            self.data_mod = torch.tensor(data_mod, dtype=DEFAULT_TORCH_DTYPE)
        else:
            self.data_mod = data_mod.float() if data_mod.dtype != DEFAULT_TORCH_DTYPE else data_mod
        
        if isinstance(data_mod_only_miss, np.ndarray):
            self.data_mod_only_miss = torch.tensor(data_mod_only_miss, dtype=DEFAULT_TORCH_DTYPE)
        else:
            self.data_mod_only_miss = data_mod_only_miss.float() if data_mod_only_miss.dtype != DEFAULT_TORCH_DTYPE else data_mod_only_miss
        
        self.missing_mask = missing_mask
        self.n_mc_samples = n_mc_samples
        self.get_phi_fn = get_phi_fn
        self.get_sigma_eps_fn = get_sigma_eps_fn
        self.rng = rng
        
        self.T, self.N = self.data_mod.shape
        
        # Cache mask tensor to avoid recreating it every __getitem__ call
        device = self.data_mod.device
        dtype = self.data_mod.dtype
        mask_tensor = torch.tensor(~self.missing_mask, device=device, dtype=dtype)
        self._mask_tensor = mask_tensor  # (T, N) - will be expanded in __getitem__
        
        # For MC sampling, we return all samples as a single batch
        # So dataset length is 1 (one batch containing all MC samples)
        self.n_samples = 1
    
    def __len__(self) -> int:
        """Return number of samples (always 1 for MC dataset - all samples in one batch)."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate MC samples and return corrupted inputs, targets, and mask.
        
        Parameters
        ----------
        idx : int
            Sample index (ignored, always returns all MC samples)
            
        Returns
        -------
        x_corrupted : torch.Tensor
            Corrupted inputs with MC noise injected, shape (n_mc_samples, T, N)
        x_target : torch.Tensor
            Target observations, shape (n_mc_samples, T, N)
        mask : torch.Tensor
            Missing data mask, shape (n_mc_samples, T, N)
        """
        if idx != 0:
            raise IndexError(f"DDFMMCDataset only has 1 sample (all MC samples), got idx={idx}")
        
        # Get current Phi and Sigma_eps from model buffers
        Phi = self.get_phi_fn()  # (N, N)
        Sigma_eps = self.get_sigma_eps_fn()  # (N,) or (N, N)
        
        # Convert to numpy for MC sampling
        Phi_np = Phi.detach().cpu().numpy()
        if Sigma_eps.ndim == 0:
            std_eps = np.ones(self.N) * float(Sigma_eps.item())
        elif Sigma_eps.ndim == 1:
            std_eps = Sigma_eps.detach().cpu().numpy()
        else:
            # Full covariance - extract diagonal
            std_eps = np.sqrt(np.diag(Sigma_eps.detach().cpu().numpy()))
        
        # Ensure std_eps is positive
        std_eps = np.maximum(std_eps, DEFAULT_EPSILON)
        
        # Generate MC samples for idio (dims = n_mc_samples x T x N)
        mu_eps = np.zeros(self.N)
        try:
            # Generate all samples at once
            eps_draws_flat = self.rng.multivariate_normal(
                mu_eps, np.diag(std_eps), size=(self.n_mc_samples * self.T)
            )
            eps_draws = eps_draws_flat.reshape(self.n_mc_samples, self.T, self.N)
        except (ValueError, np.linalg.LinAlgError) as e:
            _logger.warning(
                f"DDFMMCDataset: failed to generate MC samples: {e}. Using zero samples as fallback"
            )
            eps_draws = np.zeros((self.n_mc_samples, self.T, self.N))
        
        # Convert to torch tensors
        device = self.data_mod.device
        dtype = self.data_mod.dtype
        eps_draws_tensor = torch.tensor(eps_draws, device=device, dtype=dtype)
        
        # Create corrupted inputs: subtract MC noise from filtered data
        # data_mod shape: (T, N), eps_draws shape: (n_mc_samples, T, N)
        # Broadcast data_mod to (1, T, N) then subtract
        x_corrupted = self.data_mod[None, :, :] - eps_draws_tensor  # (n_mc_samples, T, N)
        
        # Target is the original data (with missing values)
        # Broadcast to match MC samples
        x_target = self.data_mod_only_miss[None, :, :].expand(self.n_mc_samples, -1, -1)  # (n_mc_samples, T, N)
        
        # Missing data mask (use cached tensor, broadcast to match MC samples)
        mask = self._mask_tensor[None, :, :].expand(self.n_mc_samples, -1, -1)  # (n_mc_samples, T, N)
        
        return x_corrupted, x_target, mask
    
    @staticmethod
    def create_from_model(
        model: 'DDFM',
        data_mod: torch.Tensor,
        data_mod_only_miss: torch.Tensor,
        missing_mask: np.ndarray
    ) -> 'DDFMMCDataset':
        """Create DDFMMCDataset instance from model and preprocessed data.
        
        This factory method encapsulates the dataset creation logic, moving it
        from the model class to the dataset class for better separation of concerns.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance with initialized buffers (Phi, Sigma_eps)
        data_mod : torch.Tensor
            Filtered data (T x N), after subtracting AR-idio mean
        data_mod_only_miss : torch.Tensor
            Original data with missing values (T x N)
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
            
        Returns
        -------
        DDFMMCDataset
            Initialized MC dataset instance
            
        Examples
        --------
        >>> from dfm_python.dataset.ddfm_dataset import DDFMMCDataset
        >>> dataset = DDFMMCDataset.create_from_model(
        ...     model=model,
        ...     data_mod=data_mod,
        ...     data_mod_only_miss=data_mod_only_miss,
        ...     missing_mask=missing_mask
        ... )
        """
        return DDFMMCDataset(
            data_mod=data_mod,
            data_mod_only_miss=data_mod_only_miss,
            missing_mask=missing_mask,
            n_mc_samples=model.n_mc_samples,
            get_phi_fn=lambda: model.Phi,
            get_sigma_eps_fn=lambda: model.Sigma_eps,
            rng=model.rng
        )


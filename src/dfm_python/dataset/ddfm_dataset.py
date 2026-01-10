"""PyTorch Dataset for Deep Dynamic Factor Model (DDFM)."""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
try:
    import polars as pl
    _has_polars = True
    PolarsDataFrame = pl.DataFrame
except ImportError:
    pl = None
    _has_polars = False
    PolarsDataFrame = type(None)  # Dummy type for type hints when polars not available
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ..config.constants import DEFAULT_TORCH_DTYPE


class DDFMDataset(Dataset):
    """Dataset for DDFM training.
    
    Scales target series if scaler provided. Feature series are used as-is.
    
    Parameters
    ----------
    data : pd.DataFrame | PolarsDataFrame
        Input data. Target series will be scaled if scaler provided.
    time_idx : str
        Time index column name.
    target_series : List[str], optional
        Target series column names to scale. If None or empty, all columns will be used as targets.
    target_scaler : StandardScaler | RobustScaler | MinMaxScaler, optional
        Scaler instance to scale target series. If None, no scaling.
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, PolarsDataFrame],
        time_idx: str,
        target_series: Optional[List[str]] = None,
        target_scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None,
    ):
        if _has_polars and isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        
        data = data.copy()
        data.sort_index(inplace=True)
        
        self.time_idx = time_idx
        
        # Extract time index values
        if time_idx and time_idx in data.columns:
            self.time_index = pd.Index(data[time_idx])
        else:
            self.time_index = data.index
        
        # If target_series is None or empty, use all columns as targets
        if target_series is None or len(target_series) == 0:
            target_series_list = list(data.columns)
        else:
            target_series_list = list(target_series)
            missing_cols = [col for col in target_series_list if col not in data.columns]
            if missing_cols:
                raise ValueError(
                    f"target_series columns {missing_cols} not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
        self.target_series = target_series_list
        self.data_original = data.copy()
        self.target_scaler = target_scaler
        
        y = data[target_series_list]
        X = data.drop(columns=target_series_list)
        
        # Check if data is already standardized (mean≈0, std≈1) to avoid double scaling
        # If data is already scaled, we still need target_scaler for inverse transformation
        # but we should NOT scale again
        if target_scaler is not None:
            # Check if target data appears already standardized
            y_mean = y.mean().abs().max()
            y_std = (y.std() - 1.0).abs().max()
            is_already_scaled = y_mean < 0.1 and y_std < 0.1
            
            if is_already_scaled:
                # Data is already standardized - don't scale again
                # But still fit the scaler if not already fitted (for inverse transformation)
                # The scaler should have been fitted on original unscaled data by the caller
                if not hasattr(target_scaler, 'mean_') or target_scaler.mean_ is None:
                    # If scaler not fitted, fit it on current data (assumes caller passed unscaled data)
                    self.target_scaler.fit(y.values)
                # Don't transform - data is already scaled
            else:
                # Data is not standardized - apply scaling
                self.target_scaler.fit(y.values)
                y_scaled = self.target_scaler.transform(y.values)
                y = pd.DataFrame(y_scaled, index=y.index, columns=y.columns)
        
        self.data = pd.concat([X, y], axis=1)
        self.X = X.values
        self.y = y.values
        self.missing_y = y.isna().values
        self.observed_y = ~self.missing_y

    @property
    def target_nan_ratio(self) -> float:
        """Target interpolate ratio."""
        return self.missing_y.sum() / self.missing_y.size

    @property
    def target_shape(self) -> Tuple[int, int]:
        """Target shape."""
        return self.y.shape
    
    @property
    def feature_shape(self) -> Tuple[int, int]:
        """Feature shape."""
        return self.X.shape

    @property
    def data_shape(self) -> Tuple[int, int]:
        """Data shape."""
        return self.feature_shape[0], self.feature_shape[1] + self.target_shape[1]

    @property
    def colnames(self) -> List[str]:
        """Column names from original data."""
        return list(self.data_original.columns)

    @property
    def target_columns(self) -> List[str]:
        """Target series column names."""
        return self.target_series
    
    @property
    def feature_columns(self) -> List[str]:
        """Feature column names (non-target series)."""
        return [col for col in self.colnames if col not in self.target_series]
    
    @property
    def all_columns_are_targets(self) -> bool:
        """Whether all columns are target series."""
        return len(self.target_series) == len(self.colnames)
    
    def split_features_and_targets(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Split DataFrame into features (X) and targets (y).
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to split
            
        Returns
        -------
        X : Optional[pd.DataFrame]
            Features DataFrame (None if all columns are targets)
        y : pd.DataFrame
            Targets DataFrame (all columns if all columns are targets, target columns only otherwise)
        """
        if self.all_columns_are_targets:
            return None, data
        else:
            X = data.drop(columns=self.target_series)
            y = data[self.target_series]
            return X, y
    
    @property
    def target_indices(self) -> np.ndarray:
        """Target series column indices in original data."""
        return np.array([self.colnames.index(col) for col in self.target_series])
    
    @classmethod
    def from_dataset(cls, new_data: Union[pd.DataFrame, PolarsDataFrame], dataset: 'DDFMDataset') -> 'DDFMDataset':
        """Create new dataset with new data, preserving configuration.
        
        Parameters
        ----------
        new_data : pd.DataFrame | PolarsDataFrame
            New data (same columns as original).
        dataset : DDFMDataset
            Original dataset to copy configuration from.
            
        Returns
        -------
        DDFMDataset
            New dataset with same time_idx, target_series, target_scaler.
        """
        return cls(
            data=new_data,
            time_idx=dataset.time_idx,
            target_series=dataset.target_series,
            target_scaler=dataset.target_scaler
        )
    
    def create_autoencoder_dataset(
        self,
        X: Optional[torch.Tensor],
        y_tmp: torch.Tensor,
        y_actual: torch.Tensor,
        eps_draw: torch.Tensor
    ) -> 'AutoencoderDataset':
        """Create a single AutoencoderDataset with corrupted targets.
        
        Parameters
        ----------
        X : torch.Tensor, optional
            Features (T, N_features) - already on device.
        y_tmp : torch.Tensor
            Target data (T, num_target_series) - already on device.
        y_actual : torch.Tensor
            Clean targets (T, num_target_series) - already on device.
        eps_draw : torch.Tensor
            Noise sample (T, num_target_series) - already on device.
            
        Returns
        -------
        AutoencoderDataset
            Dataset with corrupted targets.
        """
        y_corrupted = y_tmp - eps_draw
        return AutoencoderDataset(
            X=X,
            y_corrupted=y_corrupted,
            y_clean=y_actual
        )
    
    def create_pretrain_dataset(
        self,
        data: pd.DataFrame,
        device: Optional[torch.device] = None
    ) -> 'AutoencoderDataset':
        """Create AutoencoderDataset for pre-training (no corruption, clean data).
        
        Parameters
        ----------
        data : pd.DataFrame
            Pre-training data (may contain NaN values, handled by masked loss).
        device : torch.device, optional
            Device for tensors. Defaults to 'cuda'.
            
        Returns
        -------
        AutoencoderDataset
            Dataset with clean data (y_corrupted = y_clean for pre-training).
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Split into X (features) and y (targets)
        if self.all_columns_are_targets:
            X_df = None
            y_df = data
        else:
            X_df = data.drop(columns=self.target_series)
            y_df = data[self.target_series]
        
        # Convert to tensors
        X = None if X_df is None else torch.from_numpy(X_df.values).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        y = torch.from_numpy(y_df.values).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        
        # For pre-training: y_corrupted = y_clean (no corruption)
        return AutoencoderDataset(
            X=X,
            y_corrupted=y,
            y_clean=y
        )
    
    def create_autoencoder_datasets_list(
        self,
        n_mc_samples: int,
        mu_eps: np.ndarray,
        std_eps: np.ndarray,
        X: Union[np.ndarray, pd.DataFrame],
        y_tmp: Union[np.ndarray, pd.DataFrame],
        y_actual: np.ndarray,
        rng: np.random.RandomState,
        device: Optional[torch.device] = None
    ) -> List['AutoencoderDataset']:
        """Create AutoencoderDataset instances with pre-sampled MC noise.
        
        Parameters
        ----------
        n_mc_samples : int
            Number of Monte Carlo samples.
        mu_eps : np.ndarray
            Noise mean (num_target_series,).
        std_eps : np.ndarray
            Noise std (num_target_series,).
        X : np.ndarray | pd.DataFrame
            Features (T x N_features) - lags, dummies, etc. Not corrupted.
        y_tmp : np.ndarray | pd.DataFrame
            Target data (T x num_target_series) to corrupt.
        y_actual : np.ndarray
            Clean targets (T x num_target_series) for reconstruction.
        rng : np.random.RandomState
            Random number generator.
        device : torch.device, optional
            Device for tensors. Defaults to 'cuda'.
            
        Returns
        -------
        List[AutoencoderDataset]
            One dataset per MC sample.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_tmp_array = y_tmp.values if isinstance(y_tmp, pd.DataFrame) else y_tmp
        
        T = y_tmp_array.shape[0]
        has_features = X_array.size > 0
        
        # Pre-sample all MC noise at once (efficient)
        eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (n_mc_samples, T))
        
        # Convert to tensors once (efficient)
        X_tensor = torch.from_numpy(X_array).to(dtype=DEFAULT_TORCH_DTYPE, device=device) if has_features else None
        y_tmp_tensor = torch.from_numpy(y_tmp_array).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        y_actual_tensor = torch.from_numpy(y_actual).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        eps_draws_tensor = torch.from_numpy(eps_draws).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        
        # Create datasets using the single-dataset method
        datasets = []
        for i in range(n_mc_samples):
            dataset = self.create_autoencoder_dataset(
                X=X_tensor,
                y_tmp=y_tmp_tensor,
                y_actual=y_actual_tensor,
                eps_draw=eps_draws_tensor[i, :, :]
            )
            datasets.append(dataset)
        
        return datasets


class AutoencoderDataset:
    """Container for autoencoder training data with corrupted inputs and clean targets.
    
    Stores pre-loaded tensors for efficient direct slicing (no DataLoader needed).
    All tensors are expected to be on the correct device.
    
    Parameters
    ----------
    X : torch.Tensor, optional
        Features (T, N_features) - lags, dummies, etc. Not corrupted.
    y_corrupted : torch.Tensor
        Corrupted targets (T, num_target_series).
    y_clean : torch.Tensor
        Clean targets (T, num_target_series) for reconstruction.
    """
    
    def __init__(
        self,
        X: Optional[torch.Tensor],
        y_corrupted: torch.Tensor,
        y_clean: torch.Tensor
    ):
        self.X = X
        self.y_corrupted = y_corrupted
        self.y_clean = y_clean
        # Pre-compute full_input once (optimization: avoid torch.cat on every access)
        if self.X is not None:
            self._full_input = torch.cat([self.X, self.y_corrupted], dim=1)
        else:
            self._full_input = self.y_corrupted
    
    @property
    def full_input(self) -> torch.Tensor:
        """Full autoencoder input: clean X features + corrupted y targets."""
        return self._full_input
    
    def __len__(self) -> int:
        """Return number of time steps."""
        return self.y_corrupted.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index idx.
        
        Parameters
        ----------
        idx : int
            Index of time step
            
        Returns
        -------
        full_input : torch.Tensor
            Full input at time step idx (shape: (N_input,))
        y_clean : torch.Tensor
            Clean target at time step idx (shape: (N,))
        """
        return self._full_input[idx], self.y_clean[idx]

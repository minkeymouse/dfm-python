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
    
    Scales target series if scaler provided.
    
    All columns are treated as targets by default. When covariates are specified,
    they are excluded from targets (targets = all_columns - covariates).
    
    Parameters
    ----------
    data : pd.DataFrame | PolarsDataFrame
        Input data. Target series will be scaled if scaler provided.
    time_idx : str
        Time index column name.
    covariates : List[str], optional
        Series to exclude from targets (used for factor extraction but not forecasted).
        If None, all series are targets (default).
    scaler : StandardScaler | RobustScaler | MinMaxScaler, optional
        Scaler instance to scale target series. If None, no scaling.
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, PolarsDataFrame],
        time_idx: str,
        covariates: Optional[List[str]] = None,
        scaler: Optional[Union[StandardScaler, RobustScaler, MinMaxScaler]] = None,
    ):
        if _has_polars and isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        
        data = data.copy()
        data.sort_index(inplace=True)
        
        self.time_idx = time_idx
        self.time_index = pd.Index(data[time_idx]) if time_idx in data.columns else data.index
        self.data_original = data.copy()
        
        # Get variables (all columns excluding time_idx)
        variables = [col for col in data.columns if col != time_idx] if time_idx in data.columns else list(data.columns)
        
        # Filter covariates and compute targets
        covariates = [c for c in (covariates or []) if c in variables]
        self.covariates = covariates
        self.target_series = [col for col in variables if col not in covariates]
        
        # Split into covariates (X) and targets (y)
        data_for_split = data.drop(columns=[time_idx]) if time_idx in data.columns else data
        y = data_for_split[self.target_series]
        X = data_for_split.drop(columns=self.target_series) if self.target_series else pd.DataFrame()
        
        # Scale covariates separately (temporary scaler, not stored)
        if scaler is not None and not X.empty:
            X = pd.DataFrame(type(scaler)().fit_transform(X.values), index=X.index, columns=X.columns)
        
        # Scale targets and store scaler for prediction
        if scaler is not None:
            y = pd.DataFrame(scaler.fit_transform(y.values), index=y.index, columns=y.columns)
        
        self.scaler = scaler
        
        # Store processed data
        self.data = pd.concat([X, y], axis=1) if not X.empty else y
        self.X = X.values if not X.empty else np.empty((len(y), 0))
        self.y = y.values
        self.missing_y = y.isna().values
        self.observed_y = ~self.missing_y

    @property
    def target_shape(self) -> Tuple[int, int]:
        """Target shape (T, num_target_series)."""
        return self.y.shape

    @property
    def data_shape(self) -> Tuple[int, int]:
        """Data shape (T, num_features + num_target_series)."""
        return self.X.shape[0], self.X.shape[1] + self.y.shape[1]
    
    @property
    def all_columns_are_targets(self) -> bool:
        """Whether all columns are target series (no covariates)."""
        return len(self.covariates) == 0
    
    def split_features_and_targets(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Split DataFrame into features (X) and targets (y)."""
        if self.all_columns_are_targets:
            return None, data
        X = data.drop(columns=self.target_series)
        y = data[self.target_series]
        return X, y
    
    @property
    def target_indices(self) -> np.ndarray:
        """Target series column indices in processed data (self.data, excluding time_idx)."""
        # target_series are column names in the processed data (self.data)
        # self.data contains X + y (or just y), but not time_idx
        return np.array([self.data.columns.get_loc(col) for col in self.target_series])
    
    @classmethod
    def from_dataset(cls, new_data: Union[pd.DataFrame, PolarsDataFrame], dataset: 'DDFMDataset') -> 'DDFMDataset':
        """Create new dataset with new data, preserving configuration."""
        return cls(
            data=new_data,
            time_idx=dataset.time_idx,
            covariates=dataset.covariates,
            scaler=dataset.scaler
        )
    
    def create_autoencoder_dataset(
        self,
        X: Optional[torch.Tensor],
        y_tmp: torch.Tensor,
        y_actual: torch.Tensor,
        eps_draw: torch.Tensor
    ) -> 'AutoencoderDataset':
        """Create a single AutoencoderDataset with corrupted targets."""
        return AutoencoderDataset(X=X, y_corrupted=y_tmp - eps_draw, y_clean=y_actual)
    
    def create_pretrain_dataset(
        self,
        data: pd.DataFrame,
        device: Optional[torch.device] = None
    ) -> 'AutoencoderDataset':
        """Create AutoencoderDataset for pre-training (no corruption, clean data)."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_df, y_df = self.split_features_and_targets(data)
        X = None if X_df is None else torch.from_numpy(X_df.values).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        y = torch.from_numpy(y_df.values).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        
        return AutoencoderDataset(X=X, y_corrupted=y, y_clean=y)
    
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
        """Create AutoencoderDataset instances with pre-sampled MC noise."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_tmp_array = y_tmp.values if isinstance(y_tmp, pd.DataFrame) else y_tmp
        T = y_tmp_array.shape[0]
        
        # Pre-sample all MC noise at once
        eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (n_mc_samples, T))
        
        # Convert to tensors once
        X_tensor = torch.from_numpy(X_array).to(dtype=DEFAULT_TORCH_DTYPE, device=device) if X_array.size > 0 else None
        y_tmp_tensor = torch.from_numpy(y_tmp_array).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        y_actual_tensor = torch.from_numpy(y_actual).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        eps_draws_tensor = torch.from_numpy(eps_draws).to(dtype=DEFAULT_TORCH_DTYPE, device=device)
        
        # Create datasets
        return [
            self.create_autoencoder_dataset(X_tensor, y_tmp_tensor, y_actual_tensor, eps_draws_tensor[i])
            for i in range(n_mc_samples)
        ]


class AutoencoderDataset:
    """Container for autoencoder training data with corrupted inputs and clean targets.
    
    Stores pre-loaded tensors for efficient direct slicing. All tensors are expected 
    to be on the correct device. Use direct tensor slicing (e.g., `dataset.full_input[i:j]`)
    rather than indexing.
    
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

"""PyTorch Dataset for Identifiable Variational Dynamic Factor Model (iVDFM).

This module provides a dataset class for iVDFM that handles sliding window
sequences and context (auxiliary variable) generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer

from ..config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_IVDFM_AUX_DIM,
)
from ..logger import get_logger

_logger = get_logger(__name__)


class iVDFMDataset(Dataset):
    """Dataset for iVDFM training with sliding window sequences.
    
    Provides sequences of:
    - targets (the series reconstructed by the decoder and used in reconstruction loss)
    - context (the auxiliary variable u_t in the iVAE/iVDFM identifiability framework)
    
    Parameters
    ----------
    data : Union[np.ndarray, torch.Tensor, pd.DataFrame]
        Time series data. Columns may include:
        - targets (default)
        - covariates (excluded from targets)
        - context columns (optional; exogenous auxiliary variables u_t provided as columns)
        The time index column (if provided) is excluded from targets.
    window : Optional[int]
        Length of sliding windows. If None, uses full T (full series as one sequence).
        Default: None (uses full T)
    stride : int, default 1
        Step size for sliding windows. stride=1 means overlapping windows, stride=window means non-overlapping.
    time_idx : Optional[Union[str, int]]
        Time index column name (DataFrame) or column index (array). If provided, it is
        excluded from targets. If not provided, positional index is used to generate time context.
    covariates : Optional[Union[List[str], List[int]]]
        Columns excluded from targets (DDFM-style covariates). Not used as iVDFM context.
    context : Optional[Union[List[str], List[int]]]
        Optional columns that represent *exogenous auxiliary variables* u_t. If provided, these columns are
        concatenated with time features to form the iVDFM context. Context dimension is automatically
        deduced from the number of context columns plus time_context dimension.
    time_context : int, default 1
        Dimension of time-based context features. Always included.
        - time_context=1: Only normalized time step [0, 1]
        - time_context>1: Normalized time step + periodic sine features (sin(2π * (i+1) / T * t) for i=1..time_context-1)
    device : Optional[torch.device]
        Device to move tensors to
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        window: Optional[int] = None,
        stride: int = 1,
        time_idx: Optional[Union[str, int]] = None,
        covariates: Optional[Union[List[str], List[int]]] = None,
        context: Optional[Union[List[str], List[int]]] = None,
        time_context: int = 1,
        scaler: Optional[Union[str, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize iVDFM dataset."""
        # Handle DataFrame input
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            data_df = data.copy()
            # Basic missing / non-finite handling (keep simple, like DDFM practice)
            # - convert +/-inf to NaN
            # - interpolate through time
            # - forward/back fill
            # - final fill with zeros if still missing
            data_df = data_df.replace([np.inf, -np.inf], np.nan)
            if data_df.isna().any().any():
                data_df = data_df.interpolate(limit_direction="both")
                data_df = data_df.ffill().bfill()
                if data_df.isna().any().any():
                    _logger.warning("Data still contains NaN after interpolation; filling remaining NaNs with 0.0")
                    data_df = data_df.fillna(0.0)
            data_array = data_df.values
            column_names = list(data_df.columns)
        else:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data_array = np.asarray(data)
            # Basic missing / non-finite handling for arrays
            if not np.isfinite(data_array).all():
                data_array = data_array.copy()
                data_array[~np.isfinite(data_array)] = np.nan
                # linear interpolation along time for each column
                for j in range(data_array.shape[1]):
                    col = data_array[:, j]
                    nans = np.isnan(col)
                    if nans.any():
                        idx = np.arange(col.shape[0])
                        valid = ~nans
                        if valid.any():
                            col[nans] = np.interp(idx[nans], idx[valid], col[valid])
                        else:
                            col[:] = 0.0
                    data_array[:, j] = col
                # final safeguard
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)
            column_names = None
        
        T, N_total = data_array.shape
        
        # Default behavior: if window is None, use full T (full series as one sequence)
        if window is None:
            window = T
        else:
            # Clamp to valid range
            window = max(2, min(window, T))
        
        self.window = window
        self.stride = max(1, stride)  # Ensure stride >= 1
        self.total_time_steps = T
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Determine available columns (exclude time_idx from the pool used for targets/covariates/context)
        if is_dataframe:
            all_cols = list(column_names)
            if isinstance(time_idx, str) and time_idx in all_cols:
                all_cols_no_time = [c for c in all_cols if c != time_idx]
            else:
                all_cols_no_time = all_cols
        else:
            all_cols = list(range(N_total))
            all_cols_no_time = all_cols if time_idx is None else [c for c in all_cols if c != time_idx]
        
        # covariates: excluded from targets (NOT iVDFM auxiliary context)
        covariates_list = [c for c in (covariates or []) if c in all_cols_no_time]
        self.covariates = covariates_list
        
        # context columns: auxiliary variables u_t provided as columns (in addition to time context)
        context_cols = [c for c in (context or []) if c in all_cols_no_time and c not in covariates_list]
        self.context_columns = context_cols
        
        # Targets are everything else: all - covariates - context - time_idx
        self.target_series = [c for c in all_cols_no_time if c not in covariates_list and c not in context_cols]
        
        # Split into targets (y), covariates (X_cov), and context columns (X_ctx)
        if is_dataframe:
            data_for_split = data_df.drop(columns=[time_idx]) if isinstance(time_idx, str) and time_idx in data_df.columns else data_df
            y = data_for_split[self.target_series] if self.target_series else pd.DataFrame()
            X_cov = data_for_split[self.covariates] if self.covariates else pd.DataFrame()
            X_ctx = data_for_split[self.context_columns] if self.context_columns else pd.DataFrame()

            # Apply scaler to targets only (not covariates/context)
            if scaler is not None:
                scaler_instance = self._get_scaler_instance(scaler)
                if not y.empty:
                    y = pd.DataFrame(
                        scaler_instance.fit_transform(y.values),
                        index=y.index,
                        columns=y.columns
                    )
                self.scaler = scaler_instance
            else:
                self.scaler = None

            self.data = y.values
            self.covariate_data = X_cov.values if not X_cov.empty else None
            self.aux_context_data = X_ctx.values if not X_ctx.empty else None
        else:
            # Array: use indices
            target_cols = self.target_series
            covariate_cols = self.covariates
            context_cols_arr = self.context_columns

            y_data = data_array[:, target_cols] if len(target_cols) > 0 else np.empty((T, 0))
            if scaler is not None:
                scaler_instance = self._get_scaler_instance(scaler)
                if y_data.size > 0:
                    y_data = scaler_instance.fit_transform(y_data)
                self.scaler = scaler_instance
            else:
                self.scaler = None
            self.data = y_data
            self.covariate_data = data_array[:, covariate_cols] if len(covariate_cols) > 0 else None
            self.aux_context_data = data_array[:, context_cols_arr] if len(context_cols_arr) > 0 else None
        
        # iVDFM context u_t = [time features] + [exogenous context columns]
        # Covariates are *not* part of u_t.
        # Time context is always included (at least normalized time step)
        time_features = self._generate_time_context(time_idx=time_idx, data_df=data_df if is_dataframe else None, T=T, time_context=time_context)
        blocks: List[np.ndarray] = [time_features]
        if self.aux_context_data is not None and self.aux_context_data.size > 0:
            blocks.append(self.aux_context_data.astype(np.float32, copy=False))

        self._context = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        self._context_dim = int(self._context.shape[1])
        
        # Number of sequences (sliding windows with stride)
        self.num_sequences = (T - window) // self.stride + 1

        # Basic bookkeeping for user-facing clarity
        self.time_idx = time_idx
        self.all_columns = all_cols
    
    def _get_scaler_instance(
        self,
        scaler: Union[str, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]
    ) -> Union[StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]:
        """Convert scaler string to instance, or return instance if already provided.
        
        Parameters
        ----------
        scaler : Union[str, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]
            Scaler as string ('standard', 'robust', 'minmax', 'maxabs', 'quantile') or instance
            
        Returns
        -------
        Scaler instance
        """
        if isinstance(scaler, str):
            scaler_map = {
                'standard': StandardScaler,
                'robust': RobustScaler,
                'minmax': MinMaxScaler,
                'maxabs': MaxAbsScaler,
                'quantile': QuantileTransformer,
            }
            scaler_class = scaler_map.get(scaler.lower())
            if scaler_class is None:
                raise ValueError(
                    f"Unknown scaler type '{scaler}'. "
                    f"Supported: {list(scaler_map.keys())}"
                )
            return scaler_class()
        else:
            # Already an instance
            return scaler
    
    def _generate_time_context(
        self,
        *,
        time_idx: Optional[Union[str, int]],
        data_df: Optional[pd.DataFrame],
        T: int,
        time_context: int,
    ) -> np.ndarray:
        """Generate time-based context features from time index.
        
        Always generates at least normalized time step. If time_context > 1,
        adds periodic sine features.
        """
        if isinstance(time_idx, str) and data_df is not None and time_idx in data_df.columns:
            tcol = data_df[time_idx]
            if np.issubdtype(tcol.dtype, np.datetime64):
                t = pd.to_datetime(tcol).astype("int64").to_numpy(dtype=np.float32)
            else:
                # numeric-like
                t = pd.to_numeric(tcol, errors="coerce").to_numpy(dtype=np.float32)
                if np.isnan(t).any():
                    # fallback to positional index if cannot parse
                    t = np.arange(T, dtype=np.float32)
        else:
            t = np.arange(T, dtype=np.float32)

        # Normalize to [0, 1]
        t = t - float(np.nanmin(t))
        denom = float(np.nanmax(t)) if float(np.nanmax(t)) != 0.0 else 1.0
        t = t / denom
        return self._create_time_features_from_base(t, time_context)
    
    def _create_time_features_from_base(self, t01: np.ndarray, time_context: int) -> np.ndarray:
        """Create time features from a normalized base time index (shape (T,)).
        
        Parameters
        ----------
        t01 : np.ndarray
            Normalized time index in [0, 1], shape (T,)
        time_context : int
            Dimension of time features:
            - time_context=1: Only normalized time step
            - time_context>1: Normalized time step + periodic sine features
        """
        t01 = np.asarray(t01, dtype=np.float32).reshape(-1)
        if time_context <= 1:
            return t01.reshape(-1, 1)
        features = [t01.reshape(-1, 1)]  # Always include normalized time step
        T = t01.shape[0]
        for i in range(1, time_context):
            freq = 2 * np.pi * (i + 1) / max(T, 2)
            periodic = np.sin(freq * np.arange(T, dtype=np.float32))
            features.append(periodic.reshape(-1, 1))
        return np.hstack(features)
    
    
    def __len__(self) -> int:
        """Return number of sequences.
        
        Returns
        -------
        int
            Number of sequences (sliding windows)
        """
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence of observations and context."""
        start_idx = idx * self.stride
        end_idx = start_idx + self.window
        y_seq = self.target[start_idx:end_idx, :]
        u_seq = self._context[start_idx:end_idx, :]
        
        y_tensor = torch.from_numpy(y_seq).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        u_tensor = torch.from_numpy(u_seq).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        return y_tensor, u_tensor
    
    @property
    def target(self) -> np.ndarray:
        """Target matrix (T, target_length). Used for reconstruction loss."""
        return self.data
    
    @property
    def covariate(self) -> np.ndarray:
        """Covariate matrix (T, covariate_length). Not used in iVDFM loss."""
        if self.covariate_data is None:
            return np.empty((self.total_time_steps, 0), dtype=np.float32)
        return self.covariate_data
    
    @property
    def context(self) -> np.ndarray:
        """Context (auxiliary) matrix u_t (T, context_length)."""
        return self._context

    @property
    def target_length(self) -> int:
        """Number of target series (decoder output dimension)."""
        return int(self.target.shape[1]) if self.target.size > 0 else 0

    @property
    def covariate_length(self) -> int:
        """Number of covariate series (excluded from targets)."""
        return int(self.covariate.shape[1]) if self.covariate.size > 0 else 0

    @property
    def context_length(self) -> int:
        """Context dimension (auxiliary variable u_t dimension)."""
        return int(self._context_dim)

    # Backwards-compat (internal): prefer *_length names above
    @property
    def data_dim(self) -> int:
        return self.target_length

    @property
    def context_dim(self) -> int:
        return self.context_length
    
    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create a DataLoader for this dataset.
        
        Parameters
        ----------
        batch_size : int
            Batch size for the DataLoader
        shuffle : bool, default True
            Whether to shuffle the data
        num_workers : int, default 0
            Number of worker processes for data loading
        **kwargs
            Additional arguments passed to DataLoader
        
        Returns
        -------
        DataLoader
            PyTorch DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )

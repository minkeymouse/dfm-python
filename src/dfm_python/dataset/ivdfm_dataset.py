"""Sliding-window dataset of the iVDFM engine: target windows and the context u_t (time features + context columns)."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Union, Tuple, List
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer

from ..config.constants import DEFAULT_TORCH_DTYPE
from ..logger import get_logger

_logger = get_logger(__name__)

_SCALERS = {
    'standard': StandardScaler,
    'robust': RobustScaler,
    'minmax': MinMaxScaler,
    'maxabs': MaxAbsScaler,
    'quantile': QuantileTransformer,
}


def _time_features(
    time_idx: Optional[Union[str, int]], data_df: Optional[pd.DataFrame], T: int, time_context: int
) -> np.ndarray:
    """Time context (T, time_context), float32: the time step normalized to [0, 1], then sine features.

    The time step comes from the ``time_idx`` column of ``data_df`` (datetime
    or numeric), else from the position. Feature i >= 1 is
    sin(2π (i+1) / max(T, 2) * t) over the positions t.
    """
    if isinstance(time_idx, str) and data_df is not None and time_idx in data_df.columns:
        tcol = data_df[time_idx]
        if np.issubdtype(tcol.dtype, np.datetime64):
            # int64 nanoseconds whatever the column's resolution (pandas 3 parses
            # dates at a coarser unit than pandas 2's ns), so the features match.
            t = pd.to_datetime(tcol).dt.as_unit("ns").astype("int64").to_numpy(dtype=np.float32)
        else:
            t = pd.to_numeric(tcol, errors="coerce").to_numpy(dtype=np.float32)
            if np.isnan(t).any():
                t = np.arange(T, dtype=np.float32)
    else:
        t = np.arange(T, dtype=np.float32)

    t = t - float(np.nanmin(t))
    denom = float(np.nanmax(t)) if float(np.nanmax(t)) != 0.0 else 1.0
    t = t / denom
    t01 = np.asarray(t, dtype=np.float32).reshape(-1)
    if time_context <= 1:
        return t01.reshape(-1, 1)
    features = [t01.reshape(-1, 1)]
    T = t01.shape[0]
    for i in range(1, time_context):
        freq = 2 * np.pi * (i + 1) / max(T, 2)
        periodic = np.sin(freq * np.arange(T, dtype=np.float32))
        features.append(periodic.reshape(-1, 1))
    return np.hstack(features)


class iVDFMDataset(Dataset):
    """Sliding windows of targets y and context u_t = [time features | context columns] for iVDFM training.

    Missing and non-finite values are interpolated linearly through time,
    then filled forward/backward, then with zeros. Only the targets are scaled.

    Parameters
    ----------
    data : ndarray, Tensor or DataFrame, shape (T, N_total)
        Targets, plus optional ``covariates`` and ``context`` columns and a
        ``time_idx`` column, none of which are targets.
    window : int, optional
        Window length; None uses the whole series as one window. Clipped to [2, T].
    stride : int, optional
        Step between window starts; None means ``window // 2`` (at least 1).
    time_idx : str or int, optional
        Time column (DataFrame name or array index) the time features are
        computed from; without it the position is used.
    covariates : list of str or int, optional
        Columns excluded from the targets; not part of u_t.
    context : list of str or int, optional
        Exogenous columns appended to the time features in u_t.
    time_context : int, default 1
        Number of time features: the normalized time step, plus
        ``time_context - 1`` sine features.
    scaler : str or sklearn transformer, optional
        'standard', 'robust', 'minmax', 'maxabs', 'quantile', or an instance
        whose ``fit_transform`` is applied to the targets.
    device : torch.device, optional
        Device of the tensors; None means CUDA if available. On CUDA every
        window is pre-loaded as one tensor.
    """

    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        window: Optional[int] = None,
        stride: Optional[int] = None,
        time_idx: Optional[Union[str, int]] = None,
        covariates: Optional[Union[List[str], List[int]]] = None,
        context: Optional[Union[List[str], List[int]]] = None,
        time_context: int = 1,
        scaler: Optional[Union[str, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]] = None,
        device: Optional[torch.device] = None,
    ):
        is_dataframe = isinstance(data, pd.DataFrame)
        if is_dataframe:
            data_df = data.copy()
            data_df = data_df.replace([np.inf, -np.inf], np.nan)
            if data_df.isna().any().any():
                data_df = data_df.interpolate(limit_direction="both")
                data_df = data_df.ffill().bfill()
                if data_df.isna().any().any():
                    _logger.warning("Data still contains NaN after interpolation; filling remaining NaNs with 0.0")
                    data_df = data_df.fillna(0.0)
            data_array = data_df.values
        else:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data_array = np.asarray(data)
            if not np.isfinite(data_array).all():
                data_array = data_array.copy()
                data_array[~np.isfinite(data_array)] = np.nan
                for j in range(data_array.shape[1]):
                    col = data_array[:, j]
                    nans = np.isnan(col)
                    if nans.any():
                        idx = np.arange(col.shape[0])
                        valid = ~nans
                        if valid.any():
                            col[nans] = np.interp(idx[nans], idx[valid], col[valid])
                        else:
                            _logger.warning(f"Column {j} is all NaN; filling it with 0.0")
                            col[:] = 0.0

        T, N_total = data_array.shape

        if window is None:
            window = T
        else:
            window = max(2, min(window, T))
        self.window = window
        if stride is None:
            stride = max(1, window // 2)
        self.stride = max(1, stride)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # The column pool for targets/covariates/context excludes time_idx.
        if is_dataframe:
            all_cols = list(data_df.columns)
            if isinstance(time_idx, str) and time_idx in all_cols:
                all_cols_no_time = [c for c in all_cols if c != time_idx]
            else:
                all_cols_no_time = all_cols
        else:
            all_cols = list(range(N_total))
            all_cols_no_time = all_cols if time_idx is None else [c for c in all_cols if c != time_idx]

        self.covariates = [c for c in (covariates or []) if c in all_cols_no_time]
        self.context_columns = [c for c in (context or []) if c in all_cols_no_time and c not in self.covariates]
        self.target_series = [c for c in all_cols_no_time if c not in self.covariates and c not in self.context_columns]

        if is_dataframe:
            data_for_split = data_df.drop(columns=[time_idx]) if isinstance(time_idx, str) and time_idx in data_df.columns else data_df
            y_data = (data_for_split[self.target_series] if self.target_series else pd.DataFrame()).values
            X_ctx = data_for_split[self.context_columns] if self.context_columns else pd.DataFrame()
            self.aux_context_data = X_ctx.values if not X_ctx.empty else None
        else:
            y_data = data_array[:, self.target_series] if len(self.target_series) > 0 else np.empty((T, 0))
            self.aux_context_data = data_array[:, self.context_columns] if len(self.context_columns) > 0 else None

        if scaler is not None:
            self.scaler = self._get_scaler_instance(scaler)
            if y_data.size > 0:
                y_data = self.scaler.fit_transform(y_data)
        else:
            self.scaler = None
        self.data = y_data

        # u_t = [time features | context columns]; covariates are not part of it.
        blocks: List[np.ndarray] = [_time_features(time_idx, data_df if is_dataframe else None, T, time_context)]
        if self.aux_context_data is not None and self.aux_context_data.size > 0:
            blocks.append(self.aux_context_data.astype(np.float32, copy=False))
        self._context = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        self._context_dim = int(self._context.shape[1])

        self.num_sequences = (T - window) // self.stride + 1

        # On CUDA every window is pre-loaded once (no per-sample numpy -> torch -> device in __getitem__).
        self._y_gpu = None
        self._u_gpu = None
        if self.device.type == "cuda" and self.num_sequences > 0:
            N_y = int(self.data.shape[1])
            N_u = int(self._context.shape[1])
            y_all = np.zeros((self.num_sequences, self.window, N_y), dtype=np.float32)
            u_all = np.zeros((self.num_sequences, self.window, N_u), dtype=np.float32)
            for i in range(self.num_sequences):
                start = i * self.stride
                end = start + self.window
                y_all[i] = self.data[start:end, :]
                u_all[i] = self._context[start:end, :]
            self._y_gpu = torch.from_numpy(y_all).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            self._u_gpu = torch.from_numpy(u_all).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)

        self.time_idx = time_idx

    @staticmethod
    def _get_scaler_instance(scaler):
        """The scaler instance for a name ('standard', 'robust', 'minmax', 'maxabs', 'quantile') or an instance."""
        if not isinstance(scaler, str):
            return scaler
        scaler_class = _SCALERS.get(scaler.lower())
        if scaler_class is None:
            raise ValueError(f"Unknown scaler type '{scaler}'. Supported: {list(_SCALERS.keys())}")
        return scaler_class()

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Window ``idx`` as ``(y, u)`` tensors of shape (window, N) and (window, context_dim)."""
        if self._y_gpu is not None:
            return self._y_gpu[idx], self._u_gpu[idx]
        start_idx = idx * self.stride
        end_idx = start_idx + self.window
        y_seq = self.target[start_idx:end_idx, :]
        u_seq = self._context[start_idx:end_idx, :]
        y_tensor = torch.from_numpy(y_seq).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        u_tensor = torch.from_numpy(u_seq).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        return y_tensor, u_tensor

    @property
    def target(self) -> np.ndarray:
        """Scaled targets (T, target_length)."""
        return self.data

    @property
    def context(self) -> np.ndarray:
        """Context u_t (T, context_length)."""
        return self._context

    @property
    def target_length(self) -> int:
        return int(self.target.shape[1]) if self.target.size > 0 else 0

    @property
    def context_length(self) -> int:
        return int(self._context_dim)

    @property
    def data_dim(self) -> int:
        return self.target_length

    @property
    def context_dim(self) -> int:
        return self.context_length

    def get_all_tensors_on_device(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(y_all, u_all)`` of every window when pre-loaded (CUDA), else None."""
        if self._y_gpu is not None and self._u_gpu is not None:
            return self._y_gpu, self._u_gpu
        return None

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )


class RecordingDataset(iVDFMDataset):
    """The iVDFM estimator's dataset: an ``iVDFMDataset`` recording the window index of every ``__getitem__`` call.

    On CPU the engine trains and extracts its final factors through a shuffled
    ``DataLoader``, so its factor arrays are in shuffled window order. The last
    ``len(dataset)`` recorded indices are the order of that final extraction.
    Recording reads no random numbers, so the engine's numbers are unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: list[int] = []

    def __getitem__(self, idx: int):
        self.accessed.append(int(idx))
        return super().__getitem__(idx)

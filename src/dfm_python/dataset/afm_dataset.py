"""Dataset for the Attention Factor Model.

Unlike the dynamic-factor datasets (a T x N panel of time series), AFM consumes a
cross-sectional panel: at each date it needs the asset returns ``R_t`` (N,) and,
for the attention factors, the firm characteristics ``X_{t-1}`` (N, M). This
dataset holds the returns panel (T, N) and an optional characteristics tensor
(T, N, M), aligned so that row ``t`` pairs returns at ``t`` with characteristics
known at ``t`` (the model uses the lagged slice internally).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AFMDataset(Dataset):
    """Cross-sectional returns (+ characteristics) panel for AFM.

    Parameters
    ----------
    returns : pandas.DataFrame or numpy.ndarray
        Asset returns, shape (T, N).
    characteristics : numpy.ndarray, optional
        Firm characteristics, shape (T, N, M). Required for attention factors,
        omitted for PCA factors.
    window : int, optional
        Sliding-window length for batched training. If None, the full panel is a
        single sequence.
    stride : int
        Step between windows.
    scaler : str, optional
        Kept for API parity with the other datasets; characteristics in this
        literature are pre-normalized to rank quantiles, so no scaling is applied
        by default.
    """

    def __init__(self, returns: Union[pd.DataFrame, np.ndarray],
                 characteristics: Optional[np.ndarray] = None,
                 window: Optional[int] = None, stride: int = 1,
                 scaler: Optional[str] = None) -> None:
        if isinstance(returns, pd.DataFrame):
            self.columns: List[str] = list(returns.columns)
            returns_arr = returns.to_numpy(dtype=np.float64)
        else:
            returns_arr = np.asarray(returns, dtype=np.float64)
            self.columns = [f"asset_{i}" for i in range(returns_arr.shape[1])]
        if returns_arr.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, N), got {returns_arr.shape}")
        self.returns = returns_arr                                  # (T, N)
        self.T, self.N = returns_arr.shape

        if characteristics is not None:
            characteristics = np.asarray(characteristics, dtype=np.float64)
            if characteristics.shape[:2] != (self.T, self.N):
                raise ValueError(
                    "characteristics must be (T, N, M) aligned with returns, "
                    f"got {characteristics.shape} vs returns {returns_arr.shape}")
        self.characteristics = characteristics                     # (T, N, M) or None
        self.n_char: Optional[int] = None if characteristics is None else characteristics.shape[2]
        self.n_factors: Optional[int] = None
        self.window = window
        self.stride = max(1, int(stride))
        self.scaler = scaler

        if window is not None:
            starts = range(0, self.T - window + 1, self.stride)
            self._starts = list(starts)
        else:
            self._starts = [0]

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        length = self.window if self.window is not None else self.T
        r = torch.tensor(self.returns[start:start + length], dtype=torch.float32)
        if self.characteristics is None:
            return r, None
        c = torch.tensor(self.characteristics[start:start + length], dtype=torch.float32)
        return r, c

    def get_tensors(self, device: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full panel as tensors ``(T, N)`` and ``(T, N, M)`` (or None)."""
        r = torch.tensor(self.returns, dtype=torch.float32)
        c = (None if self.characteristics is None
             else torch.tensor(self.characteristics, dtype=torch.float32))
        if device is not None:
            r = r.to(device)
            c = None if c is None else c.to(device)
        return r, c

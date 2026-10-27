"""Tiny estimators on the new base class, used to exercise ``check_estimator``.

``_DummyForecaster`` (numpy mean forecaster), ``_DummyTorchForecaster`` (a
one-layer torch network) and ``_DummyAllocator`` (sign-of-trailing-mean
weights) follow the whole contract, including save and load through the
``_save_state`` / ``_load_state`` hooks. The broken variants in
``test_harness.py`` subclass them and break one rule each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from dfm_python.base import BaseFactorModel
from dfm_python.config import EstimatorConfig
from dfm_python.utils.errors import DataValidationError
from dfm_python.utils.seed import apply_seed


def _future_index(index: pd.Index, h: int) -> pd.Index:
    if isinstance(index, pd.RangeIndex):
        start = index[-1] + index.step
        return pd.RangeIndex(start, start + h * index.step, index.step, name=index.name)
    freq = index.freq or pd.infer_freq(index)
    return pd.date_range(index[-1], periods=h + 1, freq=freq, name=index.name)[1:]


def _frame(values: np.ndarray, model: BaseFactorModel, origin_index: pd.Index, columns, kind: str) -> pd.DataFrame:
    index = _future_index(origin_index, values.shape[0])
    index.name = model.index_.name if model.index_.name is not None else "time"
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64), index=index, columns=list(columns))
    frame.attrs = {"model": type(model).__name__, "kind": kind, "origin": origin_index[-1]}
    return frame


# ---------------------------------------------------------------------------
# Mean forecaster (numpy)
# ---------------------------------------------------------------------------


@dataclass
class _DummyForecasterConfig(EstimatorConfig):
    shrink: float = 0.0
    window: Optional[int] = None
    lags: tuple = (1,)
    verbose: int = 0


@dataclass
class _DummyResult:
    mean: np.ndarray
    n_obs: int
    scale: float


class _DummyForecaster(BaseFactorModel):
    """Forecast every step as the (shrunk) column means of the last ``window`` rows."""

    _tags = {
        "capability:missing_values": True,
        "capability:update": True,
        "capability:factor_forecast": True,
        "update_kind": "reattach",
        "predict_data": "full_panel",
    }
    _config_class = _DummyForecasterConfig

    def __init__(self, *, shrink=0.0, window=None, lags=(1,), verbose=0):
        self.shrink = shrink
        self.window = window
        self.lags = lags
        self.verbose = verbose

    @classmethod
    def get_test_params(cls):
        return [{}, {"shrink": 0.25, "window": 5, "lags": (1, 2)}]

    def _means(self, y: pd.DataFrame) -> np.ndarray:
        values = y.to_numpy()
        if self.window is not None:
            values = values[-self.window:]
        return (1.0 - self.shrink) * np.nanmean(values, axis=0)

    def _fit(self, data, **fit_kwargs):
        mean = self._means(data.y)
        self.mean_ = mean
        self.conditioning_index_ = data.index
        self.result_ = _DummyResult(mean=mean.copy(), n_obs=data.n_periods, scale=float(np.abs(mean).sum()))

    def predict(self, h, data=None, **predict_kwargs):
        self._check_is_fitted("predict")
        h = self._validate_horizon(h)
        origin = self.conditioning_index_ if data is None else self._validate_data(data).index
        return _frame(np.tile(self.mean_, (h, 1)), self, origin, self.feature_names_in_, "forecast")

    def predict_factors(self, h, data=None):
        self._check_is_fitted("predict_factors")
        h = self._validate_horizon(h)
        origin = self.conditioning_index_ if data is None else self._validate_data(data).index
        return _frame(np.full((h, 1), self.mean_.mean()), self, origin, ["factor_0"], "factors")

    def update(self, data, *, refit=False):
        self._check_is_fitted("update")
        panel = self._validate_data(data)
        if refit:
            self._fit(panel)
        self.conditioning_index_ = panel.index
        return self

    def _save_state(self, path):
        self._write_state(path, {"mean": self.mean_})
        self._write_result(path)

    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        arrays, _ = cls._read_state(path)
        est.mean_ = arrays["mean"]
        est.conditioning_index_ = est.index_
        est.result_ = cls._read_result(path, _DummyResult)


# ---------------------------------------------------------------------------
# One-layer torch forecaster
# ---------------------------------------------------------------------------


@dataclass
class _DummyTorchConfig(EstimatorConfig):
    max_epochs: int = 3
    learning_rate: float = 0.05
    random_state: Optional[int] = 0
    device: str = "auto"
    verbose: int = 0


@dataclass
class _DummyTorchResult:
    loss: float
    weight: np.ndarray


class _DummyTorchForecaster(BaseFactorModel):
    """Fit y_t = W y_{t-1} + b by a few full-batch SGD steps; iterate the map to forecast."""

    _tags = {"backend": "torch", "capability:device": True, "predict_data": "window"}
    _config_class = _DummyTorchConfig

    def __init__(self, *, max_epochs=3, learning_rate=0.05, random_state=0, device="auto", verbose=0):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _fit(self, data, **fit_kwargs):
        device = self._resolve_device()
        if self.random_state is not None:
            apply_seed(self.random_state)
        y = torch.as_tensor(data.y.to_numpy(), dtype=torch.float32, device=device)
        network = torch.nn.Linear(y.shape[1], y.shape[1]).to(device)
        optimizer = torch.optim.SGD(network.parameters(), lr=self.learning_rate)
        loss = torch.zeros(())
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            loss = ((network(y[:-1]) - y[1:]) ** 2).mean()
            loss.backward()
            optimizer.step()
        network.eval()
        self.network_ = network
        self.device_ = device
        self.last_ = data.y.to_numpy()[-1].copy()
        self.result_ = _DummyTorchResult(loss=float(loss.detach()), weight=network.weight.detach().cpu().numpy().copy())

    def predict(self, h, data=None, **predict_kwargs):
        self._check_is_fitted("predict")
        h = self._validate_horizon(h)
        if data is None:
            last, origin = self.last_, self.index_
        else:
            panel = self._validate_data(data)
            last, origin = panel.y.to_numpy()[-1], panel.index
        param = next(self.network_.parameters())
        x = torch.as_tensor(last, dtype=param.dtype, device=param.device)
        steps = []
        with torch.no_grad():
            for _ in range(h):
                x = self.network_(x)
                steps.append(x.cpu().numpy().astype(np.float64))
        return _frame(np.stack(steps), self, origin, self.feature_names_in_, "forecast")

    def _save_state(self, path):
        self._write_state(path, {"last": self.last_})
        self._write_result(path)
        self._write_weights(path)

    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        device = est._load_device(device)
        arrays, _ = cls._read_state(path)
        n = est.n_features_in_
        network = torch.nn.utils.skip_init(torch.nn.Linear, n, n, device=device)  # no RNG draws
        network.load_state_dict(cls._read_weights(path, device))
        network.eval()
        est.network_ = network
        est.device_ = device
        est.last_ = arrays["last"]
        est.result_ = cls._read_result(path, _DummyTorchResult)


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


@dataclass
class _DummyAllocatorConfig(EstimatorConfig):
    hist_len: int = 4
    verbose: int = 0


@dataclass
class _DummyAllocResult:
    weights: np.ndarray
    sharpe: float


class _DummyAllocator(BaseFactorModel):
    """Weight each asset by the sign of its mean return over the last ``hist_len`` steps (scaled to sum |w| = 1)."""

    _tags = {
        "task": "allocation",
        "capability:series_exog": True,
        "capability:update": True,
        "update_kind": "reattach",
    }
    _config_class = _DummyAllocatorConfig

    def __init__(self, *, hist_len=4, verbose=0):
        self.hist_len = hist_len
        self.verbose = verbose

    def _weights(self, data) -> pd.DataFrame:
        returns = data.y.to_numpy()
        rows = []
        for t in range(self.hist_len, len(returns)):
            signal = np.sign(returns[t - self.hist_len:t].mean(axis=0))
            total = np.abs(signal).sum()
            rows.append(signal / total if total > 0 else signal)
        weights = np.asarray(rows, dtype=np.float64).reshape(-1, returns.shape[1])
        frame = pd.DataFrame(weights, index=data.index[self.hist_len:], columns=list(self.feature_names_in_))
        frame.attrs = {"model": type(self).__name__, "kind": "weights", "origin": data.index[-1]}
        return frame

    def _fit(self, data, **fit_kwargs):
        weights = self._weights(data)
        book = (weights.to_numpy() * data.y.to_numpy()[self.hist_len:]).sum(axis=1)
        sharpe = float(book.mean() / book.std()) if book.std() > 0 else 0.0
        self.data_ = data
        self.result_ = _DummyAllocResult(weights=weights.to_numpy(), sharpe=sharpe)

    def allocate(self, data=None):
        self._check_is_fitted("allocate")
        if data is None and "data_" not in vars(self):
            raise DataValidationError("This model was loaded without its training data; call allocate(data=...).")
        panel = self.data_ if data is None else self._validate_data(data)
        return self._weights(panel)

    def update(self, data, *, refit=False):
        self._check_is_fitted("update")
        panel = self._validate_data(data)
        if refit:
            self._fit(panel)
        self.data_ = panel
        return self

    def _save_state(self, path):
        self._write_state(path)
        self._write_result(path)

    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        est.result_ = cls._read_result(path, _DummyAllocResult)

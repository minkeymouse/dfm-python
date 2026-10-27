"""A scaled mean forecaster and a custom scaler for the save/load tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

from dfm_python.base import BaseFactorModel
from dfm_python.config import EstimatorConfig

from ..contract._dummies import _frame


class DoublingScaler(TransformerMixin, BaseEstimator):
    """A scaler that is not one of the five known ones (so it is pickled)."""

    def __init__(self, factor=2.0):
        self.factor = factor

    def fit(self, X, y=None):
        self.scale_ = np.full(np.asarray(X).shape[1], float(self.factor))
        return self

    def transform(self, X):
        return np.asarray(X) * self.scale_

    def inverse_transform(self, X):
        return np.asarray(X) / self.scale_


SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": lambda: QuantileTransformer(n_quantiles=10),
    "custom": DoublingScaler,
}


@dataclass
class ScaledMeanConfig(EstimatorConfig):
    scaler: Any = None
    verbose: int = 0


@dataclass
class ScaledMeanResult:
    mean: np.ndarray
    n_obs: int


class ScaledMean(BaseFactorModel):
    """Forecast the column means in scaled space, mapped back with ``scaler_``.

    ``scaler`` is None, a name in :data:`SCALERS` or a scaler instance (cloned).
    """

    _tags = {"predict_data": "full_panel"}
    _config_class = ScaledMeanConfig

    def __init__(self, *, scaler=None, verbose=0):
        self.scaler = scaler
        self.verbose = verbose

    @classmethod
    def get_test_params(cls):
        return [{}, {"scaler": "standard"}, {"scaler": "quantile"}, {"scaler": "custom"}]

    def _fit(self, data, **fit_kwargs):
        scaler = self.scaler
        if isinstance(scaler, str):
            scaler = SCALERS[scaler]()
        elif scaler is not None:
            scaler = clone(scaler)
        y = data.y.to_numpy()
        z = y if scaler is None else scaler.fit_transform(y)
        self.scaler_ = scaler
        self.mean_ = z.mean(axis=0)
        self.result_ = ScaledMeanResult(mean=self.mean_.copy(), n_obs=data.n_periods)

    def predict(self, h, data=None, **predict_kwargs):
        self._check_is_fitted("predict")
        h = self._validate_horizon(h)
        origin = self.index_ if data is None else self._validate_data(data).index
        z = np.tile(self.mean_, (h, 1))
        values = z if self.scaler_ is None else self.scaler_.inverse_transform(z)
        return _frame(values, self, origin, self.feature_names_in_, "forecast")

    def _save_state(self, path):
        self._write_state(path, {"mean": self.mean_}, {"n_obs": self.result_.n_obs})
        self._write_result(path)

    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        arrays, _ = cls._read_state(path)
        est.mean_ = arrays["mean"]
        est.result_ = cls._read_result(path, ScaledMeanResult)

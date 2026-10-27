"""Tiny models on the new base classes (:mod:`dfm_python._estimator`), for ``_testing.check_estimator``.

- ``DummyForecaster`` (numpy): one factor, the loading-weighted mean of the
  standardized observed series; an AR(1) forecast with Gaussian intervals.
  It accepts mixed frequency, so slower series are masked off their period
  ends.
- ``DummyTorchForecaster``: the same model with its loadings in a torch
  module trained by Adam (``weights.pt``, ``allclose`` reload checks).
- ``DummyAllocator``: weights by the sign of the trailing mean return,
  L1-normalized; it accepts characteristics and exog and new assets in
  ``update``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from dfm_python import _identify, _time
from dfm_python._estimator import TrainingLog, _Allocator, _Forecaster
from dfm_python._namespace import Field, Namespace
from dfm_python.dataset.panel import PanelData
from dfm_python.errors import ConfigurationError


def _factor_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data["f"], index=data["index"], columns=pd.Index(data["factor_names"], name="factor"))


def _common_frame(data: dict) -> pd.DataFrame:
    values = data["center"] + data["scale"] * (data["f"] @ data["loadings"].T)
    return pd.DataFrame(values, index=data["index"], columns=pd.Index(data["series"], name="series"))


def _shape(data: dict) -> tuple:
    return (len(data["index"]), len(data["series"]))


def _time_index(index: pd.Index) -> pd.Index:
    """The namespace frames' time index: the data's, named ``"time"`` when unnamed (spec section 5)."""
    return index.rename(index.name if index.name is not None else "time")


class DummyForecaster(_Forecaster):
    """One-factor forecaster: f_t = sum_i l_i x_it / sum_i l_i^2 over observed standardized x_it."""

    _accepts_mixed_frequency = True
    _warns_on_nonconvergence = True

    def __init__(self, *, max_iter=3, tol=0.0, sign_anchors=None, clock=None, verbose=0):
        self.max_iter = max_iter
        self.tol = tol
        self.sign_anchors = sign_anchors
        self.clock = clock
        self.verbose = verbose

    @classmethod
    def _test_params(cls):
        return [{}, {"max_iter": 2, "sign_anchors": {"factor_0": "s1"}}]

    def _validate_params(self):
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ConfigurationError(f"max_iter must be a positive int, got {self.max_iter!r}.")

    def _explicit_clock(self):
        return self.clock

    # Estimation

    def _standardize(self, panel: PanelData) -> np.ndarray:
        return (panel._y.to_numpy() - self._center_) / self._scale_

    def _filter(self, x: np.ndarray, loadings: np.ndarray, phi: float) -> np.ndarray:
        f = np.zeros(len(x))
        for t in range(len(x)):
            observed = ~np.isnan(x[t])
            if observed.any():
                l_obs = loadings[observed]
                f[t] = l_obs @ x[t, observed] / max(l_obs @ l_obs, 1e-12)
            else:
                f[t] = phi * f[t - 1] if t else 0.0
        return f

    def _estimate(self, panel: PanelData, log: TrainingLog) -> str:
        y = panel._y.to_numpy()
        self._center_ = np.nanmean(y, axis=0)
        self._scale_ = np.nanstd(y, axis=0)
        x = self._standardize(panel)
        loadings = np.ones(panel.n_series)
        phi, previous, reason = 0.0, np.inf, "max_iter"
        for _ in range(self.max_iter):
            f = self._filter(x, loadings, phi)
            observed = ~np.isnan(x)
            loadings = np.array([
                (f[observed[:, i]] @ x[observed[:, i], i]) / max(f[observed[:, i]] @ f[observed[:, i]], 1e-12)
                for i in range(panel.n_series)
            ])
            phi = float(f[1:] @ f[:-1] / max(f[:-1] @ f[:-1], 1e-12))
            mse = float(np.nanmean((x - np.outer(f, loadings)) ** 2))
            if log(mse):
                reason = "callback"
                break
            if abs(previous - mse) < self.tol:
                reason = "converged"
                break
            previous = mse
        f = self._filter(x, loadings, phi)
        anchors = _identify.resolve_anchors(self.sign_anchors, ["factor_0"], panel.series_names)
        self._transform_ = _identify.identify(loadings[:, None], anchors=anchors, rotate=False)
        self._loadings_ = loadings
        self._phi_ = phi
        self._sigma2_ = float(np.var(f[1:] - phi * f[:-1]))
        self._noise_ = np.nanmean((x - np.outer(f, loadings)) ** 2, axis=0)
        self._set_params_namespace()
        return reason

    def _set_params_namespace(self) -> None:
        names = pd.Index(list(self.feature_names_in_), name="series")
        reported = _identify.apply_loadings(self._loadings_[:, None], self._transform_)
        series = pd.DataFrame({"center": self._center_, "scale": self._scale_}, index=names)
        loadings = pd.DataFrame(self._scale_[:, None] * reported, index=names, columns=pd.Index(["factor_0"], name="factor"))
        self.params_ = Namespace("DummyForecaster.params_", [
            Field("series", "center and scale per series", "data units"),
            Field("loadings", "data units per factor unit", "data units"),
        ], {"series": series, "loadings": loadings})

    def _condition(self, panel: PanelData, previous: Optional[PanelData]) -> None:
        f = self._filter(self._standardize(panel), self._loadings_, self._phi_)
        self._f_ = f
        reported = _identify.apply_factors(f[:, None], self._transform_)
        data = {
            "f": reported, "index": _time_index(panel.index), "factor_names": ["factor_0"],
            "series": list(panel.series_names), "center": self._center_, "scale": self._scale_,
            "loadings": _identify.apply_loadings(self._loadings_[:, None], self._transform_),
        }
        self.states_ = Namespace("DummyForecaster.states_", [
            Field("factors", "the filtered factor", "factor units", build=_factor_frame, shape=_shape),
            Field("common", "center + scale * l f", "data units", build=_common_frame, shape=_shape),
        ], data)
        self.factors_ = self.states_.factors

    # Moments

    def _path(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Factor means and variances at the positions (in sample: the filtered factor, variance 0)."""
        n = len(self._f_)
        steps = np.maximum(positions - (n - 1), 0)
        f = np.where(positions < n, self._f_[np.minimum(positions, n - 1)], self._f_[-1] * self._phi_ ** steps)
        var = self._sigma2_ * np.array([sum(self._phi_ ** (2 * j) for j in range(h)) for h in steps])
        return f, var

    def _moments(self, stamps, exog, coverage):
        f, var = self._path(self._target_positions(stamps))
        mean = self._center_ + self._scale_ * np.outer(f, self._loadings_)
        if coverage is None:
            return mean, None, None
        sd = self._scale_ * np.sqrt(np.outer(var, self._loadings_ ** 2) + self._noise_)
        z = norm.ppf((1 + coverage) / 2)
        return mean, mean - z * sd, mean + z * sd

    def _factor_path(self, stamps, exog):
        f, _ = self._path(self._target_positions(stamps))
        return _identify.apply_factors(f[:, None], self._transform_)

    # Save / load

    def _save_fitted(self):
        arrays = {"center": self._center_, "scale": self._scale_, "loadings": self._loadings_, "noise": self._noise_}
        return arrays, {"phi": self._phi_, "sigma2": self._sigma2_}

    def _load_fitted(self, arrays, values, *, weights=None, device=None):
        self._center_, self._scale_ = arrays["center"], arrays["scale"]
        self._loadings_, self._noise_ = arrays["loadings"], arrays["noise"]
        self._phi_, self._sigma2_ = float(values["phi"]), float(values["sigma2"])
        self._set_params_namespace()


class DummyTorchForecaster(DummyForecaster):
    """``DummyForecaster`` whose loadings are a torch ``Linear(1, N)`` trained by Adam (saved in ``weights.pt``)."""

    _requires_torch = True
    _accepts_mixed_frequency = False
    _warns_on_nonconvergence = False

    def __init__(self, *, max_iter=4, learning_rate=0.05, random_state=0, device="auto", verbose=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    @classmethod
    def _test_params(cls):
        return [{"device": "cpu"}]

    def _explicit_clock(self):
        return None

    def _module(self, n_series: int) -> Any:
        import torch

        module = torch.nn.Linear(1, n_series, bias=False, dtype=torch.float64)
        with torch.no_grad():
            module.weight.copy_(torch.ones(n_series, 1, dtype=torch.float64))
        return module.to(self._device_)

    def _estimate(self, panel, log):
        import torch

        self._device_ = self._torch_device()
        y = panel._y.to_numpy()
        self._center_ = np.nanmean(y, axis=0)
        self._scale_ = np.nanstd(y, axis=0)
        x = self._standardize(panel)
        self._network_ = self._module(panel.n_series)
        optimizer = torch.optim.Adam(self._network_.parameters(), lr=self.learning_rate)
        target = torch.as_tensor(np.nan_to_num(x), device=self._device_)
        mask = torch.as_tensor(~np.isnan(x), device=self._device_)
        generator = np.random.default_rng(self._seed_)
        reason = "max_iter"
        for _ in range(self.max_iter):
            loadings = self._network_.weight.detach().cpu().numpy()[:, 0]
            f = torch.as_tensor(self._filter(x, loadings, 0.0)[:, None], device=self._device_)
            f = f + torch.as_tensor(1e-3 * generator.standard_normal(f.shape), device=self._device_)
            loss = ((self._network_(f) - target) ** 2)[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if log(loss.item()):
                reason = "callback"
                break
        self._loadings_ = self._network_.weight.detach().cpu().numpy()[:, 0].copy()
        f = self._filter(x, self._loadings_, 0.0)
        self._phi_ = float(f[1:] @ f[:-1] / max(f[:-1] @ f[:-1], 1e-12))
        self._sigma2_ = float(np.var(f[1:] - self._phi_ * f[:-1]))
        self._noise_ = np.nanmean((x - np.outer(f, self._loadings_)) ** 2, axis=0)
        self._transform_ = _identify.identify(self._loadings_[:, None], rotate=False)
        self._set_params_namespace()
        return reason

    def _fitted_weights(self):
        return self._network_.state_dict()

    def _load_fitted(self, arrays, values, *, weights=None, device=None):
        self._device_ = device
        self._network_ = self._module(len(arrays["center"]))
        self._network_.load_state_dict(weights)
        super()._load_fitted(arrays, values, weights=weights, device=device)
        self._loadings_ = self._network_.weight.detach().cpu().numpy()[:, 0].copy()


def _weights_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data["weights"], index=data["index"], columns=pd.Index(data["series"], name="series"))


class DummyAllocator(_Allocator):
    """Weights by the sign of each asset's trailing mean return over ``window`` rows, L1-normalized."""

    _accepts_exog = True
    _accepts_characteristics = True
    _new_series_on_update = True

    def __init__(self, *, window=5, max_iter=2, random_state=0, verbose=0):
        self.window = window
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    @classmethod
    def _test_params(cls):
        return [{}, {"window": 3, "max_iter": 0}]

    def _set_params_namespace(self, scale: np.ndarray) -> None:
        series = pd.DataFrame({"scale": scale}, index=pd.Index(list(self.feature_names_in_), name="series"))
        self.params_ = Namespace("DummyAllocator.params_", [Field("series", "per-asset scale", "return units")],
                                 {"series": series})

    def _estimate(self, panel, log):
        returns = panel._y.to_numpy()
        self._asset_scale_ = np.nanstd(returns, axis=0)
        self._set_params_namespace(self._asset_scale_)
        for i in range(self.max_iter):
            if log(-float(np.nanmean(returns)) * (i + 1), stage="train"):
                return "callback"
        return "max_iter" if self.max_iter else "not_trainable"

    def _decisions(self, returns: np.ndarray) -> np.ndarray:
        """Weights decided at the end of each row (labeled by the next one), (T, N)."""
        out = np.zeros_like(returns)
        for t in range(len(returns)):
            window = returns[max(0, t - self.window + 1):t + 1]
            signs = np.sign(np.nansum(window, axis=0))
            total = np.abs(signs).sum()
            out[t] = signs / total if total else 0.0
        return out

    def _condition(self, panel, previous):
        returns = panel._y.to_numpy()
        decided = self._decisions(returns)
        index = _time_index(panel.index.append(pd.Index([_time.forward_stamp(panel.index)])).rename(panel.index.name))
        held = np.vstack([np.full((1, panel.n_series), np.nan), decided])
        factor = np.append(np.nanmean(returns, axis=1), np.nan)[:, None]
        self._next_ = decided[-1]
        data = {"weights": held, "f": factor, "index": index, "series": list(panel.series_names),
                "factor_names": ["factor_0"]}
        self.states_ = Namespace("DummyAllocator.states_", [
            Field("factors", "the cross-sectional mean return", "return units", build=_factor_frame),
            Field("weights", "the weights held over each period", "", build=_weights_frame),
        ], data)
        self.factors_ = self.states_.factors

    def _next_weights(self):
        return self._next_

    def _save_fitted(self):
        return {"asset_scale": self._asset_scale_}, {}

    def _load_fitted(self, arrays, values, *, weights=None, device=None):
        self._asset_scale_ = arrays["asset_scale"]
        self._set_params_namespace(self._asset_scale_)

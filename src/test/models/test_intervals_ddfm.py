"""Prediction intervals of DDFM."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

from dfm_python import DDFM, PanelData
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError

SERIES = ["s0", "s1", "s2", "s3"]
DDFM_PARAMS = {
    "n_factors": 2, "encoder_layers": (6,), "n_mc_samples": 3, "window_size": 32, "max_iter": 8,
    "scaler": "standard", "random_state": 0, "device": "cpu",
}
DDFM_MLP = {"decoder_type": "mlp", "encoder_layers": (6, 4)}
T_FIT = 72


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engines seed the global torch and numpy RNGs when they build their networks; restore them after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _simulate(n_time: int, seed: int, n_series: int = len(SERIES)) -> pd.DataFrame:
    """A one-factor model: f_t = 0.7 f_{t-1} + N(0, 1), x_t = 3 + L f_t + N(0, 0.5^2), monthly from 2000."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=n_series)
    f = np.zeros(n_time)
    for t in range(1, n_time):
        f[t] = 0.7 * f[t - 1] + rng.normal()
    values = 3.0 + np.outer(f, loadings) + 0.5 * rng.normal(size=(n_time, n_series))
    idx = pd.date_range("2000-01-31", periods=n_time, freq="ME", name="date")
    return pd.DataFrame(values, index=idx, columns=[f"s{i}" for i in range(n_series)])


def _fit(cls: type, data: object, **params: object) -> object:
    with torch.random.fork_rng():
        return cls(**params).fit(data)


def _bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame.xs("lower", axis=1, level="bound").to_numpy(), frame.xs("upper", axis=1, level="bound").to_numpy()


def _assert_interval_frame(frame: pd.DataFrame, point: pd.DataFrame, coverage: float) -> None:
    assert frame.index.equals(point.index) and frame.columns.names == ["series", "bound"]
    assert list(frame.columns) == [(s, b) for s in point.columns for b in ("lower", "upper")]
    assert frame.attrs == {**point.attrs, "coverage": coverage}
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    lower, upper = _bounds(frame)
    assert np.isfinite(lower).all() and (lower <= upper).all()


# -- DDFM ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _simulate(T_FIT + 12, seed=1)


@pytest.fixture(scope="module", params=["linear", "mlp"])
def ddfm(request, frame) -> DDFM:
    extra = DDFM_MLP if request.param == "mlp" else {}
    return _fit(DDFM, PanelData(y=frame.iloc[:T_FIT]), **{**DDFM_PARAMS, **extra})


@pytest.fixture(scope="module")
def ddfm_linear(frame) -> DDFM:
    """The linear decoder: Gaussian intervals in closed form, centred on predict."""
    return _fit(DDFM, PanelData(y=frame.iloc[:T_FIT]), **DDFM_PARAMS)


@pytest.fixture(scope="module")
def ddfm_mlp(frame) -> DDFM:
    """The MLP decoder: intervals from simulated paths."""
    return _fit(DDFM, PanelData(y=frame.iloc[:T_FIT]), **{**DDFM_PARAMS, **DDFM_MLP})


def _decoder64(model: DDFM) -> torch.nn.Module:
    return copy.deepcopy(model.network_.decoder).to(device="cpu", dtype=torch.float64).eval()


def _decode(decoder: torch.nn.Module, Z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return decoder(torch.tensor(np.atleast_2d(Z), dtype=torch.float64)).numpy()


def _jacobian_fd(decoder: torch.nn.Module, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite differences of the decoder at ``z`` (r,): (N, r)."""
    cols = []
    for j in range(len(z)):
        step = np.zeros_like(z)
        step[j] = eps
        cols.append((_decode(decoder, z + step)[0] - _decode(decoder, z - step)[0]) / (2 * eps))
    return np.stack(cols, axis=1)


def _ddfm_half_widths(model: DDFM, scaled: np.ndarray, h: int, coverage: float, P0: np.ndarray,
                      z_forecast: np.ndarray) -> np.ndarray:
    """``q * sd`` in the scaled space, from the forecast covariance of the state space through the decoder.

    ``P_k = A^k P0 A'^k + sum_{j<k} A^j Q A'^j`` and ``var = diag(J_k P_k J_k') + s2``, with ``J_k`` the
    decoder's Jacobian at the step-k forecast factors (finite differences here) and ``s2`` the mean
    square of the observed training residuals ``x - g(Z)`` around the decoder ``g``.
    """
    A, Q, Z = (np.asarray(m, dtype=np.float64) for m in (model.result_.A, model.result_.Q, model.result_.Z))
    decoder = _decoder64(model)
    s2 = np.nanmean((scaled - _decode(decoder, Z)) ** 2, axis=0)
    power = [np.linalg.matrix_power(A, j) for j in range(h + 1)]
    variances = []
    for k in range(1, h + 1):
        P = power[k] @ P0 @ power[k].T + sum(power[j] @ Q @ power[j].T for j in range(k))
        J = _jacobian_fd(decoder, np.asarray(z_forecast[k - 1], dtype=np.float64))
        variances.append(np.diag(J @ P @ J.T) + s2)
    return norm.ppf((1 + coverage) / 2) * np.sqrt(np.array(variances))


def _mc_covariance(model: DDFM) -> np.ndarray:
    draws = model.network_.factors[:, -1, :].astype(np.float64)
    return np.cov(draws, rowvar=False, bias=True).reshape(draws.shape[1], draws.shape[1])


def test_ddfm_interval_is_centred_on_predict(ddfm_linear):
    point = ddfm_linear.predict(5)
    frame = ddfm_linear.predict_interval(5, coverage=0.8)
    _assert_interval_frame(frame, point, 0.8)
    lower, upper = _bounds(frame)
    assert (lower < point.to_numpy()).all() and (point.to_numpy() < upper).all()
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-13, atol=1e-12)
    # As the coverage goes to 0 the interval collapses onto the point forecast, bitwise.
    lower, upper = _bounds(ddfm_linear.predict_interval(5, coverage=1e-300))
    assert lower.tobytes() == point.to_numpy().tobytes() and upper.tobytes() == point.to_numpy().tobytes()
    # Deterministic (closed form) and wider with the coverage and the horizon.
    assert ddfm_linear.predict_interval(5, coverage=0.8).equals(frame)
    narrow, wide = _bounds(frame), _bounds(ddfm_linear.predict_interval(5, coverage=0.95))
    assert ((wide[1] - wide[0]) > (narrow[1] - narrow[0])).all()
    assert (np.diff(narrow[1] - narrow[0], axis=0) >= 0).all()


def test_ddfm_bounds_match_the_forecast_covariance(ddfm_linear, frame):
    """The bounds are the Gaussian quantiles of the state space's forecast covariance through the decoder."""
    lower, upper = _bounds(ddfm_linear.predict_interval(6, coverage=0.9))
    scaled = ddfm_linear.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    zf = ddfm_linear.predict_factors(6).to_numpy()
    expected = _ddfm_half_widths(ddfm_linear, scaled, 6, 0.9, _mc_covariance(ddfm_linear), zf) * ddfm_linear.scaler_.scale_
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-6)


def test_ddfm_interval_starts_from_the_spread_of_the_last_mc_draws(ddfm_linear, frame):
    """The initial covariance is that of the MC factor draws at the last row: spread them around the same mean."""
    model = copy.deepcopy(ddfm_linear)
    factors = model.network_.factors.copy()
    factors[:, -1, :] += np.array([[0.5, 0.3], [-0.5, 0.0], [0.0, -0.3]], dtype=factors.dtype)
    model.network_.factors = factors
    P0 = _mc_covariance(model)
    assert np.trace(P0) > 0.1
    lower, upper = _bounds(model.predict_interval(3))
    scaled = model.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    expected = _ddfm_half_widths(model, scaled, 3, 0.9, P0, model.predict_factors(3).to_numpy()) * model.scaler_.scale_
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-6)
    before = _bounds(ddfm_linear.predict_interval(3))
    assert ((upper - lower) > (before[1] - before[0])).all()


def test_ddfm_interval_with_data_and_after_update(ddfm_linear, frame):
    new = PanelData(y=frame.iloc[T_FIT:T_FIT + 5])
    before = ddfm_linear.predict_interval(3)
    conditioned = ddfm_linear.predict_interval(3, data=new)
    assert ddfm_linear.predict_interval(3).equals(before)  # the model is unchanged
    point = ddfm_linear.predict(3, data=new)
    _assert_interval_frame(conditioned, point, 0.9)
    assert conditioned.index[0] == frame.index[T_FIT + 5]
    lower, upper = _bounds(conditioned)
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-13, atol=1e-12)
    # The encoder maps an appended row to one factor value: the forecast starts from a known state.
    scaled = ddfm_linear.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    r = ddfm_linear.n_factors_
    zf = ddfm_linear.predict_factors(3, data=new).to_numpy()
    expected = _ddfm_half_widths(ddfm_linear, scaled, 3, 0.9, np.zeros((r, r)), zf) * ddfm_linear.scaler_.scale_
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-6)
    updated = copy.deepcopy(ddfm_linear).update(new)
    assert updated.predict_interval(3).to_numpy().tobytes() == conditioned.to_numpy().tobytes()


def test_ddfm_interval_with_a_non_affine_scaler(frame):
    """The scaled bounds go through inverse_transform; predict, the image of the scaled centre, is their median."""
    params = {**DDFM_PARAMS, "scaler": QuantileTransformer(n_quantiles=40)}
    model = _fit(DDFM, PanelData(y=frame.iloc[:T_FIT]), **params)
    point = model.predict(4)
    frame_ = model.predict_interval(4, coverage=0.7)
    _assert_interval_frame(frame_, point, 0.7)
    lower, upper = _bounds(frame_)
    assert (lower <= point.to_numpy()).all() and (point.to_numpy() <= upper).all()
    scaled = model.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    zf = model.predict_factors(4).to_numpy()
    half = _ddfm_half_widths(model, scaled, 4, 0.7, _mc_covariance(model), zf)
    centre = model.network_.decode(zf)
    np.testing.assert_allclose(lower, model.scaler_.inverse_transform(centre - half), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(upper, model.scaler_.inverse_transform(centre + half), rtol=1e-6, atol=1e-8)


def test_ddfm_mlp_interval_is_seeded_and_widens_with_coverage(ddfm_mlp):
    point = ddfm_mlp.predict(4)
    frame = ddfm_mlp.predict_interval(4, coverage=0.8)
    _assert_interval_frame(frame, point, 0.8)
    assert ddfm_mlp.predict_interval(4, coverage=0.8).equals(frame)  # seeded with random_state
    narrow, wide = _bounds(ddfm_mlp.predict_interval(4, coverage=0.5)), _bounds(ddfm_mlp.predict_interval(4, coverage=0.95))
    assert ((wide[1] - wide[0]) > (narrow[1] - narrow[0])).all()


def test_ddfm_mlp_interval_matches_a_long_independent_simulation(ddfm_mlp, frame):
    """The quantiles of 2000 seeded paths agree with those of 40000 paths simulated here, within sampling error."""
    model, h, coverage = ddfm_mlp, 3, 0.8
    lower, upper = _bounds(model.predict_interval(h, coverage=coverage))
    A, Q, Z = (np.asarray(m, dtype=np.float64) for m in (model.result_.A, model.result_.Q, model.result_.Z))
    decoder = _decoder64(model)
    scaled = model.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    sd = np.sqrt(np.nanmean((scaled - _decode(decoder, Z)) ** 2, axis=0))
    draws = model.network_.factors[:, -1, :].astype(np.float64)
    rng = np.random.default_rng(12345)
    z = rng.multivariate_normal(draws.mean(axis=0), _mc_covariance(model), size=40000, method="eigh")
    for step in range(h):
        z = z @ A.T + rng.multivariate_normal(np.zeros(len(A)), Q, size=40000, method="eigh")
        y = _decode(decoder, z) + sd * rng.standard_normal((40000, len(sd)))
        lo, hi = np.quantile(model.scaler_.inverse_transform(y), [(1 - coverage) / 2, (1 + coverage) / 2], axis=0)
        width = hi - lo
        np.testing.assert_array_less(np.abs(lower[step] - lo), 0.1 * width)
        np.testing.assert_array_less(np.abs(upper[step] - hi), 0.1 * width)


def test_ddfm_interval_coverage_on_a_simulated_factor_model():
    """Rolling one- and three-step intervals over 100 new rows of a known linear one-factor model.

    The parameters are estimated on 160 rows and taken as known, so the coverage may fall a little short
    of the nominal 0.8 (about 0.75 here).
    """
    n_fit, n_new = 160, 100
    df = _simulate(n_fit + n_new + 3, seed=8, n_series=5)
    params = {**DDFM_PARAMS, "n_factors": 1, "activation": "tanh", "learning_rate": 0.02, "max_iter": 30}
    model = _fit(DDFM, PanelData(y=df.iloc[:n_fit]), **params)
    assert model.explained_variance_.mean() > 0.5  # the factor carries most of the variance
    hits = []
    for j in range(n_new):
        data = None if j == 0 else PanelData(y=df.iloc[n_fit:n_fit + j])
        lower, upper = _bounds(model.predict_interval(3, coverage=0.8, data=data))
        truth = df.to_numpy()[n_fit + j:n_fit + j + 3]
        hits.append((truth >= lower) & (truth <= upper))
    hits = np.array(hits)  # (origins, steps, series)
    for step in (0, 2):
        assert 0.65 <= hits[:, step].mean() <= 0.92, (step, hits[:, step].mean())


def test_ddfm_interval_errors(ddfm):
    with pytest.raises(NotFittedError, match="predict_interval"):
        DDFM(**DDFM_PARAMS).predict_interval(2)
    for coverage in (0, 1, 1.5, -0.1, float("nan"), True, "0.9", None):
        with pytest.raises(ConfigurationError, match="coverage"):
            ddfm.predict_interval(2, coverage=coverage)
    with pytest.raises(ValueError, match="positive"):
        ddfm.predict_interval(0)
    with pytest.raises(DataValidationError, match="overlaps"):
        ddfm.predict_interval(2, data=PanelData(y=_simulate(T_FIT, seed=1).iloc[-3:]))


def test_ddfm_interval_save_load_and_polars(ddfm, tmp_path):
    ddfm.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run", device="cpu")
    expected = ddfm.predict_interval(4, coverage=0.6)
    assert loaded.predict_interval(4, coverage=0.6).to_numpy().tobytes() == expected.to_numpy().tobytes()
    pl = pytest.importorskip("polars")
    out = copy.deepcopy(ddfm).set_output(output="polars").predict_interval(4, coverage=0.6)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["date", *[f"{s}__{b}" for s in SERIES for b in ("lower", "upper")]]
    np.testing.assert_array_equal(out.drop("date").to_numpy(), expected.to_numpy())

"""Prediction intervals of iVDFM, and its exact loadings with a linear decoder."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

from dfm_python import PanelData, iVDFM
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

SERIES = ["s0", "s1", "s2", "s3"]
IVDFM_PARAMS = {
    "n_factors": 2, "window": 12, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
    "encoder_n_hidden_layers": 1, "decoder_n_hidden_layers": 1, "prior_n_hidden_layers": 1,
    "batch_size": 4, "max_epochs": 4, "scaler": "standard", "random_state": 0, "device": "cpu",
}
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


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _simulate(T_FIT + 12, seed=1)


# -- iVDFM --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ivdfm(frame) -> iVDFM:
    return _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **IVDFM_PARAMS)


@pytest.fixture(scope="module")
def ivdfm_exog(frame) -> tuple[iVDFM, pd.DataFrame]:
    y, exog = frame[SERIES[:3]], frame[[SERIES[3]]].rename(columns={"s3": "u"})
    return _fit(iVDFM, PanelData(y=y.iloc[:T_FIT], exog=exog.iloc[:T_FIT]), **IVDFM_PARAMS), exog


def test_ivdfm_interval_is_seeded_nested_and_holds_the_forecast(ivdfm):
    point = ivdfm.predict(5)
    state = torch.random.get_rng_state()
    frame = ivdfm.predict_interval(5)
    assert torch.equal(torch.random.get_rng_state(), state)  # the global RNG is restored
    _assert_interval_frame(frame, point, 0.9)
    assert ivdfm.predict_interval(5).equals(frame)  # the same draws on every call
    lower, upper = _bounds(frame)
    inside = (point.to_numpy() >= lower) & (point.to_numpy() <= upper)
    assert inside.mean() >= 0.8
    # The same draws for every coverage, so the intervals are nested.
    inner = _bounds(ivdfm.predict_interval(5, coverage=0.5))
    assert (inner[0] >= lower).all() and (inner[1] <= upper).all() and ((inner[1] - inner[0]) < (upper - lower)).all()


def test_ivdfm_interval_is_the_quantiles_of_the_seeded_paths(ivdfm):
    """The bounds are the empirical quantiles of 1000 paths drawn from the engine with the run's seed."""
    with torch.random.fork_rng():
        torch.manual_seed(ivdfm.random_state_)
        paths = ivdfm.network_._sample_paths(3, None, 1000).numpy().astype(np.float64)
    assert paths.shape == (1000, 3, len(SERIES))
    paths = ivdfm.scaler_.inverse_transform(paths.reshape(-1, len(SERIES))).reshape(paths.shape)
    lower, upper = _bounds(ivdfm.predict_interval(3, coverage=0.8))
    np.testing.assert_allclose(lower, np.quantile(paths, 0.1, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(upper, np.quantile(paths, 0.9, axis=0), rtol=1e-12, atol=1e-12)


def test_ivdfm_interval_matches_the_gaussian_forecast_distribution(frame):
    """Linear decoder, Gaussian innovations, AR(1): the paths are Gaussian with a mean and variance derived here.

    Step k's factors are the deterministic forecast plus ``sum_{j<=k} H_{k-j} eta_j`` (``H`` the impulse
    response, ``eta_j ~ N(mu_j, var_j)`` from the prior at the future context), so the scaled targets are
    Gaussian with mean ``W m_k + b`` and variance ``W^2 v_k + decoder_var``. The 1000-path quantiles match
    that distribution's to within a few Monte Carlo standard errors (about 0.07 sd per bound).
    """
    h, coverage, decoder_var = 4, 0.9, 0.1
    params = {**IVDFM_PARAMS, "decoder_type": "linear", "innovation_distribution": "gaussian", "decoder_var": decoder_var}
    model = _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **params)
    engine = model.network_
    u = torch.as_tensor(engine._future_time_features(h, model.time_context_)).unsqueeze(0)
    with torch.no_grad():
        prior = engine.prior_network(u)
        impulse = engine.ssm.get_impulse_response(h).double().numpy()  # (r, h): B * A^k
    mu, var = prior["mu"][0].double().numpy(), prior["logvar"][0].double().exp().numpy()
    W = engine.decoder.linear.weight.detach().double().numpy()
    b = engine.decoder.linear.bias.detach().double().numpy()
    deterministic = model.predict_factors(h).to_numpy()
    mean = np.array([deterministic[k] + sum(impulse[:, k - j] * mu[j] for j in range(k + 1)) for k in range(h)])
    variance = np.array([sum(impulse[:, k - j] ** 2 * var[j] for j in range(k + 1)) for k in range(h)])
    scale, offset = model.scaler_.scale_, model.scaler_.mean_
    centre, sd = offset + scale * (mean @ W.T + b), scale * np.sqrt(variance @ (W**2).T + decoder_var)
    lower, upper = _bounds(model.predict_interval(h, coverage=coverage))
    z = norm.ppf((1 + coverage) / 2)
    assert (np.abs(lower - (centre - z * sd)) / sd).max() < 0.3
    assert (np.abs(upper - (centre + z * sd)) / sd).max() < 0.3


def test_ivdfm_interval_with_regimes_and_an_ar2(frame):
    """K > 1 regimes and AR(2) factors: the batched rollout of every path."""
    params = {**IVDFM_PARAMS, "n_regimes": 2, "factor_order": 2}
    model = _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **params)
    point = model.predict(4)
    interval = model.predict_interval(4)
    _assert_interval_frame(interval, point, 0.9)
    assert model.predict_interval(4).equals(interval)
    lower, upper = _bounds(interval)
    assert ((point.to_numpy() >= lower) & (point.to_numpy() <= upper)).mean() >= 0.8


def test_ivdfm_interval_without_a_seed_draws_from_the_global_rng(frame):
    model = _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **{**IVDFM_PARAMS, "random_state": None})
    torch.manual_seed(0)
    first = model.predict_interval(2)
    assert not model.predict_interval(2).equals(first)
    torch.manual_seed(0)
    assert model.predict_interval(2).equals(first)
    # With data the global RNG is restored afterwards, as for predict(h, data=...).
    window = PanelData(y=frame.iloc[T_FIT - 20:T_FIT + 6])
    state = torch.random.get_rng_state()
    conditioned = model.predict_interval(2, data=window)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert model.predict_interval(2, data=window).equals(conditioned)


def test_ivdfm_interval_with_data(ivdfm, frame):
    window = PanelData(y=frame.iloc[T_FIT - 20:T_FIT + 6])
    path = ivdfm.network_.factor_path.copy()
    conditioned = ivdfm.predict_interval(3, data=window)
    assert conditioned.equals(ivdfm.predict_interval(3, data=window))
    _assert_interval_frame(conditioned, ivdfm.predict(3, data=window), 0.9)
    assert conditioned.index[0] == frame.index[T_FIT + 6]
    np.testing.assert_array_equal(ivdfm.network_.factor_path, path)  # the model is unchanged
    assert ivdfm.conditioning_index_.equals(frame.index[:T_FIT])
    assert not conditioned.equals(ivdfm.predict_interval(3).set_axis(conditioned.index))


def test_ivdfm_interval_with_futr_exog(ivdfm_exog):
    model, exog = ivdfm_exog
    future = exog.iloc[T_FIT:T_FIT + 3]
    frame = model.predict_interval(3, futr_exog=future)
    _assert_interval_frame(frame, model.predict(3, futr_exog=future), 0.9)
    np.testing.assert_array_equal(model.predict_interval(3, futr_exog=future.to_numpy()).to_numpy(), frame.to_numpy())
    with pytest.warns(UserWarning, match="predict_interval got no futr_exog") as record:
        without = model.predict_interval(3)
    assert record[0].filename == __file__
    assert not np.array_equal(without.to_numpy(), frame.to_numpy())
    with pytest.raises(DataValidationError, match="shape"):
        model.predict_interval(3, futr_exog=future.to_numpy()[:2])


def test_ivdfm_interval_with_a_non_affine_scaler_and_student_t_innovations(frame):
    """Every path is inverse-transformed, so the bounds are its quantiles in data units for any scaler."""
    params = {**IVDFM_PARAMS, "scaler": QuantileTransformer(n_quantiles=40), "innovation_distribution": "student_t"}
    model = _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **params)
    frame_ = model.predict_interval(4, coverage=0.8)
    _assert_interval_frame(frame_, model.predict(4), 0.8)
    lower, upper = _bounds(frame_)
    observed = frame.iloc[:T_FIT].to_numpy()
    assert (lower >= observed.min(axis=0) - 1e-9).all() and (upper <= observed.max(axis=0) + 1e-9).all()
    assert np.isfinite(model.predict(4, deterministic=False, n_samples=2).to_numpy()).all()


def test_ivdfm_interval_errors_and_polars(ivdfm):
    with pytest.raises(NotFittedError, match="predict_interval"):
        iVDFM(**IVDFM_PARAMS).predict_interval(2)
    with pytest.raises(NotFittedError, match="predict"):  # before the arguments are validated
        iVDFM(**IVDFM_PARAMS).predict(2, deterministic="yes")
    with pytest.raises(ConfigurationError, match="coverage"):
        ivdfm.predict_interval(2, coverage=1.0)
    with pytest.raises(DataValidationError, match="predict_interval takes no futr_exog"):
        ivdfm.predict_interval(2, futr_exog=np.zeros((2, 1)))
    pl = pytest.importorskip("polars")
    out = copy.deepcopy(ivdfm).set_output(output="polars").predict_interval(2)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["date", *[f"{s}__{b}" for s in SERIES for b in ("lower", "upper")]]
    np.testing.assert_array_equal(out.drop("date").to_numpy(), ivdfm.predict_interval(2).to_numpy())


# -- iVDFM exact loadings -------------------------------------------------------------------


@pytest.mark.parametrize("mixing", [False, True])
def test_ivdfm_linear_decoder_has_exact_loadings(frame, mixing, tmp_path):
    y = frame.iloc[:T_FIT].copy()
    y.iloc[[4, 30], [0, 2]] = np.nan
    model = _fit(iVDFM, PanelData(y=y), **IVDFM_PARAMS, decoder_type="linear", mixing=mixing)
    engine = model.network_
    weight = engine.decoder.linear.weight.detach().double()
    if mixing:
        weight = weight @ engine.factor_mixing.M.weight.detach().double()
    assert model.loadings_kind_ == "exact"
    np.testing.assert_array_equal(model.loadings_.to_numpy(), weight.numpy())
    # decompose(by="factor"): scale * L_ij * f_jt per factor, offset + scale * b as "mean"; the parts add up.
    parts = model.decompose(by="factor")
    assert list(parts.columns.get_level_values("part").unique()) == ["factor_0", "factor_1", "mean", "idiosyncratic"]
    scale, offset = model.scaler_.scale_, model.scaler_.mean_
    f = model.factors_.to_numpy()
    for j, name in enumerate(model.factor_names_):
        np.testing.assert_allclose(parts[name].to_numpy(), np.outer(f[:, j], scale * weight.numpy()[:, j]), rtol=1e-12)
    bias = engine.decoder.linear.bias.detach().double().numpy()
    np.testing.assert_allclose(parts["mean"].to_numpy(), np.tile(offset + scale * bias, (T_FIT, 1)), rtol=1e-12)
    explained = sum(parts[name].to_numpy() for name in [*model.factor_names_, "mean"])
    total = explained + parts["idiosyncratic"].to_numpy()
    observed = y.notna().to_numpy()
    np.testing.assert_allclose(total[observed], y.to_numpy()[observed], rtol=0, atol=1e-9)
    assert np.isnan(parts["idiosyncratic"].to_numpy()[~observed]).all()
    # The explained part is the engine's float32 decoding, up to its rounding.
    np.testing.assert_allclose(explained, model.decompose()["common"].to_numpy(), rtol=1e-5, atol=1e-5)
    model.save(tmp_path / "run")
    loaded = iVDFM.load(tmp_path / "run", device="cpu")
    assert loaded.loadings_kind_ == "exact"
    pd.testing.assert_frame_equal(loaded.loadings_, model.loadings_, check_exact=True)
    pd.testing.assert_frame_equal(loaded.decompose(by="factor"), parts, check_exact=True)


def test_ivdfm_nonlinear_or_regime_decoding_keeps_the_mean_jacobian(frame):
    data = PanelData(y=frame.iloc[:T_FIT])
    regimes = _fit(iVDFM, data, **IVDFM_PARAMS, decoder_type="linear", n_regimes=2)
    residual = _fit(iVDFM, data, **IVDFM_PARAMS, decoder_type="residual")
    for model in (regimes, residual):
        assert model.loadings_kind_ == "mean_jacobian"
        with pytest.raises(UnsupportedInputError, match="nonlinear"):
            model.decompose(by="factor")
    quantile = _fit(iVDFM, data, **{**IVDFM_PARAMS, "scaler": QuantileTransformer(n_quantiles=40)},
                    decoder_type="linear")
    assert quantile.loadings_kind_ == "exact"
    with pytest.raises(UnsupportedInputError, match="affine"):
        quantile.decompose(by="factor")

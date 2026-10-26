"""Prediction intervals of DDFM and iVDFM, and iVDFM's exact loadings with a linear decoder."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

from dfm_python import DDFM, PanelData, iVDFM
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

SERIES = ["s0", "s1", "s2", "s3"]
DDFM_PARAMS = {
    "n_factors": 2, "encoder_layers": (6,), "n_mc_samples": 3, "window_size": 32, "max_iter": 8,
    "scaler": "standard", "random_state": 0, "device": "cpu",
}
DDFM_MLP = {"decoder_type": "mlp", "encoder_layers": (6, 4)}
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


# -- DDFM ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _simulate(T_FIT + 12, seed=1)


@pytest.fixture(scope="module", params=["linear", "mlp"])
def ddfm(request, frame) -> DDFM:
    extra = DDFM_MLP if request.param == "mlp" else {}
    return _fit(DDFM, PanelData(y=frame.iloc[:T_FIT]), **{**DDFM_PARAMS, **extra})


def _ddfm_half_widths(model: DDFM, scaled: np.ndarray, h: int, coverage: float, P0: np.ndarray) -> np.ndarray:
    """``q * sd`` in the scaled space, from the closed-form forecast covariance of the state space.

    ``P_k = A^k P0 A'^k + sum_{j<k} A^j Q A'^j`` and ``var = diag(C P_k C') + s2``, with ``s2`` the
    mean square of the observed training residuals ``x - Z C'``.
    """
    A, C, Q, Z = (np.asarray(m, dtype=np.float64) for m in (model.result_.A, model.result_.C, model.result_.Q,
                                                           model.result_.Z))
    s2 = np.nanmean((scaled - Z @ C.T) ** 2, axis=0)
    power = [np.linalg.matrix_power(A, j) for j in range(h + 1)]
    variances = []
    for k in range(1, h + 1):
        P = power[k] @ P0 @ power[k].T + sum(power[j] @ Q @ power[j].T for j in range(k))
        variances.append(np.diag(C @ P @ C.T) + s2)
    return norm.ppf((1 + coverage) / 2) * np.sqrt(np.array(variances))


def _mc_covariance(model: DDFM) -> np.ndarray:
    draws = model.network_.factors[:, -1, :].astype(np.float64)
    return np.cov(draws, rowvar=False, bias=True).reshape(draws.shape[1], draws.shape[1])


def test_ddfm_interval_is_centred_on_predict(ddfm):
    point = ddfm.predict(5)
    frame = ddfm.predict_interval(5, coverage=0.8)
    _assert_interval_frame(frame, point, 0.8)
    lower, upper = _bounds(frame)
    assert (lower < point.to_numpy()).all() and (point.to_numpy() < upper).all()
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-13, atol=1e-12)
    # As the coverage goes to 0 the interval collapses onto the point forecast, bitwise.
    lower, upper = _bounds(ddfm.predict_interval(5, coverage=1e-300))
    assert lower.tobytes() == point.to_numpy().tobytes() and upper.tobytes() == point.to_numpy().tobytes()
    # Deterministic (closed form) and wider with the coverage and the horizon.
    assert ddfm.predict_interval(5, coverage=0.8).equals(frame)
    narrow, wide = _bounds(frame), _bounds(ddfm.predict_interval(5, coverage=0.95))
    assert ((wide[1] - wide[0]) > (narrow[1] - narrow[0])).all()
    assert (np.diff(narrow[1] - narrow[0], axis=0) >= 0).all()


def test_ddfm_bounds_match_the_forecast_covariance(ddfm, frame):
    """Both decoders forecast with the linear state space (A, C, Q), so the bounds are its Gaussian quantiles."""
    lower, upper = _bounds(ddfm.predict_interval(6, coverage=0.9))
    scaled = ddfm.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    expected = _ddfm_half_widths(ddfm, scaled, 6, 0.9, _mc_covariance(ddfm)) * ddfm.scaler_.scale_
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-9)


def test_ddfm_interval_with_data_and_after_update(ddfm, frame):
    new = PanelData(y=frame.iloc[T_FIT:T_FIT + 5])
    before = ddfm.predict_interval(3)
    conditioned = ddfm.predict_interval(3, data=new)
    assert ddfm.predict_interval(3).equals(before)  # the model is unchanged
    point = ddfm.predict(3, data=new)
    _assert_interval_frame(conditioned, point, 0.9)
    assert conditioned.index[0] == frame.index[T_FIT + 5]
    lower, upper = _bounds(conditioned)
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-13, atol=1e-12)
    # The encoder maps an appended row to one factor value: the forecast starts from a known state.
    scaled = ddfm.scaler_.transform(frame.iloc[:T_FIT].to_numpy())
    r = ddfm.n_factors_
    expected = _ddfm_half_widths(ddfm, scaled, 3, 0.9, np.zeros((r, r))) * ddfm.scaler_.scale_
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-9)
    updated = copy.deepcopy(ddfm).update(new)
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
    half = _ddfm_half_widths(model, scaled, 4, 0.7, _mc_covariance(model))
    centre = model.predict_factors(4).to_numpy() @ model.result_.C.T
    np.testing.assert_allclose(lower, model.scaler_.inverse_transform(centre - half), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(upper, model.scaler_.inverse_transform(centre + half), rtol=1e-9, atol=1e-9)


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


def test_ivdfm_interval_without_a_seed_draws_from_the_global_rng(frame):
    model = _fit(iVDFM, PanelData(y=frame.iloc[:T_FIT]), **{**IVDFM_PARAMS, "random_state": None})
    torch.manual_seed(0)
    first = model.predict_interval(2)
    assert not model.predict_interval(2).equals(first)
    torch.manual_seed(0)
    assert model.predict_interval(2).equals(first)


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
    with pytest.warns(UserWarning, match="futr_exog") as record:
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
    with pytest.raises(ConfigurationError, match="coverage"):
        ivdfm.predict_interval(2, coverage=1.0)
    with pytest.raises(DataValidationError, match="futr_exog"):
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

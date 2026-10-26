"""Tests for the DDFM interpretation API: factor_names_, factors_, loadings_, explained_variance_ and decompose."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

from dfm_python import DDFM, PanelData
from dfm_python.utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError

TINY = {"n_factors": 2, "n_mc_samples": 2, "window_size": 16, "max_iter": 3, "random_state": 0, "device": "cpu"}
DECODERS = {"linear": {"encoder_layers": (4,)}, "mlp": {"decoder_type": "mlp", "encoder_layers": (6, 4)}}
SERIES = ["s0", "s1", "s2"]
T = 48


@pytest.fixture(autouse=True)
def _keep_global_rngs():
    """The engine seeds the global torch RNG when it builds the autoencoder; restore it after each test."""
    np_state = np.random.get_state()
    with torch.random.fork_rng():
        yield
    np.random.set_state(np_state)


def _panel(K: int, n_time: int = T) -> PanelData:
    """Three targets on different scales (two values missing) and ``K`` exog columns."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-31", periods=n_time, freq="ME", name="date")
    values = rng.standard_normal((n_time, 3)) * [1.0, 10.0, 100.0] + [0.0, 5.0, 1000.0]
    y = pd.DataFrame(values, index=idx, columns=SERIES)
    y.iloc[5, 0] = np.nan
    y.iloc[20, 2] = np.nan
    exog = pd.DataFrame(rng.standard_normal((n_time, K)), index=idx, columns=[f"x{k}" for k in range(K)]) if K else None
    return PanelData(y=y, exog=exog)


def _fit(decoder: str = "linear", K: int = 0, **params) -> DDFM:
    with torch.random.fork_rng():
        return DDFM(**{**TINY, **DECODERS[decoder], "scaler": "standard", **params}).fit(_panel(K))


@pytest.fixture(
    scope="module",
    params=[("linear", 0), ("linear", 1), ("mlp", 0), ("mlp", 1)],
    ids=["linear", "linear-exog", "mlp", "mlp-exog"],
)
def case(request) -> tuple[DDFM, PanelData]:
    decoder, K = request.param
    return _fit(decoder, K), _panel(K)


def _observed(panel: PanelData) -> np.ndarray:
    return panel.y.notna().to_numpy()


def _decoder64(model: DDFM) -> torch.nn.Module:
    return copy.deepcopy(model.network_.decoder).double()


# -- tags and attributes ---------------------------------------------------------


def test_tags_enable_interpret_and_prediction_intervals():
    tags = DDFM.get_tags()
    assert tags["capability:interpret"] is True and tags["capability:pred_int"] is True


def test_attributes_cover_the_targets(case):
    model, panel = case
    names = ["factor_0", "factor_1"]
    assert model.factor_names_.dtype == object and list(model.factor_names_) == names
    assert model.factors_.index.equals(model.index_) and list(model.factors_.columns) == names
    np.testing.assert_array_equal(model.factors_.to_numpy(), model.result_.Z.astype(np.float64), strict=True)
    # The exog columns are encoder covariates: they are not decoded, so not in loadings_.
    assert list(model.loadings_.index) == SERIES == list(model.feature_names_in_)
    assert list(model.loadings_.columns) == names
    assert all(dtype == np.float64 for dtype in model.loadings_.dtypes)
    assert np.isfinite(model.loadings_.to_numpy()).all()
    assert model.loadings_kind_ == ("exact" if model.decoder_type == "linear" else "mean_jacobian")
    ev = model.explained_variance_
    assert isinstance(ev, pd.Series) and ev.name == "explained_variance" and ev.dtype == np.float64
    assert list(ev.index) == SERIES and np.isfinite(ev.to_numpy()).all() and (ev <= 1).all()
    assert list(model.predict_factors(2).columns) == names


def test_linear_loadings_are_the_decoder_weight(case):
    model, _ = case
    if model.decoder_type != "linear":
        pytest.skip("linear decoder only")
    weight = model.network_.decoder.get_last_linear_layer().weight.detach().double().numpy()
    np.testing.assert_array_equal(model.loadings_.to_numpy(), weight, strict=True)
    np.testing.assert_array_equal(model.loadings_.to_numpy(), np.asarray(model.result_.C, dtype=np.float64))


def _finite_difference_loadings(model: DDFM) -> np.ndarray:
    """Central-difference decoder Jacobian (targets x factors), averaged over the rows of ``factors_``."""
    decoder = _decoder64(model)
    Z = torch.tensor(model.factors_.to_numpy(), dtype=torch.float64)
    step = 1e-6
    columns = []
    with torch.no_grad():
        for j in range(Z.shape[1]):
            shift = torch.zeros_like(Z)
            shift[:, j] = step
            columns.append(((decoder(Z + shift) - decoder(Z - shift)) / (2 * step)).mean(dim=0).numpy())
    return np.stack(columns, axis=1)


@pytest.mark.parametrize("decoder_type, K", [("linear", 0), ("linear", 1), ("mlp", 0), ("mlp", 1)])
def test_loadings_are_the_mean_finite_difference_jacobian(decoder_type, K):
    # tanh: smooth everywhere. With relu, factors at exactly 0 can sit on a decoder kink, where
    # central differences average the one-sided slopes but autograd takes relu'(0) = 0.
    model = _fit(decoder_type, K, activation="tanh")
    np.testing.assert_allclose(model.loadings_.to_numpy(), _finite_difference_loadings(model), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("activation", ["tanh", "sigmoid"])
def test_mlp_loadings_with_one_factor_and_one_target(activation):
    params = {**TINY, **DECODERS["mlp"], "n_factors": 1, "activation": activation, "scaler": "standard"}
    with torch.random.fork_rng():
        model = DDFM(**params).fit(_panel(1).y[["s1"]])
    assert model.loadings_.shape == (1, 1) and model.loadings_kind_ == "mean_jacobian"
    np.testing.assert_allclose(model.loadings_.to_numpy(), _finite_difference_loadings(model), rtol=1e-6, atol=1e-8)
    assert np.isfinite(model.explained_variance_.to_numpy()).all()


def test_explained_variance_matches_a_direct_computation(case):
    model, panel = case
    scaled = model.scaler_.transform(panel.y.to_numpy())
    with torch.no_grad():
        Z = torch.tensor(model.factors_.to_numpy(), dtype=torch.float32)
        common = model.network_.decoder(Z).double().numpy()
    observed = _observed(panel)
    expected = []
    for i in range(len(SERIES)):
        data = scaled[observed[:, i], i]
        expected.append(1.0 - np.var(data - common[observed[:, i], i]) / np.var(data))
    np.testing.assert_allclose(model.explained_variance_.to_numpy(), expected, rtol=1e-5, atol=1e-6)


# -- decompose -------------------------------------------------------------------


def test_decompose_by_component_adds_up_to_the_data(case):
    model, panel = case
    frame = model.decompose()
    assert frame.index.equals(model.index_)
    assert list(frame.columns.names) == ["component", "series"]
    assert list(frame.columns) == [(c, s) for c in ("common", "idiosyncratic") for s in SERIES]
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    observed = _observed(panel)
    data = panel.y.to_numpy()
    total = (frame["common"] + frame["idiosyncratic"]).to_numpy()
    np.testing.assert_allclose(total[observed], data[observed], rtol=1e-9)
    assert np.isfinite(frame["common"].to_numpy()).all()
    np.testing.assert_array_equal(np.isnan(frame["idiosyncratic"].to_numpy()), ~observed)
    # common is the decoder's output on the factors, in data units.
    with torch.no_grad():
        common = _decoder64(model)(torch.tensor(model.factors_.to_numpy(), dtype=torch.float64)).numpy()
    np.testing.assert_allclose(frame["common"].to_numpy(), model.scaler_.inverse_transform(common), rtol=1e-12)


def test_decompose_by_factor(case):
    model, panel = case
    if model.decoder_type == "mlp":
        with pytest.raises(UnsupportedInputError, match="by='factor'"):
            model.decompose(by="factor")
        return
    frame = model.decompose(by="factor")
    parts = [*model.factor_names_, "mean", "idiosyncratic"]
    assert list(frame.columns.names) == ["part", "series"]
    assert list(frame.columns) == [(p, s) for p in parts for s in SERIES]
    assert frame.index.equals(model.index_)
    observed = _observed(panel)
    total = sum(frame[part] for part in parts).to_numpy()
    np.testing.assert_allclose(total[observed], panel.y.to_numpy()[observed], rtol=1e-9)
    # In data units a factor's part is scale * H_ij * f_jt and "mean" is the scaler mean + scale * bias.
    scaler, layer = model.scaler_, model.network_.decoder.get_last_linear_layer()
    H = model.loadings_.to_numpy()
    bias = layer.bias.detach().double().numpy()
    Z = model.factors_.to_numpy()
    for j, name in enumerate(model.factor_names_):
        np.testing.assert_allclose(frame[name].to_numpy(), Z[:, [j]] * (scaler.scale_ * H[:, j]), rtol=1e-12)
    mean = frame["mean"].to_numpy()
    np.testing.assert_allclose(mean, np.tile(scaler.mean_ + scaler.scale_ * bias, (T, 1)), rtol=1e-12)
    component = model.decompose()
    common = sum(frame[part] for part in parts[:-1]).to_numpy()
    np.testing.assert_allclose(common, component["common"].to_numpy(), rtol=1e-9)
    assert frame["idiosyncratic"].equals(component["idiosyncratic"])


@pytest.mark.parametrize("scaler", [None, "robust", "minmax", "maxabs"])
def test_decompose_by_factor_with_each_affine_scaler(scaler):
    model = _fit(scaler=scaler)
    panel = _panel(0)
    frame = model.decompose(by="factor")
    observed = _observed(panel)
    total = sum(frame[part] for part in frame.columns.get_level_values("part").unique()).to_numpy()
    np.testing.assert_allclose(total[observed], panel.y.to_numpy()[observed], rtol=1e-9)
    common = (frame.drop(columns="idiosyncratic", level="part").T.groupby(level="series").sum().T)[SERIES]
    np.testing.assert_allclose(common.to_numpy(), model.decompose()["common"].to_numpy(), rtol=1e-9)


def test_decompose_by_factor_needs_an_affine_scaler():
    model = _fit(scaler=QuantileTransformer(n_quantiles=T))
    panel = _panel(0)
    with pytest.raises(UnsupportedInputError, match="affine scaler"):
        model.decompose(by="factor")
    frame = model.decompose()
    observed = _observed(panel)
    total = (frame["common"] + frame["idiosyncratic"]).to_numpy()
    np.testing.assert_allclose(total[observed], panel.y.to_numpy()[observed], rtol=1e-9)


def test_decompose_rejects_an_unknown_by(case):
    model, _ = case
    with pytest.raises(ConfigurationError, match="by must be"):
        model.decompose(by="series")


def test_decompose_before_fit():
    with pytest.raises(NotFittedError):
        DDFM().decompose()


# -- output, update, save / load -------------------------------------------------


def test_set_output_polars(case):
    pl = pytest.importorskip("polars")
    model, _ = case
    model = copy.deepcopy(model).set_output(output="polars")
    frame = model.decompose()
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["date", *[f"{c}__{s}" for c in ("common", "idiosyncratic") for s in SERIES]]
    expected = model.set_output(output="pandas").decompose()
    np.testing.assert_array_equal(frame.drop("date").to_numpy(), expected.to_numpy())
    model.set_output(output="polars")
    if model.loadings_kind_ == "exact":
        by_factor = model.decompose(by="factor")
        assert isinstance(by_factor, pl.DataFrame) and by_factor.columns[:2] == ["date", "factor_0__s0"]
    forecast, factors = model.predict(2), model.predict_factors(2)
    assert isinstance(forecast, pl.DataFrame) and forecast.columns == ["date", *SERIES]
    assert isinstance(factors, pl.DataFrame) and factors.columns == ["date", "factor_0", "factor_1"]
    # Fitted attributes stay pandas.
    assert isinstance(model.factors_, pd.DataFrame) and isinstance(model.loadings_, pd.DataFrame)


def test_update_leaves_the_interpretation_unchanged(case):
    model, panel = case
    model = copy.deepcopy(model)
    before = {name: getattr(model, name).copy() for name in ("factors_", "loadings_", "explained_variance_")}
    decomposition = model.decompose()
    full = _panel(len(panel.exog_names), n_time=T + 6)
    new = PanelData(y=full.y.iloc[T:], exog=None if full.exog is None else full.exog.iloc[T:])
    model.update(new)
    for name, value in before.items():
        assert getattr(model, name).equals(value), name
    assert model.decompose().equals(decomposition)


def test_save_load_round_trip(case, tmp_path):
    model, _ = case
    model.save(tmp_path / "run")
    loaded = DDFM.load(tmp_path / "run")
    np.testing.assert_array_equal(loaded.factor_names_, model.factor_names_, strict=True)
    assert loaded.loadings_kind_ == model.loadings_kind_
    pd.testing.assert_frame_equal(loaded.factors_, model.factors_, check_exact=True, check_freq=True)
    pd.testing.assert_frame_equal(loaded.loadings_, model.loadings_, check_exact=True)
    pd.testing.assert_series_equal(loaded.explained_variance_, model.explained_variance_, check_exact=True)
    pd.testing.assert_frame_equal(loaded.decompose(), model.decompose(), check_exact=True)
    if model.loadings_kind_ == "exact":
        pd.testing.assert_frame_equal(loaded.decompose(by="factor"), model.decompose(by="factor"), check_exact=True)

"""The interpretation API of iVDFM: factors_, window_factors_, loadings_, explained_variance_ and decompose()."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from dfm_python import PanelData, iVDFM
from dfm_python.io.codecs import read_npz, write_npz
from dfm_python.utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError

TINY = {
    "n_factors": 2, "window": 8, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
    "batch_size": 4, "max_epochs": 2, "random_state": 0, "device": "cpu",
}


def _panel(T: int = 40, N: int = 3, K: int = 0, *, index: str = "datetime", missing: bool = False) -> PanelData:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date") if index == "datetime" else pd.RangeIndex(T)
    y = pd.DataFrame(rng.standard_normal((T, N)) + 2.0, index=idx, columns=[f"s{i}" for i in range(N)])
    if missing:
        y.iloc[[3, 10, 25], [0, 2, 1]] = np.nan
    exog = pd.DataFrame(rng.standard_normal((T, K)), index=idx, columns=[f"u{k}" for k in range(K)]) if K else None
    return PanelData(y=y, exog=exog)


CASES = {
    "scaler_missing": ({"scaler": "standard"}, {"missing": True}),
    "exog_context": ({"time_context": 2}, {"K": 2}),
    "regimes": ({"n_regimes": 2, "factor_order": 2, "mixing": True}, {"index": "range"}),
}


@pytest.fixture(scope="module", params=list(CASES))
def case(request) -> tuple[iVDFM, PanelData]:
    params, data = CASES[request.param]
    panel = _panel(**data)
    return iVDFM(**TINY, **params).fit(panel), panel


def _fd_mean_jacobian(engine, path: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite differences of the float64 decoding (as get_result decodes), averaged over the path."""
    engine = copy.deepcopy(engine).double()
    p = torch.as_tensor(path, dtype=torch.float64).unsqueeze(0)
    columns = []
    with torch.no_grad():
        for j in range(p.shape[-1]):
            step = torch.zeros_like(p)
            step[..., j] = h
            diff = (engine._decode(p + step, u_1T=None) - engine._decode(p - step, u_1T=None)) / (2 * h)
            columns.append(diff[0].numpy().mean(axis=0))
    return np.stack(columns, axis=1)


def test_tags():
    tags = iVDFM.get_tags()
    assert tags["capability:interpret"] is True and tags["capability:pred_int"] is True


def test_factors_is_the_engine_factor_path(case):
    est, panel = case
    assert list(est.factor_names_) == ["factor_0", "factor_1"] and est.factor_names_.dtype == object
    frame = est.factors_
    assert frame.index.equals(panel.index) and frame.index.name == panel.index.name
    assert list(frame.columns) == list(est.factor_names_) and all(d == np.float64 for d in frame.dtypes)
    np.testing.assert_array_equal(frame.to_numpy(), est.network_.factor_path.astype(np.float64), strict=True)
    np.testing.assert_array_equal(frame.to_numpy(), est.result_.Z.astype(np.float64))


def test_window_factors_keep_the_window_layout(case):
    est, panel = case
    frame = est.window_factors_
    n_windows, window, r = est.network_.factors.shape
    assert frame.shape == (n_windows * window, r) and list(frame.columns) == list(est.factor_names_)
    assert frame.index.names == ["window", panel.index.name or "time"]
    assert all(d == np.float64 for d in frame.dtypes)
    for w in (0, n_windows - 1):
        assert frame.loc[w].index.equals(panel.index[w * est.stride_: w * est.stride_ + window])
    rows = frame.to_numpy().reshape(n_windows, window, r)
    engine_rows = {est.network_.factors[k].astype(np.float64).tobytes() for k in range(n_windows)}
    assert {rows[w].tobytes() for w in range(n_windows)} == engine_rows


def test_decompose_adds_up_to_the_data(case):
    est, panel = case
    parts = est.decompose()
    series = list(est.feature_names_in_)
    assert parts.index.equals(est.index_)
    assert parts.columns.names == ["component", "series"]
    assert list(parts.columns) == [(c, s) for c in ("common", "idiosyncratic") for s in series]
    assert all(d == np.float64 for d in parts.dtypes)
    data = panel.y.to_numpy()
    observed = ~np.isnan(data)
    common, idio = parts["common"].to_numpy(), parts["idiosyncratic"].to_numpy()
    np.testing.assert_allclose((common + idio)[observed], data[observed], rtol=1e-9, atol=0)
    assert np.isnan(idio[~observed]).all() and np.isfinite(common).all()
    expected = est.result_.x_sm.astype(np.float64)
    if est.scaler_ is not None:
        expected = est.scaler_.inverse_transform(expected)
    np.testing.assert_array_equal(common, expected)


def test_loadings_are_the_mean_decoder_jacobian(case):
    est, _ = case
    loadings = est.loadings_
    assert list(loadings.index) == list(est.feature_names_in_) and list(loadings.columns) == list(est.factor_names_)
    assert all(d == np.float64 for d in loadings.dtypes) and est.loadings_kind_ == "mean_jacobian"
    expected = _fd_mean_jacobian(est.network_, est.result_.Z)
    np.testing.assert_allclose(loadings.to_numpy(), expected, rtol=1e-4, atol=1e-6)


def test_linear_decoder_loadings_are_the_weights():
    """One regime: the loadings are exact (test_intervals_deep covers them and decompose(by='factor'))."""
    est = iVDFM(**TINY, decoder_type="linear", mixing=True).fit(_panel())
    engine = est.network_
    weight = (engine.decoder.linear.weight.double() @ engine.factor_mixing.M.weight.double()).detach().numpy()
    assert est.loadings_kind_ == "exact"
    np.testing.assert_allclose(est.loadings_.to_numpy(), weight, rtol=1e-12, atol=0)
    np.testing.assert_allclose(est.loadings_.to_numpy(), _fd_mean_jacobian(engine, est.result_.Z), rtol=1e-4, atol=1e-6)


def test_explained_variance_matches_a_direct_computation(case):
    est, panel = case
    data = panel.y.to_numpy()
    scaled = data if est.scaler_ is None else est.scaler_.transform(data)
    residual = scaled - est.result_.x_sm.astype(np.float64)
    expected = []
    for i in range(data.shape[1]):
        observed = ~np.isnan(data[:, i])
        expected.append(1.0 - np.var(residual[observed, i]) / np.var(scaled[observed, i]))
    ev = est.explained_variance_
    assert isinstance(ev, pd.Series) and ev.name == "explained_variance" and ev.dtype == np.float64
    assert list(ev.index) == list(est.feature_names_in_)
    np.testing.assert_allclose(ev.to_numpy(), expected, rtol=1e-12)


def test_non_finite_targets_count_as_missing():
    """Inf entries count as missing in the explained variance (fit does not warn)."""
    y = _panel().y
    y.iloc[[4, 9], 2] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        est = iVDFM(**TINY, scaler="standard").fit(PanelData(y=y))
    ev = est.explained_variance_.to_numpy()
    assert np.isfinite(ev).all()
    observed = np.delete(np.arange(40), [4, 9])
    scaled = est.scaler_.transform(y.to_numpy()[observed])[:, 2]
    residual = scaled - est.result_.x_sm[observed, 2].astype(np.float64)
    assert ev[2] == pytest.approx(1.0 - np.var(residual) / np.var(scaled), rel=1e-12)
    assert np.isfinite(est.decompose()["common"].to_numpy()).all()


def test_decompose_by():
    with pytest.raises(NotFittedError, match="decompose"):
        iVDFM(**TINY).decompose()
    est = iVDFM(**TINY).fit(_panel())
    with pytest.raises(UnsupportedInputError, match="nonlinear"):
        est.decompose(by="factor")
    with pytest.raises(ConfigurationError, match="by must be"):
        est.decompose(by="series")
    # A failed refit removes the training targets with the fitted attributes.
    with pytest.raises(ConfigurationError):
        est.set_params(n_factors=0).fit(_panel())
    assert "_targets_" not in vars(est) and not hasattr(est, "factors_")


def test_update_leaves_the_interpretation_unchanged():
    est = iVDFM(**TINY, scaler="standard").fit(_panel())
    before = {name: copy.deepcopy(getattr(est, name)) for name in
              ("factors_", "window_factors_", "loadings_", "explained_variance_")}
    parts = est.decompose()
    est.update(_panel(T=52))
    for name, value in before.items():
        assert getattr(est, name).equals(value), name
    assert est.decompose().equals(parts)


def test_set_output_polars(case):
    pl = pytest.importorskip("polars")
    est, _ = copy.deepcopy(case)
    futr = {"futr_exog": np.zeros((3, len(est.exog_names_in_)))} if len(est.exog_names_in_) else {}
    frames = (est.decompose(), est.predict(3, **futr), est.predict_factors(3, **futr))
    assert est.set_output(output="polars") is est
    parts, forecast, factors = est.decompose(), est.predict(3, **futr), est.predict_factors(3, **futr)
    time = est.index_.name or "time"
    assert parts.columns == [time, *(f"{c}__{s}" for c in ("common", "idiosyncratic") for s in est.feature_names_in_)]
    assert factors.columns == [time, *est.factor_names_]
    for frame, pandas_frame in zip((parts, forecast, factors), frames):
        assert isinstance(frame, pl.DataFrame)
        np.testing.assert_array_equal(frame.drop(time).to_numpy(), pandas_frame.to_numpy())
    assert isinstance(est.factors_, pd.DataFrame) and isinstance(est.set_output().predict(3, **futr), pd.DataFrame)


def test_save_load_restores_the_interpretation(case, tmp_path):
    est, _ = case
    est.save(tmp_path / "run")
    loaded = iVDFM.load(tmp_path / "run", device="cpu")
    for name in ("factors_", "window_factors_", "loadings_", "explained_variance_"):
        a, b = getattr(loaded, name), getattr(est, name)
        assert a.equals(b) and a.index.names == b.index.names and a.index.equals(b.index), name
    assert list(loaded.factor_names_) == list(est.factor_names_) and loaded.loadings_kind_ == "mean_jacobian"
    pd.testing.assert_frame_equal(loaded.decompose(), est.decompose())


def _rewrite(path, **arrays) -> None:
    """Replace (an array) or drop (None) entries of an npz archive."""
    content = read_npz(path)
    for key, value in arrays.items():
        content.pop(key, None)
        if value is not None:
            content[key] = value
    write_npz(path, content)


@pytest.mark.parametrize("release", ["0.8.3", "0.8.2"])
def test_runs_saved_by_0_8_load(release, tmp_path):
    est = iVDFM(**TINY, scaler="standard").fit(_panel())
    est.save(tmp_path / "run")
    drop = {"fit/targets": None, "fit/explained_variance": None}
    if release == "0.8.2":
        # No factor path: result_.Z is the window average, as get_result computed it then.
        drop["factor_path"] = None
        average = est.network_.factors.mean(axis=0)
        _rewrite(tmp_path / "run" / "result.npz", Z=average)
    _rewrite(tmp_path / "run" / "state.npz", **drop)

    loaded = iVDFM.load(tmp_path / "run", device="cpu")
    assert loaded.window_factors_.equals(est.window_factors_)
    if release == "0.8.3":
        assert loaded.factors_.equals(est.factors_) and loaded.loadings_.equals(est.loadings_)
        pd.testing.assert_frame_equal(loaded.predict(3), est.predict(3))
        pd.testing.assert_frame_equal(loaded.predict_interval(3), est.predict_interval(3))
    else:
        assert np.isfinite(loaded.predict_interval(3).to_numpy()).all()
        assert loaded.factors_.index.equals(est.index_[-est.window_:])
        np.testing.assert_array_equal(loaded.factors_.to_numpy(), average.astype(np.float64))
        np.testing.assert_allclose(
            loaded.loadings_.to_numpy(), _fd_mean_jacobian(loaded.network_, average), rtol=1e-4, atol=1e-6
        )
        assert np.isfinite(loaded.predict(3).to_numpy()).all()
    assert not hasattr(loaded, "explained_variance_")
    with pytest.raises(UnsupportedInputError, match="training targets"):
        loaded.decompose()
    loaded.save(tmp_path / "again")
    assert not hasattr(iVDFM.load(tmp_path / "again"), "explained_variance_")

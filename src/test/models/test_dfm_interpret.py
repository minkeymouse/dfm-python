"""DFM interpretation and interval API: factors_, states_, loadings_, explained_variance_, decompose, predict_interval."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from dfm_python import DFM, PanelData
from dfm_python.functional.state_layout import DFMStateLayout
from dfm_python.testing import _state_hash
from dfm_python.utils.errors import ConfigurationError, NotFittedError, UnsupportedInputError

TENT = [1, 2, 3, 2, 1]
USER = ["m0", "gdp", "m1", "m2", "m3", "m4"]
BLOCKS = {
    "global": {"n_factors": 2, "series": USER},
    "real": {"n_factors": 1, "series": ["m0", "m1", "gdp"]},
}
PARAMS = {"blocks": BLOCKS, "tent_weights": {"q:m": TENT}, "max_iter": 10, "scaler": "standard"}
N_FIT = 66


def _frame(T: int = 72, seed: int = 0) -> pd.DataFrame:
    """Five monthly series and a quarterly one (NaN off quarter ends), two AR(1) factors, scattered NaN and a ragged edge."""
    rng = np.random.default_rng(seed)
    f = np.zeros((T, 2))
    for t in range(1, T):
        f[t] = [0.7, 0.4] * f[t - 1] + rng.standard_normal(2)
    y = f @ rng.standard_normal((2, 6)) + 0.5 * rng.standard_normal((T, 6))
    index = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    df = pd.DataFrame(y, index=index, columns=USER)
    df.loc[df.index.month % 3 != 0, "gdp"] = np.nan
    monthly = [name for name in USER if name != "gdp"]
    mask = rng.random((T, len(monthly))) < 0.05
    df[monthly] = df[monthly].mask(mask)
    df.iloc[-2:, USER.index("m4")] = np.nan
    return df


def _panel(df: pd.DataFrame) -> PanelData:
    return PanelData(y=df, freq={"gdp": "q"})


@pytest.fixture(scope="module")
def fitted():
    df = _frame()
    train, new = _panel(df.iloc[:N_FIT]), _panel(df.iloc[N_FIT:])
    return DFM(**PARAMS).fit(train), df, new


def _layout(model: DFM) -> DFMStateLayout:
    return DFMStateLayout.from_state(model.engine_.training_state)


def _current(model: DFM) -> np.ndarray:
    """The states of the current factor values, blocks in order."""
    layout = _layout(model)
    return np.concatenate([layout.factor_current(b) for b in range(len(layout.r))])


def _internal(model: DFM, frame: pd.DataFrame) -> np.ndarray:
    return frame[list(model.internal_feature_names_)].to_numpy(np.float64)


def _scaled(model: DFM, frame: pd.DataFrame) -> np.ndarray:
    """``frame`` in the internal order, standardized with the fitted scaler's statistics."""
    return (_internal(model, frame) - model.scaler_.mean_) / model.scaler_.scale_


def _user(model: DFM, values: np.ndarray) -> np.ndarray:
    """Columns of an internal-order array in the user's order."""
    internal = list(model.internal_feature_names_)
    return values[:, [internal.index(name) for name in model.feature_names_in_]]


def _filtered_cov(model: DFM, Y: np.ndarray) -> np.ndarray:
    """``Var[x_T | y_1..y_T]`` by a plain Kalman filter over ``Y`` (NaN missing) with the fitted parameters."""
    engine = model.engine_
    A, C, Q, R = engine.A, engine.C, engine.Q, engine.R
    x, P = engine.Z_0.copy(), engine.V_0.copy()
    for t, y in enumerate(Y):
        if t:
            x, P = A @ x, A @ P @ A.T + Q
        o = ~np.isnan(y)
        if o.any():
            Co = C[o]
            K = np.linalg.solve(Co @ P @ Co.T + R[np.ix_(o, o)], Co @ P).T
            x, P = x + K @ (y[o] - Co @ x), P - K @ Co @ P
    return P


def _half_widths(model: DFM, Y: np.ndarray, h: int, coverage: float) -> np.ndarray:
    """Half widths (h x N, user order) from the forecast moments propagated from the filter over ``Y``."""
    engine = model.engine_
    A, C, Q, R = engine.A, engine.C, engine.Q, engine.R
    P = _filtered_cov(model, Y)
    sd = []
    for _ in range(h):
        P = A @ P @ A.T + Q
        sd.append(np.sqrt(np.diag(C @ P @ C.T + R)))
    return _user(model, norm.ppf(0.5 + coverage / 2) * np.array(sd) * model.scaler_.scale_)


def _bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (frame.xs("lower", axis=1, level="bound").to_numpy(), frame.xs("upper", axis=1, level="bound").to_numpy())


def _interpretation(model: DFM) -> dict:
    return {name: getattr(model, name) for name in
            ("factor_names_", "factors_", "states_", "loadings_", "loadings_kind_", "explained_variance_")}


def _assert_same_interpretation(got: DFM, expected: DFM) -> None:
    a, b = _interpretation(got), _interpretation(expected)
    np.testing.assert_array_equal(a.pop("factor_names_"), b.pop("factor_names_"), strict=True)
    assert a.pop("loadings_kind_") == b.pop("loadings_kind_") == "exact"
    for name in a:
        assert a[name].equals(b[name]), name
        assert a[name].index.equals(b[name].index), name


# -- attributes -----------------------------------------------------------------------


def test_tags():
    tags = DFM.get_tags()
    assert tags["capability:interpret"] is True and tags["capability:pred_int"] is True


@pytest.mark.parametrize("params, names", [
    ({"blocks": None}, ["factor_0"]),
    ({"blocks": None, "n_factors": 2}, ["factor_0", "factor_1"]),
    ({"blocks": {"only": {"n_factors": 2, "series": USER}}}, ["factor_0", "factor_1"]),
    ({"blocks": BLOCKS}, ["global_0", "global_1", "real_0"]),
    ({"blocks": {"real": BLOCKS["real"], "global": BLOCKS["global"]}}, ["real_0", "global_0", "global_1"]),
])
def test_factor_names(params, names):
    model = DFM(**{**PARAMS, "max_iter": 2, **params}).fit(_panel(_frame()))
    assert model.factor_names_.dtype == object and list(model.factor_names_) == names
    assert list(model.factors_.columns) == names and list(model.loadings_.columns) == names


def test_factors_are_the_current_factor_states(fitted):
    model, df, _ = fitted
    layout = _layout(model)
    Z = model.result_.Z
    assert list(model.states_.columns) == [f"state_{i}" for i in range(Z.shape[1])]
    np.testing.assert_array_equal(model.states_.to_numpy(), Z, strict=True)
    np.testing.assert_array_equal(model.factors_.to_numpy(), Z[:, _current(model)], strict=True)
    for frame in (model.factors_, model.states_):
        assert frame.index.equals(model.index_) and all(dtype == np.float64 for dtype in frame.dtypes)
    # The companion form stacks a block's factors lag by lag: state offset + r_b + j is lag 1 of factor j.
    A = model.result_.A
    for b, r_b in enumerate(layout.r):
        offset = int(layout.factor_offsets[b])
        for j in range(r_b):
            assert A[offset + r_b + j, offset + j] == 1.0


def test_loadings(fitted):
    model, _, _ = fitted
    layout = _layout(model)
    C = model.result_.C
    internal = list(model.internal_feature_names_)
    loadings = model.loadings_
    assert list(loadings.index) == USER and all(dtype == np.float64 for dtype in loadings.dtypes)
    assert model.loadings_kind_ == "exact"
    offsets, r = layout.factor_offsets, layout.r
    factors = [(b, j) for b in range(len(r)) for j in range(r[b])]
    for name in USER:
        i = internal.index(name)
        for col, (b, j) in zip(model.factor_names_, factors):
            lag0 = C[i, offsets[b] + j]
            if name == "gdp":
                # The loading on the tent aggregate: lag l of the factor carries TENT[l] times it.
                assert loadings.loc[name, col] == lag0 / TENT[0]
                lags = C[i, offsets[b] + j + r[b] * np.arange(len(TENT))]
                np.testing.assert_allclose(lags, np.array(TENT) * loadings.loc[name, col], rtol=1e-10, atol=1e-14)
            else:
                assert loadings.loc[name, col] == lag0
    # Series outside the block "real" do not load on it.
    assert (loadings.loc[["m2", "m3", "m4"], "real_0"] == 0).all()
    assert (loadings.loc[["m0", "m1", "gdp"], "real_0"] != 0).all()


def test_slower_loadings_divide_by_the_first_tent_weight():
    tent = [2, 4, 6, 4, 2]
    model = DFM(**{**PARAMS, "tent_weights": {"q:m": tent}, "max_iter": 2}).fit(_panel(_frame()))
    i = list(model.internal_feature_names_).index("gdp")
    lag0 = model.result_.C[i, _current(model)]
    np.testing.assert_array_equal(model.loadings_.loc["gdp"].to_numpy(), lag0 / tent[0], strict=True)
    assert (lag0 != 0).all()


def test_explained_variance(fitted):
    model, df, _ = fitted
    x = _scaled(model, df.iloc[:N_FIT])
    F = int(sum(_layout(model).r)) * _layout(model).L
    common = model.result_.Z[:, :F] @ model.result_.C[:, :F].T
    expected = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        o = ~np.isnan(x[:, i])
        expected[i] = 1 - np.var(x[o, i] - common[o, i]) / np.var(x[o, i])
    explained = model.explained_variance_
    assert explained.name == "explained_variance" and list(explained.index) == USER
    np.testing.assert_allclose(explained.to_numpy(), _user(model, expected[None])[0], rtol=1e-12)
    assert ((explained > 0) & (explained < 1)).all()


# -- decompose ------------------------------------------------------------------------


def _assert_adds_up(frame: pd.DataFrame, data: pd.DataFrame) -> None:
    total = frame.T.groupby(level="series", sort=False).sum().T[data.columns]
    observed = data.notna().to_numpy()
    np.testing.assert_allclose(total.to_numpy()[observed], data.to_numpy()[observed], rtol=1e-9, atol=1e-12)


def test_decompose_by_component(fitted):
    model, df, _ = fitted
    data = df.iloc[:N_FIT]
    parts = model.decompose()
    assert parts.columns.names == ["component", "series"]
    assert list(parts.columns) == [(c, s) for c in ("common", "idiosyncratic") for s in USER]
    assert parts.index.equals(model.index_) and all(dtype == np.float64 for dtype in parts.dtypes)
    common, idio = parts["common"], parts["idiosyncratic"]
    np.testing.assert_array_equal(idio.isna().to_numpy(), data.isna().to_numpy())
    assert np.isfinite(common.to_numpy()).all()
    observed = data.notna().to_numpy()
    np.testing.assert_allclose((common + idio).to_numpy()[observed], data.to_numpy()[observed], rtol=1e-9)
    # common: the factor states through C, in data units (the series mean included).
    F = int(sum(_layout(model).r)) * _layout(model).L
    scaled = model.result_.Z[:, :F] @ model.result_.C[:, :F].T
    expected = _user(model, scaled * model.scaler_.scale_ + model.scaler_.mean_)
    np.testing.assert_allclose(common.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_decompose_by_factor(fitted):
    model, df, _ = fitted
    data = df.iloc[:N_FIT]
    parts = model.decompose(by="factor")
    names = list(model.factor_names_) + ["mean", "idiosyncratic"]
    assert parts.columns.names == ["part", "series"]
    assert list(parts.columns) == [(p, s) for p in names for s in USER]
    _assert_adds_up(parts, data)
    np.testing.assert_array_equal(parts["idiosyncratic"].isna().to_numpy(), data.isna().to_numpy())
    mean = _user(model, model.scaler_.mean_[None])[0]
    np.testing.assert_array_equal(parts["mean"].to_numpy(), np.broadcast_to(mean, data.shape))
    # The factor parts and the mean add up to the common component.
    common = sum(parts[name] for name in model.factor_names_) + parts["mean"]
    np.testing.assert_allclose(common.to_numpy(), model.decompose()["common"].to_numpy(), rtol=1e-12, atol=1e-12)
    # A factor's part: lag 0 for a clock series, its tent lags for the slower one.
    layout, Z, C = _layout(model), model.result_.Z, model.result_.C
    internal = list(model.internal_feature_names_)
    b, j = 0, 1  # global_1
    offset, r_b = int(layout.factor_offsets[b]), layout.r[b]
    i = internal.index("m2")
    np.testing.assert_allclose(parts[("global_1", "m2")].to_numpy(),
                               C[i, offset + j] * Z[:, offset + j] * model.scaler_.scale_[i], rtol=1e-12)
    i = internal.index("gdp")
    lags = offset + j + r_b * np.arange(len(TENT))
    np.testing.assert_allclose(parts[("global_1", "gdp")].to_numpy(),
                               Z[:, lags] @ C[i, lags] * model.scaler_.scale_[i], rtol=1e-12)


@pytest.mark.parametrize("by", ["series", "Component", None])
def test_decompose_rejects_an_unknown_by(fitted, by):
    with pytest.raises(ConfigurationError, match="by must be"):
        fitted[0].decompose(by=by)


def test_decompose_before_fit_raises():
    with pytest.raises(NotFittedError):
        DFM(**PARAMS).decompose()


def test_decompose_with_a_nonlinear_scaler():
    df = _frame()
    model = DFM(**{**PARAMS, "max_iter": 2, "scaler": QuantileTransformer(n_quantiles=20)}).fit(_panel(df))
    parts = model.decompose()
    observed = df.notna().to_numpy()
    np.testing.assert_allclose((parts["common"] + parts["idiosyncratic"]).to_numpy()[observed],
                               df.to_numpy()[observed], rtol=1e-9, atol=1e-12)
    with pytest.raises(UnsupportedInputError, match="QuantileTransformer"):
        model.decompose(by="factor")


# -- predict_interval -------------------------------------------------------------------


def test_predict_interval_frame(fitted):
    model, _, _ = fitted
    point = model.predict(4)
    frame = model.predict_interval(4, coverage=0.8)
    assert frame.index.equals(point.index) and frame.columns.names == ["series", "bound"]
    assert list(frame.columns) == [(s, b) for s in USER for b in ("lower", "upper")]
    assert frame.attrs == {**point.attrs, "coverage": 0.8}
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    lower, upper = _bounds(frame)
    assert (lower < point.to_numpy()).all() and (point.to_numpy() < upper).all()
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-14, atol=1e-14)
    # As the coverage goes to 0 the interval collapses onto the point forecast, bitwise.
    lower, upper = _bounds(model.predict_interval(4, coverage=1e-300))
    assert lower.tobytes() == point.to_numpy().tobytes() and upper.tobytes() == point.to_numpy().tobytes()
    # Wider with the coverage and with the horizon.
    wide = _bounds(model.predict_interval(4, coverage=0.95))
    narrow = _bounds(frame)
    assert ((wide[1] - wide[0]) > (narrow[1] - narrow[0])).all()
    assert (np.diff(narrow[1] - narrow[0], axis=0) > 0).all()
    # The largest float below 1 (where (1 + coverage) / 2 rounds to 1): z = 8.29.
    top = np.nextafter(1.0, 0.0)
    widest = _bounds(model.predict_interval(4, coverage=top))
    np.testing.assert_allclose((widest[1] - widest[0]) / (narrow[1] - narrow[0]), norm.isf((1 - top) / 2) / norm.ppf(0.9),
                               rtol=1e-12)


def test_predict_interval_matches_the_propagated_moments(fitted):
    model, df, _ = fitted
    frame = model.predict_interval(5, coverage=0.9)
    lower, upper = _bounds(frame)
    expected = _half_widths(model, _scaled(model, df.iloc[:N_FIT]), 5, 0.9)
    np.testing.assert_allclose((upper - lower) / 2, expected, rtol=1e-8)


def test_predict_interval_with_data_and_after_update(fitted):
    model, df, new = fitted
    before = _state_hash(model)
    conditioned = model.predict_interval(3, data=new)
    assert _state_hash(model) == before
    point = model.predict(3, data=new)
    assert conditioned.index.equals(point.index) and conditioned.attrs == {**point.attrs, "coverage": 0.9}
    lower, upper = _bounds(conditioned)
    np.testing.assert_allclose((lower + upper) / 2, point.to_numpy(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose((upper - lower) / 2, _half_widths(model, _scaled(model, df), 3, 0.9), rtol=1e-8)
    zero = _bounds(model.predict_interval(3, coverage=1e-300, data=new))[0]
    assert zero.tobytes() == point.to_numpy().tobytes()

    updated = copy.deepcopy(model).update(new)
    after = updated.predict_interval(3)
    assert after.index.equals(conditioned.index) and after.attrs == conditioned.attrs
    np.testing.assert_allclose(after.to_numpy(), conditioned.to_numpy(), rtol=1e-12, atol=1e-12)
    # More rows after the update: the filter continues over them as well.
    more = _panel(_frame(T=75).iloc[72:])
    np.testing.assert_allclose(
        updated.predict_interval(2, data=more).to_numpy(),
        copy.deepcopy(updated).update(more).predict_interval(2).to_numpy(), rtol=1e-12, atol=1e-12,
    )


def _simulate_dfm(T: int, seed: int) -> pd.DataFrame:
    """A known one-factor DFM: f_t = 0.8 f_{t-1} + u_t, x_it = l_i f_t + e_it with AR(1) e_it."""
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, 6) * rng.choice([-1, 1], 6)
    rho = rng.uniform(0.0, 0.5, 6)
    f, e = np.zeros(T), np.zeros((T, 6))
    for t in range(1, T):
        f[t] = 0.8 * f[t - 1] + rng.standard_normal()
        e[t] = rho * e[t - 1] + 0.6 * rng.standard_normal(6)
    index = pd.date_range("2000-01-31", periods=T, freq="ME", name="date")
    return pd.DataFrame(np.outer(f, loadings) + e, index=index, columns=[f"s{i}" for i in range(6)])


def test_predict_interval_coverage_on_a_simulated_dfm():
    """1 to 3 steps ahead from 58 origins (the model updated row by row): about 90% of the values fall inside."""
    df = _simulate_dfm(210, seed=0)
    train, test = df.iloc[:150], df.iloc[150:]
    model = DFM(n_factors=1, max_iter=200, scaler="standard").fit(PanelData(y=train))
    hits = []
    for o in range(len(test) - 2):
        lower, upper = _bounds(model.predict_interval(3, coverage=0.9))
        actual = test.iloc[o:o + 3].to_numpy()
        hits.append((lower <= actual) & (actual <= upper))
        model.update(PanelData(y=test.iloc[o:o + 1]))
    assert 0.8 <= np.mean(hits) <= 0.97


@pytest.mark.parametrize("scaler", [None, "robust", "minmax", "maxabs", StandardScaler(with_std=False)])
def test_affine_scalers_convert_intervals_and_factor_parts(scaler):
    df = _frame()
    model = DFM(**{**PARAMS, "max_iter": 3, "scaler": scaler}).fit(_panel(df))
    lower, upper = _bounds(model.predict_interval(3))
    s = model.scaler_
    X = _internal(model, df)
    Y = X if s is None else s.transform(X)
    zeros, ones = np.zeros((1, X.shape[1])), np.ones((1, X.shape[1]))
    offset = np.zeros(X.shape[1]) if s is None else s.inverse_transform(zeros)[0]
    slope = np.ones(X.shape[1]) if s is None else s.inverse_transform(ones)[0] - offset
    engine = model.engine_
    P = _filtered_cov(model, Y)
    sd = []
    for _ in range(3):
        P = engine.A @ P @ engine.A.T + engine.Q
        sd.append(np.sqrt(np.diag(engine.C @ P @ engine.C.T + engine.R)) * np.abs(slope))
    np.testing.assert_allclose((upper - lower) / 2, _user(model, norm.ppf(0.95) * np.array(sd)), rtol=1e-8)
    np.testing.assert_allclose((lower + upper) / 2, model.predict(3).to_numpy(), rtol=1e-13, atol=1e-13)
    # decompose(by="factor"): a factor's states through C times the slope, and the offset as "mean".
    parts = model.decompose(by="factor")
    _assert_adds_up(parts, df)
    np.testing.assert_allclose(parts["mean"].to_numpy(), np.broadcast_to(_user(model, offset[None])[0], df.shape),
                               rtol=1e-15)
    states = _layout(model).r[0] * np.arange(_layout(model).L)  # global_0 and its lags
    Z, C = model.result_.Z, model.result_.C
    np.testing.assert_allclose(parts["global_0"].to_numpy(), _user(model, Z[:, states] @ C[:, states].T * slope),
                               rtol=1e-12, atol=1e-14)


def test_predict_interval_rejects_bad_input(fitted):
    model = fitted[0]
    for coverage in (0, 1, 1.5, -0.1, float("nan"), "0.9", True, None):
        with pytest.raises(ConfigurationError, match="coverage"):
            model.predict_interval(2, coverage=coverage)
    with pytest.raises(ValueError):
        model.predict_interval(0)
    with pytest.raises(NotFittedError):
        DFM(**PARAMS).predict_interval(2)
    quantile = DFM(**{**PARAMS, "max_iter": 2, "scaler": QuantileTransformer(n_quantiles=20)}).fit(_panel(_frame()))
    with pytest.raises(UnsupportedInputError, match="affine"):
        quantile.predict_interval(2)


# -- update, output type, save/load -------------------------------------------------------


def test_update_keeps_and_refit_refreshes_the_attributes(fitted):
    model, df, new = fitted
    updated = copy.deepcopy(model).update(new)
    _assert_same_interpretation(updated, model)
    assert updated.decompose().equals(model.decompose())

    refitted = copy.deepcopy(model).update(new, refit=True)
    assert refitted.factors_.index.equals(refitted.conditioning_index_) and len(refitted.factors_) == len(df)
    np.testing.assert_array_equal(refitted.states_.to_numpy(), refitted.result_.Z, strict=True)
    np.testing.assert_array_equal(refitted.factors_.to_numpy(), refitted.result_.Z[:, _current(refitted)], strict=True)
    internal = list(refitted.internal_feature_names_)
    assert refitted.loadings_.loc["m3", "global_0"] == refitted.result_.C[internal.index("m3"), 0]
    assert not refitted.explained_variance_.equals(model.explained_variance_)
    parts = refitted.decompose(by="factor")
    assert parts.index.equals(refitted.conditioning_index_)
    _assert_adds_up(parts, df)
    # The intervals use the refitted parameters, filtered over all rows.
    lower, upper = _bounds(refitted.predict_interval(2))
    np.testing.assert_allclose((upper - lower) / 2, _half_widths(refitted, _scaled(refitted, df), 2, 0.9), rtol=1e-8)
    np.testing.assert_allclose((lower + upper) / 2, refitted.predict(2).to_numpy(), rtol=1e-14, atol=1e-14)


def test_set_output_polars(fitted):
    pl = pytest.importorskip("polars")
    model = copy.deepcopy(fitted[0]).set_output(output="polars")
    pandas_model = fitted[0]
    calls = {
        "predict": (lambda m: m.predict(3), USER),
        "predict_factors": (lambda m: m.predict_factors(3), list(pandas_model.states_.columns)),
        "predict_interval": (lambda m: m.predict_interval(3), [f"{s}__{b}" for s in USER for b in ("lower", "upper")]),
        "decompose": (lambda m: m.decompose(), [f"{c}__{s}" for c in ("common", "idiosyncratic") for s in USER]),
        "decompose_factor": (
            lambda m: m.decompose(by="factor"),
            [f"{p}__{s}" for p in [*pandas_model.factor_names_, "mean", "idiosyncratic"] for s in USER],
        ),
    }
    for name, (call, columns) in calls.items():
        out = call(model)
        assert isinstance(out, pl.DataFrame), name
        assert out.columns == ["date", *columns], name
        np.testing.assert_array_equal(out.drop("date").to_numpy(), call(pandas_model).to_numpy(), err_msg=name)
    # The fitted attributes stay pandas.
    assert isinstance(model.factors_, pd.DataFrame) and isinstance(model.explained_variance_, pd.Series)


def _assert_same_outputs(loaded: DFM, model: DFM, new: PanelData | None) -> None:
    _assert_same_interpretation(loaded, model)
    outputs = [lambda m: m.predict_interval(4), lambda m: m.decompose(), lambda m: m.decompose(by="factor")]
    if new is not None:
        outputs.append(lambda m: m.predict_interval(4, data=new, coverage=0.5))
    for call in outputs:
        got, expected = call(loaded), call(model)
        assert got.index.equals(expected.index) and got.columns.equals(expected.columns)
        assert got.to_numpy().tobytes() == expected.to_numpy().tobytes() and got.attrs == expected.attrs


def test_save_load_round_trip(fitted, tmp_path):
    model, df, new = fitted
    model.save(tmp_path / "fitted")
    _assert_same_outputs(DFM.load(tmp_path / "fitted"), model, new)

    first, rest = _panel(df.iloc[N_FIT:N_FIT + 3]), _panel(df.iloc[N_FIT + 3:])
    updated = copy.deepcopy(model).update(first)
    updated.save(tmp_path / "updated")
    _assert_same_outputs(DFM.load(tmp_path / "updated"), updated, rest)

    refitted = copy.deepcopy(updated).update(rest, refit=True)
    refitted.save(tmp_path / "refitted")
    loaded = DFM.load(tmp_path / "refitted")
    _assert_same_outputs(loaded, refitted, None)
    assert len(loaded.factors_) == len(df)

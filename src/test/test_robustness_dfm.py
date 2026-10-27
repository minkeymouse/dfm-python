"""Input robustness of DFM: the default scaler, degenerate and infinite inputs, and the narrowed exception handlers."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import dfm_python.functional.em as em_module
import dfm_python.models.dfm.initialization as dfm_init
from dfm_python import DFM, PanelData
from dfm_python.config import DFMConfig
from dfm_python.config.schema.results import DFMResult
from dfm_python.numeric.validator import validate_dfm_initialization
from dfm_python.utils.errors import DataValidationError

_T = 36
_TINY = {
    DFM: {"max_iter": 3},
}
_CONFIGS = {DFM: DFMConfig}


def _frame(seed: int = 0) -> pd.DataFrame:
    index = pd.date_range("2001-01-31", periods=_T, freq="ME")
    values = np.random.default_rng(seed).standard_normal((_T, 3))
    return pd.DataFrame(values, index=index, columns=["a", "b", "c"])


def _no_training(monkeypatch, cls) -> None:
    def fail(self, *args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(cls, "_fit", fail)


@pytest.mark.parametrize("cls", [DFM], ids=lambda c: c.__name__)
def test_default_scaler_is_standard(cls):
    assert cls().scaler == "standard"
    assert _CONFIGS[cls]().scaler == "standard" and _CONFIGS[cls].from_dict({}).scaler == "standard"
    assert cls().get_config() == _CONFIGS[cls]()


@pytest.mark.parametrize("cls", [DFM], ids=lambda c: c.__name__)
def test_default_scaler_standardizes_the_targets(cls):
    df = _frame() * 500.0 + 1000.0
    model = cls(**_TINY[cls]).fit(df)
    assert isinstance(model.scaler_, StandardScaler)
    np.testing.assert_allclose(model.scaler_.mean_, df.mean().to_numpy())
    assert np.isfinite(model.predict(2).to_numpy()).all()


def test_dfm_scaler_none_needs_scaled_data():
    df = _frame() * 500.0 + 1000.0
    with pytest.raises(DataValidationError, match="appears unscaled"):
        DFM(max_iter=3, scaler=None).fit(df)
    assert DFM(max_iter=3, scaler=None).fit(_frame()).scaler_ is None


def _degenerate(case: str) -> pd.DataFrame:
    df = _frame()
    if case == "no observed value":
        df["b"] = np.nan
    elif case == "a single observed value":
        df["b"] = np.nan
        df.iloc[4, 1] = 0.5
    elif case == "constant":
        df["b"] = 2.5
    elif case == "constant with gaps":
        df["b"] = 2.5
        df.iloc[::3, 1] = np.nan
    elif case == "constant up to rounding":
        # 0.1 + 0.2 is 0.30000000000000004: StandardScaler leaves this series unscaled, and DFM used to
        # reject it as "unscaled".
        df["b"] = np.where(np.arange(_T) % 2 == 0, 0.1 + 0.2, 0.3)
    return df


_MESSAGES = {
    "no observed value": "'b' has no observed value",
    "a single observed value": "'b' has a single observed value",
    "constant": r"'b' is constant \(every observed value is 2.5, up to rounding\)",
    "constant with gaps": r"'b' is constant \(every observed value is 2.5, up to rounding\)",
    "constant up to rounding": r"'b' is constant \(every observed value is 0.30000000000000004, up to rounding\)",
}


@pytest.mark.parametrize("case", list(_MESSAGES))
@pytest.mark.parametrize("cls", [DFM], ids=lambda c: c.__name__)
def test_degenerate_target_is_rejected_before_training(monkeypatch, cls, case):
    _no_training(monkeypatch, cls)
    model = cls(**_TINY[cls])
    with pytest.raises(DataValidationError, match=_MESSAGES[case]) as info:
        model.fit(_degenerate(case))
    assert f"{cls.__name__} cannot be fitted" in str(info.value) and "drop these series" in str(info.value)
    assert not hasattr(model, "result_")


def test_every_degenerate_series_is_named():
    df = _frame()
    df["a"] = 1.0
    df["c"] = np.nan
    with pytest.raises(DataValidationError, match=r"'a' is constant .*; 'c' has no observed value"):
        DFM(max_iter=3).fit(df)


def test_two_distinct_observed_values_are_enough():
    df = _frame()
    df["b"] = np.nan
    df.iloc[[3, 9], 1] = [0.0, 1.0]
    DFM()._check_training_targets(PanelData(y=df))


def test_a_small_variation_above_rounding_is_not_constant():
    df = _frame()
    df["b"] = 2.5 + 1e-9 * df["b"]
    model = DFM(max_iter=2).fit(df)
    np.testing.assert_allclose(np.std(model.scaler_.transform(df.to_numpy())[:, 1]), 1.0)


def _quarterly(df: pd.DataFrame) -> PanelData:
    return PanelData(y=df, freq={"c": "q"})


def test_mixed_frequency_series_are_judged_on_their_observations():
    df = _frame()
    quarter_end = df.index.month % 3 == 0
    df.loc[~quarter_end, "c"] = np.nan
    model = DFM(tent_weights={"q:m": [1, 2, 3, 2, 1]}, max_iter=2).fit(_quarterly(df))
    assert model.freq_["c"] == "q"

    single = df.copy()
    single.loc[single.index[quarter_end][1:], "c"] = np.nan
    with pytest.raises(DataValidationError, match="'c' has a single observed value"):
        DFM(tent_weights={"q:m": [1, 2, 3, 2, 1]}).fit(_quarterly(single))
    constant = df.copy()
    constant.loc[quarter_end, "c"] = 4.0
    with pytest.raises(DataValidationError, match="'c' is constant"):
        DFM(tent_weights={"q:m": [1, 2, 3, 2, 1]}).fit(_quarterly(constant))


def test_new_rows_are_not_judged_as_training_data():
    df = _frame()
    model = DFM(max_iter=3).fit(df.iloc[:30])
    new = df.iloc[30:].copy()
    new["b"] = np.nan
    assert np.isfinite(model.predict(2, data=new).to_numpy()).all()
    model.update(new)
    assert model.conditioning_index_[-1] == df.index[-1]


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("cls", [DFM], ids=lambda c: c.__name__)
def test_infinite_target_is_rejected_before_training(monkeypatch, cls, value):
    _no_training(monkeypatch, cls)
    df = _frame()
    df.iloc[7, 1] = value
    with pytest.raises(DataValidationError, match=r"y contains infinite values in columns \['b'\]\. Replace them "
                                                  r"with finite values or NaN \(missing\)"):
        cls(**_TINY[cls]).fit(df)


def test_infinite_new_rows_are_rejected():
    df = _frame()
    model = DFM(max_iter=3).fit(df.iloc[:30])
    before = model.predict(2)
    new = df.iloc[30:].copy()
    new.iloc[2, 0] = np.inf
    with pytest.raises(DataValidationError, match=r"y contains infinite values in columns \['a'\]"):
        model.predict(2, data=new)
    pd.testing.assert_frame_equal(model.predict(2), before)


def _raising(exc: type):
    def raise_(*args, **kwargs):
        raise exc("boom")

    return raise_


@pytest.mark.parametrize(
    ("exc", "expected", "match"),
    [(np.linalg.LinAlgError, RuntimeError, "EM algorithm failed at iteration 1: boom"), (TypeError, TypeError, "boom")],
)
def test_em_step_wraps_numerical_failures_only(monkeypatch, exc, expected, match):
    monkeypatch.setattr(em_module, "em_step", _raising(exc))
    with pytest.raises(expected, match=match):
        DFM(max_iter=3).fit(_frame())


def test_result_summary_reports_unset_arrays_and_lets_bugs_through():
    assert DFMResult(r=np.array(["x"]), Z=np.zeros((4, 2))).num_factors() == 2
    text = DFMResult().summary()
    assert "Series: N/A" in text and "Factors: N/A" in text and "Time periods: N/A" in text
    with pytest.raises(AttributeError):
        DFMResult(x_sm=[[1.0]]).summary()


def test_spline_fill_falls_back_to_forward_backward_fill(monkeypatch):
    x = np.array([[np.nan], [1.0], [np.nan], [3.0], [np.nan]])
    monkeypatch.setattr(dfm_init, "_interpolate", _raising(ValueError))
    filled, _ = dfm_init.remNaNs_spline(x, {"method": 2, "k": 3})
    np.testing.assert_array_equal(filled[:, 0], [1.0, 1.0, 1.0, 3.0, 3.0])
    monkeypatch.setattr(dfm_init, "_interpolate", _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        dfm_init.remNaNs_spline(x, {"method": 2, "k": 3})


def test_spline_fill_method_3_falls_back_to_linear(monkeypatch):
    interpolate = dfm_init._interpolate

    def cubic_fails(col, kind):
        if kind == "cubic":
            raise ValueError("boom")
        return interpolate(col, kind)

    x = np.array([[0.0], [np.nan], [2.0], [3.0], [np.nan], [5.0]])
    monkeypatch.setattr(dfm_init, "_interpolate", cubic_fails)
    filled, _ = dfm_init.remNaNs_spline(x, {"method": 3, "k": 3})
    np.testing.assert_allclose(filled[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_initialization_check_warns_on_covariances_it_cannot_check(caplog):
    m, n = 2, 3
    with caplog.at_level(logging.WARNING, logger="dfm_python"):
        validate_dfm_initialization(
            A=0.5 * np.eye(m), C=np.ones((n, m)), Q=np.ones((m, m + 1)), R=np.eye(n), Z_0=np.zeros(m), V_0=np.eye(m)
        )
    assert "Could not check eigenvalues for Q" in caplog.text

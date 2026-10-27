"""Input robustness of DDFM: the default scaler, degenerate and infinite inputs, and the narrowed exception handlers."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from dfm_python import DDFM, PanelData
from dfm_python.config import DDFMConfig
from dfm_python.utils.errors import DataValidationError
from dfm_python.utils.helper import interpolate_dataframe

_T = 36
_TINY = {
    DDFM: {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 12, "max_iter": 2,
           "random_state": 0, "device": "cpu"},
}
_CONFIGS = {DDFM: DDFMConfig}


def _frame(seed: int = 0) -> pd.DataFrame:
    index = pd.date_range("2001-01-31", periods=_T, freq="ME")
    values = np.random.default_rng(seed).standard_normal((_T, 3))
    return pd.DataFrame(values, index=index, columns=["a", "b", "c"])


def _no_training(monkeypatch, cls) -> None:
    def fail(self, *args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(cls, "_fit", fail)


@pytest.mark.parametrize("cls", [DDFM], ids=lambda c: c.__name__)
def test_default_scaler_is_standard(cls):
    assert cls().scaler == "standard"
    assert _CONFIGS[cls]().scaler == "standard" and _CONFIGS[cls].from_dict({}).scaler == "standard"
    assert cls().get_config() == _CONFIGS[cls]()


@pytest.mark.parametrize("cls", [DDFM], ids=lambda c: c.__name__)
def test_default_scaler_standardizes_the_targets(cls):
    df = _frame() * 500.0 + 1000.0
    model = cls(**_TINY[cls]).fit(df)
    assert isinstance(model.scaler_, StandardScaler)
    np.testing.assert_allclose(model.scaler_.mean_, df.mean().to_numpy())
    assert np.isfinite(model.predict(2).to_numpy()).all()


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
@pytest.mark.parametrize("cls", [DDFM], ids=lambda c: c.__name__)
def test_degenerate_target_is_rejected_before_training(monkeypatch, cls, case):
    _no_training(monkeypatch, cls)
    model = cls(**_TINY[cls])
    with pytest.raises(DataValidationError, match=_MESSAGES[case]) as info:
        model.fit(_degenerate(case))
    assert f"{cls.__name__} cannot be fitted" in str(info.value) and "drop these series" in str(info.value)
    assert not hasattr(model, "result_")


def test_array_input_names_the_default_series_name():
    values = _frame().to_numpy()
    values[:, 2] = 0.0
    with pytest.raises(DataValidationError, match="'series_2' is constant"):
        DDFM(**_TINY[DDFM]).fit(values)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("cls", [DDFM], ids=lambda c: c.__name__)
def test_infinite_target_is_rejected_before_training(monkeypatch, cls, value):
    _no_training(monkeypatch, cls)
    df = _frame()
    df.iloc[7, 1] = value
    with pytest.raises(DataValidationError, match=r"y contains infinite values in columns \['b'\]\. Replace them "
                                                  r"with finite values or NaN \(missing\)"):
        cls(**_TINY[cls]).fit(df)


def test_infinite_exog_is_rejected():
    df = _frame()
    exog = pd.DataFrame({"x0": np.arange(_T, dtype=float), "x1": 1.0}, index=df.index)
    exog.iloc[3, 0] = np.inf
    with pytest.raises(DataValidationError, match=r"DDFM: exog contains infinite values in columns \['x0'\]"):
        DDFM(**_TINY[DDFM]).fit(PanelData(y=df, exog=exog))


@pytest.mark.parametrize("method", ["spline", "cubic"])
def test_dataframe_spline_with_too_few_points_falls_back_to_linear(caplog, method):
    df = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 2.0, np.nan]})
    with caplog.at_level(logging.WARNING, logger="dfm_python"):
        filled = interpolate_dataframe(df, method=method)
    pd.testing.assert_frame_equal(filled, interpolate_dataframe(df, method="linear"))
    assert "falling back to linear" in caplog.text

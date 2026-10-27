"""Input robustness of iVDFM: the default scaler, degenerate and infinite inputs, and the narrowed exception handlers."""

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import dfm_python.models.ivdfm._engine as ivdfm_engine
import dfm_python.numeric.estimator as numeric_estimator
from dfm_python import iVDFM
from dfm_python.config import iVDFMConfig
from dfm_python.config.constants import DEFAULT_PROCESS_NOISE
from dfm_python.utils.errors import DataValidationError

_T = 36
_TINY = {
    iVDFM: {"n_factors": 2, "window": 12, "encoder_hidden_dim": 4, "decoder_hidden_dim": 4, "prior_hidden_dim": 4,
            "encoder_n_hidden_layers": 1, "decoder_n_hidden_layers": 1, "prior_n_hidden_layers": 1,
            "batch_size": 4, "max_epochs": 1, "random_state": 0, "device": "cpu"},
}
_CONFIGS = {iVDFM: iVDFMConfig}
_NEEDS_OPTUNA = pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="needs optuna")


def _frame(seed: int = 0) -> pd.DataFrame:
    index = pd.date_range("2001-01-31", periods=_T, freq="ME")
    values = np.random.default_rng(seed).standard_normal((_T, 3))
    return pd.DataFrame(values, index=index, columns=["a", "b", "c"])


def _no_training(monkeypatch, cls) -> None:
    def fail(self, *args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(cls, "_fit", fail)


@pytest.mark.parametrize("cls", [iVDFM], ids=lambda c: c.__name__)
def test_default_scaler_is_standard(cls):
    assert cls().scaler == "standard"
    assert _CONFIGS[cls]().scaler == "standard" and _CONFIGS[cls].from_dict({}).scaler == "standard"
    assert cls().get_config() == _CONFIGS[cls]()


@pytest.mark.parametrize("cls", [iVDFM], ids=lambda c: c.__name__)
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
@pytest.mark.parametrize("cls", [iVDFM], ids=lambda c: c.__name__)
def test_degenerate_target_is_rejected_before_training(monkeypatch, cls, case):
    _no_training(monkeypatch, cls)
    model = cls(**_TINY[cls])
    with pytest.raises(DataValidationError, match=_MESSAGES[case]) as info:
        model.fit(_degenerate(case))
    assert f"{cls.__name__} cannot be fitted" in str(info.value) and "drop these series" in str(info.value)
    assert not hasattr(model, "result_")


def test_ivdfm_fills_infinite_values_like_missing_ones():
    nan, inf = _frame(), _frame()
    nan.iloc[7, 1] = np.nan
    inf.iloc[7, 1] = np.inf
    inf.iloc[11, 2] = -np.inf
    nan.iloc[11, 2] = np.nan
    a, b = iVDFM(**_TINY[iVDFM]).fit(nan), iVDFM(**_TINY[iVDFM]).fit(inf)
    pd.testing.assert_frame_equal(a.factors_, b.factors_)
    pd.testing.assert_series_equal(a.explained_variance_, b.explained_variance_)
    pd.testing.assert_frame_equal(a.predict(2), b.predict(2))


def test_ivdfm_series_without_finite_values_is_rejected():
    df = _frame()
    df["c"] = np.inf
    with pytest.raises(DataValidationError, match="'c' has no observed value"):
        iVDFM(**_TINY[iVDFM]).fit(df)


def _raising(exc: type):
    def raise_(*args, **kwargs):
        raise exc("boom")

    return raise_


_TUNE_FIXED = {
    "n_factors": 2, "n_regimes": 1, "factor_order": 1, "mixing": False, "encoder_hidden_dim": 8,
    "decoder_hidden_dim": 8, "prior_hidden_dim": 8, "encoder_n_hidden_layers": 1, "decoder_n_hidden_layers": 1,
    "prior_n_hidden_layers": 1,
}


_TUNE_SEARCH = {"min_window": 10, "max_window": 20, "horizon": 6, "max_epochs_per_trial": 1, "min_batch_size": 8}


@_NEEDS_OPTUNA
def test_tuning_rejects_degenerate_targets_up_front(monkeypatch):
    from dfm_python.tuning import tune_hyperparameters

    _no_training(monkeypatch, iVDFM)
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((60, 3)), columns=["a", "b", "c"])
    df["b"] = 1.0
    with pytest.raises(DataValidationError, match="'b' is constant"):
        tune_hyperparameters(iVDFM(random_state=0, device="cpu"), df, n_trials=1, fixed=_TUNE_FIXED, **_TUNE_SEARCH)


@_NEEDS_OPTUNA
def test_tuning_search_space_errors_propagate(monkeypatch):
    from dfm_python.tuning import tune_hyperparameters

    _no_training(monkeypatch, iVDFM)
    df = pd.DataFrame(np.random.default_rng(0).standard_normal((60, 3)), columns=["a", "b", "c"])
    # A fixed window of 2010 makes the lowest stride (1005) exceed the highest (995): no trial can be sampled.
    with pytest.raises(ValueError, match=r"low <= high.*low=1005, high=995"):
        tune_hyperparameters(
            iVDFM(random_state=0, device="cpu"), df, n_trials=2, fixed={**_TUNE_FIXED, "window": 2010}, **_TUNE_SEARCH
        )


@pytest.mark.parametrize("exc", [np.linalg.LinAlgError, ValueError])
def test_ivdfm_initialization_falls_back_on_pca_failures(monkeypatch, caplog, exc):
    monkeypatch.setattr(ivdfm_engine, "extract_pca_factors", _raising(exc))
    with caplog.at_level(logging.WARNING, logger="dfm_python"):
        model = iVDFM(**_TINY[iVDFM]).fit(_frame())
    assert "PCA init failed: boom" in caplog.text and "AR coefficient initialization failed: boom" in caplog.text
    assert np.isfinite(model.predict(2).to_numpy()).all()


@pytest.mark.parametrize("target", [(ivdfm_engine, "extract_pca_factors"), (numeric_estimator, "estimate_arp_ols")])
def test_ivdfm_initialization_lets_bugs_through(monkeypatch, target):
    monkeypatch.setattr(*target, _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        iVDFM(**_TINY[iVDFM]).fit(_frame())


def test_arp_ols_falls_back_to_zeros_on_solve_failures(monkeypatch):
    factors = np.random.default_rng(0).standard_normal((20, 2))
    monkeypatch.setattr(numeric_estimator, "solve_regularized_ols", _raising(np.linalg.LinAlgError))
    ar, b = numeric_estimator.estimate_arp_ols(factors, order=2)
    np.testing.assert_array_equal(ar, 0.0)
    np.testing.assert_array_equal(b, np.float32(DEFAULT_PROCESS_NOISE))
    monkeypatch.setattr(numeric_estimator, "solve_regularized_ols", _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        numeric_estimator.estimate_arp_ols(factors, order=2)

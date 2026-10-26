"""Input robustness: the default scaler, degenerate and infinite inputs, and the narrowed exception handlers."""

import dataclasses
import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import dfm_python.functional.em as em_module
import dfm_python.models.dfm.initialization as dfm_init
import dfm_python.models.ivdfm._engine as ivdfm_engine
import dfm_python.numeric.estimator as numeric_estimator
from dfm_python import AFM, DDFM, DFM, PanelData, iVDFM
from dfm_python.config import DDFMConfig, DFMConfig, iVDFMConfig
from dfm_python.config.constants import DEFAULT_PROCESS_NOISE
from dfm_python.config.estimator import EstimatorConfig, load_yaml
from dfm_python.config.schema.results import DFMResult
from dfm_python.numeric.validator import validate_dfm_initialization
from dfm_python.testing import _deep_equal
from dfm_python.utils.errors import ConfigurationError, DataValidationError
from dfm_python.utils.helper import interpolate_dataframe

_T = 36
_TINY = {
    DFM: {"max_iter": 3},
    DDFM: {"n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 12, "max_iter": 2,
           "random_state": 0, "device": "cpu"},
    iVDFM: {"n_factors": 2, "window": 12, "encoder_hidden_dim": 4, "decoder_hidden_dim": 4, "prior_hidden_dim": 4,
            "encoder_n_hidden_layers": 1, "decoder_n_hidden_layers": 1, "prior_n_hidden_layers": 1,
            "batch_size": 4, "max_epochs": 1, "random_state": 0, "device": "cpu"},
}
_FORECASTERS = [DFM, DDFM, iVDFM]
_CONFIGS = {DFM: DFMConfig, DDFM: DDFMConfig, iVDFM: iVDFMConfig}
_NEEDS_OPTUNA = pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="needs optuna")
_NEEDS_OMEGACONF = pytest.mark.skipif(importlib.util.find_spec("omegaconf") is None, reason="needs omegaconf")


def _frame(seed: int = 0) -> pd.DataFrame:
    index = pd.date_range("2001-01-31", periods=_T, freq="ME")
    values = np.random.default_rng(seed).standard_normal((_T, 3))
    return pd.DataFrame(values, index=index, columns=["a", "b", "c"])


def _no_training(monkeypatch, cls) -> None:
    def fail(self, *args, **kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(cls, "_fit", fail)


# -- default scaler -----------------------------------------------------------


@pytest.mark.parametrize("cls", _FORECASTERS, ids=lambda c: c.__name__)
def test_default_scaler_is_standard(cls):
    assert cls().scaler == "standard"
    assert _CONFIGS[cls]().scaler == "standard" and _CONFIGS[cls].from_dict({}).scaler == "standard"
    assert cls().get_config() == _CONFIGS[cls]()


@pytest.mark.parametrize("cls", _FORECASTERS, ids=lambda c: c.__name__)
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


# -- degenerate target series -------------------------------------------------


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
    return df


_MESSAGES = {
    "no observed value": "'b' has no observed value",
    "a single observed value": "'b' has a single observed value",
    "constant": r"'b' is constant \(every observed value is 2.5\)",
    "constant with gaps": r"'b' is constant \(every observed value is 2.5\)",
}


@pytest.mark.parametrize("case", list(_MESSAGES))
@pytest.mark.parametrize("cls", _FORECASTERS, ids=lambda c: c.__name__)
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


def test_array_input_names_the_default_series_name():
    values = _frame().to_numpy()
    values[:, 2] = 0.0
    with pytest.raises(DataValidationError, match="'series_2' is constant"):
        DDFM(**_TINY[DDFM]).fit(values)


def test_two_distinct_observed_values_are_enough():
    df = _frame()
    df["b"] = np.nan
    df.iloc[[3, 9], 1] = [0.0, 1.0]
    DFM()._check_training_targets(PanelData(y=df))


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


def test_allocation_model_keeps_its_own_rules():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((40, 4)) * 0.01
    returns[:, 1] = 0.0
    panel = PanelData.from_arrays(returns, series_exog=rng.standard_normal((40, 4, 2)))
    AFM()._check_training_targets(panel)


# -- infinite values ----------------------------------------------------------


@pytest.mark.parametrize("value", [np.inf, -np.inf])
@pytest.mark.parametrize("cls", [DFM, DDFM], ids=lambda c: c.__name__)
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


def test_infinite_new_rows_are_rejected():
    df = _frame()
    model = DFM(max_iter=3).fit(df.iloc[:30])
    before = model.predict(2)
    new = df.iloc[30:].copy()
    new.iloc[2, 0] = np.inf
    with pytest.raises(DataValidationError, match=r"y contains infinite values in columns \['a'\]"):
        model.predict(2, data=new)
    pd.testing.assert_frame_equal(model.predict(2), before)


@pytest.mark.parametrize("slot", ["y", "series_exog"])
def test_afm_rejects_infinite_values(monkeypatch, slot):
    _no_training(monkeypatch, AFM)
    rng = np.random.default_rng(0)
    returns, chars = rng.standard_normal((40, 4)) * 0.01, rng.standard_normal((40, 4, 2))
    (returns if slot == "y" else chars)[5, 1] = np.inf
    with pytest.raises(DataValidationError, match=f"AFM: {slot} contains infinite values .* finite values\\.$"):
        AFM(**AFM.get_test_params()[1]).fit(PanelData.from_arrays(returns, series_exog=chars))


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


# -- narrowed exception handlers ----------------------------------------------


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


@dataclasses.dataclass
class _ForwardRefConfig(EstimatorConfig):
    layers: "UndefinedWidths" = (8, 4)  # noqa: F821


def test_config_tuple_fields_fall_back_to_the_defaults_on_unresolvable_hints():
    assert _ForwardRefConfig.from_dict({"layers": [3, 2]}).layers == (3, 2)


@_NEEDS_OMEGACONF
@pytest.mark.parametrize("text", ["model: [1, 2\n", "model:\n  n_factors: ${nope}\n"], ids=["syntax", "interpolation"])
def test_yaml_read_errors_are_configuration_errors_chained_to_the_cause(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Could not read YAML file") as info:
        load_yaml(path)
    assert info.value.__cause__ is not None


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


@pytest.mark.parametrize("method", ["spline", "cubic"])
def test_dataframe_spline_with_too_few_points_falls_back_to_linear(caplog, method):
    df = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 2.0, np.nan]})
    with caplog.at_level(logging.WARNING, logger="dfm_python"):
        filled = interpolate_dataframe(df, method=method)
    pd.testing.assert_frame_equal(filled, interpolate_dataframe(df, method="linear"))
    assert "falling back to linear" in caplog.text


def test_arp_ols_falls_back_to_zeros_on_solve_failures(monkeypatch):
    factors = np.random.default_rng(0).standard_normal((20, 2))
    monkeypatch.setattr(numeric_estimator, "solve_regularized_ols", _raising(np.linalg.LinAlgError))
    ar, b = numeric_estimator.estimate_arp_ols(factors, order=2)
    np.testing.assert_array_equal(ar, 0.0)
    np.testing.assert_array_equal(b, np.float32(DEFAULT_PROCESS_NOISE))
    monkeypatch.setattr(numeric_estimator, "solve_regularized_ols", _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        numeric_estimator.estimate_arp_ols(factors, order=2)


def test_constrained_ols_ridge_falls_back_on_eigenvalue_failures(monkeypatch):
    X = np.zeros((10, 3))  # every eigenvalue of X'X is 0, so the condition number is undefined
    y = np.random.default_rng(0).standard_normal(10)
    R, q = np.array([[1.0, -1.0, 0.0]]), np.zeros(1)
    beta = numeric_estimator.estimate_constrained_ols(y, X, R, q)
    assert beta.shape == (3,) and np.isfinite(beta).all()
    monkeypatch.setattr(np.linalg, "eigvalsh", _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        numeric_estimator.estimate_constrained_ols(y, X, R, q)


def test_initialization_check_warns_on_covariances_it_cannot_check(caplog):
    m, n = 2, 3
    with caplog.at_level(logging.WARNING, logger="dfm_python"):
        validate_dfm_initialization(
            A=0.5 * np.eye(m), C=np.ones((n, m)), Q=np.ones((m, m + 1)), R=np.eye(n), Z_0=np.zeros(m), V_0=np.eye(m)
        )
    assert "Could not check eigenvalues for Q" in caplog.text


class _EqualityRaises:
    def __init__(self, exc: type) -> None:
        self.exc = exc

    def __eq__(self, other):
        raise self.exc("boom")

    __hash__ = None


def test_deep_equal_treats_incomparable_values_as_unequal():
    assert _deep_equal(_EqualityRaises(TypeError), _EqualityRaises(TypeError)) is False
    assert _deep_equal(_EqualityRaises(ValueError), _EqualityRaises(ValueError)) is False
    with pytest.raises(RuntimeError, match="boom"):
        _deep_equal(_EqualityRaises(RuntimeError), _EqualityRaises(RuntimeError))

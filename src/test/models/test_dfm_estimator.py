"""Tests for the public DFM estimator (dfm_python.models.dfm.estimator)."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import inspect
import io
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler, StandardScaler

import dfm_python
import dfm_python.models
import dfm_python.models.dfm
from dfm_python import PanelData
from dfm_python.config import DFMConfig, load_yaml
from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.config.schema.results import DFMResult
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.models.dfm._engine import DFMEngine
from dfm_python.models.dfm.estimator import _TIME_COLUMN, DFM
from dfm_python.models.dfm.tent import generate_tent_weights
from dfm_python.registry import all_estimators
from dfm_python.testing import _state_hash, make_panel
from dfm_python.utils.errors import ConfigurationError, DataValidationError, NotFittedError, UnsupportedInputError

_YAML = importlib.util.find_spec("omegaconf") is not None or importlib.util.find_spec("yaml") is not None
TENT = {"q:m": [1, 2, 3, 2, 1]}
TINY = {"max_iter": 5, "tent_weights": TENT}
_MONTHLY = ["m0", "m1", "m2", "m3"]


def _frame(T: int = 60, *, seed: int = 0, index: str = "datetime") -> pd.DataFrame:
    """Four monthly series and one quarterly series (NaN off quarter ends) driven by one AR(1) factor."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    y = np.outer(f, rng.standard_normal(5)) + 0.5 * rng.standard_normal((T, 5))
    idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date") if index == "datetime" else pd.RangeIndex(T)
    df = pd.DataFrame(y, index=idx, columns=_MONTHLY + ["gdp"])
    if index == "datetime":
        df.loc[df.index.month % 3 != 0, "gdp"] = np.nan
    else:
        df.loc[df.index % 3 != 2, "gdp"] = np.nan
    df.iloc[7, 1] = np.nan
    return df


def _mixed(order=None, **kwargs) -> PanelData:
    df = _frame(**kwargs)
    if order is not None:
        df = df[order]
    return PanelData(y=df, freq={"gdp": "q"})


def _monthly(**kwargs) -> PanelData:
    return PanelData(y=_frame(**kwargs)[_MONTHLY])


@pytest.fixture(scope="module")
def fitted() -> DFM:
    return DFM(**TINY, scaler="standard").fit(_mixed(["gdp", "m0", "m1", "m2", "m3"]))


# -- construction ---------------------------------------------------------------


def test_defaults_are_the_values_that_ran_when_omitted():
    assert DFM().get_params() == {
        "n_factors": None, "blocks": None, "clock": "m", "tent_weights": None, "max_iter": 5000, "tol": 1e-5,
        "ar_clip": None, "data_clip": None, "regularization": None, "damping_factor": None,
        "em_acceleration": "squarem", "scaler": None, "verbose": 0,
    }
    # The engine config defaults these came from.
    assert _DFMEngineConfig(blocks={"b": {"num_factors": 1, "series": []}}).max_iter == 5000
    assert _DFMEngineConfig(blocks={"b": {"num_factors": 1, "series": []}}).threshold == 1e-5
    assert _DFMEngineConfig(blocks={"b": {"num_factors": 1, "series": []}}).em_acceleration == "squarem"
    assert _DFMEngineConfig.from_dict({"blocks": {"b": {"num_factors": 1, "series": []}}}).em_acceleration == "squarem"


def test_documented_em_defaults_equal_get_params():
    """The EM defaults stated in the docstring, the config mirror and the CHANGELOG are the ones that run."""
    import re

    params = DFM().get_params()
    doc = DFM.__doc__
    max_iter = re.search(r"^\s*max_iter : int, default (\S+)\s*$", doc, re.M)
    tol = re.search(r"^\s*tol : float, default (\S+)\s*$", doc, re.M)
    assert max_iter and tol
    assert int(max_iter.group(1)) == params["max_iter"]
    assert float(tol.group(1)) == params["tol"]
    assert DFMConfig().max_iter == params["max_iter"] and DFMConfig().tol == params["tol"]
    changelog = (Path(__file__).resolve().parents[3] / "CHANGELOG.md").read_text(encoding="utf-8")
    row = re.search(r"`DFM\(\*, [^`]*`", changelog)
    assert row, "the CHANGELOG constructor row for DFM"
    assert int(re.search(r"max_iter=([^,]+),", row.group(0)).group(1)) == params["max_iter"]
    assert float(re.search(r"\btol=([^,]+),", row.group(0)).group(1)) == params["tol"]


def test_constructor_only_stores_and_is_keyword_only():
    np_state = np.random.get_state()
    est = DFM(n_factors=0, clock="x", tent_weights="nonsense", scaler="nonsense", blocks=[1])
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert vars(est) == est.get_params()
    assert est.blocks == [1]
    with pytest.raises(TypeError):
        DFM(1)  # noqa
    for old in ("dataset", "config", "threshold", "num_factors", "frequency", "idio_rho0", "idio_min_var",
                "target_scaler", "random_state", "device"):
        with pytest.raises(TypeError):
            DFM(**{old: 1})


def test_tags():
    tags = DFM.get_tags()
    assert tags["task"] == "forecast" and tags["backend"] == "numpy"
    for cap in ("mixed_frequency", "missing_values", "update", "factor_forecast"):
        assert tags[f"capability:{cap}"] is True
    for cap in ("exog", "series_exog", "device"):
        assert tags[f"capability:{cap}"] is False
    assert tags["update_kind"] == "filter" and tags["predict_data"] == "new_rows"
    assert tags["non_deterministic"] is False


def test_exports():
    assert dfm_python.DFM is DFM
    assert dfm_python.models.DFM is DFM
    assert dfm_python.models.dfm.DFM is DFM
    assert "DFM" in dfm_python.__all__ and "DFM" in dfm_python.models.__all__
    assert issubclass(DFM, dfm_python.BaseFactorModel) and DFM is not DFMEngine
    assert not hasattr(dfm_python.models.dfm, "DFMEngine")


def test_registered_for_the_contract_suite():
    """check_estimator(DFM) runs in src/test/contract for both parameter sets, update and save/load included."""
    assert dict(all_estimators())["DFM"] is DFM
    sets = DFM.get_test_params()
    assert len(sets) == 2 and all(p["tent_weights"] == {"q:m": [1, 2, 3, 2, 1]} for p in sets)
    assert sets[1]["blocks"] == {"global": {"n_factors": 1, "series": ["s0", "s1", "s2"]},
                                 "pair": {"n_factors": 1, "series": ["s1", "s2"]}}
    # The contract fixture puts the quarterly s0 first, so the stable partition moves it last.
    panel = make_panel(DFM)
    assert list(panel.series_names) == ["s0", "s1", "s2"] and panel.freq == {"s0": "q", "s1": "m", "s2": "m"}
    for params in sets:
        model = DFM(**params).fit(panel)
        assert list(model.internal_feature_names_) == ["s1", "s2", "s0"]
        assert np.isfinite(model.result_.Z).all() and np.isfinite(model.predict(3).to_numpy()).all()


# -- config mirror ----------------------------------------------------------------


def test_config_mirror():
    assert DFM._config_class is DFMConfig
    assert dfm_python.DFMConfig is DFMConfig and dfm_python.config.DFMConfig is DFMConfig
    assert DFMConfig.field_names() == tuple(inspect.signature(DFM.__init__).parameters)[1:]
    assert DFMConfig().to_dict() == DFM().get_params()
    blocks = {"g": {"n_factors": 2, "series": ["m0", "gdp"]}}
    est = DFM.from_config({"blocks": blocks, "tent_weights": TENT, "tol": 1e-3, "scaler": "robust"})
    assert est.blocks == blocks and est.tent_weights == TENT and est.tol == 1e-3 and est.scaler == "robust"
    assert est.get_config() == DFMConfig(blocks=blocks, tent_weights=TENT, tol=1e-3, scaler="robust")


@pytest.mark.parametrize(
    "old, canonical",
    [("threshold", "tol"), ("num_factors", "n_factors"), ("frequency", None), ("target_scaler", "scaler"),
     ("idio_rho0", None), ("idio_min_var", None), ("dataset", None), ("config", None)],
)
def test_config_rejects_old_names(old, canonical):
    with pytest.raises(ConfigurationError, match=old) as info:
        DFMConfig.from_dict({old: 1})
    if canonical is not None:
        assert canonical in str(info.value)


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_old_keys_name_the_canonical_ones(tmp_path):
    path = tmp_path / "dfm.yaml"
    path.write_text("model:\n  threshold: 1.0e-05\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"'threshold'.*\btol\b"):
        DFMConfig.from_yaml(path)
    path.write_text("model:\n  num_factors: 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"'num_factors' \(did you mean 'n_factors'\?\)"):
        DFMConfig.from_yaml(path)


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_yaml_values_reach_the_engine(tmp_path, monkeypatch):
    """Spec section 11, check 14: what the YAML sets is what the engine runs with (block order included)."""
    seen = {}
    original = DFMEngine.fit

    def fit(self, *args, **kwargs):
        seen["config"], seen["scaler"] = self._config, self.scaler
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFMEngine, "fit", fit)
    path = tmp_path / "dfm.yaml"
    path.write_text(
        "model:\n"
        "  blocks:\n"
        "    zeta: {n_factors: 1, series: [m0, m1, m2, m3, gdp]}\n"
        "    alpha: {n_factors: 1, series: [m2, m3]}\n"
        "  clock: m\n  tent_weights: {'q:m': [1, 2, 3, 2, 1]}\n  max_iter: 4\n  tol: 1.0e-03\n"
        "  ar_clip: {min: -0.9, max: 0.9}\n  data_clip: 6.0\n  regularization: {scale: 1.0e-05}\n"
        "  damping_factor: 0.9\n  scaler: robust\n"
        "data:\n  freq: {gdp: q}\n",
        encoding="utf-8",
    )
    model = DFM.from_config(DFMConfig.from_yaml(path))
    assert list(model.blocks) == ["zeta", "alpha"] and model.tol == 1e-3 and model.max_iter == 4
    model.fit(PanelData(y=_frame(), freq=load_yaml(path)["data"]["freq"]))
    config = seen["config"]
    assert list(config.blocks) == ["zeta", "alpha"]
    assert config.blocks["alpha"] == {"num_factors": 1, "series": ["m2", "m3"]}
    assert config.tent_weights == {"q:m": [1, 2, 3, 2, 1]} and config.max_iter == 4 and config.threshold == 1e-3
    assert config.ar_clip == {"min": -0.9, "max": 0.9} and config.data_clip == 6.0
    assert config.regularization == {"scale": 1e-5} and config.damping_factor == 0.9
    assert config.frequency == {"m0": "m", "m1": "m", "m2": "m", "m3": "m", "gdp": "q"}
    assert type(seen["scaler"]) is RobustScaler and model.n_factors_ == 2


# -- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "params, name",
    [
        ({"n_factors": 0}, "n_factors"), ({"n_factors": 1.0}, "n_factors"), ({"n_factors": True}, "n_factors"),
        ({"blocks": {}}, "blocks"), ({"blocks": [("b", 1)]}, "blocks"), ({"blocks": {"b": 1}}, "blocks"),
        ({"blocks": {"b": {"n_factors": 0, "series": ["m0"]}}}, "n_factors"),
        ({"blocks": {"b": {"n_factors": 1}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": []}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": "m0"}}}, "series"),
        ({"blocks": {"b": {"n_factors": 1, "series": ["m0", "m0"]}}}, "series"),
        ({"blocks": {1: {"n_factors": 1, "series": ["m0"]}}}, "block names"),
        ({"clock": "x"}, "clock"), ({"clock": None}, "clock"),
        ({"tent_weights": [1, 2, 1]}, "tent_weights"),
        ({"tent_weights": {"x:m": [1, 2, 1]}}, "tent_weights"),
        ({"tent_weights": {"q:w": [1, 2, 1]}}, "clock"),
        ({"tent_weights": {"w": [1, 2, 1]}}, "not slower"),
        ({"tent_weights": {"q": [1, 2, 1], "q:m": [1, 2, 1]}}, "twice"),
        ({"tent_weights": {"q:m": []}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1, 0, 1]}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1, float("nan")]}}, "tent_weights"),
        ({"tent_weights": {"q:m": "121"}}, "tent_weights"),
        ({"tent_weights": {"q:m": [1] * 13}}, "at most 12"),
        ({"max_iter": 0}, "max_iter"), ({"max_iter": 2.0}, "max_iter"),
        ({"tol": -1.0}, "tol"), ({"tol": float("inf")}, "tol"), ({"tol": "1e-5"}, "tol"),
        ({"ar_clip": {"low": -1}}, "min"), ({"ar_clip": [-1, 1]}, "ar_clip"), ({"ar_clip": {}}, "ar_clip"),
        ({"ar_clip": {"min": 1.0, "max": -1.0}}, "ar_clip"), ({"ar_clip": {"min": float("nan")}}, "ar_clip"),
        ({"data_clip": 0}, "data_clip"), ({"data_clip": -3.0}, "data_clip"),
        ({"regularization": {"scal": 1e-6}}, "scale"), ({"regularization": 1e-6}, "regularization"),
        ({"regularization": {"min_eigenvalue": 0.0}}, "min_eigenvalue"),
        ({"regularization": {"scale": -1.0}}, "scale"),
        ({"damping_factor": 0.0}, "damping_factor"), ({"damping_factor": 1.5}, "damping_factor"),
        ({"scaler": "zscore"}, "scaler"), ({"scaler": object()}, "scaler"),
        ({"verbose": -1}, "verbose"), ({"verbose": 1.0}, "verbose"),
    ],
)
def test_invalid_parameters_raise_configuration_error_naming_them(params, name, monkeypatch):
    def no_engine(*args, **kwargs):
        raise AssertionError("the engine must not be built")

    monkeypatch.setattr("dfm_python.models.dfm.estimator.DFMEngine", no_engine)
    est = DFM(**{**TINY, **params})
    with pytest.raises(ConfigurationError, match=name):
        est.fit(_mixed())
    assert not hasattr(est, "result_")


def test_n_factors_and_blocks_are_exclusive():
    est = DFM(n_factors=2, blocks={"b": {"n_factors": 1, "series": _MONTHLY}})
    with pytest.raises(ConfigurationError, match="set n_factors or blocks"):
        est.fit(_monthly())


def test_unknown_block_key_names_n_factors():
    est = DFM(blocks={"b": {"num_factors": 1, "series": _MONTHLY}})
    with pytest.raises(ConfigurationError, match="num_factors.*n_factors"):
        est.fit(_monthly())


def test_unknown_block_series_raise_data_validation_error():
    est = DFM(blocks={"b": {"n_factors": 1, "series": ["m0", "gnp"]}}, max_iter=3)
    with pytest.raises(DataValidationError, match="gnp"):
        est.fit(_monthly())


def test_valid_edge_parameters_are_accepted(monkeypatch):
    built = []

    class Stop(Exception):
        pass

    def engine(*args, **kwargs):
        built.append(kwargs)
        raise Stop

    monkeypatch.setattr("dfm_python.models.dfm.estimator.DFMEngine", engine)
    for params in ({"n_factors": np.int64(2)}, {"tol": 0}, {"scaler": "Standard"}, {"scaler": RobustScaler()},
                   {"tent_weights": {"q": np.array([1.0, 2.0, 1.0])}}, {"tent_weights": {"q:m": (1, 1, 1)}},
                   {"ar_clip": {"max": 0.99}}, {"regularization": {"scale": 0.0}}, {"damping_factor": 1},
                   {"data_clip": 5}, {"blocks": {"g": {"series": ["gdp", "m1"]}, "h": {"n_factors": 2, "series": ("m0",)}}}):
        with pytest.raises(Stop):
            DFM(**{**TINY, **params}).fit(_mixed())
    assert len(built) == 11


def test_unexpected_fit_keywords_raise():
    with pytest.raises(TypeError):
        DFM(**TINY).fit(_mixed(), max_iter=3)


def test_exog_series_exog_and_bad_layouts_are_rejected():
    df = _frame()
    with pytest.raises(UnsupportedInputError, match="exog"):
        DFM(**TINY).fit(PanelData(y=df[_MONTHLY], exog=df[["gdp"]].fillna(0.0)))
    rng = np.random.default_rng(1)
    with pytest.raises(UnsupportedInputError, match="series_exog"):
        DFM(**TINY).fit(PanelData.from_arrays(rng.standard_normal((20, 2)), series_exog=rng.standard_normal((20, 2, 3))))
    with pytest.raises(UnsupportedInputError, match="faster than the clock"):
        DFM(**TINY).fit(PanelData(y=df, freq={"m0": "w"}))
    with pytest.raises(UnsupportedInputError, match="one frequency slower"):
        DFM(**TINY).fit(PanelData(y=df, freq={"gdp": "q", "m0": "a"}))
    with pytest.raises(DataValidationError, match="strings"):
        DFM(max_iter=3).fit(pd.DataFrame(df[_MONTHLY].to_numpy(), index=df.index))


def test_slower_frequency_without_tent_weights_names_the_symmetric_kernel():
    kernel = generate_tent_weights(5).tolist()
    assert kernel == [1, 2, 3, 2, 1]
    with pytest.raises(ConfigurationError) as info:
        DFM(max_iter=3).fit(_mixed())
    message = str(info.value)
    assert "['gdp']" in message and "no default kernel" in message
    assert "generate_tent_weights(5)" in message and "tent_weights={'q:m': [1, 2, 3, 2, 1]}" in message
    # The kernel the message names fits.
    model = DFM(max_iter=3, tent_weights={"q:m": kernel}).fit(_mixed())
    assert model.engine_._config.tent_weights == {"q:m": [1, 2, 3, 2, 1]}
    # Weekly clock with a monthly series: 4 weeks per month, 7 weights.
    with pytest.raises(ConfigurationError, match=r"tent_weights=\{'m:w': \[1, 2, 3, 4, 3, 2, 1\]\}"):
        DFM(clock="w").fit(PanelData(y=_frame()[_MONTHLY], freq={"m0": "m"}))
    # Annual on a monthly clock: the symmetric kernel would exceed the supported size.
    with pytest.raises(ConfigurationError, match="23 weights, more than the 12 supported"):
        DFM().fit(PanelData(y=_frame(), freq={"gdp": "a"}))
    # A tent key for the series' frequency without the clock part works too.
    DFM(max_iter=3, tent_weights={"q": kernel}).fit(_mixed())


# -- the adapter and the engine call ----------------------------------------------


def test_every_hyperparameter_reaches_the_engine(monkeypatch):
    seen = {}
    original = DFMEngine.fit

    def fit(self, *args, **kwargs):
        seen["config"], seen["scaler"], seen["kwargs"] = self._config, self.scaler, kwargs
        seen["columns"] = list(self._dataset.variables.columns)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DFMEngine, "fit", fit)

    def callback(*args, **kwargs):
        return None

    params = {
        "blocks": {"g": {"n_factors": 2, "series": ["m0", "m1", "m2", "m3", "gdp"]}, "r": {"series": ["m2"]}},
        "clock": "m", "tent_weights": {"q:m": [1, 2, 1]}, "max_iter": 4, "tol": 1e-3,
        "ar_clip": {"min": -0.9, "max": 0.9}, "data_clip": 6.0,
        "regularization": {"scale": 1e-5, "min_eigenvalue": 1e-7, "max_eigenvalue": 0.95},
        "damping_factor": 0.9, "scaler": "robust",
    }
    model = DFM(**params).fit(_mixed(["gdp", "m0", "m1", "m2", "m3"]), checkpoint_callback=callback)
    config = seen["config"]
    assert config.blocks == {"g": {"num_factors": 2, "series": ["m0", "m1", "m2", "m3", "gdp"]},
                             "r": {"num_factors": 1, "series": ["m2"]}}
    assert config.frequency == {"m0": "m", "m1": "m", "m2": "m", "m3": "m", "gdp": "q"}
    assert list(config.frequency) == ["m0", "m1", "m2", "m3", "gdp"]
    assert config.clock == "m" and config.tent_weights == {"q:m": [1, 2, 1]}
    assert config.max_iter == 4 and config.threshold == 1e-3
    assert config.ar_clip == {"min": -0.9, "max": 0.9} and config.data_clip == 6.0
    assert config.regularization == params["regularization"] and config.damping_factor == 0.9
    em = config.to_em_config()
    assert em.regularization == 1e-5 and em.min_norm == 1e-7 and em.max_eigenval == 0.95
    assert type(seen["scaler"]) is RobustScaler and seen["kwargs"] == {"checkpoint_callback": callback}
    assert seen["columns"] == ["m0", "m1", "m2", "m3", "gdp"]
    # The engine got copies: the hyperparameters are unchanged objects of the caller.
    assert config.tent_weights["q:m"] is not params["tent_weights"]["q:m"]
    assert config.ar_clip is not params["ar_clip"]
    assert model.get_params()["blocks"] is params["blocks"]
    assert model.n_factors_ == 3


def test_default_block_is_one_global_block():
    model = DFM(max_iter=3).fit(_monthly())
    assert model.engine_._config.blocks == {"Block_Global": {"num_factors": 1, "series": []}}
    assert model.n_factors_ == 1
    model = DFM(n_factors=2, max_iter=3).fit(_monthly())
    assert model.engine_._config.blocks == {"Block_Global": {"num_factors": 2, "series": []}}
    assert model.n_factors_ == 2


def test_fit_matches_the_engine_on_the_internal_order():
    data = _mixed(["gdp", "m0", "m1", "m2", "m3"])
    model = DFM(**TINY, scaler="standard").fit(data)

    frame = _frame()[_MONTHLY + ["gdp"]].reset_index(drop=True)
    frame.insert(0, "t", np.arange(len(frame)))
    config = _DFMEngineConfig.from_dict({
        "blocks": {"Block_Global": {"num_factors": 1, "series": []}},
        "frequency": {**{c: "m" for c in _MONTHLY}, "gdp": "q"},
        "clock": "m", "max_iter": 5, "threshold": 1e-5, "tent_weights": TENT,
    })
    engine = DFMEngine(dataset=DFMDataset(config=config, data=frame, time_index="t"), config=config,
                       scaler=StandardScaler())
    engine.fit()
    result = engine.get_result()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z", "x_sm"):
        np.testing.assert_array_equal(getattr(model.result_, name), getattr(result, name), err_msg=name, strict=True)
    assert model.result_.loglik == result.loglik and model.n_iter_ == result.num_iter
    X, Z = engine.predict(horizon=4)
    np.testing.assert_array_equal(model.predict(4)[_MONTHLY + ["gdp"]].to_numpy(), X, strict=True)
    np.testing.assert_array_equal(model.predict_factors(4).to_numpy(), Z, strict=True)


def test_time_column_never_collides():
    df = _frame()[_MONTHLY].rename(columns={"m0": _TIME_COLUMN})
    model = DFM(max_iter=3).fit(df)
    assert list(model.internal_feature_names_) == [_TIME_COLUMN, "m1", "m2", "m3"]
    assert list(model.predict(2).columns) == [_TIME_COLUMN, "m1", "m2", "m3"]


def test_scaler_instance_is_cloned_and_a_fitted_one_warns():
    scaler = StandardScaler()
    model = DFM(**TINY, scaler=scaler).fit(_mixed())
    assert model.scaler_ is not scaler and not hasattr(scaler, "mean_")
    assert model.scaler_ is model.engine_.scaler and type(model.scaler_) is StandardScaler

    fitted_scaler = StandardScaler().fit(np.arange(10.0).reshape(-1, 1))
    with pytest.warns(UserWarning, match="already fitted") as record:
        model = DFM(**TINY, scaler=fitted_scaler).fit(_mixed())
    assert record[0].filename == __file__
    assert fitted_scaler.mean_.shape == (1,) and model.scaler_.mean_.shape == (5,)


def _ar_panel(phi: float, *, T: int = 80, seed: int = 0) -> PanelData:
    """Four monthly series loading on one AR(1) factor ``f[t] = phi f[t-1] + e[t]``."""
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = phi * f[t - 1] + rng.standard_normal()
    y = np.outer(f, rng.standard_normal(4) + 1.0) + 0.3 * rng.standard_normal((T, 4))
    idx = pd.date_range("2010-01-31", periods=T, freq="ME", name="date")
    return PanelData(y=pd.DataFrame(y, index=idx, columns=_MONTHLY))


def _nonstationary_warnings(record) -> list:
    return [w for w in record if "spectral radius" in str(w.message)]


def test_fit_warns_when_the_fitted_transition_is_explosive():
    """No eigenvalue cap on A: on an explosive input the MLE is explosive and fit says so."""
    with pytest.warns(UserWarning, match="spectral radius") as record:
        model = DFM(n_factors=1, max_iter=200, scaler="standard").fit(_ar_panel(1.05))
    rho = float(np.max(np.abs(np.linalg.eigvals(model.result_.A))))
    assert rho > 1.0
    found = _nonstationary_warnings(record)
    assert len(found) == 1 and found[0].filename == __file__
    message = str(found[0].message)
    assert f"{rho:.4f}" in message and "stationar" in message
    # The forecasts it warns about grow with the horizon instead of reverting to the mean.
    fc = model.predict(120).to_numpy()
    assert np.abs(fc[-1]).max() > 2 * np.abs(fc[0]).max()


def test_fit_does_not_warn_on_stationary_data():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = DFM(n_factors=1, max_iter=200, scaler="standard").fit(_ar_panel(0.7))
    assert float(np.max(np.abs(np.linalg.eigvals(model.result_.A)))) < 1.0
    assert _nonstationary_warnings(record) == []


def test_scaler_none_fits_on_the_data_as_given():
    model = DFM(**TINY).fit(_mixed())
    assert model.scaler_ is None and model.engine_.scaler is None
    np.testing.assert_array_equal(model.engine_.data_processed, _frame()[_MONTHLY + ["gdp"]].to_numpy(), strict=True)


# -- fitted attributes ----------------------------------------------------------------


def test_fitted_attributes(fitted):
    user = ["gdp", "m0", "m1", "m2", "m3"]
    assert list(fitted.feature_names_in_) == user and fitted.n_features_in_ == 5
    assert list(fitted.internal_feature_names_) == ["m0", "m1", "m2", "m3", "gdp"]
    assert fitted.freq_ == {"gdp": "q", "m0": "m", "m1": "m", "m2": "m", "m3": "m"}
    assert list(fitted.freq_) == user
    assert list(fitted.exog_names_in_) == []
    assert isinstance(fitted.engine_, DFMEngine) and fitted.engine_._dataset is None
    assert isinstance(fitted.result_, DFMResult) and fitted.result_ is not fitted.engine_.get_result()
    np.testing.assert_array_equal(fitted.result_.Z, fitted.engine_.get_result().Z, strict=True)
    assert fitted.n_iter_ == fitted.result_.num_iter and isinstance(fitted.n_iter_, int)
    assert fitted.converged_ == bool(fitted.result_.converged) and isinstance(fitted.converged_, bool)
    assert fitted.loglik_ == fitted.result_.loglik and isinstance(fitted.loglik_, float)
    assert fitted.n_factors_ == 1
    assert fitted.index_.equals(_frame().index) and fitted.conditioning_index_.equals(fitted.index_)
    # scaler_ holds the statistics of the series in the internal order.
    np.testing.assert_allclose(fitted.scaler_.mean_, np.nanmean(_frame()[list(fitted.internal_feature_names_)], axis=0))
    # result_ matrices are in the internal order: one row of C per series.
    assert fitted.result_.C.shape[0] == 5 and fitted.result_.x_sm.shape == (60, 5)


def test_factors_and_states_frames(fitted):
    states = fitted.states_
    m = fitted.result_.Z.shape[1]
    assert list(states.columns) == [f"state_{i}" for i in range(m)]
    np.testing.assert_array_equal(states.to_numpy(), fitted.result_.Z, strict=True)
    # One factor: its current value is the first state.
    assert list(fitted.factors_.columns) == ["factor_0"] and list(fitted.factor_names_) == ["factor_0"]
    np.testing.assert_array_equal(fitted.factors_.to_numpy(), fitted.result_.Z[:, :1], strict=True)
    for frame in (states, fitted.factors_):
        assert frame.index.equals(fitted.index_) and frame.index.name == "date"
        assert all(dtype == np.float64 for dtype in frame.dtypes)


def test_unfitted_access_raises_not_fitted():
    est = DFM(**TINY)
    assert not hasattr(est, "result_") and not hasattr(est, "engine_")
    for call in (lambda: est.predict(3), lambda: est.predict_factors(3)):
        with pytest.raises(NotFittedError):
            call()


def test_failed_fit_leaves_the_model_unfitted():
    model = DFM(**TINY).fit(_mixed())
    model.set_params(tent_weights=None)
    with pytest.raises(ConfigurationError):
        model.fit(_mixed())
    assert not hasattr(model, "result_") and not hasattr(model, "engine_")


def test_fit_is_repeatable_and_does_not_mutate_inputs():
    data = _mixed()
    before = data.y.copy()
    params = {**TINY, "tent_weights": {"q:m": [1, 2, 3, 2, 1]}}
    a = DFM(**params, scaler="standard").fit(data)
    b = DFM(**params, scaler="standard").fit(data)
    np.testing.assert_array_equal(a.result_.Z, b.result_.Z, strict=True)
    assert a.predict(3).equals(b.predict(3))
    assert data.y.equals(before) and params["tent_weights"] == {"q:m": [1, 2, 3, 2, 1]}


def test_fit_logs_nothing_with_verbose_zero(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        model = DFM(**TINY).fit(make_panel(DFM))
        model.predict(3)
        model.predict_factors(3)
    assert [r for r in caplog.records if r.levelno < logging.ERROR] == []


def test_fit_with_verbose_logs_the_engine_progress(caplog):
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        DFM(**{**TINY, "verbose": 1}).fit(_mixed())
    assert any(r.name.startswith("dfm_python.models.dfm") and r.levelno == logging.INFO for r in caplog.records)


# -- forecasting --------------------------------------------------------------------


def test_predict_frame_in_the_user_order(fitted):
    forecast = fitted.predict(4)
    assert list(forecast.columns) == ["gdp", "m0", "m1", "m2", "m3"]
    assert forecast.shape == (4, 5) and all(dtype == np.float64 for dtype in forecast.dtypes)
    expected_index = pd.date_range("2015-01-31", periods=4, freq="ME", name="date")
    assert forecast.index.equals(expected_index)
    assert forecast.attrs == {"model": "DFM", "kind": "forecast", "origin": fitted.index_[-1]}
    # The quarterly series is reported at every clock date.
    assert np.isfinite(forecast["gdp"].to_numpy()).all()
    X, _ = fitted.engine_.predict(horizon=4)
    internal = list(fitted.internal_feature_names_)
    np.testing.assert_array_equal(forecast.to_numpy(), pd.DataFrame(X, columns=internal)[list(forecast.columns)].to_numpy())


def test_predict_factors_frame(fitted):
    factors = fitted.predict_factors(4)
    assert list(factors.columns) == list(fitted.states_.columns)
    assert factors.index.equals(fitted.predict(4).index)
    assert factors.attrs == {"model": "DFM", "kind": "factors", "origin": fitted.index_[-1]}
    _, Z = fitted.engine_.predict(horizon=4)
    np.testing.assert_array_equal(factors.to_numpy(), Z, strict=True)


def test_range_index_continues_by_position():
    model = DFM(**TINY).fit(PanelData(y=_frame(index="range"), freq={"gdp": "q"}))
    assert model.predict(3).index.equals(pd.RangeIndex(60, 63, name="time"))
    assert model.factors_.index.equals(pd.RangeIndex(60))


def test_permuted_input_is_reported_in_the_user_order():
    """The internal order is the stable partition [clock | slower]; outputs follow the user's order."""
    ordered = DFM(**TINY, scaler="standard").fit(_mixed(_MONTHLY + ["gdp"]))
    assert list(ordered.internal_feature_names_) == _MONTHLY + ["gdp"]
    expected = ordered.predict(5)

    # Moving the slower series anywhere keeps the internal order, so the numbers are the same.
    for order in (["gdp"] + _MONTHLY, ["m0", "m1", "gdp", "m2", "m3"]):
        model = DFM(**TINY, scaler="standard").fit(_mixed(order))
        assert list(model.internal_feature_names_) == _MONTHLY + ["gdp"]
        assert list(model.feature_names_in_) == order
        np.testing.assert_array_equal(model.result_.Z, ordered.result_.Z, strict=True)
        forecast = model.predict(5)
        assert list(forecast.columns) == order
        assert forecast[_MONTHLY + ["gdp"]].equals(expected)
        assert model.predict_factors(5).equals(ordered.predict_factors(5))

    # Within the clock group the user's relative order is kept.
    permuted_order = ["m2", "gdp", "m0", "m3", "m1"]
    permuted = DFM(**TINY, scaler="standard").fit(_mixed(permuted_order))
    assert list(permuted.internal_feature_names_) == ["m2", "m0", "m3", "m1", "gdp"]
    assert list(permuted.predict(5).columns) == permuted_order
    assert list(permuted.freq_) == permuted_order


def test_predict_does_not_change_the_model(fitted):
    before = _state_hash(fitted)
    first = fitted.predict(3)
    fitted.predict_factors(3)
    assert _state_hash(fitted) == before
    assert fitted.predict(3).equals(first)


@pytest.mark.parametrize("h", [1.0, "3", None, True])
def test_horizon_must_be_an_integer(fitted, h):
    with pytest.raises(TypeError):
        fitted.predict(h)
    with pytest.raises(TypeError):
        fitted.predict_factors(h)


def test_horizon_must_be_positive(fitted):
    with pytest.raises(ValueError):
        fitted.predict(0)


# -- new rows: predict(data=) and update ----------------------------------------------

_FIT_ROWS = 54


def _split_frame(order=None, *, index: str = "datetime", n_fit: int = _FIT_ROWS):
    """The mixed panel's first ``n_fit`` rows (to fit) and the rest (the new rows), as PanelData."""
    df = _frame(index=index)
    if order is not None:
        df = df[order]
    return PanelData(y=df.iloc[:n_fit], freq={"gdp": "q"}), PanelData(y=df.iloc[n_fit:], freq={"gdp": "q"})


@pytest.fixture(scope="module")
def split_fit():
    train, new = _split_frame(["gdp", "m0", "m1", "m2", "m3"])
    return DFM(**TINY, scaler="standard").fit(train), train, new


def _internal_rows(new: PanelData) -> np.ndarray:
    return new.y[_MONTHLY + ["gdp"]].to_numpy(np.float64)


def test_predict_data_is_the_engine_path_on_a_copy(split_fit):
    model, train, new = split_fit
    before = _state_hash(model)
    forecast = model.predict(4, data=new)
    factors = model.predict_factors(4, data=new)
    assert _state_hash(model) == before
    X, Z = copy.deepcopy(model.engine_).predict(horizon=4, data=_internal_rows(new))
    assert forecast.to_numpy().tobytes() == pd.DataFrame(X, columns=_MONTHLY + ["gdp"])[
        ["gdp", "m0", "m1", "m2", "m3"]].to_numpy().tobytes()
    assert factors.to_numpy().tobytes() == np.asarray(Z, dtype=np.float64).tobytes()
    expected_index = pd.date_range(new.index[-1], periods=5, freq="ME")[1:].rename("date")
    assert forecast.index.equals(expected_index) and factors.index.equals(expected_index)
    assert forecast.attrs == {"model": "DFM", "kind": "forecast", "origin": new.index[-1]}
    assert factors.attrs["origin"] == new.index[-1] and factors.attrs["kind"] == "factors"
    # Repeated calls are bitwise equal, and the model still forecasts from its training data.
    assert model.predict(4, data=new).equals(forecast)
    assert model.predict(4).index[0] == train.index[-1] + pd.offsets.MonthEnd(1)


def test_predict_data_with_freq_none_inherits_the_fitted_frequencies(split_fit):
    model, _, new = split_fit
    assert model.predict(3, data=new.y).equals(model.predict(3, data=new))


@pytest.mark.parametrize("method", ["predict", "predict_factors", "update"])
def test_new_rows_are_validated(split_fit, method):
    model, train, new = split_fit
    model = copy.deepcopy(model)
    before = _state_hash(model)

    def call(data):
        if method == "update":
            return model.update(data)
        return getattr(model, method)(3, data=data)

    y = new.y
    with pytest.raises(DataValidationError, match="overlaps"):
        call(PanelData(y=pd.concat([train.y.iloc[-2:], y]), freq={"gdp": "q"}))
    with pytest.raises(DataValidationError, match="overlaps"):
        call(train)
    with pytest.raises(DataValidationError, match="series"):
        call(PanelData(y=y[["m0", "gdp", "m1", "m2", "m3"]], freq={"gdp": "q"}))
    with pytest.raises(DataValidationError, match="series"):
        call(PanelData(y=y.drop(columns="m3"), freq={"gdp": "q"}))
    with pytest.raises(DataValidationError, match="frequencies"):
        call(PanelData(y=y, freq={"gdp": "q", "m0": "q"}))
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        call(PanelData.from_arrays(y.to_numpy(), names=list(y.columns), freq={"gdp": "q"}))
    with pytest.raises(UnsupportedInputError, match="capability:exog"):
        call(PanelData(y=y, exog=y[["m0"]].rename(columns={"m0": "x"}), freq={"gdp": "q"}))
    assert _state_hash(model) == before


def test_new_rows_on_a_range_index_must_continue_it():
    train, new = _split_frame(index="range")
    model = DFM(**TINY).fit(train)
    assert model.predict(2, data=new).index.equals(pd.RangeIndex(60, 62, name="time"))
    gap = PanelData(y=new.y.set_axis(pd.RangeIndex(55, 61)), freq={"gdp": "q"})
    with pytest.raises(DataValidationError, match="continue the model's index at 54"):
        model.predict(2, data=gap)
    model.update(new)
    assert model.conditioning_index_.equals(pd.RangeIndex(60)) and type(model.conditioning_index_) is pd.RangeIndex
    assert model.predict(2).index.equals(pd.RangeIndex(60, 62, name="time"))


def test_extend_index_keeps_the_datetime_frequency():
    index = pd.date_range("2010-01-31", periods=5, freq="ME", name="date")
    extended = DFM._extend_index(index, pd.date_range("2010-06-30", periods=2, freq="ME"), "DFM.update")
    assert extended.freq == "ME" and extended.name == "date" and len(extended) == 7
    # A gap keeps the stamps; the forecast index then infers the step.
    gap = DFM._extend_index(index, pd.DatetimeIndex(["2010-08-31"]), "DFM.update")
    assert gap.freq is None and gap[-1] == pd.Timestamp("2010-08-31") and gap.name == "date"
    periods = pd.period_range("2010-01", periods=3, freq="M", name="p")
    extended = DFM._extend_index(periods, pd.period_range("2010-04", periods=2, freq="M"), "DFM.update")
    assert isinstance(extended, pd.PeriodIndex) and len(extended) == 5 and extended.name == "p"


def test_update_filters_the_new_rows_with_fixed_parameters(split_fit):
    fitted, train, new = split_fit
    model = copy.deepcopy(fitted)
    engine = copy.deepcopy(fitted.engine_)
    result, factors = model.result_, model.factors_.copy()
    params = {name: np.array(getattr(model.engine_, name), copy=True) for name in ("A", "C", "Q", "R", "Z_0", "V_0")}
    assert model.update(new) is model

    engine.update(_internal_rows(new))
    np.testing.assert_array_equal(model.engine_.get_result().Z, engine.get_result().Z, strict=True)
    np.testing.assert_array_equal(model.engine_.data_processed, engine.data_processed, strict=True)
    assert model.engine_.get_result().Z.shape[0] == 60 and model.engine_.data_processed.shape[0] == 60
    for name, value in params.items():
        np.testing.assert_array_equal(getattr(model.engine_, name), value, strict=True)
    # result_, factors_ and the training diagnostics are unchanged; the conditioning index grows.
    assert model.result_ is result and result.Z.shape[0] == _FIT_ROWS and model.factors_.equals(factors)
    assert model.conditioning_index_.equals(_frame().index) and model.conditioning_index_.freq == "ME"
    assert model.index_.equals(train.index)
    X, Z = engine.predict(horizon=3)
    forecast = model.predict(3)
    assert forecast[_MONTHLY + ["gdp"]].to_numpy().tobytes() == np.asarray(X).tobytes()
    assert model.predict_factors(3).to_numpy().tobytes() == np.asarray(Z).tobytes()
    assert forecast.index[0] == _frame().index[-1] + pd.offsets.MonthEnd(1)
    # predict(h, data=new) on the fitted model continues the same filter as update(new): the same forecast.
    np.testing.assert_allclose(fitted.predict(3, data=new).to_numpy(), forecast.to_numpy(), rtol=1e-12, atol=1e-12)
    assert fitted.predict(3, data=new).index.equals(forecast.index)


def test_update_one_row_then_more(split_fit):
    fitted, _, new = split_fit
    model = copy.deepcopy(fitted)
    engine = copy.deepcopy(fitted.engine_)
    rows = _internal_rows(new)
    model.update(PanelData(y=new.y.iloc[:1], freq={"gdp": "q"}))
    model.update(PanelData(y=new.y.iloc[1:], freq={"gdp": "q"}))
    engine.update(rows[:1])
    engine.update(rows[1:])
    np.testing.assert_array_equal(model.engine_.get_result().Z, engine.get_result().Z, strict=True)
    assert model.predict(2).equals(model.predict(2)) and model.conditioning_index_.equals(_frame().index)
    with pytest.raises(DataValidationError, match="overlaps"):
        model.update(new)


def test_update_refit_runs_one_em_iteration_on_all_rows(split_fit):
    fitted, _, new = split_fit
    model = copy.deepcopy(fitted)
    engine = copy.deepcopy(fitted.engine_)
    assert model.update(new, refit=True) is model
    engine.update(_internal_rows(new), retrain=True)
    result = engine.get_result()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z", "x_sm"):
        np.testing.assert_array_equal(getattr(model.result_, name), getattr(result, name), err_msg=name, strict=True)
    assert model.result_ is not model.engine_.get_result()
    assert model.n_iter_ == result.num_iter == 1 and model.converged_ == bool(result.converged)
    assert model.loglik_ == float(result.loglik)
    assert model.factors_.index.equals(model.conditioning_index_) and model.factors_.shape[0] == 60
    np.testing.assert_array_equal(model.states_.to_numpy(), result.Z, strict=True)
    assert model.engine_._config.threshold == fitted.tol
    X, _ = engine.predict(horizon=3)
    assert model.predict(3)[_MONTHLY + ["gdp"]].to_numpy().tobytes() == np.asarray(X).tobytes()


def test_update_logs_nothing_with_verbose_zero(split_fit, caplog):
    model = copy.deepcopy(split_fit[0])
    with caplog.at_level(logging.DEBUG, logger="dfm_python"):
        model.predict(3, data=split_fit[2])
        model.update(split_fit[2], refit=True)
    assert [r for r in caplog.records if r.levelno < logging.ERROR] == []


def test_update_before_fit_raises():
    with pytest.raises(NotFittedError):
        DFM(**TINY).update(_mixed())


# -- save / load ----------------------------------------------------------------------


def _assert_same_model(loaded: DFM, model: DFM, new: PanelData | None = None) -> None:
    assert type(loaded) is DFM
    for got, expected in ((loaded.predict(4), model.predict(4)), (loaded.predict_factors(4), model.predict_factors(4))):
        assert got.index.equals(expected.index) and list(got.columns) == list(expected.columns)
        assert got.to_numpy().tobytes() == expected.to_numpy().tobytes() and got.attrs == expected.attrs
    if new is not None:
        assert loaded.predict(4, data=new).to_numpy().tobytes() == model.predict(4, data=new).to_numpy().tobytes()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0", "Z", "x_sm"):
        assert getattr(loaded.result_, name).tobytes() == getattr(model.result_, name).tobytes(), name
    assert loaded.result_.loglik == model.result_.loglik and loaded.n_iter_ == model.n_iter_
    assert loaded.converged_ == model.converged_ and loaded.loglik_ == model.loglik_
    assert loaded.n_factors_ == model.n_factors_ and loaded.freq_ == model.freq_
    assert list(loaded.freq_) == list(model.freq_)
    assert list(loaded.internal_feature_names_) == list(model.internal_feature_names_)
    assert list(loaded.feature_names_in_) == list(model.feature_names_in_)
    assert loaded.index_.equals(model.index_) and loaded.conditioning_index_.equals(model.conditioning_index_)
    assert getattr(loaded.conditioning_index_, "freq", None) == getattr(model.conditioning_index_, "freq", None)
    assert type(loaded.conditioning_index_) is type(model.conditioning_index_)
    assert loaded.factors_.equals(model.factors_) and loaded.states_.equals(model.states_)
    assert loaded.get_params() == model.get_params()
    np.testing.assert_array_equal(loaded.engine_.data_processed, model.engine_.data_processed, strict=True)
    np.testing.assert_array_equal(loaded.engine_.get_result().Z, model.engine_.get_result().Z, strict=True)


def test_save_writes_the_numpy_layout(split_fit, tmp_path):
    model, _, _ = split_fit
    model.save(tmp_path / "run")
    files = {p.name for p in (tmp_path / "run").iterdir()}
    assert files == {"config.json", "data_spec.json", "state.npz", "state.json", "result.npz", "result.json",
                     "scaler.json"}
    with np.load(tmp_path / "run" / "state.npz", allow_pickle=False) as npz:
        keys = set(npz.files)
        assert npz["data/data_processed"].flags.f_contiguous == model.engine_.data_processed.flags.f_contiguous
        np.testing.assert_array_equal(npz["data/last_smoothed_state"], model.engine_.get_result().Z[-1], strict=True)
        filtered = model.engine_._create_kalman_filter().filter(np.ma.masked_invalid(model.engine_.data_processed))[1]
        np.testing.assert_array_equal(npz["data/last_smoothed_cov"], filtered[-1], strict=True)
    assert {key for key in keys if not key.startswith("scaler/")} == {
        "ssm/A", "ssm/C", "ssm/Q", "ssm/R", "ssm/Z_0", "ssm/V_0", "mixed_freq/constraint_matrix",
        "mixed_freq/constraint_vector", "mixed_freq/idio_indicator", "structure/blocks", "structure/r",
        "data/data_processed", "data/smoothed_state", "data/last_smoothed_state", "data/last_smoothed_cov",
    }
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["frequencies"] == ["m", "m", "m", "m", "q"] and state["tent_weights_dict"] == {"q": [1, 2, 3, 2, 1]}
    assert state["mixed_freq"] is True and state["n_slower_freq"] == 1 and state["num_iter"] == model.n_iter_
    spec = json.loads((tmp_path / "run" / "data_spec.json").read_text())
    assert spec["internal_feature_names_"] == _MONTHLY + ["gdp"] and spec["freq_"]["gdp"] == "q"


@pytest.mark.parametrize("loader", ["DFM.load", "dfm_python.load"])
def test_save_load_round_trip(split_fit, tmp_path, loader):
    model, _, new = split_fit
    model.save(tmp_path / "run")
    loaded = (DFM.load if loader == "DFM.load" else dfm_python.load)(tmp_path / "run")
    _assert_same_model(loaded, model, new)
    assert type(loaded.scaler_) is StandardScaler and loaded.engine_.scaler is loaded.scaler_
    assert loaded.engine_._dataset is None
    # The loaded model updates and refits like the fitted one.
    for refit in (False, True):
        a, b = copy.deepcopy(model).update(new, refit=refit), copy.deepcopy(loaded).update(new, refit=refit)
        _assert_same_model(b, a)


def test_save_load_after_update_and_refit(split_fit, tmp_path):
    fitted, _, new = split_fit
    first, rest = (PanelData(y=new.y.iloc[:2], freq={"gdp": "q"}), PanelData(y=new.y.iloc[2:], freq={"gdp": "q"}))
    updated = copy.deepcopy(fitted).update(first)
    updated.save(tmp_path / "updated")
    loaded = DFM.load(tmp_path / "updated")
    _assert_same_model(loaded, updated, rest)
    _assert_same_model(copy.deepcopy(loaded).update(rest), copy.deepcopy(updated).update(rest))
    _assert_same_model(copy.deepcopy(loaded).update(rest, refit=True), copy.deepcopy(updated).update(rest, refit=True))

    refitted = copy.deepcopy(updated).update(rest, refit=True)
    refitted.save(tmp_path / "refitted")
    _assert_same_model(DFM.load(tmp_path / "refitted"), refitted)
    # Refit, then filter more rows: the engine's smoothed state is longer than result_.Z.
    more = PanelData(y=_frame(T=66)[["gdp"] + _MONTHLY].iloc[60:], freq={"gdp": "q"})
    extended = copy.deepcopy(refitted).update(more)
    extended.save(tmp_path / "extended")
    loaded = DFM.load(tmp_path / "extended")
    _assert_same_model(loaded, extended)
    assert loaded.factors_.index.equals(extended.factors_.index) and len(loaded.factors_) == 60
    np.testing.assert_array_equal(loaded.engine_.get_result().x_sm, extended.engine_.get_result().x_sm, strict=True)


def test_save_load_keeps_the_block_order_and_needs_no_data(tmp_path):
    """JSON sorts dict keys; the loaded model keeps the block (factor) order and refits like the original."""
    blocks = {"zeta": {"n_factors": 1, "series": _MONTHLY + ["gdp"]}, "alpha": {"series": ["m2", "m3"]}}
    train, new = _split_frame()
    model = DFM(**TINY, blocks=blocks).fit(train)
    model.save(tmp_path / "run")
    loaded = DFM.load(tmp_path / "run")
    assert list(loaded.blocks) == ["zeta", "alpha"] and loaded.scaler_ is None
    _assert_same_model(loaded, model, new)
    _assert_same_model(copy.deepcopy(loaded).update(new, refit=True), copy.deepcopy(model).update(new, refit=True))
    # A clone of the loaded estimator refits to the same numbers as the original parameters.
    again = clone(loaded).fit(train)
    np.testing.assert_array_equal(again.result_.Z, model.result_.Z, strict=True)


def test_save_load_single_frequency_and_range_index(tmp_path):
    model = DFM(n_factors=2, max_iter=3).fit(PanelData(y=_frame(index="range")[_MONTHLY]))
    model.save(tmp_path / "run")
    with np.load(tmp_path / "run" / "state.npz", allow_pickle=False) as npz:
        assert "mixed_freq/idio_indicator" in npz.files
    loaded = DFM.load(tmp_path / "run")
    _assert_same_model(loaded, model)
    assert type(loaded.index_) is pd.RangeIndex


def test_save_when_the_training_filter_is_not_finite(split_fit, tmp_path, monkeypatch):
    """Some mixed-frequency fits leave parameters under which the Kalman filter over the training data is not
    finite (the filter raises NumericalError); save stores the informational last covariance as NaN and warns."""
    from dfm_python.ssm.kalman import DFMKalmanFilter
    from dfm_python.utils.errors import NumericalError

    model, _, _ = split_fit

    def not_finite(self, *args, **kwargs):
        raise NumericalError("Kalman filter: the filtered moments or the log-likelihood are not finite.")

    monkeypatch.setattr(DFMKalmanFilter, "filter", not_finite)
    with pytest.warns(RuntimeWarning, match="last_smoothed_cov is saved as NaN") as record:
        model.save(tmp_path / "run")
    assert record[0].filename == __file__
    with np.load(tmp_path / "run" / "state.npz", allow_pickle=False) as npz:
        m = model.result_.Z.shape[1]
        assert npz["data/last_smoothed_cov"].shape == (m, m) and np.isnan(npz["data/last_smoothed_cov"]).all()
    monkeypatch.undo()
    _assert_same_model(DFM.load(tmp_path / "run"), model)


def test_single_file_pickle_checkpoint_is_rejected(tmp_path):
    """The engine's pickle save/load is gone: a single-file checkpoint of an earlier release is refused, not read."""
    import pickle

    path = tmp_path / "old_dfm_model.pkl"
    path.write_bytes(pickle.dumps({"model_state": None, "config": None}))
    with pytest.raises(ConfigurationError, match="single-file"):
        DFM.load(path)
    with pytest.raises(ConfigurationError, match="single-file"):
        dfm_python.load(path)


def test_load_rejects_an_inconsistent_run(split_fit, tmp_path):
    model, _, _ = split_fit
    model.save(tmp_path / "run")
    path = tmp_path / "run" / "state.json"
    state = json.loads(path.read_text())
    state["engine_config"]["max_iter"] = 99
    path.write_text(json.dumps(state))
    with pytest.raises(ConfigurationError, match="inconsistent"):
        DFM.load(tmp_path / "run")


# -- the adapter --------------------------------------------------------------------


def test_adapter_is_the_stable_partition_and_dfm_dataset_keeps_it():
    """Replaces src/test/dataset/test_dfm_dataset.py: the adapter orders the series [clock | slower]
    (stable: the user's relative order within each group), and DFMDataset's own reorder is then the identity."""
    order = ["m2", "gdp", "m0", "m3", "m1"]
    data = _mixed(order)
    est = DFM(**TINY)
    freq = est._series_frequencies(data)
    assert list(freq) == order and freq["gdp"] == "q"
    internal = est._internal_order(freq)
    assert internal == ["m2", "m0", "m3", "m1", "gdp"]
    config, dataset = est._adapter(data, freq, internal)
    assert list(dataset.variables.columns) == internal
    assert list(config.frequency) == internal
    np.testing.assert_array_equal(dataset.variables.to_numpy(), data.y[internal].to_numpy(), strict=True)
    # DFMDataset applies the same rule to the user's order: the adapter's order is a fixed point of it.
    frame = data.y.reset_index(drop=True)
    frame.insert(0, "t", np.arange(len(frame)))
    assert list(DFMDataset(config=config, data=frame, time_index="t").variables.columns) == internal
    # Ordered input is the identity permutation.
    ordered = _mixed(_MONTHLY + ["gdp"])
    assert est._internal_order(est._series_frequencies(ordered)) == _MONTHLY + ["gdp"]


# -- ported from src/test/models/test_dfm.py (the pre-estimator DFM tests) ---------------


_WEEKLY = ["series_0", "series_1", "series_2"]
_WEEKLY_BLOCKS = {"block1": {"n_factors": 1, "series": _WEEKLY}}


def _weekly_panel() -> PanelData:
    """test_dfm.py's sample dataset: three weekly N(0, 1) series (seed 42), 50 weeks from 2020-01-05."""
    values = np.random.RandomState(42).randn(50, 3)
    index = pd.date_range(start="2020-01-01", periods=50, freq="W", name="date")
    return PanelData(y=pd.DataFrame(values, index=index, columns=_WEEKLY), freq={name: "w" for name in _WEEKLY})


def _weekly_model() -> DFM:
    """The sample dataset's config: one block of one factor over the three series, weekly clock."""
    return DFM(blocks=_WEEKLY_BLOCKS, clock="w")


def test_ported_construction_needs_no_dataset():
    """Was test_dfm_initialization_requires_dataset: the estimator is built from hyperparameters alone."""
    model = _weekly_model()
    assert vars(model) == model.get_params() and model.blocks is _WEEKLY_BLOCKS
    assert not hasattr(model, "engine_")
    DFM()  # no argument is required
    with pytest.raises(TypeError):
        DFM(dataset=_weekly_panel())  # noqa


def test_ported_fit():
    """Was test_dfm_fit: fit returns the model, and the fitted state space is in result_ and the engine."""
    model = _weekly_model()
    assert model.fit(_weekly_panel()) is model
    for name in ("A", "C", "Q", "R"):
        assert getattr(model.result_, name) is not None
        np.testing.assert_array_equal(getattr(model.result_, name), getattr(model.engine_.training_state, name))
    assert model.result_.C.shape[0] == 3 and model.n_factors_ == 1


def test_ported_predict_before_fit_raises_not_fitted():
    """Was test_dfm_predict_not_trained."""
    model = _weekly_model()
    for call in (lambda: model.predict(5), lambda: model.predict_factors(5)):
        with pytest.raises(NotFittedError):
            call()


def test_ported_predict():
    """Was test_dfm_predict_with_data: predict(horizon=5) returned (X, Z) with 5 rows each."""
    model = _weekly_model().fit(_weekly_panel())
    forecast, factors = model.predict(5), model.predict_factors(5)
    assert forecast.shape == (5, 3) and list(forecast.columns) == _WEEKLY
    assert factors.shape == (5, model.result_.Z.shape[1])
    assert forecast.index[0] == _weekly_panel().index[-1] + pd.offsets.Week(weekday=6)


def test_ported_result_before_fit():
    """Was test_dfm_get_result_not_trained: there is no result before fit."""
    model = _weekly_model()
    assert not hasattr(model, "result_")
    with pytest.raises(NotFittedError):
        model.result_  # noqa: B018
    assert not hasattr(model, "get_result")


def _sparse_monthly_frame() -> pd.DataFrame:
    """test_dfm.py's sparse monthly panel: 200 weeks, 3 weekly series and 2 monthly series observed on 30 weeks."""
    rng = np.random.RandomState(42)
    T, n_weekly, n_monthly = 200, 3, 2
    weekly = rng.randn(T, n_weekly)
    monthly = np.full((T, n_monthly), np.nan)
    observed = np.arange(0, T, step=4)[: int(T * 0.15)]  # every 4 weeks, the first 30 such weeks
    monthly[observed, :] = rng.randn(len(observed), n_monthly)
    columns = [f"weekly_{i}" for i in range(n_weekly)] + [f"monthly_{i}" for i in range(n_monthly)]
    index = pd.date_range(start="2020-01-01", periods=T, freq="W", name="date")
    return pd.DataFrame(np.hstack([weekly, monthly]), columns=columns, index=index)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_ported_mixed_freq_initialization_sparse_monthly():
    """Was test_mixed_freq_initialization_sparse_monthly: monthly series on a weekly clock with ~85% NaN.

    The frequencies come from PanelData.freq (grouped form) and the tent kernel from tent_weights={'m:w': ...};
    the engine's state space is checked as the old test checked the state fit returned, and equals the
    old dataset-and-config engine call bitwise.
    """
    df = _sparse_monthly_frame()
    columns = list(df.columns)
    monthly = [c for c in columns if c.startswith("monthly_")]
    for col in monthly:
        assert df[col].isna().mean() > 0.7
    data = PanelData(y=df, freq={"m": monthly})
    model = DFM(blocks={"Block_Global": {"n_factors": 2, "series": columns}}, clock="w",
                tent_weights={"m:w": [1, 2, 1]})
    model.fit(data)
    state = model.engine_.training_state
    internal = list(model.internal_feature_names_)
    assert internal == columns  # already [weekly | monthly]
    for name in monthly:
        loadings = state.C[internal.index(name), :]
        assert np.any(np.abs(loadings) > 1e-10), f"{name} has all-zero loadings"
    assert state.A.shape[0] == state.A.shape[1] and state.Q.shape == state.A.shape
    assert state.C.shape[0] == len(columns) and state.R.shape[0] == len(columns)
    for name in ("A", "C", "Q", "R"):
        assert np.all(np.isfinite(getattr(state, name))), name

    # The same numbers as the old engine call (DFMDataset with a per-series frequency config, scaler None).
    config = _DFMEngineConfig(
        blocks={"Block_Global": {"num_factors": 2, "series": columns}},
        frequency={c: ("m" if c in monthly else "w") for c in columns}, clock="w", tent_weights={"m:w": [1, 2, 1]},
    )
    frame = df.reset_index()
    engine = DFMEngine(dataset=DFMDataset(config=config, data=frame, time_index="date"), config=config)
    old_state = engine.fit()
    for name in ("A", "C", "Q", "R", "Z_0", "V_0"):
        np.testing.assert_array_equal(getattr(state, name), getattr(old_state, name), err_msg=name, strict=True)


def test_ported_save_and_load(tmp_path):
    """Was test_dfm_save_and_load: the forecasts, the result and the state space survive a save and a load.

    The old test pickled the engine to one file and needed the dataset to load it; DFM.load needs only the
    directory, and the loaded model is bitwise equal (the old test compared to a tolerance).
    """
    model = _weekly_model().fit(_weekly_panel())
    forecast, factors = model.predict(5), model.predict_factors(5)
    model.save(tmp_path / "run")
    assert (tmp_path / "run").is_dir() and not any(p.suffix == ".pkl" for p in (tmp_path / "run").iterdir())
    loaded = DFM.load(tmp_path / "run")
    assert loaded.predict(5).equals(forecast) and loaded.predict_factors(5).equals(factors)
    assert loaded.result_.converged == model.result_.converged and loaded.result_.num_iter == model.result_.num_iter
    assert loaded.result_.loglik == model.result_.loglik
    for name in ("A", "C", "Q"):
        np.testing.assert_array_equal(getattr(loaded.engine_.training_state, name),
                                      getattr(model.engine_.training_state, name), strict=True)


def test_ported_load_without_dataset(tmp_path):
    """Was test_dfm_save_and_load_without_dataset: loading needs no data and no config argument."""
    model = _weekly_model().fit(_weekly_panel())
    model.save(tmp_path / "run")
    for loaded in (DFM.load(tmp_path / "run"), dfm_python.load(tmp_path / "run")):
        assert type(loaded) is DFM and loaded.engine_._dataset is None
        assert loaded.get_params() == model.get_params()
        assert loaded.predict(5).equals(model.predict(5))
    with pytest.raises(TypeError):
        DFM.load(tmp_path / "run", dataset=_weekly_panel())  # noqa
    with pytest.raises(TypeError):
        DFM.load(tmp_path / "run", config=model.get_config())  # noqa


# -- shipped YAMLs and tutorials -------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]
_MACRO_WEEKLY = ["KORFBEV", "KORWHLS", "KORMCMP", "KORELEC", "KORCNST", "KORMACH", "KOROCOINF"]
_MACRO_MONTHLY = ["KOEMPTOTO", "KOHWRWEMP", "KOWRCDURE", "KOCONPRCF", "KOEQUIPTE", "KOWRCCNSE", "KOIMPCONA"]
_EXCHANGE = ["AUD_USD", "GBP_USD", "CAD_USD", "CHF_USD", "CNY_USD", "JPY_USD", "SGD_USD", "NZD_USD"]
_FINANCE_BLOCKS = {
    "Block_ALL": ["E1", "E5", "E10", "E15", "E20", "I1", "I5", "I9", "M1", "M5", "M10", "M15", "M18", "D1", "D5",
                  "D9", "P1", "P5", "P10", "P13", "S1", "S5", "S10", "V1", "V5", "V10",
                  "market_forward_excess_returns"],
    "Block_EI": ["E1", "E5", "E10", "E15", "E20", "I1", "I5", "I9"],
    "Block_MD": ["M1", "M5", "M10", "M15", "M18", "D1", "D5", "D9"],
    "Block_PSV": ["P1", "P5", "P10", "P13", "S1", "S5", "S10", "V1", "V5", "V10", "market_forward_excess_returns"],
}


def _finance_targets() -> list[str]:
    """Block_ALL's series (every block's series) in finance.csv's column order."""
    header = pd.read_csv(_REPO / "data" / "finance.csv", nrows=0).columns
    return [c for c in header if c in set(_FINANCE_BLOCKS["Block_ALL"])]


_SHIPPED = {
    "macro": {
        "model": {"blocks": {"Block_Global": {"n_factors": 3, "series": _MACRO_WEEKLY + _MACRO_MONTHLY}},
                  "clock": "w", "tent_weights": {"m:w": [1, 2, 1]}, "max_iter": 10, "tol": 1e-5,
                  "scaler": "standard"},
        "data": {"time_col": "date", "target_cols": _MACRO_WEEKLY + _MACRO_MONTHLY,
                 "freq": {"w": _MACRO_WEEKLY, "m": _MACRO_MONTHLY}},
        "csv": "macro.csv",
    },
    "exchange": {
        "model": {"n_factors": 2, "clock": "d", "max_iter": 10, "tol": 1e-4, "scaler": "standard"},
        "data": {"time_col": "date", "target_cols": _EXCHANGE},
        "csv": "exchange_rate.csv",
    },
    "finance": {
        "model": {"blocks": {name: {"n_factors": 2, "series": series} for name, series in _FINANCE_BLOCKS.items()},
                  "clock": "d", "max_iter": 10, "tol": 1e-5, "scaler": "standard"},
        "data": {"time_col": "date", "target_cols": _finance_targets()},
        "csv": "finance.csv",
    },
}


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_shipped_yaml_builds_the_estimator(name):
    path = _REPO / "config" / f"dfm_{name}.yaml"
    expected = {**DFM().get_params(), **_SHIPPED[name]["model"]}
    cfg = DFMConfig.from_yaml(path)
    assert cfg.to_dict() == expected
    model = DFM.from_config(cfg)
    assert model.get_params() == expected
    if model.blocks is not None:
        assert list(model.blocks) == list(_SHIPPED[name]["model"]["blocks"])  # the block (factor) order
    sections = load_yaml(path)
    assert sections["data"] == _SHIPPED[name]["data"]
    # Every column the data: section names is in the shipped CSV (finance.csv gets its 'date' in the tutorial).
    header = set(pd.read_csv(_REPO / "data" / _SHIPPED[name]["csv"], nrows=0).columns) | {"date"}
    assert set(sections["data"]["target_cols"]) <= header
    assert set(sections["data"].get("freq", {}).get("m", [])) <= header
    # The blocks name only series the data: section selects.
    for spec in (model.blocks or {}).values():
        assert set(spec["series"]) <= set(sections["data"]["target_cols"])


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
def test_shipped_macro_yaml_runs_the_values_of_the_flat_file():
    """The model: values and the data: frequencies are the flat dfm_macro.yaml values the golden cases ran."""
    from ..parity._cases import _DFM_MACRO_FLAT_CONFIG as flat

    path = _REPO / "config" / "dfm_macro.yaml"
    model = DFM.from_config(DFMConfig.from_yaml(path))
    assert model.blocks == {name: {"n_factors": block["num_factors"], "series": block["series"]}
                            for name, block in flat["blocks"].items()}
    assert model.clock == flat["clock"] and model.tent_weights == flat["tent_weights"]
    assert model.max_iter == flat["max_iter"] and model.tol == flat["threshold"]
    # Never read by the old config, so they ran as None: the estimator's defaults.
    assert model.ar_clip is None and model.regularization is None
    assert model.scaler == flat["scaler"] == "standard"  # the tutorial's StandardScaler
    freq = load_yaml(path)["data"]["freq"]
    per_series = {name: code for code, names in freq.items() for name in names}
    assert per_series == {name: code for code, names in flat["frequency"].items() for name in names}
    # The golden weekly/monthly case ran these values (its manifest records the engine config).
    manifest = json.loads((_REPO / "src" / "test" / "parity" / "fixtures" / "dfm_macro_weekly_monthly.json").read_text())
    engine_config = manifest["hyperparameters"]["config"]
    assert engine_config["max_iter"] == model.max_iter and engine_config["threshold"] == model.tol
    assert engine_config["tent_weights"] == model.tent_weights and engine_config["clock"] == model.clock
    assert engine_config["ar_clip"] is None and engine_config["regularization"] is None
    assert engine_config["frequency"] == per_series


@pytest.mark.skipif(not _YAML, reason="YAML needs omegaconf or PyYAML")
@pytest.mark.parametrize("name", sorted(_SHIPPED))
def test_tutorial_runs_on_truncated_data(name, tmp_path):
    """The tutorial runs end to end on 120 rows with one EM iteration: YAML, PanelData.from_wide, fit,
    predict, predict_factors, save and load."""
    path = _REPO / "tutorial" / f"tutorial_{name}_dfm.py"
    spec = importlib.util.spec_from_file_location(f"_tutorial_{name}_dfm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module.main(["--rows", "120", "--max-iter", "1", "--run-dir", str(tmp_path / "run")])
    text = out.getvalue()
    assert "6-step forecast (data units)" in text and "reloaded forecast is identical" in text
    loaded = DFM.load(tmp_path / "run")
    expected = {**_SHIPPED[name]["model"], "max_iter": 1}
    assert {k: loaded.get_params()[k] for k in expected} == expected
    assert loaded.n_iter_ == 1 and loaded.index_.size == 120
    assert list(loaded.feature_names_in_) == _SHIPPED[name]["data"]["target_cols"]
    assert np.isfinite(loaded.predict(6).to_numpy()).all()


def test_ar_clip_leaves_the_companion_shift_rows_alone():
    """``ar_clip`` bounds the estimated AR coefficients only; the identity shift entries stay 1."""
    from dfm_python.functional.state_layout import DFMStateLayout

    model = DFM(**TINY, scaler="standard", ar_clip={"min": -0.5, "max": 0.5}).fit(_monthly())
    A = model.result_.A
    layout = DFMStateLayout.from_state(model.network_.training_state if hasattr(model, "network_") else model.engine_.training_state)
    assert layout.L >= 2
    start = int(layout.factor_offsets[0])
    shift = A[start + layout.r[0]:start + layout.r[0] * layout.L, start:start + layout.r[0] * (layout.L - 1)]
    assert np.array_equal(shift, np.eye(shift.shape[0], shift.shape[1]))
    cur = layout.factor_current(0)
    lag = layout.factor_lags(0, min(layout.p, layout.L))
    assert np.all(np.abs(A[np.ix_(cur, lag)]) <= 0.5)
    ii = layout.ar1_states
    assert np.all(np.abs(A[ii, ii]) <= 0.5)


def test_refit_failure_leaves_the_model_untouched(monkeypatch):
    """When EM raises during ``update(refit=True)`` the fitted model is unchanged and still usable."""
    import dfm_python.functional.em as em_module

    data = _monthly(T=72)
    model = DFM(**TINY, scaler="standard").fit(PanelData(y=data.y.iloc[:60]))
    before = model.predict(3)
    n_rows = model.engine_.data_processed.shape[0]

    def boom(*args, **kwargs):
        raise RuntimeError("EM exploded")

    monkeypatch.setattr(em_module, "run_em_algorithm", boom)
    with pytest.raises(RuntimeError):
        model.update(PanelData(y=data.y.iloc[60:]), refit=True)
    assert model.engine_.data_processed.shape[0] == n_rows
    pd.testing.assert_frame_equal(model.predict(3), before)

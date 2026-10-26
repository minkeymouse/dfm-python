"""Tests for the private DFM engine (models.dfm._engine.DFMEngine).

The engine is the pre-redesign DFM class, moved and renamed, with one import
fixed: ``update()`` imported ``..numeric.validator`` (the non-existent
``dfm_python.models.numeric``) and always raised ``ModuleNotFoundError``.
These tests cover ``update()`` without any import shim (several rows and one
row, both continuing the training filter, checked against the independent
reference filter in ``src/test/ssm/_reference_kf.py``), the purity of
``predict(horizon, data=)``, that the public
``DFM`` names the estimator, not the engine, the engine's surface after the
removal of its old entry points (``load_config``, the pickle ``save``/``load``,
``reset``, ``result``, the default horizon and the old base class), and the
``mixed_freq.find_slower_frequency`` helper (ported from ``test_dfm.py``).
"""

import copy
import inspect
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from dfm_python.config.schema.model import _DFMEngineConfig
from dfm_python.dataset.dfm_dataset import DFMDataset
from dfm_python.config.schema.results import DFMResult
from dfm_python.models.dfm._engine import DFMEngine
from dfm_python.models.dfm.mixed_freq import find_slower_frequency
from dfm_python.ssm.kalman import DFMKalmanFilter
from dfm_python.utils.errors import ConfigurationError, ModelNotTrainedError

_PARAMS = ("A", "C", "Q", "R", "Z_0", "V_0")


def _panel(T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    loadings = rng.standard_normal(4)
    y = np.outer(f, loadings) + 0.5 * rng.standard_normal((T, 4))
    df = pd.DataFrame(y, columns=[f"s{i}" for i in range(4)])
    df.insert(0, "date", pd.date_range("2015-01-31", periods=T, freq="ME"))
    return df


@pytest.fixture
def fitted():
    df = _panel(60, seed=0)
    cols = [c for c in df.columns if c != "date"]
    config = _DFMEngineConfig.from_dict({
        "blocks": {"Block_Global": {"num_factors": 1, "series": cols}},
        "frequency": {c: "m" for c in cols},
        "clock": "m",
        "max_iter": 5,
        "threshold": 1e-5,
    })
    dataset = DFMDataset(config=config, data=df, time_index="date")
    model = DFMEngine(dataset=dataset, config=config, scaler=StandardScaler())
    model.fit()
    model.get_result()  # computes and caches the result, as update() and predict() do
    return model


def _new_rows(n: int) -> np.ndarray:
    return _panel(60 + n, seed=1).iloc[-n:, 1:].to_numpy(np.float64)


def _reference_filtered_new_rows(model: DFMEngine, new_std: np.ndarray) -> np.ndarray:
    """Filtered means of the new rows from the reference filter over the training and the new rows."""
    from ..ssm import _reference_kf as ref

    p = [getattr(model, k) for k in _PARAMS]
    X_all = np.vstack([model.data_processed, new_std])
    return ref.kalman_filter(*p, X_all)["fm"][-new_std.shape[0]:]


def _params(model: DFMEngine) -> dict[str, np.ndarray]:
    return {k: getattr(model, k).copy() for k in _PARAMS}


def _assert_params_unchanged(model: DFMEngine, before: dict[str, np.ndarray]) -> None:
    for k in _PARAMS:
        np.testing.assert_array_equal(getattr(model, k), before[k], err_msg=k, strict=True)


def test_update_multiple_rows(fitted):
    model = fitted
    params = _params(model)
    Z_before = model.get_result().Z.copy()
    data_before = model.data_processed.copy()
    new = _new_rows(4)
    new_std = model.scaler.transform(new)
    expected = _reference_filtered_new_rows(model, new_std)

    assert model.update(new) is None

    result = model.get_result()
    assert result.Z.shape == (Z_before.shape[0] + 4, Z_before.shape[1])
    np.testing.assert_array_equal(result.Z[:-4], Z_before, strict=True)
    np.testing.assert_allclose(result.Z[-4:], expected, rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(result.x_sm, result.Z @ result.C.T, strict=True)
    np.testing.assert_array_equal(model.data_processed, np.vstack([data_before, new_std]), strict=True)
    _assert_params_unchanged(model, params)
    # The validator import resolved on its own, with no module aliased in.
    assert "dfm_python.models.numeric" not in sys.modules


def test_update_single_row_continues_the_training_filter(fitted):
    model = fitted
    params = _params(model)
    Z_before = model.get_result().Z.copy()
    new = _new_rows(1)
    new_std = model.scaler.transform(new)
    expected = _reference_filtered_new_rows(model, new_std)

    with mock.patch.object(DFMKalmanFilter, "filter_and_smooth",
                           side_effect=AssertionError("update must filter, not smooth")):
        model.update(new)

    Z = model.get_result().Z
    assert Z.shape == (Z_before.shape[0] + 1, Z_before.shape[1])
    np.testing.assert_array_equal(Z[:-1], Z_before, strict=True)
    np.testing.assert_allclose(Z[-1], expected[0], rtol=1e-9, atol=1e-11)
    assert model.data_processed.shape[0] == Z.shape[0]
    _assert_params_unchanged(model, params)


def test_update_retrain_runs_em_on_all_rows(fitted):
    model = fitted
    params = _params(model)
    T = model.data_processed.shape[0]
    model.update(_new_rows(4), retrain=True)
    assert model.data_processed.shape[0] == T + 4
    assert model.get_result().Z.shape[0] == T + 4
    assert model.get_result().num_iter == 1
    assert not np.array_equal(model.A, params["A"])


def test_predict_with_data_leaves_the_model_unchanged(fitted):
    model = fitted
    state = copy.deepcopy(model.training_state)
    data_before = model.data_processed.copy()
    Z_before = model.get_result().Z.copy()
    X0, Z0 = model.predict(horizon=3)

    X_d, Z_d = model.predict(horizon=3, data=_new_rows(4))
    assert X_d.shape == (3, 4) and Z_d.shape == (3, Z_before.shape[1])
    assert not np.array_equal(Z_d, Z0)

    for k in _PARAMS:
        np.testing.assert_array_equal(getattr(model.training_state, k), getattr(state, k), err_msg=k, strict=True)
    np.testing.assert_array_equal(model.data_processed, data_before, strict=True)
    np.testing.assert_array_equal(model._result.Z, Z_before, strict=True)
    X1, Z1 = model.predict(horizon=3)
    np.testing.assert_array_equal(X1, X0, strict=True)
    np.testing.assert_array_equal(Z1, Z0, strict=True)


def test_public_dfm_is_the_estimator_not_the_engine():
    import dfm_python.models
    import dfm_python.models.dfm
    from dfm_python.models.dfm.estimator import DFM

    assert dfm_python.DFM is DFM
    assert dfm_python.models.DFM is DFM
    assert dfm_python.models.dfm.DFM is DFM
    assert DFM is not DFMEngine


# -- ported from test_dfm.py: find_slower_frequency ----------------------------------------


def test_find_slower_frequency_from_tent_weights_dict():
    """The one slower frequency is read from the tent weights (weekly series on a daily clock)."""
    assert find_slower_frequency("d", {"w": np.array([1.0, 2.0, 1.0])}) == "w"


def test_find_slower_frequency_rejects_multiple_slower_frequencies():
    """DFM supports one frequency slower than the clock."""
    with pytest.raises(ConfigurationError, match="Multiple slower frequencies"):
        find_slower_frequency("d", {"w": np.array([1.0, 2.0]), "m": np.array([5.0, 6.0])})


def test_find_slower_frequency_from_hierarchy():
    slower = find_slower_frequency("d", None)
    assert slower is None or isinstance(slower, str)


# -- the engine's surface -----------------------------------------------------------------


class TestEngineSurface:
    """What the engine keeps and what was removed with the old public entry points (DDFM precedent)."""

    @pytest.mark.parametrize("name", ["load_config", "save", "load", "reset", "result"])
    def test_old_entry_points_are_gone(self, fitted, name):
        assert not hasattr(DFMEngine, name)
        assert not hasattr(fitted, name)

    def test_no_old_base_class(self):
        assert DFMEngine.__mro__ == (DFMEngine, object)
        pytest.importorskip("torch")  # dfm_python.models.base holds the torch models' network bases
        import dfm_python.models.base

        assert not hasattr(dfm_python.models.base, "BaseFactorModel")

    def test_init_sets_the_old_base_attributes(self):
        df = _panel(30, seed=0)
        cols = [c for c in df.columns if c != "date"]
        config = _DFMEngineConfig.from_dict({"blocks": {"b": {"num_factors": 1, "series": cols}},
                                             "frequency": {c: "m" for c in cols}, "clock": "m"})
        engine = DFMEngine(dataset=DFMDataset(config=config, data=df, time_index="date"), config=config)
        assert engine._result is None and engine.data_processed is None
        assert engine._config is config and engine._dataset is not None
        assert engine.training_state is not None and engine.training_state.A is None

    def test_result_before_fit_raises(self):
        df = _panel(30, seed=0)
        cols = [c for c in df.columns if c != "date"]
        config = _DFMEngineConfig.from_dict({"blocks": {"b": {"num_factors": 1, "series": cols}},
                                             "frequency": {c: "m" for c in cols}, "clock": "m"})
        engine = DFMEngine(dataset=DFMDataset(config=config, data=df, time_index="date"), config=config)
        with pytest.raises(ModelNotTrainedError):
            engine.get_result()
        engine.training_state = None
        with pytest.raises(ModelNotTrainedError):
            engine._ensure_result()

    def test_ensure_result_caches_get_result(self, fitted):
        result = fitted.get_result()
        assert isinstance(result, DFMResult)
        assert fitted._ensure_result() is result and fitted.get_result() is result

    def test_predict_requires_the_horizon(self, fitted):
        assert inspect.signature(DFMEngine.predict).parameters["horizon"].default is inspect.Parameter.empty
        with pytest.raises(TypeError):
            fitted.predict()  # noqa
        X, Z = fitted.predict(4)
        assert X.shape == (4, 4) and Z.shape[0] == 4

    def test_compute_default_horizon_is_gone(self):
        import dfm_python.utils.misc as misc

        assert not hasattr(misc, "compute_default_horizon")



def test_prefitted_scaler_without_mean_is_used_as_is():
    # A fitted RobustScaler has no ``mean_``; the engine used to refit it on the training data.
    from sklearn.preprocessing import RobustScaler

    df = _panel(60, seed=0)
    cols = [c for c in df.columns if c != "date"]
    config = _DFMEngineConfig.from_dict({
        "blocks": {"Block_Global": {"num_factors": 1, "series": cols}},
        "frequency": {c: "m" for c in cols}, "clock": "m", "max_iter": 3, "threshold": 1e-5,
    })
    scaler = RobustScaler().fit(_panel(80, seed=1)[cols].to_numpy() * 10)
    center, scale = scaler.center_.copy(), scaler.scale_.copy()
    dataset = DFMDataset(config=config, data=df, time_index="date")
    DFMEngine(dataset=dataset, config=config, scaler=scaler).fit()
    np.testing.assert_array_equal(scaler.center_, center)
    np.testing.assert_array_equal(scaler.scale_, scale)

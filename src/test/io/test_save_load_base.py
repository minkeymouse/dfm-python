"""``BaseFactorModel.save`` / ``load`` (format 1) and ``dfm_python.load`` on the contract dummies."""

import json
import warnings

import numpy as np
import pytest
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

import dfm_python
from dfm_python import registry
from dfm_python.base import BaseFactorModel
from dfm_python.testing import check_estimator, make_panel
from dfm_python.utils.errors import ConfigurationError, NotFittedError

from ..contract._dummies import _DummyAllocator, _DummyForecaster, _DummyTorchForecaster
from ._estimators import DoublingScaler, ScaledMean

_REQUIRED = {"config.json", "data_spec.json", "state.npz", "state.json", "result.npz", "result.json"}


def _fitted(cls, **params):
    if cls is _DummyTorchForecaster:
        params.setdefault("device", "cpu")
    return cls(**params).fit(make_panel(cls))


def _files(path):
    return {p.name for p in path.iterdir()}


@pytest.fixture
def register(monkeypatch):
    """Add estimator classes to the registry for one test."""

    def add(*classes):
        entries = tuple((cls.__name__, cls.__module__, cls.__name__) for cls in classes)
        monkeypatch.setattr(registry, "_ESTIMATORS", registry._ESTIMATORS + entries)

    return add


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls, weights",
    [(_DummyForecaster, False), (_DummyTorchForecaster, True), (_DummyAllocator, False)],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_file_set_follows_the_backend(tmp_path, cls, weights):
    _fitted(cls).save(tmp_path / "run")
    expected = _REQUIRED | ({"weights.pt"} if weights else set())
    assert _files(tmp_path / "run") == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_config_json(tmp_path):
    est = _fitted(_DummyTorchForecaster, max_epochs=2, random_state=5)
    est.save(tmp_path / "run")
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["format_version"] == 1
    assert config["model_type"] == "_DummyTorchForecaster"
    assert config["library_version"] == dfm_python.__version__
    assert config["params"] == {"device": "cpu", "learning_rate": 0.05, "max_epochs": 2, "random_state": 5, "verbose": 0}
    assert config["tags"] == _DummyTorchForecaster.get_tags()
    runtime = config["runtime"]
    assert runtime["device"] == "cpu"
    assert runtime["torch_version"] == str(torch.__version__) and runtime["numpy_version"] == np.__version__
    assert runtime["deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()
    assert runtime["num_threads"] == torch.get_num_threads()
    assert list(config) == sorted(config)


def test_config_json_runtime_of_a_numpy_model_has_no_torch_settings(tmp_path):
    _fitted(_DummyForecaster).save(tmp_path / "run")
    runtime = json.loads((tmp_path / "run" / "config.json").read_text())["runtime"]
    assert runtime == {"device": None, "numpy_version": np.__version__}


def test_data_spec_json(tmp_path):
    est = _fitted(_DummyAllocator)
    est.save(tmp_path / "run")
    spec = json.loads((tmp_path / "run" / "data_spec.json").read_text())
    assert spec["feature_names_in_"] == ["s0", "s1", "s2"]
    assert spec["exog_names_in_"] == []
    assert spec["series_exog_names_in_"] == ["c0", "c1"]
    index = spec["index_"]
    assert index["kind"] == "datetime" and index["freq"] == "ME" and index["name"] == "date"
    assert index["length"] == 48 and index["first"] == "2000-01-31 00:00:00"
    assert "series_exog_names_in_" not in json.loads(
        (_save(_fitted(_DummyForecaster), tmp_path / "f") / "data_spec.json").read_text()
    )


def _save(est, path, **kwargs):
    est.save(path, **kwargs)
    return path


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_forecaster_round_trip(tmp_path):
    est = _fitted(_DummyForecaster, shrink=0.25, window=5, lags=(1, 2))
    loaded = _DummyForecaster.load(_save(est, tmp_path / "run"))
    assert type(loaded) is _DummyForecaster
    assert loaded.get_params() == est.get_params() and loaded.lags == (1, 2)
    assert loaded.result_.mean.tobytes() == est.result_.mean.tobytes()
    assert (loaded.result_.n_obs, loaded.result_.scale) == (est.result_.n_obs, est.result_.scale)
    assert loaded.index_.equals(est.index_) and loaded.index_.freq == est.index_.freq
    assert list(loaded.feature_names_in_) == list(est.feature_names_in_) and loaded.n_features_in_ == 3
    assert loaded.feature_names_in_.dtype == object
    assert loaded.predict(4).equals(est.predict(4))


def test_torch_round_trip_and_device(tmp_path):
    est = _fitted(_DummyTorchForecaster)
    path = _save(est, tmp_path / "run")
    torch_state = torch.random.get_rng_state()
    loaded = _DummyTorchForecaster.load(path, device="cpu")
    assert torch.equal(torch_state, torch.random.get_rng_state())  # skip_init: loading draws nothing
    assert loaded.device_ == torch.device("cpu")
    for (name, a), (_, b) in zip(est.network_.state_dict().items(), loaded.network_.state_dict().items()):
        assert torch.equal(a, b), name
    assert loaded.predict(3).to_numpy().tobytes() == est.predict(3).to_numpy().tobytes()
    assert loaded.result_.loss == est.result_.loss
    # device=None follows the device parameter ("cpu" here)
    assert _DummyTorchForecaster.load(path).device_ == torch.device("cpu")


def test_allocator_round_trip_without_training_data(tmp_path):
    est = _fitted(_DummyAllocator)
    data = make_panel(_DummyAllocator)
    loaded = _DummyAllocator.load(_save(est, tmp_path / "run"))
    assert loaded.allocate(data=data).equals(est.allocate(data=data))
    assert loaded.result_.sharpe == est.result_.sharpe


@pytest.mark.parametrize("cls", [_DummyForecaster, _DummyTorchForecaster, _DummyAllocator, ScaledMean])
def test_check_estimator_runs_save_load(cls):
    assert check_estimator(cls)[-1] == "save_load"


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaler", [None, "standard", "robust", "minmax", "maxabs", "quantile"])
def test_known_scalers_are_not_pickled(tmp_path, scaler):
    est = _fitted(ScaledMean, scaler=scaler)
    path = _save(est, tmp_path / "run")
    assert _files(path) == _REQUIRED | {"scaler.json"}
    loaded = ScaledMean.load(path)
    assert type(loaded.scaler_) is type(est.scaler_)
    assert loaded.predict(3).to_numpy().tobytes() == est.predict(3).to_numpy().tobytes()


def test_scaler_instances_in_params(tmp_path):
    est = _fitted(ScaledMean, scaler=RobustScaler(quantile_range=(5.0, 95.0)))
    path = _save(est, tmp_path / "run")
    param = json.loads((path / "config.json").read_text())["params"]["scaler"]
    assert param["class"] == "sklearn.preprocessing._data.RobustScaler"
    loaded = ScaledMean.load(path)
    assert isinstance(loaded.scaler, RobustScaler) and loaded.scaler.quantile_range == (5.0, 95.0)
    assert loaded.scaler is not loaded.scaler_
    assert loaded.predict(2).equals(est.predict(2))


def test_pickled_scaler_needs_allow_pickle(tmp_path):
    est = _fitted(ScaledMean, scaler="custom")
    path = _save(est, tmp_path / "run")
    assert _files(path) == _REQUIRED | {"scaler.pkl"}
    with pytest.raises(ConfigurationError, match="allow_pickle=True"):
        ScaledMean.load(path)
    loaded = ScaledMean.load(path, allow_pickle=True)
    assert isinstance(loaded.scaler_, DoublingScaler)
    assert loaded.predict(2).equals(est.predict(2))


def test_custom_estimator_param_needs_allow_pickle(tmp_path):
    est = _fitted(ScaledMean, scaler=DoublingScaler(factor=4.0))
    path = _save(est, tmp_path / "run")
    with pytest.raises(ConfigurationError, match="(?s)'scaler'.*allow_pickle=True"):
        ScaledMean.load(path)
    loaded = ScaledMean.load(path, allow_pickle=True)
    assert isinstance(loaded.scaler, DoublingScaler) and loaded.scaler.factor == 4.0


def test_preprocessing_param_is_rebuilt_without_pickle(tmp_path):
    est = _fitted(ScaledMean, scaler=StandardScaler(with_mean=False))
    loaded = ScaledMean.load(_save(est, tmp_path / "run"))
    assert isinstance(loaded.scaler, StandardScaler) and loaded.scaler.with_mean is False


# ---------------------------------------------------------------------------
# Errors and warnings
# ---------------------------------------------------------------------------


def test_save_before_fit(tmp_path):
    with pytest.raises(NotFittedError, match="save"):
        _DummyForecaster().save(tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_existing_directory_and_overwrite(tmp_path):
    path = _save(_fitted(_DummyForecaster), tmp_path / "run")
    other = _fitted(_DummyForecaster, shrink=0.5)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        other.save(path)
    assert _DummyForecaster.load(path).shrink == 0.0
    other.save(path, overwrite=True)
    assert _DummyForecaster.load(path).shrink == 0.5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_random_state_none_warns(tmp_path):
    with pytest.warns(UserWarning, match="random_state=None"):
        _fitted(_DummyTorchForecaster, random_state=None).save(tmp_path / "a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fitted(_DummyTorchForecaster, random_state=0).save(tmp_path / "b")
        _fitted(_DummyForecaster).save(tmp_path / "c")  # no random_state parameter


@pytest.mark.parametrize("name", ["model.pkl", "model.pt"])
def test_single_file_checkpoint(tmp_path, name):
    (tmp_path / name).write_bytes(b"old")
    with pytest.raises(ConfigurationError, match="release of dfm-python that wrote them"):
        _DummyForecaster.load(tmp_path / name)
    with pytest.raises(ConfigurationError, match="release of dfm-python that wrote them"):
        dfm_python.load(tmp_path / name)


def test_model_type_mismatch(tmp_path):
    path = _save(_fitted(_DummyForecaster), tmp_path / "run")
    with pytest.raises(ConfigurationError, match="holds a '_DummyForecaster' run, not _DummyAllocator"):
        _DummyAllocator.load(path)


def test_backend_files_are_checked(tmp_path):
    path = _save(_fitted(_DummyTorchForecaster), tmp_path / "torch")
    (path / "weights.pt").unlink()
    with pytest.raises(ConfigurationError, match="no weights.pt"):
        _DummyTorchForecaster.load(path)
    path = _save(_fitted(_DummyForecaster), tmp_path / "numpy")
    (path / "weights.pt").write_bytes(b"")
    with pytest.raises(ConfigurationError, match="keeps no torch weights"):
        _DummyForecaster.load(path)
    (path / "weights.pt").unlink()
    (path / "result.json").unlink()
    with pytest.raises(ConfigurationError, match="result.json missing"):
        _DummyForecaster.load(path)


def test_unknown_param_in_config(tmp_path):
    path = _save(_fitted(_DummyForecaster), tmp_path / "run")
    config = json.loads((path / "config.json").read_text())
    config["params"]["shrinkk"] = 1
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConfigurationError, match="did you mean 'shrink'"):
        _DummyForecaster.load(path)


class _NoHooks(_DummyForecaster):
    _save_state = BaseFactorModel._save_state


class _BrokenLoad(_DummyForecaster):
    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        raise RuntimeError("cannot rebuild")


class _ForgetsResult(_DummyForecaster):
    @classmethod
    def _load_state(cls, est, path, device, allow_pickle):
        est.mean_ = cls._read_state(path)[0]["mean"]


@pytest.mark.parametrize(
    "cls, error, match",
    [
        (_NoHooks, NotImplementedError, "_save_state"),
        (_BrokenLoad, RuntimeError, "cannot rebuild"),
        (_ForgetsResult, RuntimeError, "did not restore result_"),
    ],
)
def test_failed_save_leaves_no_debris_and_keeps_the_old_run(tmp_path, cls, error, match):
    with pytest.raises(error, match=match):
        _fitted(cls).save(tmp_path / "run")
    assert list(tmp_path.iterdir()) == []
    _fitted(_DummyForecaster).save(tmp_path / "run")
    with pytest.raises(error, match=match):
        _fitted(cls).save(tmp_path / "run", overwrite=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]
    assert type(_DummyForecaster.load(tmp_path / "run")) is _DummyForecaster


def test_load_hook_is_required():
    with pytest.raises(NotImplementedError, match="_load_state"):
        BaseFactorModel._load_state(None, None, None, False)


# ---------------------------------------------------------------------------
# dfm_python.load
# ---------------------------------------------------------------------------


def test_dfm_python_load_dispatches(tmp_path, register):
    register(_DummyForecaster, _DummyTorchForecaster)
    est = _fitted(_DummyTorchForecaster)
    loaded = dfm_python.load(_save(est, tmp_path / "torch"), device="cpu")
    assert type(loaded) is _DummyTorchForecaster
    assert loaded.predict(2).equals(est.predict(2))
    assert type(dfm_python.load(_save(_fitted(_DummyForecaster), tmp_path / "numpy"))) is _DummyForecaster


def test_dfm_python_load_passes_allow_pickle(tmp_path, register):
    register(ScaledMean)
    path = _save(_fitted(ScaledMean, scaler="custom"), tmp_path / "run")
    with pytest.raises(ConfigurationError, match="allow_pickle=True"):
        dfm_python.load(path)
    assert isinstance(dfm_python.load(path, allow_pickle=True).scaler_, DoublingScaler)


def test_dfm_python_load_unregistered(tmp_path):
    path = _save(_fitted(_DummyForecaster), tmp_path / "run")
    with pytest.raises(ConfigurationError, match=r"'_DummyForecaster' run, which is not a registered estimator"):
        dfm_python.load(path)
    assert dfm_python.load is dfm_python.io.load


def test_check_estimator_uses_dispatch_for_registered_classes(register):
    register(_DummyForecaster)
    assert check_estimator(_DummyForecaster)[-1] == "save_load"

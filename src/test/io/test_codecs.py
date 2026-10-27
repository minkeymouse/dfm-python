"""JSON, npz and weights codecs: plain values only, no pickle."""

import io
import json
import zipfile

import numpy as np
import pytest
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from dfm_python.io import codecs
from dfm_python.io.codecs import (
    decode_params,
    encode_params,
    from_jsonable,
    is_encoded_estimator,
    read_json,
    read_json_plain,
    read_npz,
    read_weights,
    to_jsonable,
    write_json,
    write_npz,
    write_weights,
)
from dfm_python.utils.errors import ConfigurationError

from ._estimators import DoublingScaler

# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_json_sorts_keys_and_converts_numpy(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {
        "b": np.float64(0.1),
        "a": np.int64(3),
        "c": np.bool_(True),
        "d": np.arange(3),
        "e": (1, 2),
        "f": {"z": [np.float32(0.5), None, "s"]},
        "g": np.array(["u", "v"], dtype=object),
        "h": np.str_("name"),
    })
    text = path.read_text()
    assert list(json.loads(text)) == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert text.index('"a"') < text.index('"b"')
    data = read_json(path)
    assert data == {"a": 3, "b": 0.1, "c": True, "d": [0, 1, 2], "e": [1, 2], "f": {"z": [0.5, None, "s"]},
                    "g": ["u", "v"], "h": "name"}
    assert type(data["a"]) is int and type(data["b"]) is float and type(data["c"]) is bool


def test_json_keeps_floats_exact_and_non_finite(tmp_path):
    values = [0.1 + 0.2, 1e-300, -0.0, float("inf"), float("-inf")]
    write_json(tmp_path / "x.json", {"v": values, "nan": float("nan")})
    back = read_json(tmp_path / "x.json")
    assert [v.hex() for v in back["v"]] == [v.hex() for v in values]
    assert np.isnan(back["nan"])


@pytest.mark.parametrize("bad", [object(), {1: "x"}, np.array([1 + 2j]), np.datetime64("2020-01-01"), b"bytes"])
def test_json_refuses_other_types(tmp_path, bad):
    with pytest.raises(TypeError, match="x.json"):
        write_json(tmp_path / "x.json", {"value": bad})
    assert not (tmp_path / "x.json").exists()


def test_estimators_are_written_as_class_and_params(tmp_path):
    scaler = RobustScaler(quantile_range=(10.0, 90.0), with_centering=False)
    write_json(tmp_path / "x.json", {"scaler": scaler})
    raw = json.loads((tmp_path / "x.json").read_text())
    assert raw["scaler"]["class"] == "sklearn.preprocessing._data.RobustScaler"
    assert raw["scaler"]["params"]["quantile_range"] == [10.0, 90.0]
    assert set(raw["scaler"]) == {"class", "params"}


@pytest.mark.parametrize(
    "estimator",
    [StandardScaler(with_mean=False), RobustScaler(quantile_range=(10.0, 90.0)), MinMaxScaler(feature_range=(-1, 1))],
    ids=lambda e: type(e).__name__,
)
def test_preprocessing_estimators_are_rebuilt(tmp_path, estimator):
    write_json(tmp_path / "x.json", {"scaler": estimator, "nested": [estimator]})
    back = read_json(tmp_path / "x.json")
    for rebuilt in (back["scaler"], back["nested"][0]):
        assert type(rebuilt) is type(estimator)
        assert rebuilt.get_params() == estimator.get_params()  # tuples restored, not lists


def test_other_estimators_are_rebuilt_only_with_allow_import(tmp_path):
    write_json(tmp_path / "x.json", {"model": Ridge(alpha=0.5), "custom": DoublingScaler(factor=3.0)})
    back = read_json(tmp_path / "x.json")
    assert back["model"] == {"class": "sklearn.linear_model._ridge.Ridge", "params": Ridge(alpha=0.5).get_params()}
    assert is_encoded_estimator(back["model"]) and is_encoded_estimator(back["custom"])
    back = read_json(tmp_path / "x.json", allow_import=True)
    assert isinstance(back["model"], Ridge) and back["model"].alpha == 0.5
    assert isinstance(back["custom"], DoublingScaler) and back["custom"].factor == 3.0


def test_preprocessing_prefix_cannot_reach_other_classes():
    for name in ("sklearn.preprocessing._data.BaseEstimator", "sklearn.preprocessing._data.np"):
        encoded = {"class": name, "params": {}}
        assert from_jsonable(encoded) == encoded  # not a public estimator of sklearn.preprocessing
    with pytest.raises(ConfigurationError, match="not a scikit-learn estimator"):
        from_jsonable({"class": "sklearn.preprocessing._data.scale", "params": {}})  # public, but a function


@pytest.mark.parametrize(
    "name",
    [
        "sklearn.preprocessing.tests.test_data.StandardScaler",  # public subpackage: would import its module
        "sklearn.preprocessing._data.os.path.StandardScaler",
        "sklearn.preprocessing..StandardScaler",
        "sklearn.preprocessing._data.StandardScaler()",
        "sklearn.preprocessingX.StandardScaler",
    ],
)
def test_preprocessing_names_outside_the_public_form_are_not_imported(name, monkeypatch):
    import sys

    for module in [m for m in sys.modules if m.startswith("sklearn.preprocessing.tests")]:
        monkeypatch.delitem(sys.modules, module)
    encoded = {"class": name, "params": {}}
    assert from_jsonable(encoded) == encoded
    assert not any(m.startswith("sklearn.preprocessing.tests") for m in sys.modules)


@pytest.mark.parametrize(
    "name", ["sklearn.preprocessing.StandardScaler", "sklearn.preprocessing._data.StandardScaler"]
)
def test_public_preprocessing_names_are_rebuilt(name):
    rebuilt = from_jsonable({"class": name, "params": {"with_mean": False}})
    assert type(rebuilt) is StandardScaler and rebuilt.with_mean is False


@pytest.mark.parametrize("name", ["os.system", "json.dumps"])
def test_allow_import_still_requires_an_estimator_class(name):
    with pytest.raises(ConfigurationError, match="not a scikit-learn estimator"):
        from_jsonable({"class": name, "params": {}}, allow_import=True)


def test_unimportable_class_raises():
    with pytest.raises(ConfigurationError, match="Cannot import"):
        from_jsonable({"class": "no_such_package_xyz.Thing", "params": {}}, allow_import=True)


def test_plain_dicts_with_other_keys_are_untouched():
    value = {"class": "sklearn.preprocessing.StandardScaler", "params": {}, "extra": 1}
    assert from_jsonable(value) == value
    assert to_jsonable(value) == value


# ---------------------------------------------------------------------------
# npz
# ---------------------------------------------------------------------------


def test_npz_round_trip_keeps_keys_order_dtypes_and_bits(tmp_path):
    arrays = {
        "ssm/A": np.arange(6, dtype=np.float64).reshape(2, 3) / 7,
        "ssm/Q": np.eye(2, dtype=np.float32),
        "data/mask": np.array([True, False]),
        "scaler/n_samples_seen_": np.asarray(np.int64(12)),
        "file": np.array([1, 2], dtype=np.int16),  # names np.savez reserves
        "allow_pickle": np.array(["a", "bc"]),
        "nan": np.array([np.nan, -np.nan, np.inf]),
        "fortran": np.asfortranarray(np.arange(6.0).reshape(2, 3)),
    }
    write_npz(tmp_path / "state.npz", arrays)
    back = read_npz(tmp_path / "state.npz")
    assert list(back) == list(arrays)
    for key, value in arrays.items():
        assert back[key].dtype == value.dtype and back[key].shape == value.shape, key
        assert np.ascontiguousarray(back[key]).tobytes() == np.ascontiguousarray(value).tobytes(), key
    assert back["scaler/n_samples_seen_"].ndim == 0
    with zipfile.ZipFile(tmp_path / "state.npz") as zf:
        assert sorted(zf.namelist()) == sorted(f"{k}.npy" for k in arrays)


def test_npz_is_readable_by_numpy_load(tmp_path):
    write_npz(tmp_path / "state.npz", {"ssm/A": np.eye(2)})
    with np.load(tmp_path / "state.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ssm/A"], np.eye(2))


@pytest.mark.parametrize("bad", [np.array([{"a": 1}], dtype=object), np.array(["x", None], dtype=object)])
def test_npz_refuses_object_arrays(tmp_path, bad):
    with pytest.raises(TypeError, match="dtype object"):
        write_npz(tmp_path / "state.npz", {"ok": np.zeros(2), "names": bad})
    assert not (tmp_path / "state.npz").exists()


@pytest.mark.parametrize("key", ["", "/abs", 3])
def test_npz_refuses_bad_keys(tmp_path, key):
    with pytest.raises(TypeError, match="npz keys"):
        write_npz(tmp_path / "state.npz", {key: np.zeros(1)})


def test_read_npz_never_unpickles(tmp_path):
    buffer = io.BytesIO()
    np.savez(buffer, names=np.array(["x", None], dtype=object))  # numpy pickles object arrays
    (tmp_path / "evil.npz").write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="allow_pickle"):
        read_npz(tmp_path / "evil.npz")


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_weights_round_trip_with_map_location(tmp_path):
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    write_weights(tmp_path / "weights.pt", module.state_dict())
    state = read_weights(tmp_path / "weights.pt", map_location="cpu")
    assert list(state) == list(module.state_dict())
    for key, value in module.state_dict().items():
        assert torch.equal(state[key], value) and state[key].device.type == "cpu"


def test_read_weights_refuses_pickled_objects(tmp_path):
    torch.save({"weight": torch.zeros(1), "obj": DoublingScaler()}, tmp_path / "weights.pt")
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        read_weights(tmp_path / "weights.pt")


# ---------------------------------------------------------------------------
# Component specs in format-2 params
# ---------------------------------------------------------------------------


class _Spec(BaseEstimator):
    def __init__(self, *, width=32, layers=(32,), inner=None):
        self.width = width
        self.layers = layers
        self.inner = inner


class _Model:
    def __init__(self, *, factors=None, sizes=(8, 4), blocks=None):
        self.factors = factors
        self.sizes = sizes
        self.blocks = blocks


@pytest.fixture
def specs(monkeypatch):
    monkeypatch.setattr(codecs, "_SPECS", dict(codecs._SPECS))
    return codecs.register_spec(_Spec)


def test_specs_round_trip_through_json(specs):
    params = {"factors": _Spec(width=64, layers=(16, 8), inner=_Spec(layers=())), "sizes": (3, 2),
              "blocks": {"g": ["a", "b"]}}
    encoded = json.loads(json.dumps(encode_params(params)))
    assert encoded["factors"] == {
        "spec": "_Spec",
        "params": {"inner": {"spec": "_Spec", "params": {"inner": None, "layers": [], "width": 32}},
                   "layers": [16, 8], "width": 64},
    }
    decoded = decode_params(encoded, _Model)
    assert isinstance(decoded["factors"], _Spec) and decoded["factors"].layers == (16, 8)
    assert decoded["factors"].inner.layers == () and decoded["sizes"] == (3, 2)
    assert decoded["blocks"] == {"g": ["a", "b"]}  # lists stay lists where the default is not a tuple


def test_unregistered_estimators_are_refused(specs):
    with pytest.raises(TypeError, match="not an allowed spec class"):
        encode_params({"scaler": StandardScaler()})
    with pytest.raises(ConfigurationError, match=r"names the spec 'Ridge', which is not an allowed spec class"):
        decode_params({"model": {"spec": "Ridge", "params": {}}})
    with pytest.raises(ConfigurationError, match="not a mapping"):
        decode_params({"factors": {"spec": "_Spec", "params": [1]}})
    with pytest.raises(TypeError, match="string keys"):
        encode_params({"blocks": {1: ["a"]}})


def test_register_spec_checks_its_argument(specs):
    assert codecs.register_spec(_Spec) is _Spec  # registering the same class again is fine
    other = type("_Spec", (BaseEstimator,), {})
    with pytest.raises(ValueError, match="already registered"):
        codecs.register_spec(other)
    with pytest.raises(TypeError, match="BaseEstimator"):
        codecs.register_spec(dict)


def test_read_json_plain_keeps_estimator_dicts(tmp_path):
    path = tmp_path / "x.json"
    write_json(path, {"s": StandardScaler()})
    assert read_json_plain(path)["s"] == {"class": "sklearn.preprocessing._data.StandardScaler",
                                          "params": {"copy": True, "with_mean": True, "with_std": True}}

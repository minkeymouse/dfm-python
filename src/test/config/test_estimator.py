"""``dfm_python.config.estimator``: EstimatorConfig (strict keys, dict/YAML round-trips) and load_yaml."""

import json
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from dfm_python.config import EstimatorConfig, load_yaml
from dfm_python.utils.errors import ConfigurationError


@dataclass
class _Cfg(EstimatorConfig):
    n_factors: int = 2
    tol: float = 1e-5
    layers: tuple = (16,)
    widths: Optional[tuple[int, ...]] = None
    blocks: Optional[dict] = None
    names: list = field(default_factory=list)
    scaler: Optional[str] = "standard"


@dataclass
class _Required(EstimatorConfig):
    window: int
    stride: Optional[int] = None


def _without(monkeypatch, *modules):
    for name in modules:
        monkeypatch.setitem(sys.modules, name, None)


# -- dict ---------------------------------------------------------------------


def test_to_dict_in_field_order_without_copies():
    blocks = {"g": {"n_factors": 1}}
    cfg = _Cfg(blocks=blocks)
    d = cfg.to_dict()
    assert list(d) == ["n_factors", "tol", "layers", "widths", "blocks", "names", "scaler"]
    assert d["blocks"] is blocks
    assert _Cfg.field_names() == tuple(d)


def test_from_dict_round_trip():
    cfg = _Cfg(n_factors=3, layers=(8, 4), widths=(1, 2), blocks={"g": {"n": 1}}, names=["a"])
    assert _Cfg.from_dict(cfg.to_dict()) == cfg


def test_from_dict_unknown_key_names_closest_field():
    with pytest.raises(ConfigurationError, match=r"'n_factor' \(did you mean 'n_factors'\?\)") as info:
        _Cfg.from_dict({"n_factor": 3})
    assert isinstance(info.value, ValueError)


def test_from_dict_old_names_are_unknown():
    with pytest.raises(ConfigurationError, match="'num_factors'"):
        _Cfg.from_dict({"num_factors": 3})


def test_from_dict_reports_every_unknown_key_and_lists_fields():
    with pytest.raises(ConfigurationError) as info:
        _Cfg.from_dict({"zzz": 1, "tolerance": 1e-3})
    message = str(info.value)
    assert "'zzz'" in message and "'tolerance'" in message and "Valid keys: n_factors, tol" in message


def test_from_dict_rejects_non_mappings():
    with pytest.raises(ConfigurationError, match="mapping"):
        _Cfg.from_dict([("n_factors", 3)])


def test_from_dict_missing_required_field():
    with pytest.raises(ConfigurationError, match="'window'"):
        _Required.from_dict({"stride": 2})
    assert _Required.from_dict({"window": 5}) == _Required(window=5)


def test_from_dict_turns_lists_into_tuples_for_tuple_fields():
    cfg = _Cfg.from_dict({"layers": [8, 4], "widths": [3], "names": ["a", "b"]})
    assert cfg.layers == (8, 4) and cfg.widths == (3,)
    assert cfg.names == ["a", "b"]  # list-typed field stays a list


def test_to_plain_dict_and_json_round_trip():
    cfg = _Cfg(n_factors=np.int64(4), tol=np.float64(1e-6), layers=(8,), blocks={"g": {"series": ("a", "b")}})
    plain = cfg.to_plain_dict()
    assert plain["layers"] == [8] and plain["blocks"] == {"g": {"series": ["a", "b"]}}
    assert type(plain["n_factors"]) is int and type(plain["tol"]) is float
    back = _Cfg.from_dict(json.loads(json.dumps(plain)))
    assert back.layers == (8,) and back.n_factors == 4 and back.tol == 1e-6


def test_to_plain_dict_rejects_objects():
    with pytest.raises(ConfigurationError, match="'scaler'.*object"):
        _Cfg(scaler=object()).to_plain_dict()
    with pytest.raises(ConfigurationError, match="non-string key"):
        _Cfg(blocks={1: "a"}).to_plain_dict()


# -- YAML -----------------------------------------------------------------------


def test_yaml_round_trip(tmp_path):
    cfg = _Cfg(n_factors=3, tol=1e-7, layers=(8, 4), blocks={"g": {"n": 1}}, names=["x"], scaler=None)
    path = tmp_path / "model.yaml"
    cfg.to_yaml(path)
    assert path.read_text().startswith("model:\n")
    assert _Cfg.from_yaml(path) == cfg
    sections = load_yaml(path)
    assert sections["data"] == {} and sections["model"]["layers"] == [8, 4]


def test_load_yaml_returns_both_sections(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "model:\n  n_factors: 4\n  tol: 1e-5\ndata:\n  time_col: date\n  exog: [oil]\n  freq: {gdp: q}\n"
    )
    sections = load_yaml(path)
    assert sections == {
        "model": {"n_factors": 4, "tol": 1e-5},
        "data": {"time_col": "date", "exog": ["oil"], "freq": {"gdp": "q"}},
    }
    assert _Cfg.from_yaml(path) == _Cfg(n_factors=4, tol=1e-5)


def test_flat_yaml_is_rejected(tmp_path):
    path = tmp_path / "flat.yaml"
    path.write_text("n_factors: 3\n")
    with pytest.raises(ConfigurationError, match="no top-level 'model:' section"):
        load_yaml(path)
    with pytest.raises(ConfigurationError, match="model"):
        _Cfg.from_yaml(path)


def test_unknown_key_in_yaml_model_section(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("model:\n  seed: 3\n")
    with pytest.raises(ConfigurationError, match="'seed'"):
        _Cfg.from_yaml(path)


def test_unknown_top_level_section(tmp_path):
    path = tmp_path / "extra.yaml"
    path.write_text("model: {}\nmodle: {}\n")
    with pytest.raises(ConfigurationError, match=r"'modle' \(did you mean 'model'\?\)"):
        load_yaml(path)


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "model: 3\n", "model: {}\ndata: [1]\n"])
def test_non_mapping_content_is_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigurationError, match="mapping"):
        load_yaml(path)


def test_empty_sections_are_empty_dicts(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("model:\n")
    assert load_yaml(path) == {"model": {}, "data": {}}
    assert _Cfg.from_yaml(path) == _Cfg()


def test_empty_file_is_flat(tmp_path):
    path = tmp_path / "nothing.yaml"
    path.write_text("")
    with pytest.raises(ConfigurationError, match="'model:'"):
        load_yaml(path)


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        load_yaml(tmp_path / "absent.yaml")


def test_omegaconf_interpolation_is_resolved(tmp_path):
    pytest.importorskip("omegaconf")
    path = tmp_path / "interp.yaml"
    path.write_text("model:\n  n_factors: 2\n  tol: ${model.n_factors}\n")
    assert load_yaml(path)["model"]["tol"] == 2


def test_pyyaml_fallback_reads_exponent_floats(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    _without(monkeypatch, "omegaconf")
    path = tmp_path / "floats.yaml"
    path.write_text("model:\n  tol: 1e-5\n  n_factors: 3\n  scaler: '1e-5'\n")
    model = load_yaml(path)["model"]
    assert model == {"tol": 1e-5, "n_factors": 3, "scaler": "1e-5"}
    assert isinstance(model["tol"], float)


def test_backends_agree(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    path = tmp_path / "same.yaml"
    path.write_text("model:\n  tol: 1e-5\n  layers: [8, 4]\n  scaler: null\n  flag: true\n  rate: .5\n")
    with_omegaconf = load_yaml(path)
    _without(monkeypatch, "omegaconf")
    assert load_yaml(path) == with_omegaconf


def test_write_with_omegaconf_only(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    import omegaconf  # noqa: F401  (imported before PyYAML is hidden)

    _without(monkeypatch, "yaml")
    path = tmp_path / "oc.yaml"
    _Cfg(n_factors=5, layers=(2,)).to_yaml(path)
    assert _Cfg.from_yaml(path) == _Cfg(n_factors=5, layers=(2,))


def test_no_yaml_backend_names_the_extra(tmp_path, monkeypatch):
    _without(monkeypatch, "omegaconf", "yaml")
    path = tmp_path / "x.yaml"
    path.write_text("model: {}\n")
    with pytest.raises(ImportError, match=r"dfm-python\[yaml\]"):
        load_yaml(path)
    with pytest.raises(ImportError, match=r"dfm-python\[yaml\]"):
        _Cfg().to_yaml(path)

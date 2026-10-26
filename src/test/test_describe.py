"""``describe()`` on every model and ``json_schema()`` on the public config classes."""

import copy
import inspect
import json
import warnings

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import dfm_python
from dfm_python import AFM, DDFM, DFM, AFMConfig, DDFMConfig, DFMConfig, iVDFM, iVDFMConfig
from dfm_python.config.constants import VALID_FREQUENCIES
from dfm_python.registry import all_estimators
from dfm_python.testing import make_panel
from dfm_python.utils.errors import ConfigurationError

ESTIMATORS = [cls for _, cls in all_estimators()]
CONFIGS = {DFM: DFMConfig, DDFM: DDFMConfig, iVDFM: iVDFMConfig, AFM: AFMConfig}
BASE_KEYS = {"model", "version", "params", "tags", "fitted"}
FITTED_KEYS = {"n_features_in", "feature_names", "index", "n_iter", "converged"}
OWN_KEYS = {DFM: {"loglik"}, AFM: {"sharpe"}, DDFM: set(), iVDFM: set()}
# The keywords json_schema() uses in a property, which _conforms checks.
PROPERTY_KEYWORDS = {"type", "items", "additionalProperties", "default", "enum", "description"}


def _round_trip(d: dict) -> None:
    """``d`` survives strict JSON (no NaN) unchanged."""
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def _param_sets(cls) -> list[dict]:
    """``get_params()`` of the defaults and of every ``get_test_params`` set."""
    return [cls().get_params(deep=False)] + [cls(**params).get_params(deep=False) for params in cls.get_test_params()]


def _conforms(value, schema: dict) -> bool:
    """Whether ``value`` (plain JSON) satisfies the ``type``, ``enum``, ``items`` and ``additionalProperties`` of ``schema``."""
    python_types = {"integer": int, "number": (int, float), "string": str, "boolean": bool, "null": type(None),
                    "array": list, "object": dict}
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not any(isinstance(value, python_types[t]) and (t not in ("integer", "number") or is_number) for t in types):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, list) and "items" in schema:
        return all(_conforms(v, schema["items"]) for v in value)
    if isinstance(value, dict) and "additionalProperties" in schema:
        return all(_conforms(v, schema["additionalProperties"]) for v in value.values())
    return True


@pytest.fixture(scope="module")
def fitted() -> dict:
    """Each model fitted once on its fixture panel with its first test parameter set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {cls: cls(**cls.get_test_params()[0]).fit(make_panel(cls)) for cls in ESTIMATORS}


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_describe_unfitted(cls):
    est = cls(**cls.get_test_params()[0])
    d = est.describe()
    _round_trip(d)
    assert set(d) == BASE_KEYS
    assert d["model"] == cls.__name__ and d["version"] == dfm_python.__version__ and d["fitted"] is False
    assert set(d["params"]) == set(est.get_params())
    assert d["tags"] == est.get_tags()


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_describe_fitted(cls, fitted):
    est = fitted[cls]
    d = est.describe()
    _round_trip(d)
    interpret = cls.get_tags()["capability:interpret"]
    expected = BASE_KEYS | FITTED_KEYS | OWN_KEYS[cls]
    expected |= {"factor_names", "explained_variance"} if interpret else set()
    expected |= {"stop_reason"} if hasattr(est, "stop_reason_") else set()
    assert set(d) == expected
    assert d["fitted"] is True
    assert d["n_features_in"] == 3 and d["feature_names"] == ["s0", "s1", "s2"]
    index = est.index_
    assert d["index"] == {"start": str(index[0]), "end": str(index[-1]), "length": 48, "freq": index.freqstr}
    assert d["n_iter"] == est.n_iter_ and d["converged"] is est.converged_
    if interpret:
        assert d["factor_names"] == list(est.factor_names_)
        assert list(d["explained_variance"]) == d["feature_names"]
        assert all(v is None or isinstance(v, float) for v in d["explained_variance"].values())
    if cls is DFM:
        assert d["loglik"] == est.loglik_
    if cls is AFM:
        assert d["sharpe"] == est.result_.sharpe


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_describe_survives_save_and_load(cls, fitted, tmp_path):
    est = fitted[cls]
    est.save(tmp_path / "run")
    assert dfm_python.load(tmp_path / "run").describe() == est.describe()


def test_describe_makes_params_json_safe():
    dfm = DFM(n_factors=np.int64(2), scaler=StandardScaler(), tent_weights={"q:m": np.array([1.0, 2.0, 1.0])})
    params = dfm.describe()["params"]
    assert params["scaler"] == "StandardScaler"
    assert params["n_factors"] == 2 and type(params["n_factors"]) is int
    assert params["tent_weights"] == {"q:m": [1.0, 2.0, 1.0]}
    torch = pytest.importorskip("torch")
    ddfm = DDFM(encoder_layers=(4, 2), device=torch.device("cpu"))
    params = ddfm.describe()["params"]
    assert params["encoder_layers"] == [4, 2] and params["device"] == "cpu"
    _round_trip(dfm.describe())
    _round_trip(ddfm.describe())


def test_describe_turns_non_finite_floats_into_none(fitted):
    est = copy.deepcopy(fitted[DFM])
    est.loglik_ = float("nan")
    est.explained_variance_ = est.explained_variance_.copy()
    est.explained_variance_.iloc[0] = np.nan
    est.explained_variance_.iloc[1] = np.inf
    d = est.describe()
    _round_trip(d)
    assert d["loglik"] is None
    assert d["explained_variance"]["s0"] is None and d["explained_variance"]["s1"] is None
    assert isinstance(d["explained_variance"]["s2"], float)


def test_describe_omits_missing_attributes(fitted):
    est = copy.deepcopy(fitted[iVDFM])
    del est.explained_variance_  # as in a run saved by 0.8.x
    d = est.describe()
    assert "explained_variance" not in d and "factor_names" in d


def test_describe_range_index():
    est = DFM(max_iter=2).fit(np.random.default_rng(0).standard_normal((30, 3)))
    assert est.describe()["index"] == {"start": 0, "end": 29, "length": 30, "freq": None}


# ---------------------------------------------------------------------------
# json_schema()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_json_schema_has_one_property_per_parameter(cls):
    schema = CONFIGS[cls].json_schema()
    assert json.loads(json.dumps(schema, allow_nan=False)) == schema
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == cls.__name__ and schema["description"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    names = list(inspect.signature(cls.__init__).parameters)[1:]
    assert list(schema["properties"]) == names
    assert set(schema["properties"]) == set(cls().get_params())
    defaults = CONFIGS[cls]().to_plain_dict()
    for name, prop in schema["properties"].items():
        assert set(prop) <= PROPERTY_KEYWORDS, name
        assert prop["default"] == defaults[name], name
        assert prop["description"].endswith(".") and "\n" not in prop["description"], name


def test_json_schema_types_enums_and_descriptions():
    dfm = DFMConfig.json_schema()["properties"]
    assert dfm["n_factors"]["type"] == ["integer", "null"]
    assert dfm["tol"]["type"] == "number" and dfm["max_iter"]["type"] == "integer"
    assert dfm["tent_weights"]["additionalProperties"] == {"type": "array", "items": {"type": "number"}}
    assert dfm["clock"]["enum"] == sorted(VALID_FREQUENCIES)
    assert dfm["em_acceleration"] == {
        "type": ["string", "null"], "default": "squarem", "enum": ["squarem", None],
        "description": dfm["em_acceleration"]["description"],
    }
    assert "type" not in dfm["scaler"] and dfm["scaler"]["enum"][-1] is None
    assert set(dfm["scaler"]["enum"][:-1]) == {"standard", "robust", "minmax", "maxabs", "quantile"}
    assert dfm["max_iter"]["description"] == "Maximum number of EM iterations (E-steps, each one EM map)."
    ddfm = DDFMConfig.json_schema()["properties"]
    assert ddfm["encoder_layers"] == {
        "type": "array", "items": {"type": "integer"}, "default": [16],
        "description": "Widths of the encoder's hidden layers.",
    }
    assert ddfm["optimizer"]["enum"] == ["Adam", "AdamW", "SGD"]
    ivdfm = iVDFMConfig.json_schema()["properties"]
    assert ivdfm["encoder_hidden_dim"]["type"] == ["integer", "array"]
    assert ivdfm["encoder_hidden_dim"]["items"] == {"type": "integer"}
    # A docstring entry naming several parameters describes each of them.
    assert ivdfm["encoder_hidden_dim"]["description"] == ivdfm["prior_hidden_dim"]["description"]
    assert ivdfm["use_layer_norm"]["type"] == "boolean"
    assert ivdfm["scheduler_type"]["enum"][-1] is None
    afm = AFMConfig.json_schema()["properties"]
    assert afm["factor_model"]["enum"] == ["attention", "pca"] and afm["trading"]["enum"] == ["longconv", "ou"]
    assert afm["n_factors"]["description"] == "Number of factors K."


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_json_schema_enums_match_validation(cls):
    base = {"n_factors": 2}  # valid for every model (AFM requires it)
    enums = {name: prop["enum"] for name, prop in CONFIGS[cls].json_schema()["properties"].items() if "enum" in prop}
    assert set(enums) == set(cls._param_choices)
    for name, values in enums.items():
        for value in values:
            cls(**{**base, name: value})._validate_params()
        with pytest.raises(ConfigurationError, match=name):
            cls(**{**base, name: "__not_a_choice__"})._validate_params()


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_params_conform_to_json_schema(cls):
    properties = CONFIGS[cls].json_schema()["properties"]
    for params in _param_sets(cls):
        plain = CONFIGS[cls](**params).to_plain_dict()
        for name, value in plain.items():
            assert _conforms(value, properties[name]), (name, value)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda cls: cls.__name__)
def test_json_schema_validates_with_jsonschema(cls):
    jsonschema = pytest.importorskip("jsonschema")
    schema = CONFIGS[cls].json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for params in _param_sets(cls):
        plain = CONFIGS[cls](**params).to_plain_dict()
        validator.validate(plain)
        validator.validate(cls(**params).describe()["params"])
    plain = CONFIGS[cls]().to_plain_dict()
    assert not validator.is_valid({**plain, "n_factor": 2})
    name = next(iter(cls._param_choices))
    assert not validator.is_valid({**plain, name: "__not_a_choice__"})

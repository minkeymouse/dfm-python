"""Robustness of the shared helpers: config tuple fields, YAML read errors, the constrained OLS and the harness's equality."""

import dataclasses
import importlib.util

import numpy as np
import pytest

import dfm_python.numeric.estimator as numeric_estimator
from dfm_python.config.estimator import EstimatorConfig, load_yaml
from dfm_python.testing import _deep_equal
from dfm_python.utils.errors import ConfigurationError

_NEEDS_OMEGACONF = pytest.mark.skipif(importlib.util.find_spec("omegaconf") is None, reason="needs omegaconf")


def _raising(exc: type):
    def raise_(*args, **kwargs):
        raise exc("boom")

    return raise_


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


def test_constrained_ols_ridge_falls_back_on_eigenvalue_failures(monkeypatch):
    X = np.zeros((10, 3))  # every eigenvalue of X'X is 0, so the condition number is undefined
    y = np.random.default_rng(0).standard_normal(10)
    R, q = np.array([[1.0, -1.0, 0.0]]), np.zeros(1)
    beta = numeric_estimator.estimate_constrained_ols(y, X, R, q)
    assert beta.shape == (3,) and np.isfinite(beta).all()
    monkeypatch.setattr(np.linalg, "eigvalsh", _raising(TypeError))
    with pytest.raises(TypeError, match="boom"):
        numeric_estimator.estimate_constrained_ols(y, X, R, q)


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

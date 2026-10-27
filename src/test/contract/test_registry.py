"""``dfm_python.registry``: the estimators built on ``BaseFactorModel`` that the contract tests run."""

import dfm_python
from dfm_python.base import BaseFactorModel
from dfm_python.registry import all_estimators


def test_registry_lists_estimator_classes():
    estimators = all_estimators()
    names = [name for name, _ in estimators]
    assert names == sorted(names) and len(set(names)) == len(names)
    for name, cls in estimators:
        assert isinstance(cls, type) and issubclass(cls, BaseFactorModel)
        assert cls.__name__ == name


def test_registry_lists_exactly_the_four_models():
    assert [name for name, _ in all_estimators()] == ["AFM", "DDFM", "DFM", "iVDFM"]
    for name, cls in all_estimators():
        assert getattr(dfm_python, name) is cls, name

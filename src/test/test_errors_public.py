"""The public exception module ``dfm_python.errors`` (spec section 11)."""

import pytest
from sklearn.exceptions import NotFittedError as SklearnNotFittedError

import dfm_python.errors as errors
import dfm_python.utils.errors as internal

PUBLIC = ["DFMError", "ConfigurationError", "DataValidationError", "NotFittedError", "NumericalError"]


def test_module_exports_exactly_the_five_classes():
    assert sorted(errors.__all__) == sorted(PUBLIC)
    for name in PUBLIC:
        cls = getattr(errors, name)
        assert isinstance(cls, type) and issubclass(cls, Exception)
        assert cls.__module__ == "dfm_python.errors"


@pytest.mark.parametrize(
    "name, bases",
    [
        ("DFMError", (Exception,)),
        ("ConfigurationError", (errors.DFMError, ValueError)),
        ("DataValidationError", (errors.DFMError, ValueError)),
        ("NotFittedError", (errors.DFMError, SklearnNotFittedError, ValueError, AttributeError)),
        ("NumericalError", (errors.DFMError, ArithmeticError)),
    ],
)
def test_hierarchy(name, bases):
    cls = getattr(errors, name)
    for base in bases:
        assert issubclass(cls, base), (name, base)


def test_not_fitted_error_is_caught_as_sklearn_and_attribute_error():
    for catch in (SklearnNotFittedError, AttributeError, ValueError, errors.DFMError):
        with pytest.raises(catch):
            raise errors.NotFittedError("not fitted")


def test_hasattr_is_false_on_a_not_fitted_error():
    class Model:
        @property
        def params_(self):
            raise errors.NotFittedError("Call fit first.")

    assert not hasattr(Model(), "params_")


def test_the_two_value_errors_are_distinct():
    assert not issubclass(errors.ConfigurationError, errors.DataValidationError)
    assert not issubclass(errors.DataValidationError, errors.ConfigurationError)
    with pytest.raises(ValueError):
        raise errors.DataValidationError("bad data")


def test_details_are_appended_to_the_message():
    err = errors.ConfigurationError("bad value", details="use 3")
    assert str(err) == "bad value\nDetails: use 3"
    assert err.message == "bad value" and err.details == "use 3"
    assert str(errors.DFMError("plain")) == "plain"


def test_internal_classes_are_built_on_the_public_ones():
    for name in PUBLIC:
        assert getattr(internal, name) is getattr(errors, name), name
    assert internal.ModelNotTrainedError is errors.NotFittedError
    assert issubclass(internal.UnsupportedInputError, errors.DataValidationError)
    assert issubclass(internal.ConfigValidationError, errors.ConfigurationError)
    assert issubclass(internal.NumericalStabilityError, errors.NumericalError)
    for name in ("ModelNotInitializedError", "DataError", "PredictionError"):
        assert issubclass(getattr(internal, name), errors.DFMError), name


def test_existing_raise_sites_are_caught_by_the_public_classes():
    with pytest.raises(errors.DataValidationError):
        raise internal.UnsupportedInputError("slot not accepted")
    with pytest.raises(errors.NotFittedError):
        raise internal.ModelNotTrainedError("engine not trained")
    with pytest.raises(errors.ConfigurationError):
        raise internal.ConfigValidationError("bad config")

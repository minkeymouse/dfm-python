"""The public exception classes of dfm-python.

Every error the library raises on purpose derives from :class:`DFMError`. The
four concrete classes also derive from the matching built-in (or
scikit-learn) exception, so callers can catch them the standard way:

=========================  =====================================================
Exception                  Bases
=========================  =====================================================
``DFMError``               ``Exception``
``ConfigurationError``     ``DFMError``, ``ValueError``
``DataValidationError``    ``DFMError``, ``ValueError``
``NotFittedError``         ``DFMError``, ``sklearn.exceptions.NotFittedError``
                           (so also ``ValueError`` and ``AttributeError``)
``NumericalError``         ``DFMError``, ``ArithmeticError``
=========================  =====================================================

- ``ConfigurationError``: an invalid hyperparameter or spec combination
  (checked at ``fit``); an invalid ``coverage``, ``end``, ``start``,
  ``series`` or ``period``; an unwanted ``exog``; ``set_params`` with an
  unknown name; a format-1 or inconsistent save directory.
- ``DataValidationError``: a problem with the data: names, frequencies, the
  index type, gaps or a clock mismatch, a slower series off its anchor
  stamps, infinite values, a constant or under-observed target, a slot the
  model does not accept, missing future exog.
- ``NotFittedError``: any fitted attribute or method before ``fit``.
- ``NumericalError``: a numerical failure (for example a covariance that is
  not positive definite, or non-finite values).

Wrong argument types raise ``TypeError``.
"""

from __future__ import annotations

from typing import Optional

from sklearn.exceptions import NotFittedError as _SklearnNotFittedError

__all__ = ["DFMError", "ConfigurationError", "DataValidationError", "NotFittedError", "NumericalError"]


class DFMError(Exception):
    """Base class of the dfm-python errors.

    ``details``, when given, is appended to ``str(err)`` as ``"\\nDetails: ..."``.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        if details:
            super().__init__(f"{message}\nDetails: {details}")
        else:
            super().__init__(message)


class ConfigurationError(DFMError, ValueError):
    """An invalid hyperparameter, argument value or saved-run configuration."""


class DataValidationError(DFMError, ValueError):
    """The data cannot be used as given (names, index, frequencies, values)."""


class NotFittedError(DFMError, _SklearnNotFittedError):
    """A fitted attribute or method was used before ``fit``.

    Subclasses :class:`sklearn.exceptions.NotFittedError`, so it is also a
    ``ValueError`` and an ``AttributeError``: ``hasattr(model, "params_")``
    is False before ``fit``.
    """


class NumericalError(DFMError, ArithmeticError):
    """A numerical failure: a matrix that is not positive definite, or non-finite values."""

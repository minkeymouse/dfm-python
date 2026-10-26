"""Hyperparameter search (needs the ``tune`` and ``deep`` extras: ``pip install 'dfm-python[tune,deep]'``).

:func:`tune_hyperparameters` searches the hyperparameters of an
:class:`~dfm_python.iVDFM` estimator with optuna and returns the best ones
under the estimator's names, for ``estimator.set_params(**best).fit(data)``.
Importing this module imports neither optuna nor torch; the functions that
need them raise an ``ImportError`` naming the extra when one is missing.
"""

from ._ivdfm import (
    compute_forecast_metric,
    get_optuna_subset,
    suggest_ivdfm_hyperparameters,
    suggest_ivdfm_hyperparameters_causal,
    tune_hyperparameters,
)

__all__ = [
    "tune_hyperparameters",
    "get_optuna_subset",
    "compute_forecast_metric",
    "suggest_ivdfm_hyperparameters",
    "suggest_ivdfm_hyperparameters_causal",
]

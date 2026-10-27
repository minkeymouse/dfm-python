"""Dynamic factor models in Python.

Models
------
- ``DFM``: linear, mixed-frequency dynamic factor model estimated by EM with
  Kalman filtering and smoothing. Latent factors evolve at a common clock
  frequency; lower-frequency series load on them through tent kernels.
- ``DDFM``: deep dynamic factor model with a nonlinear (PyTorch) encoder.
- ``iVDFM``: identifiable variational dynamic factor model.
- ``AFM``: attention factor model for cross-sectional returns.

``DFM`` needs numpy only. ``DDFM``, ``iVDFM`` and ``AFM`` train with torch,
the ``deep`` extra (``pip install "dfm-python[deep]"``); importing the package
and building any estimator does not import torch, and fitting or loading a
torch model without it raises ``ImportError`` naming the extra.

``AFM`` follows the estimator API: hyperparameters in the constructor, data in
one ``PanelData`` container, fitted state in attributes ending in ``_`` and a
pickle-free save directory::

    >>> from dfm_python import AFM, PanelData
    >>> data = PanelData.from_arrays(returns, characteristics=characteristics)  # (T, N), (T, N, M)
    >>> model = AFM(n_factors=3, hist_len=20, max_epochs=20, random_state=0).fit(data)
    >>> weights = model.allocate()               # DataFrame (T - hist_len) x N
    >>> model.result_.sharpe, model.book_returns_
    >>> model.save("runs/afm")
    >>> model = AFM.load("runs/afm")             # or dfm_python.load("runs/afm")

``iVDFM`` is an estimator on the same API: ``iVDFM(n_factors=3, window=200,
scaler="standard").fit(data)``, then ``predict(h, futr_exog=...)`` (data
units) and ``predict_factors(h)``. So is ``DDFM``: ``DDFM(n_factors=4,
encoder_layers=(16,), scaler="standard").fit(data)`` with encoder covariates
in ``exog``, then ``predict(h)`` and ``predict_factors(h)``. And ``DFM``:
``DFM(n_factors=1, clock="m", tent_weights={"q:m": [1, 2, 3, 2, 1]},
scaler="standard").fit(PanelData(y=df, freq={"gdp": "q"}))``, then
``predict(h)`` (series in the user's column order) and ``predict_factors(h)``;
see docs/models.md for complete examples of every model.

``dfm_python.evaluation.backtest(model, data, start=..., h=...)`` evaluates a
forecasting model walk-forward: it fits (or updates) a clone at each origin,
forecasts ``h`` steps and returns the forecasts beside the data with RMSE,
MAE, bias and interval coverage per horizon or series.
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("dfm-python")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

del PackageNotFoundError, _version

# Silent by default: the package logger gets only a NullHandler. Opt in with
# set_verbosity() or the DFM_PYTHON_VERBOSITY environment variable.
_package_logger = _logging.getLogger(__name__)
if not any(isinstance(h, _logging.NullHandler) for h in _package_logger.handlers):
    _package_logger.addHandler(_logging.NullHandler())
del _package_logger

from .config import DFMConfig, DDFMConfig, iVDFMConfig, AFMConfig
from .config import DFMResult, DDFMResult, BaseResult
from .utils.metric import calculate_rmse
from .utils.errors import NotFittedError, UnsupportedInputError
from .logger import set_verbosity, get_verbosity
from .logger.logger import _apply_env_verbosity
from .dataset.panel import PanelData
from .io.loading import load
from .base import BaseFactorModel
from .models.dfm import DFM
from .models.ddfm.estimator import DDFM
from .models.ivdfm.estimator import iVDFM
from .models.afm.estimator import AFM
from . import evaluation

__all__ = [
    # Models
    'BaseFactorModel',
    'DFM', 'DDFM', 'iVDFM', 'AFM',
    # Configs
    'DFMConfig', 'DDFMConfig', 'iVDFMConfig', 'AFMConfig',
    # Data container
    'PanelData',
    # Saved runs
    'load',
    # Results and metrics
    'BaseResult', 'DFMResult', 'DDFMResult', 'calculate_rmse',
    # Errors
    'NotFittedError', 'UnsupportedInputError',
    # Logging
    'set_verbosity', 'get_verbosity',
    # Walk-forward evaluation (the module)
    'evaluation',
]

_apply_env_verbosity()
del _apply_env_verbosity

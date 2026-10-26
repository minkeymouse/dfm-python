"""``iVDFM``: the identifiable variational dynamic factor model estimator (sklearn conventions, PanelData input).

A thin layer over the private engine
:class:`~dfm_python.models.ivdfm._engine.iVDFMEngine`: ``fit`` builds the
engine's sliding-window dataset from :class:`~dfm_python.PanelData` (targets
from ``y``, the context u_t from ``exog`` plus generated time features),
constructs the engine with every hyperparameter passed explicitly (the engine
seeds, then builds its networks) and trains it. ``predict(h)`` inverse-transforms
the engine's model-space forecast with ``scaler_``; ``update(data)`` encodes
new data with the fixed networks and appends the factors; ``predict(h, data=...)``
does the same on a copy. ``factors_``, ``loadings_``, ``explained_variance_``
and ``decompose()`` describe the training data through the encoded factor path
and its decoding. ``save``/``load`` use the pickle-free directory layout
(``state.npz``/``state.json``, ``result.npz``/``result.json``, ``weights.pt``).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ...base import BaseFactorModel
from ...config.constants import DEFAULT_TORCH_DTYPE, VALID_OPTIMIZERS
from ...config.schema.model import iVDFMConfig
from ...config.schema.params import iVDFMModelState
from ...config.schema.results import iVDFMResult
from ...config.types import to_numpy
from ...dataset.ivdfm_dataset import iVDFMDataset
from ...dataset.panel import PanelData, _polars_time_col, _to_pandas
from ...io.layout import index_from_spec, index_to_spec
from ...logger import get_logger
from ...logger.progress import fit_verbosity, quiet_engine_loggers
from ...utils.errors import ConfigurationError, DataValidationError, UnsupportedInputError
from ...utils.interpret import explained_variance
from ._engine import iVDFMEngine

__all__ = ["iVDFM"]

_logger = get_logger(__name__)

#: Scaler names accepted by the dataset, and the class each one builds.
_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
}
_INNOVATION_DISTRIBUTIONS = ("laplace", "gaussian", "student_t", "gamma", "beta", "exponential")
_DECODER_TYPES = ("linear", "residual", "mlp")
_ACTIVATIONS = ("relu", "lrelu", "tanh", "sigmoid")
_SCHEDULERS = ("step", "plateau", "cosine", "exponential", None)
_POSITIVE_INTS = (
    "n_factors", "time_context", "encoder_n_hidden_layers", "decoder_n_hidden_layers", "prior_n_hidden_layers",
    "factor_order", "n_regimes", "batch_size", "max_epochs",
)
_HIDDEN_DIMS = ("encoder_hidden_dim", "decoder_hidden_dim", "prior_hidden_dim")
# Loggers the engine (and the code it calls) writes progress to; raised to WARNING during fit with verbose=0.
_ENGINE_LOGGERS = (
    "dfm_python.models.ivdfm",
    "dfm_python.dataset.ivdfm_dataset",
    "dfm_python.numeric",
    "dfm_python.layer",
    "dfm_python.ssm",
)
# The engine's training reporter (iVDFMTrainLogger in logger/progress.py), which
# also logs its end-of-training summary ("did not converge after N epochs") as a
# warning; raised to ERROR during fit with verbose=0. converged_ carries the same information.
_PROGRESS_LOGGERS = ("dfm_python.logger.progress",)
# Name of the adapter's time column; a suffix is added if a series or exog column has this name.
_TIME_COLUMN = "__dfm_python_time__"
# result_ fields not written to result.npz/result.json, rebuilt on load: model_state_dict from
# weights.pt and config (asdict of the engine config) from the engine rebuilt from config.json.
_RESULT_REBUILT = ("model_state_dict", "config")
# Values of the engine and of the estimator that training sets, written to state.json.
_ENGINE_DIMS = ("data_dim", "context_dim", "time_context", "window")


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def _mean_decoder_jacobian(engine: iVDFMEngine, path: np.ndarray) -> np.ndarray:
    """The mean over the rows of ``path`` (T, r) of the decoder Jacobian, float64 (N, r).

    The path is decoded as the engine's ``get_result`` decodes it (``_decode``
    without context, so K > 1 mixes the decoders with uniform weights). The
    decoder acts on each time step separately, so the gradient of series i
    summed over time holds row i of every step's Jacobian.
    """
    f = torch.as_tensor(path, dtype=DEFAULT_TORCH_DTYPE, device=engine.device).unsqueeze(0).requires_grad_(True)
    with torch.enable_grad():
        y = engine._decode(f, u_1T=None)  # (1, T, N)
        rows = [torch.autograd.grad(y[..., i].sum(), f, retain_graph=True)[0][0].mean(dim=0)
                for i in range(y.shape[-1])]
    return to_numpy(torch.stack(rows)).astype(np.float64)


class _FittedScaler:
    """The fitted ``scaler_`` as the dataset's scaler: ``fit_transform`` only transforms.

    ``update`` and ``predict(data=...)`` build the dataset like ``fit`` does,
    so missing values are filled in data units before the targets are
    scaled, but the scaler is not refitted.
    """

    def __init__(self, scaler: Any) -> None:
        self.scaler = scaler

    def fit_transform(self, X: Any) -> Any:
        return self.scaler.transform(X)

    def transform(self, X: Any) -> Any:
        return self.scaler.transform(X)

    def inverse_transform(self, X: Any) -> Any:
        return self.scaler.inverse_transform(X)


class _RecordingDataset(iVDFMDataset):
    """The engine's dataset, recording the window index of every ``__getitem__`` call.

    On CPU the engine trains and extracts its final factors through a shuffled
    ``DataLoader``, so its factor arrays are in shuffled window order. The last
    ``len(dataset)`` recorded indices are the order of that final extraction.
    Recording reads no random numbers, so the engine's numbers are unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accessed: list[int] = []

    def __getitem__(self, idx: int):
        self.accessed.append(int(idx))
        return super().__getitem__(idx)


class iVDFM(BaseFactorModel):
    """Identifiable variational dynamic factor model.

    Latent factors follow AR(``factor_order``) dynamics driven by innovations
    whose conditionally independent prior given the context u_t makes the
    factors identifiable (iVAE conditions on the innovations). Training
    maximizes the ELBO on sliding windows of the panel. The context u_t is the
    ``exog`` columns of :class:`~dfm_python.PanelData` plus ``time_context``
    generated time features.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int, default 3
        Number of latent factors r.
    window : int, optional
        Length of the training windows. None uses the whole series as one
        window. A window longer than the series is clipped to its length.
    stride : int, optional
        Step between window starts. None means ``window // 2`` (at least 1).
    time_context : int, default 1
        Number of generated time features in u_t: the normalized time step,
        plus ``time_context - 1`` sine features. A ``DatetimeIndex`` (or a
        ``PeriodIndex``, through its start times) gives the time step; a
        ``RangeIndex`` gives the position.
    encoder_hidden_dim, decoder_hidden_dim, prior_hidden_dim : int or list of int
        Hidden widths of the innovation encoder (default 200), the decoder
        (default 200) and the prior network (default 100).
    encoder_n_hidden_layers, decoder_n_hidden_layers, prior_n_hidden_layers : int
        Number of hidden layers of the innovation encoder (default 2), the
        decoder (default 2) and the prior network (default 1).
    decoder_type : {"mlp", "residual", "linear"}, default "mlp"
        Decoder from the factors to the targets: an MLP, a linear map plus an
        MLP correction, or a linear map.
    activation : {"lrelu", "relu", "tanh", "sigmoid"}, default "lrelu"
        Activation of the hidden layers of every network.
    slope : float, default 0.1
        Leaky ReLU slope.
    factor_order : int, default 1
        AR order p of the factor dynamics. The AR coefficients are
        initialized by OLS on the PCA factors of the last training window;
        with ``factor_order > 1`` and ``n_regimes=1``, factors beyond the
        number of PCA factors (``n_factors`` above the number of targets or
        the window length) keep the default random initialization.
    innovation_distribution : str, default "laplace"
        One of "laplace", "gaussian", "student_t", "gamma", "beta",
        "exponential".
    decoder_var : float, default 0.01
        Observation noise variance of the decoder.
    n_regimes : int, default 1
        Number of regimes K; K > 1 adds regime-indexed priors, decoders and AR
        coefficients mixed by a regime network on u_t.
    regime_temperature : float, default 1.0
        Softmax temperature of the regime weights (> 0).
    beta_kl : float, default 1.0
        Weight of the KL term in the ELBO (>= 0).
    use_layer_norm : bool, default False
        Layer normalization in the encoder and decoder.
    mixing : bool, default False
        Global post-factor mixing f -> M f (identity initialized) before
        decoding.
    learning_rate : float, default 1e-3
        Learning rate of the optimizer.
    optimizer : {"Adam", "AdamW", "SGD"}, default "Adam"
        Optimizer of the networks' weights.
    optimizer_weight_decay : float, default 0.0
        Weight decay of the optimizer.
    optimizer_momentum : float, default 0.9
        Momentum of SGD.
    batch_size : int, default 32
        Windows per batch.
    max_epochs : int, default 100
        Maximum number of training epochs.
    tol : float, default 5e-4
        Training stops when the mean ELBO of the last five batches changes by
        less than ``tol``.
    patience : int, optional, default 20
        Early stopping: stop after this many epochs without a lower ELBO and
        restore the best epoch. None disables it.
    scheduler_type : {"step", "plateau", "cosine", "exponential", None}, default "step"
        Learning-rate scheduler; None disables it.
    scheduler_step_size : int, optional
        StepLR step; None means ``max_epochs // 3``.
    scheduler_gamma : float, default 0.5
        StepLR / ExponentialLR factor, in (0, 1].
    scheduler_patience : int, default 10
        ReduceLROnPlateau patience.
    scheduler_factor : float, default 0.1
        ReduceLROnPlateau factor, in (0, 1].
    scheduler_min_lr : float, default 0.0
        ReduceLROnPlateau minimum learning rate.
    scaler : str, sklearn transformer or None, default "standard"
        Scaler of the targets (not of exog): "standard", "robust", "minmax",
        "maxabs", "quantile", or an unfitted transformer, which is cloned. None
        trains on the data as given. Forecasts are always in data units.
    random_state : int, optional, default 3
        Seed of the global torch and numpy RNGs, applied just before the
        networks are built. None does not seed, so runs differ.
    device : str or torch.device, default "auto"
        "auto" uses CUDA if available, else CPU.
    verbose : int, default 0
        0 is silent; 1 logs the engine's progress (one record per epoch); 2
        also logs debug records.

    Attributes
    ----------
    result_ : iVDFMResult
        Factors, innovations, reconstructions and training diagnostics of the
        engine: ``Z`` is the factor path and ``x_sm`` its decoding.
    factor_names_ : numpy.ndarray of str
        ``factor_0`` .. ``factor_{r-1}``.
    factors_ : pandas.DataFrame
        The factor path: the training series encoded as one sequence with the
        trained networks (``result_.Z``), float64, indexed by ``index_``, one
        column per factor name. A run saved before 0.8.3 has no path; it holds
        the engine's average over the training windows instead, on the last
        ``window_`` stamps of ``index_``.
    window_factors_ : pandas.DataFrame
        The factors of every training window, float64, with a
        ``(window, time)`` MultiIndex in chronological order and one column
        per factor name.
    loadings_ : pandas.DataFrame
        The mean over the rows of ``factors_`` of the decoder Jacobian
        d common_t / d f_t, float64, in model (scaled) space, indexed by
        ``feature_names_in_``, one column per factor name: a local linear
        summary of the decoder, not exact loadings.
    loadings_kind_ : str
        ``"mean_jacobian"``.
    explained_variance_ : pandas.Series
        Per series, the share of the variance of the scaled training targets
        explained by the common component (the decoded factor path,
        ``result_.x_sm``): ``1 - var(y - common) / var(y)`` over the observed
        (finite) entries. Named ``"explained_variance"``, indexed by
        ``feature_names_in_``. Not set for a run saved by 0.8.x, which holds
        no training targets.
    network_ : torch.nn.Module
        The trained engine.
    device_ : torch.device
        The resolved device.
    scaler_ : sklearn transformer or None
        The target scaler fitted on the training data.
    n_features_in_, feature_names_in_, exog_names_in_, index_
        Number and names of the targets, the exog names and the training index.
    n_iter_ : int
        Training epochs (the best epoch when early stopping restored it).
    converged_ : bool
        True when training stopped before ``max_epochs`` (tolerance or
        patience).
    n_factors_ : int
        Number of factors.
    random_state_ : int or None
        The seed the run used.
    window_, stride_ : int
        The window length and stride used for training.
    time_context_ : int
        Number of generated time features in u_t.
    context_dim_ : int
        Width of u_t: ``time_context_`` plus the number of exog columns.
    conditioning_index_ : pandas.Index
        Time index of the data the forecasts start from (the training data,
        or the data of the last ``update``); ``predict(h)`` returns the ``h``
        stamps after its last entry.
    """

    _tags = {
        "task": "forecast",
        "backend": "torch",
        "capability:exog": True,
        "capability:missing_values": True,
        "capability:update": True,
        "capability:factor_forecast": True,
        "capability:device": True,
        "capability:interpret": True,
        "update_kind": "encoder_forward",
        "predict_data": "window",
        # The class default (random_state=3); an instance reports random_state is None.
        "non_deterministic": False,
    }
    _config_class = iVDFMConfig
    _param_choices = {
        "decoder_type": _DECODER_TYPES,
        "activation": _ACTIVATIONS,
        "innovation_distribution": _INNOVATION_DISTRIBUTIONS,
        "optimizer": VALID_OPTIMIZERS,
        "scheduler_type": _SCHEDULERS,
        "scaler": (*_SCALERS, None),
    }
    _interpolates_non_finite = True  # the dataset fills infinite values like missing ones

    def __init__(
        self,
        *,
        n_factors: int = 3,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        time_context: int = 1,
        encoder_hidden_dim: Any = 200,
        encoder_n_hidden_layers: int = 2,
        decoder_type: str = "mlp",
        decoder_hidden_dim: Any = 200,
        decoder_n_hidden_layers: int = 2,
        prior_hidden_dim: Any = 100,
        prior_n_hidden_layers: int = 1,
        activation: str = "lrelu",
        slope: float = 0.1,
        factor_order: int = 1,
        innovation_distribution: str = "laplace",
        decoder_var: float = 0.01,
        n_regimes: int = 1,
        regime_temperature: float = 1.0,
        beta_kl: float = 1.0,
        use_layer_norm: bool = False,
        mixing: bool = False,
        learning_rate: float = 1e-3,
        optimizer: str = "Adam",
        optimizer_weight_decay: float = 0.0,
        optimizer_momentum: float = 0.9,
        batch_size: int = 32,
        max_epochs: int = 100,
        tol: float = 5e-4,
        patience: Optional[int] = 20,
        scheduler_type: Optional[str] = "step",
        scheduler_step_size: Optional[int] = None,
        scheduler_gamma: float = 0.5,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.1,
        scheduler_min_lr: float = 0.0,
        scaler: Any = "standard",
        random_state: Optional[int] = 3,
        device: Any = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.window = window
        self.stride = stride
        self.time_context = time_context
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_hidden_layers = encoder_n_hidden_layers
        self.decoder_type = decoder_type
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_hidden_layers = decoder_n_hidden_layers
        self.prior_hidden_dim = prior_hidden_dim
        self.prior_n_hidden_layers = prior_n_hidden_layers
        self.activation = activation
        self.slope = slope
        self.factor_order = factor_order
        self.innovation_distribution = innovation_distribution
        self.decoder_var = decoder_var
        self.n_regimes = n_regimes
        self.regime_temperature = regime_temperature
        self.beta_kl = beta_kl
        self.use_layer_norm = use_layer_norm
        self.mixing = mixing
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_weight_decay = optimizer_weight_decay
        self.optimizer_momentum = optimizer_momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.scheduler_type = scheduler_type
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_min_lr = scheduler_min_lr
        self.scaler = scaler
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _instance_tags(self) -> dict:
        return {"non_deterministic": self.random_state is None}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small CPU parameter sets: one regime, and two regimes with AR(2) factors, mixing, a linear decoder and a scaler."""
        tiny = {
            "n_factors": 2, "window": 12, "encoder_hidden_dim": 8, "decoder_hidden_dim": 8, "prior_hidden_dim": 8,
            "encoder_n_hidden_layers": 1, "decoder_n_hidden_layers": 1, "prior_n_hidden_layers": 1,
            "batch_size": 4, "max_epochs": 2, "scaler": None, "random_state": 0, "device": "cpu",
        }
        return [
            tiny,
            {**tiny, "n_regimes": 2, "factor_order": 2, "time_context": 2, "mixing": True, "decoder_type": "linear",
             "scaler": "standard"},
        ]

    # -- validation ----------------------------------------------------------

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` naming the first invalid hyperparameter."""
        name = type(self).__name__

        def fail(message: str) -> None:
            raise ConfigurationError(f"{name}: {message}")

        for param in _POSITIVE_INTS:
            value = getattr(self, param)
            if not _is_int(value) or value < 1:
                fail(f"{param} must be a positive integer, got {value!r}.")
        for param in ("window", "stride", "patience", "scheduler_step_size"):
            value = getattr(self, param)
            if value is not None and (not _is_int(value) or value < 1):
                fail(f"{param} must be None or a positive integer, got {value!r}.")
        for param in _HIDDEN_DIMS:
            value = getattr(self, param)
            widths = list(value) if isinstance(value, (list, tuple)) else [value]
            if not widths or not all(_is_int(w) and w >= 1 for w in widths):
                fail(f"{param} must be a positive integer or a non-empty list of them, got {value!r}.")
        if self.decoder_type not in _DECODER_TYPES:
            fail(f"decoder_type must be one of {_DECODER_TYPES}, got {self.decoder_type!r}.")
        if self.activation not in _ACTIVATIONS:
            fail(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}.")
        if self.innovation_distribution not in _INNOVATION_DISTRIBUTIONS:
            fail(f"innovation_distribution must be one of {_INNOVATION_DISTRIBUTIONS}, got {self.innovation_distribution!r}.")
        if self.optimizer not in VALID_OPTIMIZERS:
            fail(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got {self.optimizer!r}.")
        if self.scheduler_type not in _SCHEDULERS:
            fail(f"scheduler_type must be one of {_SCHEDULERS}, got {self.scheduler_type!r}.")
        if not _is_real(self.slope) or self.slope < 0:
            fail(f"slope must be a finite number >= 0, got {self.slope!r}.")
        for param in ("decoder_var", "regime_temperature", "learning_rate"):
            value = getattr(self, param)
            if not _is_real(value) or value <= 0:
                fail(f"{param} must be a finite number > 0, got {value!r}.")
        for param in ("beta_kl", "optimizer_weight_decay", "optimizer_momentum", "tol", "scheduler_min_lr"):
            value = getattr(self, param)
            if not _is_real(value) or value < 0:
                fail(f"{param} must be a finite number >= 0, got {value!r}.")
        for param in ("scheduler_gamma", "scheduler_factor"):
            value = getattr(self, param)
            if not _is_real(value) or not 0 < value <= 1:
                fail(f"{param} must be in (0, 1], got {value!r}.")
        if not _is_int(self.scheduler_patience) or self.scheduler_patience < 0:
            fail(f"scheduler_patience must be an integer >= 0, got {self.scheduler_patience!r}.")
        for param in ("use_layer_norm", "mixing"):
            if not isinstance(getattr(self, param), (bool, np.bool_)):
                fail(f"{param} must be True or False, got {getattr(self, param)!r}.")
        self._scaler_name()
        if self.random_state is not None and not _is_int(self.random_state):
            fail(f"random_state must be None or an integer, got {self.random_state!r}.")
        if not _is_int(self.verbose) or self.verbose < 0:
            fail(f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}.")
        self._resolve_device()

    def _scaler_name(self) -> Optional[str]:
        """The scaler as the dataset's string name (``scaler`` field of the engine config), validating it."""
        scaler = self.scaler
        if scaler is None:
            return None
        if isinstance(scaler, str):
            if scaler.lower() not in _SCALERS:
                raise ConfigurationError(
                    f"{type(self).__name__}: scaler must be None, one of {sorted(_SCALERS)} or a transformer "
                    f"instance, got {scaler!r}."
                )
            return scaler
        if not (hasattr(scaler, "fit_transform") and hasattr(scaler, "inverse_transform")):
            raise ConfigurationError(
                f"{type(self).__name__}: scaler must have fit_transform and inverse_transform, got "
                f"{type(scaler).__name__}."
            )
        for key, cls in _SCALERS.items():
            if type(scaler) is cls:
                return key
        return type(scaler).__name__

    def _resolve_device(self) -> torch.device:
        """``device`` as a ``torch.device``; ``"auto"`` means CUDA if available, else CPU."""
        device = self.device
        if isinstance(device, str) and device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return torch.device(device)
        except (RuntimeError, TypeError, ValueError) as err:
            raise ConfigurationError(
                f"{type(self).__name__}: device must be 'auto', a device string such as 'cpu' or 'cuda:0', "
                f"or a torch.device; got {device!r} ({err})."
            ) from err

    # -- adapter and engine --------------------------------------------------

    def _fresh_scaler(self) -> Any:
        """The scaler handed to the dataset: the name, or a clone of the instance (warns if it was fitted)."""
        scaler = self.scaler
        if scaler is None or isinstance(scaler, str):
            return scaler
        try:
            check_is_fitted(scaler)
        except (SklearnNotFittedError, TypeError):
            pass
        else:
            warnings.warn(
                f"{type(self).__name__}: the scaler {type(scaler).__name__} passed in is already fitted; fit uses an "
                "unfitted clone and refits it on the training targets, so its statistics are discarded. The fitted "
                "copy is scaler_.",
                UserWarning,
                stacklevel=6,  # _fresh_scaler < _adapter < _fit < BaseFactorModel.fit < iVDFM.fit < caller
            )
        return clone(scaler)

    def _adapter(self, panel: PanelData, device: torch.device, *, fitted: bool = False) -> _RecordingDataset:
        """The engine's dataset built from ``panel``: a frame ``[time | y | exog]`` on a RangeIndex.

        The time column holds the ``DatetimeIndex`` (a ``PeriodIndex`` through
        ``to_timestamp()``); a ``RangeIndex`` gives no time column, so the time
        features use the position. With ``fitted=True`` (``update`` and
        ``predict(data=...)``) the dataset uses the fitted ``window_``,
        ``stride_`` and ``time_context_`` and scales the targets with
        ``scaler_`` without refitting it.
        """
        frame = panel.y.reset_index(drop=True)
        if panel.exog is not None:
            frame = pd.concat([frame, panel.exog.reset_index(drop=True)], axis=1)
        index = panel.index
        time_col = None
        if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
            stamps = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
            if stamps.tz is not None:
                stamps = stamps.tz_convert("UTC").tz_localize(None)
            # The dataset reads the stamps as int64 nanoseconds, whatever resolution the index has.
            stamps = stamps.as_unit("ns")
            time_col = _TIME_COLUMN
            while time_col in set(frame.columns):
                time_col += "_"
            frame.insert(0, time_col, np.asarray(stamps))
        if fitted:
            window, stride, time_context = self.window_, self.stride_, self.time_context_
            scaler = None if self.scaler_ is None else _FittedScaler(self.scaler_)
        else:
            window, stride, time_context, scaler = self.window, self.stride, self.time_context, self._fresh_scaler()
        return _RecordingDataset(
            data=frame,
            window=window,
            stride=stride,
            time_idx=time_col,
            context=list(panel.exog_names),
            time_context=time_context,
            scaler=scaler,
            device=device,
        )

    def _engine_params(self, window: int, stride: int) -> dict:
        """Every engine hyperparameter, passed explicitly under the engine's names.

        ``window`` and ``stride`` are the dataset's (``window=None`` is the
        series length and ``stride=None`` is ``window // 2``).
        """

        def widths(value: Any) -> Any:
            return list(value) if isinstance(value, tuple) else value

        return {
            "num_factors": self.n_factors,
            "window": window,
            "encoder_hidden_dim": widths(self.encoder_hidden_dim),
            "encoder_n_hidden_layers": self.encoder_n_hidden_layers,
            "decoder_type": self.decoder_type,
            "decoder_hidden_dim": widths(self.decoder_hidden_dim),
            "decoder_n_hidden_layers": self.decoder_n_hidden_layers,
            "prior_hidden_dim": widths(self.prior_hidden_dim),
            "prior_n_hidden_layers": self.prior_n_hidden_layers,
            "activation": self.activation,
            "slope": self.slope,
            "factor_order": self.factor_order,
            "innovation_distribution": self.innovation_distribution,
            "decoder_var": self.decoder_var,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "tolerance": self.tol,
            "seed": self.random_state,
            # The rest reach the engine's config through its **kwargs.
            "stride": stride,
            "time_context": self.time_context,
            "scaler": self._scaler_name(),
            "num_regimes": self.n_regimes,
            "regime_temperature": self.regime_temperature,
            "beta_kl": self.beta_kl,
            "use_layer_norm": self.use_layer_norm,
            "mixing": self.mixing,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_momentum": self.optimizer_momentum,
            "patience": self.patience,
            "scheduler_type": self.scheduler_type,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_min_lr": self.scheduler_min_lr,
        }

    @staticmethod
    def _restore_none_params(engine: iVDFMEngine, params: dict) -> None:
        """Give the engine the None values its constructor dropped.

        The engine drops None values before building its config, so
        ``patience=None`` would become 20 and ``scheduler_type=None`` would
        become ``"step"``. The engine reads ``patience``, ``scheduler_type``
        and ``scheduler_step_size`` from its attributes during ``fit``.
        """
        fields = {f.name for f in dataclasses.fields(engine._config)}
        dropped = {k: v for k, v in params.items() if v is None and k in fields}
        if dropped:
            engine._config = dataclasses.replace(engine._config, **dropped)
        for attr in ("patience", "scheduler_type", "scheduler_step_size"):
            setattr(engine, attr, params[attr])

    # -- fit -----------------------------------------------------------------

    def fit(self, data: Any, **fit_kwargs: Any) -> iVDFM:
        """Train on ``data`` and return ``self``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            Targets in ``y`` and, optionally, context variables in ``exog``.
            Missing and infinite values are interpolated linearly, then
            filled forward, backward and with zeros (in targets and exog).
            Mixed frequency and ``series_exog`` are rejected.

        Raises
        ------
        ConfigurationError
            For an invalid hyperparameter.
        UnsupportedInputError
            For ``series_exog`` or a mixed-frequency panel.
        DataValidationError
            For a target series with fewer than 2 observed (finite) values,
            or with the same value at every observation.

        Warns
        -----
        UserWarning
            When ``scaler`` is an already fitted transformer (it is cloned and
            refitted).

        A failed fit leaves the model unfitted (earlier fitted attributes are
        removed).
        """
        return super().fit(data, **fit_kwargs)

    def _fit(self, data: PanelData, **fit_kwargs: Any) -> None:
        if fit_kwargs:
            raise TypeError(f"iVDFM.fit() got unexpected keyword arguments {sorted(fit_kwargs)}.")
        self._validate_params()
        device = self._resolve_device()
        dataset = self._adapter(data, device)
        params = self._engine_params(dataset.window, dataset.stride)
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _PROGRESS_LOGGERS, logging.ERROR):
            # The engine seeds the global torch and numpy RNGs (random_state) and then builds its networks.
            engine = iVDFMEngine(dataset=dataset, device=device, **params)
            self._restore_none_params(engine, params)
            if self.verbose >= 2:
                _logger.debug("iVDFM: engine hyperparameters %s", params)
            engine.fit()
        engine._dataset = None  # the engine keeps no reference to the training data

        self.network_ = engine
        self.device_ = device
        self.scaler_ = dataset.scaler
        self.result_ = engine.get_result()
        self.n_iter_ = int(engine._num_iter)
        self.converged_ = bool(engine._converged)
        self.n_factors_ = int(engine.latent_dim)
        self.random_state_ = self.random_state
        self.window_ = int(dataset.window)
        self.stride_ = int(dataset.stride)
        self.time_context_ = int(self.time_context)
        self.context_dim_ = int(dataset.context_dim)
        self.conditioning_index_ = data.index
        self.window_factors_ = self._window_factors_frame(engine.factors, dataset.accessed, data.index)
        targets = data.y.to_numpy(dtype=np.float64)
        # The dataset fills non-finite targets like missing ones; they do not count as observed.
        scaled = np.where(np.isfinite(targets), dataset.target, np.nan)
        self._set_interpretation(targets, explained_variance(scaled, self.result_.x_sm))

    def _set_interpretation(self, targets: Optional[np.ndarray], explained: Optional[np.ndarray]) -> None:
        """Set ``factor_names_``, ``factors_``, ``loadings_``, ``loadings_kind_`` and ``explained_variance_``.

        ``factors_`` is ``result_.Z`` and ``loadings_`` its mean decoder
        Jacobian. ``targets`` (the training targets in data units, NaN where
        missing, kept for :meth:`decompose`) and ``explained`` (the
        explained-variance values) are None for a run saved by 0.8.x.
        """
        path = np.asarray(self.result_.Z, dtype=np.float64)
        self.factor_names_ = np.asarray([f"factor_{i}" for i in range(path.shape[1])], dtype=object)
        names, series = list(self.factor_names_), list(self.feature_names_in_)
        self.factors_ = pd.DataFrame(path, index=self.index_[len(self.index_) - len(path):], columns=names)
        self.loadings_ = pd.DataFrame(
            _mean_decoder_jacobian(self.network_, self.result_.Z), index=series, columns=names
        )
        self.loadings_kind_ = "mean_jacobian"
        self._targets_ = targets
        if explained is not None:
            self.explained_variance_ = pd.Series(explained, index=series, name="explained_variance")

    def _clear_fitted(self) -> None:
        """Remove the fitted attributes and the private training targets ``_targets_``."""
        super()._clear_fitted()
        vars(self).pop("_targets_", None)

    def _window_factors_frame(self, factors: np.ndarray, accessed: list, index: pd.Index) -> pd.DataFrame:
        """``window_factors_``: the engine's window factors in chronological order on a ``(window, time)`` MultiIndex.

        The engine's last ``len(windows)`` dataset accesses are its final
        factor extraction, so window ``order[k]`` is row ``k`` of the engine's
        factors. Without recorded accesses (the single-batch GPU path) the
        windows are in natural order.
        """
        n_windows, window, r = factors.shape
        order = np.asarray(accessed[-n_windows:] if len(accessed) >= n_windows else range(n_windows))
        chronological = np.asarray(factors)[np.argsort(order, kind="stable")]
        starts = np.arange(n_windows) * self.stride_
        positions = (starts[:, None] + np.arange(window)[None, :]).reshape(-1)
        return self._window_factors_from(
            chronological.reshape(n_windows * window, r).astype(np.float64),
            np.repeat(np.arange(n_windows), window),
            positions,
            index,
        )

    @staticmethod
    def _window_factors_from(
        values: np.ndarray, windows: np.ndarray, positions: np.ndarray, index: pd.Index
    ) -> pd.DataFrame:
        """``window_factors_`` from its float64 values, window labels and row positions in ``index``."""
        multi = pd.MultiIndex.from_arrays(
            [windows, index[positions]],
            names=["window", index.name if index.name is not None else "time"],
        )
        return pd.DataFrame(values, index=multi, columns=[f"factor_{i}" for i in range(values.shape[1])])

    # -- conditioning on new data -------------------------------------------

    def _panel(self, data: Any, method: str) -> PanelData:
        """Validate ``data`` for a fitted model: same targets and exog as in training, two or more rows."""
        panel = self._validate_data(data)  # inputs the tags exclude raise UnsupportedInputError first
        expected = list(self.feature_names_in_)
        if list(panel.series_names) != expected:
            raise DataValidationError(
                f"iVDFM.{method}: data has series {list(panel.series_names)}, but the model was fitted on "
                f"{expected}. Pass the same series in the same order."
            )
        trained = list(self.exog_names_in_)
        if list(panel.exog_names) != trained:
            raise DataValidationError(
                f"iVDFM.{method}: data has exog {list(panel.exog_names)}, but the model was fitted on {trained}. "
                "Pass the same exog columns in the same order."
            )
        if panel.n_timesteps < 2:
            raise DataValidationError(f"iVDFM.{method}: data needs at least 2 time steps, got {panel.n_timesteps}.")
        return panel

    def _condition(self, engine: iVDFMEngine, panel: PanelData) -> None:
        """Run the engine's ``update`` on ``panel``: encode it with the fixed networks and append the factors.

        The dataset is built like the training dataset, with the fitted window,
        stride and time context and the fitted ``scaler_``.
        """
        dataset = self._adapter(panel, self.device_, fitted=True)
        with fit_verbosity(self.verbose), quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _PROGRESS_LOGGERS, logging.ERROR):
            engine.update(dataset)

    def _rng_devices(self) -> list:
        """The CUDA devices whose generators ``predict(data=...)`` forks besides the CPU one."""
        device = self.device_
        if device.type != "cuda":
            return []
        return [device.index if device.index is not None else torch.cuda.current_device()]

    def update(self, data: Any, *, refit: bool = False) -> iVDFM:
        """Condition later forecasts on ``data`` and return ``self``.

        The engine encodes ``data`` as one sequence with the fixed networks
        (``no_grad``) and appends the result to its factor path, so
        ``predict(h)`` then starts from the factor state at the last time step
        of ``data``; it also encodes every window of ``data`` and appends the
        window-end factors to ``factors``.
        Missing values are filled as in ``fit`` and the targets are scaled with
        the fitted ``scaler_`` (not refitted). The network parameters,
        ``result_``, ``factors_``, ``window_factors_``, ``loadings_``,
        ``explained_variance_`` and the other fit diagnostics are unchanged,
        and :meth:`decompose` still splits the training data;
        ``conditioning_index_`` becomes ``data.index``. The encoder samples
        its innovations, so ``update`` draws from the global torch RNG.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            The same targets (and exog) as in training, in the same order; at
            least 2 time steps. A window longer than ``data`` is clipped to
            its length.
        refit : bool, default False
            Not supported: iVDFM cannot continue training from new data.

        Raises
        ------
        UnsupportedInputError
            For ``refit=True``, ``series_exog`` or a mixed-frequency panel.
        DataValidationError
            If the series or exog names differ from the training data.
        """
        self._check_is_fitted("update")
        if refit:
            raise UnsupportedInputError(
                "iVDFM.update(refit=True) is not supported: iVDFM cannot continue training on new data. Use "
                "update(data) (refit=False) to condition the forecasts on it, or fit(data) to retrain from scratch."
            )
        panel = self._panel(data, "update")
        self._condition(self.network_, panel)
        self.conditioning_index_ = panel.index
        return self

    # -- forecasting ---------------------------------------------------------

    def _futr_exog_values(self, h: int, futr_exog: Any, origin_index: pd.Index) -> Optional[np.ndarray]:
        """``futr_exog`` validated as a float64 (h, n_exog) array, or None.

        A DataFrame is matched by column name, and its index must be the
        forecast index (the ``h`` steps after ``origin_index[-1]``); a default
        ``RangeIndex`` (0 to h-1) is taken by position, like an array. A
        polars frame's single Date/Datetime column is its index.
        """
        exog_names = list(self.exog_names_in_)
        if futr_exog is None:
            if exog_names:
                warnings.warn(
                    f"iVDFM was fitted with exog {exog_names} but predict got no futr_exog; the forecast context "
                    "is generated time features in every column, as the engine does without context_data. Pass "
                    "futr_exog (h rows, one column per exog name) for the context the model was trained on.",
                    UserWarning,
                    stacklevel=4,
                )
            return None
        if not exog_names:
            raise DataValidationError(
                "iVDFM was fitted without exog, so predict takes no futr_exog; drop the argument."
            )
        time_col = _polars_time_col(futr_exog)
        futr_exog = _to_pandas(futr_exog)
        if time_col is not None:
            futr_exog = futr_exog.set_index(time_col)
        if isinstance(futr_exog, pd.DataFrame):
            missing = [c for c in exog_names if c not in futr_exog.columns]
            if missing:
                raise DataValidationError(
                    f"futr_exog lacks the exog columns {missing}; the model was fitted on {exog_names}."
                )
            values = futr_exog[exog_names].to_numpy(dtype=np.float64)
        else:
            values = np.asarray(futr_exog, dtype=np.float64)
        if values.shape != (h, len(exog_names)):
            raise DataValidationError(
                f"futr_exog must have shape (h, n_exog) = ({h}, {len(exog_names)}), got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise DataValidationError("futr_exog contains NaN or infinite values.")
        if isinstance(futr_exog, pd.DataFrame):
            index = futr_exog.index
            positional = isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1
            if not positional:
                expected = self._forecast_index(origin_index, h)
                # By value: a polars Date column arrives as datetime64[ms], the forecast index as [ns].
                same = len(index) == len(expected) and bool(np.all(index == expected))
                if not same:
                    raise DataValidationError(
                        f"futr_exog index runs {index[0]!r} to {index[-1]!r}, but the forecast index runs "
                        f"{expected[0]!r} to {expected[-1]!r} (the {h} steps after {origin_index[-1]!r}). Pass the "
                        "rows of the forecast period, or an array (or a frame with a default RangeIndex) to "
                        "match them by position."
                    )
        return values

    def _engine_forecast(
        self, engine: iVDFMEngine, h: int, futr_exog: Optional[np.ndarray], deterministic: bool, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The engine's forecast factors (h, r) and model-space observations (h, N).

        The context is ``[time features | futr_exog]`` (``context_data``), or
        None when no ``futr_exog`` is given (the engine then generates time
        features for every context column).
        """
        context_data = None
        if futr_exog is not None:
            context_data = np.hstack([engine._future_time_features(h, self.time_context_), futr_exog])
        factors, y_pred = engine._forecast(h, context_data, deterministic, n_samples)
        return to_numpy(factors.squeeze(0)), to_numpy(y_pred.squeeze(0))

    def _run_forecast(
        self, method: str, h: Any, data: Any, futr_exog: Any, deterministic: Any, n_samples: Any
    ) -> tuple[np.ndarray, np.ndarray, int, pd.Index]:
        self._check_is_fitted(method)
        h = self._validate_horizon(h)
        panel = None if data is None else self._panel(data, method)
        if not isinstance(deterministic, (bool, np.bool_)):
            raise TypeError(f"deterministic must be True or False, got {deterministic!r}.")
        if not _is_int(n_samples) or n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}.")
        futr = self._futr_exog_values(h, futr_exog, self.conditioning_index_ if panel is None else panel.index)
        if panel is None:
            factors, y_pred = self._engine_forecast(self.network_, h, futr, bool(deterministic), int(n_samples))
            return factors, y_pred, h, self.conditioning_index_
        # update(data) on a copy of the engine; the global torch RNG the update and
        # a sampled forecast draw from is restored afterwards.
        with torch.random.fork_rng(devices=self._rng_devices()):
            engine = copy.deepcopy(self.network_)
            self._condition(engine, panel)
            factors, y_pred = self._engine_forecast(engine, h, futr, bool(deterministic), int(n_samples))
        return factors, y_pred, h, panel.index

    def _frame(self, values: np.ndarray, h: int, columns: list, kind: str, origin_index: pd.Index) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(values, dtype=np.float64),
            index=self._forecast_index(origin_index, h),
            columns=columns,
        )
        frame.attrs = {"model": type(self).__name__, "kind": kind, "origin": origin_index[-1]}
        return frame

    def predict(
        self,
        h: int,
        data: Any = None,
        *,
        futr_exog: Any = None,
        deterministic: bool = True,
        n_samples: int = 1,
    ) -> pd.DataFrame:
        """Forecast the targets ``h`` steps ahead, in data units.

        The engine's closed-loop forecast starts from the factor state at the
        last time step of the conditioning series (the training data, or the
        data of the last ``update``), which the engine encodes as one sequence
        after training; the decoded observations are inverse-transformed with
        ``scaler_``.

        Parameters
        ----------
        h : int
            Forecast horizon (a positive integer; anything else raises
            ``TypeError``).
        data : PanelData, DataFrame or 2-D array, optional
            A window to condition on (tag ``predict_data="window"``): every
            row is encoded and the forecast starts after its last row. This is
            ``update(data)`` on a copy of the model, followed by the forecast,
            with the global torch RNG restored afterwards; the model is
            unchanged. Same series and exog as in training.
        futr_exog : DataFrame or array of shape (h, n_exog), optional
            Future values of the exog columns (a DataFrame is matched by column
            name, and its index must be the forecast index unless it is a
            default ``RangeIndex``, which is taken by position). They follow
            the generated time features in the context u_t.
            None generates time features for every context column, as the
            engine does without context; that warns for a model with exog.
        deterministic : bool, default True
            True uses zero innovations; False samples them from the prior.
        n_samples : int, default 1
            Prior samples averaged when ``deterministic=False``.

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows after the last conditioning time stamp, one
            column per target in the user's order; ``attrs`` holds ``model``,
            ``kind="forecast"`` and ``origin``.
        """
        _, y_pred, h, origin = self._run_forecast("predict", h, data, futr_exog, deterministic, n_samples)
        values = np.asarray(y_pred, dtype=np.float64)
        if self.scaler_ is not None:
            values = self.scaler_.inverse_transform(values)
        return self._emit(self._frame(values, h, list(self.feature_names_in_), "forecast", origin))

    def predict_factors(
        self,
        h: int,
        data: Any = None,
        *,
        futr_exog: Any = None,
        deterministic: bool = True,
        n_samples: int = 1,
    ) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (same arguments as :meth:`predict`).

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows, one column per name in ``factor_names_``;
            ``attrs["kind"]`` is ``"factors"``.
        """
        factors, _, h, origin = self._run_forecast("predict_factors", h, data, futr_exog, deterministic, n_samples)
        return self._emit(self._frame(factors, h, list(self.factor_names_), "factors", origin))

    # -- interpretation ------------------------------------------------------

    def decompose(self, *, by: str = "component") -> pd.DataFrame:
        """Split the training targets into the common component and the idiosyncratic part, in data units.

        The common component is the decoded factor path (``result_.x_sm``)
        inverse-transformed with ``scaler_``; the idiosyncratic part is the
        data minus it, NaN where the data is missing, so the two add up to the
        data.

        Parameters
        ----------
        by : {"component", "factor"}, default "component"
            Only ``"component"``: the decoder is nonlinear
            (``loadings_kind_`` is ``"mean_jacobian"``), so the common
            component has no exact split by factor.

        Returns
        -------
        pandas.DataFrame
            float64, indexed by ``index_``, MultiIndex columns
            ``(component, series)`` with ``component`` ``"common"`` or
            ``"idiosyncratic"`` and the series in the user's order.

        Raises
        ------
        ConfigurationError
            For ``by`` other than ``"component"`` and ``"factor"``.
        UnsupportedInputError
            For ``by="factor"``, and for a run saved by 0.8.x, which holds no
            training targets.
        """
        self._check_is_fitted("decompose")
        if by == "factor":
            raise UnsupportedInputError(
                "iVDFM.decompose(by='factor') is not supported: the decoder is nonlinear (loadings_kind_ "
                "'mean_jacobian'), so the common component has no exact split by factor. Use by='component'."
            )
        if by != "component":
            raise ConfigurationError(f"by must be 'component' or 'factor', got {by!r}.")
        if self._targets_ is None:
            raise UnsupportedInputError(
                "iVDFM.decompose: this run was saved by dfm-python 0.8 without its training targets. Refit the "
                "model to decompose its training data."
            )
        common = np.asarray(self.result_.x_sm, dtype=np.float64)
        if self.scaler_ is not None:
            common = np.asarray(self.scaler_.inverse_transform(common), dtype=np.float64)
        columns = pd.MultiIndex.from_product(
            [["common", "idiosyncratic"], list(self.feature_names_in_)], names=["component", "series"]
        )
        values = np.hstack([common, self._targets_ - common])
        return self._emit(pd.DataFrame(values, index=self.index_, columns=columns))

    # -- save / load ---------------------------------------------------------

    def _save_state(self, path: Path) -> None:
        """Write the iVDFM part of the saved run (spec section 7).

        ``state.npz``: the engine's current ``factors``, ``innovations``,
        ``factor_path`` and ``regime_weights`` (when it has them), ``fit/factors``, ``fit/window``
        and ``fit/position`` (``window_factors_`` as values, window labels and
        row positions in ``index_``), ``fit/targets`` (the training targets in
        data units) and ``fit/explained_variance`` (when the run has them) and
        the ``scaler/`` entries. ``state.json``: the training diagnostics, the
        engine dims, the fitted window attributes and ``conditioning_index``.
        ``result.npz``/``result.json``: ``result_`` without ``model_state_dict``
        and ``config``, rebuilt on load from ``weights.pt`` and the engine
        config. Load recomputes ``factors_`` and ``loadings_`` from ``result_.Z``.
        """
        engine = self.network_
        frame = self.window_factors_
        arrays = {"factors": engine.factors, "innovations": engine.innovations}
        if engine.regime_weights is not None:
            arrays["regime_weights"] = engine.regime_weights
        if engine.factor_path is not None:
            arrays["factor_path"] = engine.factor_path
        arrays["fit/factors"] = frame.to_numpy()
        arrays["fit/window"] = frame.index.get_level_values(0).to_numpy(dtype=np.int64)
        arrays["fit/position"] = self.index_.get_indexer(frame.index.get_level_values(1)).astype(np.int64)
        if self._targets_ is not None:
            arrays["fit/targets"] = self._targets_
            arrays["fit/explained_variance"] = self.explained_variance_.to_numpy()
        meta = {
            "num_iter": engine._num_iter,
            "loss_now": engine.loss_now,
            "elbo": engine._elbo,
            "converged": engine._converged,
            **{name: getattr(engine, name) for name in _ENGINE_DIMS},
            "window_": self.window_,
            "stride_": self.stride_,
            "time_context_": self.time_context_,
            "context_dim_": self.context_dim_,
            "conditioning_index": index_to_spec(self.conditioning_index_),
        }
        self._write_state(path, arrays, meta)
        self._write_result(path, exclude=_RESULT_REBUILT)
        self._write_weights(path)

    @classmethod
    def _load_state(cls, est: iVDFM, path: Path, device: Any, allow_pickle: bool) -> None:
        """Rebuild the engine without data or seeding, load its weights and restore the fitted attributes."""
        arrays, meta = cls._read_state(path)
        device = est._load_device(device)
        params = est._engine_params(meta["window"], meta["stride_"])
        engine = iVDFMEngine._from_saved(params, {name: meta[name] for name in _ENGINE_DIMS}, device)
        engine.load_state_dict(cls._read_weights(path, device))
        engine.factors = arrays["factors"]
        engine.innovations = arrays["innovations"]
        engine.regime_weights = arrays.get("regime_weights")
        engine.factor_path = arrays.get("factor_path")  # absent in runs saved before 0.8.3
        engine._num_iter = meta["num_iter"]
        # state.json holds the float64 means as JSON numbers; fit leaves numpy float64 values.
        engine.loss_now = None if meta["loss_now"] is None else np.float64(meta["loss_now"])
        engine._elbo = None if meta["elbo"] is None else np.float64(meta["elbo"])
        engine._converged = meta["converged"]
        engine.training_state = iVDFMModelState.from_model(engine)

        weights = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
        est.network_ = engine
        est.device_ = device
        est.result_ = cls._read_result(
            path, iVDFMResult, model_state_dict=weights, config=dataclasses.asdict(engine._config)
        )
        est.n_iter_ = int(meta["num_iter"])
        est.converged_ = bool(meta["converged"])
        est.n_factors_ = int(engine.latent_dim)
        est.random_state_ = est.random_state
        est.window_ = int(meta["window_"])
        est.stride_ = int(meta["stride_"])
        est.time_context_ = int(meta["time_context_"])
        est.context_dim_ = int(meta["context_dim_"])
        est.conditioning_index_ = index_from_spec(meta["conditioning_index"])
        est.window_factors_ = cls._window_factors_from(
            arrays["fit/factors"], arrays["fit/window"], arrays["fit/position"], est.index_
        )
        # Runs saved by 0.8.x hold no training targets.
        est._set_interpretation(arrays.get("fit/targets"), arrays.get("fit/explained_variance"))

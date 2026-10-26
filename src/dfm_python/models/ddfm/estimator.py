"""``DDFM``: the deep dynamic factor model estimator (sklearn conventions, PanelData input).

A thin layer over the private engine
:class:`~dfm_python.models.ddfm._engine.DDFMEngine`. ``fit`` builds the
engine's dataset from :class:`~dfm_python.PanelData` as the frame
``[exog | y]`` (the encoder's covariates, then the targets), constructs the
engine with every hyperparameter passed explicitly, trains it and builds the
linear state space on its factors.

``predict(h)`` returns the target forecasts in data units and
``predict_factors(h)`` the AR(1) factor forecast. ``update(data)`` takes the
rows after the conditioning data (overlap raises), scales them with the
scalers fitted at fit time, interpolates them as fit does and appends their
encoded factors; ``predict(h, data=...)`` does the same on a copy.

``fit`` (and ``load``) also set the interpretation attributes
(``factor_names_``, ``factors_``, ``loadings_``, ``loadings_kind_``,
``explained_variance_``) from the trained decoder and the training factors;
``decompose(by=...)`` splits the training targets into the decoder's
reconstruction and the residual, in data units.

``save(path)`` writes the pickle-free save layout (``state.npz``/``state.json``
with the engine's factors, residuals, state-space matrices and scaled
conditioning frame, ``result.npz``/``result.json`` with ``result_`` and
``weights.pt`` with the network weights); ``DDFM.load(path, device=None)``
rebuilds the engine from them without data or seeding.
"""

from __future__ import annotations

import copy
import logging
import math
import pickle
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
from ...config.constants import VALID_OPTIMIZERS
from ...config.schema.model import DDFMConfig
from ...config.schema.results import DDFMResult
from ...dataset.ddfm_dataset import DDFMDataset
from ...dataset.panel import PanelData
from ...io.layout import index_from_spec, index_to_spec
from ...io.scaler import SCALER_PREFIX, decode_scaler, encode_scaler, is_known_scaler
from ...logger import get_logger
from ...logger.progress import fit_verbosity, quiet_engine_loggers
from ...utils.errors import ConfigurationError, DataValidationError, UnsupportedInputError
from ...utils.interpret import affine_scaler_params, explained_variance
from ._engine import DDFMEngine

__all__ = ["DDFM"]

_logger = get_logger(__name__)

#: Scaler names accepted by ``scaler``, and the class each one builds.
_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "quantile": QuantileTransformer,
}
_DECODER_TYPES = ("linear", "mlp")
_ACTIVATIONS = ("relu", "tanh", "sigmoid")
_LIMIT_DIRECTIONS = ("forward", "backward", "both")
_POSITIVE_INTS = ("n_factors", "n_mc_samples", "window_size", "max_iter")
# The engine's progress interval (disp): it logs a summary every 10 MCMC iterations, at INFO.
_ENGINE_DISP = 10
# Loggers the engine (and the code it calls) writes progress to; raised to WARNING during fit with verbose=0.
_ENGINE_LOGGERS = (
    "dfm_python.models.ddfm",
    "dfm_python.dataset.ddfm_dataset",
    "dfm_python.numeric",
    "dfm_python.utils",
)
# The MCMC loop's variance-collapse diagnostics, logged as warnings on every
# iteration they apply to; raised to ERROR during fit with verbose=0.
_DIAGNOSTIC_LOGGERS = ("dfm_python.models.ddfm.sampling",)
# Name of the adapter's time column; it is never in the frame (a suffix is added
# if a series or exog column has this name), so the dataset keeps the frame's RangeIndex.
_TIME_COLUMN = "__dfm_python_time__"
# Engine attributes that only training reads (the MCMC state and the training
# frames); fit drops them. _update and _forecast read none of them.
_TRAINING_BUFFERS = (
    "_last_iter_datasets",
    "data_imputed",
    "data_denoised",
    "data_denoised_interpolated",
    "y_actual",
    "rng",
)
# result_ fields not written to result.npz/json: the target scaler is scaler_ (scaler.json), restored on load.
_RESULT_REBUILT = ("target_scaler",)
# The engine's state-space matrices (training_state fields), saved as ssm/<name>.
_SSM_FIELDS = ("F", "H", "Q", "R", "mu_0", "Sigma_0")
# Engine values __init__ reads from the dataset, saved in state.json and passed to DDFMEngine._from_saved.
_ENGINE_DIMS = ("input_dim", "output_dim", "num_series", "target_series", "covariates")
# state.npz prefix of the fitted covariate scaler's arrays (its JSON part is state.json's "covariate_scaler").
_COVARIATE_SCALER_PREFIX = "covariate_scaler/"
# The covariate scaler's "class" in state.json when it is pickled (a class outside the five known scalers).
_PICKLED = "pickle"


def _encode_covariate_scaler(scaler: Any) -> tuple[dict, dict]:
    """``(json_dict, arrays)`` for the engine's fitted covariate scaler.

    The five known scalers (and None) are encoded like ``scaler_``
    (:func:`~dfm_python.io.scaler.encode_scaler`), their arrays under
    ``covariate_scaler/``. Any other class (the covariate scaler is a fresh
    instance of ``scaler``'s class, so it is custom exactly when ``scaler_``
    is, and ``scaler_`` is then pickled too) is pickled into the uint8 array
    ``covariate_scaler/pickle``; load reads it only with ``allow_pickle=True``.
    """
    if is_known_scaler(scaler):
        try:
            json_dict, arrays = encode_scaler(scaler)
        except TypeError:  # a parameter JSON cannot hold: pickle, as write_scaler does
            pass
        else:
            return json_dict, {_COVARIATE_SCALER_PREFIX + k[len(SCALER_PREFIX):]: v for k, v in arrays.items()}
    blob = pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL)
    return {"class": _PICKLED}, {_COVARIATE_SCALER_PREFIX + "pickle": np.frombuffer(blob, dtype=np.uint8)}


def _decode_covariate_scaler(json_dict: Any, arrays: dict, allow_pickle: bool, path: Path) -> Any:
    """Inverse of :func:`_encode_covariate_scaler`; None for a run saved without one."""
    if json_dict is None:
        return None
    if json_dict.get("class") == _PICKLED:
        if not allow_pickle:
            raise ConfigurationError(
                f"{str(path)!r} stores its fitted covariate scaler pickled, because its class is not one of the "
                "known scalers. Unpickling can run arbitrary code, so it is off by default: pass allow_pickle=True "
                "to load(...) only if you trust the directory."
            )
        return pickle.loads(arrays[_COVARIATE_SCALER_PREFIX + "pickle"].tobytes())  # noqa: S301 - gated by allow_pickle
    prefix = _COVARIATE_SCALER_PREFIX
    return decode_scaler(json_dict, {SCALER_PREFIX + k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)})


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


class DDFM(BaseFactorModel):
    """Deep dynamic factor model.

    A nonlinear encoder maps the covariates and targets at each time step to
    ``n_factors`` latent factors, and a linear (or MLP) decoder reconstructs
    the targets from them. The autoencoder is pre-trained, then trained with
    MCMC denoising: each iteration draws ``n_mc_samples`` corrupted copies of
    the targets from the idiosyncratic residuals, trains on them and imputes
    missing targets. After training, an AR(1) transition and the decoder's
    last linear layer form a linear state space used for forecasting.

    The covariates are the ``exog`` columns of :class:`~dfm_python.PanelData`:
    they enter the encoder but are not reconstructed or forecast.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int, default 4
        Number of latent factors r (the width of the encoder's output layer).
    encoder_layers : tuple of int, default (16,)
        Widths of the encoder's hidden layers. The engine's encoder is
        ``(*encoder_layers, n_factors)``. May be empty with the linear decoder;
        the MLP decoder mirrors the hidden layers, so it needs at least one.
    decoder_type : {"linear", "mlp"}, default "linear"
        Decoder from the factors to the targets: one linear layer, or an MLP
        that mirrors the encoder's hidden layers.
    activation : {"relu", "tanh", "sigmoid"}, default "relu"
        Activation of the hidden layers of the encoder and the MLP decoder.
    learning_rate : float, default 0.005
        Learning rate of the optimizer.
    optimizer : {"Adam", "AdamW", "SGD"}, default "Adam"
        Optimizer of the autoencoder's weights.
    n_mc_samples : int, default 10
        Monte Carlo samples (corrupted copies of the targets) per MCMC
        iteration; pre-training also runs this many epochs.
    window_size : int, default 100
        Time steps per training batch.
    max_iter : int, default 200
        Maximum MCMC iterations.
    tol : float, default 5e-4
        Training stops when the relative change of the reconstruction MSE
        between iterations falls below ``tol``.
    interpolation_method : str, default "linear"
        ``pandas.DataFrame.interpolate`` method for missing values (the
        frame has a RangeIndex, so index-based methods use positions).
    interpolation_limit : int, optional, default 10
        Maximum number of consecutive missing values filled; None fills all.
    interpolation_limit_direction : {"forward", "backward", "both"}, default "both"
        Direction in which consecutive missing values are filled (the
        ``limit_direction`` of ``pandas.DataFrame.interpolate``).
    scaler : str, sklearn transformer or None, default "standard"
        Scaler of the targets: "standard", "robust", "minmax", "maxabs",
        "quantile", or an unfitted transformer, which is cloned. The
        covariates are scaled with a fresh instance of the same class, fitted
        on the training covariates. None trains on the data as given.
        Forecasts are always in data units. Both scalers are fitted once, in
        ``fit``: ``update`` and ``predict(h, data=)`` scale the new rows with
        them and never refit them.
    random_state : int, optional, default 3
        Seed of the autoencoder's initialization (global torch RNG) and of the
        local generator of the MCMC noise. None does not seed, so runs differ.
    device : str or torch.device, default "auto"
        "auto" uses CUDA if available, else CPU.
    verbose : int, default 0
        0 is silent; 1 logs the engine's progress; 2 also logs debug records.

    Attributes
    ----------
    result_ : DDFMResult
        Factors ``Z`` (T x r, averaged over the MC samples), the
        reconstruction ``x_sm`` (T x N, model space) and the state space:
        ``A`` (the AR(1) transition F), ``C`` (the loadings H, N x r, the DFM
        orientation), ``Q``, ``R``, ``Z_0``, ``V_0``.
    factor_names_ : numpy.ndarray of str
        ``factor_0`` .. ``factor_{r-1}`` (object dtype).
    factors_ : pandas.DataFrame
        The MC-averaged encoder factors ``result_.Z`` as float64 on
        ``index_``, columns ``factor_names_``.
    loadings_ : pandas.DataFrame
        float64, index ``feature_names_in_`` (the targets; the exog columns
        are not decoded), columns ``factor_names_``, in model (scaled) space.
        The linear decoder's weight; for the MLP decoder, the decoder's
        Jacobian with respect to the factors averaged over the rows of
        ``factors_`` (a local linear summary of a nonlinear map; autograd
        derivatives, so relu contributes slope 0 at its kink).
    loadings_kind_ : str
        ``"exact"`` (linear decoder) or ``"mean_jacobian"`` (MLP decoder).
    explained_variance_ : pandas.Series
        Per target (index ``feature_names_in_``, name ``"explained_variance"``):
        ``1 - var(data - common) / var(data)`` over the observed training
        entries of the scaled targets, where ``common`` is the decoder's
        output on ``factors_``.
    network_ : torch.nn.Module
        The trained engine.
    device_ : torch.device
        The resolved device.
    scaler_ : sklearn transformer or None
        The target scaler fitted on the training data (``update`` does not
        change it).
    n_features_in_, feature_names_in_, exog_names_in_, index_
        Number and names of the targets, the exog names and the training index.
    n_iter_ : int
        MCMC iterations run.
    converged_ : bool
        True when training stopped on ``tol`` before ``max_iter``.
    n_factors_ : int
        Number of factors.
    random_state_ : int or None
        The seed the run used.
    conditioning_index_ : pandas.Index
        Time index of the data the forecasts start from: ``index_``, extended
        by the rows of each ``update``; ``predict(h)`` returns the ``h``
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
        "predict_data": "new_rows",
        # The class default (random_state=3); an instance reports random_state is None.
        "non_deterministic": False,
    }
    _config_class = DDFMConfig
    _param_choices = {
        "decoder_type": _DECODER_TYPES,
        "activation": _ACTIVATIONS,
        "optimizer": VALID_OPTIMIZERS,
        "interpolation_limit_direction": _LIMIT_DIRECTIONS,
        "scaler": (*_SCALERS, None),
    }

    def __init__(
        self,
        *,
        n_factors: int = 4,
        encoder_layers: Any = (16,),
        decoder_type: str = "linear",
        activation: str = "relu",
        learning_rate: float = 0.005,
        optimizer: str = "Adam",
        n_mc_samples: int = 10,
        window_size: int = 100,
        max_iter: int = 200,
        tol: float = 5e-4,
        interpolation_method: str = "linear",
        interpolation_limit: Optional[int] = 10,
        interpolation_limit_direction: str = "both",
        scaler: Any = "standard",
        random_state: Optional[int] = 3,
        device: Any = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.encoder_layers = encoder_layers
        self.decoder_type = decoder_type
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_iter = max_iter
        self.tol = tol
        self.interpolation_method = interpolation_method
        self.interpolation_limit = interpolation_limit
        self.interpolation_limit_direction = interpolation_limit_direction
        self.scaler = scaler
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _instance_tags(self) -> dict:
        return {"non_deterministic": self.random_state is None}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small CPU parameter sets: a linear decoder, and an MLP decoder with a scaler."""
        tiny = {
            "n_factors": 2, "encoder_layers": (4,), "n_mc_samples": 2, "window_size": 16, "max_iter": 3,
            "scaler": None, "random_state": 0, "device": "cpu",
        }
        return [tiny, {**tiny, "decoder_type": "mlp", "encoder_layers": (6, 4), "scaler": "standard"}]

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
        if self.decoder_type not in _DECODER_TYPES:
            fail(f"decoder_type must be one of {_DECODER_TYPES}, got {self.decoder_type!r}.")
        layers = self.encoder_layers
        if not isinstance(layers, (list, tuple)) or not all(_is_int(w) and w >= 1 for w in layers):
            fail(f"encoder_layers must be a list or tuple of positive integers, got {layers!r}.")
        if self.decoder_type == "mlp" and len(layers) == 0:
            fail(
                "encoder_layers must not be empty with decoder_type='mlp': the MLP decoder mirrors the encoder's "
                "hidden layers. Give at least one hidden width, or use decoder_type='linear'."
            )
        if self.activation not in _ACTIVATIONS:
            fail(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}.")
        if self.optimizer not in VALID_OPTIMIZERS:
            fail(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got {self.optimizer!r}.")
        if not _is_real(self.learning_rate) or self.learning_rate <= 0:
            fail(f"learning_rate must be a finite number > 0, got {self.learning_rate!r}.")
        if not _is_real(self.tol) or self.tol < 0:
            fail(f"tol must be a finite number >= 0, got {self.tol!r}.")
        if not isinstance(self.interpolation_method, str) or not self.interpolation_method:
            fail(f"interpolation_method must be a non-empty string, got {self.interpolation_method!r}.")
        limit = self.interpolation_limit
        if limit is not None and (not _is_int(limit) or limit < 1):
            fail(f"interpolation_limit must be None or a positive integer, got {limit!r}.")
        if self.interpolation_limit_direction not in _LIMIT_DIRECTIONS:
            fail(
                f"interpolation_limit_direction must be one of {_LIMIT_DIRECTIONS}, got "
                f"{self.interpolation_limit_direction!r}."
            )
        self._check_scaler()
        if self.random_state is not None and not _is_int(self.random_state):
            fail(f"random_state must be None or an integer, got {self.random_state!r}.")
        if not _is_int(self.verbose) or self.verbose < 0:
            fail(f"verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}.")
        self._resolve_device()

    def _check_scaler(self) -> None:
        scaler = self.scaler
        if scaler is None:
            return
        if isinstance(scaler, str):
            if scaler.lower() not in _SCALERS:
                raise ConfigurationError(
                    f"{type(self).__name__}: scaler must be None, one of {sorted(_SCALERS)} or a transformer "
                    f"instance, got {scaler!r}."
                )
            return
        if not (hasattr(scaler, "fit_transform") and hasattr(scaler, "inverse_transform")):
            raise ConfigurationError(
                f"{type(self).__name__}: scaler must have fit_transform and inverse_transform, got "
                f"{type(scaler).__name__}."
            )

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
        """The scaler handed to the dataset: a new instance for a name, or a clone of the instance (warns if fitted)."""
        scaler = self.scaler
        if scaler is None:
            return None
        if isinstance(scaler, str):
            return _SCALERS[scaler.lower()]()
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
                stacklevel=6,  # _fresh_scaler < _adapter < _fit < BaseFactorModel.fit < DDFM.fit < caller
            )
        return clone(scaler)

    @staticmethod
    def _engine_frame(panel: PanelData) -> pd.DataFrame:
        """``panel`` as the engine's frame ``[exog | y]`` on a RangeIndex, in data units."""
        y = panel.y.reset_index(drop=True)
        if panel.exog is None:
            return y
        return pd.concat([panel.exog.reset_index(drop=True), y], axis=1)

    def _adapter(self, panel: PanelData) -> DDFMDataset:
        """The engine's dataset built from ``panel``: the frame ``[exog | y]`` on a RangeIndex.

        The exog columns are the dataset's covariates and the ``y`` columns its
        targets, in that order (the column order of the engine's input). The
        time column name is absent from the frame, so the dataset keeps the
        RangeIndex. The dataset fits the target scaler and scales the
        covariates with a fresh instance of the same class, which it keeps
        (the engine's ``covariate_scaler``).
        """
        frame = self._engine_frame(panel)
        time_col = _TIME_COLUMN
        while time_col in set(frame.columns):
            time_col += "_"
        return DDFMDataset(
            data=frame, time_idx=time_col, covariates=list(panel.exog_names), scaler=self._fresh_scaler()
        )

    def _engine_params(self) -> dict:
        """Every engine hyperparameter, passed explicitly under the engine's names."""
        return {
            "encoder_size": tuple(int(w) for w in self.encoder_layers) + (int(self.n_factors),),
            "decoder_type": self.decoder_type,
            "seed": self.random_state,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "n_mc_samples": self.n_mc_samples,
            "window_size": self.window_size,
            "max_iter": self.max_iter,
            "tolerance": self.tol,
            "disp": _ENGINE_DISP,
            "interpolation_method": self.interpolation_method,
            "interpolation_limit": self.interpolation_limit,
            "interpolation_limit_direction": self.interpolation_limit_direction,
        }

    # -- fit -----------------------------------------------------------------

    def fit(self, data: Any, **fit_kwargs: Any) -> DDFM:
        """Train on ``data`` and return ``self``.

        Pre-trains the autoencoder, runs the MCMC denoising loop and builds
        the linear state space (AR(1) factors, decoder loadings) used by
        ``predict``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            Targets in ``y`` and, optionally, encoder covariates in ``exog``.
            Missing values are allowed (interpolated for the encoder, and
            missing targets imputed during training). Mixed frequency and
            ``series_exog`` are rejected.

        Raises
        ------
        ConfigurationError
            For an invalid hyperparameter.
        UnsupportedInputError
            For ``series_exog`` or a mixed-frequency panel.
        DataValidationError
            For infinite values, or a target series with fewer than 2
            observed values or the same value (up to rounding) at every
            observation.

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
            raise TypeError(f"DDFM.fit() got unexpected keyword arguments {sorted(fit_kwargs)}.")
        self._validate_params()
        device = self._resolve_device()
        dataset = self._adapter(data)
        params = self._engine_params()
        with quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _DIAGNOSTIC_LOGGERS, logging.ERROR):
            if self.verbose >= 2:
                _logger.debug("DDFM: engine hyperparameters %s", params)
            engine = DDFMEngine(dataset=dataset, device=device, **params)
            engine.fit()
            engine.build_state_space()
            result = engine.get_result()
        # The engine keeps no reference to the training data and drops the MCMC
        # buffers; it keeps its scaled training frame (engine.data), which update reads.
        engine._dataset = None
        for name in _TRAINING_BUFFERS:
            engine.__dict__.pop(name, None)

        self.network_ = engine
        self.device_ = device
        self.scaler_ = engine.scaler
        self.result_ = result
        self.n_iter_ = int(engine._num_iter)
        self.converged_ = bool(getattr(engine, "_converged", False))
        self.n_factors_ = int(self.n_factors)
        self.random_state_ = self.random_state
        self.conditioning_index_ = data.index
        self._set_interpretation()

    # -- interpretation ------------------------------------------------------

    def _decoder64(self) -> torch.nn.Module:
        """A float64 CPU copy of the trained decoder."""
        return copy.deepcopy(self.network_.decoder).to(device="cpu", dtype=torch.float64)

    def _training_targets(self) -> np.ndarray:
        """The scaled training targets (T x N, NaN where missing): the first ``len(index_)`` conditioning rows."""
        engine = self.network_
        return engine.data[engine.target_series].to_numpy(dtype=np.float64)[: len(self.index_)]

    def _factor_path(self) -> torch.Tensor:
        """``factors_`` as a float64 tensor (a copy: under pandas copy-on-write the frame's array is read-only)."""
        return torch.tensor(self.factors_.to_numpy(), dtype=torch.float64)

    def _common_scaled(self, decoder: torch.nn.Module) -> np.ndarray:
        """``decoder`` (from :meth:`_decoder64`) applied to ``factors_``: the common component (T x N, model space)."""
        with torch.no_grad():
            return decoder(self._factor_path()).numpy()

    def _set_interpretation(self) -> None:
        """Set ``factor_names_``, ``factors_``, ``loadings_``, ``loadings_kind_`` and ``explained_variance_``.

        Computed from ``result_.Z``, the decoder weights and the scaled
        training targets, so ``load`` recomputes them exactly.
        """
        names = [f"factor_{j}" for j in range(self.n_factors_)]
        self.factor_names_ = np.asarray(names, dtype=object)
        self.factors_ = pd.DataFrame(np.asarray(self.result_.Z, dtype=np.float64), index=self.index_, columns=names)
        decoder = self._decoder64()
        if self.network_.decoder_type == "linear":
            loadings = decoder.get_last_linear_layer().weight.detach().numpy()
            self.loadings_kind_ = "exact"
        else:
            # The output layer is linear, so the mean Jacobian is its weight times the mean Jacobian of
            # the hidden layers. Reverse mode through the hidden layers alone costs memory in the last
            # hidden width per row; through the whole decoder it costs T x N x N.
            hidden = torch.func.vmap(torch.func.jacrev(decoder.get_intermediate()))(self._factor_path())
            loadings = (decoder.get_last_linear_layer().weight @ hidden.mean(dim=0)).detach().numpy()
            self.loadings_kind_ = "mean_jacobian"
        series = pd.Index(self.feature_names_in_)
        self.loadings_ = pd.DataFrame(loadings, index=series, columns=names)
        data = self._training_targets()
        explained = explained_variance(data, self._common_scaled(decoder))
        self.explained_variance_ = pd.Series(explained, index=series, name="explained_variance")

    def _to_data_units(self, values: np.ndarray) -> np.ndarray:
        """Scaled target values in data units (``scaler_.inverse_transform``; unchanged without a scaler)."""
        if self.scaler_ is None:
            return values
        return np.asarray(self.scaler_.inverse_transform(values), dtype=np.float64)

    def decompose(self, *, by: str = "component") -> pd.DataFrame:
        """Split the training targets into the decoder's reconstruction and the residual, in data units.

        Parameters
        ----------
        by : {"component", "factor"}, default "component"
            ``"component"``: MultiIndex columns ``(component, series)`` with
            ``"common"``, the decoder's output on ``factors_`` inverse-scaled
            with ``scaler_`` (so it includes the series mean), and
            ``"idiosyncratic"``, the data minus common (NaN where the data is
            missing).
            ``"factor"`` (linear decoder only): MultiIndex columns ``(part,
            series)``. Writing the scaler as ``data = offset + scale *
            scaled`` per target, factor ``j``'s part of target ``i`` is
            ``scale_i * H_ij * f_jt`` (``H`` is ``loadings_``), ``"mean"`` is
            ``offset_i + scale_i * b_i`` (``b`` the decoder bias) and
            ``"idiosyncratic"`` is as for ``"component"``.
            In both, the parts add up to the data.

        Returns
        -------
        pandas.DataFrame
            float64, indexed by ``index_``, one column per part and target
            (targets in the user's order; the exog columns are not decoded).

        Raises
        ------
        ConfigurationError
            For another ``by``.
        UnsupportedInputError
            For ``by="factor"`` with the MLP decoder (``loadings_kind_`` is
            ``"mean_jacobian"``), or with a scaler that is not affine (other
            than None, ``StandardScaler``, ``RobustScaler``, ``MinMaxScaler``
            and ``MaxAbsScaler``).
        """
        self._check_is_fitted("decompose")
        if by not in ("component", "factor"):
            raise ConfigurationError(f"DDFM.decompose: by must be 'component' or 'factor', got {by!r}.")
        affine = affine_scaler_params(self.scaler_, self.n_features_in_)
        if by == "factor" and self.loadings_kind_ != "exact":
            raise UnsupportedInputError(
                "DDFM.decompose(by='factor') needs targets linear in the factors, but the MLP decoder is nonlinear "
                "(loadings_kind_='mean_jacobian'). Use by='component', or decoder_type='linear'."
            )
        if by == "factor" and affine is None:
            raise UnsupportedInputError(
                f"DDFM.decompose(by='factor') needs an affine scaler (None, 'standard', 'robust', 'minmax' or "
                f"'maxabs'), got {type(self.scaler_).__name__}: the factor parts do not add up in data units. "
                "Use by='component'."
            )
        decoder = self._decoder64()
        common = self._to_data_units(self._common_scaled(decoder))
        idiosyncratic = self._to_data_units(self._training_targets()) - common
        if by == "component":
            parts, level = {"common": common, "idiosyncratic": idiosyncratic}, "component"
        else:
            offset, scale = affine
            factors, loadings = self.factors_.to_numpy(), self.loadings_.to_numpy()
            parts = {name: factors[:, [j]] * (scale * loadings[:, j]) for j, name in enumerate(self.factor_names_)}
            bias = decoder.get_last_linear_layer().bias.detach().numpy()
            parts["mean"] = np.tile(offset + scale * bias, (len(factors), 1))
            parts["idiosyncratic"] = idiosyncratic
            level = "part"
        series = list(self.feature_names_in_)
        frame = pd.concat(
            {key: pd.DataFrame(values, index=self.index_, columns=series) for key, values in parts.items()},
            axis=1,
            names=[level, "series"],
        )
        return self._emit(frame)

    # -- conditioning on new data -------------------------------------------

    def _new_rows(self, data: Any, method: str) -> tuple[pd.DataFrame, pd.Index]:
        """Validate new rows for a fitted model; return them as the engine's frame and the extended index.

        ``data`` must hold the training targets and exog (same names, same
        order), an index of the training index's type and at least one row,
        all strictly after the last conditioning time stamp (a ``RangeIndex``
        must continue the model's).

        Returns
        -------
        frame : pandas.DataFrame
            The new rows as ``[exog | y]`` on a RangeIndex, in data units (the
            engine scales them with the fitted scalers).
        extended : pandas.Index
            ``conditioning_index_`` followed by ``data.index``.
        """
        panel = self._validate_data(data)  # inputs the tags exclude raise UnsupportedInputError first
        expected = list(self.feature_names_in_)
        if list(panel.series_names) != expected:
            raise DataValidationError(
                f"DDFM.{method}: data has series {list(panel.series_names)}, but the model was fitted on "
                f"{expected}. Pass the same series in the same order."
            )
        trained = list(self.exog_names_in_)
        if list(panel.exog_names) != trained:
            raise DataValidationError(
                f"DDFM.{method}: data has exog {list(panel.exog_names)}, but the model was fitted on {trained}. "
                "Pass the same exog columns in the same order."
            )
        if panel.n_timesteps < 1:
            raise DataValidationError(f"DDFM.{method}: data has no time steps.")
        extended = self._extend_conditioning_index(panel.index, f"DDFM.{method}")
        return self._engine_frame(panel), extended

    def _condition(self, engine: DDFMEngine, frame: pd.DataFrame) -> None:
        """Run the engine's ``_update`` on the new rows ``frame`` (see :meth:`update`)."""
        with fit_verbosity(self.verbose), quiet_engine_loggers(self.verbose, _ENGINE_LOGGERS), \
                quiet_engine_loggers(self.verbose, _DIAGNOSTIC_LOGGERS, logging.ERROR):
            engine._update(frame)

    def _rng_devices(self) -> list:
        """The CUDA devices whose generators ``predict(data=...)`` forks besides the CPU one."""
        device = self.device_
        if device.type != "cuda":
            return []
        return [device.index if device.index is not None else torch.cuda.current_device()]

    def update(self, data: Any, *, refit: bool = False) -> DDFM:
        """Condition later forecasts on new rows ``data`` and return ``self``.

        Tag ``update_kind`` is ``"encoder_forward"``: the trained encoder maps
        each new row to its factors, which are appended to the engine's, so
        ``predict(h)`` then starts from the last new row. ``data`` holds only
        the rows after the last conditioning time stamp (tag
        ``predict_data="new_rows"``); overlap raises. The targets are scaled
        with ``scaler_`` and the covariates with the covariate scaler, both
        fitted in ``fit`` and never refitted. Missing values are interpolated
        as in ``fit``, over the conditioning rows and the new rows together;
        gaps the interpolation leaves carry the last value forward, so no NaN
        reaches the encoder.

        The network parameters, ``scaler_``, ``result_`` and the
        interpretation attributes (``factors_``, ``loadings_``,
        ``explained_variance_``; :meth:`decompose` covers the training rows)
        are unchanged; ``conditioning_index_`` is extended by ``data.index``.
        ``predict(h, data=new)`` forecasts from the same factors as
        ``update(new).predict(h)``, without changing the model.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            The new rows of the training targets (and exog), same names and
            order, on an index of the training index's type, strictly after
            ``conditioning_index_``. A ``RangeIndex`` must continue the
            model's index. Missing values are allowed.
        refit : bool, default False
            Not supported: DDFM cannot continue training from new data.

        Raises
        ------
        UnsupportedInputError
            For ``refit=True``, ``series_exog`` or a mixed-frequency panel.
        DataValidationError
            If the series or exog names differ from the training data, the
            index type differs, the rows overlap the conditioning data,
            ``data`` is empty or holds infinite values.
        """
        self._check_is_fitted("update")
        if refit:
            raise UnsupportedInputError(
                "DDFM.update(refit=True) is not supported: DDFM cannot continue training on new data. Use "
                "update(data) (refit=False) to condition the forecasts on it, or fit(data) to retrain from scratch."
            )
        frame, extended = self._new_rows(data, "update")
        self._condition(self.network_, frame)
        self.conditioning_index_ = extended
        return self

    # -- forecasting ---------------------------------------------------------

    def _run_forecast(self, method: str, h: Any, data: Any) -> tuple[np.ndarray, np.ndarray, int, pd.Index]:
        self._check_is_fitted(method)
        h = self._validate_horizon(h)
        if data is None:
            y_forecast, z_forecast = self.network_._forecast(h)
            return y_forecast, z_forecast, h, self.conditioning_index_
        frame, extended = self._new_rows(data, method)
        # update(data) on a copy of the engine; the global torch RNG is restored afterwards.
        with torch.random.fork_rng(devices=self._rng_devices()):
            engine = copy.deepcopy(self.network_)
            self._condition(engine, frame)
            y_forecast, z_forecast = engine._forecast(h)
        return y_forecast, z_forecast, h, extended

    def _frame(self, values: np.ndarray, h: int, columns: list, kind: str, origin_index: pd.Index) -> pd.DataFrame:
        frame = pd.DataFrame(
            np.asarray(values, dtype=np.float64),
            index=self._forecast_index(origin_index, h),
            columns=columns,
        )
        frame.attrs = {"model": type(self).__name__, "kind": kind, "origin": origin_index[-1]}
        return frame

    def predict(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the targets ``h`` steps ahead, in data units.

        The factors follow the fitted AR(1) transition from the last factor
        state, and the targets are their image under the loadings,
        inverse-transformed with ``scaler_`` (the identity with
        ``scaler=None``).

        Parameters
        ----------
        h : int
            Forecast horizon (a positive integer; anything else raises
            ``TypeError``).
        data : PanelData, DataFrame or 2-D array, optional
            New rows to condition on (tag ``predict_data="new_rows"``: only
            the rows strictly after ``conditioning_index_``, as for
            :meth:`update`). This is :meth:`update` on a deep copy of the
            model, followed by the forecast, which starts after
            ``data.index[-1]``; the forecast equals ``update(data)`` followed
            by ``predict(h)``. The model and the global torch RNG are
            unchanged.

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows after the last conditioning time stamp, one
            column per target in the user's order; ``attrs`` holds ``model``,
            ``kind="forecast"`` and ``origin``. A ``polars.DataFrame`` after
            ``set_output(output="polars")`` (so are :meth:`predict_factors`
            and :meth:`decompose`).

        Raises
        ------
        DataValidationError
            For ``data`` that does not hold the training series and exog,
            overlaps the conditioning data or is empty (see :meth:`update`).
        """
        y_forecast, _, h, origin = self._run_forecast("predict", h, data)
        return self._emit(self._frame(y_forecast, h, list(self.feature_names_in_), "forecast", origin))

    def predict_factors(self, h: int, data: Any = None) -> pd.DataFrame:
        """Forecast the factors ``h`` steps ahead (same arguments as :meth:`predict`).

        Returns
        -------
        pandas.DataFrame
            float64, ``h`` rows, columns ``factor_names_``; ``attrs["kind"]``
            is ``"factors"``.
        """
        _, z_forecast, h, origin = self._run_forecast("predict_factors", h, data)
        return self._emit(self._frame(z_forecast, h, list(self.factor_names_), "factors", origin))

    # -- save / load ---------------------------------------------------------

    def _save_state(self, path: Path) -> None:
        """Write the DDFM part of the saved run (spec section 7).

        ``state.npz`` holds the engine's ``factors`` (MC x T x r, the rows
        ``update`` appended included), ``eps``, ``last_neurons``, the
        state-space matrices ``ssm/<name>``, ``data/frame`` (the scaled
        conditioning frame ``[exog | y]``, NaN kept, which ``update``
        interpolates the new rows against) and the covariate scaler's arrays
        under ``covariate_scaler/``. ``state.json`` holds the training
        diagnostics (``loss_now`` is null when training stopped before
        computing it), the engine dims and sizes, ``data_columns``, the
        covariate scaler's JSON part and ``conditioning_index``.
        ``result.npz``/``result.json`` hold ``result_`` without
        ``target_scaler`` (load sets it to ``scaler_``) and ``weights.pt`` the
        state dict.
        """
        engine = self.network_
        state = engine.training_state
        arrays = {
            "factors": engine.factors,
            "eps": engine.eps,
            "last_neurons": engine.last_neurons,
            **{f"ssm/{name}": getattr(state, name) for name in _SSM_FIELDS},
            "data/frame": engine.data.to_numpy(),
        }
        # Keep each array's memory order: the state-space builder leaves F as a
        # column-major view, and the forecast's matrix products round
        # differently on a row-major copy. A column-major copy is written with
        # fortran_order and read back column-major, so a loaded model forecasts
        # bitwise like the fitted one.
        arrays = {key: np.array(value, order="K", copy=True) for key, value in arrays.items()}
        covariate_scaler, covariate_arrays = _encode_covariate_scaler(engine.covariate_scaler)
        arrays.update(covariate_arrays)
        loss_now = getattr(engine, "loss_now", None)
        meta = {
            "num_iter": int(engine._num_iter),
            "loss_now": None if loss_now is None else float(loss_now),
            "converged": bool(getattr(engine, "_converged", False)),
            **{name: getattr(engine, name) for name in _ENGINE_DIMS},
            "encoder_size": list(engine.encoder_size),
            "decoder_size": None if engine.decoder_size is None else list(engine.decoder_size),
            "data_columns": list(engine.data.columns),
            "covariate_scaler": covariate_scaler,
            "conditioning_index": index_to_spec(self.conditioning_index_),
        }
        self._write_state(path, arrays, meta)
        self._write_result(path, exclude=_RESULT_REBUILT)
        self._write_weights(path)

    @classmethod
    def _load_state(cls, est: DDFM, path: Path, device: Any, allow_pickle: bool) -> None:
        """Rebuild the engine without data or seeding, load its weights and restore the fitted attributes.

        The interpretation attributes are recomputed from the restored weights,
        factors and scaled training frame (they are not saved).
        """
        arrays, meta = cls._read_state(path)
        device = est._load_device(device)
        engine = DDFMEngine._from_saved(est._engine_params(), {name: meta[name] for name in _ENGINE_DIMS}, device)
        saved_sizes = (meta["encoder_size"], meta["decoder_size"])
        built_sizes = (list(engine.encoder_size), None if engine.decoder_size is None else list(engine.decoder_size))
        if saved_sizes != built_sizes:
            raise ConfigurationError(
                f"{str(path)!r}: state.json holds encoder/decoder sizes {saved_sizes}, but config.json gives "
                f"{built_sizes}; the run is inconsistent."
            )
        engine.load_state_dict(cls._read_weights(path, device), strict=True)
        engine.factors = arrays["factors"]
        engine.eps = arrays["eps"]
        engine.last_neurons = arrays["last_neurons"]
        engine._num_iter = meta["num_iter"]
        # The engine computes loss_now from the second MCMC iteration on, as a Python float.
        if meta["loss_now"] is not None:
            engine.loss_now = float(meta["loss_now"])
        engine._converged = meta["converged"]
        for name in _SSM_FIELDS:
            setattr(engine.training_state, name, arrays[f"ssm/{name}"])
        engine.data = pd.DataFrame(arrays["data/frame"], columns=list(meta["data_columns"]))
        engine.target_indices = np.array([engine.data.columns.get_loc(col) for col in engine.target_series])
        engine.scaler = est.scaler_
        engine.covariate_scaler = _decode_covariate_scaler(meta.get("covariate_scaler"), arrays, allow_pickle, path)

        est.network_ = engine
        est.device_ = device
        est.result_ = cls._read_result(path, DDFMResult, target_scaler=est.scaler_)
        est.n_iter_ = int(meta["num_iter"])
        est.converged_ = bool(meta["converged"])
        est.n_factors_ = int(est.n_factors)
        est.random_state_ = est.random_state
        est.conditioning_index_ = index_from_spec(meta["conditioning_index"])
        est._set_interpretation()

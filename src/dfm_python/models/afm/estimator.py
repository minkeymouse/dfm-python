"""``AFM``: the Attention Factor Model estimator (sklearn conventions, PanelData input).

A thin layer over the private engine
:class:`~dfm_python.models.afm._engine.AFMEngine`. ``fit`` builds the engine's
dataset from :class:`~dfm_python.PanelData` (returns from ``y``,
characteristics from ``series_exog``), constructs the engine with every
hyperparameter passed explicitly (the engine seeds, then builds its network)
and trains it.

AFM allocates rather than forecasts: :meth:`AFM.allocate` returns portfolio
weights, :meth:`AFM.backtest` evaluates the fitted policy on new data, and
``predict`` raises ``TypeError``. ``save``/``load`` write and read the
pickle-free save directory; a loaded model has no data attached.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ...base import BaseFactorModel
from ...config.schema.model import AFMConfig
from ...config.schema.params import AFMModelState
from ...config.schema.results import AFMResult
from ...dataset.afm_dataset import AFMDataset
from ...dataset.panel import PanelData
from ...io.codecs import read_json, read_npz
from ...io.layout import RESULT_JSON, RESULT_NPZ
from ...io.results import result_from_files
from ...logger import get_logger
from ...logger.progress import fit_verbosity
from ...utils.errors import ConfigurationError, DataValidationError, UnsupportedInputError
from ._engine import AFMEngine

__all__ = ["AFM"]

_FACTOR_MODELS = ("attention", "pca")
_TRADING = ("longconv", "ou")
_POSITIVE_INTS = ("embed_dim", "hist_len", "n_kernels", "pca_window", "reestim", "max_epochs")
_NON_NEGATIVE_FLOATS = ("ridge", "squash_lambda", "lambda_var", "turnover_cost", "short_cost")
# AFMResult fields that hold the same array as residual_portfolios.
_RESULT_ALIASES = ("Z", "x_sm")

_logger = get_logger(__name__)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


class AFM(BaseFactorModel):
    """Attention Factor Model for statistical arbitrage (Epstein, Wang, Choi and Pelger, 2025).

    Learns a conditional factor model whose residuals are traded and a trading
    policy on the residual history, under an after-cost Sharpe objective with
    an explained-variance regularizer. Two switches select the paper's three
    model classes:

    - ``factor_model="attention", trading="longconv"``: Attention Factors
      (needs ``series_exog``, the firm characteristics);
    - ``factor_model="pca", trading="longconv"``: PCA + LongConv benchmark;
    - ``factor_model="pca", trading="ou"``: PCA + Avellaneda-Lee.

    The constructor only stores its arguments; they are validated in ``fit``.

    Parameters
    ----------
    n_factors : int
        Number of factors K. Required (the default None raises in ``fit``).
    factor_model : {"attention", "pca"}, default "attention"
        Attention factors on the characteristics, or rolling-window PCA on the
        returns. PCA does not read ``series_exog``.
    trading : {"longconv", "ou"}, default "longconv"
        Learned LongConv filter, or the Avellaneda-Lee OU rule.
    embed_dim : int, default 32
        Characteristic embedding dimension (attention).
    hist_len : int, default 20
        Residual-history length fed to the trading policy. The first
        ``hist_len`` time steps get no weights.
    n_kernels : int, default 32
        Number of long convolutions (LongConv).
    ridge : float, default 1e-2
        Ridge penalty of the closed-form loadings.
    squash_lambda : float, default 1e-3
        LongConv Squash regularizer strength.
    pca_window : int, default 252
        Trailing window of the PCA benchmark. The first estimate uses the
        first ``min(pca_window, max(2 * n_factors + 1, 30))`` dates; earlier
        dates get zero factor weights (residuals equal the returns).
    reestim : int, default 21
        PCA re-estimation stride.
    lambda_var : float, default 0.1
        Weight of the explained-variance regularizer.
    turnover_cost : float, default 5e-4
        Cost per unit turnover (5 bps).
    short_cost : float, default 1e-4
        Cost per unit short position (1 bp).
    risk_free : float, default 0.0
        Risk-free rate subtracted in the Sharpe ratio.
    learning_rate : float, default 1e-3
        Adam learning rate.
    max_epochs : int, default 50
        Full-batch training epochs.
    patience : int, optional
        Stop after this many epochs without improvement; None trains for
        ``max_epochs``.
    random_state : int, optional
        Seed of the global torch RNG, applied just before the network is
        built. None (default) does not seed, so runs differ.
    device : str or torch.device, default "auto"
        ``"auto"`` uses CUDA if available, else CPU. A ``torch.device`` is
        written to ``config.json`` and YAML as its string.
    verbose : int, default 0
        0 is silent; 1 logs one record when training starts and one when it
        ends (``fit`` and ``update(refit=True)``); 2 also logs the engine
        hyperparameters. Training is full-batch, with no per-epoch log.

    Attributes
    ----------
    result_ : AFMResult
        Residuals, book returns, Sharpe ratio and explained variance of the
        last training pass.
    residuals_ : pandas.DataFrame
        Traded residual portfolios, T x N on ``index_``.
    book_returns_ : pandas.Series
        After-cost book returns on ``index_[hist_len:]``.
    network_ : torch.nn.Module
        The trained engine.
    device_ : torch.device
        The resolved device.
    data_ : PanelData
        The data ``allocate()`` uses: the training data, or the data of the
        last ``update``. A loaded model has none until ``update(data)``.
    n_features_in_, feature_names_in_, exog_names_in_, series_exog_names_in_, index_
        Number and names of the assets, exog names (always empty),
        characteristic names and the training index.
    n_iter_ : int
        Training epochs run.
    converged_ : bool
        True when training stopped early through ``patience``.
    n_factors_ : int
    n_char_ : int or None
        Number of characteristics (None without ``series_exog``).
    scaler_ : None
        AFM does not scale its inputs.
    random_state_ : int or None
        The seed the run used.
    """

    _tags = {
        "task": "allocation",
        "backend": "torch",
        "capability:series_exog": True,
        "capability:update": True,
        "capability:device": True,
        "update_kind": "reattach",
        "predict_data": None,
        "non_deterministic": True,  # the class default random_state=None; see _instance_tags
    }
    _config_class = AFMConfig
    _param_choices = {"factor_model": _FACTOR_MODELS, "trading": _TRADING}

    def __init__(
        self,
        *,
        n_factors: Optional[int] = None,
        factor_model: str = "attention",
        trading: str = "longconv",
        embed_dim: int = 32,
        hist_len: int = 20,
        n_kernels: int = 32,
        ridge: float = 1e-2,
        squash_lambda: float = 1e-3,
        pca_window: int = 252,
        reestim: int = 21,
        lambda_var: float = 0.1,
        turnover_cost: float = 5e-4,
        short_cost: float = 1e-4,
        risk_free: float = 0.0,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        patience: Optional[int] = None,
        random_state: Optional[int] = None,
        device: Any = "auto",
        verbose: int = 0,
    ) -> None:
        self.n_factors = n_factors
        self.factor_model = factor_model
        self.trading = trading
        self.embed_dim = embed_dim
        self.hist_len = hist_len
        self.n_kernels = n_kernels
        self.ridge = ridge
        self.squash_lambda = squash_lambda
        self.pca_window = pca_window
        self.reestim = reestim
        self.lambda_var = lambda_var
        self.turnover_cost = turnover_cost
        self.short_cost = short_cost
        self.risk_free = risk_free
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _instance_tags(self) -> dict:
        return {"non_deterministic": self.random_state is None}

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Small CPU parameter sets for the contract checks: the paper's three model classes."""
        tiny = {"n_factors": 2, "hist_len": 5, "max_epochs": 2, "random_state": 0, "device": "cpu"}
        return [
            {**tiny, "factor_model": "attention", "trading": "longconv", "embed_dim": 4, "n_kernels": 4},
            {**tiny, "factor_model": "pca", "trading": "longconv", "n_kernels": 4, "pca_window": 20, "reestim": 5},
            {**tiny, "factor_model": "pca", "trading": "ou", "pca_window": 20, "reestim": 5},
        ]

    def _describe_fitted(self) -> dict:
        return {"sharpe": self.result_.sharpe}

    def _validate_params(self) -> None:
        """Raise ``ConfigurationError`` naming the first invalid hyperparameter."""
        name = type(self).__name__
        if self.n_factors is None:
            raise ConfigurationError(f"{name}: n_factors is required; pass a positive integer, e.g. {name}(n_factors=3).")
        if not _is_int(self.n_factors) or self.n_factors < 1:
            raise ConfigurationError(f"{name}: n_factors must be a positive integer, got {self.n_factors!r}.")
        if self.factor_model not in _FACTOR_MODELS:
            raise ConfigurationError(f"{name}: factor_model must be one of {_FACTOR_MODELS}, got {self.factor_model!r}.")
        if self.trading not in _TRADING:
            raise ConfigurationError(f"{name}: trading must be one of {_TRADING}, got {self.trading!r}.")
        for param in _POSITIVE_INTS:
            value = getattr(self, param)
            if not _is_int(value) or value < 1:
                raise ConfigurationError(f"{name}: {param} must be a positive integer, got {value!r}.")
        for param in _NON_NEGATIVE_FLOATS:
            value = getattr(self, param)
            if not _is_real(value) or not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name}: {param} must be a finite number >= 0, got {value!r}.")
        if not _is_real(self.risk_free) or not math.isfinite(self.risk_free):
            raise ConfigurationError(f"{name}: risk_free must be a finite number, got {self.risk_free!r}.")
        if not _is_real(self.learning_rate) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConfigurationError(f"{name}: learning_rate must be a finite number > 0, got {self.learning_rate!r}.")
        if self.patience is not None and (not _is_int(self.patience) or self.patience < 1):
            raise ConfigurationError(f"{name}: patience must be None or a positive integer, got {self.patience!r}.")
        if self.random_state is not None and not _is_int(self.random_state):
            raise ConfigurationError(f"{name}: random_state must be None or an integer, got {self.random_state!r}.")
        if not _is_int(self.verbose) or self.verbose < 0:
            raise ConfigurationError(
                f"{name}: verbose must be a non-negative integer (0 silent, 1 progress, 2 debug), got {self.verbose!r}."
            )
        self._resolve_device()

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

    @staticmethod
    def _require_characteristics(panel: PanelData, factor_model: str) -> None:
        if factor_model == "attention" and not panel.series_exog_names:
            raise UnsupportedInputError(
                "AFM with factor_model='attention' needs firm characteristics in series_exog "
                "(PanelData(..., series_exog=...) or PanelData.from_arrays(..., series_exog=chars)); got none. "
                "Pass them, or use factor_model='pca'."
            )

    def _panel(self, data: Any, method: str) -> PanelData:
        """Validate ``data`` for a fitted model: same assets and, for attention factors, same characteristics."""
        panel = self._validate_data(data)
        expected = list(self.feature_names_in_)
        if list(panel.series_names) != expected:
            raise DataValidationError(
                f"AFM.{method}: data has series {list(panel.series_names)}, but the model was fitted on {expected}. "
                "Pass the same assets in the same order."
            )
        trained = list(self.series_exog_names_in_)
        if (self.network_.factor_model == "attention" and panel.series_exog_names
                and list(panel.series_exog_names) != trained):
            raise DataValidationError(
                f"AFM.{method}: data has series_exog features {list(panel.series_exog_names)}, but the model was "
                f"fitted on {trained}. Pass the same characteristics in the same order."
            )
        self._require_characteristics(panel, self.network_.factor_model)
        return panel

    @staticmethod
    def _adapter(panel: PanelData) -> AFMDataset:
        """The engine's dataset: returns (T, N) from ``y``, characteristics (T, N, M) from ``series_exog``.

        Both arrays are made C-contiguous: ``DataFrame.to_numpy`` can return a
        Fortran-ordered array, whose strided float32 tensor takes other BLAS
        paths and rounds differently.
        """
        returns = np.ascontiguousarray(panel.y.to_numpy(np.float64))
        characteristics = panel.series_exog_array()
        if characteristics is not None:
            characteristics = np.ascontiguousarray(characteristics)
        return AFMDataset(returns=returns, characteristics=characteristics)

    def _engine_params(self) -> dict:
        """Every engine hyperparameter, passed explicitly (``random_state`` is the engine's ``seed``)."""
        return {
            "n_factors": self.n_factors,
            "embed_dim": self.embed_dim,
            "hist_len": self.hist_len,
            "n_kernels": self.n_kernels,
            "ridge": self.ridge,
            "squash_lambda": self.squash_lambda,
            "pca_window": self.pca_window,
            "reestim": self.reestim,
            "factor_model": self.factor_model,
            "trading": self.trading,
            "lambda_var": self.lambda_var,
            "turnover_cost": self.turnover_cost,
            "short_cost": self.short_cost,
            "risk_free": self.risk_free,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.random_state,
        }

    def fit(self, data: Any, **fit_kwargs: Any) -> AFM:
        """Train on ``data`` and return ``self``.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array
            Asset returns in ``y`` (T x N) and, for attention factors, firm
            characteristics in ``series_exog``. ``exog``, mixed frequency and
            missing values are rejected.

        Raises
        ------
        ConfigurationError
            For an invalid hyperparameter.
        UnsupportedInputError
            For ``exog``, mixed frequency, NaN, or attention factors without
            ``series_exog``.

        A failed fit leaves the model unfitted (earlier fitted attributes are
        removed).
        """
        return super().fit(data, **fit_kwargs)

    def _fit(self, data: PanelData, **fit_kwargs: Any) -> None:
        if fit_kwargs:
            raise TypeError(f"AFM.fit() got unexpected keyword arguments {sorted(fit_kwargs)}.")
        self._validate_params()
        self._require_characteristics(data, self.factor_model)
        device = self._resolve_device()
        dataset = self._adapter(data)
        # The engine seeds the global torch RNG (random_state) and then builds its network.
        engine = AFMEngine(dataset, **self._engine_params(), device=device)
        self._train(engine, dataset, lambda: engine.fit())
        self._set_fitted(engine, device)
        self.data_ = data
        self._set_results(data.index)

    def _train(self, engine: AFMEngine, dataset: AFMDataset, run: Any) -> None:
        """Call ``run()`` (the engine's training) between the progress records of ``verbose``."""
        if self.verbose >= 1:
            _logger.info(
                "AFM: training factor_model=%r, trading=%r, n_factors=%d on T=%d, N=%d for up to %d epochs on %s.",
                engine.factor_model, engine.trading, engine.n_factors, dataset.T, dataset.N, engine.max_epochs, engine.device,
            )
        if self.verbose >= 2:
            _logger.debug("AFM: engine hyperparameters %s", self._engine_params())
        run()
        if self.verbose >= 1:
            state = engine.training_state
            if not any(p.requires_grad for p in engine.parameters()):
                stop = "no trainable parameters"
            elif state.converged:
                stop = f"early stop (patience={engine.patience})"
            else:
                stop = "no early stop"
            _logger.info(
                "AFM: finished after %d epoch(s), %s; loss %.6g, Sharpe %.6g, explained variance %.6g.",
                state.num_iter, stop, state.loss_now, state.sharpe, state.explained_variance,
            )

    def _set_fitted(self, engine: AFMEngine, device: torch.device) -> None:
        """Set the fitted attributes that come from the engine (shared by ``fit`` and ``load``)."""
        self.network_ = engine
        self.device_ = device
        self.n_factors_ = int(engine.n_factors)
        self.n_char_ = None if engine.n_char is None else int(engine.n_char)
        self.scaler_ = None
        self.random_state_ = self.random_state

    def _set_results(self, index: pd.Index) -> None:
        """Set ``result_`` and the attributes derived from it from the engine's last training pass."""
        engine = self.network_
        result = engine.get_result()
        self.result_ = result
        self.n_iter_ = int(result.num_iter)
        self.converged_ = bool(result.converged)
        columns = list(self.feature_names_in_)
        self.residuals_ = pd.DataFrame(
            np.asarray(result.residual_portfolios[0], dtype=np.float64), index=index, columns=columns
        )
        self.book_returns_ = pd.Series(
            np.asarray(result.book_returns[0], dtype=np.float64), index=index[engine.hist_len:], name="book_return"
        )

    def allocate(self, data: Any = None) -> pd.DataFrame:
        """Portfolio weights of the fitted policy.

        Parameters
        ----------
        data : PanelData, DataFrame or 2-D array, optional
            The panel to allocate on (same assets and characteristics as in
            training). None uses the attached data (``data_``).

        Returns
        -------
        pandas.DataFrame
            float64 weights, index ``data.index[hist_len:]`` (row *i* holds the
            weights applied to the returns at that timestamp), one column per
            asset; ``attrs`` holds ``model``, ``kind="weights"`` and ``origin``
            (the last timestamp of ``data``). A ``polars.DataFrame`` after
            ``set_output(output="polars")``.
        """
        self._check_is_fitted("allocate")
        if data is None:
            if "data_" not in vars(self):
                raise DataValidationError(
                    "AFM has no attached data: a loaded model keeps no training data. Pass the panel to allocate on, "
                    "allocate(data=...), or attach it once with update(data)."
                )
            panel = self.data_
        else:
            panel = self._panel(data, "allocate")
        out = self.network_.predict(data=self._adapter(panel))  # runs under torch.no_grad()
        frame = pd.DataFrame(
            np.asarray(out["weights"][0], dtype=np.float64),
            index=panel.index[self.network_.hist_len:],
            columns=list(self.feature_names_in_),
        )
        frame.attrs = {"model": type(self).__name__, "kind": "weights", "origin": panel.index[-1]}
        return self._emit(frame)

    def backtest(self, data: Any) -> AFMResult:
        """Evaluate the fitted policy on ``data`` without training.

        Returns
        -------
        AFMResult
            Residual portfolios ``(1, T, N)``, weights ``(1, T - hist_len, N)``,
            after-cost book returns ``(1, T - hist_len)``, Sharpe ratio and
            explained variance computed on ``data``; ``num_iter`` and
            ``converged`` describe the fitted model.
        """
        self._check_is_fitted("backtest")
        panel = self._panel(data, "backtest")
        engine = self.network_
        returns, characteristics = engine._as_batch(self._adapter(panel))
        with torch.no_grad():
            out = engine.forward(returns, characteristics)
        residuals = out["residuals"].cpu().numpy()
        sharpe = float(out["sharpe"])
        return AFMResult(
            Z=residuals,
            x_sm=residuals,
            converged=self.converged_,
            num_iter=self.n_iter_,
            objective=sharpe,
            residual_portfolios=residuals,
            book_returns=out["book_returns"].cpu().numpy(),
            sharpe=sharpe,
            explained_variance=float(out["ev"]),
            factor_model=engine.factor_model,
            trading=engine.trading,
            num_factors=engine.n_factors,
            weights=out["weights"].cpu().numpy(),
        )

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """AFM does not forecast; always raises ``TypeError``."""
        raise TypeError("AFM allocates; use allocate() for portfolio weights or backtest(data) to evaluate them.")

    def update(self, data: Any, *, refit: bool = False) -> AFM:
        """Attach ``data`` for later ``allocate()``; with ``refit=True`` also continue training on it.

        ``refit=True`` trains from the current weights (the engine's
        ``update(..., retrain=True)``) and refreshes ``result_``,
        ``residuals_``, ``book_returns_``, ``n_iter_``, ``converged_`` and
        ``index_``. Returns ``self``.
        """
        self._check_is_fitted("update")
        panel = self._panel(data, "update")
        if refit:
            dataset = self._adapter(panel)
            with fit_verbosity(self.verbose):
                self._train(self.network_, dataset, lambda: self.network_.update(dataset, retrain=True))
            self._set_data_attributes(panel)
            self._set_results(panel.index)
        self.data_ = panel
        return self

    def _save_state(self, path: Path) -> None:
        """Write the AFM part of the saved run.

        - ``state.npz``: ``residuals`` and ``book_returns`` of the last training
          pass; ``state.json``: the training diagnostics and ``n_char``;
        - ``result.npz`` / ``result.json``: ``result_`` without ``Z`` and
          ``x_sm``, which alias ``residual_portfolios`` and are restored as such;
        - ``weights.pt``: ``network_.state_dict()``.
        """
        state = self.network_.training_state
        arrays = {"residuals": state.residuals, "book_returns": state.book_returns}
        meta = {
            "num_iter": state.num_iter,
            "converged": state.converged,
            "loss_now": state.loss_now,
            "sharpe": state.sharpe,
            "explained_variance": state.explained_variance,
            "n_char": self.n_char_,
        }
        self._write_state(path, arrays, meta)
        self._write_result(path, exclude=_RESULT_ALIASES)
        self._write_weights(path)

    @classmethod
    def _load_state(cls, est: AFM, path: Path, device: Any, allow_pickle: bool) -> None:
        """Rebuild the engine without data or seeding, load its weights and restore the fitted attributes."""
        arrays, meta = cls._read_state(path)
        device = est._load_device(device)
        engine = AFMEngine._from_saved(est._engine_params(), meta["n_char"], device)
        engine.load_state_dict(cls._read_weights(path, device))
        state = AFMModelState(
            num_iter=meta["num_iter"],
            converged=meta["converged"],
            loss_now=meta["loss_now"],
            sharpe=meta["sharpe"],
            explained_variance=meta["explained_variance"],
            residuals=arrays["residuals"],
            book_returns=arrays["book_returns"],
        )
        state.apply_to_model(engine)
        engine.training_state = state

        result_arrays = read_npz(Path(path) / RESULT_NPZ)
        residuals = result_arrays.get("residual_portfolios")
        engine._result = result_from_files(
            AFMResult,
            result_arrays,
            read_json(Path(path) / RESULT_JSON),
            **{name: residuals for name in _RESULT_ALIASES},
        )
        est._set_fitted(engine, device)
        est._set_results(est.index_)  # result_ is the engine's (cached) result, as after fit

"""Attention Factor Model (AFM) for statistical arbitrage.

Implements Epstein, Wang, Choi and Pelger (2025), "Attention Factors for
Statistical Arbitrage" (arXiv:2510.11616). The model jointly learns a
conditional factor model whose residuals are traded portfolios and a
residual-history trading policy, under an after-cost Sharpe objective with an
explained-variance regularizer:

    max_{omega_F, omega_port}  Sharpe(R_net) + lambda_var * EV,   s.t. ||omega||_1 = 1.

Two switches cover the paper's three model classes:
  * ``factor_model='attention', trading='longconv'`` -> Attention Factors;
  * ``factor_model='pca',       trading='longconv'`` -> PCA + LongConv benchmark;
  * ``factor_model='pca',       trading='ou'``       -> PCA + Avellaneda--Lee.

This is a from-the-paper implementation (no reference code exists) meant to be
correct and runnable on modest data, not a reproduction of the paper's CRSP
study, which needs licensed point-in-time data.

``AFMEngine`` is the private engine behind the public estimator
:class:`dfm_python.models.afm.estimator.AFM`, which builds its dataset from
``PanelData`` and passes every hyperparameter explicitly. The estimator saves
and loads it through :meth:`AFMEngine._from_saved`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ...config.schema.results import AFMResult
from ...config.schema.params import AFMModelState
from ...utils.errors import (ConfigurationError, ModelNotTrainedError,
                             DataValidationError)
from ...utils.run_control import RunControl
from ...utils.seed import apply_seed
from .factors import AttentionFactors, PCAFactors, residuals, rolling_pca_weights
from .longconv import LongConv1d
from .baselines import avellaneda_lee_weights

_EPS = 1e-8

#: The hyperparameters ``__init__`` stores, which ``_from_saved`` restores.
_HYPERPARAMS = ("factor_model", "trading", "n_factors", "embed_dim", "hist_len",
                "n_kernels", "ridge", "squash_lambda", "pca_window", "reestim",
                "lambda_var", "turnover_cost", "short_cost", "risk_free",
                "learning_rate", "max_epochs", "patience", "seed")


def _init_engine_state(engine: Any) -> None:
    engine._result = None
    engine.training_state = None
    engine._dataset = None


class AFMEngine(nn.Module):
    """Attention Factor Model engine.

    Parameters
    ----------
    dataset : AFMDataset, optional
        Exposes ``returns`` (T, N) and, for attention, ``characteristics``
        (T, N, M); None builds nothing until ``n_factors`` (and ``n_char``
        for attention) are known.
    n_char, n_factors, embed_dim, hist_len, n_kernels, ridge : int/float
        Architecture. ``hist_len`` is the residual history length ``s``.
    factor_model : {'attention', 'pca'}
    trading : {'longconv', 'ou'}
    lambda_var, turnover_cost, short_cost, risk_free : float
        Objective and cost parameters (defaults are the paper's 5bps/1bp).
    learning_rate, max_epochs, patience, seed
        Training controls. ``seed`` seeds the global torch RNG before the
        network is built.
    device : torch.device or str, optional
        None uses CUDA if available, else CPU.
    """

    def __init__(self, dataset: Optional[Any] = None, *,
                 n_char: Optional[int] = None, n_factors: Optional[int] = None,
                 embed_dim: int = 32, hist_len: int = 20, n_kernels: int = 32,
                 ridge: float = 1e-2, squash_lambda: float = 1e-3,
                 pca_window: int = 252, reestim: int = 21,
                 factor_model: str = "attention", trading: str = "longconv",
                 lambda_var: float = 0.1, turnover_cost: float = 5e-4,
                 short_cost: float = 1e-4, risk_free: float = 0.0,
                 learning_rate: float = 1e-3, max_epochs: int = 50,
                 patience: Optional[int] = None, seed: Optional[int] = None,
                 device: Optional[torch.device] = None) -> None:
        _init_engine_state(self)
        nn.Module.__init__(self)

        self.factor_model = factor_model
        self.trading = trading
        self.n_factors = n_factors
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
        self.seed = seed

        if self.factor_model not in ("attention", "pca"):
            raise ConfigurationError(
                f"factor_model must be 'attention' or 'pca', got '{self.factor_model}'")
        if self.trading not in ("longconv", "ou"):
            raise ConfigurationError(
                f"trading must be 'longconv' or 'ou', got '{self.trading}'")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if self.seed is not None:
            apply_seed(int(self.seed))

        self._dataset = dataset
        if dataset is not None:
            self._infer_dims(dataset, n_char)
        else:
            self.n_char = n_char
        self._built = False
        if self.n_factors is not None and (self.factor_model == "pca" or
                                            self.n_char is not None):
            self._build_components()

        # Training diagnostics (read by AFMModelState.from_model).
        self.loss_now: Optional[float] = None
        self._num_iter: int = 0
        self._converged: bool = False
        self._sharpe: Optional[float] = None
        self._ev: Optional[float] = None

    @classmethod
    def _from_saved(cls, params: dict, n_char: Optional[int],
                    device: Any) -> AFMEngine:
        """Rebuild an engine without ``__init__``: no dataset, no seeding.

        ``params`` holds the ``_HYPERPARAMS``. The components are built on
        ``device`` inside ``torch.random.fork_rng``, so their initial draws do
        not consume the caller's CPU RNG; load the saved weights over them.
        """
        with torch.random.fork_rng(devices=[]):
            self = cls.__new__(cls)
            _init_engine_state(self)
            nn.Module.__init__(self)
            for name in _HYPERPARAMS:
                setattr(self, name, params[name])
            self.device = torch.device(device)
            self.n_char = n_char
            self._built = False
            self.loss_now = None
            self._num_iter = 0
            self._converged = False
            self._sharpe = None
            self._ev = None
            self._build_components()
        return self

    def _infer_dims(self, dataset: Any, n_char: Optional[int]) -> None:
        self.n_char = n_char if n_char is not None else getattr(dataset, "n_char", None)
        if self.n_factors is None:
            self.n_factors = getattr(dataset, "n_factors", None)

    def _build_components(self) -> None:
        if self.n_factors is None or self.n_factors < 1:
            raise ConfigurationError("n_factors must be a positive integer")
        if self.factor_model == "attention":
            if self.n_char is None:
                raise ConfigurationError("attention factors require n_char")
            self.factor_block: Any = AttentionFactors(
                self.n_char, self.n_factors, self.embed_dim, self.ridge)
        else:
            self.factor_block = PCAFactors(self.n_factors, self.ridge)
        if self.trading == "longconv":
            self.filter: Optional[nn.Module] = LongConv1d(
                self.hist_len, self.n_kernels, self.squash_lambda)
        else:
            self.filter = None
        self.to(self.device)
        self._built = True

    def _factor_weights(self, returns: torch.Tensor,
                        characteristics: Optional[torch.Tensor]):
        """Per-date ``omega_f`` (B, W, K, N) and ``beta_t`` (B, W, N, K)."""
        if self.factor_model == "attention":
            if characteristics is None:
                raise DataValidationError("attention factors require characteristics")
            return self.factor_block(characteristics)
        # The paper's benchmark re-estimates PCA on a rolling trailing window.
        return rolling_pca_weights(returns, self.n_factors, self.pca_window,
                                   self.reestim, self.ridge)

    def forward(self, returns: torch.Tensor,
                characteristics: Optional[torch.Tensor] = None) -> dict:
        """Compute residuals, book returns, and the training objective.

        Parameters
        ----------
        returns : torch.Tensor
            Asset returns, shape ``(B, W, N)``.
        characteristics : torch.Tensor, optional
            Firm characteristics ``(B, W, N, M)`` (required for attention).

        Returns
        -------
        dict
            ``loss`` (scalar), ``sharpe``, ``ev``, ``residuals`` (B, W, N),
            ``book_returns`` (B, n_eval), ``weights`` (B, n_eval, N).
        """
        if returns.dim() != 3:
            raise DataValidationError(
                f"returns must be (B, W, N), got shape {tuple(returns.shape)}")
        B, W, N = returns.shape
        if W <= self.hist_len:
            raise DataValidationError(
                f"window length {W} must exceed hist_len {self.hist_len}")

        returns = returns.to(self.device)
        if characteristics is not None:
            characteristics = characteristics.to(self.device)

        omega_f, beta_t = self._factor_weights(returns, characteristics)
        eps = residuals(omega_f, beta_t, returns)                  # (B, W, N)

        book, weights = [], []
        prev_w = torch.zeros(B, N, device=returns.device, dtype=returns.dtype)
        for w in range(self.hist_len, W):
            hist = eps[:, w - self.hist_len:w, :].transpose(1, 2)  # (B, N, s)
            if self.trading == "longconv":
                omega_port = self.filter(hist)                     # (B, N)
            else:
                omega_port = avellaneda_lee_weights(hist)          # (B, N)
            of, bt = omega_f[:, w - 1], beta_t[:, w - 1]           # (B,K,N),(B,N,K)
            # omega_eps^T omega_port = omega_port - omega_F^T (beta^T omega_port)
            z = (bt.transpose(-1, -2) @ omega_port.unsqueeze(-1)).squeeze(-1)  # (B,K)
            omega_asset = omega_port - (of.transpose(-1, -2) @ z.unsqueeze(-1)).squeeze(-1)
            norm = omega_asset.abs().sum(-1, keepdim=True) + _EPS
            omega_asset = omega_asset / norm                       # ||omega||_1 = 1
            r_port = (returns[:, w] * omega_asset).sum(-1)         # (B,)
            cost = (self.turnover_cost * (omega_asset - prev_w).abs().sum(-1)
                    + self.short_cost * torch.relu(-omega_asset).sum(-1))
            book.append(r_port - cost)
            weights.append(omega_asset)
            prev_w = omega_asset

        book_returns = torch.stack(book, dim=1)                    # (B, n_eval)
        weights = torch.stack(weights, dim=1)                      # (B, n_eval, N)

        mean = book_returns.mean(-1) - self.risk_free
        std = book_returns.std(-1) + _EPS
        sharpe = (mean / std).mean()
        var_eps = eps.var(dim=1)                                   # (B, N)
        var_r = returns.var(dim=1) + _EPS
        ev = (1.0 - var_eps / var_r).mean()
        loss = -(sharpe + self.lambda_var * ev)
        return {"loss": loss, "sharpe": sharpe, "ev": ev, "residuals": eps,
                "book_returns": book_returns, "weights": weights}

    def fit(self, control: Optional[RunControl] = None) -> AFMModelState:
        """Train full-batch by maximizing Sharpe + lambda_var * EV on the attached dataset.

        Training stops after ``max_epochs`` epochs, on ``patience``, after one
        epoch when nothing is trainable, or when ``control`` (optional; called
        after every epoch with the epoch's loss) asks for a stop before those;
        ``_stop_reason`` records which (``"converged"`` when nothing is
        trainable).
        """
        if self._dataset is None:
            raise ConfigurationError("no dataset attached; pass one to __init__ or update")
        if not self._built:
            self._build_components()
        returns, characteristics = self._as_batch(self._dataset)
        params = [p for p in self.parameters() if p.requires_grad]
        from ...numeric.builder import build_afm_optimizer
        optimizer = build_afm_optimizer(params, self.learning_rate) if params else None

        best = float("inf")
        no_improve = 0
        last = None
        self._converged = False
        self._stop_reason = "max_iter"
        for epoch in range(int(self.max_epochs)):
            out = self.forward(returns, characteristics)
            loss = out["loss"]
            if optimizer is not None and torch.is_grad_enabled() and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
            last = out
            self.loss_now = float(loss.detach())
            self._num_iter = epoch + 1
            if self.loss_now < best - 1e-6:
                best = self.loss_now
                no_improve = 0
            else:
                no_improve += 1
            halt = None if control is None else control.step(epoch + 1, self.loss_now)
            if optimizer is None:                                  # nothing to learn
                self._stop_reason = "converged"
                break
            if self.patience is not None and no_improve >= self.patience:
                self._converged = True
                self._stop_reason = "early_stopping"
                break
            if halt is not None and epoch + 1 < self.max_epochs:
                self._stop_reason = halt
                break

        self._sharpe = float(last["sharpe"].detach())
        self._ev = float(last["ev"].detach())
        self._residuals = last["residuals"].detach().cpu().numpy()
        self._book_returns = last["book_returns"].detach().cpu().numpy()
        self.training_state = AFMModelState.from_model(self)
        self._result = None
        return self.training_state

    def predict(self, *, data: Optional[Any] = None) -> dict:
        """Residuals, weights and book returns (numpy) on ``data`` or the attached dataset.

        AFM is not a horizon forecaster: the trading output is its prediction.
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        ds = data if data is not None else self._dataset
        if ds is None:
            raise DataValidationError("no data to predict on")
        returns, characteristics = self._as_batch(ds)
        with torch.no_grad():
            out = self.forward(returns, characteristics)
        return {"residuals": out["residuals"].cpu().numpy(),
                "weights": out["weights"].cpu().numpy(),
                "book_returns": out["book_returns"].cpu().numpy()}

    def _as_batch(self, ds: Any):
        """The dataset's full panel as a batch of one: (1, T, N) and (1, T, N, M) or None."""
        returns, characteristics = ds.get_tensors(self.device)
        if returns.dim() == 2:
            returns = returns.unsqueeze(0)
        if characteristics is not None and characteristics.dim() == 3:
            characteristics = characteristics.unsqueeze(0)
        return returns, characteristics

    def update(self, data: Any, *, retrain: bool = False, control: Optional[RunControl] = None) -> None:
        """Attach ``data``; with ``retrain`` continue training on it from the current weights (``fit(control)``)."""
        self._dataset = data
        self._infer_dims(data, getattr(self, "n_char", None))
        if retrain:
            self.fit(control)

    def get_result(self) -> AFMResult:
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        if self._result is not None:
            return self._result
        st = self.training_state
        self._result = AFMResult(
            Z=st.residuals, x_sm=st.residuals, converged=st.converged,
            num_iter=st.num_iter, objective=st.sharpe,
            residual_portfolios=st.residuals, book_returns=st.book_returns,
            sharpe=st.sharpe, explained_variance=st.explained_variance,
            factor_model=self.factor_model, trading=self.trading,
            num_factors=self.n_factors)
        return self._result

"""Result dataclasses: ``BaseResult`` and the per-model ``DFMResult``, ``DDFMResult``, ``iVDFMResult``, ``AFMResult``."""

import numpy as np
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .params import DDFMModelState


@dataclass
class BaseResult(ABC):
    """Model-agnostic result fields shared by the linear and deep models.

    ``Z`` holds the factors (T x r, or the state T x m) and ``x_sm`` the
    reconstruction or smoothed series (T x N), both in model space. The
    state-space fields ``C``, ``R``, ``A``, ``Q``, ``Z_0``, ``V_0`` are set by
    the linear models (DFM, DDFM) only.
    """
    x_sm: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    r: Optional[np.ndarray] = None         # factors per block, or [num_factors]
    p: Optional[int] = None                # AR order / factor order
    target_scaler: Optional[Any] = None    # scaler of the targets, for inverse transforms
    converged: bool = False
    num_iter: int = 0                      # epochs or iterations, model-specific
    objective: Optional[float] = None      # loglik / ELBO / loss, model-specific
    loglik: Optional[float] = None

    C: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    Z_0: Optional[np.ndarray] = None
    V_0: Optional[np.ndarray] = None

    @property
    def factors(self) -> Optional[np.ndarray]:
        """Alias of ``Z``."""
        return self.Z

    @property
    def reconstructed(self) -> Optional[np.ndarray]:
        """Alias of ``x_sm``."""
        return self.x_sm

    @property
    def x_hat(self) -> Optional[np.ndarray]:
        """Alias of ``x_sm``."""
        return self.x_sm

    def num_series(self) -> int:
        """Columns of ``x_sm``."""
        if self.x_sm is None:
            raise ValueError("x_sm is not available in this result")
        return int(self.x_sm.shape[1])

    def num_state(self) -> int:
        """Columns of ``Z``."""
        if self.Z is None:
            raise ValueError("Z is not available in this result")
        return int(self.Z.shape[1])

    def num_periods(self) -> int:
        """Rows of ``Z`` (or of ``x_sm``)."""
        if self.Z is not None:
            return int(self.Z.shape[0])
        if self.x_sm is not None:
            return int(self.x_sm.shape[0])
        raise ValueError("Neither Z nor x_sm is available in this result")

    def num_factors(self) -> int:
        """``sum(r)``, or the state dimension when ``r`` is not set."""
        if self.r is not None:
            try:
                return int(np.sum(self.r))
            except (TypeError, ValueError):  # r holds something that does not sum to a number
                pass
        return self.num_state()

    def get_x_sm_original_scale(self, target_scaler: Optional[Any] = None) -> np.ndarray:
        """``x_sm`` inverse-transformed by ``target_scaler`` (default ``self.target_scaler``), or as is when there is none."""
        if self.x_sm is None:
            raise ValueError("x_sm is not available in this result")
        scaler = target_scaler if target_scaler is not None else self.target_scaler
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            return scaler.inverse_transform(self.x_sm)
        return self.x_sm

    def to_pandas_factors(self, time_index: Optional[object] = None, factor_names: Optional[List[str]] = None):
        """``Z`` as a DataFrame (columns ``F1..``), with a ``time`` column when ``time_index`` is iterable."""
        import pandas as pd
        cols = factor_names or [f"F{i+1}" for i in range(self.num_state())]
        if self.Z is None:
            raise ValueError("Z is not available in this result")
        df_dict = {col: self.Z[:, i] for i, col in enumerate(cols)}
        if time_index is not None:
            if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                df_dict['time'] = list(time_index)
        return pd.DataFrame(df_dict)

    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None, target_scaler: Optional[Any] = None):
        """``get_x_sm_original_scale`` as a DataFrame (columns ``S1..``), with a ``time`` column when ``time_index`` is iterable."""
        import pandas as pd
        x_sm_original = self.get_x_sm_original_scale(target_scaler)
        cols = series_ids or [f"S{i+1}" for i in range(self.num_series())]
        df_dict = {col: x_sm_original[:, i] for i, col in enumerate(cols)}
        if time_index is not None:
            if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                df_dict['time'] = list(time_index)
        return pd.DataFrame(df_dict)

    def summary(self) -> str:
        """Formatted summary: dimensions, factor structure and training diagnostics."""
        model_type = self.__class__.__name__.replace('Result', '')

        lines = ["=" * 80, f"{model_type} Model Summary", "=" * 80, ""]

        lines.append("Data Dimensions:")
        # The num_* methods raise ValueError when the array they read is not set.
        try:
            lines.append(f"  Series: {self.num_series()}")
        except ValueError:
            lines.append("  Series: N/A")
        try:
            lines.append(f"  Factors: {self.num_factors()} (state dim: {self.num_state()})")
        except ValueError:
            lines.append("  Factors: N/A")
        try:
            lines.append(f"  Time periods: {self.num_periods()}")
        except ValueError:
            lines.append("  Time periods: N/A")
        lines.append("")

        lines.append("Factor Structure:")
        if self.r is not None and hasattr(self.r, "__len__") and len(self.r) > 0:
            lines.append(f"  Factors per block: {self.r}")
        else:
            try:
                lines.append(f"  Total factors: {self.num_factors()}")
            except ValueError:
                lines.append("  Total factors: N/A")
        lines.append(f"  AR order: {self.p if self.p is not None else 'N/A'}")
        lines.append("")

        lines.append("Training Diagnostics:")
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Iterations: {self.num_iter}")
        if self.objective is not None:
            lines.append(f"  Objective: {self.objective:.6f}")
        if self.loglik is not None:
            lines.append(f"  Log-likelihood: {self.loglik:.4f}")
        lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class DFMResult(BaseResult):
    """DFM (EM) estimation results, exposed by the :class:`~dfm_python.DFM` estimator as ``result_``.

    ``converged`` and ``num_iter`` refer to the EM iterations. The
    series-indexed matrices (``C``, ``R``, ``x_sm``) are in the estimator's
    internal order ``internal_feature_names_`` (``[clock | slower]``), not
    the user's column order; ``res.Z @ res.C.T`` equals ``res.x_sm``. The
    ``training_*`` fields are never set.
    """
    training_max_iter: Optional[int] = None
    training_threshold: Optional[float] = None
    training_regularization_scale: Optional[float] = None

    C: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    Z_0: Optional[np.ndarray] = None
    V_0: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Base summary plus the ``training_*`` hyperparameters that are set."""
        lines = super().summary().split("\n")

        # Insert before the final separator
        insert_idx = len(lines) - 1
        training_info = []
        if self.training_max_iter is not None:
            training_info.append(f"  Max iterations: {self.training_max_iter}")
        if self.training_threshold is not None:
            training_info.append(f"  Convergence threshold: {self.training_threshold:.6f}")
        if self.training_regularization_scale is not None:
            training_info.append(f"  Regularization scale: {self.training_regularization_scale:.6f}")

        if training_info:
            lines[insert_idx:insert_idx] = ["Training Hyperparameters:", *training_info, ""]

        return "\n".join(lines)


@dataclass
class DDFMResult(BaseResult):
    """DDFM estimation results (``DDFM.result_``).

    ``DDFM.fit`` sets the BaseResult fields from the engine's state space:
    ``Z`` (T x r, factors averaged over the MC samples), ``x_sm`` (T x N, the
    reconstruction ``Z @ C.T`` in model space), ``A`` (the AR(1) transition
    F), ``C`` (the loadings H, N x r, the DFM orientation), ``Q``, ``R``,
    ``Z_0``, ``V_0``, ``r``, ``p``, ``target_scaler`` (the estimator's
    ``scaler_``), ``converged`` (the MCMC denoising loop stopped on ``tol``
    before ``max_iter``), ``num_iter`` (MCMC iterations) and ``loglik``
    (-inf; DDFM computes no log-likelihood).

    ``training_loss`` is the last MCMC loss, ``encoder_layers`` and the
    ``training_*`` fields hold the training settings. ``use_idiosyncratic``
    and ``fit_params`` are never set (always None).
    """
    training_loss: Optional[float] = None
    encoder_layers: Optional[List[int]] = None
    use_idiosyncratic: Optional[bool] = None
    fit_params: Optional[DDFMModelState] = None

    training_max_iter: Optional[int] = None
    training_tolerance: Optional[float] = None
    training_n_mc_samples: Optional[int] = None
    training_window_size: Optional[int] = None
    training_learning_rate: Optional[float] = None
    training_optimizer: Optional[str] = None
    training_decay_learning_rate: Optional[bool] = None
    training_encoder_size: Optional[tuple] = None
    training_decoder_size: Optional[tuple] = None
    training_target_series: Optional[List[str]] = None
    training_use_bias: Optional[bool] = None
    training_batch_norm: Optional[bool] = None
    training_activation: Optional[str] = None

    def summary(self) -> str:
        """Base summary plus the network training fields that are set."""
        lines = super().summary().split("\n")

        # Insert before the final separator
        insert_idx = len(lines) - 1
        ddfm_info = []
        if self.training_loss is not None:
            ddfm_info.append(f"  Final training loss: {self.training_loss:.4f}")
        if self.encoder_layers is not None:
            ddfm_info.append(f"  Encoder architecture: {self.encoder_layers}")
        if self.training_encoder_size is not None:
            ddfm_info.append(f"  Encoder size: {self.training_encoder_size}")
        if self.training_decoder_size is not None:
            ddfm_info.append(f"  Decoder size: {self.training_decoder_size}")
        if self.training_max_iter is not None:
            ddfm_info.append(f"  Max MCMC iterations: {self.training_max_iter}")
        if self.training_tolerance is not None:
            ddfm_info.append(f"  Convergence tolerance: {self.training_tolerance:.6f}")
        if self.training_n_mc_samples is not None:
            ddfm_info.append(f"  MC samples per iteration: {self.training_n_mc_samples}")
        if self.training_window_size is not None:
            ddfm_info.append(f"  Window size: {self.training_window_size}")
        if self.training_learning_rate is not None:
            ddfm_info.append(f"  Learning rate: {self.training_learning_rate:.6f}")
        if self.training_optimizer is not None:
            ddfm_info.append(f"  Optimizer: {self.training_optimizer}")
        if self.training_decay_learning_rate is not None:
            ddfm_info.append(f"  Learning rate decay: {self.training_decay_learning_rate}")

        if ddfm_info:
            lines[insert_idx:insert_idx] = ["Neural Network Training:", *ddfm_info, ""]

        return "\n".join(lines)


@dataclass
class iVDFMResult(BaseResult):
    """iVDFM results: learned innovations and factors, decoder reconstructions, diagnostics and a weights-only checkpoint.

    iVDFM is not a linear state-space model, so the ``C``/``R``/``Q`` fields
    of BaseResult stay unset.
    """
    innovations: Optional[np.ndarray] = None          # (T, r) or (batch, T, r)
    reconstructions: Optional[np.ndarray] = None      # decoded factors, (T, N) or (batch, T, N)
    full_state: Optional[np.ndarray] = None           # companion-form state: (T x r*p) for p>1, (T x r) for p=1

    # Regimes and mixing (when num_regimes > 1 or mixing=True)
    num_regimes: Optional[int] = None                # K; 1 = baseline
    regime_temperature: Optional[float] = None        # softmax temperature of pi
    regime_weights: Optional[np.ndarray] = None      # (T, K) or (num_windows, K) pi_t from regime_net
    mixing: Optional[bool] = None                    # whether post-factor mixing f -> M f was used
    mixing_matrix: Optional[np.ndarray] = None       # (r, r) learned M when mixing=True

    training_elbo: Optional[float] = None
    training_loss: Optional[float] = None
    num_epochs: Optional[int] = None

    config: Optional[Dict[str, Any]] = None  # plain-python configuration snapshot
    model_state_dict: Optional[Dict[str, Any]] = None  # weights-only checkpoint (torch tensors)

    def summary(self) -> str:
        lines = ["=" * 80, "iVDFM Result Summary", "=" * 80, ""]
        if self.factors is not None:
            lines.append(f"Factors shape: {getattr(self.factors, 'shape', None)}")
        if self.innovations is not None:
            lines.append(f"Innovations shape: {getattr(self.innovations, 'shape', None)}")
        if self.reconstructions is not None:
            lines.append(f"Reconstructions shape: {getattr(self.reconstructions, 'shape', None)}")
        if self.full_state is not None:
            lines.append(f"Full state (augmented) shape: {getattr(self.full_state, 'shape', None)}")
        if self.num_regimes is not None:
            lines.append(f"Num regimes: {self.num_regimes}")
        if self.regime_temperature is not None:
            lines.append(f"Regime temperature: {self.regime_temperature}")
        if self.mixing is not None:
            lines.append(f"Mixing: {self.mixing}")
        if self.mixing_matrix is not None:
            lines.append(f"Mixing matrix shape: {getattr(self.mixing_matrix, 'shape', None)}")
        lines.append("")
        lines.append("Training:")
        lines.append(f"  Converged: {self.converged}")
        if self.num_epochs is not None:
            lines.append(f"  Epochs: {self.num_epochs}")
        if self.training_loss is not None:
            lines.append(f"  Final loss: {self.training_loss:.6f}")
        if self.training_elbo is not None:
            lines.append(f"  Final ELBO: {self.training_elbo:.6f}")
        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


@dataclass
class AFMResult(BaseResult):
    """AFM results (statistical-arbitrage residual trading).

    AFM is not a linear state-space model; its artifacts are the traded
    residual portfolios, the after-cost book returns and the objective values.
    ``Z`` and ``x_sm`` alias the residuals so the common accessors work.
    """
    residual_portfolios: Optional[np.ndarray] = None   # (B, T, N) traded residuals
    book_returns: Optional[np.ndarray] = None          # (B, n_eval) after-cost returns
    sharpe: Optional[float] = None                     # after-cost Sharpe (objective)
    explained_variance: Optional[float] = None         # EV regularizer value
    factor_model: Optional[str] = None                 # 'attention' | 'pca'
    trading: Optional[str] = None                      # 'longconv' | 'ou'
    num_factors: Optional[int] = None
    weights: Optional[np.ndarray] = None               # (B, n_eval, N) asset weights (set by AFM.backtest)

    def summary(self) -> str:
        lines = ["=" * 80, "AFM Result Summary", "=" * 80, ""]
        lines.append(f"Factor model: {self.factor_model}, trading: {self.trading}")
        if self.num_factors is not None:
            lines.append(f"Num factors: {self.num_factors}")
        if self.sharpe is not None:
            lines.append(f"After-cost Sharpe (objective): {self.sharpe:.4f}")
        if self.explained_variance is not None:
            lines.append(f"Explained variance: {self.explained_variance:.4f}")
        if self.book_returns is not None:
            lines.append(f"Book returns shape: {getattr(self.book_returns, 'shape', None)}")
        lines += ["", "=" * 80]
        return "\n".join(lines)

"""Private engine of the identifiable variational dynamic factor model (iVDFM).

``iVDFMEngine`` is the pre-redesign ``iVDFM`` class behind the public
estimator :class:`dfm_python.models.ivdfm.estimator.iVDFM`, which builds the
engine's dataset from ``PanelData`` and passes every hyperparameter
explicitly. Identifiability comes from the iVAE conditions applied to the
innovation process that drives the factor dynamics (conditional
exponential-family priors on the innovations, not on the states).

Compared with the old public class: ``predict`` is split into
:meth:`iVDFMEngine._future_time_features` and :meth:`iVDFMEngine._forecast`;
:meth:`iVDFMEngine._from_saved` rebuilds a saved run without ``__init__``
(no dataset, no seeding); ``set_dataset``, the ``torch.save`` checkpoint
methods, the raw-data branch of ``update`` and
``hyperparameter_optimization`` (see :func:`dfm_python.tuning.tune_hyperparameters`)
are gone, and the engine no longer inherits the deleted engine base class:
:func:`_init_engine_state` sets the attributes that base set.
"""

import copy
import dataclasses
import time
from typing import Optional, Any, Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ...logger import get_logger
from ...logger.progress import iVDFMTrainLogger
from ...config.constants import DEFAULT_LOSS_LOG_PRECISION
from ...config.types import to_numpy
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_SEED,
    DEFAULT_DTYPE,
    DEFAULT_TOLERANCE,
    DEFAULT_IVDFM_LATENT_DIM,
    DEFAULT_REGULARIZATION,
)
from ...utils.errors import ModelNotTrainedError, ConfigurationError, DataValidationError
from ...utils.run_control import RunControl
from ...utils.seed import apply_seed
from ...utils.loss import compute_elbo_loss
from ...layer.pca import extract_pca_factors
from .encoder import iVDFMInnovationEncoder
from .decoder import iVDFMDecoder
from .prior import iVDFMPriorNetwork
from ...layer.regime import RegimeNet
from ...layer.mixing import build_factor_mixing
from ...ssm.companion import iVDFMCompanionSSM
from ...dataset.ivdfm_dataset import iVDFMDataset
from ...config.schema.model import _iVDFMEngineConfig
from ...config.schema.params import iVDFMModelState
from ...config.schema.results import iVDFMResult
from ...numeric.builder import build_ivdfm_optimizer

_logger = get_logger(__name__)


def _init_engine_state(engine: Any) -> None:
    """Set the attributes the old ``models.base.BaseFactorModel.__init__`` set."""
    engine._config = None
    engine._result = None
    engine.training_state = None
    engine._dataset = None
    engine.data_processed = None


class iVDFMEngine(nn.Module):
    """Identifiable variational dynamic factor model: iVAE identifiability on the innovations of AR(p) factors.

    Parameters
    ----------
    dataset : iVDFMDataset, optional
        Training data; ``data_dim``, ``context_dim`` and ``window`` default to its.
    data_dim : int, optional
        Dimension N of the observations.
    num_factors : int, optional
        Number r of factors.
    context_dim : int, optional
        Width of u_t (time features + context columns); inferred from the dataset in ``fit``.
    window : int, optional
        Window length T; None uses the whole series.
    config : _iVDFMEngineConfig or dict, optional
        Base values; the explicit arguments and ``kwargs`` override it.
    encoder_hidden_dim, decoder_hidden_dim, prior_hidden_dim : int or list of int
    encoder_n_hidden_layers, decoder_n_hidden_layers, prior_n_hidden_layers : int
    decoder_type : str, optional
        'linear', 'residual' or 'mlp'.
    activation : str
    slope : float
        Leaky ReLU slope.
    factor_order : int
        AR order p.
    innovation_distribution : str
    decoder_var : float
        Observation noise variance.
    learning_rate, optimizer, batch_size, max_epochs : optional
        None keeps the config value.
    tolerance : float
    device : torch.device, optional
        None means CUDA if available.
    seed : int
        Seeds the global torch and numpy RNGs before the networks are built.
    **kwargs
        Further config fields (``num_regimes``, ``patience``, the scheduler, ...).

    The components are built in ``__init__`` when ``data_dim`` and
    ``context_dim`` are known, else in ``fit``.
    """

    def __init__(
        self,
        dataset: Optional["iVDFMDataset"] = None,
        data_dim: Optional[int] = None,
        num_factors: Optional[int] = None,
        context_dim: Optional[int] = None,
        window: Optional[int] = None,
        config: Optional[_iVDFMEngineConfig] = None,
        encoder_hidden_dim: Union[int, list] = 200,
        encoder_n_hidden_layers: int = 2,
        decoder_type: Optional[str] = None,
        decoder_hidden_dim: Union[int, list] = 200,
        decoder_n_hidden_layers: int = 2,
        prior_hidden_dim: Union[int, list] = 100,
        prior_n_hidden_layers: int = 1,
        activation: str = 'lrelu',
        slope: float = 0.1,
        factor_order: int = 1,
        innovation_distribution: str = 'laplace',
        decoder_var: float = 0.01,
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        tolerance: float = DEFAULT_TOLERANCE,
        device: Optional[torch.device] = None,
        seed: int = DEFAULT_SEED,
        **kwargs
    ):
        _init_engine_state(self)
        nn.Module.__init__(self)

        # Precedence: kwargs > explicit non-None arguments > config > the config defaults.
        if config is None:
            config_dict = {}
        elif isinstance(config, _iVDFMEngineConfig):
            config_dict = dataclasses.asdict(config)
        elif isinstance(config, dict):
            config_dict = config.copy()
        else:
            raise ConfigurationError(
                f"config must be _iVDFMEngineConfig instance or dict, got {type(config)}"
            )
        explicit = {
            'num_factors': num_factors,
            'encoder_hidden_dim': encoder_hidden_dim,
            'encoder_n_hidden_layers': encoder_n_hidden_layers,
            'decoder_type': decoder_type,
            'decoder_hidden_dim': decoder_hidden_dim,
            'decoder_n_hidden_layers': decoder_n_hidden_layers,
            'prior_hidden_dim': prior_hidden_dim,
            'prior_n_hidden_layers': prior_n_hidden_layers,
            'activation': activation,
            'slope': slope,
            'factor_order': factor_order,
            'innovation_distribution': innovation_distribution,
            'decoder_var': decoder_var,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'tolerance': tolerance,
            'seed': seed,
            'data_dim': data_dim,
            'context_dim': context_dim,
            'window': window,
        }
        config_dict.update({k: v for k, v in explicit.items() if v is not None})
        config_dict.update(kwargs)

        if dataset is not None:
            self._dataset = dataset
            if data_dim is None:
                data_dim = dataset.data_dim
            if context_dim is None:
                context_dim = dataset.context_dim
            if window is None:
                window = dataset.window

        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        self._config = _iVDFMEngineConfig.from_dict(config_dict) if config_dict else _iVDFMEngineConfig()
        self._read_config()

        self.data_dim = data_dim  # None until fit infers it
        # context_dim is not a config field: it is time_context + the context columns, known from the dataset.
        self.context_dim = context_dim
        self.window = window if window is not None else self._config.window

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.seed = seed
        if seed is not None:
            apply_seed(seed, torch_global=True, numpy_global=True)

        if self.data_dim is not None and self.context_dim is not None:
            self._build_components()
        else:
            self.innovation_encoder = None
            self.prior_network = None
            self.prior_networks = None
            self.decoder = None
            self.ssm = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.factors: Optional[np.ndarray] = None
        self.innovations: Optional[np.ndarray] = None
        self.regime_weights: Optional[np.ndarray] = None
        self.factor_path: Optional[np.ndarray] = None  # (T, r): the conditioning series encoded as one sequence

        self.to(self.device)

    @classmethod
    def _from_saved(cls, params: Dict[str, Any], dims: Dict[str, Any], device: Any) -> 'iVDFMEngine':
        """Rebuild an engine without ``__init__``: no dataset, no seeding.

        ``params`` holds the keyword arguments the estimator passes to
        ``__init__``; ``_config`` is built from them as ``__init__`` builds it,
        except that None values are kept (the estimator sets them on a fitted
        engine too). ``dims`` holds the values training sets: ``data_dim``,
        ``context_dim``, ``time_context`` (after fit, the width of u_t) and
        ``window``. The components are built on ``device`` inside
        ``torch.random.fork_rng`` (their builders seed and draw from the global
        torch RNG, which is restored afterwards). The engine has no optimizer,
        scheduler or training state and is in eval mode, as after ``fit``;
        load the saved weights over the components.
        """
        device = torch.device(device)
        config_dict = {k: v for k, v in params.items() if v is not None}
        config = _iVDFMEngineConfig.from_dict(config_dict) if config_dict else _iVDFMEngineConfig()
        fields = {f.name for f in dataclasses.fields(config)}
        dropped = {k: v for k, v in params.items() if v is None and k in fields}
        if dropped:
            config = dataclasses.replace(config, **dropped)

        self = cls.__new__(cls)
        _init_engine_state(self)
        nn.Module.__init__(self)
        self._config = config
        self._read_config()
        self.data_dim = dims["data_dim"]
        self.context_dim = dims["context_dim"]
        self.time_context = dims["time_context"]
        self.window = dims["window"]
        self.device = device
        self.seed = params.get("seed")

        # The builders call torch.manual_seed (all devices), so every CUDA generator is forked too.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            self._build_components()
        self.to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.factors = None
        self.innovations = None
        self.regime_weights = None
        self.factor_path = None
        self._num_iter = 0
        self.loss_now = None
        self._elbo = None
        self._converged = False
        self.eval()
        return self

    def _read_config(self) -> None:
        """Copy the hyperparameters of ``_config`` onto the engine's attributes."""
        cfg = self._config
        self.latent_dim = cfg.num_factors or DEFAULT_IVDFM_LATENT_DIM
        self.time_context = cfg.time_context
        self.encoder_hidden_dim = cfg.encoder_hidden_dim
        self.encoder_n_hidden_layers = cfg.encoder_n_hidden_layers
        self.decoder_type = cfg.decoder_type
        self.decoder_hidden_dim = cfg.decoder_hidden_dim
        self.decoder_n_hidden_layers = cfg.decoder_n_hidden_layers
        self.prior_hidden_dim = cfg.prior_hidden_dim
        self.prior_n_hidden_layers = cfg.prior_n_hidden_layers
        self.activation = cfg.activation
        self.slope = cfg.slope
        self.factor_order = cfg.factor_order
        self.innovation_distribution = cfg.innovation_distribution
        self.decoder_var = cfg.decoder_var
        self.num_regimes = cfg.num_regimes
        self.regime_temperature = cfg.regime_temperature
        self.beta_kl = cfg.beta_kl
        self.use_layer_norm = cfg.use_layer_norm
        self.mixing = cfg.mixing
        self.learning_rate = cfg.learning_rate
        self.optimizer_type = cfg.optimizer
        self.optimizer_weight_decay = cfg.optimizer_weight_decay
        self.optimizer_momentum = cfg.optimizer_momentum
        self.batch_size = cfg.batch_size
        self.max_epochs = cfg.max_epochs
        self.tolerance = cfg.tolerance
        self.patience = cfg.patience  # early stopping; None disables it
        self.scheduler_type = cfg.scheduler_type
        self.scheduler_step_size = cfg.scheduler_step_size
        self.scheduler_gamma = cfg.scheduler_gamma
        self.scheduler_patience = cfg.scheduler_patience
        self.scheduler_factor = cfg.scheduler_factor
        self.scheduler_min_lr = cfg.scheduler_min_lr

    def _set_random_f0(self) -> None:
        with torch.no_grad():
            self.ssm.f0.data = torch.randn(self.latent_dim, device=self.device) * 0.1

    def _normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """Pad ``f0`` to ``latent_dim`` and fix its sign (sum >= 0) for reproducibility."""
        if f0.shape[0] < self.latent_dim:
            f0 = np.pad(f0, (0, self.latent_dim - f0.shape[0]))
        if np.sum(f0) < 0:
            f0 = -f0
        return f0

    def _normalize_factors_shape(self, factors: np.ndarray) -> np.ndarray:
        """(batch, T, r) averaged over the batch to (T, r); (T, r) unchanged."""
        if factors.ndim == 3:
            return np.mean(factors, axis=0)
        return factors

    def _compute_full_state(self, z_np: np.ndarray) -> np.ndarray:
        """The companion-form state: the factors (T, r) for p = 1, else (T, r*p) with the lags.

        s_t[i*p : (i+1)*p] = [f_t[i], f_{t-1}[i], ..., f_{t-p+1}[i]], with f0 before t = 0.
        """
        T, r = z_np.shape
        if self.factor_order == 1:
            return z_np

        full_state = np.zeros((T, r * self.factor_order), dtype=z_np.dtype)
        f0_np = self.ssm.f0.data.cpu().numpy()
        for t in range(T):
            for i in range(r):
                for lag in range(self.factor_order):
                    idx = t - lag
                    if idx >= 0:
                        full_state[t, i * self.factor_order + lag] = z_np[idx, i]
                    elif i < len(f0_np):
                        full_state[t, i * self.factor_order + lag] = f0_np[i]
        return full_state

    def _set_ssm_ar_params(self, ar_coeffs: np.ndarray, B_diag: np.ndarray) -> None:
        """Set the SSM's AR coefficients (r, p) and innovation scales (r).

        For ``factor_order == 1`` the SSM applies a sigmoid to its AR
        parameter, so the coefficients are stored as logits (clipped to
        [0.01, 0.99] first). With K > 1 regimes, regime 0 of ``A_diag``
        (K, p, r) and ``B_diag`` (K, r) gets the values and the other regimes
        a copy plus N(0, 0.02²) noise.
        """
        if self.factor_order == 1:
            a = np.clip(np.asarray(ar_coeffs, dtype=np.float64), 0.01, 0.99)
            ar_coeffs = np.log(a / (1.0 - a)).astype(np.asarray(ar_coeffs).dtype)
        with torch.no_grad():
            if self.factor_order == 1:
                A_flat = ar_coeffs[:, 0] if ar_coeffs.ndim > 1 else ar_coeffs
                self.ssm.A.data = torch.tensor(A_flat, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            else:
                self.ssm.ar_coeffs.data = torch.tensor(
                    ar_coeffs, dtype=DEFAULT_TORCH_DTYPE, device=self.device
                )
                self.ssm.B.data = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            if self.A_diag is not None and self.B_diag is not None:
                K = self.A_diag.shape[0]
                r = self.latent_dim
                ar_t = torch.tensor(ar_coeffs, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                if ar_t.dim() == 1:
                    ar_t = ar_t.unsqueeze(0).expand(r, -1)
                B_t = torch.tensor(B_diag, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
                self.A_diag.data[0] = ar_t.permute(1, 0)  # (p, r)
                self.B_diag.data[0] = B_t
                for k in range(1, K):
                    self.A_diag.data[k] = self.A_diag.data[0].clone() + torch.randn_like(self.A_diag.data[0], device=self.device) * 0.02
                    self.B_diag.data[k] = self.B_diag.data[0].clone() + torch.randn_like(self.B_diag.data[0], device=self.device) * 0.02

    def _build_components(self):
        """Build the encoder, the prior(s), the decoder(s), the optional mixing, the SSM and the regime AR parameters."""
        self.innovation_encoder = iVDFMInnovationEncoder(
            data_dim=self.data_dim,
            latent_dim=self.latent_dim,
            aux_dim=self.context_dim,
            hidden_dim=self.encoder_hidden_dim,
            n_hidden_layers=self.encoder_n_hidden_layers,
            activation=self.activation,
            slope=self.slope,
            use_layer_norm=self.use_layer_norm,
            device=self.device,
            seed=self.seed,
        )

        def make_prior() -> iVDFMPriorNetwork:
            return iVDFMPriorNetwork(
                aux_dim=self.context_dim,
                latent_dim=self.latent_dim,
                hidden_dim=self.prior_hidden_dim,
                n_hidden_layers=self.prior_n_hidden_layers,
                activation=self.activation,
                slope=self.slope,
                innovation_distribution=self.innovation_distribution,
                device=self.device,
                seed=self.seed,
            )

        def make_decoder() -> iVDFMDecoder:
            return iVDFMDecoder(
                latent_dim=self.latent_dim,
                data_dim=self.data_dim,
                decoder_type=self.decoder_type,
                hidden_dim=self.decoder_hidden_dim,
                n_hidden_layers=self.decoder_n_hidden_layers,
                activation=self.activation,
                slope=self.slope,
                use_layer_norm=self.use_layer_norm,
                decoder_var=self.decoder_var,
                device=self.device,
                seed=self.seed,
            )

        # K = 1: one prior p(η_t | u_t) and one decoder; K > 1: regime-indexed priors p_k and
        # decoders mixed by the regime network's π(u_t).
        K = self.num_regimes
        if K == 1:
            self.prior_network = make_prior()
            self.prior_networks = None
            self.regime_net = None
            self.decoders = None
            self.decoder = make_decoder()
        else:
            self.prior_network = None
            self.prior_networks = nn.ModuleList([make_prior() for _ in range(K)])
            self.regime_net = RegimeNet(
                input_dim=self.context_dim,
                num_regimes=K,
                hidden_dim=32,
                n_hidden_layers=1,
                activation=self.activation,
                slope=self.slope,
                device=self.device,
                seed=self.seed,
            )
            self.decoder = None
            self.decoders = nn.ModuleList([make_decoder() for _ in range(K)])

        # Optional global post-factor mixing f -> M f (identity init).
        self.factor_mixing = (
            build_factor_mixing(self.latent_dim, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            if self.mixing
            else None
        )

        self.ssm = iVDFMCompanionSSM(
            latent_dim=self.latent_dim,
            factor_order=self.factor_order,
            device=self.device,
        )

        # Regime-indexed AR(p): A_diag (K, p, r), B_diag (K, r); per window A_eff = Σ_k π_k A_diag[k].
        if K > 1:
            p = self.factor_order
            init_scale = 0.1
            self.A_diag = nn.Parameter(torch.randn(K, p, self.latent_dim, device=self.device) * init_scale)
            self.B_diag = nn.Parameter(torch.randn(K, self.latent_dim, device=self.device) * init_scale)
        else:
            self.A_diag = None
            self.B_diag = None

    def _effective_regime_coeffs(self, pi_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """A_eff (batch, r, p) = Σ_k π_k A_diag[k] and B_eff (batch, r) = Σ_k π_k B_diag[k] for ``pi_w`` (batch, K)."""
        A_eff = (pi_w.unsqueeze(2).unsqueeze(3) * self.A_diag.unsqueeze(0)).sum(dim=1)  # (batch, p, r)
        A_eff = A_eff.permute(0, 2, 1)  # (batch, r, p)
        B_eff = (pi_w.unsqueeze(2) * self.B_diag.unsqueeze(0)).sum(dim=1)  # (batch, r)
        return A_eff, B_eff

    def _decode(
        self,
        factors: torch.Tensor,
        u_1T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode factors (T, r) or (batch, T, r), after the optional mixing; K > 1 mixes the decoders by π(u_1T), uniform without u_1T."""
        squeeze_out = factors.dim() == 2
        if squeeze_out:
            factors = factors.unsqueeze(0)
        mixed = self.factor_mixing(factors) if self.factor_mixing is not None else factors
        if self.num_regimes == 1:
            out = self.decoder(mixed)
        else:
            if u_1T is not None:
                if u_1T.dim() == 2:
                    u_1T = u_1T.unsqueeze(0)
                pi = self.regime_net(u_1T, temperature=self.regime_temperature)
            else:
                batch, T, _ = mixed.shape
                pi = torch.full(
                    (batch, T, self.num_regimes),
                    1.0 / self.num_regimes,
                    device=mixed.device,
                    dtype=mixed.dtype,
                )
            out = torch.zeros(
                mixed.shape[0],
                mixed.shape[1],
                self.data_dim,
                device=mixed.device,
                dtype=mixed.dtype,
            )
            for k in range(self.num_regimes):
                out = out + pi[:, :, k : k + 1] * self.decoders[k](mixed)
        if squeeze_out:
            out = out.squeeze(0)
        return out

    def _sample_prior(self, u: torch.Tensor) -> torch.Tensor:
        """Sample innovations (batch, T, r) from the prior; K > 1 draws r_t ~ Cat(π(u_t)), then η_t ~ p_{r_t}(· | u_t)."""
        if self.num_regimes == 1:
            return self.prior_network.sample(u)
        pi = self.regime_net(u, temperature=self.regime_temperature)  # (batch, T, K)
        batch, T, _ = u.shape
        eta_list = []
        for t in range(T):
            pi_t = pi[:, t, :]  # (batch, K)
            r = torch.multinomial(pi_t + 1e-8, 1).squeeze(-1)  # (batch,)
            eta_t = torch.zeros(batch, self.latent_dim, device=u.device, dtype=u.dtype)
            for k in range(self.num_regimes):
                mask = r == k
                if mask.any():
                    u_t = u[mask, t : t + 1, :]  # (n, 1, context_dim)
                    eta_t[mask] = self.prior_networks[k].sample(u_t).squeeze(1)
            eta_list.append(eta_t)
        return torch.stack(eta_list, dim=1)  # (batch, T, r)

    def _initialize_f0_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Set f_0 to the mean PCA factor of the last window of the data (random on failure)."""
        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total
        if T_init < 2:
            self._set_random_f0()
            _logger.warning(f"Insufficient data for PCA init (T={T_init}). Using random f_0.")
            return

        max_components = min(self.data_dim, T_init, self.latent_dim)
        try:
            y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
            f_init, _ = extract_pca_factors(y_win, n_components=max_components)
            f0_mean = np.mean(f_init, axis=0)  # (max_components,)
        except (np.linalg.LinAlgError, ValueError) as e:  # the PCA's failures (SVD, non-finite data)
            self._set_random_f0()
            _logger.warning(f"PCA init failed: {e}. Using random f_0.")
            return

        f0_mean = self._normalize_f0(f0_mean)
        with torch.no_grad():
            self.ssm.f0.data = torch.tensor(f0_mean, dtype=DEFAULT_TORCH_DTYPE, device=self.device)

        _logger.info(
            f"Initialized f_0 using PCA. "
            f"f_0 range: [{f0_mean.min():.4f}, {f0_mean.max():.4f}]"
        )

    def _initialize_ar_coeffs_from_data(self, dataset: 'iVDFMDataset') -> None:
        """Initialize the AR(p) coefficients by OLS on the PCA factors of the last window.

        The window has min(data_dim, T_init, latent_dim) PCA factors. With
        ``factor_order > 1`` and one regime, the factors beyond those keep the
        SSM's own (non-data) initialization of their AR coefficients and
        innovation scale (an OLS fit on all-zero padded factors would give
        all-zero coefficient rows). With ``factor_order == 1``, or with
        ``num_regimes > 1`` (whose regime coefficients start from the OLS
        values and trained finitely), the padded factors get the OLS values of
        a zero series.
        """
        from ...numeric.estimator import estimate_arp_ols

        T_total = len(dataset.data)
        T_init = min(self.window, T_total) if self.window is not None else T_total

        if T_init < self.factor_order + 1:
            _logger.warning(f"Insufficient data for AR init (T={T_init}, order={self.factor_order}). Skipping.")
            return

        try:
            y_win = dataset.data[T_total - T_init:T_total, :]  # (T_init, N)
            max_components = min(self.data_dim, T_init, self.latent_dim)
            factors, _ = extract_pca_factors(y_win, n_components=max_components)
            n_data = factors.shape[1]
            keep_default = self.factor_order > 1 and n_data < self.latent_dim and self.num_regimes == 1

            if factors.shape[1] < self.latent_dim and not keep_default:
                factors = np.pad(factors, ((0, 0), (0, self.latent_dim - factors.shape[1])))

            ar_coeffs, B_diag = estimate_arp_ols(
                factors=factors,
                order=self.factor_order,
                regularization=DEFAULT_REGULARIZATION,
                dtype=DEFAULT_DTYPE
            )

            if keep_default:
                ar_default = self.ssm.ar_coeffs.detach().cpu().numpy().astype(ar_coeffs.dtype)
                B_default = self.ssm.B.detach().cpu().numpy().astype(B_diag.dtype)
                ar_coeffs = np.concatenate([ar_coeffs, ar_default[n_data:]], axis=0)
                B_diag = np.concatenate([B_diag, B_default[n_data:]], axis=0)
                _logger.info(
                    f"AR({self.factor_order}) initialization: {n_data} PCA factors for {self.latent_dim} "
                    f"factors; factors {n_data}..{self.latent_dim - 1} keep the default initialization."
                )

            self._set_ssm_ar_params(ar_coeffs, B_diag)

            _logger.info(
                f"Initialized AR({self.factor_order}) coefficients using OLS from PCA factors. "
                f"AR coeffs range: [{ar_coeffs.min():.4f}, {ar_coeffs.max():.4f}], "
                f"B range: [{B_diag.min():.4f}, {B_diag.max():.4f}]"
            )
        except (np.linalg.LinAlgError, ValueError) as e:  # the PCA's or the OLS's failures
            _logger.warning(f"AR coefficient initialization failed: {e}. Using random initialization.")

    def _build_optimizer(self):
        self.optimizer, self.scheduler = build_ivdfm_optimizer(
            model=self,
            learning_rate=self.learning_rate,
            optimizer_type=self.optimizer_type,
            max_epochs=self.max_epochs,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_momentum=self.optimizer_momentum,
            scheduler_type=self.scheduler_type,
            scheduler_step_size=self.scheduler_step_size,
            scheduler_gamma=self.scheduler_gamma,
            scheduler_patience=self.scheduler_patience,
            scheduler_factor=self.scheduler_factor,
            scheduler_min_lr=self.scheduler_min_lr,
        )

    def forward(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode, sample and propagate the innovations of a batch of windows, then decode.

        ``y_1T`` is (batch, T, N) and ``u_1T`` (batch, T, context_dim). Returns
        ``y_pred`` (batch, T, N), ``eta`` and ``factors`` (batch, T, r),
        ``encoder_params`` and ``prior_params`` (one dict per time step) and,
        with K > 1, ``regime_weights`` (batch, T, K) and
        ``prior_params_per_regime`` (K lists of T dicts).
        """
        batch_size, T, _ = y_1T.shape

        mu_all, logvar_all = self.innovation_encoder.forward(y_1T, u_1T)  # (batch, T, r)
        std_all = torch.exp(0.5 * logvar_all)
        eps = torch.randn_like(mu_all)
        eta_1T = mu_all + eps * std_all  # (batch, T, r)

        encoder_params_list = [
            {'mu': mu_all[:, t, :], 'logvar': logvar_all[:, t, :]}
            for t in range(T)
        ]
        if self.num_regimes == 1:
            prior_params_all = self.prior_network(u_1T)
            prior_keys = list(prior_params_all.keys())
            prior_params_list = [
                {key: prior_params_all[key][:, t, :] for key in prior_keys}
                for t in range(T)
            ]
            regime_weights = None
            prior_params_per_regime = None
        else:
            pi = self.regime_net(u_1T, temperature=self.regime_temperature)  # (batch, T, K)
            prior_params_per_k = [self.prior_networks[k](u_1T) for k in range(self.num_regimes)]
            prior_keys = list(prior_params_per_k[0].keys())
            prior_params_per_regime = [
                [{key: prior_params_per_k[k][key][:, t, :] for key in prior_keys} for t in range(T)]
                for k in range(self.num_regimes)
            ]
            prior_params_list = prior_params_per_regime[0]  # for a reader of prior_params only
            regime_weights = pi

        # K > 1: the AR(p) coefficients of a window are the π-mix at its origin, frozen over the window.
        if self.num_regimes > 1 and self.A_diag is not None:
            pi_w = regime_weights[:, 0, :]  # (batch, K)
            A_eff, B_eff = self._effective_regime_coeffs(pi_w)  # (batch, r, p), (batch, r)
            group_id = pi_w.argmax(dim=-1)  # dominant regime per window, for the grouped Krylov path
            factors_1T = self.ssm.forward_with_coeffs(eta_1T, None, A_eff, B_eff, group_id=group_id)
        else:
            factors_1T = self.ssm.forward(eta_1T)  # (batch, T, r)

        y_pred = self._decode(factors_1T, u_1T)

        out = {
            'y_pred': y_pred,
            'eta': eta_1T,
            'factors': factors_1T,
            'encoder_params': encoder_params_list,
            'prior_params': prior_params_list,
        }
        if regime_weights is not None:
            out['regime_weights'] = regime_weights
            out['prior_params_per_regime'] = prior_params_per_regime
        return out

    def elbo(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor,
        N: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """The negative ELBO (a loss) of a batch and its components ``reconstruction`` and ``kl``.

        ELBO = E[log p(y_t | f_t)] - Σ_t KL(q(η_t | y_t, u_t) || p(η_t | u_t)); with
        K > 1 the KL is Σ_t Σ_k π_{t,k} KL(q(η_t) || p_k(η_t | u_t)). ``N`` is unused.
        """
        outputs = self.forward(y_1T, u_1T)
        elbo, loss_dict = compute_elbo_loss(
            y_true=y_1T,
            y_pred=outputs['y_pred'],
            encoder_params=outputs['encoder_params'],
            prior_params=outputs['prior_params'],
            innovation_distribution=self.innovation_distribution,
            decoder_variance=self.decoder_var,
            beta_kl=self.beta_kl,
            reduction='mean',
            regime_weights=outputs.get('regime_weights'),
            prior_params_per_regime=outputs.get('prior_params_per_regime'),
        )
        return elbo, loss_dict

    def fit(self, control: Optional[RunControl] = None) -> 'iVDFMEngine':
        """Train on the dataset passed to ``__init__`` as ``dataset=``; then extract the factors of every window.

        Training stops on early stopping (``patience``, restoring the best
        epoch), on ``tolerance``, after ``max_epochs`` epochs, or when
        ``control`` (optional; called after every epoch with the epoch's mean
        ELBO loss) asks for a stop before those; ``_stop_reason`` records which.
        """
        dataset = self._dataset
        if dataset is None or not isinstance(dataset, iVDFMDataset):
            raise ModelNotTrainedError(
                "fit() requires a dataset. Pass an iVDFMDataset to the engine as dataset=."
            )

        actual_data_dim = dataset.target_length
        actual_context_dim = dataset.context_length
        self.time_context = actual_context_dim

        need_rebuild = False
        if self.data_dim is None:
            self.data_dim = actual_data_dim
            need_rebuild = True
        elif self.data_dim != actual_data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, dataset has {actual_data_dim}",
                details="If data_dim was set in __init__, it must match the observation dimension after extracting context columns"
            )

        if self.context_dim is None:
            self.context_dim = actual_context_dim
            need_rebuild = True
        elif self.context_dim != actual_context_dim:
            _logger.warning(
                f"context_dim mismatch: model expects {self.context_dim}, dataset has {actual_context_dim}. "
                f"Using dataset's inferred dimension {actual_context_dim}."
            )
            self.context_dim = actual_context_dim
            need_rebuild = True

        if need_rebuild:
            self._build_components()

        self._initialize_f0_from_data(dataset)
        self._initialize_ar_coeffs_from_data(dataset)
        self._build_optimizer()

        # A dataset pre-loaded on the GPU that fits in one batch trains on that one tensor, without a DataLoader.
        all_tensors = dataset.get_all_tensors_on_device()
        use_single_batch = all_tensors is not None and len(dataset) <= self.batch_size
        if use_single_batch:
            y_all, u_all = all_tensors
            dataloader = None
        else:
            dataloader = dataset.get_dataloader(
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
            )

        train_logger = iVDFMTrainLogger(verbose=True)
        train_logger.start(
            config={
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_type,
            },
            data_info={
                'num_sequences': len(dataset),
                'data_dim': self.data_dim,
                'context_dim': self.context_dim,
                'latent_dim': self.latent_dim,
            }
        )

        self.train()
        N_total = len(dataset)

        self._num_iter = 0
        self.loss_now = None
        self._elbo = None
        self._converged = False

        # Early stopping on the epoch ELBO (a loss: lower is better).
        best_elbo = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        batch_elbos: list[float] = []   # every batch of every epoch, for the tolerance check

        early_stop_enabled = self.patience is not None and self.patience > 0
        self._stop_reason = "max_iter"
        patience_str = f"early stopping patience={self.patience}" if early_stop_enabled else "early stopping disabled"
        batches_per_epoch = 1 if use_single_batch else len(dataloader)
        _logger.info(f"Starting training: {self.max_epochs} epochs, {batches_per_epoch} batches/epoch, {patience_str}")

        start_time = time.time()
        t_fwd, t_bwd, t_clip, t_step, t_data = 0.0, 0.0, 0.0, 0.0, 0.0
        _device = next(self.parameters()).device
        _sync = lambda: torch.cuda.synchronize() if _device.type == "cuda" else None
        t_data_start = time.perf_counter()

        for epoch in range(self.max_epochs):
            epoch_recon_losses = []
            epoch_kl_losses = []
            epoch_elbos = []

            batch_iter = [(y_all, u_all)] if use_single_batch else dataloader
            for batch_idx, (y_batch, u_batch) in enumerate(batch_iter):
                _sync()
                t_data += time.perf_counter() - t_data_start
                self.optimizer.zero_grad()

                t0 = time.perf_counter()
                elbo, loss_dict = self.elbo(y_batch, u_batch, N_total)
                _sync()
                t_fwd += time.perf_counter() - t0

                if torch.isnan(elbo) or torch.isinf(elbo):
                    _logger.error(
                        f"NaN/Inf detected in ELBO at epoch {epoch}, batch {batch_idx}. "
                        f"ELBO={elbo.item()}, Recon={loss_dict['reconstruction'].item()}, "
                        f"KL={loss_dict['kl'].item()}"
                    )
                    raise ValueError("Training failed: NaN/Inf in loss")

                t0 = time.perf_counter()
                elbo.backward()
                _sync()
                t_bwd += time.perf_counter() - t0

                t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                _sync()
                t_clip += time.perf_counter() - t0

                t0 = time.perf_counter()
                self.optimizer.step()
                _sync()
                t_step += time.perf_counter() - t0

                recon_val = loss_dict['reconstruction'].item()
                kl_val = loss_dict['kl'].item()
                elbo_val = elbo.item()
                # A KL below zero means a broken KL formula; fail fast.
                if kl_val < 0:
                    raise ValueError(
                        f"Training failed: negative KL at epoch {epoch}, batch {batch_idx}: "
                        f"KL={kl_val:.6f} (Recon={recon_val:.6f}, ELBO={elbo_val:.6f})"
                    )

                epoch_recon_losses.append(recon_val)
                epoch_kl_losses.append(kl_val)
                epoch_elbos.append(elbo_val)
                batch_elbos.append(elbo_val)
                t_data_start = time.perf_counter()

            self._num_iter = epoch + 1
            self.loss_now = np.mean(epoch_recon_losses)
            kl_loss_mean = np.mean(epoch_kl_losses)
            self._elbo = np.mean(epoch_elbos)

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self._elbo)
                else:
                    self.scheduler.step()

            elapsed_time = time.time() - start_time
            train_logger.log_epoch(
                epoch=epoch + 1,
                elbo=self._elbo,
                recon_loss=self.loss_now,
                kl_loss=kl_loss_mean,
                learning_rate=current_lr,
                time_elapsed=f"{elapsed_time:.2f}s",
            )
            if (epoch + 1) % 10 == 0 or epoch == 0:
                progress = 100 * (epoch + 1) / self.max_epochs
                _logger.info(
                    f"Progress: {progress:.1f}% | "
                    f"ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"Recon: {self.loss_now:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"KL: {kl_loss_mean:.{DEFAULT_LOSS_LOG_PRECISION}f} | "
                    f"LR: {current_lr:.2e}"
                )

            halt = None if control is None else control.step(epoch + 1, self._elbo)
            if early_stop_enabled:
                if self._elbo < best_elbo:
                    best_elbo = self._elbo
                    epochs_without_improvement = 0
                    best_model_state = {
                        'model_state_dict': copy.deepcopy(self.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                        'epoch': epoch + 1,
                        'elbo': self._elbo,
                    }
                    _logger.debug(f"New best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f} at epoch {epoch + 1}")
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        _logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}: "
                            f"no improvement for {epochs_without_improvement} epochs "
                            f"(best ELBO: {best_elbo:.{DEFAULT_LOSS_LOG_PRECISION}f}, "
                            f"current ELBO: {self._elbo:.{DEFAULT_LOSS_LOG_PRECISION}f})"
                        )
                        # best_model_state was set on the first epoch (its ELBO is below inf).
                        self.load_state_dict(best_model_state['model_state_dict'])
                        self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                        self._elbo = best_model_state['elbo']
                        self._num_iter = best_model_state['epoch']
                        _logger.info(f"Restored best model from epoch {best_model_state['epoch']}")

                        self._converged = True
                        self._stop_reason = "early_stopping"
                        train_logger.log_convergence(
                            converged=True,
                            num_epochs=epoch + 1,
                            final_loss=best_elbo,
                            reason="early_stopping"
                        )
                        break

            if self._check_convergence(batch_elbos):
                self._converged = True
                self._stop_reason = "converged"
                train_logger.log_convergence(
                    converged=True,
                    num_epochs=epoch + 1,
                    final_loss=self._elbo,
                    reason="converged"
                )
                _logger.info(f"Converged at epoch {epoch + 1}")
                break

            if halt is not None and epoch + 1 < self.max_epochs:
                self._stop_reason = halt
                break

        total_step = t_fwd + t_bwd + t_clip + t_step + t_data
        if total_step > 0:
            _logger.info(
                "Training time breakdown: data_load=%.2fs (%.0f%%), forward=%.2fs (%.0f%%), backward=%.2fs (%.0f%%), "
                "clip_grad=%.2fs (%.0f%%), optimizer_step=%.2fs (%.0f%%)",
                t_data, 100 * t_data / total_step,
                t_fwd, 100 * t_fwd / total_step,
                t_bwd, 100 * t_bwd / total_step,
                t_clip, 100 * t_clip / total_step,
                t_step, 100 * t_step / total_step,
            )

        if not self._converged:
            train_logger.log_convergence(
                converged=False,
                num_epochs=self._num_iter,
                final_loss=self._elbo,
                reason=self._stop_reason
            )

        # Final extraction over every window (in the loader's order when shuffled).
        self.eval()
        with torch.no_grad():
            all_factors = []
            all_innovations = []
            all_regime_weights = []
            iter_extract = [(y_all, u_all)] if use_single_batch else dataloader
            for y_batch, u_batch in iter_extract:
                outputs = self.forward(y_batch, u_batch)
                all_factors.append(to_numpy(outputs['factors']))
                all_innovations.append(to_numpy(outputs['eta']))
                if outputs.get('regime_weights') is not None:
                    all_regime_weights.append(to_numpy(outputs['regime_weights']))

            self.factors = np.concatenate(all_factors, axis=0)  # (windows, T, r)
            self.innovations = np.concatenate(all_innovations, axis=0)
            self.factor_path = self._encode_sequence(dataset)
            if all_regime_weights:
                rw = np.concatenate(all_regime_weights, axis=0)  # (windows, T, K)
                self.regime_weights = rw[:, -1, :]  # (windows, K): π at each window end
            else:
                self.regime_weights = None

        self.training_state = iVDFMModelState.from_model(self)
        return self

    def _check_convergence(self, elbo_history: list, window: int = 5) -> bool:
        """True when the mean of the last ``window`` batch ELBOs (across epochs) moved by less than ``tolerance``."""
        if len(elbo_history) < window + 1:
            return False
        recent = elbo_history[-window:]
        previous = elbo_history[-window-1:-1]
        improvement = np.mean(recent) - np.mean(previous)
        return abs(improvement) < self.tolerance

    def predict(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]] = None,
        context_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Union[List[str], List[int]]] = None,
        horizon: int = 1,
        deterministic: bool = True,
        num_samples: int = 1,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Forecast ``horizon`` steps ahead in model space, shape (horizon, data_dim).

        ``context_data`` (horizon, context_dim) is the future u_t; None
        generates time features. ``deterministic`` uses zero innovations,
        else ``num_samples`` prior samples are averaged. ``data``, ``context``
        and the extra arguments are ignored.
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before prediction")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        factors_future, y_pred = self._forecast(horizon, context_data, deterministic, num_samples)
        return to_numpy(y_pred.squeeze(0))  # (horizon, data_dim)

    def _factor_path(self) -> np.ndarray:
        """The factor state (T, r) the forecast starts from.

        ``factor_path`` is the conditioning series (the training data, or the
        data of the last ``update``) encoded as one sequence, so its last row
        is the factor state at the last observed time step. A model saved
        before 0.8.3 has no path; it falls back to the window average of
        ``factors``, which mixes the window-end states of different dates.
        """
        if self.factor_path is not None:
            return self.factor_path
        if self.factors.ndim == 3:
            return np.mean(self.factors, axis=0)
        return self.factors

    def _encode_sequence(self, dataset: iVDFMDataset) -> np.ndarray:
        """Factors (T, r) of the whole series of ``dataset``, encoded as one sequence with the fixed networks."""
        y = torch.as_tensor(np.asarray(dataset.target), dtype=DEFAULT_TORCH_DTYPE, device=self.device).unsqueeze(0)
        u = torch.as_tensor(np.asarray(dataset.context), dtype=DEFAULT_TORCH_DTYPE, device=self.device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            return to_numpy(self.forward(y, u)['factors'])[0]

    def _future_time_features(self, horizon: int, time_context: int) -> np.ndarray:
        """Time features (horizon, time_context), float32, for the ``horizon`` steps after the factor path.

        The normalized time step continues from ``T_train`` (the length of
        the factor path); ``time_context > 1`` adds the sine features.
        """
        T_train = self._factor_path().shape[0]
        time_indices = np.arange(T_train, T_train + horizon, dtype=np.float32)
        if T_train > 1:
            time_indices = time_indices / (T_train - 1)
        else:
            time_indices = time_indices / T_train

        if time_context == 1:
            return time_indices.reshape(-1, 1)
        features = [time_indices.reshape(-1, 1)]
        for i in range(1, time_context):
            freq = 2 * np.pi * (i + 1) / T_train if T_train > 1 else 2 * np.pi * (i + 1)
            periodic = np.sin(freq * np.arange(T_train, T_train + horizon, dtype=np.float32))
            features.append(periodic.reshape(-1, 1))
        return np.hstack(features)

    def _forecast(
        self,
        horizon: int,
        context_data: Optional[Union[np.ndarray, torch.Tensor]],
        deterministic: bool,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-loop forecast from the last factor state: factors (1, horizon, r) and model-space y (1, horizon, data_dim)."""
        self.eval()
        with torch.no_grad():
            f_last = self._factor_path()[-1, :]  # (r,)
            f_last_tensor = torch.from_numpy(f_last).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, r)

            if context_data is None:
                u_future = self._future_time_features(horizon, self.time_context)
                if self.context_dim > self.time_context:
                    _logger.warning(
                        f"Model has custom context columns (context_dim={self.context_dim} > time_context={self.time_context}). "
                        f"Only time features generated. Provide context_data parameter for full context."
                    )
            else:
                if isinstance(context_data, torch.Tensor):
                    u_future = context_data.cpu().numpy()
                else:
                    u_future = context_data
                if u_future.shape[0] != horizon:
                    raise ValueError(
                        f"context_data shape[0] ({u_future.shape[0]}) must match horizon ({horizon})"
                    )
                if u_future.shape[1] != self.context_dim:
                    raise ValueError(
                        f"context_data shape[1] ({u_future.shape[1]}) must match context_dim ({self.context_dim})"
                    )

            u_future_tensor = torch.from_numpy(u_future).to(
                dtype=DEFAULT_TORCH_DTYPE,
                device=self.device
            ).unsqueeze(0)  # (1, horizon, context_dim)

            if deterministic:
                eta_future = torch.zeros(1, horizon, self.latent_dim, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            elif num_samples == 1:
                eta_future = self._sample_prior(u_future_tensor)
            else:
                samples = [self._sample_prior(u_future_tensor) for _ in range(num_samples)]  # (1, horizon, r) each
                eta_future = torch.stack(samples, dim=0).mean(dim=0)

            # K > 1: the AR(p) coefficients are the π-mix at the first future step.
            if self.num_regimes > 1 and self.A_diag is not None:
                pi_w = self.regime_net(u_future_tensor[:, 0:1, :], temperature=self.regime_temperature).squeeze(1)  # (1, K)
                A_eff, B_eff = self._effective_regime_coeffs(pi_w)  # (1, r, p), (1, r)
                group_id = pi_w.argmax(dim=-1)  # (1,) for the grouped path
                factors_future = self.ssm.forward_closed_loop_with_coeffs(
                    f_current=f_last_tensor,
                    eta_future=eta_future,
                    ar_coeffs=A_eff,
                    B_diag=B_eff,
                    group_id=group_id,
                )
            else:
                factors_future = self.ssm.forward_closed_loop(
                    f_current=f_last_tensor,
                    eta_future=eta_future,
                    horizon=horizon,
                )

            y_pred = self._decode(factors_future, u_future_tensor)
            return factors_future, y_pred

    def update(
        self,
        dataset: iVDFMDataset,
        *,
        append: bool = True,
        verbose: bool = False,
    ) -> None:
        """Encode every window of ``dataset`` with the fixed networks and append (or replace) the factor state.

        One factor, innovation and regime-weight row per window end is kept;
        with ``append`` they follow the existing state, whose window
        dimension is averaged away first.
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before update")
        if not isinstance(dataset, iVDFMDataset):
            raise TypeError(f"update() takes an iVDFMDataset, got {type(dataset).__name__}")

        if dataset.target_length != self.data_dim:
            raise DataValidationError(
                f"data_dim mismatch: model expects {self.data_dim}, new data has {dataset.target_length}"
            )
        if dataset.context_length != self.context_dim:
            raise DataValidationError(
                f"context_dim mismatch: model expects {self.context_dim}, new data has {dataset.context_length}"
            )

        dataloader = dataset.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.eval()
        with torch.no_grad():
            all_factors: list[np.ndarray] = []
            all_innovations: list[np.ndarray] = []
            all_regime_weights: list[np.ndarray] = []
            for y_batch, u_batch in dataloader:
                outputs = self.forward(y_batch, u_batch)
                all_factors.append(to_numpy(outputs['factors']))
                all_innovations.append(to_numpy(outputs['eta']))
                if outputs.get('regime_weights') is not None:
                    all_regime_weights.append(to_numpy(outputs['regime_weights']))

            # The state at the end of each window, (num_windows, r), aligns with full-series models (e.g. DDFM).
            new_factors = np.concatenate(all_factors, axis=0)[:, -1, :]
            new_innovations = np.concatenate(all_innovations, axis=0)[:, -1, :]
            new_regime_weights = np.concatenate(all_regime_weights, axis=0)[:, -1, :] if all_regime_weights else None

            new_path = self._encode_sequence(dataset)
            if append and self.factor_path is not None:
                self.factor_path = np.concatenate([self.factor_path, new_path], axis=0)
            else:
                self.factor_path = new_path

            if append and self.factors is not None and self.innovations is not None:
                factors_existing = self._normalize_factors_shape(self.factors)
                innovations_existing = self._normalize_factors_shape(self.innovations)
                self.factors = np.concatenate([factors_existing, new_factors], axis=0)
                self.innovations = np.concatenate([innovations_existing, new_innovations], axis=0)
                if new_regime_weights is not None and self.regime_weights is not None:
                    self.regime_weights = np.concatenate([self.regime_weights, new_regime_weights], axis=0)
                elif new_regime_weights is not None:
                    self.regime_weights = new_regime_weights
            else:
                self.factors = new_factors
                self.innovations = new_innovations
                if new_regime_weights is not None:
                    self.regime_weights = new_regime_weights

            self.training_state = iVDFMModelState.from_model(self)

            if verbose:
                _logger.info(
                    f"Model state updated with {len(dataset)} new sequences. "
                    f"Factors shape: {self.factors.shape}, Innovations shape: {self.innovations.shape}"
                )

    def get_result(self) -> iVDFMResult:
        """The result of the trained model: factors, innovations, reconstructions, diagnostics and weights."""
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        if self.factors is None or self.innovations is None:
            raise ModelNotTrainedError("Factors and innovations not available")

        # No u at result time: K > 1 decodes with uniform π.
        # Z and x_sm are the (T, r) factor path (the series encoded as one
        # sequence; the window average without a path) and its (T, N) decoding;
        # ``reconstructions`` decodes every window.
        z_np = np.asarray(self._factor_path())
        with torch.no_grad():
            z_tensor = torch.from_numpy(self.factors).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device)
            recon = to_numpy(self._decode(z_tensor, u_1T=None))
            path_tensor = torch.from_numpy(z_np).to(dtype=DEFAULT_TORCH_DTYPE, device=self.device).unsqueeze(0)
            x_sm = to_numpy(self._decode(path_tensor, u_1T=None))[0]

        weights = {k: v.detach().cpu() for k, v in self.state_dict().items()}

        mixing_matrix = None
        if self.factor_mixing is not None:
            mixing_matrix = to_numpy(self.factor_mixing.M.weight.detach())

        return iVDFMResult(
            innovations=self.innovations,
            reconstructions=recon,
            full_state=self._compute_full_state(z_np),
            x_sm=x_sm,
            Z=z_np,
            r=np.array([self.latent_dim], dtype=int),
            p=int(self.factor_order),
            target_scaler=self._config.target_scaler,
            num_iter=int(self._num_iter),
            objective=self._elbo,
            training_elbo=self._elbo,
            training_loss=self.loss_now,
            num_epochs=int(self._num_iter),
            converged=bool(self._converged),
            config=dataclasses.asdict(self._config),
            model_state_dict=weights,
            regime_weights=self.regime_weights,
            num_regimes=self.num_regimes,
            regime_temperature=self.regime_temperature,
            mixing=self.mixing,
            mixing_matrix=mixing_matrix,
        )

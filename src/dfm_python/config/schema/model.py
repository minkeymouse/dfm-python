"""Configuration dataclasses.

- ``BaseModelConfig``: shared engine structure (frequency dict, clock, scaler).
- ``_DFMEngineConfig(BaseModelConfig)``: the private DFM engine's blocks and EM
  parameters.
- ``_iVDFMEngineConfig(BaseModelConfig)``: the private iVDFM engine's parameters.
- ``AFMConfig``, ``iVDFMConfig``, ``DDFMConfig``, ``DFMConfig``: the public
  mirrors of the estimators' constructors (:class:`EstimatorConfig`), read from
  YAML with ``<Model>Config.from_yaml`` or :func:`dfm_python.config.load_yaml`.

Series are given by a frequency dict mapping column names to frequency codes.
Blocks (DFM only) are ``{"block_name": {"num_factors": int, "series": [str]}}``.
The engine configs are private; their parsing helpers are in ``config.adapter``.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field, fields

from ..estimator import EstimatorConfig

if TYPE_CHECKING:
    from ...functional.em import EMConfig

from ..constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY,
    DEFAULT_IVDFM_OPTIMIZER_MOMENTUM,
    DEFAULT_IVDFM_SCHEDULER_TYPE,
    DEFAULT_IVDFM_SCHEDULER_STEP_SIZE,
    DEFAULT_IVDFM_SCHEDULER_GAMMA,
    DEFAULT_IVDFM_SCHEDULER_PATIENCE,
    DEFAULT_IVDFM_SCHEDULER_FACTOR,
    DEFAULT_IVDFM_SCHEDULER_MIN_LR,
    DEFAULT_IVDFM_EARLY_STOP_PATIENCE,
    DEFAULT_REGULARIZATION_SCALE,
    DEFAULT_REGULARIZATION,
    DEFAULT_EM_THRESHOLD,
    DEFAULT_EM_MAX_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_IDIO_RHO0,
    MIN_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    VAR_STABILITY_THRESHOLD,
    VALID_FREQUENCIES,
    VALID_OPTIMIZERS,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_AFM_FACTOR_MODEL,
    DEFAULT_AFM_TRADING,
    DEFAULT_AFM_EMBED_DIM,
    DEFAULT_AFM_HIST_LEN,
    DEFAULT_AFM_N_KERNELS,
    DEFAULT_AFM_RIDGE,
    DEFAULT_AFM_SQUASH_LAMBDA,
    DEFAULT_AFM_PCA_WINDOW,
    DEFAULT_AFM_REESTIM,
    DEFAULT_AFM_LAMBDA_VAR,
    DEFAULT_AFM_TURNOVER_COST,
    DEFAULT_AFM_SHORT_COST,
    DEFAULT_AFM_MAX_EPOCHS,
    DEFAULT_IVDFM_ENCODER_HIDDEN_DIM,
    DEFAULT_IVDFM_ENCODER_N_HIDDEN_LAYERS,
    DEFAULT_IVDFM_NUM_REGIMES,
    DEFAULT_IVDFM_REGIME_TEMPERATURE,
    DEFAULT_IVDFM_MIXING,
    DEFAULT_IVDFM_DECODER_TYPE,
    DEFAULT_IVDFM_DECODER_HIDDEN_DIM,
    DEFAULT_IVDFM_DECODER_N_HIDDEN_LAYERS,
    DEFAULT_IVDFM_PRIOR_HIDDEN_DIM,
    DEFAULT_IVDFM_PRIOR_N_HIDDEN_LAYERS,
    DEFAULT_IVDFM_FACTOR_ORDER,
    DEFAULT_IVDFM_INNOVATION_DIST,
    DEFAULT_IVDFM_DECODER_VAR,
    DEFAULT_IVDFM_ACTIVATION,
    DEFAULT_IVDFM_SLOPE,
    DEFAULT_IVDFM_BATCH_SIZE,
    DEFAULT_IVDFM_MAX_EPOCHS,
)


@dataclass
class BaseModelConfig:
    """Model structure shared by the engine configs.

    ``frequency`` maps column names to frequency codes; it may be given in the
    grouped form ``{'w': [series...], 'm': [...]}`` to ``from_dict``. When it
    is None, or a column is missing from it, the column uses ``clock``.
    """
    frequency: Optional[Dict[str, str]] = None
    clock: str = 'm'  # base frequency of the latent factors: 'd', 'w', 'm', 'q', 'sa', 'a'
    target_scaler: Optional[Any] = None  # fitted sklearn scaler for the target series
    scaler: Optional[str] = None  # 'standard', 'robust', 'minmax', 'maxabs', 'quantile' or None; targets only

    def __post_init__(self):
        """Validate ``clock`` and the frequency dict; raises ``ConfigurationError``."""
        from ...config.adapter import _raise_config_error, _is_dict_like

        self.clock = validate_frequency(self.clock)
        if self.frequency is not None:
            if not _is_dict_like(self.frequency):
                _raise_config_error(
                    f"frequency must be a dict mapping column names to frequencies, got {type(self.frequency)}"
                )
            for col_name, freq in self.frequency.items():
                if not isinstance(col_name, str):
                    _raise_config_error(f"frequency dict keys must be strings (column names), got {type(col_name)}")
                validate_frequency(freq)

    def get_frequencies(self, columns: Optional[List[str]] = None) -> List[str]:
        """Frequencies of ``columns`` (``clock`` for missing ones), or of the frequency dict.

        With ``columns`` and no frequency dict, the dict is created from them.
        """
        if columns is not None:
            if self.frequency is None:
                self.frequency = {col: self.clock for col in columns}
            return [self.frequency.get(col, self.clock) for col in columns]
        if self.frequency is None:
            return []
        return list(self.frequency.values())

    @classmethod
    def _extract_base(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """The shared fields of a config dict (frequency from any accepted form)."""
        from ...config.adapter import _extract_frequency_dict

        base_params = {
            'clock': data.get('clock', DEFAULT_CLOCK_FREQUENCY),
            'target_scaler': data.get('target_scaler', None),
        }
        frequency_dict = _extract_frequency_dict(data, base_params['clock'])
        if frequency_dict is not None:
            base_params['frequency'] = frequency_dict
        return base_params


@dataclass
class _DFMEngineConfig(BaseModelConfig):
    """Configuration of the private DFM engine (the pre-redesign ``DFMConfig``).

    Used only by :class:`~dfm_python.models.dfm._engine.DFMEngine` and the
    dataset it reads; the DFM estimator builds it from its constructor
    parameters, and the public :class:`DFMConfig` mirrors that constructor.
    Factors are organized in blocks (global + sector-specific) and always
    follow AR(1) dynamics. ``ar_clip``, ``data_clip``, ``regularization`` and
    ``damping_factor`` are off when None. ``em_acceleration`` is ``"squarem"``
    or None (plain EM).
    """
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"name": {"num_factors": int, "series": [str]}}
    block_names: List[str] = field(init=False)  # derived from blocks
    factors_per_block: List[int] = field(init=False)  # derived from blocks
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cached_blocks_key: Optional[tuple] = field(default=None, init=False, repr=False)

    threshold: float = DEFAULT_EM_THRESHOLD  # EM convergence threshold
    max_iter: int = DEFAULT_EM_MAX_ITER

    ar_clip: Optional[Dict[str, float]] = None  # {"min": float, "max": float}
    data_clip: Optional[float] = None  # clip values beyond this many standard deviations
    regularization: Optional[Dict[str, float]] = None  # {"scale", "min_eigenvalue", "max_eigenvalue"}
    damping_factor: Optional[float] = None  # 0.8 = 80% new, 20% old
    em_acceleration: Optional[str] = "squarem"  # "squarem" or None (plain EM)

    idio_rho0: float = DEFAULT_IDIO_RHO0  # initial AR coefficient of the idiosyncratic components
    idio_min_var: float = MIN_DIAGONAL_VARIANCE  # floor of the idiosyncratic innovation variance

    # Required for mixed-frequency data: {'m:w': [1, 2, 1]} or {'m': [1, 2, 1]} per slower frequency.
    tent_weights: Optional[Union[Dict[str, List[float]], Dict[str, np.ndarray]]] = None

    def __post_init__(self):
        """Validate the blocks and derive ``block_names`` and ``factors_per_block``.

        Series named in a block but missing from the frequency dict are added
        with the clock frequency.
        """
        super().__post_init__()
        from ...config.adapter import _raise_config_error

        if not self.blocks:
            _raise_config_error("DFM configuration must contain at least one block.")

        object.__setattr__(self, 'block_names', list(self.blocks.keys()))
        object.__setattr__(self, 'factors_per_block',
                           [self.blocks[name].get('num_factors', 1) for name in self.block_names])

        for block_name, block_cfg in self.blocks.items():
            num_factors = block_cfg.get('num_factors', 1)
            series_list = block_cfg.get('series', [])
            if num_factors < 1:
                _raise_config_error(f"Block '{block_name}' must have num_factors >= 1, got {num_factors}")
            if not isinstance(series_list, list):
                _raise_config_error(f"Block '{block_name}' must have 'series' as a list, got {type(series_list)}")
            if self.frequency is not None:
                for series_name in series_list:
                    if series_name not in self.frequency:
                        self.frequency[series_name] = self.clock

    def to_em_config(self) -> 'EMConfig':
        """The ``EMConfig`` of this configuration (regularization, clipping, damping, acceleration; the rest defaults)."""
        from ...functional.em import EMConfig

        if isinstance(self.regularization, dict):
            reg_scale = self.regularization.get('scale', DEFAULT_REGULARIZATION_SCALE)
            min_eigenval = self.regularization.get('min_eigenvalue', MIN_EIGENVALUE)
            max_eigenval = self.regularization.get('max_eigenvalue', VAR_STABILITY_THRESHOLD)
        else:
            reg_scale = DEFAULT_REGULARIZATION
            min_eigenval = MIN_EIGENVALUE
            max_eigenval = VAR_STABILITY_THRESHOLD

        return EMConfig(
            regularization=reg_scale,
            min_norm=min_eigenval,
            max_eigenval=max_eigenval,
            ar_clip=self.ar_clip,
            damping_factor=self.damping_factor,
            data_clip=self.data_clip,
            em_acceleration=self.em_acceleration,
        )

    def get_blocks_array(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """Block membership (N x B): 1 if the series loads on the block, cached per ``columns``.

        With ``columns``, a single ``Block_Global`` block with an empty series
        list means "all series".
        """
        key = None if columns is None else tuple(columns)
        if self._cached_blocks is None or self._cached_blocks_key != key:
            if self.frequency is None:
                if columns is None:
                    from ...config.adapter import _raise_config_error
                    _raise_config_error("frequency dict or columns required")
                self.frequency = {col: self.clock for col in columns}

            series_ids = list(self.frequency.keys()) if columns is None else columns

            block_series_sets: Dict[str, set] = {}
            for name in self.block_names:
                series_list = self.blocks[name].get('series', [])
                if (
                    columns is not None
                    and (series_list is None or len(series_list) == 0)
                    and len(self.block_names) == 1
                    and name == "Block_Global"
                ):
                    block_series_sets[name] = set(series_ids)
                else:
                    block_series_sets[name] = set(series_list or [])

            blocks_list = [
                [1 if series_id in block_series_sets[name] else 0 for name in self.block_names]
                for series_id in series_ids
            ]
            self._cached_blocks = np.array(blocks_list, dtype=int)
            self._cached_blocks_key = key
        return self._cached_blocks

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '_DFMEngineConfig':
        """Build the engine config from ``{'frequency': ..., 'blocks': ..., 'threshold': ..., ...}``.

        Without ``blocks``, a single ``Block_Global`` with ``num_factors``
        (default 1) factors and series bound from the data columns is used.
        A dictionary that looks like a DDFM config (``model_type: ddfm`` or
        ``deep``, or one of ``encoder_layers``, ``epochs``, ``learning_rate``,
        ``batch_size`` without ``blocks``) raises ``ConfigurationError`` naming
        ``DDFMConfig.from_dict`` and ``DDFM.from_config``.
        """
        from ...config.adapter import (
            detect_config_type, MODEL_TYPE_DDFM, _normalize_blocks_dict, _raise_config_error, _is_dict_like,
        )
        from ...utils.errors import ConfigurationError

        base_params = cls._extract_base(data)

        if detect_config_type(data) == MODEL_TYPE_DDFM:
            raise ConfigurationError(
                "_DFMEngineConfig.from_dict: this dictionary holds DDFM parameters (model_type, encoder_layers, epochs, "
                "learning_rate or batch_size). DDFM has its own config: use DDFMConfig.from_dict(params) or "
                "DDFM.from_config(params) with the DDFM constructor parameters (n_factors, encoder_layers, ...)."
            )

        blocks_dict = data.get('blocks', {})
        if not blocks_dict:
            default_num_factors = int(data.get('num_factors', 1) or 1)
            if default_num_factors < 1:
                _raise_config_error(f"num_factors must be >= 1, got {default_num_factors}")
            blocks_dict = {"Block_Global": {"num_factors": default_num_factors, "series": []}}
        if not _is_dict_like(blocks_dict):
            _raise_config_error(f"blocks must be a dict, got {type(blocks_dict)}")

        dfm_params = {
            'threshold': data.get('threshold', DEFAULT_EM_THRESHOLD),
            'max_iter': data.get('max_iter', DEFAULT_EM_MAX_ITER),
            'idio_rho0': data.get('idio_rho0', DEFAULT_IDIO_RHO0),
            'idio_min_var': data.get('idio_min_var', MIN_DIAGONAL_VARIANCE),
            'ar_clip': data.get('ar_clip', None),
            'data_clip': data.get('data_clip', None),
            'regularization': data.get('regularization', None),
            'damping_factor': data.get('damping_factor', None),
            'em_acceleration': data.get('em_acceleration', "squarem"),
            'tent_weights': data.get('tent_weights', None),
        }
        return cls(blocks=_normalize_blocks_dict(blocks_dict), **base_params, **dfm_params)


# Fields of BaseModelConfig that _extract_base reads.
_BASE_FIELDS = ('frequency', 'clock', 'target_scaler')


@dataclass
class _iVDFMEngineConfig(BaseModelConfig):
    """Configuration of the private iVDFM engine (the pre-redesign ``iVDFMConfig``).

    Used only by :class:`~dfm_python.models.ivdfm._engine.iVDFMEngine`; the
    public :class:`iVDFMConfig` mirrors the estimator's constructor. iVDFM has
    no blocks: ``num_factors`` is the number of factors. The ``scaler`` string
    applies to the targets only, not to the context variables.
    """
    num_factors: Optional[int] = None
    window: Optional[int] = None  # sliding window length; None = the full series as one sequence
    stride: int = 1  # window step; stride=window gives non-overlapping windows
    context: Optional[Union[List[str], List[int]]] = None  # exogenous context columns (names or indices)
    time_context: int = 1  # time feature dimension; 1 = time step only, >1 adds periodic sine features

    encoder_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_ENCODER_HIDDEN_DIM
    encoder_n_hidden_layers: int = DEFAULT_IVDFM_ENCODER_N_HIDDEN_LAYERS
    decoder_type: str = DEFAULT_IVDFM_DECODER_TYPE  # 'linear', 'residual' or 'mlp'
    decoder_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_DECODER_HIDDEN_DIM  # residual/mlp decoders
    decoder_n_hidden_layers: int = DEFAULT_IVDFM_DECODER_N_HIDDEN_LAYERS
    prior_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_PRIOR_HIDDEN_DIM
    prior_n_hidden_layers: int = DEFAULT_IVDFM_PRIOR_N_HIDDEN_LAYERS
    activation: str = DEFAULT_IVDFM_ACTIVATION
    slope: float = DEFAULT_IVDFM_SLOPE  # leaky ReLU

    num_regimes: int = DEFAULT_IVDFM_NUM_REGIMES  # K; 1 = one global dynamics and one decoder
    regime_temperature: float = DEFAULT_IVDFM_REGIME_TEMPERATURE  # softmax temperature of pi; below 1 sharpens
    factor_order: int = DEFAULT_IVDFM_FACTOR_ORDER  # AR order of the factors
    innovation_distribution: str = DEFAULT_IVDFM_INNOVATION_DIST
    decoder_var: float = DEFAULT_IVDFM_DECODER_VAR
    beta_kl: float = 1.0  # ELBO = recon_loss + beta_kl * kl_loss
    use_layer_norm: bool = False
    mixing: bool = DEFAULT_IVDFM_MIXING  # global post-factor mixing f -> M f (identity init) before decoding

    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = 'Adam'
    optimizer_weight_decay: float = DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY
    optimizer_momentum: float = DEFAULT_IVDFM_OPTIMIZER_MOMENTUM  # SGD
    batch_size: int = DEFAULT_IVDFM_BATCH_SIZE
    max_epochs: int = DEFAULT_IVDFM_MAX_EPOCHS
    tolerance: float = DEFAULT_TOLERANCE
    patience: Optional[int] = DEFAULT_IVDFM_EARLY_STOP_PATIENCE  # early stopping; None disables
    seed: Optional[int] = None

    scheduler_type: Optional[str] = DEFAULT_IVDFM_SCHEDULER_TYPE  # 'step', 'plateau', 'cosine', 'exponential', None
    scheduler_step_size: Optional[int] = DEFAULT_IVDFM_SCHEDULER_STEP_SIZE  # StepLR; None = max_epochs // 3
    scheduler_gamma: float = DEFAULT_IVDFM_SCHEDULER_GAMMA  # StepLR/ExponentialLR
    scheduler_patience: int = DEFAULT_IVDFM_SCHEDULER_PATIENCE  # ReduceLROnPlateau
    scheduler_factor: float = DEFAULT_IVDFM_SCHEDULER_FACTOR  # ReduceLROnPlateau
    scheduler_min_lr: float = DEFAULT_IVDFM_SCHEDULER_MIN_LR  # ReduceLROnPlateau

    def __post_init__(self):
        """Validate the ranges and choices; raises ``ConfigurationError``."""
        super().__post_init__()
        from ...utils.errors import ConfigurationError

        if self.window is not None and self.window < 1:
            raise ConfigurationError(f"window must be >= 1, got {self.window}")
        if self.time_context < 1:
            raise ConfigurationError(f"time_context must be >= 1, got {self.time_context}")
        if self.patience is not None and self.patience < 1:
            raise ConfigurationError(f"patience must be >= 1 or None (to disable), got {self.patience}")
        if self.num_factors is not None and self.num_factors < 1:
            raise ConfigurationError(f"num_factors must be >= 1, got {self.num_factors}")
        if self.factor_order < 1:
            raise ConfigurationError(f"factor_order must be >= 1, got {self.factor_order}")
        valid_dists = {'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'}
        if self.innovation_distribution not in valid_dists:
            raise ConfigurationError(
                f"innovation_distribution must be one of {valid_dists}, got '{self.innovation_distribution}'"
            )
        if self.num_regimes < 1:
            raise ConfigurationError(f"num_regimes must be >= 1, got {self.num_regimes}")
        if self.regime_temperature <= 0.0:
            raise ConfigurationError(f"regime_temperature must be > 0, got {self.regime_temperature}")
        if self.decoder_type not in ("linear", "residual", "mlp"):
            raise ConfigurationError(
                f"decoder_type must be 'linear', 'residual', or 'mlp', got '{self.decoder_type}'"
            )
        valid_activations = {'relu', 'lrelu', 'tanh', 'sigmoid'}
        if self.activation not in valid_activations:
            raise ConfigurationError(f"activation must be one of {valid_activations}, got '{self.activation}'")
        if self.beta_kl < 0:
            raise ConfigurationError(f"beta_kl must be >= 0, got {self.beta_kl}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ConfigurationError(f"optimizer must be one of {VALID_OPTIMIZERS}, got '{self.optimizer}'")
        valid_schedulers = {'step', 'plateau', 'cosine', 'exponential', None}
        if self.scheduler_type not in valid_schedulers:
            raise ConfigurationError(
                f"scheduler_type must be one of {valid_schedulers}, got '{self.scheduler_type}'"
            )
        if self.scheduler_gamma <= 0 or self.scheduler_gamma > 1:
            raise ConfigurationError(f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}")
        if self.scheduler_patience < 0:
            raise ConfigurationError(f"scheduler_patience must be >= 0, got {self.scheduler_patience}")
        if self.scheduler_factor <= 0 or self.scheduler_factor > 1:
            raise ConfigurationError(f"scheduler_factor must be in (0, 1], got {self.scheduler_factor}")
        if self.scheduler_min_lr < 0:
            raise ConfigurationError(f"scheduler_min_lr must be >= 0, got {self.scheduler_min_lr}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '_iVDFMEngineConfig':
        """Build the engine config from a dict; missing keys take the field defaults, unknown keys are ignored."""
        base_params = cls._extract_base(data)
        params = {f.name: data.get(f.name, f.default) for f in fields(cls) if f.name not in _BASE_FIELDS}
        return cls(**base_params, **params)


def validate_frequency(frequency: str) -> str:
    """Return ``frequency`` if it is a valid code; raises ``ConfigurationError`` otherwise."""
    from ...utils.errors import ConfigurationError

    if not isinstance(frequency, str):
        raise ConfigurationError(
            f"Frequency must be a string, got {type(frequency).__name__}: {frequency}"
        )
    if frequency not in VALID_FREQUENCIES:
        raise ConfigurationError(
            f"Invalid frequency: '{frequency}'. Must be one of {VALID_FREQUENCIES}. "
            f"Common frequencies: 'd' (daily), 'w' (weekly), 'm' (monthly), "
            f"'q' (quarterly), 'sa' (semi-annual), 'a' (annual)."
        )
    return frequency


@dataclass
class AFMConfig(EstimatorConfig):
    """Config mirror of the :class:`~dfm_python.models.afm.estimator.AFM` estimator.

    The fields and defaults are exactly the estimator's constructor
    parameters, so ``AFM.from_config(cfg)`` is ``AFM(**cfg.to_dict())`` and
    ``AFM(...).get_config()`` returns one whose fields equal ``get_params()``.
    Values are validated when the estimator is fitted, not here. Unknown keys
    (including the old names ``num_factors`` and ``seed``) raise
    ``ConfigurationError`` in ``from_dict`` and ``from_yaml``.
    """

    n_factors: Optional[int] = None
    factor_model: str = DEFAULT_AFM_FACTOR_MODEL  # 'attention' (learned) | 'pca'
    trading: str = DEFAULT_AFM_TRADING     # 'longconv' (learned) | 'ou' (Avellaneda-Lee)
    embed_dim: int = DEFAULT_AFM_EMBED_DIM  # characteristic embedding dim d
    hist_len: int = DEFAULT_AFM_HIST_LEN   # residual-history length s for the filter
    n_kernels: int = DEFAULT_AFM_N_KERNELS  # number of long convolutions
    ridge: float = DEFAULT_AFM_RIDGE       # ridge penalty for closed-form loadings
    squash_lambda: float = DEFAULT_AFM_SQUASH_LAMBDA  # LongConv Squash regularizer
    pca_window: int = DEFAULT_AFM_PCA_WINDOW  # trailing window for the PCA benchmark
    reestim: int = DEFAULT_AFM_REESTIM     # PCA re-estimation stride
    lambda_var: float = DEFAULT_AFM_LAMBDA_VAR  # explained-variance regularizer weight
    turnover_cost: float = DEFAULT_AFM_TURNOVER_COST  # 5bps per unit turnover (paper)
    short_cost: float = DEFAULT_AFM_SHORT_COST  # 1bp shorting cost (paper)
    risk_free: float = 0.0
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_AFM_MAX_EPOCHS
    patience: Optional[int] = None
    max_time: Optional[float] = None  # training budget in seconds
    random_state: Optional[int] = None
    device: str = "auto"
    verbose: int = 0


@dataclass
class iVDFMConfig(EstimatorConfig):
    """Config mirror of the :class:`~dfm_python.models.ivdfm.estimator.iVDFM` estimator.

    The fields and defaults are exactly the estimator's constructor
    parameters, so ``iVDFM.from_config(cfg)`` is ``iVDFM(**cfg.to_dict())`` and
    ``iVDFM(...).get_config()`` returns one whose fields equal ``get_params()``.
    Values are validated when the estimator is fitted, not here. Unknown keys
    (including the old names ``num_factors``, ``num_regimes``, ``tolerance``
    and ``seed``) raise ``ConfigurationError`` in ``from_dict`` and
    ``from_yaml``. A list given for a hidden width in YAML or JSON becomes a
    tuple.
    """

    n_factors: int = 3
    window: Optional[int] = None
    stride: Optional[int] = None
    time_context: int = 1
    encoder_hidden_dim: Union[int, Tuple[int, ...]] = 200
    encoder_n_hidden_layers: int = 2
    decoder_type: str = "mlp"
    decoder_hidden_dim: Union[int, Tuple[int, ...]] = 200
    decoder_n_hidden_layers: int = 2
    prior_hidden_dim: Union[int, Tuple[int, ...]] = 100
    prior_n_hidden_layers: int = 1
    activation: str = "lrelu"
    slope: float = 0.1
    factor_order: int = 1
    innovation_distribution: str = "laplace"
    decoder_var: float = 0.01
    n_regimes: int = 1
    regime_temperature: float = 1.0
    beta_kl: float = 1.0
    use_layer_norm: bool = False
    mixing: bool = False
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    optimizer_weight_decay: float = 0.0
    optimizer_momentum: float = 0.9
    batch_size: int = 32
    max_epochs: int = 100
    tol: float = 5e-4
    patience: Optional[int] = 20
    max_time: Optional[float] = None
    scheduler_type: Optional[str] = "step"
    scheduler_step_size: Optional[int] = None
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 10
    scheduler_factor: float = 0.1
    scheduler_min_lr: float = 0.0
    scaler: Any = "standard"
    random_state: Optional[int] = 3
    device: str = "auto"
    verbose: int = 0


@dataclass
class DDFMConfig(EstimatorConfig):
    """Config mirror of the :class:`~dfm_python.models.ddfm.estimator.DDFM` estimator.

    The fields and defaults are exactly the estimator's constructor
    parameters, so ``DDFM.from_config(cfg)`` is ``DDFM(**cfg.to_dict())`` and
    ``DDFM(...).get_config()`` returns one whose fields equal ``get_params()``.
    Values are validated when the estimator is fitted, not here. Unknown keys
    (including the old names ``encoder_size``, ``num_factors``, ``tolerance``,
    ``seed``, ``disp``, ``max_epoch`` and the never-read ``use_batch_norm``,
    ``use_idiosyncratic``, ``min_obs_idio`` and ``lags_input``) raise
    ``ConfigurationError`` in ``from_dict`` and ``from_yaml``. A list given
    for ``encoder_layers`` in YAML or JSON becomes a tuple.
    """

    n_factors: int = 4
    encoder_layers: Tuple[int, ...] = (16,)
    decoder_type: str = "linear"
    activation: str = "relu"
    learning_rate: float = 0.005
    optimizer: str = "Adam"
    n_mc_samples: int = 10
    window_size: int = 100
    max_iter: int = 200
    tol: float = 5e-4
    max_time: Optional[float] = None
    interpolation_method: str = "linear"
    interpolation_limit: Optional[int] = 10
    interpolation_limit_direction: str = "both"
    scaler: Any = "standard"
    random_state: Optional[int] = 3
    device: str = "auto"
    verbose: int = 0


@dataclass
class DFMConfig(EstimatorConfig):
    """Config mirror of the :class:`~dfm_python.models.dfm.estimator.DFM` estimator.

    The fields and defaults are exactly the estimator's constructor
    parameters, so ``DFM.from_config(cfg)`` is ``DFM(**cfg.to_dict())`` and
    ``DFM(...).get_config()`` returns one whose fields equal ``get_params()``.
    Values are validated when the estimator is fitted, not here. Unknown keys
    (including the old names ``threshold``, ``num_factors``, ``frequency``,
    ``target_scaler``, ``idio_rho0`` and ``idio_min_var``) raise
    ``ConfigurationError`` in ``from_dict`` and ``from_yaml``. A block is
    ``{"n_factors": int, "series": [names]}``; the order of ``blocks`` is the
    order of the factors.
    """

    n_factors: Optional[int] = None
    blocks: Optional[Dict[str, Dict[str, Any]]] = None
    clock: str = "m"
    tent_weights: Optional[Dict[str, List[float]]] = None
    max_iter: int = 5000
    tol: float = 1e-6
    max_time: Optional[float] = None
    ar_clip: Optional[Dict[str, float]] = None
    data_clip: Optional[float] = None
    regularization: Optional[Dict[str, float]] = None
    damping_factor: Optional[float] = None
    em_acceleration: Optional[str] = "squarem"
    scaler: Any = "standard"
    verbose: int = 0

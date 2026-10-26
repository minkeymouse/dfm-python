"""Builders for model structure, observation matrices, optimizers and state spaces.

The optimizer and companion builders of the torch models import torch when
called; the DFM builders need numpy only.
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Any, List

from ..logger import get_logger
from ..config.constants import (
    DEFAULT_IDENTITY_SCALE,
    DEFAULT_DTYPE,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_ADAM_BETA1,
    DEFAULT_ADAM_BETA2,
    DEFAULT_ADAM_EPS,
    DEFAULT_LR_DECAY_RATE,
    DEFAULT_DDFM_OBSERVATION_NOISE,
)
from .stability import create_scaled_identity
from .estimator import get_transition_params
from ..utils.errors import ConfigurationError, ModelNotTrainedError, NumericalError

if TYPE_CHECKING:
    import torch

_logger = get_logger(__name__)


def build_dfm_structure(config: Any, *, columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build the DFM block structure from an engine config.

    Parameters
    ----------
    config : _DFMEngineConfig
        Provides ``get_blocks_array()`` and optionally ``factors_per_block``.
    columns : list[str], optional
        Series names; lets a config that omits ``frequency`` map every series
        to the clock frequency.

    Returns
    -------
    blocks : np.ndarray
        Block membership (N x n_blocks).
    r : np.ndarray
        Factors per block (n_blocks,); ones when the config does not set it.
    num_factors : int
        ``sum(r)``.
    p : int
        Factor VAR order, always ``DEFAULT_FACTOR_ORDER`` (1).
    """
    blocks_array = config.get_blocks_array(columns=columns)
    blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)

    factors_per_block = getattr(config, 'factors_per_block', None)
    if factors_per_block is not None:
        r = np.array(factors_per_block, dtype=DEFAULT_DTYPE)
    else:
        r = np.ones(blocks_array.shape[1], dtype=DEFAULT_DTYPE)

    return blocks, r, int(np.sum(r)), DEFAULT_FACTOR_ORDER


def log_blocks_diagnostics(
    blocks: np.ndarray,
    columns: Optional[List[str]],
    N_actual: int
) -> None:
    """Log how many of the ``N_actual`` series load on the global block."""
    n_in_block = np.sum(blocks[:, 0] > 0) if blocks.shape[1] > 0 else 0
    _logger.info(f"Blocks array: shape={blocks.shape}, series in Block_Global: {n_in_block}/{N_actual}")
    
    if n_in_block < N_actual:
        missing_indices = np.where(blocks[:, 0] == 0)[0]
        if columns and len(columns) > 0:
            missing_series = [columns[i] for i in missing_indices[:10]]
            _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} ({missing_series[:5]}...)")
        else:
            _logger.warning(f"  Series NOT in Block_Global: {len(missing_indices)} (indices: {missing_indices[:10].tolist()}...)")


def build_dfm_blocks(
    blocks: np.ndarray,
    config: Any,
    columns: Optional[List[str]],
    N_actual: int
) -> np.ndarray:
    """Rebuild the blocks array (N x n_blocks) so it has ``N_actual`` rows.

    With ``columns`` the array is rebuilt from the config (its block cache is
    cleared first); otherwise it is zero-padded or truncated.
    """
    if columns is not None:
        if hasattr(config, '_cached_blocks'):
            config._cached_blocks = None
        new_blocks = np.array(config.get_blocks_array(columns=columns), dtype=DEFAULT_DTYPE)
        _logger.info(f"Rebuilt blocks array: shape={new_blocks.shape}")
        log_blocks_diagnostics(new_blocks, columns, N_actual)
        return new_blocks
    if blocks.shape[0] < N_actual:
        padding = np.zeros((N_actual - blocks.shape[0], blocks.shape[1]), dtype=DEFAULT_DTYPE)
        _logger.warning(f"Padded blocks array with zeros: {N_actual - blocks.shape[0]} rows")
        return np.vstack([blocks, padding])
    if blocks.shape[0] > N_actual:
        _logger.warning(f"Truncated blocks array: {blocks.shape[0]} -> {N_actual} rows")
        return blocks[:N_actual, :]
    return blocks


def build_dfm_slower_freq_observation_matrix(
    N: int,
    n_clock_freq: int,
    n_slower_freq: int,
    tent_weights: np.ndarray,
    dtype: type = np.float32
) -> np.ndarray:
    """Observation block (N x (len(tent_weights) * n_slower_freq)) of the slower-frequency idiosyncratic chains.

    Rows ``n_clock_freq:`` hold ``kron(I_{n_slower_freq}, tent_weights)``;
    the clock-frequency rows are zero.
    """
    tent_kernel_size = len(tent_weights)
    C_slower_freq = np.zeros((N, tent_kernel_size * n_slower_freq), dtype=dtype)
    C_slower_freq[n_clock_freq:, :] = np.kron(create_scaled_identity(n_slower_freq, DEFAULT_IDENTITY_SCALE, dtype=dtype), tent_weights.reshape(1, -1))
    return C_slower_freq


def build_lag_matrix(
    factors: np.ndarray,
    T: int,
    num_factors: int,
    tent_kernel_size: int,
    p: int,
    dtype: type = np.float32
) -> np.ndarray:
    """Stack lagged factors (T x num_factors) into (T x (num_factors * num_lags)).

    ``num_lags = max(p + 1, tent_kernel_size)``; row ``t``, column block ``k``
    holds ``factors[t - k]`` (zero for ``t < k``), as FRBNY's
    ``[f_t, f_{t-1}, ..., f_{t-num_lags+1}]``.
    """
    num_lags = max(p + 1, tent_kernel_size)
    lag_matrix = np.zeros((T, num_factors * num_lags), dtype=dtype)
    for lag_idx in range(min(num_lags, T)):
        col_start = lag_idx * num_factors
        lag_matrix[lag_idx:, col_start:col_start + num_factors] = factors[:T - lag_idx, :num_factors]
    return lag_matrix


def _pick_optimizer(optimizers: dict, optimizer_type: str):
    """Look up ``optimizer_type`` in ``optimizers`` or raise ``ConfigurationError``."""
    try:
        return optimizers[optimizer_type]
    except KeyError:
        raise ConfigurationError(
            f"optimizer must be one of {sorted(optimizers)}, got {optimizer_type!r}"
        ) from None


def build_ddfm_optimizer(
    model: Any,
    learning_rate: float,
    optimizer_type: str,
    n_mc_samples: int
) -> Tuple[Any, Any]:
    """Build the DDFM optimizer ('Adam', 'AdamW' or 'SGD') and its LambdaLR scheduler.

    The scheduler is stepped once per batch and multiplies the learning rate
    by ``DEFAULT_LR_DECAY_RATE ** (step // n_mc_samples)``, the staircase
    ``ExponentialDecay(decay_steps=n_mc_samples)`` of the reference
    TensorFlow implementation.
    """
    import torch

    optimizers = {
        'Adam': lambda: torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
            eps=DEFAULT_ADAM_EPS
        ),
        'AdamW': lambda: torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
            eps=DEFAULT_ADAM_EPS
        ),
        'SGD': lambda: torch.optim.SGD(model.parameters(), lr=learning_rate)
    }
    optimizer = _pick_optimizer(optimizers, optimizer_type)()

    def lr_lambda(step: int) -> float:
        return DEFAULT_LR_DECAY_RATE ** (step // n_mc_samples)

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_ivdfm_optimizer(
    model: Any,
    learning_rate: float,
    optimizer_type: str,
    max_epochs: int,
    optimizer_weight_decay: float = 0.0,
    optimizer_momentum: float = 0.9,
    scheduler_type: Optional[str] = 'step',
    scheduler_step_size: Optional[int] = None,
    scheduler_gamma: float = 0.5,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.1,
    scheduler_min_lr: float = 0.0,
) -> Tuple[Any, Optional[Any]]:
    """Build the iVDFM optimizer and an epoch-stepped scheduler.

    Parameters
    ----------
    model : torch.nn.Module
    learning_rate : float
    optimizer_type : str
        'Adam', 'AdamW' or 'SGD'.
    max_epochs : int
        Sets the default StepLR step size (``max_epochs // 3``) and cosine ``T_max``.
    optimizer_weight_decay : float, default 0.0
    optimizer_momentum : float, default 0.9
        SGD only.
    scheduler_type : str or None, default 'step'
        'step', 'plateau', 'cosine', 'exponential' or None for no scheduler.
    scheduler_step_size : int, optional
        StepLR step size; None means ``max(1, max_epochs // 3)``.
    scheduler_gamma : float, default 0.5
        StepLR / ExponentialLR decay factor.
    scheduler_patience : int, default 10
    scheduler_factor : float, default 0.1
    scheduler_min_lr : float, default 0.0
        ReduceLROnPlateau settings (``min_lr`` also serves as cosine ``eta_min``).

    Returns
    -------
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler or None
    """
    import torch

    optimizers = {
        'Adam': lambda: torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
            eps=DEFAULT_ADAM_EPS,
            weight_decay=optimizer_weight_decay
        ),
        'AdamW': lambda: torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(DEFAULT_ADAM_BETA1, DEFAULT_ADAM_BETA2),
            eps=DEFAULT_ADAM_EPS,
            weight_decay=optimizer_weight_decay
        ),
        'SGD': lambda: torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=optimizer_momentum,
            weight_decay=optimizer_weight_decay
        )
    }
    optimizer = _pick_optimizer(optimizers, optimizer_type)()

    if scheduler_type is None:
        return optimizer, None
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=scheduler_min_lr
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_gamma
        )
    elif scheduler_type == 'step':
        step_size = scheduler_step_size if scheduler_step_size is not None else max(1, max_epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=scheduler_gamma
        )
    else:
        raise ConfigurationError(
            "scheduler_type must be one of ['step', 'plateau', 'cosine', 'exponential'] or None, "
            f"got {scheduler_type!r}"
        )

    return optimizer, scheduler


def build_ddfm_state_space(
    factors: np.ndarray,
    eps: np.ndarray,
    decoder_weight: np.ndarray,
    observed_y: np.ndarray,
    model_name: str = "DDFM"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the DDFM state space (F, Q, mu_0, Sigma_0, H, R) from a trained autoencoder.

    Non-finite entries of ``factors`` / ``eps`` are replaced by zero with a
    warning. The transition block is the factor part of
    :func:`get_transition_params`; ``H`` is the first ``num_factors`` columns
    of ``decoder_weight`` and ``R`` is ``DEFAULT_DDFM_OBSERVATION_NOISE * I``.

    Parameters
    ----------
    factors : np.ndarray
        MC-averaged factors (T x num_factors).
    eps : np.ndarray
        Idiosyncratic residuals (T x N).
    decoder_weight : np.ndarray
        Decoder linear weight (N x >= num_factors).
    observed_y : np.ndarray
        Boolean observation mask (T x N).
    model_name : str, default "DDFM"
        Used in error messages.

    Returns
    -------
    F, Q, Sigma_0 : np.ndarray
        (num_factors x num_factors); ``mu_0`` is (num_factors,), ``H`` is
        (N x num_factors) and ``R`` is (N x N).

    Raises
    ------
    ModelNotTrainedError
        If ``factors`` or ``eps`` is empty.
    ValueError
        If ``decoder_weight`` has fewer than ``num_factors`` columns.
    NumericalError
        If transition parameter estimation fails.
    """
    if factors is None or len(factors) == 0:
        raise ModelNotTrainedError(
            f"{model_name}: Cannot build state space - factors are empty. "
            "Model must be trained before building state space."
        )
    if eps is None or len(eps) == 0:
        raise ModelNotTrainedError(
            f"{model_name}: Cannot build state space - residuals are empty. "
            "Model must be trained before building state space."
        )

    if not np.all(np.isfinite(factors)):
        _logger.warning(
            "build_ddfm_state_space: Factors contain non-finite values. "
            "Replacing with zeros to allow state space construction."
        )
        factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.all(np.isfinite(eps)):
        _logger.warning(
            "build_ddfm_state_space: Residuals contain non-finite values. "
            "Replacing with zeros to allow state space construction."
        )
        eps = np.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
    
    num_factors = factors.shape[1]
    if decoder_weight.shape[1] < num_factors:
        raise ValueError(
            f"build_ddfm_state_space: Decoder weight shape {decoder_weight.shape} incompatible with "
            f"{num_factors} factors. Expected at least {num_factors} columns."
        )
    H = decoder_weight[:, :num_factors]

    try:
        F_full, Q_full, mu_0_full, Sigma_0_full, _ = get_transition_params(
            factors, eps, bool_no_miss=observed_y
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        raise NumericalError(
            f"{model_name}: Failed to estimate transition parameters. "
            f"Error: {type(e).__name__}: {str(e)}. "
            "This may indicate numerical instability in factor estimation. "
            "Consider: (1) Checking data quality, (2) Reducing number of factors, "
            "(3) Increasing training iterations."
        ) from e

    # The full transition is block-diagonal [[A_f, 0], [0, Phi]] over (factors, idiosyncratic).
    F = F_full[:num_factors, :num_factors]
    Q = Q_full[:num_factors, :num_factors]
    mu_0 = mu_0_full[:num_factors]
    Sigma_0 = Sigma_0_full[:num_factors, :num_factors]
    R = np.eye(eps.shape[1], dtype=DEFAULT_DTYPE) * DEFAULT_DDFM_OBSERVATION_NOISE

    return F, Q, mu_0, Sigma_0, H, R


def ivdfm_companion_from_p(p: "torch.Tensor") -> "torch.Tensor":
    """Companion matrix (..., p, p) of AR coefficients ``p`` of shape (..., p).

    For the state ``[f_t, f_{t-1}, ..., f_{t-p+1}]`` (current value first, the
    layout ``iVDFMCompanionSSM`` reads with ``C``): the coefficients in the
    first row (``p_k`` multiplies ``f_{t-k}``) and a shift on the sub-diagonal::

        [[p_0, p_1, ..., p_{p-1}],
         [1,   0,   ..., 0      ],
         ...
         [0, ..., 1,        0   ]]
    """
    import torch

    d = p.shape[-1]
    A = torch.zeros(*p.shape[:-1], d, d, dtype=p.dtype, device=p.device)
    if d > 1:
        A[..., 1:, :-1] = torch.eye(d - 1, dtype=p.dtype, device=p.device)
    A[..., 0, :] = p
    return A


def build_ivdfm_diagonal_companion(
    ar_coeffs: "torch.Tensor"
) -> "torch.Tensor":
    """Block-diagonal (r*p, r*p) matrix of per-factor AR(p) companions from ``ar_coeffs`` (r, p).

    Factors stay independent, which the iVDFM identification requires.
    """
    import torch

    r, p = ar_coeffs.shape
    A_block = torch.zeros(r * p, r * p, dtype=ar_coeffs.dtype, device=ar_coeffs.device)
    for i in range(r):
        A_block[i * p:(i + 1) * p, i * p:(i + 1) * p] = ivdfm_companion_from_p(ar_coeffs[i, :])
    return A_block


def build_afm_optimizer(
    params: List[Any],
    learning_rate: float,
    optimizer_type: str = 'Adam',
    weight_decay: float = 0.0,
) -> Any:
    """Build the AFM optimizer over ``params`` (no scheduler).

    ``optimizer_type`` is 'Adam', 'AdamW' or 'SGD', matched case-insensitively.
    """
    import torch

    opt = optimizer_type.lower()
    if opt == 'adamw':
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if opt == 'sgd':
        return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    if opt == 'adam':
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    raise ConfigurationError(f"optimizer must be one of ['Adam', 'AdamW', 'SGD'], got {optimizer_type!r}")


__all__ = [
    'build_dfm_structure',
    'build_dfm_blocks',
    'build_dfm_slower_freq_observation_matrix',
    'build_lag_matrix',
    'build_ddfm_optimizer',
    'build_ivdfm_optimizer',
    'build_afm_optimizer',
    'build_ddfm_state_space',
    'ivdfm_companion_from_p',
    'build_ivdfm_diagonal_companion',
]


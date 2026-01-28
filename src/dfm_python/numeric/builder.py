"""State-space model building functions.

This module provides functions for building state-space models,
including observation matrix construction and state-space assembly.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List

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
from ..utils.errors import ModelNotTrainedError, NumericalError

_logger = get_logger(__name__)


def build_dfm_structure(config: Any, *, columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build DFM model structure from configuration.
    
    Parameters
    ----------
    config : Any
        DFMConfig instance with get_blocks_array() method
    columns : list[str], optional
        If provided, used to auto-create a single-frequency mapping (all series use clock)
        when config.frequency is missing. This enables "minimal" configs that omit frequency.
    
    Returns
    -------
    blocks : np.ndarray
        Block structure array (N x n_blocks)
    r : np.ndarray
        Number of factors per block (n_blocks,)
    num_factors : int
        Total number of factors
    p : int
        VAR lag order (always 1 for factors)
    """
    # Get model structure (stored as NumPy arrays)
    # Cache blocks array to avoid multiple calls to get_blocks_array()
    blocks_array = config.get_blocks_array(columns=columns) if columns is not None else config.get_blocks_array()
    blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)
    
    # Get factors per block (r)
    factors_per_block = getattr(config, 'factors_per_block', None)
    if factors_per_block is not None:
        r = np.array(factors_per_block, dtype=DEFAULT_DTYPE)
    else:
        r = np.ones(blocks_array.shape[1], dtype=DEFAULT_DTYPE)
    
    # Total number of factors (computed from r to avoid redundancy)
    num_factors = int(np.sum(r))
    
    # AR order (always AR(1) for factors)
    p = DEFAULT_FACTOR_ORDER
    
    return blocks, r, num_factors, p


def build_dfm_blocks(
    blocks: np.ndarray,
    config: Any,
    columns: Optional[List[str]],
    N_actual: int
) -> np.ndarray:
    """Rebuild DFM blocks array to match data dimensions.
    
    Parameters
    ----------
    blocks : np.ndarray
        Current blocks array
    config : Any
        Config object with get_blocks_array() method
    columns : Optional[List[str]]
        Column names if available
    N_actual : int
        Expected number of series
        
    Returns
    -------
    np.ndarray
        Updated blocks array matching data dimensions
    """
    from ..logger.dfm_logger import log_blocks_diagnostics
    
    if columns is not None:
        # Clear cache and rebuild from config
        if hasattr(config, '_cached_blocks'):
            config._cached_blocks = None
        blocks_array = config.get_blocks_array(columns=columns)
        new_blocks = np.array(blocks_array, dtype=DEFAULT_DTYPE)
        _logger.info(f"Rebuilt blocks array: shape={new_blocks.shape}")
        log_blocks_diagnostics(new_blocks, columns, N_actual)
        return new_blocks
    else:
        # Fallback: pad or truncate to match dimensions
        n_blocks = blocks.shape[1]
        if blocks.shape[0] < N_actual:
            padding = np.zeros((N_actual - blocks.shape[0], n_blocks), dtype=DEFAULT_DTYPE)
            new_blocks = np.vstack([blocks, padding])
            _logger.warning(f"Padded blocks array with zeros: {N_actual - blocks.shape[0]} rows")
            return new_blocks
        elif blocks.shape[0] > N_actual:
            new_blocks = blocks[:N_actual, :]
            _logger.warning(f"Truncated blocks array: {blocks.shape[0]} -> {N_actual} rows")
            return new_blocks
        else:
            return blocks


def build_dfm_slower_freq_observation_matrix(
    N: int,
    n_clock_freq: int,
    n_slower_freq: int,
    tent_weights: np.ndarray,
    dtype: type = np.float32
) -> np.ndarray:
    """Build observation matrix for slower-frequency idiosyncratic chains.
    
    Parameters
    ----------
    N : int
        Total number of series
    n_clock_freq : int
        Number of clock-frequency series (series at the clock frequency, generic)
    n_slower_freq : int
        Number of slower-frequency series (series slower than clock frequency, generic)
    tent_weights : np.ndarray
        Tent weights array (e.g., [1, 2, 3, 2, 1])
    dtype : type, default np.float32
        Data type for output matrix
        
    Returns
    -------
    np.ndarray
        Observation matrix (N x (tent_kernel_size * n_slower_freq))
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
    """Build lag matrix for factors.
    
    Parameters
    ----------
    factors : np.ndarray
        Factor matrix (T x num_factors)
    T : int
        Number of time periods
    num_factors : int
        Number of factors
    tent_kernel_size : int
        Tent kernel size
    p : int
        AR lag order
    dtype : type
        Data type
        
    Returns
    -------
    np.ndarray
        Lag matrix (T x (num_factors * num_lags))
    """
    num_lags = max(p + 1, tent_kernel_size)
    lag_matrix = np.zeros((T, num_factors * num_lags), dtype=dtype)
    
    # Vectorized implementation: build all lags at once
    for lag_idx in range(num_lags):
        start_idx = max(0, tent_kernel_size - lag_idx)
        end_idx = T - lag_idx
        if start_idx < end_idx:
            col_start = lag_idx * num_factors
            col_end = col_start + num_factors
            # Use advanced indexing for better performance
            lag_matrix[start_idx:end_idx, col_start:col_end] = factors[start_idx:end_idx, :num_factors].copy()
    
    return lag_matrix


def build_ddfm_optimizer(
    model: Any,
    learning_rate: float,
    optimizer_type: str,
    n_mc_samples: int
) -> Tuple[Any, Any]:
    """Build optimizer and scheduler for DDFM training.
    
    Creates optimizer (Adam/AdamW/SGD) and learning rate scheduler (LambdaLR).
    
    **Learning Rate Decay Implementation:**
    TensorFlow's ExponentialDecay with decay_steps=n_mc_samples (DEFAULT_N_MC_SAMPLES) and staircase=True
    decays every n_mc_samples optimizer steps (batches), not every n_mc_samples epochs.
    
    **Implementation (Fixed 2026-01-07):**
    - Scheduler steps after each batch in autoencoder.fit() (simple_autoencoder.py:269)
    - LambdaLR scheduler uses step count (number of batches) to compute decay
    - Decays every n_mc_samples scheduler steps (batches) → matches TensorFlow behavior
    - Learning rate multiplier: decay_rate ^ (step // n_mc_samples)
    - Mathematical verification: Matches TensorFlow's ExponentialDecay(decay_steps=n_mc_samples, decay_rate=0.96, staircase=True)
    
    Parameters
    ----------
    model : Any
        PyTorch model (autoencoder) with parameters() method
    learning_rate : float
        Initial learning rate
    optimizer_type : str
        Optimizer type ('Adam', 'AdamW', or 'SGD')
    n_mc_samples : int
        Number of Monte Carlo samples (used for learning rate decay steps)
        
    Returns
    -------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer instance
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler instance
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
    optimizer = optimizers.get(optimizer_type, optimizers['SGD'])()
    
    def lr_lambda(step: int) -> float:
        """Compute learning rate multiplier for per-batch decay (matches TensorFlow behavior).
        
        TensorFlow: ExponentialDecay(decay_steps=n_mc_samples, decay_rate=DEFAULT_LR_DECAY_RATE, staircase=True)
        - Decays every n_mc_samples optimizer steps (batches)
        
        Our implementation (fixed 2026-01-07):
        - Scheduler steps after each batch in autoencoder.fit() (simple_autoencoder.py:269)
        - step parameter is scheduler step count (number of batches completed)
        - Decays every n_mc_samples scheduler steps (batches) → matches TensorFlow behavior
        - Mathematical equivalence: DEFAULT_LR_DECAY_RATE ^ (step // n_mc_samples) matches TensorFlow's staircase=True behavior
        
        Returns:
            Learning rate multiplier: DEFAULT_LR_DECAY_RATE ^ (step // n_mc_samples)
        """
        # Decay every n_mc_samples scheduler steps (batches)
        # Scheduler steps after each batch, so step count equals batch count
        decay_steps = step // n_mc_samples
        return DEFAULT_LR_DECAY_RATE ** decay_steps
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    
    return optimizer, scheduler


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
    """Build optimizer and scheduler for iVDFM training.
    
    Creates optimizer (Adam/AdamW/SGD) with configurable parameters and
    learning rate scheduler (StepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR).
    Uses epoch-based learning rate decay, unlike DDFM which uses per-batch decay.
    
    Parameters
    ----------
    model : Any
        PyTorch model (iVDFM) with parameters() method
    learning_rate : float
        Initial learning rate
    optimizer_type : str
        Optimizer type ('Adam', 'AdamW', or 'SGD')
    max_epochs : int
        Maximum number of training epochs (used for scheduler step_size)
    optimizer_weight_decay : float, default 0.0
        Weight decay (L2 regularization) for optimizer
    optimizer_momentum : float, default 0.9
        Momentum for SGD optimizer
    scheduler_type : Optional[str], default 'step'
        Scheduler type: 'step', 'plateau', 'cosine', 'exponential', or None
    scheduler_step_size : Optional[int], default None
        Step size for StepLR (None = auto: max_epochs // 3)
    scheduler_gamma : float, default 0.5
        Gamma (decay factor) for StepLR/ExponentialLR
    scheduler_patience : int, default 10
        Patience for ReduceLROnPlateau
    scheduler_factor : float, default 0.1
        Factor for ReduceLROnPlateau
    scheduler_min_lr : float, default 0.0
        Minimum learning rate for ReduceLROnPlateau
        
    Returns
    -------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer instance
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler instance (None if scheduler_type is None)
    """
    import torch
    
    # Build optimizer with configurable parameters
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
    optimizer = optimizers.get(optimizer_type, optimizers['Adam'])()
    
    # Build scheduler based on type
    scheduler = None
    if scheduler_type is None:
        return optimizer, None
    
    if scheduler_type == 'step':
        # StepLR: decays every step_size epochs
        step_size = scheduler_step_size if scheduler_step_size is not None else max(1, max_epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=scheduler_gamma
        )
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau: reduces LR when metric plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr
        )
    elif scheduler_type == 'cosine':
        # CosineAnnealingLR: cosine annealing schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=scheduler_min_lr
        )
    elif scheduler_type == 'exponential':
        # ExponentialLR: exponential decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_gamma
        )
    else:
        _logger.warning(
            f"Unknown scheduler_type '{scheduler_type}', using StepLR as default"
        )
        step_size = max(1, max_epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=scheduler_gamma
        )
    
    return optimizer, scheduler


def build_ddfm_state_space(
    factors: np.ndarray,
    eps: np.ndarray,
    decoder_weight: np.ndarray,
    observed_y: np.ndarray,
    model_name: str = "DDFM"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build DDFM state-space model parameters from trained autoencoder.
    
    Parameters
    ----------
    factors : np.ndarray
        Extracted factors (T x num_factors) - averaged across MC samples
    eps : np.ndarray
        Idiosyncratic residuals (T x num_target_series)
    decoder_weight : np.ndarray
        Decoder linear layer weight matrix (output_dim x input_dim)
    observed_y : np.ndarray
        Boolean mask for observed target values (T x num_target_series)
    model_name : str, default "DDFM"
        Model name for error messages
        
    Returns
    -------
    F : np.ndarray
        Factor transition matrix (num_factors x num_factors)
    Q : np.ndarray
        Factor process noise covariance (num_factors x num_factors)
    mu_0 : np.ndarray
        Initial factor mean (num_factors,)
    Sigma_0 : np.ndarray
        Initial factor covariance (num_factors x num_factors)
    H : np.ndarray
        Observation matrix (num_target_series x num_factors)
    R : np.ndarray
        Observation noise covariance (num_target_series x num_target_series)
        
    Raises
    ------
    ModelNotTrainedError
        If factors or residuals are empty
    ValueError
        If decoder weight shape is incompatible with number of factors
    NumericalError
        If transition parameter estimation fails
    """
    # Validate factors and residuals
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
    
    # Check for non-finite values
    if not np.all(np.isfinite(factors)):
        _logger.warning(
            f"build_ddfm_state_space: Factors contain non-finite values. "
            "Replacing with zeros to allow state space construction."
        )
        factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.all(np.isfinite(eps)):
        _logger.warning(
            f"build_ddfm_state_space: Residuals contain non-finite values. "
            "Replacing with zeros to allow state space construction."
        )
        eps = np.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
    
    num_factors = factors.shape[1]
    
    # Validate decoder weight shape
    if decoder_weight.shape[1] < num_factors:
        raise ValueError(
            f"build_ddfm_state_space: Decoder weight shape {decoder_weight.shape} incompatible with "
            f"{num_factors} factors. Expected at least {num_factors} columns."
        )
    
    # Extract observation matrix from decoder weights
    H = decoder_weight[:, :num_factors]
    
    # Get transition equation params (factor_order is fixed to 1)
    # F_full includes both factors and idiosyncratic components: shape (m + N, m + N)
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
    
    # F_full structure: [[A_f, 0], [0, Phi]] where A_f is (m x m) factor transition
    F = F_full[:num_factors, :num_factors]  # Factor transition matrix (m x m)
    Q = Q_full[:num_factors, :num_factors]  # Factor process noise (m x m)
    mu_0 = mu_0_full[:num_factors]  # Initial factor mean (m,)
    Sigma_0 = Sigma_0_full[:num_factors, :num_factors]  # Initial factor covariance (m x m)
    
    # Observation noise covariance (diagonal, small values)
    R = np.eye(eps.shape[1], dtype=DEFAULT_DTYPE) * DEFAULT_DDFM_OBSERVATION_NOISE
    
    return F, Q, mu_0, Sigma_0, H, R


def ivdfm_companion_from_p(p: torch.Tensor) -> torch.Tensor:
    """Construct companion matrix from AR coefficients for iVDFM.
    
    For AR(p) with coefficients p = [p_0, p_1, ..., p_{p-1}]:
    A = [[0, 1, 0, ..., 0],
         [0, 0, 1, ..., 0],
         ...
         [p_0, p_1, ..., p_{p-1}]]
    
    Parameters
    ----------
    p : torch.Tensor
        AR coefficients, shape (..., p)
        
    Returns
    -------
    torch.Tensor
        Companion matrix, shape (..., p, p)
    """
    d = p.shape[-1]
    batch_dims = p.shape[:-1]
    
    A = torch.zeros(*batch_dims, d, d, dtype=p.dtype, device=p.device)
    # Shift matrix (upper diagonal)
    if d > 1:
        A[..., 1:, :-1] = torch.eye(d - 1, dtype=p.dtype, device=p.device)
    # Last row = AR coefficients
    A[..., -1, :] = p
    
    return A


def build_ivdfm_diagonal_companion(
    ar_coeffs: torch.Tensor
) -> torch.Tensor:
    """Build block-diagonal companion matrix for iVDFM multiple factors.
    
    Each factor has its own AR(p) dynamics. Creates block-diagonal structure
    to preserve component-wise independence (identifiability requirement).
    
    Parameters
    ----------
    ar_coeffs : torch.Tensor
        AR coefficients per factor, shape (r, p) where:
        - r: number of factors
        - p: AR order
        
    Returns
    -------
    torch.Tensor
        Block-diagonal companion matrix, shape (r*p, r*p)
        Each block is (p, p) companion matrix
    """
    r, p = ar_coeffs.shape
    
    # Initialize block-diagonal matrix
    A_block = torch.zeros(r * p, r * p, dtype=ar_coeffs.dtype, device=ar_coeffs.device)
    
    # Build companion matrix for each factor
    for i in range(r):
        start_idx = i * p
        end_idx = (i + 1) * p
        
        # Get AR coefficients for this factor
        p_i = ar_coeffs[i, :]  # (p,)
        
        # Build companion matrix for this factor
        A_i = ivdfm_companion_from_p(p_i)  # (p, p)
        
        # Place in block-diagonal position
        A_block[start_idx:end_idx, start_idx:end_idx] = A_i
    
    return A_block


__all__ = [
    'build_dfm_structure',
    'build_dfm_blocks',
    'build_dfm_slower_freq_observation_matrix',
    'build_lag_matrix',
    'build_ddfm_optimizer',
    'build_ivdfm_optimizer',
    'build_ddfm_state_space',
    'ivdfm_companion_from_p',
    'build_ivdfm_diagonal_companion',
]


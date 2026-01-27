"""MCMC sampling logic for DDFM training.

This module contains the core MCMC-based denoising training logic,
including denoising, MC sample generation, and training iteration.
"""

import numpy as np
import torch
import pandas as pd
from typing import Tuple, Optional, List, Any
from ...logger import get_logger
from ...numeric.estimator import get_idio
from ...numeric.statistic import (
    compute_variance_mean, compute_tensor_stats, diagnose_variance_collapse
)
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_MCMC_EPOCHS,
    DEFAULT_VARIANCE_COLLAPSE_THRESHOLD,
    DEFAULT_LOSS_LOG_PRECISION,
)
from ...config.types import to_numpy
from ...utils.helper import interpolate_dataframe

_logger = get_logger(__name__)


def denoise_targets(
    eps: np.ndarray,
    data_imputed: pd.DataFrame,
    data_denoised: pd.DataFrame,
    dataset: Any,
    lags_input: int,
    interpolation_method: str,
    interpolation_limit: Optional[int],
    interpolation_limit_direction: str,
    min_obs: int = 1
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Denoise target series using AR-idio model.
    
    Estimates AR-idio parameters from residuals, then subtracts the conditional
    AR-idio mean from target series to denoise the data.
    
    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals (T x num_target_series)
    data_imputed : pd.DataFrame
        Imputed data (with predictions filling missing values)
    data_denoised : pd.DataFrame
        Denoised data (will be updated in-place)
    dataset : Any
        DDFMDataset instance (for observed_y and target_indices)
    lags_input : int
        Number of lagged inputs (offset for denoising)
    interpolation_method : str
        Interpolation method for denoised data
    interpolation_limit : Optional[int]
        Maximum number of consecutive NaNs to interpolate
    interpolation_limit_direction : str
        Interpolation direction ('forward', 'backward', or 'both')
    min_obs : int, default 1
        Minimum observations required for AR-idio estimation
        
    Returns
    -------
    data_denoised_interpolated : pd.DataFrame
        Denoised and interpolated data
    Phi : np.ndarray
        AR-idio transition matrix
    mu_eps : np.ndarray
        AR-idio mean (num_target_series,)
    std_eps : np.ndarray
        AR-idio std (num_target_series,)
    """
    # Estimate AR-idio parameters from residuals
    Phi, mu_eps, std_eps = get_idio(eps, dataset.observed_y, min_obs=min_obs)
    
    # Denoise: subtract conditional AR-idio mean from target series only
    # Features (X) are only used for encoder input, not for denoising
    # eps @ Phi gives (T-1, num_target_series), update only target columns
    eps_denoise = eps[:-1, :] @ Phi  # (T-1, num_target_series)
    data_denoised.values[lags_input+1:, dataset.target_indices] = \
        data_imputed.values[lags_input+1:, dataset.target_indices] - eps_denoise
    
    # Interpolate denoised data
    data_denoised_interpolated = interpolate_dataframe(
        data_denoised,
        method=interpolation_method,
        limit=interpolation_limit,
        limit_direction=interpolation_limit_direction
    )
    
    return data_denoised_interpolated, Phi, mu_eps, std_eps


def train_on_mc_samples(
    autoencoder: Any,
    autoencoder_datasets: List[Any],
    window_size: int,
    learning_rate: float,
    optimizer_type: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    target_indices: Optional[torch.Tensor],
    device: torch.device
) -> None:
    """Train autoencoder on Monte Carlo samples.
    
    Parameters
    ----------
    autoencoder : Any
        Autoencoder model to train
    autoencoder_datasets : List[Any]
        List of AutoencoderDataset instances (one per MC sample)
    window_size : int
        Batch size for training
    learning_rate : float
        Learning rate (may be overridden by scheduler)
    optimizer_type : str
        Optimizer type (for logging/debugging)
    optimizer : torch.optim.Optimizer
        PyTorch optimizer instance
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    target_indices : Optional[torch.Tensor]
        Target column indices tensor (None if all columns are targets)
    device : torch.device
        Device for training
    """
    autoencoder.train()
    for ae_dataset in autoencoder_datasets:
        autoencoder.fit(
            dataset=ae_dataset,
            epochs=DEFAULT_MCMC_EPOCHS,
            batch_size=window_size,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            optimizer=optimizer,
            scheduler=scheduler,
            target_indices=target_indices
        )


def extract_predictions_from_mc_samples(
    encoder: Any,
    decoder: Any,
    autoencoder_datasets: List[Any],
    n_mc_samples: int,
    extract_target_predictions: callable,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract factors and predictions from Monte Carlo samples.
    
    Parameters
    ----------
    encoder : Any
        Encoder model
    decoder : Any
        Decoder model
    autoencoder_datasets : List[Any]
        List of AutoencoderDataset instances (one per MC sample)
    n_mc_samples : int
        Expected number of MC samples (for validation)
    extract_target_predictions : callable
        Function to extract target predictions from full predictions
    device : torch.device
        Device for computation
        
    Returns
    -------
    factors : np.ndarray
        Extracted factors (n_mc_samples x T x num_factors)
    y_pred : np.ndarray
        Target predictions (T x num_target_series)
    y_pred_full : np.ndarray
        Full predictions (T x num_series)
    y_pred_std : np.ndarray
        Prediction standard deviation (T x num_target_series)
    factor_std : np.ndarray
        Factor standard deviation (T x num_factors)
    factors_mean : np.ndarray
        Factor mean (T x num_factors)
    """
    with torch.no_grad():
        factors_list = [encoder(ae_dataset.full_input) for ae_dataset in autoencoder_datasets]
        factors_tensor = torch.stack(factors_list, dim=0)
        
        y_pred_samples_tensor = torch.stack([decoder(f) for f in factors_list], dim=0)
        
        # Validate MC sample dimension
        if factors_tensor.shape[0] != n_mc_samples:
            raise ValueError(
                f"MC samples dimension mismatch: factors_tensor.shape[0]={factors_tensor.shape[0]} != n_mc_samples={n_mc_samples}"
            )
        if y_pred_samples_tensor.shape[0] != n_mc_samples:
            raise ValueError(
                f"MC samples dimension mismatch: y_pred_samples_tensor.shape[0]={y_pred_samples_tensor.shape[0]} != n_mc_samples={n_mc_samples}"
            )
        
        y_pred_full_tensor, y_pred_std_tensor = compute_tensor_stats(y_pred_samples_tensor)
        y_pred_tensor = extract_target_predictions(y_pred_full_tensor)
        y_pred_full = to_numpy(y_pred_full_tensor)
        y_pred = to_numpy(y_pred_tensor)
        y_pred_std = to_numpy(y_pred_std_tensor)
        factors = to_numpy(factors_tensor)
        
        _, factors_std_tensor = compute_tensor_stats(factors_tensor)
        factor_std = to_numpy(factors_std_tensor)
        factors_mean_tensor, _ = compute_tensor_stats(factors_tensor)
        factors_mean = to_numpy(factors_mean_tensor)
    
    return factors, y_pred, y_pred_full, y_pred_std, factor_std, factors_mean


def check_variance_collapse(
    y_pred_std: np.ndarray,
    y_pred_full: np.ndarray,
    factors_mean: np.ndarray,
    y_actual: np.ndarray,
    target_scaler: Any,
    encoder: Any,
    decoder: Any,
    factors_std: np.ndarray,
    num_iter: int,
    disp: int
) -> Optional[dict]:
    """Check for variance collapse in predictions and factors.
    
    Parameters
    ----------
    y_pred_std : np.ndarray
        Prediction standard deviation (T x num_target_series)
    y_pred_full : np.ndarray
        Full predictions (T x num_series)
    factors_mean : np.ndarray
        Factor mean (T x num_factors)
    y_actual : np.ndarray
        Actual target values (T x num_target_series)
    target_scaler : Any
        Scaler for target series
    encoder : Any
        Encoder model
    decoder : Any
        Decoder model
    factors_std : np.ndarray
        Factor standard deviation (T x num_factors)
    num_iter : int
        Current iteration number
    disp : int
        Display frequency (check variance every disp iterations)
        
    Returns
    -------
    Optional[dict]
        Variance diagnostics dict if checked, None otherwise
    """
    y_pred_std_mean_check = compute_variance_mean(y_pred_std)
    should_check_variance = (
        (y_pred_std_mean_check is not None and y_pred_std_mean_check < DEFAULT_VARIANCE_COLLAPSE_THRESHOLD) or
        (num_iter % disp == 0)
    )
    
    if should_check_variance:
        variance_diagnostics = diagnose_variance_collapse(
            prediction_std=y_pred_std,
            prediction_mean=y_pred_full,
            factors_mean=factors_mean,
            y_actual=y_actual,
            target_scaler=target_scaler,
            encoder=encoder,
            decoder=decoder,
            factors_std=factors_std
        )
        if variance_diagnostics['variance_collapse_detected']:
            _logger.warning(f"Variance collapse detected at iteration {num_iter}: {', '.join(variance_diagnostics['warnings'])}")
        return variance_diagnostics
    
    return None


def run_mcmc_iteration(
    eps: np.ndarray,
    data_imputed: pd.DataFrame,
    data_denoised: pd.DataFrame,
    dataset: Any,
    encoder: Any,
    decoder: Any,
    autoencoder: Any,
    y_actual: np.ndarray,
    lags_input: int,
    n_mc_samples: int,
    window_size: int,
    learning_rate: float,
    optimizer_type: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    extract_target_predictions: callable,
    interpolation_method: str,
    interpolation_limit: Optional[int],
    interpolation_limit_direction: str,
    target_scaler: Any,
    num_iter: int,
    disp: int,
    device: torch.device,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[Any]]:
    """Run a single MCMC iteration: denoise, sample, train, predict.
    
    Parameters
    ----------
    eps : np.ndarray
        Idiosyncratic residuals (T x num_target_series)
    data_imputed : pd.DataFrame
        Imputed data (with predictions filling missing values)
    data_denoised : pd.DataFrame
        Denoised data (will be updated in-place)
    dataset : Any
        DDFMDataset instance (provides observed_y, target_indices, all_columns_are_targets)
    encoder : Any
        Encoder model
    decoder : Any
        Decoder model
    autoencoder : Any
        Autoencoder model
    y_actual : np.ndarray
        Actual target values (T x num_target_series)
    lags_input : int
        Number of lagged inputs
    n_mc_samples : int
        Number of Monte Carlo samples
    window_size : int
        Batch size for training
    learning_rate : float
        Learning rate
    optimizer_type : str
        Optimizer type
    optimizer : torch.optim.Optimizer
        PyTorch optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    extract_target_predictions : callable
        Function to extract target predictions
    interpolation_method : str
        Interpolation method
    interpolation_limit : Optional[int]
        Interpolation limit
    interpolation_limit_direction : str
        Interpolation direction
    target_scaler : Any
        Target scaler
    num_iter : int
        Current iteration number
    disp : int
        Display frequency
    device : torch.device
        Device for computation
    rng : np.random.RandomState
        Random number generator
        
    Returns
    -------
    factors : np.ndarray
        Extracted factors (n_mc_samples x T x num_factors)
    y_pred : np.ndarray
        Target predictions (T x num_target_series)
    y_pred_full : np.ndarray
        Full predictions (T x num_series)
    y_pred_std : np.ndarray
        Prediction standard deviation (T x num_target_series)
    factor_std : np.ndarray
        Factor standard deviation (T x num_factors)
    data_denoised_interpolated : pd.DataFrame
        Denoised and interpolated data
    autoencoder_datasets : List[Any]
        List of AutoencoderDataset instances (for MLP decoder last_neurons extraction)
    """
    # Step 1: Denoise targets
    data_denoised_interpolated, Phi, mu_eps, std_eps = denoise_targets(
        eps=eps,
        data_imputed=data_imputed,
        data_denoised=data_denoised,
        dataset=dataset,
        lags_input=lags_input,
        interpolation_method=interpolation_method,
        interpolation_limit=interpolation_limit,
        interpolation_limit_direction=interpolation_limit_direction
    )
    
    # Step 2: Generate MC samples
    X_features_df, y_tmp = dataset.split_features_and_targets(data_denoised_interpolated)
    X_features = X_features_df if X_features_df is not None else pd.DataFrame()
    
    autoencoder_datasets = dataset.create_autoencoder_datasets_list(
        n_mc_samples=n_mc_samples,
        mu_eps=mu_eps,
        std_eps=std_eps,
        X=X_features,
        y_tmp=y_tmp,
        y_actual=y_actual,
        rng=rng,
        device=device
    )
    
    # Step 3: Train on MC samples
    train_on_mc_samples(
        autoencoder=autoencoder,
        autoencoder_datasets=autoencoder_datasets,
        window_size=window_size,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        optimizer=optimizer,
        scheduler=scheduler,
        target_indices=None,  # Not used - decoder output_dim is already num_target_series
        device=device
    )
    
    # Step 4: Extract predictions
    factors, y_pred, y_pred_full, y_pred_std, factor_std, factors_mean = extract_predictions_from_mc_samples(
        encoder=encoder,
        decoder=decoder,
        autoencoder_datasets=autoencoder_datasets,
        n_mc_samples=n_mc_samples,
        extract_target_predictions=extract_target_predictions,
        device=device
    )
    
    # Step 5: Check variance collapse
    check_variance_collapse(
        y_pred_std=y_pred_std,
        y_pred_full=y_pred_full,
        factors_mean=factors_mean,
        y_actual=y_actual,
        target_scaler=target_scaler,
        encoder=encoder,
        decoder=decoder,
        factors_std=factor_std,
        num_iter=num_iter,
        disp=disp
    )
    
    return factors, y_pred, y_pred_full, y_pred_std, factor_std, data_denoised_interpolated, autoencoder_datasets

"""MCMC iterative procedure for DDFM training (PyTorch version of original TensorFlow implementation)."""

import numpy as np
from typing import Optional, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None

from ..core.helpers import get_logger

_logger = get_logger(__name__)


def fit_ddfm_mcmc(
    encoder: nn.Module,
    decoder: nn.Module,
    x_standardized: np.ndarray,
    x_clean: np.ndarray,
    missing_mask: np.ndarray,
    epochs_per_iter: int,
    batch_size: int,
    learning_rate: float,
    max_iter: int,
    tolerance: float,
    disp: int,
    device: torch.device,
    rng: np.random.RandomState,
    use_idiosyncratic: bool,
    min_obs_idio: int,
    lags_input: int,
    get_idio_func,
    convergence_checker_func,
    train_autoencoder_func,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """MCMC iterative training procedure (matches original TensorFlow DDFM).
    
    Parameters
    ----------
    encoder : nn.Module
        Encoder network
    decoder : nn.Module
        Decoder network
    x_standardized : np.ndarray
        Standardized data with missing values (T x N)
    x_clean : np.ndarray
        Clean data (interpolated, T x N)
    missing_mask : np.ndarray
        Missing data mask (T x N), True where data is missing
    epochs_per_iter : int
        Number of epochs per MCMC iteration
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for Adam optimizer
    max_iter : int
        Maximum number of MCMC iterations
    tolerance : float
        Convergence tolerance
    disp : int
        Display progress every 'disp' iterations
    device : torch.device
        Device to run training on
    rng : np.random.RandomState
        Random number generator for MC sampling
    use_idiosyncratic : bool
        Whether to model idiosyncratic components
    min_obs_idio : int
        Minimum observations for idio estimation
    lags_input : int
        Number of input lags (currently not used, for future extension)
    get_idio_func : callable
        Function to estimate idio parameters: get_idio(eps, idx_no_missings, min_obs)
    convergence_checker_func : callable
        Function to check convergence: convergence_checker(y_prev, y_now, y_actual)
    train_autoencoder_func : callable
        Function to train autoencoder: train_autoencoder(encoder, decoder, X, epochs, ...)
        
    Returns
    -------
    factors : np.ndarray
        Final extracted factors (T x num_factors)
    prediction_iter : np.ndarray
        Final reconstruction (T x N)
    converged : bool
        Whether convergence was achieved
    num_iter : int
        Number of iterations completed
    """
    T, N = x_clean.shape
    bool_no_miss = ~missing_mask
    
    # Initialize data structures (matching original TensorFlow code)
    data_mod_only_miss = x_standardized.copy()  # Original with missing values
    data_mod = x_clean.copy()  # Clean data (will be modified during MCMC)
    
    # Initial prediction
    x_tensor = torch.FloatTensor(data_mod).to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        factors_init = encoder(x_tensor).cpu().numpy()
        prediction_iter = decoder(torch.FloatTensor(factors_init).to(device)).cpu().numpy()
    
    # Update missing values with initial prediction
    bool_miss = missing_mask
    if bool_miss.any():
        data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
    
    # Initial residuals
    eps = data_mod_only_miss - prediction_iter
    
    # MCMC loop
    iter_count = 0
    not_converged = True
    prediction_prev_iter = None
    
    _logger.info(f"Starting MCMC training: max_iter={max_iter}, tolerance={tolerance}")
    
    while not_converged and iter_count < max_iter:
        iter_count += 1
        
        # Get idio distribution
        if use_idiosyncratic:
            phi, mu_eps, std_eps = get_idio_func(eps, bool_no_miss, min_obs_idio)
        else:
            # No idio modeling: use zero mean, small std
            phi = np.zeros((N, N))
            mu_eps = np.zeros(N)
            std_eps = np.ones(N) * 1e-8
        
        # Subtract conditional AR-idio mean from x (matching original code)
        if use_idiosyncratic and eps.shape[0] > 1:
            # data_mod[lags_input + 1:] = data_mod_only_miss[lags_input + 1:] - eps[:-1, :] @ phi
            data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
            # For first observations set to 0 the idio
            data_mod[:1] = data_mod_only_miss[:1]
        else:
            data_mod = data_mod_only_miss.copy()
        
        # Generate MC samples for idio (dims = epochs x T x N)
        # Original: eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (epochs, T))
        # But multivariate_normal expects 2D, so we sample per time step
        eps_draws = np.zeros((epochs_per_iter, T, N))
        for t in range(T):
            # Sample for each time step independently
            eps_draws[:, t, :] = rng.multivariate_normal(
                mu_eps, np.diag(std_eps), size=epochs_per_iter
            )
        
        # Initialize noisy inputs (dims = epochs x T x N)
        x_sim_den = np.zeros((epochs_per_iter, T, N))
        
        # Loop over MC samples (matching original code)
        factors_samples = []
        for i in range(epochs_per_iter):
            x_sim_den[i, :, :] = data_mod.copy()
            # Corrupt input data, only current observations
            x_sim_den[i, :, :] = x_sim_den[i, :, :] - eps_draws[i, :, :]
            
            # Train autoencoder on corrupted sample (1 epoch)
            train_autoencoder_func(
                encoder, decoder, x_sim_den[i, :, :],
                epochs=1, batch_size=batch_size,
                learning_rate=learning_rate, device=device,
                verbose=False
            )
            
            # Extract factors from this sample
            x_sample_tensor = torch.FloatTensor(x_sim_den[i, :, :]).to(device)
            encoder.eval()
            with torch.no_grad():
                factors_sample = encoder(x_sample_tensor).cpu().numpy()
            factors_samples.append(factors_sample)
        
        # Update factors: average over all MC samples
        factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
        
        # Check convergence
        decoder.eval()
        with torch.no_grad():
            factors_tensor = torch.FloatTensor(factors).to(device)
            prediction_iter = decoder(factors_tensor).cpu().numpy()
        
        if iter_count > 1:
            delta, loss_now = convergence_checker_func(
                prediction_prev_iter, prediction_iter, data_mod_only_miss
            )
            
            if iter_count % disp == 0:
                _logger.info(
                    f"Iteration {iter_count}/{max_iter}: loss={loss_now:.6f}, delta={delta:.6f}"
                )
            
            if delta < tolerance:
                not_converged = False
                _logger.info(
                    f"Convergence achieved in {iter_count} iterations: "
                    f"loss={loss_now:.6f}, delta={delta:.6f} < {tolerance}"
                )
        
        # Store previous prediction for convergence checking
        prediction_prev_iter = prediction_iter.copy()
        
        # Update missing values
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Update residuals
        eps = data_mod_only_miss - prediction_iter
    
    if not_converged:
        _logger.warning(
            f"Convergence not achieved within {max_iter} iterations. "
            f"Final delta: {delta if iter_count > 1 else 'N/A'}"
        )
    
    converged = not not_converged
    return factors, prediction_iter, converged, iter_count



"""MCMC training procedure for Deep Dynamic Factor Model (DDFM).

This module contains the Markov Chain Monte Carlo (MCMC) training procedure
for DDFM, which alternates between estimating idiosyncratic dynamics,
generating Monte Carlo samples, and training the autoencoder.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
import torch

from ..logger import get_logger
from ..utils.statespace import estimate_idio_dynamics

if TYPE_CHECKING:
    from .ddfm import DDFM, DDFMTrainingState

_logger = get_logger(__name__)


class DDFMMCMCTrainer:
    """MCMC training procedure for DDFM.
    
    This class encapsulates the MCMC iterative training procedure for DDFM,
    which alternates between:
    1. Estimating idiosyncratic dynamics (AR parameters)
    2. Generating Monte Carlo samples from the state-space model
    3. Training the autoencoder (encoder/decoder) on MC samples
    4. Extracting factors from trained encoder
    5. Checking convergence based on MSE between predictions and data
    
    The procedure continues until convergence (MSE change < tolerance) or
    maximum iterations reached.
    """
    
    def __init__(self, model: 'DDFM'):
        """Initialize MCMC trainer with DDFM model instance.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance to train. The trainer will access model's
            encoder, decoder, and other attributes/methods.
        """
        self.model = model
    
    def fit(
        self,
        X: torch.Tensor,
        x_clean: torch.Tensor,
        missing_mask: np.ndarray,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
        disp: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> 'DDFMTrainingState':
        """Run MCMC iterative training procedure for DDFM.
        
        This method implements the Markov Chain Monte Carlo (MCMC) procedure for
        Deep Dynamic Factor Model training. It alternates between:
        1. Estimating idiosyncratic dynamics (AR parameters)
        2. Generating Monte Carlo samples from the state-space model
        3. Training the autoencoder (encoder/decoder) on MC samples
        4. Extracting factors from trained encoder
        5. Checking convergence based on MSE between predictions and data
        
        The procedure continues until convergence (MSE change < tolerance) or
        maximum iterations reached.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data with missing values, shape (T x N), where T is number
            of time periods and N is number of series. Missing values should be NaN
            or handled via missing_mask.
        x_clean : torch.Tensor
            Clean data (interpolated), shape (T x N), used for initial autoencoder
            training. Should have same shape as X.
        missing_mask : np.ndarray
            Missing data mask, shape (T x N), boolean array where True indicates
            missing data. Must match shape of X.
        Mx : np.ndarray, optional
            Mean values for unstandardization, shape (N,). Used to convert standardized
            predictions back to original scale. If None, uses values from data module.
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization, shape (N,). Used to convert
            standardized predictions back to original scale. If None, uses values from
            data module.
        max_iter : int, optional
            Maximum number of MCMC iterations. If None, uses model.max_iter (default: 200).
        tolerance : float, optional
            Convergence tolerance for MSE change between iterations. If None, uses
            model.tolerance (default: 0.0005). Training stops when |MSE_new - MSE_old| < tolerance.
        disp : int, optional
            Display progress every 'disp' iterations. If None, uses model.disp (default: 10).
            Set to 0 to disable progress output.
        seed : int, optional
            Random seed for reproducibility. If None, uses current random state.
            Sets both NumPy and PyTorch random seeds.
            
        Returns
        -------
        DDFMTrainingState
            Final training state containing:
            - factors: np.ndarray, shape (T x num_factors) - extracted factors
            - prediction: np.ndarray, shape (T x N) - final predictions
            - converged: bool - whether convergence was achieved
            - num_iter: int - number of iterations completed
            - training_loss: float - final training loss (MSE)
            
        Raises
        ------
        ValueError
            If X and x_clean have mismatched shapes.
            If missing_mask shape doesn't match X shape.
            If numerical issues occur (NaN/Inf propagation) that cannot be handled.
            
        Notes
        -----
        The MCMC procedure includes comprehensive numerical stability checks:
        
        1. **Shape validation**: Validates X, x_clean, and missing_mask have consistent shapes.
        
        2. **Numerical stability**: 
           - Checks for NaN/Inf after each major step
           - Uses fallback mechanisms when numerical issues occur
           - Logs warnings when fallbacks are used
        
        3. **Convergence checking**:
           - Computes MSE between predictions and observed data (excluding missing values)
           - Checks for convergence: |MSE_new - MSE_old| < tolerance
           - Stops early if convergence achieved
        
        4. **Error handling**:
           - VAR estimation failures: Uses fallback matrices
           - Factor extraction failures: Uses previous iteration factors
           - Prediction failures: Uses previous iteration predictions
        
        The method is designed to be robust to numerical edge cases and provides
        graceful degradation when estimation fails.
        
        Examples
        --------
        >>> from dfm_python.models.ddfm import DDFM
        >>> from dfm_python.models.mcmc import DDFMMCMCTrainer
        >>> 
        >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config.yaml')
        >>> # X, x_clean, missing_mask prepared from data module
        >>> trainer = DDFMMCMCTrainer(model)
        >>> state = trainer.fit(X, x_clean, missing_mask, max_iter=50, tolerance=1e-4)
        >>> factors = state.factors  # (T x 2) factor estimates
        >>> print(f"Converged: {state.converged}, Iterations: {state.num_iter}")
        """
        # Store standardization parameters in model
        self.model.Mx = Mx
        self.model.Wx = Wx
        self.model.data_processed = X
        
        device = X.device
        dtype = X.dtype
        T, N = X.shape
        
        # Validate shape consistency between X, x_clean, and missing_mask
        if x_clean.shape != X.shape:
            raise ValueError(
                f"{self.model.__class__.__name__} fit_mcmc failed: shape mismatch between X ({X.shape}) and x_clean ({x_clean.shape}). "
                f"Both X and x_clean must have the same shape (T x N). "
                f"This indicates data preprocessing inconsistency. "
                f"Please ensure x_clean is created from the same data as X."
            )
        if missing_mask.shape != (T, N):
            raise ValueError(
                f"{self.model.__class__.__name__} fit_mcmc failed: shape mismatch between X ({X.shape}) and missing_mask ({missing_mask.shape}). "
                f"missing_mask must have shape (T x N) matching X. "
                f"This indicates missing_mask was created from data with different shape. "
                f"Please ensure missing_mask is created from the same data passed as X parameter."
            )
        
        # Use instance attributes if not provided
        max_iter = max_iter if max_iter is not None else self.model.max_iter
        tolerance = tolerance if tolerance is not None else self.model.tolerance
        disp = disp if disp is not None else self.model.disp
        
        # Initialize networks if not done
        if self.model.encoder is None or self.model.decoder is None:
            self.model.initialize_networks(N)
        
        # Ensure encoder and decoder are on the correct device
        # Ensure encoder/decoder are initialized
        # as the device might differ or they might not have been moved properly
        self.model.encoder = self.model.encoder.to(device)
        self.model.decoder = self.model.decoder.to(device)
        
        # Random number generator for MC sampling
        rng = np.random.RandomState(seed if seed is not None else (self.model.rng.randint(0, 2**31) if hasattr(self.model.rng, 'randint') else 3))
        
        # Convert to numpy for MCMC procedure (some operations are easier in numpy)
        x_standardized_np = X.cpu().numpy()
        x_clean_np = x_clean.cpu().numpy()
        bool_no_miss = ~missing_mask
        
        # Initialize data structures
        data_mod_only_miss = x_standardized_np.copy()  # Original with missing values
        data_mod = x_clean_np.copy()  # Clean data (will be modified during MCMC)
        z_actual = x_standardized_np.copy()  # Actual observations (target for training, like original DDFM)
        
        # Initial prediction
        x_tensor = x_clean.to(device)
        self.model.encoder.eval()
        self.model.decoder.eval()
        with torch.no_grad():
            factors_init = self.model.encoder(x_tensor).cpu().numpy()
            # Check for NaN/Inf in initial factors
            factors_init = self.model._check_finite(factors_init, "initial factors", context="at iteration 0")
            
            factors_tensor = torch.tensor(factors_init, device=device, dtype=dtype)
            prediction_iter = self.model.decoder(factors_tensor).cpu().numpy()
            # Check for NaN/Inf in initial prediction
            prediction_iter = self.model._check_finite(prediction_iter, "initial prediction", context="at iteration 0")
        
        # Initialize factors
        factors = factors_init.copy()
        
        # Update missing values with initial prediction
        bool_miss = missing_mask
        if bool_miss.any():
            data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
        
        # Initial residuals
        eps = data_mod_only_miss - prediction_iter
        # Check for NaN/Inf in initial residuals
        eps = self.model._check_finite(eps, "initial residuals", context="at iteration 0")
        
        # MCMC loop
        iter_count = 0
        not_converged = True
        prediction_prev_iter = None
        delta = float('inf')
        loss_now = float('inf')
        
        # Check for very small dataset and warn about potential instability
        if T < 10:
            _logger.warning(
                f"{self.model.__class__.__name__} MCMC training: very small dataset (T={T} < 10) may cause unstable MCMC sampling. "
                f"With only {T} time periods, encoder/decoder training per iteration may have high variance. "
                f"Factor extraction and VAR estimation will use fallback strategies. Results may be less reliable. "
                f"Monitor convergence carefully. Consider reducing num_factors or using smaller encoder_layers for better stability"
            )
        
        _logger.info(f"Starting MCMC training: max_iter={max_iter}, tolerance={tolerance}, epochs_per_iter={self.model.epochs_per_iter}")
        
        # Create optimizer once per MCMC iteration (reused across MC samples)
        optimizer = None
        
        while not_converged and iter_count < max_iter:
            iter_count += 1
            self.model.mcmc_iteration = iter_count
            
            # Create optimizer for this MCMC iteration (reused across MC samples)
            optimizer = self.model._create_optimizer()
            
            # Get idiosyncratic distribution
            if self.model.use_idiosyncratic:
                A_eps, Q_eps = estimate_idio_dynamics(eps, missing_mask, self.model.min_obs_idio)
                # Check for NaN/Inf in estimated dynamics
                A_eps = self.model._check_finite(A_eps, f"idiosyncratic AR coefficients (A_eps)", context=f"at iteration {iter_count}")
                Q_eps = self.model._check_finite(Q_eps, f"idiosyncratic innovation covariance (Q_eps)", context=f"at iteration {iter_count}")
                
                # Convert to format expected by MCMC procedure
                phi = A_eps if A_eps.ndim == 2 else np.diag(A_eps) if A_eps.ndim == 1 else np.eye(N)
                mu_eps = np.zeros(N)
                if Q_eps.ndim == 2:
                    std_eps = np.sqrt(np.diag(Q_eps))
                elif Q_eps.ndim == 1:
                    std_eps = np.sqrt(Q_eps)
                else:
                    std_eps = np.ones(N) * 0.1
                
                # Ensure std_eps is finite and positive
                std_eps = np.maximum(std_eps, 1e-8)  # Floor to prevent zero/negative
                std_eps = self.model._check_finite(std_eps, f"idiosyncratic std (std_eps)", context=f"at iteration {iter_count}")
            else:
                phi = np.zeros((N, N))
                mu_eps = np.zeros(N)
                std_eps = np.ones(N) * 1e-8
            
            # Subtract conditional AR-idio mean from x
            if self.model.use_idiosyncratic and eps.shape[0] > 1:
                data_mod[1:] = data_mod_only_miss[1:] - eps[:-1, :] @ phi
                data_mod[:1] = data_mod_only_miss[:1]
            else:
                data_mod = data_mod_only_miss.copy()
            
            # Generate MC samples for idio (dims = epochs_per_iter x T x N)
            eps_draws = np.zeros((self.model.epochs_per_iter, T, N))
            try:
                for t in range(T):
                    eps_draws[:, t, :] = rng.multivariate_normal(
                        mu_eps, np.diag(std_eps), size=self.model.epochs_per_iter
                    )
                # Check for NaN/Inf in MC samples
                eps_draws = self.model._check_finite(eps_draws, f"MC samples (eps_draws)", context=f"at iteration {iter_count}")
            except (ValueError, np.linalg.LinAlgError) as e:
                _logger.warning(
                    f"{self.model.__class__.__name__} MCMC iteration {iter_count}: failed to generate MC samples: {e}. "
                    f"Using zero samples as fallback"
                )
                # Use zero samples as fallback
                eps_draws = np.zeros((self.model.epochs_per_iter, T, N))
            
            # Initialize noisy inputs
            x_sim_den = np.zeros((self.model.epochs_per_iter, T, N))
            
            # Loop over MC samples
            factors_samples = []
            for i in range(self.model.epochs_per_iter):
                x_sim_den[i, :, :] = data_mod.copy()
                # Corrupt input data by subtracting sampled idio innovations
                x_sim_den[i, :, :] = x_sim_den[i, :, :] - eps_draws[i, :, :]
                
                # Train autoencoder on corrupted sample (1 epoch)
                # Convert to torch and create dataset
                x_sample = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                z_actual_tensor = torch.tensor(z_actual, device=device, dtype=dtype)
                dataset = torch.utils.data.TensorDataset(x_sample, z_actual_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.model.batch_size, shuffle=True
                )
                
                # Train for 1 epoch
                self.model.encoder.train()
                self.model.decoder.train()
                # Optimizer is created once per MCMC iteration and reused across MC samples
                
                for batch_data, batch_target in dataloader:
                    optimizer.zero_grad()
                    reconstructed = self.model.forward(batch_data)
                    # Use missing-aware loss (mse_missing) for consistency
                    # Create mask for missing values (though in MCMC loop, missing values are filled)
                    mask = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), torch.ones_like(batch_target))
                    target_clean = torch.where(torch.isnan(batch_target), torch.zeros_like(batch_target), batch_target)
                    reconstructed_masked = reconstructed * mask
                    squared_diff = (target_clean - reconstructed_masked) ** 2
                    loss = torch.sum(squared_diff) / (torch.sum(mask) + 1e-8)
                    loss.backward()
                    # Gradient clipping to prevent NaN and improve stability
                    if self.model.grad_clip_val > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
                            max_norm=self.model.grad_clip_val
                        )
                    optimizer.step()
                # Extract factors from this sample
                x_sample_tensor = torch.tensor(x_sim_den[i, :, :], device=device, dtype=dtype)
                self.model.encoder.eval()
                with torch.no_grad():
                    factors_sample = self.model.encoder(x_sample_tensor).cpu().numpy()
                    # Check for NaN/Inf in factor sample
                    factors_sample = self.model._check_finite(
                        factors_sample, 
                        f"factor sample {i+1}/{self.model.epochs_per_iter}", 
                        context=f"at iteration {iter_count}"
                    )
                factors_samples.append(factors_sample)
            
            # Update factors: average over all MC samples
            factors = np.mean(np.array(factors_samples), axis=0)  # T x num_factors
            # Check for NaN/Inf in averaged factors
            factors = self.model._check_finite(factors, "averaged factors", context=f"at iteration {iter_count}", fallback=factors_init)
            
            # Clip extreme factor values to prevent numerical instability
            # Use configurable clipping threshold (default: 10 standard deviations)
            clip_threshold = 10.0
            factor_mean = np.mean(factors, axis=0)
            factor_std = np.std(factors, axis=0)
            # Avoid division by zero
            factor_std = np.maximum(factor_std, 1e-8)
            
            clipped_count = 0
            for i in range(factors.shape[1]):
                lower_bound = factor_mean[i] - clip_threshold * factor_std[i]
                upper_bound = factor_mean[i] + clip_threshold * factor_std[i]
                before_clip = factors[:, i].copy()
                factors[:, i] = np.clip(factors[:, i], lower_bound, upper_bound)
                # Count how many values were clipped
                clipped_count += np.sum((before_clip != factors[:, i]))
            
            if clipped_count > 0:
                _logger.warning(
                    f"{self.model.__class__.__name__} MCMC iteration {iter_count}: clipped {clipped_count} extreme factor values (>{clip_threshold} std devs). "
                    f"This prevents numerical instability in encoder/decoder forward passes. "
                    f"If clipping occurs frequently, consider: (1) Reducing learning_rate, (2) Using smaller encoder_layers, (3) Checking data scaling"
                )
            
            # Check convergence
            self.model.decoder.eval()
            with torch.no_grad():
                factors_tensor = torch.tensor(factors, device=device, dtype=dtype)
                prediction_iter = self.model.decoder(factors_tensor).cpu().numpy()
                # Check for NaN/Inf in prediction
                prediction_iter = self.model._check_finite(
                    prediction_iter, 
                    "prediction_iter", 
                    context=f"at iteration {iter_count}",
                    fallback=prediction_prev_iter if prediction_prev_iter is not None else prediction_iter
                )
            
            if iter_count > 1:
                # Compute MSE on non-missing values
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    mse = np.nanmean((prediction_prev_iter[mask] - prediction_iter[mask]) ** 2)
                    # Ensure MSE is finite
                    if not np.isfinite(mse):
                        _logger.warning(
                            f"{self.model.__class__.__name__} MCMC iteration {iter_count}: MSE is not finite ({mse}). "
                            f"Using previous delta value"
                        )
                        mse = delta if np.isfinite(delta) else tolerance * 10
                    delta = mse
                    loss_now = mse
                else:
                    delta = float('inf')
                    loss_now = float('inf')
                
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
            else:
                # First iteration: compute initial loss
                mask = ~np.isnan(data_mod_only_miss)
                if np.sum(mask) > 0:
                    loss_now = np.nanmean((data_mod_only_miss[mask] - prediction_iter[mask]) ** 2)
                    # Ensure loss is finite
                    if not np.isfinite(loss_now):
                        _logger.warning(
                            f"{self.model.__class__.__name__} MCMC iteration {iter_count}: initial loss is not finite ({loss_now}). "
                            f"Using large default value"
                        )
                        loss_now = 1e6
                else:
                    loss_now = float('inf')
            
            # Store previous prediction for convergence checking
            prediction_prev_iter = prediction_iter.copy()
            
            # Update missing values with current prediction
            if bool_miss.any():
                data_mod_only_miss[bool_miss] = prediction_iter[bool_miss]
            
            # Update residuals
            eps = data_mod_only_miss - prediction_iter
            # Check for NaN/Inf in residuals before next iteration
            eps = self.model._check_finite(eps, "residuals (eps)", context=f"at iteration {iter_count}")
        
        if not_converged:
            delta_str = f"{delta:.6f}" if iter_count > 1 else "N/A"
            _logger.warning(
                f"{self.model.__class__.__name__} MCMC training: convergence not achieved within {max_iter} iterations. "
                f"Final delta: {delta_str}"
            )
        
        converged = not not_converged
        
        # Validate and normalize factors shape before storing
        factors = self.model._validate_factors(factors, operation="fit_mcmc")
        
        # Import DDFMTrainingState here to avoid circular dependency
        from .ddfm import DDFMTrainingState
        
        # Store final state
        self.model.training_state = DDFMTrainingState(
            factors=factors,
            prediction=prediction_iter,
            converged=converged,
            num_iter=iter_count,
            training_loss=loss_now
        )
        
        return self.model.training_state


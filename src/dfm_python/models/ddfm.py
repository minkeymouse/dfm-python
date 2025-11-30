"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
import logging
from ..core.helpers import get_logger

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _has_torch = True
except ImportError:
    _has_torch = False
    torch = None
    nn = None
    optim = None

from .base import BaseFactorModel
from ..config import DFMConfig, DEFAULT_GLOBAL_BLOCK_NAME
from ..core.results import DFMResult
from ..core.state_space import run_kf
from ..dataloader.loader import rem_nans_spline
from ..core.helpers import (
    safe_get_attr,
    get_clock_frequency,
    resolve_param,
    standardize_data,
)
from ..core.structure import get_periods_per_year
from ..models.ddfm_utils import (
    train_autoencoder,
    estimate_var1,
    estimate_var2,
    estimate_idiosyncratic_dynamics,
    build_observation_matrix,
    build_state_space,
    extract_decoder_params,
    fit_ddfm_mcmc,
)
from .ddfm_fit_mcmc import fit_ddfm_mcmc
# Training, state-space, and utility functions are now in this file (merged from submodules)

_logger = get_logger(__name__)


if _has_torch:
    class Encoder(nn.Module):
        """Nonlinear encoder network for DDFM.
        
        Maps observed variables X_t to latent factors f_t using a multi-layer perceptron.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            output_dim: int,
            activation: str = 'tanh',
            use_batch_norm: bool = True,
        ):
            """Initialize encoder network.
            
            Parameters
            ----------
            input_dim : int
                Number of input features (number of series)
            hidden_dims : List[int]
                List of hidden layer dimensions
            output_dim : int
                Number of factors (output dimension)
            activation : str
                Activation function ('tanh', 'relu', 'sigmoid')
            use_batch_norm : bool
                Whether to use batch normalization
            """
            super().__init__()
            
            self.layers = nn.ModuleList()
            self.use_batch_norm = use_batch_norm
            self.batch_norms = nn.ModuleList() if use_batch_norm else None
            
            # Activation function
            if activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Build layers
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer (no activation, linear)
            self.output_layer = nn.Linear(prev_dim, output_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through encoder.
            
            Parameters
            ----------
            x : torch.Tensor
                Input data (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Encoded factors (batch_size x output_dim)
            """
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = self.activation(x)
            
            # Output layer (linear, no activation)
            x = self.output_layer(x)
            return x
    
    
    class Decoder(nn.Module):
        """Linear decoder network for DDFM.
        
        Maps latent factors f_t back to observed variables X_t using a linear transformation.
        This preserves interpretability and allows Kalman filtering.
        """
        
        def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
            """Initialize linear decoder.
            
            Parameters
            ----------
            input_dim : int
                Number of factors (input dimension)
            output_dim : int
                Number of series (output dimension)
            use_bias : bool
                Whether to use bias term
            """
            super().__init__()
            self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        def forward(self, f: torch.Tensor) -> torch.Tensor:
            """Forward pass through decoder.
            
            Parameters
            ----------
            f : torch.Tensor
                Factors (batch_size x input_dim)
                
            Returns
            -------
            torch.Tensor
                Reconstructed observations (batch_size x output_dim)
            """
            return self.decoder(f)


class DDFM(BaseFactorModel):
    """Deep Dynamic Factor Model using PyTorch.
    
    This class implements a DDFM with:
    - Nonlinear encoder (MLP) to extract factors from observations
    - Linear decoder for interpretability
    - Linear factor dynamics (VAR)
    - Kalman filtering for final smoothing
    
    The model is trained using gradient descent (Adam optimizer) to minimize
    reconstruction error, then factor dynamics are estimated via OLS, and
    final smoothing is performed using Kalman filter.
    """
    
    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        activation: str = 'tanh',
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        lags_input: int = 0,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize DDFM model.
        
        Parameters
        ----------
        encoder_layers : List[int], optional
            Hidden layer dimensions for encoder. Default: [64, 32]
        num_factors : int, optional
            Number of factors. If None, will be inferred from config during fit.
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid'). Default: 'tanh'
        use_batch_norm : bool
            Whether to use batch normalization in encoder. Default: True
        learning_rate : float
            Learning rate for Adam optimizer. Default: 0.001
        epochs : int
            Number of epochs per MCMC iteration. Default: 100
        batch_size : int
            Batch size for training. Default: 32
        factor_order : int
            VAR lag order for factor dynamics (1 or 2). Default: 1
        use_idiosyncratic : bool
            Whether to model idiosyncratic components with AR(1) dynamics. Default: True
        min_obs_idio : int
            Minimum number of observations required for idio AR(1) estimation. Default: 5
        lags_input : int
            Number of lags of inputs on encoder (default 0, i.e. same inputs and outputs). Default: 0
        max_iter : int
            Maximum number of MCMC iterations. Default: 200
        tolerance : float
            Convergence tolerance. Default: 0.0005
        disp : int
            Display intermediate results every 'disp' iterations. Default: 10
        seed : int, optional
            Random seed for reproducibility. Default: None
        """
        super().__init__()
        
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for DDFM. Install with: pip install dfm-python[deep]"
            )
        
        if factor_order not in [1, 2]:
            raise ValueError(f"factor_order must be 1 or 2, got {factor_order}")
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.num_factors = num_factors
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs  # Epochs per MCMC iteration
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        self.lags_input = lags_input
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        
        # PyTorch modules (will be initialized in fit)
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random number generator for MC sampling
        self.rng = np.random.RandomState(seed if seed is not None else 3)
    
    def fit(self, X: np.ndarray, config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the DDFM model.
        
        Training process:
        1. Standardize data
        2. Train autoencoder (encoder + decoder) to minimize reconstruction error
        3. Extract factors using trained encoder
        4. Extract decoder parameters (C, bias) directly from trained decoder
        5. Compute residuals and estimate idiosyncratic AR(1) dynamics
        6. Estimate factor dynamics (VAR(1) or VAR(2)) via OLS
        7. Build complete state-space model (factor + idio)
        8. Apply Kalman smoothing for final state estimates
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N), where T is time periods and N is number of series.
        config : DFMConfig
            Configuration object. Used to determine number of factors if not specified.
        **kwargs
            Additional parameters:
            - epochs: Override default epochs
            - batch_size: Override default batch size
            - learning_rate: Override default learning rate
            
        Returns
        -------
        DFMResult
            Estimation results compatible with DFMResult structure.
        """
        if not _has_torch:
            raise ImportError("PyTorch is required for DDFM")
        
        # Store config and data
        self._config = config
        self._data = X
        
        # Override hyperparameters from kwargs
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        
        # Determine number of factors
        if self.num_factors is None:
            # Infer from config (sum of factors per block)
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                num_factors = int(np.sum(config.factors_per_block))
            else:
                # Default: use first block's factors or 1
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    num_factors = int(np.sum(blocks[:, 0]))  # First block
                else:
                    num_factors = 1
        else:
            num_factors = self.num_factors
        
        T, N = X.shape
        
        # Step 1: Standardize data
        clip_data = kwargs.get('clip_data_values', safe_get_attr(config, 'clip_data_values', True))
        clip_threshold = kwargs.get('data_clip_threshold', safe_get_attr(config, 'data_clip_threshold', 100.0))
        x_standardized, Mx, Wx = standardize_data(X, clip_data, clip_threshold)
        
        # Step 2: Handle missing data (simple interpolation for now)
        nan_method = kwargs.get('nan_method', safe_get_attr(config, 'nan_method', 2))
        nan_k = kwargs.get('nan_k', safe_get_attr(config, 'nan_k', 3))
        x_clean, _ = rem_nans_spline(x_standardized, method=nan_method, k=nan_k)
        
        # Step 3: Initialize encoder and decoder
        self.encoder = Encoder(
            input_dim=N,
            hidden_dims=self.encoder_layers,
            output_dim=num_factors,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device)
        
        self.decoder = Decoder(
            input_dim=num_factors,
            output_dim=N,
            use_bias=True,
        ).to(self.device)
        
        # Step 4: Optional pre-training (matching original TensorFlow implementation)
        pre_train_epochs = kwargs.get('pre_train_epochs', None)
        if pre_train_epochs is not None and pre_train_epochs > 0:
            _logger.info(f"Pre-training DDFM autoencoder: {pre_train_epochs} epochs")
            train_autoencoder(
                self.encoder,
                self.decoder,
                x_clean,
                pre_train_epochs,
                batch_size,
                learning_rate,
                self.device,
                verbose=True,
            )
        
        # Step 5: MCMC iterative training procedure
        missing_mask = np.isnan(x_standardized)
        _logger.info(f"Starting MCMC iterative training: epochs_per_iter={epochs}, max_iter={self.max_iter}")
        
        factors, prediction_iter, converged, num_iter = fit_ddfm_mcmc(
            encoder=self.encoder,
            decoder=self.decoder,
            x_standardized=x_standardized,
            x_clean=x_clean,
            missing_mask=missing_mask,
            epochs_per_iter=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            disp=self.disp,
            device=self.device,
            rng=self.rng,
            use_idiosyncratic=self.use_idiosyncratic,
            min_obs_idio=self.min_obs_idio,
            lags_input=self.lags_input,
        )
        
        # Step 6: Extract decoder parameters (C, bias)
        C, bias = extract_decoder_params(self.decoder)
        
        # Step 7: Compute residuals and estimate idiosyncratic dynamics for state-space
        if self.use_idiosyncratic:
            # Use final prediction from MCMC
            residuals = x_standardized - prediction_iter
            # Fill missing with prediction
            residuals_filled = residuals.copy()
            residuals_filled[missing_mask] = 0.0  # Missing residuals set to 0
            
            # Estimate idio AR(1) dynamics for state-space model
            A_eps, Q_eps = estimate_idiosyncratic_dynamics(
                residuals_filled, missing_mask, self.min_obs_idio
            )
        else:
            # No idio modeling: use diagonal R only
            A_eps = np.zeros((N, N))
            Q_eps = np.eye(N) * 1e-8
        
        # Step 8: Estimate factor dynamics (VAR(1) or VAR(2))
        if self.factor_order == 1:
            A_f, Q_f = estimate_var1(factors)
        elif self.factor_order == 2:
            A_f, Q_f = estimate_var2(factors)
        else:
            raise ValueError(f"factor_order must be 1 or 2, got {self.factor_order}")
        
        # Step 9: Build state-space model
        if self.use_idiosyncratic:
            A, Q, Z_0, V_0 = build_state_space(
                factors, A_f, Q_f, A_eps, Q_eps, self.factor_order
            )
            
            # Build observation matrix H = [C, I] or [C, 0, I]
            H = build_observation_matrix(C, self.factor_order, N)
            
            # Observation noise (small, mainly for numerical stability)
            R = np.eye(N) * 1e-15
        else:
            # Simplified: factor-only state-space
            A = A_f
            Q = Q_f
            Z_0 = factors[0, :]
            V_0 = np.cov(factors.T)
            H = C
            # Estimate R from residuals
            residuals = x_clean - factors @ C.T
            R_diag = np.var(residuals, axis=0)
            R = np.diag(np.maximum(R_diag, 1e-8))
        
        # Step 10: Kalman smoothing
        y = x_standardized.T  # N x T (with missing data)
        zsmooth, _, _, loglik = run_kf(y, A, H, Q, R, Z_0, V_0)
        Zsmooth = zsmooth.T  # (T+1) x state_dim
        
        # Step 11: Extract factors from smoothed state
        if self.use_idiosyncratic:
            if self.factor_order == 1:
                # State: [f_t, eps_t], extract f_t
                Z = Zsmooth[1:, :num_factors]  # T x m
            else:  # VAR(2)
                # State: [f_t, f_{t-1}, eps_t], extract f_t
                Z = Zsmooth[1:, :num_factors]  # T x m
        else:
            Z = Zsmooth[1:, :]  # T x m
        
        # Step 12: Compute smoothed data
        if self.use_idiosyncratic:
            # Use full state: y_t = C @ f_t + eps_t
            # Extract both factors and idio from smoothed state
            if self.factor_order == 1:
                factors_smooth = Zsmooth[1:, :num_factors]  # T x m
                idio_smooth = Zsmooth[1:, num_factors:]  # T x N
            else:  # VAR(2)
                factors_smooth = Zsmooth[1:, :num_factors]  # T x m
                idio_smooth = Zsmooth[1:, 2*num_factors:]  # T x N
            
            x_sm = factors_smooth @ C.T + idio_smooth  # T x N (standardized)
        else:
            x_sm = Z @ C.T  # T x N (standardized)
        
        Wx_clean = np.where(np.isnan(Wx), 1.0, Wx)
        Mx_clean = np.where(np.isnan(Mx), 0.0, Mx)
        X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized)
        
        # Step 13: Create DFMResult-compatible result
        # Store factor dynamics (A_f) and observation matrix (C) for compatibility
        # Note: A and Q in result represent factor-only dynamics for compatibility
        r = np.array([num_factors])  # Single block
        p = self.factor_order
        
        result = DFMResult(
            x_sm=x_sm,
            X_sm=X_sm,
            Z=Z,  # T x m (factors only)
            C=C,
            R=R,
            A=A_f,  # Factor dynamics only (for compatibility)
            Q=Q_f,  # Factor innovation only (for compatibility)
            Mx=Mx,
            Wx=Wx,
            Z_0=Z_0[:num_factors] if self.use_idiosyncratic else Z_0,  # Factor initial state
            V_0=V_0[:num_factors, :num_factors] if self.use_idiosyncratic else V_0,  # Factor initial covariance
            r=r,
            p=p,
            converged=converged,  # MCMC convergence status
            num_iter=num_iter,  # Number of MCMC iterations
            loglik=loglik,
            series_ids=safe_get_attr(config, 'get_series_ids', lambda: [])(),
            block_names=[DEFAULT_GLOBAL_BLOCK_NAME],
        )
        
        self._result = result
        return result
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, default True
            Whether to return forecasted series.
        return_factors : bool, default True
            Whether to return forecasted factors.
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted series and/or factors
        """
        if self._result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Default horizon
        if horizon is None:
            if self._config is not None:
                clock = get_clock_frequency(self._config, 'm')
                horizon = get_periods_per_year(clock)
            else:
                horizon = 12  # Default to 12 periods if no config
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        # Extract parameters
        A = self._result.A  # Factor dynamics (m x m) for VAR(1) or (m x 2m) for VAR(2)
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]  # Last factor estimate (m,)
        p = self._result.p  # VAR order
        
        # Deterministic forecast
        if p == 1:
            # VAR(1): f_t = A @ f_{t-1}
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            Z_forecast[0, :] = A @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        elif p == 2:
            # VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            # Need last two factor values
            if self._result.Z.shape[0] < 2:
                # Fallback to VAR(1) if not enough history
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                A1 = A[:, :Z_last.shape[0]]
                Z_forecast[0, :] = A1 @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
            else:
                Z_prev = self._result.Z[-2, :]  # f_{t-2}
                A1 = A[:, :Z_last.shape[0]]
                A2 = A[:, Z_last.shape[0]:]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                if horizon > 1:
                    Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                for h in range(2, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
        else:
            raise ValueError(f"Unsupported VAR order: {p}")
        
        # Transform to observations
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast

if not _has_torch:
    # Placeholder when PyTorch is not available
    class DDFM(BaseFactorModel):
        """Placeholder DDFM class when PyTorch is not available."""
        
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError(
                "PyTorch is required for DDFM. Install with: pip install dfm-python[deep]"
            )
        
        def fit(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM")
        
        def predict(self, horizon: Optional[int] = None, *, return_series: bool = True, return_factors: bool = True):
            raise ImportError("PyTorch is required for DDFM")


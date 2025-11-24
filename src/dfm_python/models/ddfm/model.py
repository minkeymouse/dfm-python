"""Deep Dynamic Factor Model (DDFM) using PyTorch.

This module implements a PyTorch-based Deep Dynamic Factor Model that uses
a nonlinear encoder (autoencoder) to extract factors, while maintaining
linear dynamics and decoder for interpretability and compatibility with
Kalman filtering.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
import logging

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

from ..base import BaseFactorModel
from ...config import DFMConfig
from ...dfm import DFMResult
from ...engine.kalman import run_kf
from ...data import rem_nans_spline  # rem_nans_spline stays in data.py (not engine)
from ...engine.helpers import (
    safe_get_attr,
    get_clock_frequency,
    resolve_param,
    standardize_data,
)
from ...engine.utils import get_periods_per_year

_logger = logging.getLogger(__name__)


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
            Number of training epochs. Default: 100
        batch_size : int
            Batch size for training. Default: 32
        """
        super().__init__()
        
        if not _has_torch:
            raise ImportError(
                "PyTorch is required for DDFM. Install with: pip install dfm-python[deep]"
            )
        
        self.encoder_layers = encoder_layers or [64, 32]
        self.num_factors = num_factors
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # PyTorch modules (will be initialized in fit)
        self.encoder: Optional[Encoder] = None
        self.decoder: Optional[Decoder] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, X: np.ndarray, config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the DDFM model.
        
        Training process:
        1. Standardize data
        2. Train autoencoder (encoder + decoder) to minimize reconstruction error
        3. Extract factors using trained encoder
        4. Estimate factor dynamics via OLS
        5. Estimate observation equation parameters (C, R)
        6. Apply Kalman smoothing for final factor estimates
        
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
        
        # Step 4: Train autoencoder
        _logger.info(f"Training DDFM autoencoder: {epochs} epochs, batch_size={batch_size}")
        self._train_autoencoder(x_clean, epochs, batch_size, learning_rate)
        
        # Step 5: Extract factors
        x_tensor = torch.FloatTensor(x_clean).to(self.device)
        with torch.no_grad():
            factors = self.encoder(x_tensor).cpu().numpy()  # T x num_factors
        
        # Step 6: Estimate factor dynamics (VAR(1) via OLS)
        A, Q = self._estimate_factor_dynamics(factors)
        
        # Step 7: Estimate observation equation (C, R)
        C, R = self._estimate_observation_equation(x_clean, factors)
        
        # Step 8: Initial conditions
        Z_0 = factors[0, :]
        V_0 = np.cov(factors.T)  # Initial covariance from factor sample covariance
        
        # Step 9: Kalman smoothing
        y = x_standardized.T  # N x T (with missing data)
        zsmooth, _, _, loglik = run_kf(y, A, C, Q, R, Z_0, V_0)
        Zsmooth = zsmooth.T  # (T+1) x m
        
        # Step 10: Compute smoothed data
        x_sm = Zsmooth[1:, :] @ C.T  # T x N (standardized)
        Wx_clean = np.where(np.isnan(Wx), 1.0, Wx)
        Mx_clean = np.where(np.isnan(Mx), 0.0, Mx)
        X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized)
        
        # Create DFMResult-compatible result
        # For DDFM, we use simplified structure (no block structure, no idio chains)
        r = np.array([num_factors])  # Single block
        p = 1  # VAR(1)
        
        result = DFMResult(
            x_sm=x_sm,
            X_sm=X_sm,
            Z=Zsmooth[1:, :],  # T x m
            C=C,
            R=R,
            A=A,
            Q=Q,
            Mx=Mx,
            Wx=Wx,
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=p,
            converged=True,  # Training completed
            num_iter=epochs,
            loglik=loglik,
            series_ids=safe_get_attr(config, 'get_series_ids', lambda: [])(),
            block_names=['Block_Global'],
        )
        
        self._result = result
        return result
    
    def _train_autoencoder(
        self,
        X: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        """Train the autoencoder using PyTorch.
        
        Parameters
        ----------
        X : np.ndarray
            Standardized data (T x N)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate for Adam optimizer
        """
        T, N = X.shape
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )
        
        # Loss function (MSE)
        criterion = nn.MSELoss()
        
        # Training loop
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_target in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                factors = self.encoder(batch_X)
                reconstructed = self.decoder(factors)
                
                # Compute loss (only on non-missing values)
                loss = criterion(reconstructed, batch_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
                _logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
        
        self.encoder.eval()
        self.decoder.eval()
    
    def _estimate_factor_dynamics(
        self,
        factors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate factor dynamics via OLS.
        
        Estimates VAR(1) model: f_t = A @ f_{t-1} + u_t
        
        Parameters
        ----------
        factors : np.ndarray
            Extracted factors (T x m)
            
        Returns
        -------
        A : np.ndarray
            Transition matrix (m x m)
        Q : np.ndarray
            Innovation covariance (m x m)
        """
        T, m = factors.shape
        
        if T < 2:
            # Not enough data, use identity
            A = np.eye(m)
            Q = np.eye(m) * 0.1
            return A, Q
        
        # Prepare data for OLS: f_t = A @ f_{t-1}
        Y = factors[1:, :]  # T-1 x m (dependent)
        X = factors[:-1, :]  # T-1 x m (independent)
        
        # OLS: A = (X'X)^{-1} X'Y
        try:
            A = np.linalg.solve(X.T @ X + np.eye(m) * 1e-6, X.T @ Y).T
        except np.linalg.LinAlgError:
            # Fallback to pinv
            A = np.linalg.pinv(X) @ Y
        
        # Ensure stability: clip eigenvalues
        eigenvals = np.linalg.eigvals(A)
        max_eigenval = np.max(np.abs(eigenvals))
        if max_eigenval >= 0.99:
            A = A * (0.99 / max_eigenval)
        
        # Estimate innovation covariance
        residuals = Y - X @ A.T
        Q = np.cov(residuals.T)
        
        # Ensure Q is positive definite
        Q = (Q + Q.T) / 2  # Symmetrize
        eigenvals_Q = np.linalg.eigvals(Q)
        min_eigenval = np.min(eigenvals_Q)
        if min_eigenval < 1e-8:
            Q = Q + np.eye(m) * (1e-8 - min_eigenval)
        
        # Floor for Q (similar to linear DFM)
        Q = np.maximum(Q, np.eye(m) * 0.01)
        
        return A, Q
    
    def _estimate_observation_equation(
        self,
        X: np.ndarray,
        factors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate observation equation parameters.
        
        Estimates: x_t = C @ f_t + e_t
        
        Parameters
        ----------
        X : np.ndarray
            Standardized observations (T x N)
        factors : np.ndarray
            Extracted factors (T x m)
            
        Returns
        -------
        C : np.ndarray
            Loading matrix (N x m)
        R : np.ndarray
            Observation covariance (N x N), diagonal
        """
        T, N = X.shape
        T_f, m = factors.shape
        
        # Use minimum T
        T_min = min(T, T_f)
        X_use = X[:T_min, :]
        F_use = factors[:T_min, :]
        
        # OLS: C = (F'F)^{-1} F'X
        try:
            C = np.linalg.solve(F_use.T @ F_use + np.eye(m) * 1e-6, F_use.T @ X_use).T
        except np.linalg.LinAlgError:
            C = np.linalg.pinv(F_use) @ X_use
        
        # Normalize C (similar to linear DFM)
        C_norms = np.linalg.norm(C, axis=0)
        C_norms = np.where(C_norms < 1e-10, 1.0, C_norms)
        C = C / C_norms
        
        # Estimate observation covariance (diagonal)
        residuals = X_use - F_use @ C.T
        R_diag = np.var(residuals, axis=0)
        R = np.diag(np.maximum(R_diag, 1e-8))  # Floor
        
        return C, R
    
    def predict(self, horizon: Optional[int] = None, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted series and/or factors
        """
        if self._result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return_series = kwargs.get('return_series', True)
        return_factors = kwargs.get('return_factors', True)
        
        # Default horizon
        if horizon is None:
            clock = get_clock_frequency(self._config, 'm')
            horizon = get_periods_per_year(clock)
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        # Extract parameters
        A = self._result.A
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]
        
        # Deterministic forecast
        Z_forecast = np.zeros((horizon, Z_last.shape[0]))
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Transform to observations
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast

else:
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
        
        def predict(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM")


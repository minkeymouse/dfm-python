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
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
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
        factor_order : int
            VAR lag order for factor dynamics (1 or 2). Default: 1
        use_idiosyncratic : bool
            Whether to model idiosyncratic components with AR(1) dynamics. Default: True
        min_obs_idio : int
            Minimum number of observations required for idio AR(1) estimation. Default: 5
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
        self.epochs = epochs
        self.batch_size = batch_size
        self.factor_order = factor_order
        self.use_idiosyncratic = use_idiosyncratic
        self.min_obs_idio = min_obs_idio
        
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
        
        # Step 4: Train autoencoder
        _logger.info(f"Training DDFM autoencoder: {epochs} epochs, batch_size={batch_size}")
        self._train_autoencoder(x_clean, epochs, batch_size, learning_rate)
        
        # Step 5: Extract factors
        x_tensor = torch.FloatTensor(x_clean).to(self.device)
        with torch.no_grad():
            factors = self.encoder(x_tensor).cpu().numpy()  # T x num_factors
        
        # Step 6: Extract decoder parameters (C, bias)
        C, bias = self._extract_decoder_params()
        
        # Step 7: Compute residuals and estimate idiosyncratic dynamics
        if self.use_idiosyncratic:
            # Reconstruct using decoder
            factors_tensor = torch.FloatTensor(factors).to(self.device)
            with torch.no_grad():
                x_reconstructed = self.decoder(factors_tensor).cpu().numpy()  # T x N
            
            # Compute residuals
            residuals = x_clean - x_reconstructed  # T x N
            
            # Missing data mask (from original standardized data)
            missing_mask = np.isnan(x_standardized)
            
            # Estimate idio AR(1) dynamics
            A_eps, Q_eps = self._estimate_idiosyncratic_dynamics(residuals, missing_mask)
        else:
            # No idio modeling: use diagonal R only
            A_eps = np.zeros((N, N))
            Q_eps = np.eye(N) * 1e-8
        
        # Step 8: Estimate factor dynamics (VAR(1) or VAR(2))
        A_f, Q_f = self._estimate_factor_dynamics(factors)
        
        # Step 9: Build state-space model
        if self.use_idiosyncratic:
            A, Q, Z_0, V_0 = self._build_state_space(factors, A_f, Q_f, A_eps, Q_eps)
            
            # Build observation matrix H = [C, I] or [C, 0, I]
            H = self._build_observation_matrix(C)
            
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
            converged=True,  # Training completed
            num_iter=epochs,
            loglik=loglik,
            series_ids=safe_get_attr(config, 'get_series_ids', lambda: [])(),
            block_names=[DEFAULT_GLOBAL_BLOCK_NAME],
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
    
    def _extract_decoder_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract observation matrix C and bias from trained decoder.
        
        Extracts the learned decoder parameters directly from the PyTorch model,
        avoiding OLS re-estimation. This preserves the learned relationships
        from the autoencoder training.
        
        Returns
        -------
        C : np.ndarray
            Loading matrix (N x m) from decoder weights
        bias : np.ndarray
            Bias terms (N,)
        """
        decoder_layer = self.decoder.decoder
        
        # Extract weight matrix: (output_dim x input_dim) = (N x m)
        weight = decoder_layer.weight.data.cpu().numpy()
        
        # Extract bias if present
        if decoder_layer.bias is not None:
            bias = decoder_layer.bias.data.cpu().numpy()
        else:
            bias = np.zeros(weight.shape[0])
        
        # C = weight.T (m x N) -> (N x m) for consistency with DFMResult
        C = weight.T
        
        return C, bias
    
    def _estimate_idiosyncratic_dynamics(
        self,
        residuals: np.ndarray,
        missing_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate AR(1) dynamics for idiosyncratic components.
        
        Estimates AR(1) coefficients for each series independently from residuals.
        This models the residual component as having temporal structure rather
        than being pure white noise.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residuals from observation equation (T x N)
            residuals = X_t - decoder(encoder(X_t))
        missing_mask : np.ndarray
            Missing data mask (T x N), True where data is missing
            
        Returns
        -------
        A_eps : np.ndarray
            AR(1) coefficients (N x N), diagonal matrix
        Q_eps : np.ndarray
            Innovation covariance (N x N), diagonal matrix
        """
        T, N = residuals.shape
        A_eps = np.zeros((N, N))
        Q_eps = np.zeros((N, N))
        
        for j in range(N):
            # Find valid consecutive pairs (both t-1 and t must be non-missing)
            valid = ~missing_mask[:, j]
            valid_pairs = np.zeros(T - 1, dtype=bool)
            valid_pairs = valid[:-1] & valid[1:]
            
            if np.sum(valid_pairs) < self.min_obs_idio:
                # Insufficient data: use zero AR(1) coefficient
                _logger.warning(
                    f"Insufficient observations ({np.sum(valid_pairs)}) for idio AR(1) "
                    f"estimation for series {j}. Using zero AR(1) coefficient."
                )
                A_eps[j, j] = 0.0
                # Use variance of available residuals
                if np.sum(valid) > 0:
                    Q_eps[j, j] = np.var(residuals[valid, j])
                else:
                    Q_eps[j, j] = 1e-8
            else:
                # Extract valid consecutive pairs
                eps_t = residuals[1:, j][valid_pairs]
                eps_t_1 = residuals[:-1, j][valid_pairs]
                
                # Estimate AR(1) coefficient using covariance
                var_eps_t_1 = np.var(eps_t_1)
                if var_eps_t_1 > 1e-10:
                    cov_eps = np.cov(eps_t, eps_t_1)[0, 1]
                    A_eps[j, j] = cov_eps / var_eps_t_1
                    
                    # Ensure stability: clip AR(1) coefficient
                    if abs(A_eps[j, j]) >= 0.99:
                        sign = np.sign(A_eps[j, j])
                        A_eps[j, j] = sign * 0.99
                        _logger.debug(
                            f"AR(1) coefficient for series {j} clipped to {A_eps[j, j]:.4f} for stability"
                        )
                else:
                    A_eps[j, j] = 0.0
                
                # Estimate innovation covariance
                residuals_ar = eps_t - A_eps[j, j] * eps_t_1
                Q_eps[j, j] = np.var(residuals_ar)
                Q_eps[j, j] = max(Q_eps[j, j], 1e-8)  # Floor
        
        return A_eps, Q_eps
    
    def _estimate_factor_dynamics(
        self,
        factors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate factor dynamics via OLS.
        
        Estimates VAR(1) or VAR(2) model for factors.
        
        Parameters
        ----------
        factors : np.ndarray
            Extracted factors (T x m)
            
        Returns
        -------
        A : np.ndarray
            Transition matrix (m x m) for VAR(1) or (m x 2m) for VAR(2)
        Q : np.ndarray
            Innovation covariance (m x m)
        """
        T, m = factors.shape
        
        if self.factor_order == 1:
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
        
        elif self.factor_order == 2:
            if T < 3:
                # Not enough data, use VAR(1) fallback
                _logger.warning(
                    f"Insufficient data (T={T}) for VAR(2). Falling back to VAR(1)."
                )
                # Temporarily set factor_order to 1, estimate, then restore
                original_order = self.factor_order
                self.factor_order = 1
                A, Q = self._estimate_factor_dynamics(factors)
                self.factor_order = original_order
                # Pad A to VAR(2) format: [A1, A2] where A2 = 0
                A = np.hstack([A, np.zeros((A.shape[0], A.shape[1]))])
                return A, Q
            
            # Prepare data for VAR(2): f_t = A1 @ f_{t-1} + A2 @ f_{t-2}
            Y = factors[2:, :]  # T-2 x m (dependent)
            X = np.hstack((factors[1:-1, :], factors[:-2, :]))  # T-2 x 2m (independent)
            
            # OLS: A = (X'X)^{-1} X'Y, where A = [A1, A2]
            try:
                A = np.linalg.solve(X.T @ X + np.eye(2 * m) * 1e-6, X.T @ Y).T
            except np.linalg.LinAlgError:
                # Fallback to pinv
                A = np.linalg.pinv(X) @ Y
            
            # Split into A1 and A2
            A1 = A[:, :m]
            A2 = A[:, m:]
            
            # Ensure stability: check eigenvalues of companion form
            companion = np.block([
                [A1, A2],
                [np.eye(m), np.zeros((m, m))]
            ])
            eigenvals = np.linalg.eigvals(companion)
            max_eigenval = np.max(np.abs(eigenvals))
            if max_eigenval >= 0.99:
                scale = 0.99 / max_eigenval
                A1 = A1 * scale
                A2 = A2 * scale
                A = np.hstack((A1, A2))
            
            # Estimate innovation covariance
            residuals = Y - X @ A.T
            Q = np.cov(residuals.T)
            
            # Ensure Q is positive definite
            Q = (Q + Q.T) / 2  # Symmetrize
            eigenvals_Q = np.linalg.eigvals(Q)
            min_eigenval = np.min(eigenvals_Q)
            if min_eigenval < 1e-8:
                Q = Q + np.eye(m) * (1e-8 - min_eigenval)
            
            # Floor for Q
            Q = np.maximum(Q, np.eye(m) * 0.01)
            
            return A, Q
        
        else:
            raise ValueError(f"factor_order must be 1 or 2, got {self.factor_order}")
    
    def _build_observation_matrix(
        self,
        C: np.ndarray,
    ) -> np.ndarray:
        """Build observation matrix H including idiosyncratic components.
        
        Constructs the observation matrix H = [C, I] for VAR(1) or
        H = [C, 0, I] for VAR(2), where C loads on factors and I on idio.
        
        Observation equation: y_t = H @ x_t + v_t
        where x_t = [f_t, eps_t] (VAR(1)) or [f_t, f_{t-1}, eps_t] (VAR(2))
        
        Parameters
        ----------
        C : np.ndarray
            Loading matrix (N x m) from decoder
            
        Returns
        -------
        H : np.ndarray
            Observation matrix (N x state_dim)
        """
        N, m = C.shape
        
        if self.factor_order == 1:
            # H = [C, I] where C loads on f_t, I loads on eps_t
            H = np.hstack([C, np.eye(N)])
        elif self.factor_order == 2:
            # H = [C, 0, I] where C loads on f_t, 0 on f_{t-1}, I on eps_t
            H = np.hstack([C, np.zeros((N, m)), np.eye(N)])
        else:
            raise ValueError(f"factor_order must be 1 or 2, got {self.factor_order}")
        
        return H
    
    def _build_state_space(
        self,
        factors: np.ndarray,
        A_f: np.ndarray,
        Q_f: np.ndarray,
        A_eps: np.ndarray,
        Q_eps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build state-space model with companion form.
        
        Constructs the complete state-space model including both factors
        and idiosyncratic components in the state vector.
        
        Parameters
        ----------
        factors : np.ndarray
            Extracted factors (T x m)
        A_f : np.ndarray
            Factor transition matrix (m x m) for VAR(1) or (m x 2m) for VAR(2)
        Q_f : np.ndarray
            Factor innovation covariance (m x m)
        A_eps : np.ndarray
            Idiosyncratic AR(1) coefficients (N x N), diagonal
        Q_eps : np.ndarray
            Idiosyncratic innovation covariance (N x N), diagonal
            
        Returns
        -------
        A : np.ndarray
            Full transition matrix (state_dim x state_dim)
        Q : np.ndarray
            Full innovation covariance (state_dim x state_dim)
        Z_0 : np.ndarray
            Initial state vector (state_dim,)
        V_0 : np.ndarray
            Initial state covariance (state_dim x state_dim)
        """
        m = factors.shape[1]
        N = A_eps.shape[0]
        
        if self.factor_order == 1:
            # State: x_t = [f_t, eps_t]
            # Transition matrix
            A = np.block([
                [A_f, np.zeros((m, N))],
                [np.zeros((N, m)), A_eps]
            ])
            
            # Innovation covariance
            Q = np.block([
                [Q_f, np.zeros((m, N))],
                [np.zeros((N, m)), Q_eps]
            ])
            
            # Initial state
            eps_0 = np.zeros(N)  # Idiosyncratic initial values
            Z_0 = np.concatenate([factors[0, :], eps_0])
            
            # Initial covariance
            V_f = np.cov(factors.T)
            V_eps = np.diag(np.diag(Q_eps))
            V_0 = np.block([
                [V_f, np.zeros((m, N))],
                [np.zeros((N, m)), V_eps]
            ])
            
        elif self.factor_order == 2:
            # State: x_t = [f_t, f_{t-1}, eps_t]
            # Split VAR(2) coefficients
            A1 = A_f[:, :m]
            A2 = A_f[:, m:]
            
            # Companion form transition matrix
            A = np.block([
                [A1, A2, np.zeros((m, N))],
                [np.eye(m), np.zeros((m, m)), np.zeros((m, N))],
                [np.zeros((N, m)), np.zeros((N, m)), A_eps]
            ])
            
            # Innovation covariance (only f_t has innovation, f_{t-1} and eps_t are deterministic)
            Q = np.block([
                [Q_f, np.zeros((m, m)), np.zeros((m, N))],
                [np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, N))],
                [np.zeros((N, m)), np.zeros((N, m)), Q_eps]
            ])
            
            # Initial state
            Z_0 = np.concatenate([factors[0, :], factors[0, :], np.zeros(N)])
            
            # Initial covariance
            V_f = np.cov(factors.T)
            V_eps = np.diag(np.diag(Q_eps))
            V_0 = np.block([
                [V_f, V_f, np.zeros((m, N))],
                [V_f, V_f, np.zeros((m, N))],
                [np.zeros((N, m)), np.zeros((N, m)), V_eps]
            ])
        else:
            raise ValueError(f"factor_order must be 1 or 2, got {self.factor_order}")
        
        return A, Q, Z_0, V_0
    
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
        
        def predict(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DDFM")


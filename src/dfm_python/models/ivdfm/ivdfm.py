"""Identifiable Variational Dynamic Factor Model (iVDFM).

Implements the iVDFM framework that combines identifiable latent-variable modeling
with explicit stochastic dynamics. Identifiability is achieved by applying iVAE
conditions to the innovation process driving dynamics.
"""

import time
from pathlib import Path
from typing import Optional, Any, Union, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import BaseFactorModel
from ...logger import get_logger
from ...config.types import to_tensor, to_numpy
from ...config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_SEED,
    DEFAULT_DTYPE,
    DEFAULT_TOLERANCE,
)
from ...utils.errors import ModelNotTrainedError, ModelNotInitializedError, ConfigurationError
from ...utils.validation import check_condition

_logger = get_logger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron for iVDFM components."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Union[int, list],
        n_layers: int,
        activation: str = 'lrelu',
        slope: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(f'Wrong argument type for hidden_dim: {type(hidden_dim)}')
        
        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError(f'Wrong argument type for activation: {type(activation)}')
        
        # Build activation functions
        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(F.relu)
            elif act == 'tanh':
                self._act_f.append(torch.tanh)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                raise ValueError(f'Incorrect activation: {act}')
        
        # Build layers
        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[-1], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class iVDFM(BaseFactorModel, nn.Module):
    """Identifiable Variational Dynamic Factor Model.
    
    Combines identifiable latent-variable modeling (iVAE) with explicit
    stochastic dynamics. Identifiability is achieved by applying conditional
    exponential-family priors to innovations rather than states.
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        aux_dim: int,
        sequence_length: int,
        config: Optional[Any] = None,
        # Network architecture
        encoder_hidden_dim: Union[int, list] = 200,
        encoder_n_layers: int = 3,
        decoder_hidden_dim: Union[int, list] = 200,
        decoder_n_layers: int = 3,
        prior_hidden_dim: Union[int, list] = 100,
        prior_n_layers: int = 2,
        activation: str = 'lrelu',
        slope: float = 0.1,
        # Dynamics parameters
        factor_order: int = 1,  # AR order for factors (p in AR(p))
        use_companion_form: bool = True,  # Use companion form for higher-order dynamics
        # Prior/innovation parameters
        innovation_distribution: str = 'laplace',  # 'laplace', 'student_t', etc.
        decoder_var: float = 0.01,
        # Training parameters
        learning_rate: float = 1e-3,
        optimizer: str = 'Adam',
        batch_size: int = 32,
        max_epochs: int = 100,
        tolerance: float = DEFAULT_TOLERANCE,
        # Auxiliary variable
        aux_variable_type: str = 'time',  # 'time', 'regime', 'custom'
        # Device
        device: Optional[torch.device] = None,
        seed: int = DEFAULT_SEED,
    ):
        """Initialize iVDFM model.
        
        Parameters
        ----------
        data_dim : int
            Dimension of observed data (N in paper)
        latent_dim : int
            Dimension of latent factors (r in paper)
        aux_dim : int
            Dimension of auxiliary variable u_t
        sequence_length : int
            Length of time series sequences (T in paper)
        config : Optional[Any]
            Configuration object
        encoder_hidden_dim : Union[int, list]
            Hidden dimensions for innovation encoder
        encoder_n_layers : int
            Number of layers in innovation encoder
        decoder_hidden_dim : Union[int, list]
            Hidden dimensions for decoder
        decoder_n_layers : int
            Number of layers in decoder
        prior_hidden_dim : Union[int, list]
            Hidden dimensions for prior network
        prior_n_layers : int
            Number of layers in prior network
        activation : str
            Activation function ('lrelu', 'relu', 'tanh')
        slope : float
            Slope for leaky ReLU
        factor_order : int
            AR order for factor dynamics (p in AR(p))
        use_companion_form : bool
            Whether to use companion form for higher-order dynamics
        innovation_distribution : str
            Distribution for innovations ('laplace', 'student_t', etc.)
        decoder_var : float
            Decoder variance (observation noise)
        learning_rate : float
            Learning rate for optimizer
        optimizer : str
            Optimizer type ('Adam', 'AdamW', 'SGD')
        batch_size : int
            Batch size for training
        max_epochs : int
            Maximum number of training epochs
        tolerance : float
            Convergence tolerance
        aux_variable_type : str
            Type of auxiliary variable ('time', 'regime', 'custom')
        device : Optional[torch.device]
            Device for computation (None for auto-detect)
        seed : int
            Random seed
        """
        BaseFactorModel.__init__(self)
        nn.Module.__init__(self)
        
        self._config = config
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.sequence_length = sequence_length
        
        # Network architecture
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers
        self.prior_hidden_dim = prior_hidden_dim
        self.prior_n_layers = prior_n_layers
        self.activation = activation
        self.slope = slope
        
        # Dynamics parameters
        self.factor_order = factor_order
        self.use_companion_form = use_companion_form
        
        # Prior/innovation parameters
        self.innovation_distribution = innovation_distribution
        self.decoder_var = decoder_var
        
        # Training parameters
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Auxiliary variable
        self.aux_variable_type = aux_variable_type
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Random seed
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize components
        self._build_components()
        
        # Optimizer (built during training)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Training state
        self.training_state: Optional[Dict] = None
        self.factors: Optional[np.ndarray] = None
        self.innovations: Optional[np.ndarray] = None
        
        # Move to device
        self.to(self.device)
    
    def _build_components(self):
        """Build model components: encoder, decoder, prior, dynamics."""
        # Innovation encoder: q(η_t | y_{1:T}, u_t)
        # Takes full sequence and auxiliary variable, outputs variational params
        encoder_input_dim = self.data_dim * self.sequence_length + self.aux_dim
        self.innovation_encoder_mu = MLP(
            encoder_input_dim, self.latent_dim,
            self.encoder_hidden_dim, self.encoder_n_layers,
            activation=self.activation, slope=self.slope, device=str(self.device)
        )
        self.innovation_encoder_logvar = MLP(
            encoder_input_dim, self.latent_dim,
            self.encoder_hidden_dim, self.encoder_n_layers,
            activation=self.activation, slope=self.slope, device=str(self.device)
        )
        
        # Prior network: p(η_t | u_t) - outputs natural parameters λ(u_t)
        # For exponential family, we need sufficient statistics parameters
        # For Laplace: location and scale parameters
        # For now, output location and log-scale (can be extended for other distributions)
        self.prior_network = MLP(
            self.aux_dim, self.latent_dim * 2,  # location and log-scale
            self.prior_hidden_dim, self.prior_n_layers,
            activation=self.activation, slope=self.slope, device=str(self.device)
        )
        
        # Decoder: g(f_t) → y_t
        self.decoder = MLP(
            self.latent_dim, self.data_dim,
            self.decoder_hidden_dim, self.decoder_n_layers,
            activation=self.activation, slope=self.slope, device=str(self.device)
        )
        
        # Dynamics: diagonal transition matrices A and B
        # For AR(p) with companion form, we need block-diagonal structure
        if self.use_companion_form and self.factor_order > 1:
            # Block-diagonal companion form: each factor has its own AR(p) block
            state_dim = self.latent_dim * self.factor_order
            # A: block-diagonal companion matrices (one per factor)
            # B: block structure for innovations
            # For now, use learnable parameters (will be structured in forward)
            self.A_params = nn.Parameter(torch.randn(self.latent_dim, self.factor_order))
            self.B = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim))
        else:
            # First-order: simple diagonal matrices
            self.A = nn.Parameter(torch.randn(self.latent_dim))
            self.B = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim))
        
        # Initial state f_0 (learnable or zero)
        self.f0 = nn.Parameter(torch.zeros(self.latent_dim))
    
    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        optimizers = {
            'Adam': lambda: torch.optim.Adam(self.parameters(), lr=self.learning_rate),
            'AdamW': lambda: torch.optim.AdamW(self.parameters(), lr=self.learning_rate),
            'SGD': lambda: torch.optim.SGD(self.parameters(), lr=self.learning_rate),
        }
        self.optimizer = optimizers.get(self.optimizer_type, optimizers['Adam'])()
        
        # Simple step scheduler (can be customized)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.max_epochs // 3, gamma=0.5
        )
    
    def innovation_encoder(self, y_1T: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode innovations: q(η_t | y_{1:T}, u_t).
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Full observation sequence, shape (batch, T, N)
        u_t : torch.Tensor
            Auxiliary variable at time t, shape (batch, aux_dim)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Mean and log-variance of innovation posterior
        """
        # Flatten sequence: (batch, T, N) -> (batch, T*N)
        batch_size = y_1T.shape[0]
        y_flat = y_1T.view(batch_size, -1)
        
        # Concatenate with auxiliary variable
        xu = torch.cat([y_flat, u_t], dim=1)
        
        # Get variational parameters
        mu = self.innovation_encoder_mu(xu)
        logvar = self.innovation_encoder_logvar(xu)
        
        return mu, logvar
    
    def prior_params(self, u_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get prior parameters: p(η_t | u_t).
        
        Parameters
        ----------
        u_t : torch.Tensor
            Auxiliary variable, shape (batch, aux_dim)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with prior parameters (distribution-dependent)
        """
        # Output natural parameters for exponential family
        params = self.prior_network(u_t)
        
        if self.innovation_distribution == 'laplace':
            # Split into location and log-scale
            location = params[:, :self.latent_dim]
            log_scale = params[:, self.latent_dim:]
            return {'location': location, 'log_scale': log_scale}
        else:
            raise NotImplementedError(
                f"Distribution {self.innovation_distribution} not implemented"
            )
    
    def sample_innovation(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample innovation using reparameterization trick.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of innovation posterior
        logvar : torch.Tensor
            Log-variance of innovation posterior
        
        Returns
        -------
        torch.Tensor
            Sampled innovations
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std
    
    def compute_factors_from_innovations(
        self,
        eta_1T: torch.Tensor,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute factors deterministically from innovations: f_t = H(η_{1:t}).
        
        Implements: f_t = A^t f_0 + Σ_{k=0}^{t-1} A^k B η_{t-1-k}
        
        Parameters
        ----------
        eta_1T : torch.Tensor
            Innovation sequence, shape (batch, T, r)
        f0 : Optional[torch.Tensor]
            Initial state, shape (batch, r) or (r,)
        
        Returns
        -------
        torch.Tensor
            Factor sequence, shape (batch, T, r)
        """
        batch_size, T, r = eta_1T.shape
        
        if f0 is None:
            f0 = self.f0.unsqueeze(0).expand(batch_size, -1)
        elif f0.dim() == 1:
            f0 = f0.unsqueeze(0).expand(batch_size, -1)
        
        # TODO: Implement efficient convolution using companion form + Krylov
        # For now, use sequential computation
        # This should be replaced with companion_krylov for efficiency
        factors = []
        f_t = f0
        
        if self.use_companion_form and self.factor_order > 1:
            # Companion form implementation needed
            raise NotImplementedError("Companion form dynamics not yet implemented")
        else:
            # First-order: f_{t+1} = A * f_t + B * η_t
            # A is diagonal, B is diagonal or full
            A_diag = torch.sigmoid(self.A)  # Ensure stability
            B_matrix = self.B
            
            for t in range(T):
                eta_t = eta_1T[:, t, :]
                f_t = A_diag * f_t + (B_matrix @ eta_t.unsqueeze(-1)).squeeze(-1)
                factors.append(f_t)
        
        return torch.stack(factors, dim=1)  # (batch, T, r)
    
    def forward(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through iVDFM.
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Auxiliary variable sequence, shape (batch, T, aux_dim)
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - y_pred: predicted observations
            - eta: innovations
            - factors: latent factors
            - encoder_params: encoder parameters
            - prior_params: prior parameters
        """
        batch_size, T, _ = y_1T.shape
        
        # Encode innovations for each time step
        eta_list = []
        encoder_params_list = []
        prior_params_list = []
        
        for t in range(T):
            u_t = u_1T[:, t, :]
            mu, logvar = self.innovation_encoder(y_1T, u_t)
            eta_t = self.sample_innovation(mu, logvar)
            prior_params_t = self.prior_params(u_t)
            
            eta_list.append(eta_t)
            encoder_params_list.append({'mu': mu, 'logvar': logvar})
            prior_params_list.append(prior_params_t)
        
        eta_1T = torch.stack(eta_list, dim=1)  # (batch, T, r)
        
        # Compute factors deterministically
        factors_1T = self.compute_factors_from_innovations(eta_1T)
        
        # Decode observations
        y_pred_list = []
        for t in range(T):
            f_t = factors_1T[:, t, :]
            y_pred_t = self.decoder(f_t)
            y_pred_list.append(y_pred_t)
        y_pred = torch.stack(y_pred_list, dim=1)  # (batch, T, N)
        
        return {
            'y_pred': y_pred,
            'eta': eta_1T,
            'factors': factors_1T,
            'encoder_params': encoder_params_list,
            'prior_params': prior_params_list,
        }
    
    def elbo(
        self,
        y_1T: torch.Tensor,
        u_1T: torch.Tensor,
        N: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Evidence Lower Bound (ELBO).
        
        ELBO = E[log p(y_t | f_t)] - Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
        
        Parameters
        ----------
        y_1T : torch.Tensor
            Observation sequence, shape (batch, T, N)
        u_1T : torch.Tensor
            Auxiliary variable sequence, shape (batch, T, aux_dim)
        N : int
            Total number of samples in dataset (for TC computation)
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            ELBO loss and dictionary of component losses
        """
        # Forward pass
        outputs = self.forward(y_1T, u_1T)
        y_pred = outputs['y_pred']
        eta_1T = outputs['eta']
        encoder_params = outputs['encoder_params']
        prior_params = outputs['prior_params']
        
        batch_size, T, _ = y_1T.shape
        
        # Reconstruction term: E[log p(y_t | f_t)]
        # Assuming Gaussian observation model
        recon_loss = -0.5 * (
            np.log(2 * np.pi * self.decoder_var) +
            ((y_1T - y_pred) ** 2) / self.decoder_var
        ).sum(dim=-1).mean()  # Sum over data dim, mean over batch and time
        
        # KL divergence terms: Σ_t KL(q(η_t | y_{1:T}, u_t) || p(η_t | u_t))
        kl_losses = []
        for t in range(T):
            mu = encoder_params[t]['mu']
            logvar = encoder_params[t]['logvar']
            prior_p = prior_params[t]
            
            if self.innovation_distribution == 'laplace':
                # KL between Gaussian q and Laplace p
                # TODO: Implement proper KL divergence
                # For now, use simplified version
                location = prior_p['location']
                log_scale = prior_p['log_scale']
                scale = torch.exp(log_scale)
                
                # Simplified KL (needs proper implementation)
                kl_t = 0.5 * (
                    logvar - log_scale * 2 +
                    (torch.exp(logvar) + (mu - location) ** 2) / (scale ** 2) - 1
                ).sum(dim=-1)
                kl_losses.append(kl_t)
            else:
                raise NotImplementedError(
                    f"KL for {self.innovation_distribution} not implemented"
                )
        
        kl_loss = torch.stack(kl_losses, dim=1).mean()  # Mean over batch and time
        
        # Total ELBO (negative because we minimize)
        elbo = -(recon_loss - kl_loss)
        
        loss_dict = {
            'elbo': elbo,
            'reconstruction': recon_loss,
            'kl': kl_loss,
        }
        
        return elbo, loss_dict
    
    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor],
        aux_data: Union[np.ndarray, torch.Tensor],
        *args,
        **kwargs
    ) -> 'iVDFM':
        """Fit iVDFM model.
        
        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Time series data, shape (N_samples, T, N) or (T, N)
        aux_data : Union[np.ndarray, torch.Tensor]
            Auxiliary variable data, shape (N_samples, T, aux_dim) or (T, aux_dim)
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        iVDFM
            Fitted model
        """
        # Convert to tensors
        if isinstance(data, np.ndarray):
            data = to_tensor(data, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        if isinstance(aux_data, np.ndarray):
            aux_data = to_tensor(aux_data, dtype=DEFAULT_TORCH_DTYPE, device=self.device)
        
        # Ensure correct shape
        if data.dim() == 2:
            data = data.unsqueeze(0)  # Add batch dimension
        if aux_data.dim() == 2:
            aux_data = aux_data.unsqueeze(0)
        
        # Build optimizer
        self._build_optimizer()
        
        # Training loop
        self.train()
        N = data.shape[0]  # Total number of samples
        
        for epoch in range(self.max_epochs):
            # TODO: Implement proper batching and data loading
            # For now, use full batch
            self.optimizer.zero_grad()
            
            elbo, loss_dict = self.elbo(data, aux_data, N)
            elbo.backward()
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if epoch % 10 == 0:
                _logger.info(
                    f"Epoch {epoch}/{self.max_epochs}: "
                    f"ELBO={elbo.item():.4f}, "
                    f"Recon={loss_dict['reconstruction'].item():.4f}, "
                    f"KL={loss_dict['kl'].item():.4f}"
                )
        
        # Extract factors and innovations
        self.eval()
        with torch.no_grad():
            outputs = self.forward(data, aux_data)
            self.factors = to_numpy(outputs['factors'])
            self.innovations = to_numpy(outputs['eta'])
        
        # Store training state
        self.training_state = {
            'epochs': self.max_epochs,
            'final_elbo': elbo.item(),
        }
        
        return self
    
    def predict(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        aux_data: Optional[Union[np.ndarray, torch.Tensor]] = None,
        horizon: int = 1,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Predict future values.
        
        Parameters
        ----------
        data : Optional[Union[np.ndarray, torch.Tensor]]
            Historical data for prediction
        aux_data : Optional[Union[np.ndarray, torch.Tensor]]
            Auxiliary variables for prediction
        horizon : int
            Prediction horizon
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        np.ndarray
            Predictions, shape (horizon, N) or (batch, horizon, N)
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model must be trained before prediction")
        
        # TODO: Implement prediction logic
        # For now, return zeros
        raise NotImplementedError("Prediction not yet implemented")
    
    def update(
        self,
        data: Union[np.ndarray, Any],
        *args,
        **kwargs
    ) -> None:
        """Update model state with new observations.
        
        Parameters
        ----------
        data : Union[np.ndarray, Any]
            New observation data
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        """
        # TODO: Implement online update logic
        raise NotImplementedError("Update not yet implemented")
    
    def get_result(self) -> Any:
        """Extract result from trained model.
        
        Returns
        -------
        Any
            Model-specific result object
        """
        if self.training_state is None:
            raise ModelNotTrainedError("Model has not been trained yet")
        
        # TODO: Create proper result object
        return {
            'factors': self.factors,
            'innovations': self.innovations,
            'training_state': self.training_state,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model to file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'data_dim': self.data_dim,
                'latent_dim': self.latent_dim,
                'aux_dim': self.aux_dim,
                'sequence_length': self.sequence_length,
                'factor_order': self.factor_order,
                'innovation_distribution': self.innovation_distribution,
            },
            'training_state': self.training_state,
        }, path)
        _logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], *args, **kwargs) -> 'iVDFM':
        """Load model from file.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to load model from
        *args
            Additional arguments
        **kwargs
            Additional keyword arguments
        
        Returns
        -------
        iVDFM
            Loaded model instance
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')
        
        config = checkpoint['config']
        model = cls(**config, **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_state = checkpoint.get('training_state')
        
        _logger.info(f"Model loaded from {path}")
        return model

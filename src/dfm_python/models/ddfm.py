"""Deep Dynamic Factor Model (DDFM) using PyTorch.

Simplified implementation matching original TensorFlow DDFM structure.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any, Union
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    StandardScaler = None

from .base import BaseFactorModel
from ..logger import get_logger
from ..numeric.stability import convergence_checker
from ..numeric.estimator import get_idio, get_transition_params
from ..encoder.simple_autoencoder import convert_decoder_to_numpy, SimpleAutoencoder
from ..decoder.base import build_decoder
from ..utils.common import ensure_numpy, ensure_tensor
from ..config.schema.params import DDFMFitParams
from ..config.constants import (
    DEFAULT_TORCH_DTYPE,
    DEFAULT_DDFM_OBSERVATION_NOISE,
)

# Import DDFMDataset with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..dataset.ddfm_dataset import DDFMDataset

_logger = get_logger(__name__)


class DDFM(BaseFactorModel, nn.Module):
    """Deep Dynamic Factor Model using PyTorch - simplified to match original structure."""
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        dataset: Optional['DDFMDataset'] = None,
        target_series: Optional[List[str]] = None,
        encoder_size: tuple = (16, 4),
        decoder_size: tuple = None,
        scaler: Optional[Any] = None,
        seed: int = 3,
        batch_norm: bool = True,
        activation: str = 'relu',
        learning_rate: float = 0.005,
        optimizer: str = 'Adam',
        decay_learning_rate: bool = True,
        n_mc_samples: int = 100,
        window_size: int = 100,
        max_iter: int = 200,
        tolerance: float = 0.0005,
        disp: int = 10,
    ):
        """Initialize DDFM (matching original TensorFlow DDFM __init__).
        
        Args:
            data: Input data (preprocessed by user, including any lags if needed).
                  Required if dataset is not provided.
            dataset: DDFMDataset instance. If provided, extracts data, target_series, and scaler from it.
                    If both data and dataset are provided, dataset takes precedence.
            target_series: List of column names to target for reconstruction loss.
                          If None, uses all columns. Similar to original lags_input behavior
                          where lagged inputs weren't included in reconstruction loss.
                          Ignored if dataset is provided (uses dataset.target_series).
            encoder_size: Encoder architecture tuple specifying layer sizes (e.g., (16, 4))
            decoder_size: Decoder architecture tuple (None for linear decoder, i.e., single layer)
            scaler: sklearn scaler instance (e.g., StandardScaler, RobustScaler) for data standardization.
                    If None, defaults to StandardScaler(). The scaler will be fitted on the data and used for transformation.
                    Only used when dataset is not provided. If dataset is provided, scaler is extracted from dataset.
            seed: Random seed
            batch_norm: Whether to use batch normalization
            activation: Activation function ('relu' or 'tanh')
            learning_rate: Learning rate
            optimizer: Optimizer type ('Adam' or 'SGD')
            decay_learning_rate: Whether to decay learning rate
            n_mc_samples: Number of Monte Carlo samples per MCMC iteration
            window_size: Batch/window size for training
            max_iter: Maximum MCMC iterations
            tolerance: Convergence tolerance
            disp: Display interval
        """
        BaseFactorModel.__init__(self)
        nn.Module.__init__(self)
        
        # Handle dataset parameter - extract data and other info if provided
        if dataset is not None:
            # Extract data from dataset
            if hasattr(dataset, 'data_processed'):
                # Convert torch.Tensor to DataFrame
                data_array = ensure_numpy(dataset.data_processed)
                # Get column names from dataset if available
                if hasattr(dataset, 'config') and dataset.config is not None:
                    # Try to get column names from config
                    try:
                        cols = list(dataset.config.frequency.keys())
                    except (AttributeError, TypeError):
                        cols = [f'col_{i}' for i in range(data_array.shape[1])]
                else:
                    cols = [f'col_{i}' for i in range(data_array.shape[1])]
                data = pd.DataFrame(data_array, columns=cols)
            else:
                raise ValueError("DDFMDataset must have data_processed attribute")
            
            # Extract target_series from dataset if not provided
            if target_series is None and hasattr(dataset, 'target_series') and dataset.target_series is not None:
                target_series = dataset.target_series if isinstance(dataset.target_series, list) else [dataset.target_series]
            
            # Extract scaler from dataset (dataset handles scaling)
            if hasattr(dataset, 'scaler') and dataset.scaler is not None:
                self.scaler = dataset.scaler
            else:
                raise ValueError("DDFMDataset must have a scaler. Dataset should handle scaling.")
        else:
            # Use provided data (original behavior)
            if data is None:
                raise ValueError("Either data or dataset must be provided")
            
            # Use sklearn scaler (default to StandardScaler if None)
            if scaler is None:
                if not _has_sklearn:
                    raise ImportError(
                        "sklearn is required for data scaling. Install with: pip install scikit-learn"
                    )
                self.scaler = StandardScaler()
            else:
                self.scaler = scaler
            
            # Fit and transform using sklearn scaler
            self.scaler.fit(data.values)
            data_scaled = self.scaler.transform(data.values)
            data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        
        # Factor order is fixed to 1 (VAR(1) only)
        self.target_series = target_series if target_series is not None else list(data.columns)
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.batch_norm = batch_norm
        self.activation = activation
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.disp = disp
        
        # Data preprocessing (user should preprocess, but we handle scaling here)
        print("@Info - Note: Sorting data.")
        data.sort_index(inplace=True)
        
        # Extract mean/center and scale from scaler generically
        # Scaler was already fit (by dataset or in direct mode), so extract from scaler attributes
        # Note: For dataset mode, scaler was fit on original data before scaling
        # For direct mode, scaler was fit on original data before transformation
        self.mean_z = self._extract_scaler_mean(self.scaler, data.values)
        self.sigma_z = self._extract_scaler_scale(self.scaler, data.values)
        
        # Store data
        self.data = data
        
        # Get target series indices
        if self.target_series is None or len(self.target_series) == 0:
            self.target_series = list(data.columns)
            self.target_indices = list(range(data.shape[1]))
        elif isinstance(self.target_series[0], str):
            self.target_indices = [data.columns.get_loc(col) for col in self.target_series]
        else:
            self.target_indices = self.target_series
        
        self.bool_miss = self.data.isnull().values
        self.bool_no_miss = self.bool_miss == False
        self.data_mod_only_miss = self.data.copy()
        self.data_mod = self.data.copy()
        self.data_tmp = self.data.copy()
        self.z_actual = self.data.values
        
        if self.decoder_size is None:
            self.filter_type = "KalmanFilter"
        else:
            self.filter_type = "ToBeDefined"
        
        self.rng = np.random.RandomState(seed)
        self.initializer_seed = seed
        
        # Optimizer setup (matching original lines 89-97)
        if decay_learning_rate:
            from torch.optim.lr_scheduler import ExponentialLR
            self.base_lr = learning_rate
            self.learning_rate = learning_rate
        else:
            self.learning_rate = learning_rate
            self.base_lr = learning_rate
        
        if optimizer == 'Adam':
            self.optimizer_type = 'Adam'
        elif optimizer == 'SGD':
            self.optimizer_type = 'SGD'
        else:
            raise KeyError("Optimizer must be SGD or Adam")
        
        self.decay_learning_rate = decay_learning_rate
        
        # Attributes to be populated
        self.loss_now = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.eps = None
        self.factors = None
        self.last_neurons = None
        self.factors_filtered = None
        self.state_space = None
        # Initialize DDFMFitParams as None - will be populated in build_state_space()
        self.state_space_params: Optional[DDFMFitParams] = None
        self.latents = dict()
    
    @staticmethod
    def _extract_scaler_mean(scaler: Any, original_data: np.ndarray) -> np.ndarray:
        """Extract mean/center from sklearn scaler generically.
        
        Tries common attribute names used by different scaler types:
        - StandardScaler: mean_
        - RobustScaler: center_
        - Other scalers: tries inverse_transform with zeros
        
        Parameters
        ----------
        scaler : Any
            sklearn scaler instance
        original_data : np.ndarray
            Original data (fallback if scaler doesn't have mean attributes)
            
        Returns
        -------
        np.ndarray
            Mean/center values
        """
        # Try common attribute names
        if hasattr(scaler, 'mean_'):
            return scaler.mean_
        elif hasattr(scaler, 'center_'):
            return scaler.center_
        else:
            # Fallback: use inverse_transform with zeros to get mean
            # This works for any scaler that implements inverse_transform
            try:
                zero_data = np.zeros((1, original_data.shape[1]))
                mean_estimate = scaler.inverse_transform(zero_data)[0]
                return mean_estimate
            except (AttributeError, ValueError):
                # Last resort: compute from original data
                return np.mean(original_data, axis=0)
    
    @staticmethod
    def _extract_scaler_scale(scaler: Any, original_data: np.ndarray) -> np.ndarray:
        """Extract scale from sklearn scaler generically.
        
        Tries common attribute names used by different scaler types:
        - StandardScaler, RobustScaler: scale_
        - Other scalers: estimates from inverse_transform
        
        Parameters
        ----------
        scaler : Any
            sklearn scaler instance
        original_data : np.ndarray
            Original data (fallback if scaler doesn't have scale attributes)
            
        Returns
        -------
        np.ndarray
            Scale values
        """
        # Try common attribute names
        if hasattr(scaler, 'scale_'):
            return scaler.scale_
        else:
            # Fallback: estimate scale using inverse_transform
            # For a scaler: scaled = (original - mean) / scale
            # So: original = scaled * scale + mean
            # If we inverse_transform(1) and inverse_transform(0), we get:
            # inverse_transform(1) = 1 * scale + mean = scale + mean
            # inverse_transform(0) = 0 * scale + mean = mean
            # So: scale = inverse_transform(1) - inverse_transform(0)
            try:
                zero_data = np.zeros((1, original_data.shape[1]))
                one_data = np.ones((1, original_data.shape[1]))
                
                zero_inv = scaler.inverse_transform(zero_data)[0]
                one_inv = scaler.inverse_transform(one_data)[0]
                
                # Scale is the difference (absolute value to handle negative scales)
                scale_estimate = np.abs(one_inv - zero_inv)
                # Avoid division by zero - if scale is too small, use 1.0
                scale_estimate = np.where(scale_estimate < 1e-10, 1.0, scale_estimate)
                return scale_estimate
            except (AttributeError, ValueError, TypeError):
                # Last resort: compute from original data
                return np.std(original_data, axis=0, ddof=0)
    
    def build_inputs(self, interpolate: bool = True) -> None:
        """Build inputs from dataset (user should preprocess, but we handle missing values)."""
        # User should preprocess data including lags, so we just use data directly
        self.data_tmp = self.data_mod.copy()
        
        if interpolate and self.data_tmp.isna().sum().sum() > 0:
            self.data_tmp.interpolate(method='spline', limit_direction='both', inplace=True, order=3)
    
    def build_model(self) -> None:
        """Build encoder, decoder, and autoencoder (matching original lines 137-176)."""
        # Input dimension is just the number of features (user should include lags if needed)
        input_dim = self.data.shape[1]
        
        # Build encoder (matching original lines 143-158)
        layers_list = []
        if len(self.encoder_size) > 1:
            layers_list.append(nn.Linear(input_dim, self.encoder_size[0]))
            if self.batch_norm:
                layers_list.append(nn.BatchNorm1d(self.encoder_size[0]))
            layers_list.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())
            
            for j in self.encoder_size[1:]:
                layers_list.append(nn.Linear(self.encoder_size[0] if len(layers_list) == 3 else layers_list[-3].out_features, j))
                if self.batch_norm:
                    layers_list.append(nn.BatchNorm1d(j))
                layers_list.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())
        else:
            layers_list.append(nn.Linear(input_dim, self.encoder_size[0]))
        
        # Initialize with Xavier (GlorotNormal) matching original
        for layer in layers_list:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
        self.encoder = nn.Sequential(*layers_list)
        
        # Build decoder (matching original lines 159-173)
        # Decoder outputs only target series (similar to original where lagged inputs weren't in reconstruction)
        latent_dim = self.encoder_size[-1]
        output_dim = len(self.target_indices)
        
        self.decoder = build_decoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            decoder_size=self.decoder_size,
            activation=self.activation,
            seed=self.initializer_seed
        )
        
        # Build autoencoder
        self.autoencoder = SimpleAutoencoder(self.encoder, self.decoder)
    
    def pre_train(self, min_obs: int = 50, mult_epoch_pre: int = 1) -> None:
        """Pre-train model (matching original lines 178-204)."""
        self.build_inputs(interpolate=False)
        
        if len(self.data_tmp.dropna()) >= min_obs:
            inpt_pre_train = self.data_tmp.dropna().values
            loss_fn = nn.MSELoss()
        else:
            self.build_inputs()
            inpt_pre_train = self.data_tmp.dropna().values
            # Use masked loss for missing data
            def masked_loss(pred, target):
                mask = ~torch.isnan(target)
                return nn.functional.mse_loss(pred[mask], target[mask])
            loss_fn = masked_loss
        
        # Only use target series for output (similar to original where lagged inputs weren't in reconstruction)
        oupt_pre_train = self.data_tmp.dropna()[self.target_series].values
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate) if self.optimizer_type == 'Adam' else torch.optim.SGD(self.autoencoder.parameters(), lr=self.learning_rate)
        
        inpt_tensor = ensure_tensor(inpt_pre_train, dtype=DEFAULT_TORCH_DTYPE)
        oupt_tensor = ensure_tensor(oupt_pre_train, dtype=DEFAULT_TORCH_DTYPE)
        
        self.autoencoder.train()
        for epoch in range(self.n_mc_samples * mult_epoch_pre):
            optimizer.zero_grad()
            pred = self.autoencoder.forward(inpt_tensor)
            loss = loss_fn(pred, oupt_tensor)
            loss.backward()
            optimizer.step()
                
    def train(self) -> None:
        """Train DDFM (matching original lines 206-265)."""
        # Re-compile with masked loss
        self.build_inputs()
        
        # Initial prediction
        data_tmp_tensor = ensure_tensor(self.data_tmp.values, dtype=DEFAULT_TORCH_DTYPE)
        self.autoencoder.eval()
        with torch.no_grad():
            prediction_iter_target = self.autoencoder.forward(data_tmp_tensor)
        prediction_iter_np = ensure_numpy(prediction_iter_target)
        
        # Expand prediction to full data shape (for missing value imputation)
        prediction_iter_full = np.zeros((prediction_iter_np.shape[0], self.data.shape[1]))
        prediction_iter_full[:, self.target_indices] = prediction_iter_np
        
        # Update missings
        self.data_mod_only_miss.values[self.bool_miss] = prediction_iter_full[self.bool_miss]
        
        # Get idio (only for target series)
        self.eps = self.data_mod_only_miss[self.target_series].values - prediction_iter_np
        
        iter = 0
        not_converged = True
        
        # MCMC loop
        while not_converged and iter < self.max_iter:
            # Get idio distr (Phi is AR(1) coefficient matrix)
            Phi, mu_eps, std_eps = get_idio(self.eps, self.bool_no_miss[:, self.target_indices])
            
            # Subtract conditional AR-idio mean from x
            eps_expanded = np.zeros((self.eps.shape[0], self.data.shape[1]))
            eps_expanded[:, self.target_indices] = self.eps
            self.data_mod[1:] = self.data_mod_only_miss[1:] - eps_expanded[:-1, :] @ Phi
            self.data_mod[:1] = self.data_mod_only_miss[:1]
            
            # Build inputs
            self.build_inputs()
            
            # Generate MC samples (only for target series)
            eps_draws = self.rng.multivariate_normal(mu_eps, np.diag(std_eps), (self.n_mc_samples, self.data_tmp.shape[0]))
            
            # Initialize noisy inputs
            x_sim_den = np.zeros((eps_draws.shape[0], eps_draws.shape[1], self.data_tmp.shape[1]))
            
            # Loop over MC samples
            for i in range(self.n_mc_samples):
                x_sim_den[i, :, :] = self.data_tmp.values.copy()
                # Corrupt input data (only target series)
                x_sim_den[i, :, self.target_indices] = x_sim_den[i, :, self.target_indices] - eps_draws[i, :, :]
                
                # Fit autoencoder (matching original line 246)
                # y is only target series
                self.autoencoder.fit(
                    x=x_sim_den[i, :, :],
                    y=self.z_actual[:, self.target_indices],
                    epochs=1,  # epochs=1 per MC sample (matching original)
                    batch_size=self.window_size,
                    learning_rate=self.learning_rate,
                    optimizer_type=self.optimizer_type,
                    decay_learning_rate=self.decay_learning_rate,
                    verbose=0
                )
            
            # Update factors
            x_sim_den_tensor = ensure_tensor(x_sim_den, dtype=DEFAULT_TORCH_DTYPE)
            self.autoencoder.eval()
            with torch.no_grad():
                factors_list = [self.encoder(x_sim_den_tensor[i]) for i in range(x_sim_den_tensor.shape[0])]
            self.factors = np.array([ensure_numpy(f) for f in factors_list])
            
            # Check convergence
            predictions_list = [self.decoder(factors_list[i]) for i in range(len(factors_list))]
            prediction_iter_target = np.mean(np.array([ensure_numpy(p) for p in predictions_list]), axis=0)
            
            # Expand to full data shape for missing value imputation
            prediction_iter_full = np.zeros((prediction_iter_target.shape[0], self.data.shape[1]))
            prediction_iter_full[:, self.target_indices] = prediction_iter_target
            
            if iter > 1:
                delta, self.loss_now = convergence_checker(prediction_prev_iter, prediction_iter_target, self.z_actual[:, self.target_indices])
                if iter % self.disp == 0:
                    print(f'@Info: iteration: {iter} - new loss: {self.loss_now} - delta: {delta}')
                if delta < self.tolerance:
                    not_converged = False
                    print(f'@Info: Convergence achieved in {iter} iterations - new loss: {self.loss_now} - delta: {delta} < {self.tolerance}')
            
            prediction_prev_iter = prediction_iter_target.copy()
            
            # Update missings
            self.data_mod_only_miss.values[self.bool_miss] = prediction_iter_full[self.bool_miss]
            
            # Update idio (only for target series)
            self.eps = self.data_mod_only_miss[self.target_series].values - prediction_iter_target
            iter += 1
        
        # Get last neurons
        if self.decoder_size is None:
            self.last_neurons = self.factors
        else:
            # Extract second-to-last layer output
            decoder_intermediate = nn.Sequential(*list(self.decoder.children())[:-1])
            self.autoencoder.eval()
            with torch.no_grad():
                last_neurons_list = [decoder_intermediate(self.encoder(ensure_tensor(x_sim_den[i, :, :], dtype=DEFAULT_TORCH_DTYPE))) for i in range(x_sim_den.shape[0])]
            self.last_neurons = np.array([ensure_numpy(ln) for ln in last_neurons_list])
        
        if not_converged:
            print("@Info: Convergence not achieved within the maximum number of iteration!")
    
    def build_state_space(self) -> None:
        """Build state-space model (matching original lines 279-314)."""
        f_t = np.mean(self.factors, axis=0)
        eps_t = self.eps
        
        # Get params from decoder (factor_order is fixed to 1)
        # Determine use_bias from decoder's last layer (decoder is nn.Sequential from build_decoder)
        linear_layers = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)]
        use_bias = linear_layers[-1].bias is not None if linear_layers else True
        bs, H = convert_decoder_to_numpy(self.decoder, has_bias=use_bias, factor_order=1)
        
        # Modify mean with bias term
        self.mean_z = self.mean_z + bs * self.sigma_z
        
        # Get transition equation params (factor_order is fixed to 1)
        F, Q, mu_0, Sigma_0, x_t = get_transition_params(f_t, eps_t, bool_no_miss=self.bool_no_miss[:, self.target_indices])
        
        self.latents["ae_states"] = x_t
        
        R = np.eye(eps_t.shape[1]) * DEFAULT_DDFM_OBSERVATION_NOISE
        
        # Initialize and store state-space parameters in DDFMFitParams dataclass
        self.state_space_params = DDFMFitParams(
            F=F,
            Q=Q,
            mu_0=mu_0,
            Sigma_0=Sigma_0,
            H=H,
            R=R
        )
        
        # Note: StateSpace class would need to be implemented or imported
        # For now, just store the params
        self.state_space = self.state_space_params
    
    def fit(self, build_state_space: bool = False):
        """Fit DDFM (matching original lines 316-329)."""
        self.build_model()
        self.pre_train()
        self.train()
        if build_state_space:
            self.build_state_space()
            # Filtering would go here if StateSpace class is available
            # self.latents["filtered"], self.latents["sigma_kf"] = self.filter(self.data.values)
            # self.factors_filtered = self.latents["filtered"][:, 1:self.encoder_size[-1] + 1]

"""Kernelized Dynamic Factor Model (KDFM) implementation.

This module implements KDFM with two-stage VARMA architecture:
- Stage 1 (AR): Companion SSM for VAR coefficients
- Stage 2 (MA): MA Companion SSM for moving average dynamics
- Structural identification: Transform residuals to structural shocks
- Gradient descent training (not EM algorithm)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..config import (
    ConfigSource,
    MergedConfigSource,
    make_config_source,
)
from ..config.schema import KDFMConfig
from ..config.results import KDFMResult
from ..logger import get_logger
from ..ssm.companion import CompanionSSM, MACompanionSSM
from .functional.structural import StructuralIdentificationSSM
from .functional.irf import compute_irf
from .base import BaseFactorModel

if TYPE_CHECKING:
    from ..datamodule import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class KDFMTrainingState:
    """State tracking for KDFM training."""
    ar_coeffs: np.ndarray
    ma_coeffs: Optional[np.ndarray]
    structural_matrix: np.ndarray
    training_loss: float
    num_iter: int
    converged: bool


import pytorch_lightning as pl


class KDFM(BaseFactorModel, pl.LightningModule):
    """High-level API for Kernelized Dynamic Factor Model (PyTorch Lightning module).
    
    This class implements KDFM with two-stage VARMA architecture:
    - Stage 1 (AR): h_{t+1} = A^AR h_t + B ε_t, z_t = C h_t
    - Stage 2 (MA): h'_{t+1} = A^MA h'_t + B' z_t, y_t = C' h'_t
    
    Uses gradient descent training (like DDFM), not EM algorithm (like DFM).
    Uses Krylov FFT for efficient O(T log T) forward pass.
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import KDFM, KDFMDataModule, KDFMTrainer
        >>> import pandas as pd
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pd.read_csv('data/your_data.csv')
        >>> df_processed = df[[col for col in df.columns if col != 'date']]
        >>> 
        >>> # Step 2: Create DataModule
        >>> dm = KDFMDataModule(config_path='config/kdfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = KDFM(ar_order=1, ma_order=0)
        >>> model.load_config('config/kdfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = KDFMTrainer(max_epochs=100)
        >>> trainer.fit(model, dm)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
        config: Optional[KDFMConfig] = None,
        ar_order: int = 1,
        ma_order: int = 0,
        learning_rate: Optional[float] = None,
        max_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        weight_decay: Optional[float] = None,
        grad_clip_val: Optional[float] = None,
        structural_method: str = 'cholesky',
        structural_reg_weight: Optional[float] = None,
        **kwargs
    ):
        """Initialize KDFM instance.
        
        Parameters
        ----------
        config : KDFMConfig, optional
            KDFM configuration. Can be loaded later via load_config().
        ar_order : int, default=1
            VAR order p
        ma_order : int, default=0
            MA order q (0 = pure VAR)
        learning_rate : float, default=0.001
            Learning rate for Adam optimizer
        max_epochs : int, default=100
            Maximum training epochs
        batch_size : int, default=32
            Batch size for training
        weight_decay : float, default=1e-5
            Weight decay (L2 regularization)
        grad_clip_val : float, default=1.0
            Gradient clipping value
        structural_method : str, default='cholesky'
            Structural identification method: 'cholesky', 'full', 'low_rank'
        structural_reg_weight : float, default=0.1
            Weight for structural regularization loss
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        BaseFactorModel.__init__(self)
        pl.LightningModule.__init__(self)
        
        # Import constants for defaults (consolidated import)
        from ..config.constants import (
            DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS, DEFAULT_BATCH_SIZE,
            DEFAULT_REGULARIZATION_SCALE, DEFAULT_GRAD_CLIP_VAL,
            DEFAULT_STRUCTURAL_REG_WEIGHT
        )
        
        # Store parameters (use constants if not provided)
        self.ar_order = ar_order
        self.ma_order = ma_order
        
        # Initialize config using base class pattern
        # Create temporary config if none provided (will be replaced via load_config if needed)
        if config is None:
            config = self._create_temp_config()
        # Use type: ignore for config assignment since KDFMConfig is compatible with BaseModelConfig
        self._config = config  # type: ignore[assignment]
        
        # Set parameters with defaults from constants
        self.learning_rate = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
        self.max_epochs = max_epochs if max_epochs is not None else DEFAULT_MAX_EPOCHS
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        self.weight_decay = weight_decay if weight_decay is not None else DEFAULT_REGULARIZATION_SCALE
        self.grad_clip_val = grad_clip_val if grad_clip_val is not None else DEFAULT_GRAD_CLIP_VAL
        self.structural_method = structural_method
        self.structural_reg_weight = structural_reg_weight if structural_reg_weight is not None else DEFAULT_STRUCTURAL_REG_WEIGHT
        
        # Will be initialized in setup() when data dimensions are known
        self.companion_ar: Optional[CompanionSSM] = None
        self.companion_ma: Optional[MACompanionSSM] = None
        self.structural_id: Optional[StructuralIdentificationSSM] = None
        
        # Training state
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        
        # Use automatic optimization for gradient descent
        self.automatic_optimization = True
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model components when data dimensions are known.
        
        This is called by Lightning before training starts.
        """
        # Will be initialized in initialize_from_data() or first training step
        pass
    
    def initialize_from_data(self, X: torch.Tensor) -> None:
        """Initialize parameters from data.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        """
        T, N = X.shape
        K = N  # Number of variables
        
        # Initialize AR companion SSM
        self.companion_ar = CompanionSSM(
            n_vars=K,
            lag_order=self.ar_order,
            n_kernels=1,
            kernel_init='normal',
            norm_order=1
        )
        
        # Initialize MA companion SSM (if q > 0)
        if self.ma_order > 0:
            self.companion_ma = MACompanionSSM(
                n_vars=K,
                ma_order=self.ma_order,
                n_kernels=1,
                kernel_init='normal',
                norm_order=1
            )
        else:
            self.companion_ma = None
        
        # Initialize structural identification
        self.structural_id = StructuralIdentificationSSM(
            n_vars=K,
            lag_order=self.ar_order,
            method=self.structural_method,
            align_with_latent_state=True
        )
        
        # Move all components to same device as data
        device = X.device
        self._move_components_to_device(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through two-stage VARMA architecture.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data (B x T x N) or (T x N)
            
        Returns
        -------
        y_pred : torch.Tensor
            Predictions (B x T x N) or (T x N)
        """
        if self.companion_ar is None:
            raise RuntimeError("KDFM forward: Model not initialized. Call initialize_from_data() first.")
        
        # Handle different input shapes
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, T, N)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, T, N = x.shape
        
        # Transform input to structural shocks via structural identification
        # Reshape for structural identification: (B*T, N)
        residuals_flat = x.view(B * T, N)
        if self.structural_id is not None:
            structural_shocks = self.structural_id(residuals_flat)  # (B*T, shock_dim)
        else:
            structural_shocks = residuals_flat  # Fallback if not initialized
        
        # Reshape back: (B, T, shock_dim)
        structural_shocks = structural_shocks.view(B, T, -1)
        
        # Stage 1 (AR): Forward pass through AR companion SSM
        z_t = self.companion_ar(structural_shocks)  # (B, T, K)
        
        # Stage 2 (MA): Forward pass through MA companion SSM (if q > 0)
        if self.companion_ma is not None:
            y_pred = self.companion_ma(z_t)  # (B, T, K)
        else:
            y_pred = z_t  # Pure VAR: no MA stage
        
        if squeeze_output:
            y_pred = y_pred.squeeze(0)  # (T, N)
        
        return y_pred
    
    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Training step for KDFM.
        
        Parameters
        ----------
        batch : torch.Tensor or tuple
            Data tensor or (data, target) tuple
        batch_idx : int
            Batch index
            
        Returns
        -------
        loss : torch.Tensor
            Total loss (prediction + structural regularization)
        """
        # Handle batch format
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        else:
            data = batch
            target = data
        
        # Ensure data is on same device as model
        device = next(self.parameters()).device
        data = data.to(device)
        target = target.to(device)
        
        # Initialize if needed
        if self.companion_ar is None:
            # Use first batch to determine dimensions
            if data.ndim == 2:
                self.initialize_from_data(data)
            else:
                self.initialize_from_data(data[0])
        
        # Forward pass
        y_pred = self.forward(data)
        
        # Prediction loss: MSE
        pred_loss = nn.functional.mse_loss(y_pred, target)
        
        # Structural regularization loss: orthogonality constraint on structural shocks
        struct_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if self.structural_id is not None:
            # Get structural matrix
            S = self.structural_id.get_structural_matrix()
            # Compute orthogonality loss: ||S @ S^T - I||^2
            S_S_T = S @ S.T
            I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
            struct_loss = nn.functional.mse_loss(S_S_T, I)
        
        # Total loss
        total_loss = pred_loss + self.structural_reg_weight * struct_loss
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('pred_loss', pred_loss, on_step=True, on_epoch=True)
        self.log('struct_loss', struct_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def _get_model_components(self) -> List[Optional[Union[CompanionSSM, MACompanionSSM, StructuralIdentificationSSM]]]:
        """Get list of all model components.
        
        Returns
        -------
        List
            List of model components (may contain None)
        """
        return [self.companion_ar, self.companion_ma, self.structural_id]
    
    def _move_components_to_device(self, device: torch.device) -> None:
        """Move all model components to specified device.
        
        Parameters
        ----------
        device : torch.device
            Target device
        """
        for component in self._get_model_components():
            if component is not None:
                component.to(device)
    
    def _collect_parameters(self) -> List[torch.nn.Parameter]:
        """Collect all trainable parameters from model components.
        
        Returns
        -------
        List[torch.nn.Parameter]
            List of all trainable parameters
        """
        params = []
        for component in self._get_model_components():
            if component is not None:
                params.extend(component.parameters())
        return params
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizer for KDFM training.
        
        Returns
        -------
        List[torch.optim.Optimizer]
            List containing Adam optimizer with learning rate and weight decay.
            Returns dummy optimizer if model parameters not yet initialized.
        """
        params = self._collect_parameters()
        
        if not params:
            # Return dummy optimizer if no parameters yet (will be updated when model is initialized)
            return [self._create_dummy_optimizer(self.learning_rate)]
        
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return [optimizer]
    
    def _check_trained(self) -> None:
        """Check if model is trained, raise error if not.
        
        Override base class to check if model components are initialized,
        and try to extract result if model is initialized but _result is None.
        """
        if self._result is None:
            # Try to extract result if model is initialized
            if self.companion_ar is not None:
                try:
                    self._result = self.get_result()
                    return
                except (NotImplementedError, AttributeError, ValueError) as e:
                    # get_result() failed, model not fully trained
                    _logger.debug(f"KDFM _check_trained: get_result() failed: {e}")
            
            # Fall back to base class check
            BaseFactorModel._check_trained(self)
    
    def predict(  # type: ignore[override]
        self,
        horizon: Optional[int] = None,
        *,
        history: Optional[int] = None,  # Unused, kept for base class compatibility
        return_series: bool = True,
        return_factors: bool = True,
        target: Optional[List[str]] = None  # Unused, kept for base class compatibility
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict future values.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods to forecast. If None, uses default.
        return_series : bool, default=True
            Whether to return forecasted series
        return_factors : bool, default=True
            Whether to return forecasted factors
            
        Returns
        -------
        Xf : np.ndarray or Tuple[np.ndarray, np.ndarray]
            Forecasted observations (horizon x N) and optionally factors
        """
        if horizon is None:
            horizon = 6  # Default horizon
        self._check_trained()
        
        # Get last state from result
        result = self.get_result()
        Z_last = result.Z[-1, :]  # Last factor state
        
        # Forecast using companion matrix structure
        # This is a simplified version - full implementation would use
        # companion matrix powers for forecasting
        if self.companion_ar is None:
            raise ValueError("KDFM predict: Model not initialized. Train model first.")
        
        # Forecast factors using VAR dynamics
        Z_forecast = self._forecast_var_factors(
            Z_last,
            result.A,
            self.ar_order,
            horizon
        )
        
        # Transform to observations
        X_forecast = self._transform_factors_to_observations(
            Z_forecast,
            result.C,
            result.Wx,
            result.Mx
        )
        
        # Return based on flags
        if return_series and return_factors:
            return X_forecast, Z_forecast
        elif return_series:
            return X_forecast
        else:
            return Z_forecast
    
    def update(  # type: ignore[override]
        self,
        X_std: np.ndarray,
        *,
        history: Optional[int] = None,  # Unused, kept for base class compatibility
        kalman_filter: Optional[Any] = None,  # Unused, kept for base class compatibility
        scaler: Optional[Any] = None  # Unused, kept for base class compatibility
    ) -> 'KDFM':
        """Update model state with standardized data for nowcasting.
        
        Parameters
        ----------
        X_std : np.ndarray
            Standardized data (T_new x N)
            
        Returns
        -------
        self : KDFM
            Returns self for method chaining
        """
        # Update internal state (simplified - full implementation would update
        # companion SSM states)
        self.data_processed = torch.tensor(X_std, dtype=torch.float32)
        
        return self
    
    def _create_temp_config(self, block_name: Optional[str] = None) -> KDFMConfig:  # type: ignore[override]
        """Create temporary KDFMConfig for initialization.
        
        Parameters
        ----------
        block_name : str, optional
            Block name (ignored for KDFM)
        
        Returns
        -------
        KDFMConfig
            Temporary configuration object
        """
        from ..config.schema import SeriesConfig
        # KDFM does not use blocks structure - only series
        return KDFMConfig(
            series=[SeriesConfig(series_id='temp', frequency='m')],
            ar_order=self.ar_order if hasattr(self, 'ar_order') else 1,
            ma_order=self.ma_order if hasattr(self, 'ma_order') else 0
        )
    
    def _extract_companion_params(self, companion_ssm: Optional[Union[CompanionSSM, MACompanionSSM]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract companion matrix and B, C parameters from companion SSM.
        
        Parameters
        ----------
        companion_ssm : CompanionSSM or MACompanionSSM, optional
            Companion SSM instance (AR or MA)
            
        Returns
        -------
        tuple
            (A_matrix, B_matrix, C_matrix) as numpy arrays, or None if not available
        """
        if companion_ssm is None:
            return None, None, None
        
        try:
            # Extract companion matrix A
            A = companion_ssm.get_companion_matrix()
            # Handle both 2D and 3D matrices (n_kernels=1 returns 2D, n_kernels>1 returns 3D)
            if A.ndim == 3:
                A_np = A[0].detach().cpu().numpy()
            else:
                A_np = A.detach().cpu().numpy()
            
            # Extract B and C parameters (both CompanionSSM and MACompanionSSM have these)
            B_param = companion_ssm.B
            C_param = companion_ssm.C
            B_data = B_param.data[0] if B_param.ndim > 2 else B_param.data
            C_data = C_param.data[0] if C_param.ndim > 2 else C_param.data
            B_np = B_data.detach().cpu().numpy()
            C_np = C_data.detach().cpu().numpy()
            
            return A_np, B_np, C_np
        except Exception as e:
            _logger.warning(f"KDFM _extract_companion_params: Failed to extract parameters: {e}")
            return None, None, None
    
    def _can_compute_irf(
        self,
        ma_transition: Optional[np.ndarray],
        ar_input: Optional[np.ndarray],
        ar_output: Optional[np.ndarray],
        ma_input: Optional[np.ndarray],
        ma_output: Optional[np.ndarray],
        structural_matrix: Optional[np.ndarray]
    ) -> bool:
        """Check if all required parameters are available for IRF computation.
        
        Parameters
        ----------
        ma_transition : np.ndarray, optional
            MA stage transition matrix
        ar_input : np.ndarray, optional
            AR stage input matrix
        ar_output : np.ndarray, optional
            AR stage output matrix
        ma_input : np.ndarray, optional
            MA stage input matrix
        ma_output : np.ndarray, optional
            MA stage output matrix
        structural_matrix : np.ndarray, optional
            Structural identification matrix
            
        Returns
        -------
        bool
            True if all required parameters are available
        """
        return (self.companion_ma is not None and 
                ma_transition is not None and
                ar_input is not None and ar_output is not None and
                ma_input is not None and ma_output is not None and
                structural_matrix is not None)
    
    def get_result(self) -> KDFMResult:
        """Extract parameters and create KDFMResult.
        
        Returns
        -------
        KDFMResult
            KDFM estimation results
        """
        from ..config.constants import DEFAULT_REGULARIZATION
        
        if self.companion_ar is None:
            raise ValueError("KDFM get_result: Model not initialized. Train model first.")
        
        # Extract parameters and convert to numpy
        ar_coeffs_np = self.companion_ar.extract_coefficients().detach().cpu().numpy()
        
        ma_coeffs_np = None
        if self.companion_ma is not None:
            ma_coeffs_np = self.companion_ma.extract_coefficients().detach().cpu().numpy()
        
        # Get structural matrix
        S_np = None
        if self.structural_id is not None:
            S_np = self.structural_id.get_structural_matrix().detach().cpu().numpy()
        
        # Extract companion matrices and parameters using helper method
        ar_transition, ar_input, ar_output = self._extract_companion_params(self.companion_ar)
        ma_transition, ma_input, ma_output = self._extract_companion_params(self.companion_ma)
        
        # Compute IRFs if all required parameters are available
        irf_reduced = None
        irf_structural = None
        if self._can_compute_irf(ma_transition, ar_input, ar_output, ma_input, ma_output, S_np):
            try:
                from ..config.constants import DEFAULT_IRF_HORIZON
                irf_reduced, irf_structural = compute_irf(
                    torch.tensor(ar_transition),
                    torch.tensor(ma_transition),
                    torch.tensor(ar_input),
                    torch.tensor(ar_output),
                    torch.tensor(ma_input),
                    torch.tensor(ma_output),
                    torch.tensor(S_np),
                    horizon=DEFAULT_IRF_HORIZON
                )
            except Exception as e:
                _logger.warning(f"KDFM get_result: Failed to compute IRFs: {e}")
        
        # Determine dimensions from extracted parameters
        n_vars = ar_coeffs_np.shape[1] if ar_coeffs_np is not None else 1
        n_factors = n_vars  # KDFM uses same dimension for factors and variables in AR stage
        
        # Create result with proper dimensions
        # KDFM uses a two-stage VARMA structure rather than traditional factor model,
        # so some result fields (x_sm, X_sm, Z) are minimal placeholders
        result = KDFMResult(
            x_sm=np.zeros((1, n_vars)),
            X_sm=np.zeros((1, n_vars)),
            Z=np.zeros((1, n_factors)),
            C=ar_output if ar_output is not None else np.eye(n_factors, n_vars),
            R=np.eye(n_vars) * DEFAULT_REGULARIZATION,  # Small noise covariance
            A=ar_transition[:n_factors, :n_factors] if ar_transition is not None else np.eye(n_factors),
            Q=np.eye(n_factors) * DEFAULT_REGULARIZATION,  # Small process noise
            Mx=self.Mx if self.Mx is not None else np.zeros(n_vars),
            Wx=self.Wx if self.Wx is not None else np.ones(n_vars),
            Z_0=np.zeros(n_factors),
            V_0=np.eye(n_factors) * DEFAULT_REGULARIZATION,
            r=np.array([n_factors]),
            p=self.ar_order,
            converged=True,
            num_iter=0,
            # KDFM-specific fields
            S=S_np,
            structural_shocks=None,  # Would be computed during training
            irf_reduced=irf_reduced,
            irf_structural=irf_structural,
            ar_coeffs=ar_coeffs_np,
            ma_coeffs=ma_coeffs_np
        )
        
        return result


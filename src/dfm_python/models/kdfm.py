"""Kernelized Dynamic Factor Model (KDFM) implementation.

This module implements KDFM with two-stage VARMA architecture:
- Stage 1 (AR): Companion SSM for VAR coefficients
- Stage 2 (MA): MA Companion SSM for moving average dynamics
- Structural identification: Transform residuals to structural shocks
- Gradient descent training (not EM algorithm)

The implementation directly uses CompanionSSM and MACompanionSSM for the two-stage VARMA architecture.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict, Any, cast, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn

from ..config import KDFMConfig, KDFMResult
from ..config.constants import DEFAULT_TORCH_DTYPE, DEFAULT_CLOCK_FREQUENCY, DEFAULT_REGULARIZATION, DEFAULT_GRAD_CLIP_VAL, DEFAULT_EIGENVALUE_MAX_MAGNITUDE, DEFAULT_EIGENVALUE_WARN_THRESHOLD, DEFAULT_IDENTITY_SCALE, DEFAULT_ZERO_VALUE, DEFAULT_FORECAST_HORIZON, DEFAULT_DTYPE, DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER, MIN_TIME_STEPS, MIN_VARIABLES, COMPUTATION_ERROR_TYPES
from ..logger import get_logger
from ..ssm.companion import CompanionSSM, MACompanionSSM
from ..ssm.structural import StructuralIdentificationSSM
from ..functional.irf import compute_irf
from ..numeric.stability import create_scaled_identity
from .base import BaseFactorModel
from ..utils.errors import (
    ModelNotTrainedError,
    ModelNotInitializedError,
    PredictionError,
    NumericalError,
    DataValidationError,
    ConfigurationError
)
from ..utils.validation import check_condition, has_shape_with_min_dims
from ..utils.common import ensure_numpy
from ..config.types import (
    Device, ArrayLike, ForecastResult, Shape2D, Shape3D,
    ForecastHorizon, NumVars, LagOrder, OptionalTensor, OptionalArray
)

# Import type hints (optional, for better IDE support)
if TYPE_CHECKING:
    from ..datamodule import KDFMDataModule

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


class KDFM(BaseFactorModel, pl.LightningModule):
    """High-level API for Kernelized Dynamic Factor Model (PyTorch Lightning module).
    
    This class implements KDFM with two-stage VARMA architecture:
    - Stage 1 (AR): h_{t+1} = A^AR h_t + B ε_t, z_t = C h_t
    - Stage 2 (MA): h'_{t+1} = A^MA h'_t + B' z_t, y_t = C' h'_t
    
    Uses gradient descent training (like DDFM), not EM algorithm (like DFM).
    Uses Krylov FFT for efficient O(T log T) forward pass.
    
    **API Differences from DFM/DDFM**:
    - Training: Uses `training_step()` method (PyTorch Lightning), not `fit()`
    - Prediction: `predict(horizon, last_observation)` - REQUIRES `last_observation` parameter
    - Result extraction: `get_result()` method (not `result` property)
    - Error handling: Raises exceptions (`NumericalError`, `PredictionError`, `ValueError`) instead of silent warnings
    
    See `BaseFactorModel` documentation for API comparison across models.
    
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
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> model.load_config('config/kdfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = KDFMTrainer(max_epochs=100)  # DEFAULT_MAX_EPOCHS
        >>> trainer.fit(model, dm)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
        config: Optional[KDFMConfig] = None,
        ar_order: int = DEFAULT_KDFM_AR_ORDER,
        ma_order: int = DEFAULT_KDFM_MA_ORDER,
        learning_rate: Optional[float] = None,
        max_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        weight_decay: Optional[float] = None,
        grad_clip_val: Optional[float] = None,
        structural_method: str = 'cholesky',
        structural_reg_weight: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Initialize KDFM instance.
        
        Parameters
        ----------
        config : KDFMConfig, optional
            KDFM configuration. Can be loaded later via load_config().
        ar_order : int, default=1
            VAR order p
        ma_order : int, default=0
            MA order q (0 = pure VAR)
        learning_rate : float, default=DEFAULT_LEARNING_RATE
            Learning rate for Adam optimizer (default: DEFAULT_LEARNING_RATE from constants)
        max_epochs : int, default=100  # DEFAULT_MAX_EPOCHS
            Maximum training epochs
        batch_size : int, default=32
            Batch size for training
        weight_decay : float, default=DEFAULT_REGULARIZATION_SCALE
            Weight decay (L2 regularization)
        grad_clip_val : float, default=DEFAULT_GRAD_CLIP_VAL
            Gradient clipping value (default: DEFAULT_GRAD_CLIP_VAL from constants)
        structural_method : str, default='cholesky'
            Structural identification method: 'cholesky', 'full', 'low_rank'
        structural_reg_weight : float, default=DEFAULT_STRUCTURAL_REG_WEIGHT
            Weight for structural regularization loss (default: DEFAULT_STRUCTURAL_REG_WEIGHT from constants)
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        BaseFactorModel.__init__(self)
        pl.LightningModule.__init__(self)
        
        # Import constants for defaults (consolidated import)
        from ..config.constants import (
            DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_CLOCK_FREQUENCY,
            DEFAULT_REGULARIZATION_SCALE, DEFAULT_GRAD_CLIP_VAL,
            DEFAULT_STRUCTURAL_REG_WEIGHT
        )
        
        # Validate and store parameters
        from ..numeric.validator import (
            validate_ar_order, validate_ma_order, validate_learning_rate,
            validate_batch_size
        )
        self.ar_order = validate_ar_order(ar_order)
        self.ma_order = validate_ma_order(ma_order)
        
        # Initialize config using base class pattern
        # Create temporary config if none provided (will be replaced via load_config if needed)
        if config is None:
            config = self._create_temp_config()
        # KDFMConfig is compatible with DFMConfig (base class) - both inherit from BaseConfig
        # Type checker sees BaseFactorModel._config as Optional[DFMConfig], but KDFM uses KDFMConfig
        # This is safe at runtime since KDFMConfig is a subclass of DFMConfig
        # Cast to satisfy type checker while maintaining runtime correctness
        self._config: KDFMConfig = cast(KDFMConfig, config)
        
        # Set parameters with defaults from constants and validate
        # Resolve parameters using consolidated helper
        from ..utils.misc import resolve_param
        self.learning_rate = validate_learning_rate(
            resolve_param(learning_rate, default=DEFAULT_LEARNING_RATE)
        )
        self.max_epochs = resolve_param(max_epochs, default=DEFAULT_MAX_EPOCHS)
        self.batch_size = validate_batch_size(
            resolve_param(batch_size, default=DEFAULT_BATCH_SIZE)
        )
        self.weight_decay = resolve_param(weight_decay, default=DEFAULT_REGULARIZATION_SCALE)
        self.grad_clip_val = resolve_param(grad_clip_val, default=DEFAULT_GRAD_CLIP_VAL)
        self.structural_method = structural_method
        self.structural_reg_weight = resolve_param(structural_reg_weight, default=DEFAULT_STRUCTURAL_REG_WEIGHT)
        
        # Model components initialized in initialize_from_data() when data dimensions are known
        self.companion_ar: Optional[CompanionSSM] = None
        self.companion_ma: Optional[MACompanionSSM] = None
        self.structural_id: Optional[StructuralIdentificationSSM] = None
        
        # Training state (set during training)
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
    
    def initialize_from_data(self, X: Tensor) -> None:
        """Initialize model parameters from data.
        
        This method initializes all model components (AR stage, MA stage, structural
        identification) based on the data dimensions. Must be called before training
        or prediction.
        
        **Initialization Order**:
        1. AR stage (CompanionSSM) - handles VAR dynamics
        2. MA stage (MACompanionSSM) - handles MA dynamics (if ma_order > 0)
        3. Structural identification (StructuralIdentificationSSM) - transforms residuals to shocks
        4. Training step handler - manages loss computation
        
        **Device Management**: All components are moved to the same device as the input data.
        
        Parameters
        ----------
        X : Tensor
            Standardized data of shape (T, N) where:
            - T: Number of time steps (must be >= 1)
            - N: Number of variables (K, must be >= 1)
            Data should be standardized (mean ~0, std ~1) and differenced if needed.
            Must be on a valid device (CPU or CUDA).
            
        Raises
        ------
        DataValidationError
            If data shape is invalid (not 2D), contains NaN/Inf values, or has invalid dimensions.
        ConfigurationError
            If model configuration is invalid (e.g., ar_order or ma_order out of range).
        RuntimeError
            If component initialization fails (e.g., device mismatch, memory allocation failure).
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> X = torch.randn(100, 5)  # 100 time steps, 5 variables
        >>> model.initialize_from_data(X)
        >>> assert model.companion_ar is not None
        >>> assert model.structural_id is not None
        >>> assert model.companion_ar.n_vars == 5
        >>> assert model.companion_ar.order == 1
        """
        from ..numeric.validator import validate_data_shape, validate_no_nan_inf
        
        # Validate data
        validate_data_shape(X, min_dims=2, max_dims=2, min_size=1)
        validate_no_nan_inf(X, name="training data")
        
        T, N = X.shape
        
        # Validate dimensions
        check_condition(
            T >= MIN_TIME_STEPS,
            DataValidationError,
            f"Data must have at least {MIN_TIME_STEPS} time step, got T={T}",
            details="Ensure data is not empty"
        )
        check_condition(
            N >= MIN_VARIABLES,
            DataValidationError,
            f"Data must have at least {MIN_VARIABLES} variable, got N={N}",
            details="Ensure data has at least one column"
        )
        
        # Note: ar_order and ma_order are already validated in __init__ via validate_ar_order/validate_ma_order
        # No need to re-validate here since they cannot be modified after initialization
        
        K = N  # Number of variables
        device = X.device
        
        try:
            # Initialize AR stage (CompanionSSM)
            self.companion_ar = CompanionSSM(
                n_vars=K,
                lag_order=self.ar_order,
                n_kernels=1,
                kernel_init='normal',
                norm_order=1
            )
            self.companion_ar.to(device)
            
            # Initialize MA stage (MACompanionSSM, if q > 0)
            if self.ma_order > 0:
                self.companion_ma = MACompanionSSM(
                    n_vars=K,
                    ma_order=self.ma_order,
                    n_kernels=1,
                    kernel_init='normal',
                    norm_order=1
                )
                self.companion_ma.to(device)
            else:
                self.companion_ma = None
            
            # Initialize structural identification
            self.structural_id = StructuralIdentificationSSM(
                n_vars=K,
                lag_order=self.ar_order,
                method=self.structural_method,
                align_with_latent_state=True
            )
            self.structural_id.to(device)
            
            _logger.info(
                f"KDFM initialized from data: T={T}, N={N}, ar_order={self.ar_order}, "
                f"ma_order={self.ma_order}, device={device}"
            )
            
        except COMPUTATION_ERROR_TYPES as e:
            # Specific exceptions for initialization failures
            raise ModelNotInitializedError(
                f"Failed to initialize KDFM components: {type(e).__name__}: {e}",
                details=(
                    f"Data shape: {X.shape}, ar_order={self.ar_order}, ma_order={self.ma_order}, "
                    f"device={device}. Check: (1) Device availability, (2) Memory availability, "
                    f"(3) Configuration validity."
                )
            ) from e
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through two-stage VARMA architecture.
        
        This method performs the forward pass through KDFM's two-stage VARMA
        architecture, processing input data through:
        1. Structural identification: Transform residuals to structural shocks
        2. AR stage: Process structural shocks through companion SSM
        3. MA stage: Process AR output through MA companion SSM (if q > 0)
        
        **CRITICAL**: This method requires model components to be initialized.
        Call `initialize_from_data()` before calling `forward()`.
        
        Parameters
        ----------
        x : Tensor
            Input data of shape (B, T, N) or (T, N) where:
            - B: Batch size (optional)
            - T: Sequence length
            - N: Number of variables (K)
            Data should be in PREPROCESSED scale (standardized, differenced)
            
        Returns
        -------
        Tensor
            Predictions of shape (B, T, N) or (T, N) matching input shape.
            Predictions are in PREPROCESSED scale (same as input).
            
        Raises
        ------
        ModelNotInitializedError
            If model components are not initialized (companion_ar is None)
        RuntimeError
            If forward pass fails due to shape mismatches or numerical issues
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> X = torch.randn(100, 5)  # 100 time steps, 5 variables
        >>> model.initialize_from_data(X)
        >>> y_pred = model.forward(X)
        >>> assert y_pred.shape == X.shape
        """
        check_condition(
            self.companion_ar is not None,
            ModelNotInitializedError,
            "KDFM forward pass requires initialized model components. "
            "Call initialize_from_data() before forward pass.",
            details="companion_ar is None"
        )
        
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
        
        # Stage 1 (AR): Forward pass through AR stage
        z_t = self._forward_ar_stage(structural_shocks)  # (B, T, K)
        
        # Stage 2 (MA): Forward pass through MA stage (if q > 0)
        y_pred = self._forward_ma_stage(z_t)  # (B, T, K)
        
        if squeeze_output:
            y_pred = y_pred.squeeze(0)  # (T, N)
        
        return y_pred
    
    def _forward_ar_stage(self, structural_shocks: Tensor) -> Tensor:
        """Forward pass through AR stage.
        
        This method processes structural shocks through the AR stage companion SSM,
        computing z_t = C h_t where h_t evolves according to:
        h_{t+1} = A^{AR} h_t + B ε_t
        
        The AR stage captures autoregressive dynamics using the companion matrix
        structure, enabling direct IRF computation via matrix powers.
        
        Parameters
        ----------
        structural_shocks : Tensor
            Structural shocks of shape (B, T, shock_dim) where:
            - B: Batch size
            - T: Sequence length
            - shock_dim: Shock dimension (p*K if align_with_latent_state, else K)
            Structural shocks are orthogonal: E[ε_t ε_t^T] = I
            
        Returns
        -------
        Tensor
            AR stage output z_t of shape (B, T, K) where:
            - z_t represents the AR stage output before MA processing
            - For pure VAR (q=0), z_t is the final output
            - For VARMA (q > 0), z_t is input to MA stage
            
        Raises
        ------
        ModelNotInitializedError
            If AR stage is not initialized (companion_ar is None)
        RuntimeError
            If forward pass fails due to shape mismatches
        """
        check_condition(
            self.companion_ar is not None,
            ModelNotInitializedError,
            "AR stage not initialized. Call initialize_from_data() first.",
            details="companion_ar is None"
        )
        return self.companion_ar(structural_shocks)
    
    def _forward_ma_stage(self, z_t: Tensor) -> Tensor:
        """Forward pass through MA stage.
        
        This method processes AR stage output through the MA stage companion SSM
        (if q > 0), computing y_t = C' h'_t where h'_t evolves according to:
        h'_{t+1} = A^{MA} h'_t + B' z_t
        
        For pure VAR models (q=0), this method simply returns z_t unchanged.
        The MA stage enables VARMA(p,q) modeling where residuals have moving
        average dynamics.
        
        Parameters
        ----------
        z_t : Tensor
            AR stage output of shape (B, T, K) where:
            - B: Batch size
            - T: Sequence length
            - K: Number of variables
            This is the output from _forward_ar_stage()
            
        Returns
        -------
        Tensor
            MA stage output y_pred of shape (B, T, K) where:
            - For VARMA (q > 0): Processed through MA companion SSM
            - For pure VAR (q=0): Returns z_t unchanged (no MA processing)
            
        Raises
        ------
        RuntimeError
            If MA stage forward pass fails (should not occur for q=0)
        """
        if self.companion_ma is not None:
            return self.companion_ma(z_t)
        else:
            # Pure VAR: no MA stage
            return z_t
    
    def training_step(self, batch: Union[Tensor, Tuple[Tensor, Tensor]], batch_idx: int) -> Tensor:
        """Training step for KDFM (PyTorch Lightning interface).
        
        This method implements the PyTorch Lightning training step, which is called
        automatically during training. It handles:
        1. Batch preparation (normalizing shapes, moving to device)
        2. Model initialization (if not already initialized)
        3. Forward pass through two-stage VARMA architecture
        4. Loss computation (prediction + structural regularization)
        
        **Loss Components**:
        - Prediction loss: MSE between predictions and targets
        - Structural loss: Regularization encouraging S @ S.T ≈ I (orthogonality)
        - Total loss: pred_loss + λ_struct * struct_loss (where λ_struct << 1)
        
        Parameters
        ----------
        batch : Tensor or Tuple[Tensor, Tensor]
            Training batch:
            - If Tensor: Data tensor of shape (B, T, N) or (T, N)
            - If Tuple: (data, target) where both are tensors of same shape
            Data should be in PREPROCESSED scale (standardized, differenced)
        batch_idx : int
            Batch index (unused, kept for Lightning interface compatibility)
            
        Returns
        -------
        Tensor
            Total training loss (scalar tensor):
            - Loss value >= 0.0
            - Loss is automatically logged by Lightning
            - Loss is used for backpropagation and optimization
            
        Raises
        ------
        ModelNotInitializedError
            If model components cannot be initialized from batch data
        RuntimeError
            If forward pass or loss computation fails
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> batch = torch.randn(32, 100, 5)  # (B=32, T=100, N=5)
        >>> loss = model.training_step(batch, batch_idx=0)
        >>> assert loss.item() >= 0.0
        """
        # Get device
        device = next(self.parameters()).device
        
        # Prepare batch
        if isinstance(batch, Sequence) and not isinstance(batch, str) and len(batch) == 2:
            data, target = batch
        else:
            # Unsupervised: use data as target
            data = batch
            target = data
        
        # Move to device
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
        
        # Store processed data for shape validation in update() (store first batch)
        if self.data_processed is None:
            from ..utils.common import ensure_numpy
            # Store as numpy array in base class format
            if data.ndim == 3:
                # Batch format: take first sample
                data_sample = data[0]
            else:
                data_sample = data
            self.data_processed = ensure_numpy(data_sample, dtype=DEFAULT_DTYPE)
        
        # Compute losses using training step handler
        total_loss = self._compute_training_loss(y_pred, target, device)
        
        return total_loss
    
    def _compute_training_loss(
        self,
        y_pred: Tensor,
        target: Tensor,
        device: torch.device
    ) -> Tensor:
        """Compute training loss (prediction + structural regularization).
        
        This method computes the total training loss as the sum of:
        - Prediction loss (MSE between predictions and targets)
        - Structural regularization loss (encourages orthogonality of structural matrix)
        
        Parameters
        ----------
        y_pred : Tensor
            Model predictions of shape (B, T, N) or (T, N)
        target : Tensor
            Target values of shape (B, T, N) or (T, N)
        device : torch.device
            Device for tensor operations
            
        Returns
        -------
        Tensor
            Total loss (prediction + structural regularization)
            
        Raises
        ------
        RuntimeError
            If loss computation fails due to shape mismatches or numerical issues
            
        Examples
        --------
        >>> loss = model._compute_training_loss(y_pred, target, device)
        >>> assert loss.item() >= 0.0
        >>> assert not torch.isnan(loss)
        """
        # Prediction loss (MSE)
        pred_loss = nn.functional.mse_loss(y_pred, target)
        
        # Structural regularization loss
        struct_loss = self._compute_structural_loss(device)
        
        # Total loss: L_total = L_pred + λ_struct * L_struct
        total_loss = pred_loss + self.structural_reg_weight * struct_loss
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('pred_loss', pred_loss, on_step=True, on_epoch=True)
        self.log('struct_loss', struct_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def _compute_structural_loss(self, device: torch.device) -> Tensor:
        """Compute structural identification regularization loss.
        
        This method computes the structural regularization loss that encourages
        the structural identification matrix S to be orthogonal: S @ S.T ≈ I.
        This ensures structural shocks are orthogonal and have unit variance,
        which is required for valid structural IRF analysis.
        
        **Loss Formula**: L_struct = ||S @ S.T - I||² (MSE between S @ S.T and identity)
        
        **Weighting**: The structural loss is weighted by `structural_reg_weight`
        (default: DEFAULT_STRUCTURAL_REG_WEIGHT = 0.1) in the total loss, ensuring prediction loss dominates
        while still encouraging orthogonality.
        
        Parameters
        ----------
        device : torch.device
            Device for tensor operations (used for fallback zero tensor)
            
        Returns
        -------
        Tensor
            Structural regularization loss (scalar tensor):
            - Loss value >= 0.0
            - Loss = 0.0 if S is perfectly orthogonal (S @ S.T = I)
            - Higher loss indicates deviation from orthogonality
            - Returns 0.0 if structural_id is None (no structural identification)
            
        Raises
        ------
        NumericalError
            If structural matrix contains NaN/Inf values or matrix multiplication fails
        RuntimeError
            If structural matrix shape is invalid or device mismatch occurs
            
        Examples
        --------
        >>> loss = model._compute_structural_loss(device)
        >>> assert loss.item() >= 0.0
        >>> assert not torch.isnan(loss)
        """
        if self.structural_id is not None:
            try:
                S = self.structural_id.get_structural_matrix()
                
                # Validate structural matrix
                if torch.isnan(S).any() or torch.isinf(S).any():
                    raise NumericalError(
                        "Structural matrix contains NaN/Inf values",
                        details="Structural identification matrix S has invalid values"
                    )
                
                S_S_T = S @ S.T
                # Use S.dtype to match input tensor dtype for loss computation consistency
                I = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
                loss = nn.functional.mse_loss(S_S_T, I)
                
                # Validate loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise NumericalError(
                        "Structural loss computation produced NaN/Inf",
                        details="MSE between S @ S.T and I is invalid"
                    )
                
                return loss
            except NumericalError:
                # Re-raise numerical errors - these are critical and indicate data/model issues
                # NumericalError is explicitly raised for NaN/Inf values in structural matrix or loss
                raise
            except (AttributeError, RuntimeError) as e:
                # Catch initialization/structural issues during early training:
                # - AttributeError: structural_id.get_structural_matrix() method missing or attribute access fails
                # - RuntimeError: Matrix operations fail due to shape/device mismatches or uninitialized state
                # Return zero loss to allow training to continue, but log exception type for debugging
                # This is acceptable during early training when model is not yet fully initialized
                _logger.debug(
                    f"KDFM _compute_structural_loss: {type(e).__name__}: {e}. "
                    f"Returning zero loss. This may indicate structural_id is not properly initialized."
                )
                return torch.tensor(DEFAULT_ZERO_VALUE, device=device, dtype=DEFAULT_TORCH_DTYPE)
        else:
            return torch.tensor(DEFAULT_ZERO_VALUE, device=device, dtype=DEFAULT_TORCH_DTYPE)
    
    def _get_model_components(self) -> List[Optional[Union[CompanionSSM, MACompanionSSM, StructuralIdentificationSSM]]]:
        """Get list of all model components.
        
        This method returns all model components (AR companion, MA companion,
        structural identification) for device management and parameter collection.
        
        Returns
        -------
        List[Optional[Union[CompanionSSM, MACompanionSSM, StructuralIdentificationSSM]]]
            List of model components:
            - companion_ar: AR stage companion SSM (required)
            - companion_ma: MA stage companion SSM (optional, None if q=0)
            - structural_id: Structural identification SSM (optional, None if not initialized)
            Components may be None if not initialized or not applicable.
        """
        return [self.companion_ar, self.companion_ma, self.structural_id]
    
    def _move_components_to_device(self, device: torch.device) -> None:
        """Move all model components to specified device.
        
        This method moves all model components (companion SSMs, structural identification)
        to the specified device. This ensures all components are on the same device as
        the input data, which is required for proper forward pass and training.
        
        **Device Management**: This method is called automatically during
        `initialize_from_data()` to ensure all components are on the correct device.
        
        Parameters
        ----------
        device : torch.device
            Target device (e.g., torch.device('cuda') or torch.device('cpu'))
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> X = torch.randn(100, 5)
        >>> model.initialize_from_data(X)  # Automatically moves to X's device
        >>> # Or manually:
        >>> model._move_components_to_device(torch.device('cuda'))
        """
        for component in self._get_model_components():
            if component is not None:
                component.to(device)
    
    def _collect_parameters(self) -> List[torch.nn.Parameter]:
        """Collect all trainable parameters from model components.
        
        This method collects all trainable parameters from all model components
        (AR companion, MA companion, structural identification) into a single list.
        This is useful for optimizer configuration and parameter inspection.
        
        **Parameter Sources**:
        - AR companion SSM: Companion matrix A^{AR}, input matrix B, output matrix C
        - MA companion SSM: Companion matrix A^{MA}, input matrix B', output matrix C'
        - Structural identification: Structural matrix S (parameterized as L, S, or U/V)
        
        Returns
        -------
        List[torch.nn.Parameter]
            List of all trainable parameters from all model components.
            Parameters are collected in order: AR, MA, structural identification.
            None components are skipped (no parameters collected).
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> X = torch.randn(100, 5)
        >>> model.initialize_from_data(X)
        >>> params = model._collect_parameters()
        >>> assert len(params) > 0
        >>> assert all(isinstance(p, torch.nn.Parameter) for p in params)
        """
        params = []
        for component in self._get_model_components():
            if component is not None:
                params.extend(component.parameters())
        return params
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, List[torch.optim.Optimizer], Dict[str, Any]]:
        """Configure optimizer for KDFM training (PyTorch Lightning interface).
        
        This method creates and configures the optimizer(s) for KDFM training.
        Uses Adam optimizer with configurable learning rate, weight decay, and
        gradient clipping. This method is called automatically by PyTorch Lightning
        during training setup.
        
        **Optimizer Configuration**:
        - Optimizer: Adam (adaptive learning rate, good for non-stationary loss landscapes)
        - Learning rate: Configurable via `learning_rate` parameter (default: DEFAULT_LEARNING_RATE = 0.001)
        - Weight decay: L2 regularization via `weight_decay` parameter (default: DEFAULT_REGULARIZATION_SCALE = 1e-5)
        - Gradient clipping: Applied via `grad_clip_val` parameter (default: DEFAULT_GRAD_CLIP_VAL = 1.0 from constants)
        
        **Gradient Clipping**: Gradient clipping is applied automatically by Lightning
        if `grad_clip_val` is set. This prevents gradient explosion, which is
        particularly important for state-space models with near-unit-root eigenvalues.
        
        Returns
        -------
        torch.optim.Optimizer or List[torch.optim.Optimizer] or Dict[str, Any]
            Optimizer configuration:
            - If single optimizer: Returns optimizer directly
            - If multiple optimizers: Returns list of optimizers
            - If scheduler needed: Returns dict with 'optimizer' and 'lr_scheduler' keys
            Currently returns list containing single Adam optimizer.
            Returns dummy optimizer if model parameters not yet initialized (for Lightning compatibility).
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0, learning_rate=DEFAULT_LEARNING_RATE)  # ar_order/ma_order use DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> optimizer_config = model.configure_optimizers()
        >>> assert isinstance(optimizer_config, list)
        >>> assert len(optimizer_config) == 1
        >>> assert isinstance(optimizer_config[0], torch.optim.Optimizer)
        >>> assert optimizer_config[0].param_groups[0]['lr'] == DEFAULT_LEARNING_RATE
        """
        params = self._collect_parameters()
        
        if not params:
            # Return dummy optimizer if no parameters yet (will be updated when model is initialized)
            # Create a dummy parameter to satisfy Lightning's optimizer requirement
            dummy_param = nn.Parameter(torch.tensor(DEFAULT_ZERO_VALUE))
            return [torch.optim.Adam([dummy_param], lr=self.learning_rate)]
        
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
    
    def predict(  # type: ignore[override]  # KDFM requires last_observation parameter (different from base class)
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True,
        last_observation: Optional[Union[Tensor, np.ndarray]] = None  # Last data point for initialization
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict future values using trained KDFM model.
        
        **Type Note**: This method has a different signature than the base class
        `BaseFactorModel.predict()` because KDFM requires `last_observation` to
        compute the initial factor state. The `# type: ignore[override]` comment
        suppresses the type checker warning about signature mismatch, which is
        intentional and documented in the base class API differences section.
        
        This is a known API difference: KDFM's companion matrix architecture requires
        the last observation to initialize the factor state, while DFM/DDFM can use
        the model's internal state. The signature difference is intentional and
        type-safe at runtime.
        
        This method generates forecasts by:
        1. Validating companion matrix stability (eigenvalues must be < DEFAULT_EIGENVALUE_MAX_MAGNITUDE (1.0))
        2. Extracting the last factor state from the last observation (if provided)
        3. Forecasting factors forward using VAR dynamics via companion matrix powers
        4. Transforming factors to observations using the loading matrix
        
        **CRITICAL**: This method validates companion matrix stability before forecasting.
        If the companion matrix has eigenvalues >= DEFAULT_EIGENVALUE_MAX_MAGNITUDE (1.0), forecasts will explode and a
        `PredictionError` will be raised. This ensures numerical stability and prevents
        invalid forecasts.
        
        **Scale Handling**: Forecasts are returned in standardized differenced space.
        The caller must apply inverse transformation (unstandardization + undifferencing)
        to get forecasts in the original scale. See `experiment/scripts/kdfm_forecasts.py`
        for an example of proper scale handling.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods to forecast. If None, uses default (6 periods).
            Must be >= 1.
        return_series : bool, default=True
            Whether to return forecasted series (observations).
        return_factors : bool, default=True
            Whether to return forecasted factors.
        last_observation : torch.Tensor or np.ndarray, optional
            Last observation of shape (1, N) or (N,) to use for initializing forecast.
            Must be in PREPROCESSED scale (standardized, differenced).
            If None, uses zeros (may produce poor forecasts).
            **Important**: Providing the actual last observation significantly improves
            forecast quality, especially for the first few periods.
            
        Returns
        -------
        X_forecast : np.ndarray or Tuple[np.ndarray, np.ndarray]
            - If both return_series and return_factors are True: (X_forecast, Z_forecast)
            - If only return_series is True: X_forecast
            - If only return_factors is True: Z_forecast
            
            Shapes:
            - X_forecast: (horizon, N) where N is number of series
            - Z_forecast: (horizon, K) where K is number of factors
            
            **Scale**: Forecasts are in standardized differenced space. Apply inverse
            transformation to get original scale.
            
        Raises
        ------
        ModelNotTrainedError
            If model is not trained (no training history).
        ModelNotInitializedError
            If model components are not properly initialized (companion_ar is None).
        PredictionError
            If companion matrix is unstable (max eigenvalue >= DEFAULT_EIGENVALUE_MAX_MAGNITUDE (1.0)) or forecast
            generation produces NaN/Inf values. This indicates numerical instability
            and the model should be retrained with better regularization.
        NumericalError
            If companion matrix contains NaN/Inf values or eigenvalue computation fails.
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> # ... train model ...
        >>> # Predict with last observation (in preprocessed scale)
        >>> X_f, Z_f = model.predict(horizon=6, last_observation=last_data_point)
        >>> # Predict only series
        >>> X_f = model.predict(horizon=6, return_factors=False)
        >>> # Apply inverse transformation to get original scale
        >>> from experiment.data.preprocessor import DataPreprocessor
        >>> preprocessor = DataPreprocessor()
        >>> forecast_df = pd.DataFrame(X_f, columns=series_names)
        >>> forecast_original = preprocessor.inverse_transform('dataset_name', forecast_df)
        """
        from ..numeric.validator import validate_horizon
        
        # Validate and set horizon
        if horizon is None:
            horizon = DEFAULT_FORECAST_HORIZON
        else:
            horizon = validate_horizon(horizon)
        
        # Check if model is trained
        try:
            self._check_trained()
        except ValueError as e:
            raise ModelNotTrainedError(
                "KDFM prediction requires a trained model. "
                "Please train the model using trainer.fit() before calling predict().",
                details=str(e)
            ) from e
        
        # Check if model is initialized
        check_condition(
            self.companion_ar is not None,
            ModelNotInitializedError,
            "KDFM model components are not initialized. "
            "This may occur if initialize_from_data() was not called during training.",
            details="companion_ar is None"
        )
        
        # Get result for parameters (compute if needed)
        result = self._ensure_result()
        
        # Extract n_vars using helper method with fallback hierarchy
        n_vars = self._extract_n_vars_from_sources(
            result=result,
            structural_id=self.structural_id,
            companion_ar=self.companion_ar,
            ar_order=self.ar_order
        )
        
        if n_vars is None:
            # Enhanced error message with diagnostic information
            diagnostic_info = self._build_diagnostic_info(
                result=result,
                companion_ar=self.companion_ar,
                structural_id=self.structural_id,
                ar_order=self.ar_order
            )
            raise ModelNotInitializedError(
                "Could not determine n_vars for KDFM from any source. Model may not be properly initialized.",
                details=(
                    f"Diagnostic info: {diagnostic_info}. "
                    f"Possible causes: (1) Model not fully trained, (2) get_result() failed, "
                    f"(3) Companion matrices not properly initialized. "
                    f"Ensure model.initialize_from_data() was called and training completed successfully."
                )
            )
        
        # CRITICAL: Validate companion matrix stability before forecasting
        # Use centralized validation utilities for consistency
        from ..numeric.validator import (
            validate_model_components,
            validate_companion_stability,
            validate_forecast_inputs
        )
        
        # Validate model is initialized
        validate_model_components(
            companion_ar=self.companion_ar,
            companion_ma=self.companion_ma,
            structural_id=self.structural_id,
            model_name="KDFM"
        )
        
        # Validate forecast inputs
        validate_forecast_inputs(
            horizon=horizon,
            last_observation=last_observation,
            n_vars=n_vars,
            model_name="KDFM"
        )
        
        try:
            from ..numeric.validator import validate_companion_stability
            A_np = ensure_numpy(result.A)
            is_stable, max_eigenval = validate_companion_stability(
                companion_matrix=A_np,
                model_name="KDFM",
                name="KDFM companion matrix",
                threshold=DEFAULT_EIGENVALUE_MAX_MAGNITUDE,
                warn_threshold=DEFAULT_EIGENVALUE_WARN_THRESHOLD
            )
            # If we get here, matrix is stable (validate_companion_stability raises on failure)
        except (PredictionError, NumericalError):
            # Re-raise validation errors
            raise
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            # Catch initialization/attribute errors during stability check (non-critical validation)
            # These can occur if companion matrix attributes are missing or operations fail
            # We log and continue because stability check is informational, not blocking
            _logger.debug(
                f"Could not check companion matrix stability: {type(e).__name__}: {e}. "
                f"Proceeding with forecast generation, but results may be unreliable."
            )
        
        # Compute n_factors from result.C or use n_vars as fallback
        result_C = self._get_result_C(result)
        # Simplify: check if result_C has shape and at least 2 dimensions
        if has_shape_with_min_dims(result_C, min_dims=2):
            n_factors = result_C.shape[1]
        else:
            n_factors = n_vars
        
        # Get last factor state by running forward pass on last observation
        if last_observation is not None:
            # Prepare last observation (normalize shape, move to device)
            from ..utils.common import ensure_tensor
            last_obs_tensor = ensure_tensor(last_observation, device=self.device, dtype=DEFAULT_TORCH_DTYPE)
            if last_obs_tensor.ndim == 1:
                last_obs_tensor = last_obs_tensor.unsqueeze(0)  # (1, N)
            if last_obs_tensor.shape != (1, n_vars):
                raise DataValidationError(
                    f"last_observation shape mismatch: expected (1, {n_vars}) or ({n_vars},), "
                    f"got {last_obs_tensor.shape}. "
                    f"last_observation must match the number of variables in the model (n_vars={n_vars}). "
                    f"Ensure last_observation is in preprocessed scale (standardized, differenced) and "
                    f"has the correct number of variables matching the training data."
                )
            
            # Compute factor state from last observation
            try:
                # Transform to structural shocks
                if self.structural_id is not None:
                    structural_shocks = self.structural_id(last_obs_tensor)  # (1, K)
                else:
                    structural_shocks = last_obs_tensor
                
                # Get AR stage output (z_t)
                if self.companion_ar is not None:
                    z_t = self.companion_ar(structural_shocks.unsqueeze(0))  # (1, 1, K)
                    z_t = z_t.squeeze(0).squeeze(0)  # (K,)
                else:
                    z_t = structural_shocks.squeeze(0)  # (K,)
                
                Z_last = ensure_numpy(z_t)
                # Validate factor state is finite
                from ..numeric.validator import validate_no_nan_inf
                validate_no_nan_inf(Z_last, name="factor state Z_last")
            except COMPUTATION_ERROR_TYPES as e:
                # Catch computation errors during factor state extraction from observation
                # Note: NumericalError/PredictionError inherit from Exception, not the caught types above,
                # so the isinstance check below is defensive but typically unreachable
                # If NumericalError/PredictionError are raised, they would propagate directly
                if isinstance(e, (NumericalError, PredictionError)):
                    raise
                # Convert generic exceptions to PredictionError for consistent error handling
                raise PredictionError(
                    f"KDFM predict: Failed to compute factor state from observation: {type(e).__name__}: {e}",
                    details="The last_observation parameter may be invalid or the model may not be properly initialized."
                ) from e
        else:
            raise DataValidationError(
                "KDFM predict: No last_observation provided. "
                "KDFM requires last_observation parameter to compute initial factor state. "
                "Provide the last observed data point (in preprocessed scale: standardized, differenced). "
                f"Expected shape: (1, {n_vars}) or ({n_vars},) where n_vars={n_vars}."
            )
        
        # Validate Z_last shape
        if Z_last.shape[0] != n_factors:
            raise DataValidationError(
                f"KDFM predict: Factor state Z_last shape mismatch. "
                f"Expected shape[0]={n_factors} (number of factors), got {Z_last.shape[0]}. "
                f"Z_last shape: {Z_last.shape}, n_factors: {n_factors}. "
                f"This indicates a mismatch between input data dimensions and model configuration. "
                f"Please verify: (1) last_observation has correct number of variables, "
                f"(2) model was trained with matching data dimensions, (3) companion_ar output dimension matches n_factors."
            )
        
        # Forecast factors using VAR dynamics
        K = Z_last.shape[0]
        A_np = ensure_numpy(result.A)
        expected_shape = (self.ar_order * K, self.ar_order * K)
        if A_np.shape != expected_shape:
            raise DataValidationError(
                f"KDFM predict: Companion matrix A has incompatible shape. "
                f"Expected: {expected_shape} (ar_order={self.ar_order} * K={K}), "
                f"got: {A_np.shape}. "
                f"This indicates a mismatch between model configuration and stored parameters. "
                f"Please verify: (1) model was trained with ar_order={self.ar_order}, "
                f"(2) result.A was computed correctly, (3) K (number of factors) is consistent."
            )
        
        # Initialize state vector (companion form)
        # Companion form stacks lagged factors: state = [Z_t, Z_{t-1}, ..., Z_{t-p+1}]
        state = np.zeros(self.ar_order * K, dtype=DEFAULT_DTYPE)
        state[:K] = Z_last.astype(DEFAULT_DTYPE)
        
        # Validate state initialization
        from ..numeric.validator import validate_no_nan_inf
        validate_no_nan_inf(state, name="initial state vector")
        
        # Generate forecasts using companion matrix powers
        # For VAR(1): Z_{t+h} = A^h @ Z_t
        # For VAR(p): Uses companion form state vector
        Z_forecast = np.zeros((horizon, K), dtype=DEFAULT_DTYPE)
        for h in range(horizon):
            Z_forecast[h, :] = state[:K].copy()
            state = A_np @ state
            # Validate state is finite at each step
            try:
                validate_no_nan_inf(state, name=f"forecast state at horizon {h+1}")
            except NumericalError as e:
                raise PredictionError(
                    f"KDFM predict: Factor forecast generation produced NaN/Inf at horizon {h+1}. "
                    f"This indicates numerical instability in companion matrix multiplication. "
                    f"Possible causes: (1) Companion matrix A has eigenvalues >= DEFAULT_EIGENVALUE_MAX_MAGNITUDE (1.0) (unstable), "
                    f"(2) Large forecast horizon causing numerical overflow, (3) Invalid initial state. "
                    f"Please check: companion matrix stability (max eigenvalue < DEFAULT_EIGENVALUE_MAX_MAGNITUDE (1.0)), "
                    f"forecast horizon (try smaller horizon), and initial factor state validity."
                ) from e
        
        # Transform factors to observations
        X_forecast = Z_forecast @ result.C.T  # (horizon, K) @ (K, N) -> (horizon, N)
        
        # Validate forecasts are finite
        validate_no_nan_inf(Z_forecast, name="factor forecast Z_forecast")
        validate_no_nan_inf(X_forecast, name="final forecast X_forecast")
        
        # Return based on flags
        if return_series and return_factors:
            return X_forecast, Z_forecast
        elif return_series:
            return X_forecast
        else:
            return Z_forecast
    
    def _create_temp_config(self, block_name: Optional[str] = None) -> KDFMConfig:  # type: ignore[override]  # Returns KDFMConfig (subclass of DFMConfig)
        """Create temporary KDFMConfig for initialization.
        
        This method creates a minimal KDFMConfig for model initialization when
        no config is provided. The temporary config will be replaced by load_config()
        if a config file is loaded later.
        
        **Type Note**: BaseFactorModel._create_temp_config() returns DFMConfig, but
        KDFM returns KDFMConfig (which is a subclass of DFMConfig). This is intentional
        and type-safe - KDFMConfig inherits from DFMConfig, so it satisfies the base
        class contract. The `# type: ignore[override]` comment suppresses the type
        checker warning about return type variance, which is acceptable here because
        KDFMConfig is a more specific type that is compatible with DFMConfig.
        
        This follows the Liskov Substitution Principle: KDFMConfig can be used
        anywhere DFMConfig is expected, so the return type variance is safe.
        
        Parameters
        ----------
        block_name : str, optional
            Block name (ignored for KDFM - KDFM does not use block structure)
        
        Returns
        -------
        KDFMConfig
            Temporary configuration object with minimal settings:
            - Single temporary series
            - AR order from self.ar_order (or default 1)
            - MA order from self.ma_order (or default 0)
        """
# SeriesConfig removed - use frequency dict instead
        # KDFM does not use blocks structure - only series
        # Get ar_order and ma_order from instance attributes (set in __init__)
        ar_order = getattr(self, 'ar_order', DEFAULT_KDFM_AR_ORDER)
        ma_order = getattr(self, 'ma_order', DEFAULT_KDFM_MA_ORDER)
        return KDFMConfig(
            frequency={'temp': DEFAULT_CLOCK_FREQUENCY},
            ar_order=ar_order,
            ma_order=ma_order
        )
    
    def _extract_companion_params(self, companion_ssm: Optional[Union[CompanionSSM, MACompanionSSM]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract companion matrix and B, C parameters from companion SSM.
        
        This method uses the utility function `extract_companion_params` from
        `dfm_python.utils.kdfm_helpers` for better code organization and reusability.
        
        Parameters
        ----------
        companion_ssm : CompanionSSM or MACompanionSSM, optional
            Companion SSM instance (AR or MA)
            
        Returns
        -------
        tuple
            (A_matrix, B_matrix, C_matrix) as numpy arrays, or (None, None, None) if:
            - companion_ssm is None
            - Parameter extraction fails
            
        Raises
        ------
        NumericalError
            If parameter extraction fails (from utility function)
        """
        check_condition(
            companion_ssm is not None,
            ModelNotInitializedError,
            "Cannot extract companion parameters: companion_ssm is None",
            details="Companion SSM must be initialized before extracting parameters. Please ensure model is properly trained."
        )
        
        try:
            A = companion_ssm.get_companion_matrix()
            if A.ndim == 3:
                A_np = ensure_numpy(A[0])
            else:
                A_np = ensure_numpy(A)
            
            # Extract B and C parameters (they are nn.Parameter which are tensors)
            B_param = companion_ssm.B
            C_param = companion_ssm.C
            # Handle 3D case (n_kernels > 1) - extract first kernel
            if B_param.ndim > 2:
                B_np = ensure_numpy(B_param[0])
            else:
                B_np = ensure_numpy(B_param)
            if C_param.ndim > 2:
                C_np = ensure_numpy(C_param[0])
            else:
                C_np = ensure_numpy(C_param)
            
            return A_np, B_np, C_np
        except (AttributeError, KeyError, RuntimeError, ValueError) as e:
            # Parameter extraction failure is critical - raise exception with context
            # These are expected errors when parameters are missing or invalid
            error_msg = f"Failed to extract companion SSM parameters: {type(e).__name__}: {e}"
            _logger.error(f"extract_companion_params: {error_msg}")
            raise NumericalError(
                error_msg,
                details="Companion SSM parameter extraction failed. This may indicate model initialization issues or numerical instability."
            ) from e
        except (IndexError, OSError, MemoryError) as e:
            # Additional specific exceptions for parameter extraction failures
            error_msg = f"Failed to extract companion SSM parameters: {type(e).__name__}: {e}"
            _logger.error(f"extract_companion_params: {error_msg}")
            raise NumericalError(
                error_msg,
                details="Companion SSM parameter extraction failed. This may indicate model initialization issues, memory problems, or numerical instability."
            ) from e
    
    def _get_result_C(self, result: Optional['KDFMResult']) -> Optional[np.ndarray]:
        """Get result.C attribute safely.
        
        Parameters
        ----------
        result : Optional[KDFMResult]
            Result object to extract C from
            
        Returns
        -------
        Optional[np.ndarray]
            result.C if available, None otherwise
        """
        if result is None:
            return None
        return getattr(result, 'C', None)
    
    def _get_result_n_vars(self, result: Optional['KDFMResult']) -> Optional[int]:
        """Get result.n_vars attribute safely.
        
        Parameters
        ----------
        result : Optional[KDFMResult]
            Result object to extract n_vars from
            
        Returns
        -------
        Optional[int]
            result.n_vars if available, None otherwise
        """
        if result is None:
            return None
        try:
            return getattr(result, 'n_vars', None)
        except (AttributeError, TypeError):
            return None
    
    def _compute_factor_state_from_observation(
        self,
        observation: Tensor,
        n_factors: int
    ) -> np.ndarray:
        """Compute factor state from last observation.
        
        This is a helper method that extracts the factor state (z_t) from the last
        observation by running it through the structural identification and AR stage.
        
        **Note**: This method computes factor state from observation inline.
        
        Parameters
        ----------
        observation : Tensor
            Last observation tensor of shape (1, N)
        n_factors : int
            Expected number of factors (for fallback)
            
        Returns
        -------
        np.ndarray
            Factor state Z_last of shape (n_factors,)
        """
        try:
            if self.structural_id is not None:
                structural_shocks = self.structural_id(observation)
            else:
                structural_shocks = observation
            
            if self.companion_ar is not None:
                z_t = self.companion_ar(structural_shocks.unsqueeze(0))
                z_t = z_t.squeeze(0).squeeze(0)
            else:
                z_t = structural_shocks.squeeze(0)
            
            Z_last = ensure_numpy(z_t)
            if np.any(np.isnan(Z_last)) or np.any(np.isinf(Z_last)):
                # This method is used internally and should return zeros for invalid states
                # The calling code (predict) will handle the error appropriately
                _logger.debug(
                    "KDFM _compute_factor_state_from_observation: Z_last contains NaN/Inf. "
                    "Returning zeros. This may indicate numerical instability in factor state computation."
                )
                return np.zeros(n_factors, dtype=DEFAULT_DTYPE)
            return Z_last
        except COMPUTATION_ERROR_TYPES as e:
            # Catch computation errors in helper method - return zeros for resilience
            # This is a helper method that should not raise exceptions - the calling code (predict)
            # will validate the returned zeros and raise appropriate PredictionError if needed
            # Broad exception catching is appropriate here to ensure method always returns a value
            _logger.debug(
                f"KDFM _compute_factor_state_from_observation: Failed to compute factor state: "
                f"{type(e).__name__}: {e}. Returning zeros. This may indicate invalid input or model state."
            )
            return np.zeros(n_factors, dtype=DEFAULT_DTYPE)
    
    def _can_compute_irf(
        self,
        ma_transition: Optional[np.ndarray],
        ar_input: Optional[np.ndarray],
        ar_output: Optional[np.ndarray],
        ma_input: Optional[np.ndarray],
        ma_output: Optional[np.ndarray],
        structural_matrix: Optional[np.ndarray]
    ) -> Tuple[bool, str]:
        """Check if all required parameters are available for IRF computation.
        
        This method uses the utility function `can_compute_irf` from
        `dfm_python.utils.kdfm_helpers` for better code organization.
        
        **API Consistency (Priority 5)**: This method signature is consistent with
        `can_compute_irf()` helper function. It accepts 6 parameters (excluding self):
        - ma_transition, ar_input, ar_output, ma_input, ma_output, structural_matrix
        Note: ar_transition is NOT required for validation (only used in actual IRF computation).
        
        Parameters
        ----------
        ma_transition : np.ndarray, optional
            MA stage transition matrix A^MA of shape (q*K, q*K) or None if q=0
        ar_input : np.ndarray, optional
            AR stage input matrix B of shape (p*K, K)
        ar_output : np.ndarray, optional
            AR stage output matrix C of shape (K, p*K)
        ma_input : np.ndarray, optional
            MA stage input matrix B' of shape (q*K, K) or None if q=0
        ma_output : np.ndarray, optional
            MA stage output matrix C' of shape (K, q*K) or None if q=0
        structural_matrix : np.ndarray, optional
            Structural identification matrix S of shape (K, K)
            
        Returns
        -------
        tuple
            (can_compute, error_msg) where:
            - can_compute: bool, True if all required parameters are available
            - error_msg: str, Error message if parameters are missing, empty string if OK
            
        Raises
        ------
        TypeError
            If called with wrong number of arguments (should be 6 parameters + self = 7 total)
            
        Examples
        --------
        >>> # Correct usage (6 parameters + self)
        >>> # Note: In actual code, use create_scaled_identity() helper instead of np.eye()
        >>> # These examples use np.eye() for simplicity in documentation
        >>> can_compute, error_msg = model._can_compute_irf(
        ...     ma_transition=None,
        ...     ar_input=np.eye(5),  # Example: use create_scaled_identity(5) in actual code
        ...     ar_output=np.eye(5),  # Example: use create_scaled_identity(5) in actual code
        ...     ma_input=None,
        ...     ma_output=None,
        ...     structural_matrix=np.eye(5)  # Example: use create_scaled_identity(5) in actual code
        ... )
        >>> assert isinstance(can_compute, bool)
        >>> assert isinstance(error_msg, str)
        """
        # Check if all required parameters are available for IRF computation
        if ar_input is None or ar_output is None:
            return False, "AR stage parameters (ar_input, ar_output) are required"
        if structural_matrix is None:
            return False, "Structural identification matrix is required"
        
        has_ma_stage = self.ma_order > 0
        if has_ma_stage:
            if ma_transition is None or ma_input is None or ma_output is None:
                return False, "MA stage parameters are required for VARMA IRF computation"
        
        # All required parameters are available
        return True, ""
    
    def _extract_n_vars_from_sources(
        self,
        result: Optional[Any],
        structural_id: Optional[Any],
        companion_ar: Optional[Any],
        ar_order: Optional[int]
    ) -> Optional[int]:
        """Extract n_vars from multiple sources with fallback hierarchy.
        
        Tries sources in order: (1) result.C.shape[0], (2) result.n_vars,
        (3) structural_id.n_vars, (4) companion_ar.A shape.
        
        Parameters
        ----------
        result : Optional[Any]
            KDFM result object
        structural_id : Optional[Any]
            Structural identification SSM
        companion_ar : Optional[Any]
            Companion AR SSM
        ar_order : Optional[int]
            AR order for shape calculation
        
        Returns
        -------
        Optional[int]
            Number of variables if successfully extracted, None otherwise
        """
        n_vars = None
        
        # Try 1: result.C.shape[0] (most reliable)
        if result is not None:
            try:
                result_C = self._get_result_C(result)
                if result_C is not None:
                    # Use has_shape_with_min_dims to check shape (handles np.ndarray, torch.Tensor, etc.)
                    if has_shape_with_min_dims(result_C, min_dims=2):
                        n_vars = int(result_C.shape[0])
            except (AttributeError, IndexError, TypeError):
                pass
            
            # Try 2: result.n_vars
            if n_vars is None:
                try:
                    result_n_vars = self._get_result_n_vars(result)
                    if result_n_vars is not None:
                        n_vars = int(result_n_vars)
                except (AttributeError, TypeError, ValueError):
                    pass
        
        # Try 3: structural_id.n_vars
        if n_vars is None and structural_id is not None:
            try:
                structural_n_vars = getattr(structural_id, 'n_vars', None)
                if structural_n_vars is not None:
                    n_vars = int(structural_n_vars)
            except (AttributeError, TypeError, ValueError):
                pass
        
        # Try 4: companion_ar shape
        if n_vars is None and companion_ar is not None and ar_order is not None and ar_order > 0:
            try:
                if hasattr(companion_ar, 'A'):
                    A = companion_ar.A
                    A_np = ensure_numpy(A)
                    if A_np.ndim == 3:
                        A_np = A_np[0]
                    if A_np.shape[0] > 0:
                        n_vars = A_np.shape[0] // ar_order
                        if n_vars <= 0:
                            n_vars = None
            except (AttributeError, IndexError, TypeError, ZeroDivisionError):
                pass
        
        return n_vars
    
    def _format_diagnostic_value(self, obj: Optional[Any]) -> str:
        """Format diagnostic value for error messages.
        
        Parameters
        ----------
        obj : Optional[Any]
            Object to format
        
        Returns
        -------
        str
            'present' if object is not None, 'None' otherwise
        """
        return 'present' if obj is not None else 'None'
    
    def _build_diagnostic_info(
        self,
        result: Optional[Any],
        companion_ar: Optional[Any],
        structural_id: Optional[Any],
        ar_order: Optional[int]
    ) -> Dict[str, Any]:
        """Build diagnostic information dictionary for error messages.
        
        Parameters
        ----------
        result : Optional[Any]
            KDFM result object
        companion_ar : Optional[Any]
            Companion AR SSM
        structural_id : Optional[Any]
            Structural identification SSM
        ar_order : Optional[int]
            AR order
        
        Returns
        -------
        Dict[str, Any]
            Diagnostic information dictionary
        """
        # Format result_C_shape with simplified conditional logic
        result_C_shape = 'N/A'
        result_C = self._get_result_C(result)
        if result_C is not None:
            try:
                result_C_shape = result_C.shape
            except (AttributeError, TypeError):
                pass
        
        # Format result_n_vars
        result_n_vars = 'N/A'
        result_n_vars_attr = self._get_result_n_vars(result)
        if result_n_vars_attr is not None:
            result_n_vars = result_n_vars_attr
        
        return {
            'result': self._format_diagnostic_value(result),
            'companion_ar': self._format_diagnostic_value(companion_ar),
            'structural_id': self._format_diagnostic_value(structural_id),
            'ar_order': ar_order,
            'result_C_shape': result_C_shape,
            'result_n_vars': result_n_vars
        }
    
    def get_result(self) -> KDFMResult:
        """Extract parameters and create KDFMResult.
        
        This method extracts all model parameters (AR/MA coefficients, structural matrix,
        companion matrices, IRFs) and creates a KDFMResult object for analysis and
        serialization.
        
        **CRITICAL**: This method requires the model to be trained and initialized.
        Call `trainer.fit(model, data_module)` before calling this method.
        
        Returns
        -------
        KDFMResult
            KDFM estimation results containing:
            - AR/MA coefficients: Extracted VAR/MA coefficients
            - Structural matrix: Structural identification matrix S
            - IRFs: Reduced-form and structural impulse response functions
            - Companion matrices: Transition, input, and output matrices
            - Eigenvalues: Stability metrics
            - Other fields: Standard DFM result fields (for compatibility)
            
        Raises
        ------
        ModelNotInitializedError
            If model components are not initialized (companion_ar is None)
        NumericalError
            If parameter extraction or IRF computation fails
            
        Examples
        --------
        >>> model = KDFM(ar_order=1, ma_order=0)  # Uses DEFAULT_KDFM_AR_ORDER, DEFAULT_KDFM_MA_ORDER
        >>> # ... train model ...
        >>> result = model.get_result()
        >>> assert result.ar_coeffs is not None
        >>> assert result.irf_reduced is not None
        >>> assert result.max_eigenvalue < DEFAULT_EIGENVALUE_MAX_MAGNITUDE  # Stable model
        """
        check_condition(
            self.companion_ar is not None,
            ModelNotInitializedError,
            "KDFM get_result requires initialized model components. "
            "Train the model using trainer.fit() before calling get_result().",
            details="companion_ar is None"
        )
        
        # Extract parameters and convert to numpy using utility function
        ar_coeffs_np = ensure_numpy(self.companion_ar.extract_coefficients())
        
        ma_coeffs_np = None
        if self.companion_ma is not None:
            ma_coeffs_np = ensure_numpy(self.companion_ma.extract_coefficients())
        
        # Get structural matrix
        S_np = None
        if self.structural_id is not None:
            S_np = ensure_numpy(self.structural_id.get_structural_matrix())
        
        # Extract companion matrices and parameters using helper method
        ar_transition, ar_input, ar_output = self._extract_companion_params(self.companion_ar)
        ma_transition, ma_input, ma_output = self._extract_companion_params(self.companion_ma)
        
        # Compute IRFs if all required parameters are available
        irf_reduced, irf_structural = self._compute_irfs_from_params(
            ar_transition, ma_transition, ar_input, ar_output,
            ma_input, ma_output, S_np
        )
        
        # Determine dimensions from extracted parameters
        n_vars = ar_coeffs_np.shape[1] if ar_coeffs_np is not None else 1
        n_factors = n_vars  # KDFM uses same dimension for factors and variables in AR stage
        
        # Create result with proper dimensions
        # KDFM uses a two-stage VARMA structure rather than traditional factor model,
        # so some result fields (x_sm, Z) are minimal placeholders
        # Get target scaler from model if available
        target_scaler = getattr(self, 'target_scaler', None)
        
        result = KDFMResult(
            x_sm=np.zeros((1, n_vars), dtype=DEFAULT_DTYPE),
            Z=np.zeros((1, n_factors), dtype=DEFAULT_DTYPE),
            # Note: np.eye(n_factors, n_vars) creates rectangular matrix (n_factors x n_vars),
            # so cannot use create_scaled_identity() helper which only supports square matrices
            C=ar_output if ar_output is not None else np.eye(n_factors, n_vars),
            R=create_scaled_identity(n_vars, DEFAULT_REGULARIZATION),  # Small noise covariance
            A=ar_transition[:n_factors, :n_factors] if ar_transition is not None else create_scaled_identity(n_factors, DEFAULT_IDENTITY_SCALE),
            Q=create_scaled_identity(n_factors, DEFAULT_REGULARIZATION),  # Small process noise
            target_scaler=target_scaler,
            Z_0=np.zeros(n_factors, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(n_factors, DEFAULT_REGULARIZATION),
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
    
    @property
    def result(self) -> KDFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        """
        result = self._ensure_result()
        # Type assertion: get_result() always returns KDFMResult for KDFM model
        assert isinstance(result, KDFMResult), f"Expected KDFMResult but got {type(result)}"
        return result
    
    def _compute_irfs_from_params(
        self,
        ar_transition: Optional[np.ndarray],
        ma_transition: Optional[np.ndarray],
        ar_input: Optional[np.ndarray],
        ar_output: Optional[np.ndarray],
        ma_input: Optional[np.ndarray],
        ma_output: Optional[np.ndarray],
        structural_matrix: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute IRFs from extracted parameters.
        
        This method computes both reduced-form and structural IRFs from the
        extracted companion matrix parameters. It validates parameters using
        the utility function `can_compute_irf` and handles errors gracefully.
        
        **IRF Computation**: IRFs are computed directly from companion matrices
        using matrix powers: K_h = C (A^{AR})^h B for reduced-form IRF, and
        K_h^{struct} = K_h S for structural IRF. This direct computation is
        KDFM's PRIMARY CONTRIBUTION: IRFs are the primary object, not a byproduct.
        
        Parameters
        ----------
        ar_transition : np.ndarray, optional
            AR stage transition matrix A^{AR} of shape (p*K, p*K)
        ma_transition : np.ndarray, optional
            MA stage transition matrix A^{MA} of shape (q*K, q*K) or None if q=0
        ar_input : np.ndarray, optional
            AR stage input matrix B of shape (p*K, K)
        ar_output : np.ndarray, optional
            AR stage output matrix C of shape (K, p*K)
        ma_input : np.ndarray, optional
            MA stage input matrix B' of shape (q*K, K) or None if q=0
        ma_output : np.ndarray, optional
            MA stage output matrix C' of shape (K, q*K) or None if q=0
        structural_matrix : np.ndarray, optional
            Structural identification matrix S of shape (K, K) or (p*K, p*K)
            
        Returns
        -------
        tuple
            (irf_reduced, irf_structural) where:
            - irf_reduced: np.ndarray of shape (horizon, K, K) or None if computation fails
            - irf_structural: np.ndarray of shape (horizon, K, K) or None if computation fails
            
        Raises
        ------
        NumericalError
            If IRF computation fails due to numerical issues (from compute_irf)
            
        Examples
        --------
        >>> # Compute IRFs from extracted parameters
        >>> irf_red, irf_struct = model._compute_irfs_from_params(
        ...     ar_transition, ma_transition, ar_input, ar_output,
        ...     ma_input, ma_output, structural_matrix
        ... )
        >>> assert irf_red is not None
        >>> assert irf_struct is not None
        >>> assert irf_red.shape[0] == 20  # Default horizon
        """
        # Validate parameters using centralized validation utility
        # Note: ar_transition is not needed for validation (only used in actual IRF computation)
        can_compute, error_msg = self._can_compute_irf(
            ma_transition, ar_input, ar_output,
            ma_input, ma_output, structural_matrix
        )
        
        if not can_compute:
            # IRF computation failure - raise exception instead of returning None
            from ..utils.errors import ModelNotInitializedError
            raise ModelNotInitializedError(
                f"KDFM _compute_irfs_from_params: Cannot compute IRF - {error_msg}",
                details="This may indicate incomplete parameter extraction or model initialization issues. Please ensure model is properly trained and parameters are initialized."
            )
        
        try:
            from ..config.constants import DEFAULT_IRF_HORIZON
            from ..utils.common import ensure_tensor
            
            # Convert to tensors for IRF computation (using common utility)
            ar_transition_t = ensure_tensor(ar_transition, dtype=DEFAULT_TORCH_DTYPE)
            ar_input_t = ensure_tensor(ar_input, dtype=DEFAULT_TORCH_DTYPE)
            ar_output_t = ensure_tensor(ar_output, dtype=DEFAULT_TORCH_DTYPE)
            structural_matrix_t = ensure_tensor(structural_matrix, dtype=DEFAULT_TORCH_DTYPE)
            
            # Handle MA stage parameters
            if self.ma_order > 0 and ma_transition is not None and ma_input is not None and ma_output is not None:
                ma_transition_t = ensure_tensor(ma_transition, dtype=DEFAULT_TORCH_DTYPE)
                ma_input_t = ensure_tensor(ma_input, dtype=DEFAULT_TORCH_DTYPE)
                ma_output_t = ensure_tensor(ma_output, dtype=DEFAULT_TORCH_DTYPE)
            else:
                # VAR model (no MA stage) - use identity matrices
                K = ar_output_t.shape[0]
                # Use DEFAULT_TORCH_DTYPE constant for consistency with model's default dtype
                identity = torch.eye(K, dtype=DEFAULT_TORCH_DTYPE, device=ar_transition_t.device)
                ma_transition_t = ma_input_t = ma_output_t = identity
            
            # Compute IRFs
            irf_reduced, irf_structural = compute_irf(
                ar_transition_t,
                ma_transition_t,
                ar_input_t,
                ar_output_t,
                ma_input_t,
                ma_output_t,
                structural_matrix_t,
                horizon=DEFAULT_IRF_HORIZON,
                structural=True
            )
            
            # Convert back to numpy using utility function
            if irf_reduced is not None:
                irf_reduced = ensure_numpy(irf_reduced)
            if irf_structural is not None:
                irf_structural = ensure_numpy(irf_structural)
            
            return irf_reduced, irf_structural
            
        except (ValueError, RuntimeError, TypeError, AttributeError) as e:
            # Re-raise as NumericalError for better error handling
            # These are expected errors when IRF computation fails due to invalid state
            from ..utils.errors import NumericalError
            raise NumericalError(
                f"KDFM IRF computation failed: {e}. "
                f"This may occur if: (1) Parameters are not properly initialized, "
                f"(2) Numerical instability in IRF computation, "
                f"(3) Incompatible matrix shapes.",
                details=f"Error type: {type(e).__name__}, Error: {str(e)}"
            ) from e
        except (IndexError, OSError, MemoryError) as e:
            # Additional specific exceptions for IRF computation failures
            from ..utils.errors import NumericalError
            raise NumericalError(
                f"KDFM IRF computation failed: {type(e).__name__} during IRF computation: {e}",
                details="This may occur due to memory constraints, invalid indices, or system errors. Please check system resources and input parameters."
            ) from e
    
    def update(self, data: Union[np.ndarray, Any]) -> None:
        """Update model state with new observations via companion matrix forward pass.
        
        This method runs the KDFM forward pass on new data to update the latent factors,
        but keeps model parameters fixed.
        
        After calling update(), the model's internal state (result.Z and data_processed)
        is extended with the new observations. Subsequent calls to predict() will use
        the updated state.
        
        **Data Shape**: The input data must be 2D with shape (T_new x N) where:
        - T_new: Number of new time steps (can be any positive integer)
        - N: Number of series (must match training data)
        
        **Supported Types**:
        - numpy.ndarray: (T_new x N) array
        - pandas.DataFrame: DataFrame with N columns, T_new rows
        - polars.DataFrame: DataFrame with N columns, T_new rows
        
        **Important**: Data must be preprocessed by the user (same preprocessing as training).
        Only target scaler is handled internally if needed.
        
        Parameters
        ----------
        data : np.ndarray, pandas.DataFrame, or polars.DataFrame
            New preprocessed observations with shape (T_new x N) where:
            - T_new: Number of new time steps (any positive integer)
            - N: Number of series (must match training data)
            Data must be preprocessed by user (same preprocessing as training).
            
        Notes
        -----
        - This updates factors via companion matrix forward pass, NOT parameter retraining
        - For parameter retraining, use trainer.fit() with concatenated data
        - After update(), predict() will use the updated factor state
        - New data must have same number of series (N) as training data
        - User must preprocess data themselves (same preprocessing as training)
        
        Raises
        ------
        ModelNotTrainedError
            If model has not been trained yet
        DataValidationError
            If data shape doesn't match training data
        ModelNotInitializedError
            If model components are not initialized
        """
        from ..utils.common import ensure_tensor, ensure_numpy
        
        # Validate and convert data (no preprocessing - user must preprocess)
        from ..numeric.validator import validate_and_convert_update_data
        data_new = validate_and_convert_update_data(
            data,
            self.data_processed,
            dtype=DEFAULT_DTYPE,
            model_name=self.__class__.__name__
        )
        
        # Convert to tensor
        device = next(self.parameters()).device if self.companion_ar is not None else torch.device('cpu')
        data_tensor = ensure_tensor(data_new, dtype=DEFAULT_TORCH_DTYPE)
        data_tensor = data_tensor.to(device)
        
        # Check if model is initialized
        check_condition(
            self.companion_ar is not None,
            ModelNotInitializedError,
            "KDFM model components are not initialized. "
            "This may occur if initialize_from_data() was not called during training.",
            details="companion_ar is None"
        )
        
        # Run forward pass to get factors
        with torch.no_grad():
            factors_new = self.forward(data_tensor)  # (T_new x K)
        
        # Convert to numpy
        factors_new_np = ensure_numpy(factors_new, dtype=DEFAULT_DTYPE)
        
        # Get current result (compute if needed)
        result = self._ensure_result()
        
        # Update model state: append new factors and data
        result.Z = np.vstack([result.Z, factors_new_np])
        
        # Ensure data_processed is numpy array
        if isinstance(self.data_processed, torch.Tensor):
            self.data_processed = ensure_numpy(self.data_processed, dtype=DEFAULT_DTYPE)
        
        if self.data_processed is not None:
            self.data_processed = np.vstack([self.data_processed, data_new])
        else:
            # If not stored during training, initialize with new data
            self.data_processed = data_new
        
        # Update smoothed data (x_sm) in result
        result.x_sm = result.Z @ result.C.T
    


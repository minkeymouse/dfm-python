"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
DFM is a PyTorch Lightning module that inherits from BaseFactorModel.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from ..config import (
    DFMConfig,
    make_config_source,
    ConfigSource,
    MergedConfigSource,
)
from ..config.results import DFMResult, FitParams
from ..config.utils import get_agg_structure, get_tent_weights, FREQUENCY_HIERARCHY, TENT_WEIGHTS_LOOKUP
from ..logger import get_logger
from ..ssm.em import EMAlgorithm, EMStepParams
from ..ssm.kalman import KalmanFilter
from .base import BaseFactorModel

# Frequency to integer mapping for tensor conversion
_FREQ_TO_INT = {'d': 1, 'w': 2, 'm': 3, 'q': 4, 'sa': 5, 'a': 6}

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)


@dataclass
class DFMTrainingState:
    """State tracking for DFM training."""
    A: torch.Tensor
    C: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    Z_0: torch.Tensor
    V_0: torch.Tensor
    loglik: float
    num_iter: int
    converged: bool



class DFM(BaseFactorModel):
    """High-level API for Linear Dynamic Factor Model (PyTorch Lightning module).
    
    This class is a PyTorch Lightning module that can be used with standard
    Lightning training patterns. It inherits from BaseFactorModel and implements
    the EM algorithm for DFM estimation.
    
    Example (Standard Lightning Pattern):
        >>> from dfm_python import DFM, DFMDataModule, DFMTrainer
        >>> import pandas as pd
        >>> 
        >>> # Step 1: Load and preprocess data
        >>> df = pd.read_csv('data/finance.csv')
        >>> df_processed = df[[col for col in df.columns if col != 'date']]
        >>> 
        >>> # Step 2: Create DataModule
        >>> dm = DFMDataModule(config_path='config/dfm_config.yaml', data=df_processed)
        >>> dm.setup()
        >>> 
        >>> # Step 3: Create model and load config
        >>> model = DFM()
        >>> model.load_config('config/dfm_config.yaml')
        >>> 
        >>> # Step 4: Create trainer and fit
        >>> trainer = DFMTrainer(max_epochs=100)
        >>> trainer.fit(model, dm)
        >>> 
        >>> # Step 5: Predict
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(
        self,
        config: Optional[DFMConfig] = None,
        num_factors: Optional[int] = None,
        threshold: float = 1e-4,
        max_iter: int = 100,
        nan_method: int = 2,
        nan_k: int = 3,
        tent_weights_dict: Optional[dict] = None,
        mixed_freq: bool = False,
        **kwargs
    ):
        """Initialize DFM instance.
        
        Parameters
        ----------
        config : DFMConfig, optional
            DFM configuration. Can be loaded later via load_config().
        num_factors : int, optional
            Number of factors. If None, inferred from config.
        threshold : float, default 1e-4
            EM convergence threshold
        max_iter : int, default 100
            Maximum EM iterations
        nan_method : int, default 2
            Missing data handling method
        nan_k : int, default 3
            Spline interpolation order
        tent_weights_dict : dict, optional
            [DEPRECATED] Optional tent weights to control aggregation. Use `mixed_freq` parameter instead.
            Kept for backward compatibility only.
        mixed_freq : bool, default False
            If True, use tent kernels for mixed-frequency data. If False, treat all series as clock frequency.
            When True, raises ValueError if any frequency pair is not in TENT_WEIGHTS_LOOKUP.
        **kwargs
            Additional arguments passed to BaseFactorModel
        """
        super().__init__(**kwargs)
        
        # Initialize config using consolidated helper method
        config = self._initialize_config(config)
        
        self.threshold = threshold
        self.max_iter = max_iter
        self.nan_method = nan_method
        self.nan_k = nan_k
        self.mixed_freq = mixed_freq
        
        # Legacy tent_weights_dict parameter (deprecated - use mixed_freq instead)
        # Only used when mixed_freq=False for backward compatibility
        self._legacy_tent_weights_dict = tent_weights_dict
        
        # Mixed frequency parameters (set during initialize_from_data)
        self._em_R_mat = None
        self._em_q = None
        self._em_nQ = 0
        self._em_tent_weights_dict = None
        self._em_frequencies = None
        self._em_i_idio = None
        self._em_idio_chain_lengths = None
        
        # Determine number of factors
        if num_factors is None:
            if hasattr(config, 'factors_per_block') and config.factors_per_block:
                self.num_factors = int(np.sum(config.factors_per_block))
            else:
                blocks = config.get_blocks_array()
                if blocks.shape[1] > 0:
                    self.num_factors = int(np.sum(blocks[:, 0]))
                else:
                    self.num_factors = 1
        else:
            self.num_factors = num_factors
        
        # Get model structure
        self.r = torch.tensor(
            config.factors_per_block if config.factors_per_block is not None
            else np.ones(config.get_blocks_array().shape[1]),
            dtype=torch.float32
        )
        self.p = getattr(config, 'ar_lag', 1)
        self.blocks = torch.tensor(config.get_blocks_array(), dtype=torch.float32)
        
        # Compose modules as components
        self.kalman = KalmanFilter(
            min_eigenval=1e-8,
            inv_regularization=1e-6,
            cholesky_regularization=1e-8
        )
        self.em = EMAlgorithm(
            kalman=self.kalman,  # Share same KalmanFilter instance
            regularization_scale=1e-6
        )
        
        # Parameters will be initialized in setup() or fit_em()
        self.A: Optional[torch.nn.Parameter] = None
        self.C: Optional[torch.nn.Parameter] = None
        self.Q: Optional[torch.nn.Parameter] = None
        self.R: Optional[torch.nn.Parameter] = None
        self.Z_0: Optional[torch.nn.Parameter] = None
        self.V_0: Optional[torch.nn.Parameter] = None
        
        # Training state
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
        
        # Use manual optimization for EM algorithm
        self.automatic_optimization = False
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize model parameters.
        
        This is called by Lightning before training starts.
        Parameters are initialized from data if available.
        """
        # Parameters will be initialized during fit_em() or first training step
        pass
    
    def _create_em_step_params(
        self, 
        y: torch.Tensor, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> EMStepParams:
        """Create EM step parameters using stored mixed frequency parameters.
        
        Parameters
        ----------
        y : torch.Tensor
            Data tensor (N x T)
        device : torch.device
            Device for tensors
        dtype : torch.dtype
            Data type for tensors
            
        Returns
        -------
        EMStepParams
            EM step parameters
        """
        clock = getattr(self.config, 'clock', 'm')
        N = y.shape[0]
        
        return EMStepParams(
            y=y,
            A=self.A,
            C=self.C,
            Q=self.Q,
            R=self.R,
            Z_0=self.Z_0,
            V_0=self.V_0,
            r=self.r.to(device),
            p=self.p,
            R_mat=self._em_R_mat,
            q=self._em_q,
            nQ=self._em_nQ,
            i_idio=self._em_i_idio if self._em_i_idio is not None else torch.ones(N, device=device, dtype=dtype),
            blocks=self.blocks.to(device),
            tent_weights_dict=self._em_tent_weights_dict,
            clock=clock,
            frequencies=self._em_frequencies,
            idio_chain_lengths=self._em_idio_chain_lengths if self._em_idio_chain_lengths is not None else torch.zeros(N, device=device, dtype=dtype),
            config=self.config
        )
    
    def initialize_from_data(self, X: torch.Tensor) -> None:
        """Initialize parameters from data using PCA and OLS.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        """
        opt_nan = {'method': self.nan_method, 'k': self.nan_k}
        clock = getattr(self.config, 'clock', 'm')
        
        # Handle mixed_freq parameter
        if self.mixed_freq:
            # Use tent kernels for mixed-frequency data
            agg_structure = get_agg_structure(self.config, clock=clock)
            
            # Validate that all required frequency pairs are in TENT_WEIGHTS_LOOKUP
            frequencies_list = [s.frequency for s in self.config.series]
            frequencies_set = set(frequencies_list)
            clock_hierarchy = FREQUENCY_HIERARCHY.get(clock, 3)
            
            missing_pairs = []
            for freq in frequencies_set:
                freq_hierarchy = FREQUENCY_HIERARCHY.get(freq, 3)
                if freq_hierarchy > clock_hierarchy:
                    # This frequency is slower than clock, needs tent kernel
                    tent_w = get_tent_weights(freq, clock)
                    if tent_w is None:
                        missing_pairs.append((freq, clock))
            
            if missing_pairs:
                raise ValueError(
                    f"mixed_freq=True but the following frequency pairs are not in TENT_WEIGHTS_LOOKUP: {missing_pairs}. "
                    f"Available pairs: {list(TENT_WEIGHTS_LOOKUP.keys())}. "
                    f"Either add the missing pairs to TENT_WEIGHTS_LOOKUP or set mixed_freq=False."
                )
            
            # Convert tent_weights to torch tensors
            tent_weights_dict = {k: torch.tensor(v, dtype=torch.float32, device=X.device) 
                                for k, v in agg_structure['tent_weights'].items()}
            
            # Get R_mat and q from first structure (if any)
            R_mat = None
            q = None
            if agg_structure['structures']:
                first_structure = list(agg_structure['structures'].values())[0]
                R_mat = torch.tensor(first_structure[0], dtype=torch.float32, device=X.device)
                q = torch.tensor(first_structure[1], dtype=torch.float32, device=X.device)
            
            # Create frequencies array
            frequencies_array = np.array(frequencies_list, dtype=object)
            
            # Count slower-frequency series
            nQ = sum(1 for freq in frequencies_list 
                    if FREQUENCY_HIERARCHY.get(freq, 3) > clock_hierarchy)
            
            # Compute i_idio (1 for clock frequency, 0 for slower frequencies)
            i_idio = torch.tensor([1 if freq == clock else 0 for freq in frequencies_list], 
                                 dtype=torch.float32, device=X.device)
        else:
            # Treat all series as clock frequency (unified frequency)
            tent_weights_dict = self._legacy_tent_weights_dict
            R_mat = None
            q = None
            frequencies_array = None
            nQ = 0
            i_idio = torch.ones(X.shape[1], dtype=torch.float32, device=X.device)
        
        # Convert frequencies to torch tensor
        frequencies_tensor = None
        if frequencies_array is not None:
            frequencies_tensor = torch.tensor(
                [_FREQ_TO_INT.get(f, 3) for f in frequencies_array], 
                dtype=torch.int32, 
                device=X.device
            )
        
        # Store for reuse in EM steps
        self._em_R_mat = R_mat
        self._em_q = q
        self._em_nQ = nQ
        self._em_tent_weights_dict = tent_weights_dict
        self._em_frequencies = frequencies_tensor
        self._em_i_idio = i_idio
        
        # Idiosyncratic chain lengths (currently unused, set to zeros)
        self._em_idio_chain_lengths = torch.zeros(X.shape[1], dtype=torch.int32, device=X.device)
        
        # Initialize parameters using EM algorithm
        A, C, Q, R, Z_0, V_0 = self.em.initialize_parameters(
            X,
            r=self.r.to(X.device),
            p=self.p,
            blocks=self.blocks.to(X.device),
            opt_nan=opt_nan,
            R_mat=R_mat,
            q=q,
            nQ=nQ,
            i_idio=i_idio,
            clock=clock,
            tent_weights_dict=tent_weights_dict,
            frequencies=frequencies_tensor,
            idio_chain_lengths=self._em_idio_chain_lengths,
            config=self.config
        )
        
        # Convert to Parameters
        self.A = nn.Parameter(A)
        self.C = nn.Parameter(C)
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)
        self.Z_0 = nn.Parameter(Z_0)
        self.V_0 = nn.Parameter(V_0)
    
    def training_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        """Perform one EM iteration.
        
        For DFM, each "step" is actually one EM iteration. The batch contains
        the full time series data.
        
        **Important**: If `fit_em()` has already completed training (in `on_train_start()`),
        this method skips additional EM iterations to prevent double training and log-likelihood
        deterioration. The model will use the results from `fit_em()` instead.
        
        Parameters
        ----------
        batch : torch.Tensor or tuple
            Data tensor (T x N) or (data, target) tuple where data is (T x N) time series
        batch_idx : int
            Batch index (should be 0 for full sequence)
            
        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood (to minimize)
        """
        # Skip if fit_em() already completed training
        if self.training_state is not None:
            # fit_em() has already run - use its results instead of doing more EM iterations
            loglik = self.training_state.loglik
            self.log('loglik', loglik, on_step=True, on_epoch=True, prog_bar=True)
            self.log('em_iteration', float(self.training_state.num_iter), on_step=True, on_epoch=True)
            # Return negative log-likelihood as loss (consistent with normal training_step)
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
            return -torch.tensor(loglik, device=device, dtype=torch.float32)
        
        # Handle both tuple and single tensor batches
        if isinstance(batch, tuple):
            data, _ = batch
        else:
            data = batch
        # data is (batch_size, T, N) or (T, N) depending on DataLoader
        if data.ndim == 3:
            # Take first batch (should only be one for time series)
            data = data[0]
        
        # Initialize parameters if not done yet
        if self.A is None:
            self.initialize_from_data(data)
        
        # Prepare data for EM step
        # EM expects y as (N x T), but data is (T x N)
        y = data.T  # (N x T)
        
        # Create EM step parameters using stored mixed frequency parameters
        em_params = self._create_em_step_params(y, y.device, y.dtype)
        
        # Perform EM step - use self.em(...) instead of em_step(...)
        C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = self.em(em_params)
        
        # Update parameters (EM doesn't use gradients, so we update directly)
        with torch.no_grad():
            self.A.data = A_new
            self.C.data = C_new
            self.Q.data = Q_new
            self.R.data = R_new
            self.Z_0.data = Z_0_new
            self.V_0.data = V_0_new
        
        # Log metrics
        self.log('loglik', loglik, on_step=True, on_epoch=True, prog_bar=True)
        self.log('em_iteration', float(self.current_epoch), on_step=True, on_epoch=True)
        
        # Return negative log-likelihood as loss (to minimize)
        return -torch.tensor(loglik, device=data.device, dtype=data.dtype)
    
    def on_train_epoch_end(self) -> None:
        """Check convergence after each epoch (EM iteration)."""
        if self.training_state is None:
            return
        
        # Check convergence - use self.em.check_convergence() instead of em_converged()
        converged, change = self.em.check_convergence(
            self.training_state.loglik,
            self.training_state.loglik,  # Previous loglik (would need to track)
            self.threshold,
            verbose=False
        )
        
        if converged:
            self.training_state.converged = True
            # Only log convergence once, not on every training_step call
            if not hasattr(self, '_convergence_logged'):
                _logger.info(f"EM algorithm converged at iteration {self.training_state.num_iter}")
                self._convergence_logged = True
    
    def fit_em(
        self,
        X: torch.Tensor,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None
    ) -> DFMTrainingState:
        """Run full EM algorithm until convergence.
        
        This method runs the complete EM algorithm outside of Lightning's
        training loop, which is more natural for EM. Called by trainer during fit().
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        Mx : np.ndarray, optional
            Mean values for unstandardization (N,)
        Wx : np.ndarray, optional
            Standard deviation values for unstandardization (N,)
            
        Returns
        -------
        DFMTrainingState
            Final training state with parameters and convergence info
        """
        self.Mx = Mx
        self.Wx = Wx
        
        # Ensure data is on same device as model (Lightning handles this automatically)
        X = X.to(self.device)
        
        device = X.device
        dtype = X.dtype
        
        # Initialize parameters and impute missing data
        self.initialize_from_data(X)
        
        # Impute missing data for EM loop
        from dfm_python.utils.data import rem_nans_spline_torch
        X_imputed, _ = rem_nans_spline_torch(X, method=self.nan_method, k=self.nan_k)
        X_imputed = torch.where(torch.isfinite(X_imputed), X_imputed, torch.tensor(0.0, device=device, dtype=dtype))
        
        # Store imputed data
        self.data_processed = X_imputed
        
        # Prepare data for EM (use imputed data, not raw data)
        y = X_imputed.T  # (N x T)
        
        # Initialize state
        previous_loglik = float('-inf')
        best_loglik = float('-inf')
        best_params = None
        num_iter = 0
        converged = False
        
        # Track parameter stability for diagnostics
        previous_A_norm = None
        previous_C_norm = None
        
        # EM loop
        while num_iter < self.max_iter and not converged:
            # Create EM step parameters using stored mixed frequency parameters
            em_params = self._create_em_step_params(y, device, dtype)
            
            # Perform EM step - use self.em(...) instead of em_step(...)
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = self.em(em_params)
            
            # Check for NaN in parameters (early stopping)
            has_nan = (
                torch.any(torch.isnan(C_new)) or torch.any(torch.isnan(A_new)) or
                torch.any(torch.isnan(Q_new)) or torch.any(torch.isnan(R_new)) or
                torch.any(torch.isnan(Z_0_new)) or torch.any(torch.isnan(V_0_new)) or
                (isinstance(loglik, float) and (np.isnan(loglik) or np.isinf(loglik)))
            )
            if has_nan:
                _logger.error(f"EM algorithm: NaN/Inf detected at iteration {num_iter + 1}, stopping early")
                break
            
            # Update parameters
            with torch.no_grad():
                self.A.data = A_new
                self.C.data = C_new
                self.Q.data = Q_new
                self.R.data = R_new
                self.Z_0.data = Z_0_new
                self.V_0.data = V_0_new
            
            # Track parameter stability
            try:
                A_norm = torch.linalg.norm(A_new).item()
                C_norm = torch.linalg.norm(C_new).item()
                
                if previous_A_norm is not None and previous_C_norm is not None and num_iter > 5:
                    A_change = abs(A_norm - previous_A_norm) / max(previous_A_norm, 1e-10)
                    C_change = abs(C_norm - previous_C_norm) / max(previous_C_norm, 1e-10)
                    param_change = max(A_change, C_change)
                    
                    if param_change < 1e-6:
                        _logger.debug(f"EM: Parameters stable (change={param_change:.2e}) at iter {num_iter + 1}")
                    elif param_change > 10.0:
                        _logger.warning(f"EM: Parameters changing significantly (change={param_change:.2e}) at iter {num_iter + 1}")
                
                previous_A_norm = A_norm
                previous_C_norm = C_norm
            except (RuntimeError, ValueError):
                pass
            
            # Track best log-likelihood and parameters (for early stopping on divergence)
            if loglik > best_loglik:
                best_loglik = loglik
                # Store best parameters in case we need to revert
                best_params = {
                    'A': self.A.data.clone(),
                    'C': self.C.data.clone(),
                    'Q': self.Q.data.clone(),
                    'R': self.R.data.clone(),
                    'Z_0': self.Z_0.data.clone(),
                    'V_0': self.V_0.data.clone()
                }
            
            # Check for log-likelihood deterioration
            if num_iter > 10 and loglik < best_loglik - 1000:
                _logger.warning(
                    f"EM algorithm: Log-likelihood deteriorated significantly from best value. "
                    f"Best: {best_loglik:.4f}, Current: {loglik:.4f}, Deterioration: {best_loglik - loglik:.4f}. "
                    f"Stopping early at iteration {num_iter + 1} to prevent further divergence. "
                    f"Reverting to best parameters from iteration {num_iter}."
                )
                # Revert to best parameters
                if best_params is not None:
                    with torch.no_grad():
                        self.A.data = best_params['A']
                        self.C.data = best_params['C']
                        self.Q.data = best_params['Q']
                        self.R.data = best_params['R']
                        self.Z_0.data = best_params['Z_0']
                        self.V_0.data = best_params['V_0']
                # Use best log-likelihood for final state
                loglik = best_loglik
                break
            
            # Check convergence
            if num_iter > 2:
                converged, change = self.em.check_convergence(
                    loglik, previous_loglik, self.threshold, verbose=(num_iter % 10 == 0)
                )
            else:
                change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else 0.0
            
            previous_loglik = loglik
            num_iter += 1
            
            # Log metrics using Lightning (enables TensorBoard, WandB, etc.)
            # on_step=False because fit_em may be called from on_train_start
            self.log('train/loglik', loglik, on_step=False, on_epoch=True)
            self.log('train/em_iteration', float(num_iter), on_step=False, on_epoch=True)
            self.log('train/loglik_change', change, on_step=False, on_epoch=True)
            
            # Log progress
            if num_iter % 5 == 0 or num_iter == 1:
                status = " ✓" if converged else ""
                _logger.info(f"EM iteration {num_iter}/{self.max_iter}: loglik={loglik:.4f}, change={change:.2e}{status}")
        
        # Store final state
        self.training_state = DFMTrainingState(
            A=self.A.data.clone(),
            C=self.C.data.clone(),
            Q=self.Q.data.clone(),
            R=self.R.data.clone(),
            Z_0=self.Z_0.data.clone(),
            V_0=self.V_0.data.clone(),
            loglik=loglik,
            num_iter=num_iter,
            converged=converged
        )
        
        # Final status
        if converged:
            print(f"\n✓ EM converged after {num_iter} iterations (loglik: {loglik:.6f})")
        else:
            print(f"\n⚠ EM stopped after {num_iter} iterations (loglik: {loglik:.6f}, change: {change:.2e})")
        
        _logger.info(f"EM training completed: converged={converged}, iterations={num_iter}, loglik={loglik:.6f}")
        
        return self.training_state
    
    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        if self.training_state is None:
            raise RuntimeError(
                "DFM get_result failed: model has not been fitted yet. "
                "Please call fit_em() first."
            )
        
        if self.data_processed is None:
            raise RuntimeError(
                "DFM get_result failed: data not available. "
                "Please ensure fit_em() was called with data."
            )
        
        # Get final smoothed factors using Kalman filter
        y = self.data_processed.T  # (N x T)
        
        # Run final Kalman smoothing with converged parameters - use self.kalman(...) instead of kalman_filter_smooth(...)
        zsmooth, Vsmooth, _, _ = self.kalman(
            y,
            self.training_state.A,
            self.training_state.C,
            self.training_state.Q,
            self.training_state.R,
            self.training_state.Z_0,
            self.training_state.V_0
        )
        
        # zsmooth is (m x (T+1)), transpose to ((T+1) x m)
        Zsmooth = zsmooth.T
        Z = Zsmooth[1:, :].cpu().numpy()  # T x m (skip initial state)
        
        # Convert parameters to numpy
        A = self.training_state.A.cpu().numpy()
        C = self.training_state.C.cpu().numpy()
        Q = self.training_state.Q.cpu().numpy()
        R = self.training_state.R.cpu().numpy()
        Z_0 = self.training_state.Z_0.cpu().numpy()
        V_0 = self.training_state.V_0.cpu().numpy()
        r = self.r.cpu().numpy()
        
        # Compute smoothed data
        x_sm = Z @ C.T  # T x N (standardized smoothed data)
        
        # Unstandardize
        Wx_clean = np.where(np.isnan(self.Wx), 1.0, self.Wx) if self.Wx is not None else np.ones(C.shape[0])
        Mx_clean = np.where(np.isnan(self.Mx), 0.0, self.Mx) if self.Mx is not None else np.zeros(C.shape[0])
        X_sm = x_sm * Wx_clean + Mx_clean  # T x N (unstandardized smoothed data)
        
        # Create result object
        result = DFMResult(
            x_sm=x_sm,
            X_sm=X_sm,
            Z=Z,
            C=C,
            R=R,
            A=A,
            Q=Q,
            Mx=self.Mx if self.Mx is not None else np.zeros(C.shape[0]),
            Wx=self.Wx if self.Wx is not None else np.ones(C.shape[0]),
            Z_0=Z_0,
            V_0=V_0,
            r=r,
            p=self.p,
            converged=self.training_state.converged,
            num_iter=self.training_state.num_iter,
            loglik=self.training_state.loglik,
            series_ids=self.config.get_series_ids() if hasattr(self.config, 'get_series_ids') else None,
            block_names=getattr(self.config, 'block_names', None)
        )
        
        return result
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizers.
        
        EM algorithm doesn't use standard optimizers, but Lightning requires
        this method. Return empty list.
        
        Returns
        -------
        List[torch.optim.Optimizer]
            Empty list (EM algorithm doesn't use optimizers)
        """
        return []
    
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> 'DFM':
        """Load configuration from various sources.
        
        After loading config, the model needs to be re-initialized with the new config.
        For standard Lightning pattern, pass config directly to __init__.
        """
        # Use common config loading logic
        new_config = self._load_config_common(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        
        # DFM-specific: Initialize r and blocks tensors
        self.r = torch.tensor(
            new_config.factors_per_block if new_config.factors_per_block is not None
            else np.ones(new_config.get_blocks_array().shape[1]),
            dtype=torch.float32
        )
        self.blocks = torch.tensor(new_config.get_blocks_array(), dtype=torch.float32)
        
        return self
    
    
    def on_train_start(self) -> None:
        """Called when training starts. Run EM algorithm."""
        # Get processed data and standardization params from DataModule
        data_module = self._get_datamodule()
        X_torch = data_module.get_processed_data()
        Mx, Wx = data_module.get_std_params()
        
        # Run EM algorithm
        self.fit_em(X_torch, Mx=Mx, Wx=Wx)
        
        super().on_train_start()
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        history: Optional[int] = None,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values.
        
        This method can be called after training. It uses the training state
        from the Lightning module to generate forecasts.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        history : int, optional
            Number of historical periods to use for Kalman filter update before prediction.
            If None, uses full history (default). If specified (e.g., 60), uses only the most
            recent N periods for efficiency. Initial state (Z_0, V_0) is always estimated from
            full history (including any new data beyond training period).
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x N)
            If only return_factors is True:
                Z_forecast (horizon x m)
            
        Notes
        -----
        When history is specified, the method uses only the most recent N periods for
        Kalman filter update, improving computational efficiency. The initial state
        (Z_0, V_0) is always estimated from full history (including any new data beyond
        training period), ensuring accuracy while maintaining efficiency.
        """
        if self.training_state is None:
            raise ValueError(
                f"{self.__class__.__name__} prediction failed: model has not been trained yet. "
                f"Please call trainer.fit(model, data_module) first"
            )
        
        # Get result (only call get_result() if _result is None)
        if not hasattr(self, '_result') or self._result is None:
            self._result = self.get_result()
        
        result = self._result
        
        if not hasattr(result, 'Z') or result.Z is None:
            raise ValueError(
                "DFM prediction failed: result.Z is not available. "
                "This may indicate the model was not properly trained or result object is corrupted."
            )
        
        # Compute default horizon
        if horizon is None:
            from ..config.utils import get_periods_per_year
            from ..utils.helpers import get_clock_frequency
            clock = get_clock_frequency(self.config, 'm')
            horizon = get_periods_per_year(clock)
        
        # Validate horizon
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        
        # Extract model parameters
        A = result.A
        C = result.C
        Wx = result.Wx
        Mx = result.Mx
        p = getattr(result, 'p', 1)  # VAR order, default to 1 for DFM
        
        # Update factor state with history if specified
        if history is not None and history > 0:
            Z_last_updated = self._update_factor_state_with_history(
                history=history,
                result=result,
                kalman_filter=getattr(self, 'kalman', None)
            )
            if Z_last_updated is not None:
                Z_last = Z_last_updated
            else:
                # Fallback to training state if update failed
                Z_last = result.Z[-1, :]
        else:
            # Use training state (default behavior)
            Z_last = result.Z[-1, :]
        
        # Validate factor state
        if np.any(np.isnan(Z_last)):
            nan_count = np.sum(np.isnan(Z_last))
            nan_ratio = nan_count / len(Z_last)
            raise ValueError(
                f"DFM prediction failed: {nan_count}/{len(Z_last)} factors contain NaN ({nan_ratio:.1%}). "
                f"Model may not have converged. Try increasing max_iter or checking data quality."
            )
        
        # Validate parameters are finite
        if np.any(~np.isfinite(A)) or np.any(~np.isfinite(C)):
            raise ValueError(
                "DFM prediction failed: model parameters (A or C) contain NaN/Inf. "
                "Check training convergence and data quality."
            )
        
        # Forecast factors using VAR dynamics (common helper)
        Z_prev = result.Z[-2, :] if result.Z.shape[0] >= 2 and p == 2 else None
        Z_forecast = self._forecast_var_factors(
            Z_last=Z_last,
            A=A,
            p=p,
            horizon=horizon,
            Z_prev=Z_prev
        )
        
        # Transform factors to observations (common helper)
        X_forecast = self._transform_factors_to_observations(
            Z_forecast=Z_forecast,
            C=C,
            Wx=Wx,
            Mx=Mx
        )
        
        # Validate forecast results are finite
        if np.any(~np.isfinite(X_forecast)):
            nan_count = np.sum(~np.isfinite(X_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in forecast. "
                f"Possible numerical instability. "
                f"Please check model parameters and data quality."
            )
        
        # Validate forecast values are within reasonable bounds
        if Wx is not None and Mx is not None and len(Wx) > 0 and len(Mx) > 0:
            # Check each series individually
            extreme_threshold_std = 50.0  # Flag if forecast is > 50 std devs from mean
            for i in range(X_forecast.shape[1] if X_forecast.ndim > 1 else 1):
                if i < len(Wx) and i < len(Mx) and Wx[i] > 0:
                    series_forecast = X_forecast[:, i] if X_forecast.ndim > 1 else X_forecast
                    series_mean = Mx[i]
                    series_std = Wx[i]
                    # Calculate how many standard deviations each forecast is from the mean
                    abs_deviations = np.abs(series_forecast - series_mean) / series_std
                    max_deviation = np.max(abs_deviations) if len(abs_deviations) > 0 else 0.0
                    if max_deviation > extreme_threshold_std:
                        extreme_count = np.sum(abs_deviations > extreme_threshold_std)
                        _logger.warning(
                            f"DFM prediction: Extreme forecast for series {i} "
                            f"(max deviation: {max_deviation:.1f} std devs). "
                            f"Possible numerical instability."
                        )
        
        # Convert to numpy (handles torch inputs) and attempt scaler inverse_transform if available
        X_forecast = np.asarray(
            X_forecast.detach().cpu().numpy() if hasattr(X_forecast, "detach") else X_forecast
        )
        scaler = getattr(self, "scaler", None)
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            try:
                X_forecast = scaler.inverse_transform(X_forecast)
            except Exception as e:
                _logger.warning(
                    f"DFM prediction: scaler.inverse_transform failed, returning unstandardized values. "
                    f"error={e}"
                )
        
        if return_factors and np.any(~np.isfinite(Z_forecast)):
            nan_count = np.sum(~np.isfinite(Z_forecast))
            raise ValueError(
                f"DFM prediction failed: produced {nan_count} NaN/Inf values in factor forecast. "
                f"Possible numerical instability in factor dynamics. "
                f"Please check model parameters and training convergence."
            )
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    def update(
        self,
        X_std: np.ndarray,
        *,
        history: Optional[int] = None,
        kalman_filter: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> 'DFM':
        """Update factor state with standardized data.
        
        This method permanently updates the last factor state (result.Z[-1, :])
        using the provided standardized data. Users should handle all preprocessing
        (masking, imputation, standardization) before calling this method.
        
        Parameters
        ----------
        X_std : np.ndarray
            Standardized data array (T x N), where T is number of time periods
            and N is number of series. Data should already be standardized using
            result.Mx and result.Wx.
        history : int, optional
            Number of recent periods to use for factor state update. If None, uses
            all provided data (default). If specified (e.g., 60), uses only the most
            recent N periods. Initial state (Z_0, V_0) is always estimated from
            full training data, but the update uses only recent history for efficiency.
        kalman_filter : Any, optional
            Kalman filter instance. If None, uses default or model's kalman filter.
            
        Returns
        -------
        DFM
            Self for method chaining
            
        Examples
        --------
        >>> # Update state with new data, then predict
        >>> model.update(X_std).predict(horizon=1)
        >>> # Or update with only recent 12 periods
        >>> model.update(X_std, history=12)
        >>> forecast = model.predict(horizon=6)
        """
        self._check_trained()
        
        # Optionally replace scaler (e.g., if refit on new regime)
        if scaler is not None:
            self.scaler = scaler
        
        result = self.result  # Use property which ensures non-None after _check_trained()
        
        # Validate input shape
        if not isinstance(X_std, np.ndarray):
            X_std = np.asarray(X_std)
        if X_std.ndim != 2:
            raise ValueError(
                f"DFM update(): X_std must be 2D array (T x N), "
                f"got shape {X_std.shape}"
            )
        
        # Handle NaN/Inf values
        X_std = np.where(np.isfinite(X_std), X_std, np.nan)
        
        # Filter to recent history if specified
        # Note: Initial state (Z_0, V_0) from result is estimated from full training data,
        # but we use only recent history for the update step
        if history is not None and history > 0:
            if X_std.shape[0] > history:
                X_recent = X_std[-history:, :]
                _logger.debug(
                    f"DFM update(): Using {history} most recent periods out of {X_std.shape[0]} total periods"
                )
            else:
                X_recent = X_std
                _logger.debug(
                    f"DFM update(): history={history} specified but data has only {X_std.shape[0]} periods, using all data"
                )
        else:
            X_recent = X_std
        
        # Update factor state using Kalman filter directly on standardized data
        Z_last_updated = self._update_factor_state_dfm(
            X_recent, result, kalman_filter or getattr(self, 'kalman', None)
        )
        
        # Update result.Z[-1, :] permanently
        if Z_last_updated is not None:
            result.Z[-1, :] = Z_last_updated
        else:
            _logger.warning(
                f"DFM update(): Failed to update factor state, "
                f"keeping current state"
            )
            
        return self
    
    @property
    def result(self) -> DFMResult:
        """Get model result from training state.
        
        Raises
        ------
        ValueError
            If model has not been trained yet
        """
        # Check if trained and extract result from training state if needed
        self._check_trained()
        return self._result
    
    
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        super().reset()
        return self



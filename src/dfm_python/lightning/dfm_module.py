"""PyTorch Lightning module for Linear Dynamic Factor Model.

This module provides DFMLightningModule which integrates the PyTorch EM algorithm
with PyTorch Lightning for training linear DFM models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
import pytorch_lightning as pl
from dataclasses import dataclass

from ..config import DFMConfig
from ..config.results import DFMResult
from ..ssm.kalman import KalmanFilter
from ..ssm.em import EMAlgorithm, EMStepParams
from ..logger import get_logger

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


class DFMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Linear Dynamic Factor Model.
    
    This module implements the EM algorithm for DFM estimation using PyTorch
    operations. It uses manual optimization since EM doesn't fit standard
    gradient-based training patterns.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
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
    """
    
    def __init__(
        self,
        config: DFMConfig,
        num_factors: Optional[int] = None,
        threshold: float = 1e-4,
        max_iter: int = 100,
        nan_method: int = 2,
        nan_k: int = 3,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.threshold = threshold
        self.max_iter = max_iter
        self.nan_method = nan_method
        self.nan_k = nan_k
        
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
        
        # Parameters will be initialized in setup() or fit()
        self.A: Optional[torch.nn.Parameter] = None
        self.C: Optional[torch.nn.Parameter] = None
        self.Q: Optional[torch.nn.Parameter] = None
        self.R: Optional[torch.nn.Parameter] = None
        self.Z_0: Optional[torch.nn.Parameter] = None
        self.V_0: Optional[torch.nn.Parameter] = None
        
        # Training state
        self.training_state: Optional[DFMTrainingState] = None
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
        # Parameters will be initialized during fit() or first training step
        pass
    
    def initialize_from_data(self, X: torch.Tensor) -> None:
        """Initialize parameters from data using PCA and OLS.
        
        Parameters
        ----------
        X : torch.Tensor
            Standardized data (T x N)
        """
        opt_nan = {'method': self.nan_method, 'k': self.nan_k}
        
        # Use self.em.initialize_parameters() with direct tensor operations (no CPU transfers)
        A, C, Q, R, Z_0, V_0 = self.em.initialize_parameters(
            X,
            r=self.r.to(X.device),
            p=self.p,
            blocks=self.blocks.to(X.device),
            opt_nan=opt_nan,
            R_mat=None,
            q=None,
            nQ=0,
            i_idio=None,
            clock=getattr(self.config, 'clock', 'm'),
            tent_weights_dict=None,
            frequencies=None,
            idio_chain_lengths=None,
            config=self.config
        )
        
        # Convert to Parameters
        self.A = nn.Parameter(A)
        self.C = nn.Parameter(C)
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)
        self.Z_0 = nn.Parameter(Z_0)
        self.V_0 = nn.Parameter(V_0)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform one EM iteration.
        
        For DFM, each "step" is actually one EM iteration. The batch contains
        the full time series data.
        
        Parameters
        ----------
        batch : tuple
            (data, target) where data is (T x N) time series
        batch_idx : int
            Batch index (should be 0 for full sequence)
            
        Returns
        -------
        loss : torch.Tensor
            Negative log-likelihood (to minimize)
        """
        data, _ = batch
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
        
        # Create EM step parameters
        em_params = EMStepParams(
            y=y,
            A=self.A,
            C=self.C,
            Q=self.Q,
            R=self.R,
            Z_0=self.Z_0,
            V_0=self.V_0,
            r=self.r.to(y.device),
            p=self.p,
            R_mat=None,
            q=None,
            nQ=0,
            i_idio=torch.ones(y.shape[0], device=y.device, dtype=y.dtype),
            blocks=self.blocks.to(y.device),
            tent_weights_dict={},
            clock=getattr(self.config, 'clock', 'm'),
            frequencies=None,
            idio_chain_lengths=torch.zeros(y.shape[0], device=y.device, dtype=y.dtype),
            config=self.config
        )
        
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
            _logger.info(f"EM algorithm converged at iteration {self.current_epoch}")
    
    def fit_em(
        self,
        X: torch.Tensor,
        Mx: Optional[np.ndarray] = None,
        Wx: Optional[np.ndarray] = None
    ) -> DFMTrainingState:
        """Run full EM algorithm until convergence.
        
        This method runs the complete EM algorithm outside of Lightning's
        training loop, which is more natural for EM.
        
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
        self.data_processed = X
        
        device = X.device
        dtype = X.dtype
        
        # Initialize parameters
        self.initialize_from_data(X)
        
        # Prepare data for EM
        y = X.T  # (N x T)
        
        # Initialize state
        previous_loglik = float('-inf')
        num_iter = 0
        converged = False
        
        # EM loop
        while num_iter < self.max_iter and not converged:
            # Create EM step parameters
            em_params = EMStepParams(
                y=y,
                A=self.A,
                C=self.C,
                Q=self.Q,
                R=self.R,
                Z_0=self.Z_0,
                V_0=self.V_0,
                r=self.r.to(device),
                p=self.p,
                R_mat=None,
                q=None,
                nQ=0,
                i_idio=torch.ones(y.shape[0], device=device, dtype=dtype),
                blocks=self.blocks.to(device),
                tent_weights_dict={},
                clock=getattr(self.config, 'clock', 'm'),
                frequencies=None,
                idio_chain_lengths=torch.zeros(y.shape[0], device=device, dtype=dtype),
                config=self.config
            )
            
            # Perform EM step - use self.em(...) instead of em_step(...)
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = self.em(em_params)
            
            # Update parameters
            with torch.no_grad():
                self.A.data = A_new
                self.C.data = C_new
                self.Q.data = Q_new
                self.R.data = R_new
                self.Z_0.data = Z_0_new
                self.V_0.data = V_0_new
            
            # Check convergence - use self.em.check_convergence() instead of em_converged()
            if num_iter > 2:
                converged, change = self.em.check_convergence(
                    loglik,
                    previous_loglik,
                    self.threshold,
                    verbose=(num_iter % 10 == 0)
                )
            else:
                change = abs(loglik - previous_loglik) if previous_loglik != float('-inf') else 0.0
            
            previous_loglik = loglik
            num_iter += 1
            
            # Log metrics using Lightning (enables TensorBoard, WandB, etc.)
            self.log('train/loglik', loglik, on_step=True, on_epoch=False)
            self.log('train/em_iteration', float(num_iter), on_step=True, on_epoch=False)
            self.log('train/loglik_change', change, on_step=True, on_epoch=False)
            
            if num_iter % 10 == 0:
                _logger.info(
                    f"EM iteration {num_iter}/{self.max_iter}: "
                    f"loglik={loglik:.4f}, change={change:.2e}"
                )
        
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
        
        return self.training_state
    
    def get_result(self) -> DFMResult:
        """Extract DFMResult from trained model.
        
        Returns
        -------
        DFMResult
            Estimation results with parameters, factors, and diagnostics
        """
        if self.training_state is None:
            raise RuntimeError("Model must be fitted before getting results. Call fit_em() first.")
        
        if self.data_processed is None:
            raise RuntimeError("Data not available. Ensure fit_em() was called with data.")
        
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
    
    def configure_optimizers(self):
        """Configure optimizers.
        
        EM algorithm doesn't use standard optimizers, but Lightning requires
        this method. Return empty list.
        """
        return []


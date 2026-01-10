"""Result structures for Dynamic Factor Model estimation.

This module contains model-specific result dataclasses:
- DFMResult(BaseResult): Results for linear DFM
- DDFMResult(BaseResult): Results for Deep DFM
"""

import numpy as np
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .model import DFMConfig
from .params import DDFMStateSpaceParams



# ============================================================================
# Base Result Structure
# ============================================================================

@dataclass
class BaseResult(ABC):
    """Base class for all factor model result structures.
    
    This abstract base class defines the core model outputs shared by all
    factor model results (DFM, DDFM, KDFM, etc.). Only essential model parameters
    and outputs are included - no user-specific metadata.
    
    Attributes
    ----------
    x_sm : np.ndarray
        Standardized smoothed data matrix (T x N), where T is time periods
        and N is number of series. Data is standardized (zero mean, unit variance).
        This is the internal representation used by the model.
        To get unstandardized data: use get_x_sm_original_scale(target_scaler) method.
    Z : np.ndarray
        Smoothed factor estimates (T x m), where m is the state dimension.
        Columns represent different factors (common factors and idiosyncratic components).
    C : np.ndarray
        Observation/loading matrix (N x m). Each row corresponds to a series,
        each column to a factor. C[i, j] gives the loading of series i on factor j.
    R : np.ndarray
        Covariance matrix for observation equation residuals (N x N).
        Typically diagonal, representing idiosyncratic variances.
    A : np.ndarray
        Transition matrix (m x m) for the state equation. Describes how factors
        evolve over time: Z_t = A @ Z_{t-1} + error.
    Q : np.ndarray
        Covariance matrix for transition equation residuals (m x m).
        Describes the covariance of factor innovations.
    target_scaler : Any, optional
        Sklearn scaler instance (StandardScaler, RobustScaler, etc.) for target series only.
        Used for unstandardization: X = scaler.inverse_transform(x).
        If None, assumes data is already in original scale.
    Z_0 : np.ndarray
        Initial state vector (m,). Starting values for factors at t=0.
    V_0 : np.ndarray
        Initial covariance matrix (m x m) for factors. Uncertainty about Z_0.
    r : np.ndarray
        Number of factors per block (n_blocks,). Each element specifies
        how many factors are in each block structure.
    p : int
        Number of lags in the autoregressive structure of factors. Typically p=1.
    converged : bool
        Whether estimation algorithm converged.
    num_iter : int
        Number of iterations performed.
    loglik : float
        Final log-likelihood value.
    """
    # Core state-space model parameters (required fields)
    x_sm: np.ndarray      # Standardized smoothed data (T x N)
    Z: np.ndarray         # Smoothed factors (T x m)
    C: np.ndarray         # Observation matrix (N x m)
    R: np.ndarray         # Covariance for observation residuals (N x N)
    A: np.ndarray         # Transition matrix (m x m)
    Q: np.ndarray         # Covariance for transition residuals (m x m)
    Z_0: np.ndarray       # Initial state (m,)
    V_0: np.ndarray       # Initial covariance (m x m)
    r: np.ndarray         # Number of factors per block
    p: int                # Number of lags
    # Optional fields (must come after required fields)
    target_scaler: Optional[Any] = None  # Sklearn scaler for target series unstandardization
    # Training diagnostics
    converged: bool = False  # Whether algorithm converged
    num_iter: int = 0     # Number of iterations completed
    loglik: float = -np.inf  # Final log-likelihood

    # ----------------------------
    # Convenience methods (OOP)
    # ----------------------------
    def num_series(self) -> int:
        """Return number of series (rows in C)."""
        return int(self.C.shape[0])

    def num_state(self) -> int:
        """Return state dimension (columns in Z/C)."""
        return int(self.Z.shape[1])

    def num_periods(self) -> int:
        """Return number of time periods (rows in Z/x_sm)."""
        return int(self.Z.shape[0])
    
    def num_factors(self) -> int:
        """Return number of primary factors (sum of r)."""
        try:
            return int(np.sum(self.r))
        except (ValueError, AttributeError, TypeError):
            return self.num_state()
    
    def get_x_sm_original_scale(self, target_scaler: Optional[Any] = None) -> np.ndarray:
        """Get smoothed data in original scale by inverse transforming target series.
        
        Note: DDFM trains and evaluates on scaled data. This method inverse transforms
        the target series back to original scale for user convenience.
        
        Parameters
        ----------
        target_scaler : Any, optional
            Sklearn scaler instance. If None, uses self.target_scaler.
            If both are None, returns x_sm as-is (assumes already in original scale).
        
        Returns
        -------
        np.ndarray
            Smoothed data with target series in original scale (T x N)
        """
        scaler = target_scaler if target_scaler is not None else self.target_scaler
        if scaler is not None and hasattr(scaler, 'inverse_transform'):
            return scaler.inverse_transform(self.x_sm)
        return self.x_sm
    
    def to_pandas_factors(self, time_index: Optional[object] = None, factor_names: Optional[List[str]] = None):
        """Return factors as pandas DataFrame."""
        try:
            import pandas as pd
            cols = factor_names or [f"F{i+1}" for i in range(self.num_state())]
            df_dict = {col: self.Z[:, i] for i, col in enumerate(cols)}
            if time_index is not None:
                if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                    df_dict['time'] = list(time_index)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.Z
    
    def to_pandas_smoothed(self, time_index: Optional[object] = None, series_ids: Optional[List[str]] = None, target_scaler: Optional[Any] = None):
        """Return smoothed data as pandas DataFrame."""
        try:
            import pandas as pd
            x_sm_original = self.get_x_sm_original_scale(target_scaler)
            cols = series_ids or [f"S{i+1}" for i in range(self.num_series())]
            df_dict = {col: x_sm_original[:, i] for i, col in enumerate(cols)}
            if time_index is not None:
                if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
                    df_dict['time'] = list(time_index)
            return pd.DataFrame(df_dict)
        except ImportError:
            return self.get_x_sm_original_scale(target_scaler)
    
    def summary(self) -> str:
        """Return a formatted summary of the model results.
        
        Returns
        -------
        str
            Formatted string containing model summary including:
            - Model type and structure
            - Data dimensions (series, factors, periods)
            - Training diagnostics (convergence, iterations, log-likelihood)
            - Factor structure (AR order, factors per block)
        """
        # Determine model type from class name
        model_type = self.__class__.__name__.replace('Result', '')
        
        # Build summary lines
        lines = []
        lines.append("=" * 80)
        lines.append(f"{model_type} Model Summary")
        lines.append("=" * 80)
        lines.append("")
        
        # Data dimensions
        lines.append("Data Dimensions:")
        lines.append(f"  Series: {self.num_series()}")
        lines.append(f"  Factors: {self.num_factors()} (total state dimension: {self.num_state()})")
        lines.append(f"  Time periods: {self.num_periods()}")
        lines.append("")
        
        # Factor structure
        lines.append("Factor Structure:")
        if hasattr(self.r, '__len__') and len(self.r) > 0:
            if len(self.r) == 1:
                lines.append(f"  Factors per block: {self.r[0]}")
            else:
                lines.append(f"  Factors per block: {self.r}")
        else:
            lines.append(f"  Total factors: {self.num_factors()}")
        lines.append(f"  AR order: {self.p}")
        lines.append("")
        
        # Training diagnostics
        lines.append("Training Diagnostics:")
        lines.append(f"  Converged: {self.converged}")
        lines.append(f"  Iterations: {self.num_iter}")
        lines.append(f"  Log-likelihood: {self.loglik:.4f}")
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)



# ============================================================================
# Model-Specific Result Classes
# ============================================================================
# BaseResult is imported from base.py - no duplicate definition needed

@dataclass
class DFMResult(BaseResult):
    """DFM estimation results structure.
    
    This dataclass contains all outputs from the DFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Inherits all fields and methods from BaseResult. This class is specifically
    for linear DFM results estimated using the EM algorithm.
    
    Attributes
    ----------
    converged : bool
        Whether EM algorithm converged.
    num_iter : int
        Number of EM iterations performed.
    training_max_iter : int, optional
        Maximum EM iterations used (from model.max_iter or config.max_iter)
    training_threshold : float, optional
        Convergence threshold used (from model.threshold or config.threshold)
    training_regularization_scale : float, optional
        Regularization scale used (from config.regularization if provided)
    
    Examples
    --------
    >>> from dfm_python import DFM
    >>> model = DFM()
    >>> Res = model.fit(X, config, threshold=1e-4)
    >>> # Access smoothed factors
    >>> common_factor = Res.Z[:, 0]
    >>> # Access factor loadings for first series
    >>> loadings = Res.C[0, :]
    >>> # Reconstruct smoothed series from factors
    >>> reconstructed = Res.Z @ Res.C.T
    >>> # Access training hyperparameters
    >>> print(f"Used {Res.training_max_iter} max iterations")
    >>> print(f"Convergence threshold: {Res.training_threshold}")
    """
    # All fields inherited from BaseResult
    # converged and num_iter have specific meaning for EM algorithm
    
    # Training hyperparameters (from config, no separate fit_params needed)
    training_max_iter: Optional[int] = None
    training_threshold: Optional[float] = None
    training_regularization_scale: Optional[float] = None
    
    def summary(self) -> str:
        """Return a formatted summary of the DFM results."""
        summary_text = super().summary()
        lines = summary_text.split("\n")
        
        # Insert training hyperparameters before the final separator
        insert_idx = len(lines) - 1
        training_info = []
        if self.training_max_iter is not None:
            training_info.append(f"  Max iterations: {self.training_max_iter}")
        if self.training_threshold is not None:
            training_info.append(f"  Convergence threshold: {self.training_threshold:.6f}")
        if self.training_regularization_scale is not None:
            training_info.append(f"  Regularization scale: {self.training_regularization_scale:.6f}")
        
        if training_info:
            lines.insert(insert_idx, "")
            lines.insert(insert_idx, "Training Hyperparameters:")
            for info in training_info:
                lines.insert(insert_idx + 1, info)
        
        return "\n".join(lines)


@dataclass
class DDFMResult(BaseResult):
    """DDFM estimation results structure.
    
    This dataclass contains all outputs from the DDFM estimation procedure,
    including estimated parameters, smoothed data, and factors.
    
    Inherits all fields and methods from BaseResult. This class is specifically
    for Deep Dynamic Factor Model results estimated using gradient descent.
    
    Attributes
    ----------
    converged : bool
        Whether MCMC/gradient descent algorithm converged.
    num_iter : int
        Number of MCMC iterations or epochs performed.
    training_loss : float, optional
        Final training loss from neural network training.
    encoder_layers : List[int], optional
        Architecture of the encoder network used.
    use_idiosyncratic : bool, optional
        Whether idiosyncratic components were modeled.
    fit_params : DDFMStateSpaceParams, optional
        State-space model parameters created during fit (F, Q, mu_0, sigma_0, H, R).
        These are computed during build_state_space() and represent the fitted
        state-space model structure.
    # Training hyperparameters
    training_max_iter : int, optional
        Maximum MCMC iterations used (from model.max_iter)
    training_tolerance : float, optional
        Convergence tolerance used (from model.tolerance)
    training_n_mc_samples : int, optional
        Number of Monte Carlo samples per MCMC iteration (from model.n_mc_samples)
    training_window_size : int, optional
        Batch/window size used during training (from model.window_size)
    training_learning_rate : float, optional
        Learning rate used (from model.learning_rate)
    training_optimizer : str, optional
        Optimizer type used ('Adam' or 'SGD', from model.optimizer_type)
    training_decay_learning_rate : bool, optional
        Whether learning rate decay was used (from model.decay_learning_rate)
    training_encoder_size : tuple, optional
        Encoder architecture tuple (from model.encoder_size)
    training_decoder_size : tuple, optional
        Decoder architecture tuple (from model.decoder_size, None for linear)
    training_target_series : List[str], optional
        Target series used for reconstruction loss (from model.target_series)
    training_use_bias : bool, optional
        Whether bias was used in final decoder layer (from model.use_bias).
        The bias term is extracted and used to adjust data mean during state-space construction.
    training_batch_norm : bool, optional
        Whether batch normalization was used (from model.batch_norm)
    training_activation : str, optional
        Activation function used ('relu' or 'tanh', from model.activation)
    
    Examples
    --------
    >>> from dfm_python import DDFM
    >>> model = DDFM(encoder_layers=[64, 32], num_factors=2)
    >>> Res = model.fit(X, config, epochs=100)
    >>> # Access smoothed factors
    >>> common_factor = Res.Z[:, 0]
    >>> # Access factor loadings
    >>> loadings = Res.C[0, :]
    >>> # Access state-space parameters
    >>> if Res.fit_params is not None:
    ...     transition_matrix = Res.fit_params.F
    ...     observation_matrix = Res.fit_params.H
    """
    # All fields inherited from BaseResult
    # Additional DDFM-specific fields
    training_loss: Optional[float] = None  # Final training loss
    encoder_layers: Optional[List[int]] = None  # Encoder architecture
    use_idiosyncratic: Optional[bool] = None  # Whether idio components were used
    
    # State-space parameters (fitted during training)
    fit_params: Optional[DDFMStateSpaceParams] = None
    
    # Training hyperparameters
    training_max_iter: Optional[int] = None
    training_tolerance: Optional[float] = None
    training_n_mc_samples: Optional[int] = None
    training_window_size: Optional[int] = None
    training_learning_rate: Optional[float] = None
    training_optimizer: Optional[str] = None
    training_decay_learning_rate: Optional[bool] = None
    training_encoder_size: Optional[tuple] = None
    training_decoder_size: Optional[tuple] = None
    training_target_series: Optional[List[str]] = None
    training_use_bias: Optional[bool] = None
    training_batch_norm: Optional[bool] = None
    training_activation: Optional[str] = None
    
    def summary(self) -> str:
        """Return a formatted summary of the DDFM results."""
        summary_text = super().summary()
        lines = summary_text.split("\n")
        
        # Insert DDFM-specific information before the final separator
        insert_idx = len(lines) - 1
        ddfm_info = []
        
        if self.training_loss is not None:
            ddfm_info.append(f"  Final training loss: {self.training_loss:.4f}")
        if self.encoder_layers is not None:
            ddfm_info.append(f"  Encoder architecture: {self.encoder_layers}")
        if self.training_encoder_size is not None:
            ddfm_info.append(f"  Encoder size: {self.training_encoder_size}")
        if self.training_decoder_size is not None:
            ddfm_info.append(f"  Decoder size: {self.training_decoder_size}")
        if self.training_max_iter is not None:
            ddfm_info.append(f"  Max MCMC iterations: {self.training_max_iter}")
        if self.training_tolerance is not None:
            ddfm_info.append(f"  Convergence tolerance: {self.training_tolerance:.6f}")
        if self.training_n_mc_samples is not None:
            ddfm_info.append(f"  MC samples per iteration: {self.training_n_mc_samples}")
        if self.training_window_size is not None:
            ddfm_info.append(f"  Window size: {self.training_window_size}")
        if self.training_learning_rate is not None:
            ddfm_info.append(f"  Learning rate: {self.training_learning_rate:.6f}")
        if self.training_optimizer is not None:
            ddfm_info.append(f"  Optimizer: {self.training_optimizer}")
        if self.training_decay_learning_rate is not None:
            ddfm_info.append(f"  Learning rate decay: {self.training_decay_learning_rate}")
        
        if ddfm_info:
            lines.insert(insert_idx, "")
            lines.insert(insert_idx, "Neural Network Training:")
            for info in ddfm_info:
                lines.insert(insert_idx + 1, info)
        
        return "\n".join(lines)

# FitParams moved to config.schema.params as DFMParams


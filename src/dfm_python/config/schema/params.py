"""Model parameters for DFM and DDFM.

This module defines dataclasses for storing fit-time parameters and state-space parameters.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class DFMFitParams:
    """Fit-time parameter overrides for DFM estimation.
    
    This dataclass allows overriding configuration parameters at fit time
    without modifying the configuration object itself.
    
    Based on DFM.fit() usage, these parameters are passed to the EM algorithm.
    
    Attributes
    ----------
    max_iter : int, optional
        Maximum EM iterations (overrides config.max_iter)
        Used in: kalman_filter.em(max_iter=...)
    threshold : float, optional
        Convergence threshold (overrides config.threshold)
        Used in: kalman_filter.em(threshold=...)
    regularization_scale : float, optional
        Regularization scale for matrix operations (overrides EM config)
        Note: This affects EMConfig.regularization and EMConfig.matrix_regularization
    """
    max_iter: Optional[int] = None
    threshold: Optional[float] = None
    regularization_scale: Optional[float] = None

@dataclass
class DDFMTrainingState:
    """Training state for DDFM model.
    
    Stores the current state of DDFM training, including convergence status,
    loss, factors, and residuals. Used for checkpointing and resuming training.
    
    Attributes
    ----------
    num_iter : int
        Current iteration number in MCMC training loop.
    loss_now : float, optional
        Current training loss value.
    converged : bool
        Whether training has converged.
    eps : np.ndarray, optional
        Idiosyncratic residuals (T x num_target_series).
    factors : np.ndarray, optional
        Extracted factors (n_mc_samples x T x num_factors) or (T x num_factors).
    last_neurons : np.ndarray, optional
        Last layer neurons for MLP decoder (n_mc_samples x T x num_neurons) or (T x num_neurons).
        For linear decoder, this equals factors.
    """
    num_iter: int = 0
    loss_now: Optional[float] = None
    converged: bool = False
    eps: Optional[np.ndarray] = None
    factors: Optional[np.ndarray] = None
    last_neurons: Optional[np.ndarray] = None

@dataclass
class DDFMFitParams:
    """State-space model parameters for DDFM (created during fit).
    
    These parameters are computed during DDFM.build_state_space() and represent
    the fitted state-space model structure. Naming follows original paper conventions.
    
    Parameters
    ----------
    F : np.ndarray
        Transition matrix (m x m) - VAR(1) dynamics for factors (A in paper)
    Q : np.ndarray
        Process noise covariance (m x m) - innovation covariance (W in original code)
    mu_0 : np.ndarray
        Initial state mean (m,) - initial factor values
    Sigma_0 : np.ndarray
        Initial state covariance (m x m) - initial uncertainty
    H : np.ndarray
        Observation matrix (N x m) - decoder weights (measurement equation, theta in paper)
    R : np.ndarray
        Observation noise covariance (N x N) - typically diagonal, small values
    """
    F: np.ndarray  # Transition matrix (m x m) - A in paper
    Q: np.ndarray  # Process noise covariance (m x m) - W in original code
    mu_0: np.ndarray  # Initial state mean (m,)
    Sigma_0: np.ndarray  # Initial state covariance (m x m)
    H: np.ndarray  # Observation matrix (N x m) - theta in paper
    R: np.ndarray  # Observation noise covariance (N x N)


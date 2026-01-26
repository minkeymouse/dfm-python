"""Model parameters for DFM and DDFM.

This module defines dataclasses for storing state-space parameters and model structure.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Any


@dataclass
class DFMModelState:
    """DFM model state including structure, mixed-frequency parameters, and state-space parameters.
    
    Consolidates all DFM model state for checkpointing: model structure, mixed-frequency
    configuration, and fitted state-space parameters.
    
    Attributes
    ----------
    num_factors : int
        Number of factors
    r : np.ndarray
        Number of factors per block
    p : int
        AR lag order
    blocks : np.ndarray
        Block structure array
    mixed_freq : bool, optional
        Whether mixed-frequency data is used
    constraint_matrix : np.ndarray, optional
        Constraint matrix for tent kernel aggregation
    constraint_vector : np.ndarray, optional
        Constraint vector for tent kernel aggregation
    n_slower_freq : int
        Number of slower-frequency series
    n_clock_freq : int, optional
        Number of clock-frequency series
    tent_weights_dict : dict, optional
        Dictionary of tent weights by frequency
    frequencies : np.ndarray, optional
        Frequency array for each series
    idio_indicator : np.ndarray, optional
        Indicator for idiosyncratic components
    max_lag_size : int, optional
        Maximum lag size for state dimension
    A : np.ndarray, optional
        Transition matrix (m x m) - VAR dynamics for factors
    C : np.ndarray, optional
        Observation matrix (N x m) - factor loadings
    Q : np.ndarray, optional
        Process noise covariance (m x m) - innovation covariance
    R : np.ndarray, optional
        Observation noise covariance (N x N) - typically diagonal
    Z_0 : np.ndarray, optional
        Initial state mean (m,) - initial factor values
    V_0 : np.ndarray, optional
        Initial state covariance (m x m) - initial uncertainty
    """
    num_factors: int
    r: np.ndarray
    p: int
    blocks: np.ndarray
    mixed_freq: Optional[bool] = None
    constraint_matrix: Optional[np.ndarray] = None
    constraint_vector: Optional[np.ndarray] = None
    n_slower_freq: int = 0
    n_clock_freq: Optional[int] = None
    tent_weights_dict: Optional[dict] = None
    frequencies: Optional[np.ndarray] = None
    idio_indicator: Optional[np.ndarray] = None
    max_lag_size: Optional[int] = None
    # State-space parameters (fitted during training)
    A: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    Z_0: Optional[np.ndarray] = None
    V_0: Optional[np.ndarray] = None
    
    @classmethod
    def from_model(cls, model: Any) -> 'DFMModelState':
        """Create DFMModelState from DFM model instance.
        
        Parameters
        ----------
        model : DFM
            DFM model instance
            
        Returns
        -------
        DFMModelState
            Model state dataclass
        """
        # Get state-space parameters from training_state if available
        A = C = Q = R = Z_0 = V_0 = None
        if hasattr(model, 'training_state') and model.training_state is not None:
            # Handle both old DFMStateSpaceParams format and new DFMModelState format
            ts = model.training_state
            if hasattr(ts, 'A'):
                A = ts.A
                C = ts.C
                Q = ts.Q
                R = ts.R
                Z_0 = ts.Z_0
                V_0 = ts.V_0
        else:
            # Fallback: get from model attributes directly
            A = getattr(model, 'A', None)
            C = getattr(model, 'C', None)
            Q = getattr(model, 'Q', None)
            R = getattr(model, 'R', None)
            Z_0 = getattr(model, 'Z_0', None)
            V_0 = getattr(model, 'V_0', None)
        
        return cls(
            num_factors=model.num_factors,
            r=model.r,
            p=model.p,
            blocks=model.blocks,
            mixed_freq=getattr(model, '_mixed_freq', None),
            constraint_matrix=getattr(model, '_constraint_matrix', None),
            constraint_vector=getattr(model, '_constraint_vector', None),
            n_slower_freq=getattr(model, '_n_slower_freq', 0),
            n_clock_freq=getattr(model, '_n_clock_freq', None),
            tent_weights_dict=getattr(model, '_tent_weights_dict', None),
            frequencies=getattr(model, '_frequencies', None),
            idio_indicator=getattr(model, '_idio_indicator', None),
            max_lag_size=getattr(model, '_max_lag_size', None),
            A=A,
            C=C,
            Q=Q,
            R=R,
            Z_0=Z_0,
            V_0=V_0
        )
    
    def apply_to_model(self, model: Any) -> None:
        """Apply this state to DFM model instance.
        
        Parameters
        ----------
        model : DFM
            DFM model instance to update
        """
        model.num_factors = self.num_factors
        model.r = self.r
        model.p = self.p
        model.blocks = self.blocks
        model._mixed_freq = self.mixed_freq
        model._constraint_matrix = self.constraint_matrix
        model._constraint_vector = self.constraint_vector
        model._n_slower_freq = self.n_slower_freq
        model._n_clock_freq = self.n_clock_freq
        model._tent_weights_dict = self.tent_weights_dict
        model._frequencies = self.frequencies
        model._idio_indicator = self.idio_indicator
        model._max_lag_size = self.max_lag_size
        
        # Apply state-space parameters if available
        if self.A is not None and self.C is not None and self.Q is not None:
            model._update_parameters(
                self.A, self.C, self.Q, self.R, self.Z_0, self.V_0
            )

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
    
    def sync_from_model(self, model: Any) -> 'DDFMTrainingState':
        """Update this dataclass from model instance attributes.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance
            
        Returns
        -------
        DDFMTrainingState
            Self (for chaining)
        """
        if hasattr(model, '_has_factors') and model._has_factors:
            self.factors = getattr(model, 'factors', None)
            self.eps = getattr(model, 'eps', None)
            self.last_neurons = getattr(model, 'last_neurons', None)
            self.num_iter = getattr(model, '_num_iter', 0)
            self.loss_now = getattr(model, 'loss_now', None)
            self.converged = getattr(model, '_converged', False)
        return self

@dataclass
class DDFMStateSpaceParams:
    """State-space model parameters for DDFM (fitted during model training).
    
    These parameters are computed during DDFM.build_state_space() and represent
    the fitted state-space model structure. Similar to DFMStateSpaceParams, this contains
    the fitted state-space parameters but for DDFM. Naming follows original paper conventions.
    
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

# Backward compatibility alias
DFMStateSpaceParams = DFMModelState


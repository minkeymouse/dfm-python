"""Model parameters for DFM and DDFM.

This module defines dataclasses for storing state-space parameters and model structure.
"""

from dataclasses import dataclass
import numpy as np
import torch
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
class DDFMModelState:
    """DDFM model state including training state and state-space parameters.
    
    Consolidates all DDFM model state for checkpointing: training state (convergence,
    loss, factors, residuals) and fitted state-space parameters.
    
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
    F : np.ndarray, optional
        Transition matrix (m x m) - VAR(1) dynamics for factors (A in paper)
    Q : np.ndarray, optional
        Process noise covariance (m x m) - innovation covariance (W in original code)
    mu_0 : np.ndarray, optional
        Initial state mean (m,) - initial factor values
    Sigma_0 : np.ndarray, optional
        Initial state covariance (m x m) - initial uncertainty
    H : np.ndarray, optional
        Observation matrix (N x m) - decoder weights (measurement equation, theta in paper)
    R : np.ndarray, optional
        Observation noise covariance (N x N) - typically diagonal, small values
    """
    # Training state
    num_iter: int = 0
    loss_now: Optional[float] = None
    converged: bool = False
    eps: Optional[np.ndarray] = None
    factors: Optional[np.ndarray] = None
    last_neurons: Optional[np.ndarray] = None
    # State-space parameters (fitted during build_state_space)
    F: Optional[np.ndarray] = None  # Transition matrix (m x m) - A in paper
    Q: Optional[np.ndarray] = None  # Process noise covariance (m x m) - W in original code
    mu_0: Optional[np.ndarray] = None  # Initial state mean (m,)
    Sigma_0: Optional[np.ndarray] = None  # Initial state covariance (m x m)
    H: Optional[np.ndarray] = None  # Observation matrix (N x m) - theta in paper
    R: Optional[np.ndarray] = None  # Observation noise covariance (N x N)
    
    @classmethod
    def from_model(cls, model: Any) -> 'DDFMModelState':
        """Create DDFMModelState from DDFM model instance.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance
            
        Returns
        -------
        DDFMModelState
            Model state dataclass
        """
        # Get training state from model attributes
        factors = getattr(model, 'factors', None)
        eps = getattr(model, 'eps', None)
        last_neurons = getattr(model, 'last_neurons', None)
        num_iter = getattr(model, '_num_iter', 0)
        loss_now = getattr(model, 'loss_now', None)
        converged = getattr(model, '_converged', False)
        
        # Get state-space parameters from training_state
        F = Q = mu_0 = Sigma_0 = H = R = None
        if hasattr(model, 'training_state') and model.training_state is not None:
            ts = model.training_state
            if hasattr(ts, 'F'):
                F = ts.F
                Q = ts.Q
                mu_0 = ts.mu_0
                Sigma_0 = ts.Sigma_0
                H = ts.H
                R = ts.R
        
        return cls(
            num_iter=num_iter,
            loss_now=loss_now,
            converged=converged,
            eps=eps,
            factors=factors,
            last_neurons=last_neurons,
            F=F,
            Q=Q,
            mu_0=mu_0,
            Sigma_0=Sigma_0,
            H=H,
            R=R
        )
    
    def sync_from_model(self, model: Any) -> 'DDFMModelState':
        """Update this dataclass from model instance attributes.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance
            
        Returns
        -------
        DDFMModelState
            Self (for chaining)
        """
        if hasattr(model, '_has_factors') and model._has_factors:
            self.factors = getattr(model, 'factors', None)
            self.eps = getattr(model, 'eps', None)
            self.last_neurons = getattr(model, 'last_neurons', None)
            self.num_iter = getattr(model, '_num_iter', 0)
            self.loss_now = getattr(model, 'loss_now', None)
            self.converged = getattr(model, '_converged', False)
        
        # Sync state-space parameters if available
        if hasattr(model, 'training_state') and model.training_state is not None:
            ts = model.training_state
            if hasattr(ts, 'F'):
                self.F = ts.F
                self.Q = ts.Q
                self.mu_0 = ts.mu_0
                self.Sigma_0 = ts.Sigma_0
                self.H = ts.H
                self.R = ts.R
        
        return self
    
    def apply_to_model(self, model: Any) -> None:
        """Apply this state to DDFM model instance.
        
        Parameters
        ----------
        model : DDFM
            DDFM model instance to update
        """
        # Apply training state
        if self.factors is not None:
            model.factors = self.factors
        if self.eps is not None:
            model.eps = self.eps
        if self.last_neurons is not None:
            model.last_neurons = self.last_neurons
        model._num_iter = self.num_iter
        model.loss_now = self.loss_now
        model._converged = self.converged
        
        # Apply state-space parameters if available
        if self.F is not None and self.Q is not None and self.mu_0 is not None:
            if hasattr(model, 'training_state'):
                model.training_state.F = self.F
                model.training_state.Q = self.Q
                model.training_state.mu_0 = self.mu_0
                model.training_state.Sigma_0 = self.Sigma_0
                model.training_state.H = self.H
                model.training_state.R = self.R

# Backward compatibility aliases
DDFMTrainingState = DDFMModelState


@dataclass
class iVDFMModelState:
    """iVDFM model state including training state and SSM parameters.
    
    Consolidates all iVDFM model state for checkpointing: training state (convergence,
    loss, factors, innovations) and fitted state-space model parameters.
    
    Attributes
    ----------
    num_iter : int
        Current training iteration/epoch.
    loss_now : float, optional
        Current training loss value.
    elbo : float, optional
        Final ELBO value.
    converged : bool
        Whether training has converged.
    factors : np.ndarray, optional
        Extracted factors (T x r) or (batch x T x r).
    innovations : np.ndarray, optional
        Innovations (T x r) or (batch x T x r).
    factor_order : int
        AR order for factor dynamics (p in AR(p)).
    A : np.ndarray, optional
        Transition matrix (r x r) for p=1, or AR coefficients (r x p) for p>1.
    B : np.ndarray, optional
        Innovation loading matrix (r x r).
    f0 : np.ndarray, optional
        Initial state (r,).
    """
    # Training state
    num_iter: int = 0
    loss_now: Optional[float] = None
    elbo: Optional[float] = None
    converged: bool = False
    factors: Optional[np.ndarray] = None
    innovations: Optional[np.ndarray] = None
    full_state: Optional[np.ndarray] = None  # Augmented state for companion form (T x r*p) for p>1, (T x r) for p=1
    
    # SSM parameters
    factor_order: int = 1
    A: Optional[np.ndarray] = None  # Transition matrix or AR coefficients
    B: Optional[np.ndarray] = None  # Innovation loading matrix
    f0: Optional[np.ndarray] = None  # Initial state
    
    @classmethod
    def from_model(cls, model: Any) -> 'iVDFMModelState':
        """Create iVDFMModelState from iVDFM model instance.
        
        Parameters
        ----------
        model : iVDFM
            iVDFM model instance
            
        Returns
        -------
        iVDFMModelState
            Model state dataclass
        """
        # Get training state from model attributes
        factors = getattr(model, 'factors', None)
        innovations = getattr(model, 'innovations', None)
        num_iter = getattr(model, '_num_iter', 0)
        loss_now = getattr(model, 'loss_now', None)
        elbo = getattr(model, '_elbo', None)
        converged = getattr(model, '_converged', False)
        
        # Get SSM parameters
        factor_order = getattr(model, 'factor_order', 1)
        A = B = f0 = None
        
        # Extract from SSM if available
        if hasattr(model, 'ssm') and model.ssm is not None:
            ssm = model.ssm
            if factor_order == 1:
                # First-order: A is diagonal, stored as (r,)
                if hasattr(ssm, 'A'):
                    A = ssm.A.data.cpu().numpy() if hasattr(ssm.A, 'data') else ssm.A
            else:
                # Higher-order: AR coefficients stored as (r, p)
                if hasattr(ssm, 'ar_coeffs'):
                    A = ssm.ar_coeffs.data.cpu().numpy() if hasattr(ssm.ar_coeffs, 'data') else ssm.ar_coeffs
            
            if hasattr(ssm, 'B'):
                B = ssm.B.data.cpu().numpy() if hasattr(ssm.B, 'data') else ssm.B
            
            if hasattr(ssm, 'f0'):
                f0 = ssm.f0.data.cpu().numpy() if hasattr(ssm.f0, 'data') else ssm.f0
        
        # Fallback: get from model attributes directly
        if A is None:
            A = getattr(model, 'A', None)
            if A is not None and hasattr(A, 'data'):
                A = A.data.cpu().numpy()
        if B is None:
            B = getattr(model, 'B', None)
            if B is not None and hasattr(B, 'data'):
                B = B.data.cpu().numpy()
        if f0 is None:
            f0 = getattr(model, 'f0', None)
            if f0 is not None and hasattr(f0, 'data'):
                f0 = f0.data.cpu().numpy()
        
        # Compute full_state (augmented state for companion form)
        full_state = None
        if factors is not None:
            factors_np = factors
            if isinstance(factors_np, np.ndarray):
                # Average over batch if needed
                if factors_np.ndim == 3:
                    factors_np = np.mean(factors_np, axis=0)  # (T, r)
                
                T, r = factors_np.shape
                if factor_order == 1:
                    # For p=1, full_state is just factors
                    full_state = factors_np  # (T, r)
                else:
                    # For p>1, construct augmented state (T, r*p)
                    # s_t[i*p : (i+1)*p] = [f_t[i], f_{t-1}[i], ..., f_{t-p+1}[i]]
                    full_state = np.zeros((T, r * factor_order), dtype=factors_np.dtype)
                    for t in range(T):
                        for i in range(r):
                            for lag in range(factor_order):
                                idx = t - lag
                                if idx >= 0:
                                    full_state[t, i * factor_order + lag] = factors_np[idx, i]
                                else:
                                    # Use f0 for negative indices
                                    if f0 is not None and i < len(f0):
                                        full_state[t, i * factor_order + lag] = f0[i]
        
        return cls(
            num_iter=num_iter,
            loss_now=loss_now,
            elbo=elbo,
            converged=converged,
            factors=factors,
            innovations=innovations,
            full_state=full_state,
            factor_order=factor_order,
            A=A,
            B=B,
            f0=f0
        )
    
    def apply_to_model(self, model: Any) -> None:
        """Apply this state to iVDFM model instance.
        
        Parameters
        ----------
        model : iVDFM
            iVDFM model instance to update
        """
        # Apply training state
        if self.factors is not None:
            model.factors = self.factors
        if self.innovations is not None:
            model.innovations = self.innovations
        model._num_iter = self.num_iter
        model.loss_now = self.loss_now
        if self.elbo is not None:
            model._elbo = self.elbo
        model._converged = self.converged
        
        # Apply SSM parameters if available
        if self.A is not None and self.B is not None:
            if hasattr(model, 'ssm') and model.ssm is not None:
                ssm = model.ssm
                if self.factor_order == 1:
                    if hasattr(ssm, 'A'):
                        ssm.A.data = torch.from_numpy(self.A).to(ssm.A.device)
                else:
                    if hasattr(ssm, 'ar_coeffs'):
                        ssm.ar_coeffs.data = torch.from_numpy(self.A).to(ssm.ar_coeffs.device)
                
                if hasattr(ssm, 'B'):
                    ssm.B.data = torch.from_numpy(self.B).to(ssm.B.device)
                
                if self.f0 is not None and hasattr(ssm, 'f0'):
                    ssm.f0.data = torch.from_numpy(self.f0).to(ssm.f0.device)
DDFMStateSpaceParams = DDFMModelState

# Backward compatibility alias
DFMStateSpaceParams = DFMModelState


@dataclass
class AFMModelState:
    """Attention Factor Model training state for checkpointing."""
    num_iter: int = 0
    converged: bool = False
    loss_now: Optional[float] = None
    sharpe: Optional[float] = None
    explained_variance: Optional[float] = None
    residuals: Optional[np.ndarray] = None       # (B, T, N) traded residual portfolios
    book_returns: Optional[np.ndarray] = None    # (B, n_eval) after-cost book returns

    @classmethod
    def from_model(cls, model: Any) -> 'AFMModelState':
        """Create state from a trained AFM instance."""
        return cls(
            num_iter=getattr(model, '_num_iter', 0),
            converged=getattr(model, '_converged', False),
            loss_now=getattr(model, 'loss_now', None),
            sharpe=getattr(model, '_sharpe', None),
            explained_variance=getattr(model, '_ev', None),
            residuals=getattr(model, '_residuals', None),
            book_returns=getattr(model, '_book_returns', None),
        )

    def apply_to_model(self, model: Any) -> None:
        """Restore training diagnostics onto a model instance."""
        model._num_iter = self.num_iter
        model._converged = self.converged
        model.loss_now = self.loss_now
        model._sharpe = self.sharpe
        model._ev = self.explained_variance
        model._residuals = self.residuals
        model._book_returns = self.book_returns


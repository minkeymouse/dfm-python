"""Synthetic Data Generating Process (DGP) for testing factor models.

This module provides a synthetic DGP based on the Monte Carlo exercise from
Banbura & Modugno (2010), augmented with polynomial and sign features for
testing Deep Dynamic Factor Models.

Reference:
    Banbura, Marta and Modugno, Michele,
    "Maximum Likelihood Estimation of Factor Models on Data Sets with 
    Arbitrary Pattern of Missing Data"
    ECB Working Paper No. 1189, 2010.
"""

import numpy as np
from typing import Optional, Tuple

try:
    from sklearn.preprocessing import PolynomialFeatures
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    PolynomialFeatures = None


class SyntheticDGP:
    """Synthetic Data Generating Process for factor models.
    
    Simulates data from a factor model with:
    - Common factors following VAR(1) dynamics
    - Optional polynomial transformations of factors
    - Optional sign-based features
    - Idiosyncratic components with serial correlation
    - Configurable missing data patterns
    
    The DGP is based on Banbura & Modugno (2010), extended to support
    nonlinear factor structures for testing Deep Dynamic Factor Models.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    n : int, default=10
        Number of observable series
    r : int, default=1
        Number of common factors (linear)
    poly_degree : int, default=1
        Polynomial degree for factor transformation (1=linear, 2=quadratic, etc.)
    sign_features : int, default=0
        Number of sign-based features to add (0=none)
    rho : float, default=0.7
        Serial correlation parameter for common factors
    alpha : float, default=0.2
        Serial correlation parameter for idiosyncratic components
    u : float, default=0.1
        Signal-to-noise ratio parameter (lower = more signal)
    tau : float, default=0.0
        Cross-correlation parameter for idiosyncratic components
    """
    
    def __init__(
        self,
        seed: int,
        n: int = 10,
        r: int = 1,
        poly_degree: int = 1,
        sign_features: int = 0,
        rho: float = 0.7,
        alpha: float = 0.2,
        u: float = 0.1,
        tau: float = 0.0,
    ):
        self.rng = np.random.RandomState(seed)
        self.n = n
        self.r = r
        self.poly_degree = poly_degree
        self.sign_features = sign_features
        self.rho = rho
        self.alpha = alpha
        self.u = u
        self.tau = tau
        
        # Store generated factors
        self.linear_f: Optional[np.ndarray] = None  # Linear factors (r x T)
        self.f: Optional[np.ndarray] = None  # Transformed factors (r_transformed x T)
        self.lambda_matrix: Optional[np.ndarray] = None  # Loading matrix (n x r_transformed)
        self.eps: Optional[np.ndarray] = None  # Idiosyncratic components (T x n)
    
    def simulate(
        self,
        t_obs: int,
        portion_missings: float = 0.0,
    ) -> np.ndarray:
        """Simulate data from the DGP.
        
        Parameters
        ----------
        t_obs : int
            Number of time observations to simulate
        portion_missings : float, default=0.0
            Proportion of missing data (0.0 to 1.0)
            
        Returns
        -------
        np.ndarray
            Simulated observable variables (T x N)
            May contain NaN values if portion_missings > 0
        """
        # Step 1: Generate common factors (VAR(1))
        u_t = self.rng.multivariate_normal(
            np.zeros(self.r),
            np.eye(self.r),
            t_obs
        )
        A = np.diag(self.rho * np.ones(self.r))
        
        f_linear = np.zeros((t_obs, self.r))
        for t in range(t_obs):
            if t > 0:
                f_linear[t, :] = f_linear[t - 1, :] @ A + u_t[t, :]
            else:
                f_linear[t, :] = u_t[t, :]
        
        self.linear_f = f_linear.copy()
        
        # Step 2: Apply polynomial transformation if needed
        if self.poly_degree > 1:
            if not _has_sklearn:
                raise ImportError(
                    "sklearn is required for polynomial transformations. "
                    "Install with: pip install scikit-learn"
                )
            poly = PolynomialFeatures(self.poly_degree, include_bias=False)
            f_transformed = poly.fit_transform(f_linear)
        else:
            f_transformed = f_linear.copy()
        
        # Step 3: Add sign-based features if requested
        if self.sign_features > 0:
            sign_features = np.sign(f_linear[:, :self.sign_features])
            f_transformed = np.hstack([f_transformed, sign_features])
        
        self.f = f_transformed.copy()
        r_transformed = f_transformed.shape[1]
        
        # Step 4: Generate loading matrix
        Lambda = self.rng.multivariate_normal(
            np.zeros(r_transformed),
            np.eye(r_transformed),
            self.n
        )
        self.lambda_matrix = Lambda
        
        # Step 5: Generate idiosyncratic components
        # Compute variance parameters
        beta = self.rng.uniform(self.u, 1 - self.u, self.n)
        gamma = np.zeros(self.n)
        for i in range(self.n):
            gamma[i] = (
                beta[i] / (1 - beta[i]) * 
                1 / (1 - self.alpha ** 2) * 
                np.sum(Lambda[i, :] ** 2)
            )
        
        # Cross-correlation matrix for idiosyncratic components
        phi = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                phi[i, j] = (
                    (self.tau ** np.abs(i - j)) * 
                    (1 - self.alpha ** 2) * 
                    np.sqrt(gamma[i] * gamma[j])
                )
        
        v_t = self.rng.multivariate_normal(np.zeros(self.n), phi, t_obs)
        D = np.diag(self.alpha * np.ones(self.n))
        
        eps = np.zeros((t_obs, self.n))
        for t in range(t_obs):
            if t > 0:
                eps[t, :] = eps[t - 1, :] @ D + v_t[t, :]
            else:
                eps[t, :] = v_t[t, :]
        
        self.eps = eps
        
        # Step 6: Generate observables
        x = f_transformed @ Lambda.T + eps
        
        # Step 7: Insert missing values if requested
        if portion_missings > 0:
            if not (0 < portion_missings < 1):
                raise ValueError("portion_missings must be between 0 and 1")
            
            n_missings = int(t_obs * self.n * portion_missings)
            missing_locations = set()
            
            while len(missing_locations) < n_missings:
                row = self.rng.choice(t_obs)
                col = self.rng.choice(self.n)
                if (row, col) not in missing_locations:
                    missing_locations.add((row, col))
                    x[row, col] = np.nan
        
        return x
    
    def evaluate(
        self,
        f_hat: np.ndarray,
        f_true: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate factor recovery using trace R².
        
        Computes the trace R² metric between estimated and true factors.
        This metric measures how well the estimated factors span the space
        of the true factors.
        
        Parameters
        ----------
        f_hat : np.ndarray
            Estimated factors (T x r_hat)
        f_true : np.ndarray, optional
            True factors (T x r_true). If None, uses self.f
            
        Returns
        -------
        float
            Trace R² score (0 to 1, higher is better)
        """
        if f_true is None:
            f_true = self.f
        
        if f_true is None:
            raise ValueError("No true factors available. Run simulate() first or provide f_true.")
        
        # Ensure same number of observations
        T = min(f_hat.shape[0], f_true.shape[0])
        f_hat = f_hat[:T, :]
        f_true = f_true[:T, :]
        
        # Compute trace R²
        # R² = trace(F_true' @ F_hat @ (F_hat' @ F_hat)^-1 @ F_hat' @ F_true) / trace(F_true' @ F_true)
        try:
            numerator = np.trace(
                f_true.T @ f_hat @ 
                np.linalg.pinv(f_hat.T @ f_hat) @ 
                f_hat.T @ f_true
            )
            denominator = np.trace(f_true.T @ f_true)
            
            if denominator < 1e-10:
                return 0.0
            
            return numerator / denominator
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
    
    def get_true_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get true linear and transformed factors.
        
        Returns
        -------
        linear_f : np.ndarray
            True linear factors (T x r)
        f : np.ndarray
            True transformed factors (T x r_transformed)
        """
        if self.linear_f is None or self.f is None:
            raise ValueError("No factors available. Run simulate() first.")
        
        return self.linear_f, self.f


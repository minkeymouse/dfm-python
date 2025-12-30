"""Companion SSM for Kernelized Dynamic Factor Model.

Implements companion form state-space models for VAR (AR) and VARMA (MA) stages.
All companion features (base class, AR, MA) are in this single file.
"""

from typing import Optional, Literal, Tuple, Union, Any, cast
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..utils.errors import ConfigurationError, NumericalError, NumericalStabilityError
from ..config.types import Tensor, Device, Shape2D, Shape3D, NumVars, LagOrder, OptionalTensor
from ..numeric.validator import (
    validate_eigenvalue_bounds,
    validate_matrix_condition,
    validate_no_nan_inf
)


class CompanionSSMBase(nn.Module):
    """Base class for companion SSM implementations.
    
    Provides common initialization, normalization, and forward pass logic
    shared between AR and MA companion SSMs.
    """
    
    # Default initialization constants
    DEFAULT_INIT_SCALE = 0.01
    DEFAULT_KERNEL_INIT_SCALE = 0.1
    DEFAULT_MIN_NORM = 1e-4
    DEFAULT_EPS = 1e-8
    
    def __init__(
        self,
        n_vars: int,
        order: int,
        n_kernels: int = 1,
        kernel_init: Literal['normal', 'xavier'] = 'normal',
        norm_order: int = 1,
        init_scale: Optional[float] = None,
        kernel_init_scale: Optional[float] = None,
        min_norm: Optional[float] = None,
        eps: Optional[float] = None
    ) -> None:
        """Initialize base companion SSM.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (K)
        order : int
            Lag order (p for AR, q for MA)
        n_kernels : int, default=1
            Number of kernels/heads
        kernel_init : str, default='normal'
            Initialization method: 'normal' or 'xavier'
        norm_order : int, default=1
            Norm order for normalization (0 = no normalization)
        init_scale : float, optional
            Initialization scale for B and C matrices. Defaults to DEFAULT_INIT_SCALE.
        kernel_init_scale : float, optional
            Initialization scale for coefficient matrices. Defaults to DEFAULT_KERNEL_INIT_SCALE.
        min_norm : float, optional
            Minimum norm threshold. Defaults to DEFAULT_MIN_NORM.
        eps : float, optional
            Epsilon for numerical stability. Defaults to DEFAULT_EPS.
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.order = order
        self.n_kernels = n_kernels
        self.kernel_init = kernel_init
        self.norm_order = norm_order
        self.latent_dim = order * n_vars
        
        # Initialization constants (use defaults if not provided)
        self.init_scale = init_scale if init_scale is not None else self.DEFAULT_INIT_SCALE
        self.kernel_init_scale = kernel_init_scale if kernel_init_scale is not None else self.DEFAULT_KERNEL_INIT_SCALE
        self.min_norm = min_norm if min_norm is not None else self.DEFAULT_MIN_NORM
        self.eps = eps if eps is not None else self.DEFAULT_EPS
    
    def _build_shift_matrix(self) -> torch.Tensor:
        """Build shift matrix with identity blocks on sub-diagonal.
        
        Returns
        -------
        shift_matrix : torch.Tensor
            Shift matrix of shape (n_kernels, latent_dim, latent_dim)
        """
        shift_matrix = torch.zeros(self.n_kernels, self.latent_dim, self.latent_dim)
        for i in range(1, self.order):
            start_row = i * self.n_vars
            end_row = (i + 1) * self.n_vars
            start_col = (i - 1) * self.n_vars
            end_col = i * self.n_vars
            shift_matrix[:, start_row:end_row, start_col:end_col] = \
                torch.eye(self.n_vars).unsqueeze(0)
        return shift_matrix
    
    def _init_kernel_weights(self, size: int) -> torch.Tensor:
        """Initialize kernel weights.
        
        Parameters
        ----------
        size : int
            Total size of kernel weights
            
        Returns
        -------
        weights : torch.Tensor
            Initialized weights of shape (n_kernels, size)
        """
        if self.kernel_init == 'normal':
            return torch.randn(self.n_kernels, size) * self.kernel_init_scale
        elif self.kernel_init == 'xavier':
            stdv = 1.0 / (size ** 0.5)
            return torch.empty(self.n_kernels, size).uniform_(-stdv, stdv)
        else:
            raise ConfigurationError(
                f"Unknown kernel_init: {self.kernel_init}. "
                f"Must be 'normal' or 'xavier'."
            )
    
    def _init_b_matrix(self) -> torch.Tensor:
        """Initialize B matrix with identity in first block.
        
        Returns
        -------
        b : torch.Tensor
            B matrix of shape (n_kernels, latent_dim, n_vars)
        """
        b = torch.zeros(self.n_kernels, self.latent_dim, self.n_vars)
        b[:, :self.n_vars, :] = torch.eye(self.n_vars).unsqueeze(0)
        b = b + torch.randn_like(b) * self.init_scale
        return b
    
    def _init_c_matrix(self) -> torch.Tensor:
        """Initialize C matrix with identity in first block.
        
        Returns
        -------
        c : torch.Tensor
            C matrix of shape (n_kernels, n_vars, latent_dim)
        """
        c = torch.zeros(self.n_kernels, self.n_vars, self.latent_dim)
        c[:, :, :self.n_vars] = torch.eye(self.n_vars).unsqueeze(0)
        c = c + torch.randn_like(c) * self.init_scale
        return c
    
    def norm(self, x: torch.Tensor, ord: Optional[int] = None) -> torch.Tensor:
        """Normalize tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor to normalize
        ord : int, optional
            Norm order. If None, uses self.norm_order.
            
        Returns
        -------
        x_norm : torch.Tensor
            Normalized tensor
        """
        if ord is None:
            ord = self.norm_order
        if ord == 0:
            return x
        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mean_norm = torch.abs(x_norm).mean()
        if mean_norm > self.min_norm:
            x = x / (x_norm + self.eps)
        return x
    
    def get_kernel(
        self, 
        u: Tensor, 
        c: Optional[Tensor] = None, 
        l: Optional[int] = None
    ) -> Tensor:
        """Get impulse response kernel using Krylov method.
        
        Computes the impulse response kernel K_h = C @ A^h @ B efficiently
        using Krylov subspace methods with FFT convolution.
        
        Parameters
        ----------
        u : torch.Tensor
            Input of shape (B, D, L) or (B, L, D) where:
            - B: batch size
            - D: dimension (n_vars or latent_dim)
            - L: sequence length
        c : torch.Tensor, optional
            Output matrix C of shape (n_vars, latent_dim).
            If None, uses self.C[0] (first kernel's C matrix)
        l : int, optional
            Kernel length (number of time steps).
            If None, uses u.shape[-1] (sequence length)
            
        Returns
        -------
        kernel : torch.Tensor
            Impulse response kernel of shape (K, l) or (l,) where:
            - K: number of variables (if c provided) or latent_dim
            - l: kernel length
            
        Raises
        ------
        NumericalError
            If Krylov computation fails or produces invalid results
        """
        if l is None:
            l = u.shape[-1]
        
        if c is None:
            c = self.C[0]
        
        # Get coefficient parameter (subclasses should override get_coefficient_param)
        coeff = self.get_coefficient_param()
        coeff_norm = self.norm(coeff, ord=self.norm_order) if self.norm_order > 0 else coeff
        A = self.get_companion_matrix(coeff_norm)
        b = self.B[0]
        
        # Handle both 2D and 3D companion matrices
        # If A is 3D (n_kernels > 1), extract first kernel; if 2D (n_kernels == 1), use as-is
        if A.ndim == 3:
            A = A[0]  # Extract first kernel: (latent_dim, latent_dim)
        
        # Lazy import to avoid circular dependency
        try:
            from ..functional.krylov import krylov
            kernel = krylov(l, A, b, c=c)
            
            # Validate kernel output
            if torch.any(torch.isnan(kernel)) or torch.any(torch.isinf(kernel)):
                raise NumericalError(
                    "Krylov kernel computation produced NaN/Inf values.",
                    details=(
                        f"Kernel shape: {kernel.shape}, "
                        f"A shape: {A.shape}, "
                        f"b shape: {b.shape}, "
                        f"c shape: {c.shape if c is not None else None}. "
                        f"Consider: (1) Regularization, (2) Checking companion matrix stability, "
                        f"(3) Lower initialization scale."
                    )
                )
            
            return kernel
        except ImportError as e:
            raise NumericalError(
                "Krylov module not available. Cannot compute impulse response kernel.",
                details=str(e)
            ) from e
        except Exception as e:
            raise NumericalError(
                "Failed to compute impulse response kernel using Krylov method.",
                details=f"Error: {str(e)}, l={l}, A shape={A.shape if A is not None else None}"
            ) from e
    
    def fft_conv(self, u_input: Tensor, v_kernel: Tensor) -> Tensor:
        """Convolve u with v in O(n log n) time with FFT (n = len(u)).
        
        This method uses Fast Fourier Transform (FFT) to compute convolution
        efficiently, achieving O(n log n) complexity instead of O(n²) for
        direct convolution. This is a key efficiency advantage of KDFM.
        
        Parameters
        ----------
        u_input : torch.Tensor
            Input tensor of shape (B, H, L) or (B, L, H) where:
            - B: Batch size
            - H: Number of heads/kernels
            - L: Sequence length
        v_kernel : torch.Tensor
            Kernel tensor of shape (H, L) where:
            - H: Number of heads/kernels (must match u_input)
            - L: Sequence length
            
        Returns
        -------
        y : torch.Tensor
            Convolved output of shape (B, H, L) where:
            - B: Batch size
            - H: Number of heads/kernels
            - L: Sequence length
            
        Examples
        --------
        >>> ssm = CompanionSSM(n_vars=5, lag_order=1)
        >>> u = torch.randn(2, 1, 100)  # (B=2, H=1, L=100)
        >>> v = torch.randn(1, 100)  # (H=1, L=100)
        >>> y = ssm.fft_conv(u, v)
        >>> assert y.shape == (2, 1, 100)
        """
        # Ensure u is (B, H, L)
        if u_input.dim() == 3 and u_input.shape[1] != v_kernel.shape[0]:
            u_input = rearrange(u_input, 'b l h -> b h l')
        
        L = u_input.shape[-1]
        u_f = torch.fft.rfft(u_input, n=2*L)
        v_f = torch.fft.rfft(v_kernel[:, :L], n=2*L)
        
        y_f = torch.einsum('b h l, h l -> b h l', u_f, v_f)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]
        return y
    
    def forward(self, u: Tensor) -> Tensor:
        """Forward pass through companion SSM.
        
        This method performs the forward pass through the companion SSM,
        computing the output by convolving the input with the impulse response
        kernel. The computation uses FFT for O(n log n) efficiency.
        
        Parameters
        ----------
        u : torch.Tensor
            Input tensor of shape (B, L, D) where:
            - B: Batch size
            - L: Sequence length
            - D: Input dimension (number of variables)
            
        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (B, L, D) where:
            - B: Batch size
            - L: Sequence length
            - D: Output dimension (matches input dimension)
            
        Raises
        ------
        NumericalError
            If kernel computation fails due to numerical instability
            
        Examples
        --------
        >>> ssm = CompanionSSM(n_vars=5, lag_order=1)
        >>> u = torch.randn(2, 100, 5)  # (B=2, L=100, D=5)
        >>> y = ssm(u)
        >>> assert y.shape == (2, 100, 5)
        """
        # Rearrange to (B, D, L) for convolution
        u = rearrange(u, 'b l d -> b d l')
        
        # Get kernel
        kernel = self.get_kernel(u)
        
        # Convolve
        y = self.fft_conv(u, kernel)
        
        # Rearrange back to (B, L, D)
        y = rearrange(y, 'b d l -> b l d')
        
        return y
    
    def get_coefficient_param(self) -> Tensor:
        """Get coefficient parameter tensor.
        
        Subclasses must implement this to return the appropriate parameter
        ('a' for AR, 'm' for MA).
        
        Returns
        -------
        coeff : Tensor
            Coefficient parameter tensor
        """
        raise NotImplementedError("Subclasses must implement get_coefficient_param")
    
    def get_companion_matrix(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Construct companion matrix from coefficients.
        
        Subclasses must implement this to construct the companion matrix
        from their specific coefficient parameter. The companion matrix
        enables direct IRF computation via matrix powers, which is KDFM's
        PRIMARY CONTRIBUTION.
        
        Parameters
        ----------
        coeff : Tensor, optional
            Coefficient tensor. If None, uses get_coefficient_param().
            Shape depends on subclass implementation.
            
        Returns
        -------
        Tensor
            Companion matrix of shape (n_kernels, latent_dim, latent_dim) where:
            - n_kernels: Number of kernels/heads
            - latent_dim: Latent dimension (order * n_vars)
            The companion matrix structure enables direct IRF computation:
            K_h = C (A^{AR})^h B for reduced-form IRF.
            
        Raises
        ------
        NotImplementedError
            If subclass does not implement this method (abstract method).
        NumericalError
            If companion matrix construction fails or produces invalid values.
            
        Examples
        --------
        >>> ssm = CompanionSSM(n_vars=5, lag_order=1)
        >>> A = ssm.get_companion_matrix()
        >>> assert A.shape == (1, 5, 5)  # (n_kernels=1, latent_dim=5, latent_dim=5)
        """
        raise NotImplementedError("Subclasses must implement get_companion_matrix")
    
    def extract_coefficients(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Extract coefficients from learned parameters.
        
        Subclasses must implement this to extract their specific coefficients
        (VAR coefficients for AR, MA coefficients for MA). This enables
        direct coefficient extraction for interpretability, which is one of
        KDFM's key advantages.
        
        Parameters
        ----------
        coeff : Tensor, optional
            Coefficient tensor. If None, uses get_coefficient_param().
            Shape depends on subclass implementation.
            
        Returns
        -------
        Tensor
            Extracted coefficients of shape (order, n_vars, n_vars) where:
            - order: Lag order (p for AR, q for MA)
            - n_vars: Number of variables (K)
            These coefficients can be directly interpreted as VAR/MA coefficients,
            maintaining full interpretability (KDFM's explainability advantage).
            
        Raises
        ------
        NotImplementedError
            If subclass does not implement this method (abstract method).
        NumericalError
            If coefficient extraction fails or produces invalid values.
            
        Examples
        --------
        >>> ssm = CompanionSSM(n_vars=5, lag_order=1)
        >>> coeffs = ssm.extract_coefficients()
        >>> assert coeffs.shape == (1, 5, 5)  # (order=1, n_vars=5, n_vars=5)
        """
        raise NotImplementedError("Subclasses must implement extract_coefficients")


class CompanionSSM(CompanionSSMBase):
    """Companion SSM for VAR coefficient learning (AR stage).
    
    This SSM learns parameters a, b, c that form a companion matrix structure.
    It computes efficient kernels via Krylov method and uses FFT convolution.
    Learns efficient parameterization while maintaining companion structure for interpretability.
    """
    
    def __init__(
        self,
        n_vars: int,
        lag_order: int,
        n_kernels: int = 1,
        kernel_init: Literal['normal', 'xavier'] = 'normal',
        norm_order: int = 1,
        **kwargs: Any
    ) -> None:
        """Initialize Companion SSM.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (K)
        lag_order : int
            Lag order (p)
        n_kernels : int, default=1
            Number of kernels/heads
        kernel_init : str, default='normal'
            Initialization method: 'normal' or 'xavier'
        norm_order : int, default=1
            Norm order for normalization (0 = no normalization)
        **kwargs
            Additional arguments passed to CompanionSSMBase
        """
        super().__init__(
            n_vars=n_vars,
            order=lag_order,
            n_kernels=n_kernels,
            kernel_init=kernel_init,
            norm_order=norm_order,
            **kwargs
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize companion matrix components."""
        # Shift matrix: identity blocks on sub-diagonal
        shift_matrix = self._build_shift_matrix()
        self.register_buffer('shift_matrix', shift_matrix)
        
        # VAR coefficients a = [A_1, ..., A_p], shape: (n_kernels, p*K*K)
        a = self._init_kernel_weights(self.order * self.n_vars * self.n_vars)
        self.register_parameter('a', nn.Parameter(a))
        
        # B and C matrices
        b = self._init_b_matrix()
        c = self._init_c_matrix()
        self.register_parameter('B', nn.Parameter(b))
        self.register_parameter('C', nn.Parameter(c))
    
    def get_coefficient_param(self) -> Tensor:
        """Get VAR coefficient parameter.
        
        Returns the learnable VAR coefficient parameter tensor a, which contains
        the VAR coefficients A_1, ..., A_p in flattened form.
        
        Returns
        -------
        Tensor
            VAR coefficient parameter of shape (n_kernels, p*K*K) where:
            - n_kernels: Number of kernels/heads
            - p: Lag order
            - K: Number of variables
        """
        return self.a
    
    def _build_companion_from_coeffs(self, coeff: Tensor) -> Tensor:
        """Build companion matrix from coefficient tensor (shared logic).
        
        Parameters
        ----------
        coeff : torch.Tensor
            Coefficient tensor of shape (n_kernels, order*K*K)
            
        Returns
        -------
        torch.Tensor
            Companion matrix of shape (n_kernels, latent_dim, latent_dim)
        """
        # Reshape: (n_kernels, order*K*K) -> (n_kernels, order, K, K)
        coeff_reshaped = coeff.view(self.n_kernels, self.order, self.n_vars, self.n_vars)
        
        if self.norm_order > 0:
            coeff_reshaped = self.norm(coeff_reshaped, ord=self.norm_order)
        
        # Construct first K rows of companion matrix
        companion_top = torch.zeros(
            self.n_kernels, self.n_vars, self.latent_dim,
            device=coeff.device, dtype=coeff.dtype
        )
        for i in range(self.n_vars):
            for j in range(self.order):
                start_col = j * self.n_vars
                end_col = (j + 1) * self.n_vars
                companion_top[:, i, start_col:end_col] = coeff_reshaped[:, j, i, :]
        
        # Combine with shift matrix
        A = self.shift_matrix.clone()
        A[:, :self.n_vars, :] = companion_top
        return A
    
    def get_companion_matrix(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Construct companion matrix from VAR coefficients.
        
        This method builds the companion matrix A from VAR coefficients A_1, ..., A_p.
        The companion matrix has shape (pK, pK) where p is the VAR order and K is the
        number of variables.
        
        **Stability Check**: The companion matrix should have all eigenvalues < 1.0 for
        stability. If max eigenvalue >= 1.0, forecasts will explode.
        
        Parameters
        ----------
        coeff : Tensor, optional
            VAR coefficient parameter. If None, uses self.a (learned parameters).
            Shape: (n_kernels, p*K*K)
            
        Returns
        -------
        Tensor
            Companion matrix A of shape (n_kernels, pK, pK)
            
        Examples
        --------
        >>> companion_ssm = CompanionSSM(n_vars=3, lag_order=2)
        >>> A = companion_ssm.get_companion_matrix()
        >>> A.shape  # (1, 6, 6) for p=2, K=3
        torch.Size([1, 6, 6])
        >>> eigenvals = torch.linalg.eigvals(A[0])
        >>> max_eigenval = torch.max(torch.abs(eigenvals))
        >>> assert max_eigenval < 1.0, "Companion matrix is unstable"
        """
        if coeff is None:
            coeff = self.a
        return self._build_companion_from_coeffs(coeff)
    
    def check_stability(self, coeff: Optional[Tensor] = None, threshold: float = 1.0) -> Tuple[bool, float]:
        """Check if companion matrix is stable (all eigenvalues < threshold).
        
        This method computes the companion matrix and checks if its maximum eigenvalue
        is below the stability threshold (default 1.0). For forecasting, eigenvalues
        must be < 1.0 to prevent forecast explosion.
        
        Parameters
        ----------
        coeff : Tensor, optional
            VAR coefficient parameter. If None, uses self.a.
        threshold : float, default=1.0
            Stability threshold. Matrix is stable if max eigenvalue < threshold.
            Must be > 0.
            
        Returns
        -------
        tuple
            (is_stable, max_eigenvalue)
            - is_stable: bool, True if max eigenvalue < threshold
            - max_eigenvalue: float, maximum absolute eigenvalue
            
        Raises
        ------
        NumericalError
            If eigenvalue computation fails or matrix contains NaN/Inf
        ConfigurationError
            If threshold is invalid
            
        Examples
        --------
        >>> companion_ssm = CompanionSSM(n_vars=3, lag_order=2)
        >>> is_stable, max_eig = companion_ssm.check_stability()
        >>> if not is_stable:
        ...     print(f"Warning: Companion matrix is unstable (max eigenvalue = {max_eig:.6f})")
        """
        if threshold <= 0:
            raise ConfigurationError(
                f"Stability threshold must be > 0, got {threshold}",
                details="Threshold represents maximum allowed eigenvalue magnitude"
            )
        
        try:
            A = self.get_companion_matrix(coeff)
            
            # Validate matrix
            if A is None:
                raise NumericalError(
                    "Cannot check stability: companion matrix is None",
                    details="Ensure model has been properly initialized"
                )
            
            # Get first kernel for eigenvalue computation
            if A.ndim == 3:
                A_np = A[0].detach().cpu().numpy()  # (latent_dim, latent_dim)
            elif A.ndim == 2:
                A_np = A.detach().cpu().numpy()
            else:
                raise NumericalError(
                    f"Invalid companion matrix shape: {A.shape}",
                    details="Expected 2D or 3D tensor"
                )
            
            # Validate matrix content
            validate_no_nan_inf(A_np, name="companion matrix")
            
            # Compute eigenvalues
            try:
                eigenvals = np.linalg.eigvals(A_np)
            except (np.linalg.LinAlgError, ValueError) as e:
                raise NumericalError(
                    f"Eigenvalue computation failed: {e}",
                    details=(
                        f"Matrix shape: {A_np.shape}, "
                        f"Matrix may be singular or ill-conditioned. "
                        f"Consider regularization or checking initialization."
                    )
                ) from e
            
            # Validate eigenvalues
            validate_no_nan_inf(eigenvals, name="eigenvalues")
            
            max_eigenval = float(np.max(np.abs(eigenvals)))
            is_stable = max_eigenval < threshold
            
            return is_stable, max_eigenval
            
        except Exception as e:
            if isinstance(e, (NumericalError, ConfigurationError)):
                raise
            raise NumericalError(
                f"Stability check failed: {e}",
                details="Check that model components are properly initialized"
            ) from e
    
    def _extract_coeffs_reshaped(self, coeff: Tensor) -> Tensor:
        """Extract and reshape coefficients (shared logic).
        
        Parameters
        ----------
        coeff : torch.Tensor
            Coefficient tensor of shape (n_kernels, order*K*K)
            
        Returns
        -------
        torch.Tensor
            Reshaped coefficients of shape (order, n_vars, n_vars)
        """
        coeff_reshaped = coeff.view(self.n_kernels, self.order, self.n_vars, self.n_vars)
        if self.norm_order > 0:
            coeff_reshaped = self.norm(coeff_reshaped, ord=self.norm_order)
        return coeff_reshaped[0]  # (order, K, K)
    
    def extract_coefficients(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Extract VAR coefficients A_1, ..., A_p from learned parameters.
        
        This method extracts the VAR coefficients from the learned parameter tensor a,
        reshaping them into the standard form (p, K, K) where:
        - p: Lag order
        - K: Number of variables
        - A_i: VAR coefficient matrix for lag i
        
        Parameters
        ----------
        coeff : Tensor, optional
            Coefficient parameter tensor. If None, uses self.a (learned parameters).
            Shape: (n_kernels, p*K*K)
            
        Returns
        -------
        Tensor
            VAR coefficients of shape (p, K, K) where:
            - A[i, :, :] = VAR coefficient matrix A_{i+1} for lag i+1
            - Coefficients are normalized if norm_order > 0
        """
        if coeff is None:
            coeff = self.a
        return self._extract_coeffs_reshaped(coeff)
    
    def predict_from_var_coefficients(self, y_t: Tensor, A_coeffs: Optional[Tensor] = None) -> Tensor:
        """Predict using VAR coefficients.
        
        This method computes VAR predictions using the standard VAR formula:
        y_pred_t = A_1 y_{t-1} + A_2 y_{t-2} + ... + A_p y_{t-p}
        
        Uses vectorized operations (einsum) for efficiency, avoiding explicit loops.
        This is useful for extracting VAR coefficients and computing residuals.
        
        Parameters
        ----------
        y_t : Tensor
            Time series of shape (B, T, K) where:
            - B: Batch size
            - T: Sequence length
            - K: Number of variables
        A_coeffs : Tensor, optional
            VAR coefficients of shape (p, K, K) where:
            - p: Lag order
            - A_coeffs[i, :, :] = VAR coefficient matrix A_{i+1} for lag i+1
            If None, extracts from learned parameter a using extract_coefficients()
            
        Returns
        -------
        Tensor
            Predictions of shape (B, T, K) where:
            - First p time steps are zero-padded (cannot predict without lags)
            - Remaining time steps contain VAR predictions
            - Predictions are in the same scale as input y_t
            
        Examples
        --------
        >>> companion_ssm = CompanionSSM(n_vars=3, lag_order=2)
        >>> y_t = torch.randn(2, 100, 3)  # (B=2, T=100, K=3)
        >>> y_pred = companion_ssm.predict_from_var_coefficients(y_t)
        >>> assert y_pred.shape == (2, 100, 3)
        >>> assert torch.allclose(y_pred[:, :2, :], torch.zeros(2, 2, 3))  # First 2 steps zero
        """
        if A_coeffs is None:
            A_coeffs = self.extract_coefficients()
        
        B, T, K = y_t.shape
        p = self.order
        
        # Initialize predictions
        y_pred = torch.zeros(B, T, K, device=y_t.device, dtype=y_t.dtype)
        
        # Vectorized prediction for t >= p
        # Stack lagged values: (B, T-p, p, K)
        lagged = torch.stack([
            y_t[:, (p - i - 1):(T - i - 1), :] 
            for i in range(p)
        ], dim=2)  # (B, T-p, p, K)
        
        # Apply VAR coefficients: sum over lags
        # A_coeffs: (p, K, K), lagged: (B, T-p, p, K)
        # Compute: sum_i A_i @ y_{t-i-1} for each t
        predictions = torch.einsum('pij,btpj->bti', A_coeffs, lagged)  # (B, T-p, K)
        
        # Fill predictions (skip first p time steps)
        y_pred[:, p:, :] = predictions
        
        return y_pred
    
    def compute_residuals_from_coefficients(self, y_t: Tensor, A_coeffs: Optional[Tensor] = None) -> Tensor:
        """Compute reduced-form residuals from VAR coefficients.
        
        This method computes reduced-form residuals e_t = y_t - y_pred_t where
        y_pred_t is computed from VAR coefficients. Residuals are used for structural
        identification to obtain structural shocks ε_t.
        
        Parameters
        ----------
        y_t : Tensor
            Time series of shape (B, T, K)
        A_coeffs : Tensor, optional
            VAR coefficients of shape (p, K, K). If None, extracts from learned a.
            
        Returns
        -------
        Tensor
            Reduced-form residuals of shape (B, T-p, K)
            First p time steps are skipped (cannot compute residuals without lags)
        """
        if A_coeffs is None:
            A_coeffs = self.extract_coefficients()
        
        y_pred = self.predict_from_var_coefficients(y_t, A_coeffs)
        return y_t[:, self.order:, :] - y_pred[:, self.order:, :]  # (B, T-p, K)


class MACompanionSSM(CompanionSSMBase):
    """MA Companion SSM for learnable moving average structure (MA stage).
    
    This SSM learns MA coefficients M_1, ..., M_q through companion matrix structure,
    enabling VARMA(p,q) representation where residuals have moving average dynamics.
    
    The MA stage processes AR stage output z_t through a companion SSM structure,
    similar to the AR stage but for moving average components. This enables
    flexible VARMA modeling where both autoregressive and moving average dynamics
    are learnable through gradient descent.
    
    **Architecture**: The MA stage uses the same companion matrix structure as the
    AR stage, but with MA coefficients instead of VAR coefficients. This maintains
    the direct IRF computation capability: MA IRFs are computed as matrix powers
    from the MA companion matrix.
    """
    
    def __init__(
        self,
        n_vars: int,
        ma_order: int,
        n_kernels: int = 1,
        kernel_init: Literal['normal', 'xavier'] = 'normal',
        norm_order: int = 1,
        **kwargs
    ):
        """Initialize MA Companion SSM.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (K)
        ma_order : int
            MA order (q)
        n_kernels : int, default=1
            Number of kernels/heads
        kernel_init : str, default='normal'
            Initialization method: 'normal' or 'xavier'
        norm_order : int, default=1
            Norm order for normalization (0 = no normalization)
        **kwargs
            Additional arguments passed to CompanionSSMBase
        """
        super().__init__(
            n_vars=n_vars,
            order=ma_order,
            n_kernels=n_kernels,
            kernel_init=kernel_init,
            norm_order=norm_order,
            **kwargs
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize MA companion matrix components."""
        # Shift matrix: identity blocks on sub-diagonal
        shift_matrix = self._build_shift_matrix()
        self.register_buffer('shift_matrix', shift_matrix)
        
        # MA coefficients m = [M_1, ..., M_q], shape: (n_kernels, q*K*K)
        m = self._init_kernel_weights(self.order * self.n_vars * self.n_vars)
        self.register_parameter('m', nn.Parameter(m))
        
        # B and C matrices
        b = self._init_b_matrix()
        c = self._init_c_matrix()
        self.register_parameter('B', nn.Parameter(b))
        self.register_parameter('C', nn.Parameter(c))
    
    def get_coefficient_param(self) -> Tensor:
        """Get MA coefficient parameter.
        
        Returns the learnable MA coefficient parameter tensor m, which contains
        the MA coefficients M_1, ..., M_q in flattened form.
        
        Returns
        -------
        Tensor
            MA coefficient parameter of shape (n_kernels, q*K*K) where:
            - n_kernels: Number of kernels/heads
            - q: MA order
            - K: Number of variables
        """
        return self.m
    
    def get_companion_matrix(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Construct companion matrix from MA coefficients."""
        if coeff is None:
            coeff = self.m
        return self._build_companion_from_coeffs(coeff)
    
    def extract_coefficients(self, coeff: Optional[Tensor] = None) -> Tensor:
        """Extract MA coefficients M_1, ..., M_q from learned parameters.
        
        This method extracts the MA coefficients from the learned parameter tensor m,
        reshaping them into the standard form (q, K, K) where:
        - q: MA order
        - K: Number of variables
        - M_i: MA coefficient matrix for lag i
        
        Parameters
        ----------
        coeff : Tensor, optional
            Coefficient parameter tensor. If None, uses self.m (learned parameters).
            Shape: (n_kernels, q*K*K)
            
        Returns
        -------
        Tensor
            MA coefficients of shape (q, K, K) where:
            - M[i, :, :] = MA coefficient matrix M_{i+1} for lag i+1
            - Coefficients are normalized if norm_order > 0
        """
        if coeff is None:
            coeff = self.m
        return self._extract_coeffs_reshaped(coeff)
    
    def _build_companion_from_coeffs(self, coeff: Tensor) -> Tensor:
        """Build companion matrix from MA coefficients.
        
        Same logic as CompanionSSM but for MA coefficients.
        """
        # Reshape coefficients: (n_kernels, q*K*K) -> (n_kernels, q, K, K)
        coeff_reshaped = coeff.view(self.n_kernels, self.order, self.n_vars, self.n_vars)
        
        # Build companion matrix: (n_kernels, q*K, q*K)
        companion = torch.zeros(
            self.n_kernels, 
            self.latent_dim, 
            self.latent_dim,
            device=coeff.device,
            dtype=coeff.dtype
        )
        
        # Shift matrix (identity blocks on sub-diagonal)
        companion += self.shift_matrix
        
        # Add MA coefficient blocks in first block row
        for i in range(self.order):
            start_col = i * self.n_vars
            end_col = (i + 1) * self.n_vars
            companion[:, :self.n_vars, start_col:end_col] = coeff_reshaped[:, i, :, :]
        
        # Return consistent shape: always 3D for consistency with CompanionSSM
        # This ensures get_kernel() works correctly
        return companion
    
    def _extract_coeffs_reshaped(self, coeff: Tensor) -> Tensor:
        """Extract MA coefficients reshaped to (order, K, K).
        
        Same logic as CompanionSSM but for MA coefficients.
        """
        # Reshape: (n_kernels, q*K*K) -> (n_kernels, q, K, K)
        coeff_reshaped = coeff.view(self.n_kernels, self.order, self.n_vars, self.n_vars)
        return coeff_reshaped[0]  # (order, K, K)


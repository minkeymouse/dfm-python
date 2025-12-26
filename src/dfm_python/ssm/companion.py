"""Companion SSM for Kernelized Dynamic Factor Model.

Implements companion form state-space models for VAR (AR) and VARMA (MA) stages.
All companion features (base class, AR, MA) are in this single file.
"""

from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange


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
        kernel_init: str = 'normal',
        norm_order: int = 1,
        init_scale: Optional[float] = None,
        kernel_init_scale: Optional[float] = None,
        min_norm: Optional[float] = None,
        eps: Optional[float] = None
    ):
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
            raise ValueError(f"Unknown kernel_init: {self.kernel_init}")
    
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
        u: torch.Tensor, 
        c: Optional[torch.Tensor] = None, 
        l: Optional[int] = None
    ) -> torch.Tensor:
        """Get impulse response kernel using Krylov method.
        
        Parameters
        ----------
        u : torch.Tensor
            Input of shape (B, D, L) or (B, L, D)
        c : torch.Tensor, optional
            Output matrix C. If None, uses self.C[0]
        l : int, optional
            Kernel length. If None, uses u.shape[-1]
            
        Returns
        -------
        kernel : torch.Tensor
            Kernel of shape (K, l) or (l,)
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
        from ..models.functional.krylov import krylov
        return krylov(l, A, b, c=c)
    
    def fft_conv(self, u_input: torch.Tensor, v_kernel: torch.Tensor) -> torch.Tensor:
        """Convolve u with v in O(n log n) time with FFT (n = len(u)).
        
        Parameters
        ----------
        u_input : torch.Tensor
            Input of shape (B, H, L) or (B, L, H)
        v_kernel : torch.Tensor
            Kernel of shape (H, L)
            
        Returns
        -------
        y : torch.Tensor
            Convolved output of shape (B, H, L)
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
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass through companion SSM.
        
        Parameters
        ----------
        u : torch.Tensor
            Input of shape (B, L, D)
            
        Returns
        -------
        y : torch.Tensor
            Output of shape (B, L, D)
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
    
    def get_coefficient_param(self) -> torch.Tensor:
        """Get coefficient parameter tensor.
        
        Subclasses must implement this to return the appropriate parameter
        ('a' for AR, 'm' for MA).
        
        Returns
        -------
        coeff : torch.Tensor
            Coefficient parameter tensor
        """
        raise NotImplementedError("Subclasses must implement get_coefficient_param")
    
    def get_companion_matrix(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct companion matrix from coefficients.
        
        Subclasses must implement this to construct the companion matrix
        from their specific coefficient parameter.
        
        Parameters
        ----------
        coeff : torch.Tensor, optional
            Coefficient tensor. If None, uses get_coefficient_param()
            
        Returns
        -------
        A : torch.Tensor
            Companion matrix of shape (n_kernels, latent_dim, latent_dim)
        """
        raise NotImplementedError("Subclasses must implement get_companion_matrix")
    
    def extract_coefficients(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract coefficients from learned parameters.
        
        Subclasses must implement this to extract their specific coefficients
        (VAR coefficients for AR, MA coefficients for MA).
        
        Parameters
        ----------
        coeff : torch.Tensor, optional
            Coefficient tensor. If None, uses get_coefficient_param()
            
        Returns
        -------
        coeffs : torch.Tensor
            Extracted coefficients of shape (order, n_vars, n_vars)
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
        kernel_init: str = 'normal',
        norm_order: int = 1,
        **kwargs
    ):
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
    
    def get_coefficient_param(self) -> torch.Tensor:
        """Get VAR coefficient parameter."""
        return self.a
    
    def _build_companion_from_coeffs(self, coeff: torch.Tensor) -> torch.Tensor:
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
    
    def get_companion_matrix(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct companion matrix from VAR coefficients."""
        if coeff is None:
            coeff = self.a
        return self._build_companion_from_coeffs(coeff)
    
    def _extract_coeffs_reshaped(self, coeff: torch.Tensor) -> torch.Tensor:
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
    
    def extract_coefficients(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract VAR coefficients A_1, ..., A_p from learned parameters."""
        if coeff is None:
            coeff = self.a
        return self._extract_coeffs_reshaped(coeff)
    
    def predict_from_var_coefficients(self, y_t: torch.Tensor, A_coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict using VAR coefficients.
        
        Computes: y_pred_t = A_1 y_{t-1} + A_2 y_{t-2} + ... + A_p y_{t-p}
        Uses vectorized operations for efficiency.
        
        Parameters
        ----------
        y_t : torch.Tensor
            Time series of shape (B, T, K)
        A_coeffs : torch.Tensor, optional
            VAR coefficients of shape (p, K, K). If None, extracts from learned a.
            
        Returns
        -------
        y_pred : torch.Tensor
            Predictions of shape (B, T, K)
            First p time steps are zero-padded
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
    
    def compute_residuals_from_coefficients(self, y_t: torch.Tensor, A_coeffs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reduced-form residuals from VAR coefficients."""
        if A_coeffs is None:
            A_coeffs = self.extract_coefficients()
        
        y_pred = self.predict_from_var_coefficients(y_t, A_coeffs)
        return y_t[:, self.order:, :] - y_pred[:, self.order:, :]  # (B, T-p, K)


class MACompanionSSM(CompanionSSMBase):
    """MA Companion SSM for learnable moving average structure (MA stage).
    
    Learns MA coefficients M_1, ..., M_q through companion matrix structure,
    enabling VARMA(p,q) representation where residuals have MA dynamics.
    """
    
    def __init__(
        self,
        n_vars: int,
        ma_order: int,
        n_kernels: int = 1,
        kernel_init: str = 'normal',
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
    
    def get_coefficient_param(self) -> torch.Tensor:
        """Get MA coefficient parameter."""
        return self.m
    
    def get_companion_matrix(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct companion matrix from MA coefficients."""
        if coeff is None:
            coeff = self.m
        return self._build_companion_from_coeffs(coeff)
    
    def extract_coefficients(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract MA coefficients M_1, ..., M_q from learned parameters."""
        if coeff is None:
            coeff = self.m
        return self._extract_coeffs_reshaped(coeff)
    
    def _build_companion_from_coeffs(self, coeff: torch.Tensor) -> torch.Tensor:
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
    
    def _extract_coeffs_reshaped(self, coeff: torch.Tensor) -> torch.Tensor:
        """Extract MA coefficients reshaped to (order, K, K).
        
        Same logic as CompanionSSM but for MA coefficients.
        """
        # Reshape: (n_kernels, q*K*K) -> (n_kernels, q, K, K)
        coeff_reshaped = coeff.view(self.n_kernels, self.order, self.n_vars, self.n_vars)
        return coeff_reshaped[0]  # (order, K, K)


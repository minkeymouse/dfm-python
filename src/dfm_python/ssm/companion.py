"""Block-diagonal companion-form SSM of iVDFM (innovations to factors).

AR(1) factors use a direct elementwise recursion; AR(p > 1) factors use the
companion form with Krylov impulse responses and FFT convolution.
"""

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

from .base import BaseSSM
from ..functional.krylov import krylov
from ..numeric.builder import build_ivdfm_diagonal_companion

# norm() leaves a row (along the last dimension) whose norm is at or below this unnormalized.
_ZERO_ROW_NORM = 1e-12


class iVDFMCompanionSSM(BaseSSM):
    """Companion-form SSM with one block per factor (identifiable: diagonal A, B).

    For ``factor_order == 1`` the parameters are the diagonals ``A`` (through a
    sigmoid, so in (0, 1)) and ``B``; for ``factor_order > 1`` they are
    ``ar_coeffs`` (r, p), the diagonal ``B`` and ``C`` (r, r*p), which reads
    the first state of each block. ``f0`` is the learned initial state.

    Parameters
    ----------
    latent_dim : int
        Number of factors (r).
    factor_order : int, default 1
        AR order p of the factor dynamics.
    norm_order : int, default 1
        Order of the norm that ``B`` and the AR coefficients are divided by
        (0: no normalization).
    device : torch.device, optional
        Defaults to CUDA when available, else CPU.
    init_scale : float, default 0.1
        Standard deviation of the random initial ``A``/``ar_coeffs`` and ``B``.
    """

    def __init__(
        self,
        latent_dim: int,
        factor_order: int = 1,
        norm_order: int = 1,
        device: Optional[torch.device] = None,
        init_scale: float = 0.1,
    ):
        super().__init__(latent_dim, factor_order, device)
        self.norm_order = norm_order

        if self.factor_order == 1:
            self.A = nn.Parameter(torch.randn(latent_dim) * init_scale)
            self.B = nn.Parameter(torch.randn(latent_dim) * init_scale)
        else:
            self.ar_coeffs = nn.Parameter(
                torch.randn(latent_dim, factor_order) * init_scale
            )
            self.B = nn.Parameter(torch.randn(latent_dim) * init_scale)

        if self.factor_order > 1:
            # C[i, i*p] = 1: factor i is the first state of its block. A fixed
            # selector (a buffer, not trained), so the factors stay independent.
            C = torch.zeros(latent_dim, latent_dim * factor_order)
            C[torch.arange(latent_dim), torch.arange(latent_dim) * factor_order] = 1.0
            self.register_buffer("C", C)
        else:
            self.C = None  # the AR(1) paths never read C

        self.f0 = nn.Parameter(torch.zeros(latent_dim))

        # Impulse responses by length, detached; see get_impulse_response.
        self._impulse_response_cache: dict[int, torch.Tensor] = {}

        self.to(self.device)

    def clear_cache(self) -> None:
        """Clear the impulse response cache (needed after the parameters change)."""
        self._impulse_response_cache.clear()

    def train(self, mode: bool = True):
        """Set the training mode; entering it clears the impulse response cache."""
        result = super().train(mode)
        if mode:
            self.clear_cache()
        return result

    def norm(self, x: torch.Tensor, ord: int = None) -> torch.Tensor:
        """Divide each row of ``x`` (last dimension) by its ``ord``-norm.

        ``ord`` defaults to ``norm_order``; ``ord == 0`` returns ``x``. Nothing
        is divided when the mean row norm is at most 1e-4, and rows with norm at
        most ``_ZERO_ROW_NORM`` are left as they are (dividing them gives NaN).
        """
        if ord is None:
            ord = self.norm_order

        if ord == 0:
            return x

        x_norm = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        if torch.abs(x_norm).mean().item() > 1e-4:
            zero_rows = x_norm <= _ZERO_ROW_NORM
            if bool(zero_rows.any()):
                x_norm = torch.where(zero_rows, torch.ones_like(x_norm), x_norm)
            x = x / x_norm
        return x

    def _normed(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, ord=self.norm_order) if self.norm_order > 0 else x

    def _initial_state(self, f0: torch.Tensor) -> torch.Tensor:
        """The companion state (batch, r*p) from ``f0``: a full state as is, factors (batch, r) with zero lags."""
        r, p = self.latent_dim, self.factor_order
        if f0.shape[-1] == r * p:
            return f0
        s0 = torch.zeros(f0.shape[0], r * p, device=f0.device, dtype=f0.dtype)
        s0[:, ::p] = f0
        return s0

    def _batched_f0(self, f0: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """``f0`` as (batch, r); None means the learned ``self.f0``."""
        if f0 is None:
            return self.f0.unsqueeze(0).expand(batch_size, -1)
        if f0.dim() == 1:
            return f0.unsqueeze(0).expand(batch_size, -1)
        return f0

    @staticmethod
    def _diagonal_scan(A: torch.Tensor, B: torch.Tensor, eta_sequence: torch.Tensor,
                       f0: torch.Tensor) -> torch.Tensor:
        """``f_t = A * f_{t-1} + B * eta_t`` from ``f_{-1} = f0``; (batch, T, r)."""
        # An O(T) scan avoids the a^{-t} blowup of a closed form; it is
        # vectorized across batch and factors.
        batch_size, T, r = eta_sequence.shape
        factors = torch.empty(batch_size, T, r, device=eta_sequence.device, dtype=eta_sequence.dtype)
        f_t = f0
        for t in range(T):
            f_t = A * f_t + B * eta_sequence[:, t, :]
            factors[:, t, :] = f_t
        return factors

    def get_impulse_response(self, length: int, use_cache: bool = True) -> torch.Tensor:
        """Impulse response ``H_k = C A^k B`` for ``k < length``, (latent_dim, length).

        In eval mode with ``use_cache`` a detached copy is cached by ``length``
        and returned on later calls, until ``clear_cache`` or ``train()``. In
        training mode nothing is cached: the response is recomputed from the
        current parameters so that gradients reach ``B`` and ``ar_coeffs``.
        """
        use_cache = use_cache and not self.training
        if use_cache and length in self._impulse_response_cache:
            return self._impulse_response_cache[length]

        if self.factor_order == 1:
            # Diagonal A and B: H[i, k] = B[i] * A[i]^k, with A in (0, 1).
            A_diag = torch.sigmoid(self.A)
            B_diag = self._normed(self.B)
            k = torch.arange(length, device=A_diag.device, dtype=A_diag.dtype)
            A_powers = A_diag.unsqueeze(-1) ** k
            H = B_diag.unsqueeze(-1) * A_powers
        else:
            H = self.get_impulse_response_with_coeffs(length, self.ar_coeffs, self.B)

        if use_cache:
            self._impulse_response_cache[length] = H.detach().clone()

        return H

    def get_impulse_response_with_coeffs(
        self,
        length: int,
        ar_coeffs: torch.Tensor,
        B_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Impulse response (r, length) of the given ``ar_coeffs`` and ``B_diag``.

        ``ar_coeffs`` is (r, p) and ``B_diag`` (r,). For ``factor_order == 1``
        ``ar_coeffs[:, 0]`` plays the role of ``A`` (before the sigmoid).
        Nothing is cached.
        """
        r, p = ar_coeffs.shape
        assert r == self.latent_dim and p == self.factor_order
        if self.factor_order == 1:
            A_diag = torch.sigmoid(ar_coeffs[:, 0])
            B_norm = self._normed(B_diag)
            k = torch.arange(length, device=ar_coeffs.device, dtype=ar_coeffs.dtype)
            A_powers = A_diag.unsqueeze(-1) ** k
            H = B_norm.unsqueeze(-1) * A_powers
            return H
        ar_coeffs_norm = self._normed(ar_coeffs)
        A_block = build_ivdfm_diagonal_companion(ar_coeffs_norm)
        # B maps innovation i to the first state of block i.
        B_block = torch.zeros(r * p, r, device=B_diag.device, dtype=B_diag.dtype)
        B_norm = self._normed(B_diag)
        for i in range(r):
            B_block[i * p, i] = B_norm[i]
        # Factor i only sees innovation i: H[i] = C[i] (B_i, A B_i, A^2 B_i, ...).
        H_list = []
        for i in range(r):
            b_i = B_block[:, i]
            krylov_seq = krylov(length, A_block, b_i, c=None)
            H_i = torch.einsum('j, jl -> l', self.C[i, :], krylov_seq)
            H_list.append(H_i)
        return torch.stack(H_list, dim=0)

    def forward_with_coeffs(
        self,
        eta_sequence: torch.Tensor,
        f0: Optional[torch.Tensor],
        ar_coeffs: torch.Tensor,
        B_diag: torch.Tensor,
        group_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``forward`` with per-window AR coefficients (regime AR(p), frozen per window).

        Parameters
        ----------
        eta_sequence : torch.Tensor
            Innovations, (batch, T, r).
        f0 : torch.Tensor or None
            Initial state, (batch, r) or (r,); None uses ``self.f0``.
        ar_coeffs : torch.Tensor
            (batch, r, p).
        B_diag : torch.Tensor
            (batch, r).
        group_id : torch.Tensor, optional
            (batch,) integer group (typically the dominant regime,
            ``pi_w.argmax(dim=-1)``). For ``p > 1`` the windows of a group share
            the coefficients of the group's first window and are convolved in
            one batch; without it every window is computed on its own.

        Returns
        -------
        torch.Tensor
            Factors, (batch, T, r).
        """
        batch_size = eta_sequence.shape[0]
        f0 = self._batched_f0(f0, batch_size)
        if self.factor_order == 1:
            A = torch.sigmoid(ar_coeffs[:, :, 0])
            B = self._normed(B_diag)
            return self._diagonal_scan(A, B, eta_sequence, f0)
        if group_id is not None:
            return self._forward_with_coeffs_grouped(eta_sequence, f0, ar_coeffs, B_diag, group_id)
        return self._forward_with_coeffs_loop(eta_sequence, f0, ar_coeffs, B_diag)

    def _forward_with_coeffs_loop(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor,
        ar_coeffs: torch.Tensor,
        B_diag: torch.Tensor,
    ) -> torch.Tensor:
        """One impulse response and Krylov initial-state term per window."""
        batch_size, T, r = eta_sequence.shape
        factors_list = []
        for b in range(batch_size):
            H = self.get_impulse_response_with_coeffs(T, ar_coeffs[b], B_diag[b])
            eta_b = eta_sequence[b : b + 1]
            eta_T = rearrange(eta_b, 'b t r -> b r t')
            factors_b = self.fft_conv(eta_T, H)
            factors_b = rearrange(factors_b, 'b r t -> b t r')
            s0 = self._initial_state(f0[b : b + 1])
            ar_norm = self._normed(ar_coeffs[b])
            A_block = build_ivdfm_diagonal_companion(ar_norm)
            A_block_b = A_block.unsqueeze(0)
            s_seq = krylov(T, A_block_b, s0, c=None)
            f_init = torch.einsum('ij, bjt -> bit', self.C, s_seq)
            f_init = rearrange(f_init, 'b r t -> b t r')
            factors_b = factors_b + f_init
            factors_list.append(factors_b)
        return torch.cat(factors_list, dim=0)

    def _forward_with_coeffs_grouped(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor,
        ar_coeffs: torch.Tensor,
        B_diag: torch.Tensor,
        group_id: torch.Tensor,
    ) -> torch.Tensor:
        """One impulse response per group (from its first window), batched in the group."""
        batch_size, T, r = eta_sequence.shape
        factors = torch.empty(batch_size, T, r, device=eta_sequence.device, dtype=eta_sequence.dtype)
        uniq = group_id.unique()
        for g in uniq:
            idx = (group_id == g).nonzero(as_tuple=True)[0]
            ar_g = ar_coeffs[idx[0]]  # (r, p)
            B_g = B_diag[idx[0]]      # (r,)
            H_g = self.get_impulse_response_with_coeffs(T, ar_g, B_g)  # (r, T)
            eta_g = eta_sequence[idx]  # (|g|, T, r)
            eta_g_t = rearrange(eta_g, 'b t r -> b r t')
            f_conv_g = self.fft_conv(eta_g_t, H_g)  # (|g|, r, T)
            f_conv_g = rearrange(f_conv_g, 'b r t -> b t r')
            s0_g = self._initial_state(f0[idx])
            ar_norm_g = self._normed(ar_g)
            A_block_g = build_ivdfm_diagonal_companion(ar_norm_g)
            A_block_batched = A_block_g.unsqueeze(0).expand(idx.shape[0], -1, -1)
            s_seq_g = krylov(T, A_block_batched, s0_g, c=None)  # (|g|, r*p, T)
            f_init_g = torch.einsum('ij, bjt -> bit', self.C, s_seq_g)  # (|g|, r, T)
            f_init_g = rearrange(f_init_g, 'b r t -> b t r')
            factors[idx] = f_conv_g + f_init_g
        return factors

    def forward_closed_loop(self, f_current: torch.Tensor, eta_future: torch.Tensor, horizon: int) -> torch.Tensor:
        """Forecast rollout ``f_{t+h} = C A^h s_t + sum_{k=0}^{h-1} H_k eta_{t+h-k}`` (see ``BaseSSM``).

        ``f_current`` is the factors (batch, r) or, for ``factor_order > 1``,
        the companion state (batch, r*p) ``[f_t, f_{t-1}, ..., f_{t-p+1}]`` per
        factor (with only the factors the lags are taken as zero). The AR(1)
        scan treats its ``f0`` as the state before the first row, so it runs
        on ``eta_future`` from ``f_current`` directly; the AR(p) companion path
        treats ``f0`` as the first row's state and takes the padded rollout of
        ``BaseSSM``.
        """
        if self.factor_order == 1:
            return self.forward(eta_future, f0=f_current)
        return super().forward_closed_loop(f_current, eta_future, horizon)

    def forward_closed_loop_with_coeffs(
        self,
        f_current: torch.Tensor,
        eta_future: torch.Tensor,
        ar_coeffs: torch.Tensor,
        B_diag: torch.Tensor,
        group_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``forward_closed_loop`` with per-window coefficients (``forward_with_coeffs``)."""
        if self.factor_order == 1:  # the AR(1) scan starts from the state before the first row
            return self.forward_with_coeffs(eta_future, f_current, ar_coeffs, B_diag, group_id=group_id)
        batch_size = f_current.shape[0]
        eta_padded = torch.cat([
            torch.zeros(batch_size, 1, self.latent_dim, device=eta_future.device, dtype=eta_future.dtype),
            eta_future,
        ], dim=1)
        factors = self.forward_with_coeffs(eta_padded, f_current, ar_coeffs, B_diag, group_id=group_id)
        return factors[:, 1:, :]

    def forward(
        self,
        eta_sequence: torch.Tensor,
        f0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Factors ``f_t = A^t f_0 + sum_{k<t} H_k eta_{t-1-k}`` of the innovations.

        ``eta_sequence`` is (batch, T, latent_dim); ``f0`` is (batch, latent_dim)
        or (latent_dim,), None for the learned ``self.f0``. Returns
        (batch, T, latent_dim).
        """
        batch_size, _, r = eta_sequence.shape
        assert r == self.latent_dim, f"Expected latent_dim={self.latent_dim}, got {r}"
        f0 = self._batched_f0(f0, batch_size)

        if self.factor_order == 1:
            return self._forward_first_order(eta_sequence, f0)
        else:
            return self._forward_companion(eta_sequence, f0)

    def _forward_first_order(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """AR(1) path: ``f_t = A * f_{t-1} + B * eta_t`` with diagonal A and B."""
        A_diag = torch.sigmoid(self.A)
        B_diag = self._normed(self.B)
        return self._diagonal_scan(A_diag, B_diag, eta_sequence, f0)

    def _forward_companion(
        self,
        eta_sequence: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """AR(p) path: FFT convolution with the impulse response, plus ``C A^t s_0``."""
        batch_size, T, r = eta_sequence.shape

        H = self.get_impulse_response(T)  # (r, T)
        eta_T = rearrange(eta_sequence, 'b t r -> b r t')
        factors = self.fft_conv(eta_T, H)  # (batch, r, T)
        factors = rearrange(factors, 'b r t -> b t r')

        # Initial-state term: the companion state s_0 (from f0, see _initial_state), propagated by A^t.
        ar_coeffs_norm = self._normed(self.ar_coeffs)
        A_block = build_ivdfm_diagonal_companion(ar_coeffs_norm)
        s0 = self._initial_state(f0)
        # A is broadcast over the batch so that each s0[b] gets its own sequence.
        A_block_b = A_block.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, r*p, r*p)
        s_seq = krylov(T, A_block_b, s0, c=None)  # (batch, r*p, T)
        f_init = torch.einsum('ij, bjt -> bit', self.C, s_seq)  # (batch, r, T)
        f_init = rearrange(f_init, 'b r t -> b t r')

        factors = factors + f_init

        return factors

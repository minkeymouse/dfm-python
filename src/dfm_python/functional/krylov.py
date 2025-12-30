"""Krylov computation for efficient matrix power operations.

Computes [b, Ab, A²b, ..., A^(L-1)b] efficiently using squaring trick.
"""

from typing import Optional, Tuple, Union
import torch


def krylov_sequential(
    L: int, 
    A: torch.Tensor, 
    b: torch.Tensor, 
    c: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sequential Krylov computation (for small L or debugging).
    
    Parameters
    ----------
    L : int
        Number of terms to compute
    A : torch.Tensor
        Matrix of shape (..., N, N)
    b : torch.Tensor
        Vector of shape (..., N)
    c : torch.Tensor, optional
        Vector of shape (..., N). If provided, computes c^T A^l b.
        
    Returns
    -------
    x : torch.Tensor
        If c is None: shape (..., N, L) where x[..., l] = A^l @ b
        If c is provided: shape (..., L) where x[..., l] = c^T @ A^l @ b
    """
    # Check which of dim b and c is smaller to save memory
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c * b_, dim=-1)  # (...)
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x


def krylov(
    L: int, 
    A: torch.Tensor, 
    b: torch.Tensor, 
    c: Optional[torch.Tensor] = None, 
    return_power: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Krylov matrix efficiently using squaring trick.
    
    Computes [b, Ab, A²b, ..., A^(L-1)b] or [c^T b, c^T Ab, ...] if c provided.
    
    Parameters
    ----------
    L : int
        Number of terms to compute
    A : torch.Tensor
        Matrix of shape (..., N, N)
    b : torch.Tensor
        Vector of shape (..., N)
    c : torch.Tensor, optional
        Vector of shape (..., N). If provided, computes c^T A^l b.
    return_power : bool, default=False
        If True, also return A^(L-1)
        
    Returns
    -------
    x : torch.Tensor
        If c is None: shape (..., N, L) where x[..., l] = A^l @ b
        If c is provided: shape (..., L) where x[..., l] = c^T @ A^l @ b
    AL : torch.Tensor, optional
        A^(L-1) if return_power=True
    """
    x = b.unsqueeze(-1)  # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L - 1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1:
                AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else:
            _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1)
        if not done:
            A_ = A_ @ A_

    assert x.shape[-1] == L, f"Expected length {L}, got {x.shape[-1]}"

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous()
    
    if return_power:
        return x, AL
    else:
        return x


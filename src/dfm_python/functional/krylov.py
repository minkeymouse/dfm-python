""" Compute a Krylov function efficiently. (S3 renames the Krylov function to a "state space kernel")

original code from Spacetime paper

A : (N, N)
b : (N,)
c : (N,)
Return: [c^T A^i b for i in [L]]
"""


import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def krylov_sequential(L, A, b, c=None):
    """ Constant matrix A

    A : (..., N, N)
    b : (..., N)
    c : (..., N)

    Returns
    if c:
    x : (..., L)
    x[i, l] = c[i] @ A^l @ b[i]

    else:
    x : (..., N, L)
    x[i, l] = A^l @ b[i]
    """

    # Check which of dim b and c is smaller to save memory
    if c is not None and c.numel() < b.numel():
        return krylov_sequential(L, A.transpose(-1, -2), c, b)

    b_ = b
    x = []
    for _ in range(L):
        if c is not None:
            x_ = torch.sum(c*b_, dim=-1) # (...) # could be faster with matmul or einsum?
        else:
            x_ = b_
        x.append(x_)
        b_ = (A @ b_.unsqueeze(-1)).squeeze(-1)

    x = torch.stack(x, dim=-1)
    return x


def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        try:
            _x = A_ @ _x
        except Exception as e:
            print(e)
            breakpoint()
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    try:
        assert x.shape[-1] == L
    except:
        print('x.shape', x.shape)
        print('L', L)
        breakpoint()

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


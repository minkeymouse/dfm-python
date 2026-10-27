"""Factor identification for reporting (spec section 4.10).

After ``fit`` each model fixes a reporting transform ``D`` (r x r) from its
estimated parameters, and every factor-indexed output reports the factors
``f~ = D' f``. ``D`` is orthogonal (a rotation, a permutation and sign flips),
so observation-space outputs are unchanged: ``L f = (L D)(D' f)``.

- :func:`rotation`: the eigenvectors of ``L'L`` by decreasing eigenvalue, so
  the reported loading columns are orthogonal with squared norms equal to the
  eigenvalues; :func:`block_rotation` applies it per block (DFM).
- :func:`sign_flips`: the sign rule. Factor ``j`` is flipped when the sum of
  its standardized loadings is negative, or, with an anchor
  ``{j: i}``, when its loading on series ``i`` is negative (an anchor
  loading below ``1e-8`` in absolute value falls back to the default).
- :func:`order_by` and :func:`permutation`: the iVDFM factor order.
- :func:`identify`: rotation (optional) then signs, as one ``D``.
- :func:`resolve_anchors`: ``sign_anchors={factor_name: series_name}`` as
  ``{factor index: series index}``.
- ``apply_*`` and :func:`contributions`: ``D`` applied to factors,
  standard deviations, loadings, transition matrices, covariances and
  per-factor contributions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from .errors import ConfigurationError

__all__ = [
    "ANCHOR_TOL",
    "apply_cov",
    "apply_factors",
    "apply_loadings",
    "apply_std",
    "apply_transition",
    "block_rotation",
    "contributions",
    "identify",
    "order_by",
    "permutation",
    "resolve_anchors",
    "rotation",
    "sign_flips",
]

#: An anchor loading smaller than this in absolute value cannot fix a sign; the default rule applies.
ANCHOR_TOL = 1e-8


def rotation(loadings: np.ndarray) -> np.ndarray:
    """The orthogonal ``D`` whose columns are the eigenvectors of ``L'L``, by decreasing eigenvalue.

    Parameters
    ----------
    loadings : ndarray, shape (N, r)
        The standardized loadings ``L``.

    Returns
    -------
    ndarray, shape (r, r)
        ``L D`` has orthogonal columns with squared norms equal to the
        eigenvalues of ``L'L``, in decreasing order. Signs are arbitrary until
        :func:`sign_flips`.
    """
    L = np.asarray(loadings, dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh(L.T @ L)
    return vectors[:, np.argsort(-eigenvalues, kind="stable")]


def block_rotation(loadings: np.ndarray, blocks: Sequence[Sequence[int]]) -> np.ndarray:
    """Block-diagonal ``D``: :func:`rotation` of each block's loading columns.

    Parameters
    ----------
    loadings : ndarray, shape (N, r)
        The standardized loadings.
    blocks : sequence of sequences of int
        The factor (column) indices of each block; together they cover
        ``0..r-1`` once. A one-factor block keeps its factor (``D_b = 1``).
    """
    L = np.asarray(loadings, dtype=np.float64)
    r = L.shape[1]
    covered = sorted(j for block in blocks for j in block)
    if covered != list(range(r)):
        raise ValueError(f"blocks must cover the factor columns 0..{r - 1} once, got {[list(b) for b in blocks]}.")
    D = np.zeros((r, r))
    for block in blocks:
        cols = list(block)
        D[np.ix_(cols, cols)] = rotation(L[:, cols]) if len(cols) > 1 else 1.0
    return D


def sign_flips(loadings: np.ndarray, anchors: Optional[Mapping[int, int]] = None) -> np.ndarray:
    """``+1`` or ``-1`` per factor: the sign rule applied to (already rotated) loadings.

    Parameters
    ----------
    loadings : ndarray, shape (N, r)
        Standardized loadings in the reported coordinates (before the flips).
    anchors : mapping, optional
        ``{factor index: series index}``: that factor's loading on that series
        is made non-negative, unless it is below :data:`ANCHOR_TOL` in absolute
        value, in which case the default rule applies.

    Returns
    -------
    ndarray, shape (r,)
        By default a factor is flipped when the sum of its loadings is negative.
    """
    L = np.asarray(loadings, dtype=np.float64)
    signs = np.where(L.sum(axis=0) < 0, -1.0, 1.0)
    for j, i in (anchors or {}).items():
        value = L[i, j]
        if abs(value) >= ANCHOR_TOL:
            signs[j] = -1.0 if value < 0 else 1.0
    return signs


def identify(
    loadings: np.ndarray,
    *,
    blocks: Optional[Sequence[Sequence[int]]] = None,
    anchors: Optional[Mapping[int, int]] = None,
    rotate: bool = True,
) -> np.ndarray:
    """The reporting transform ``D``: the rotation (per block when ``blocks`` is given) and then the signs.

    With ``rotate=False`` only the signs are fixed (``D`` is diagonal).
    """
    L = np.asarray(loadings, dtype=np.float64)
    r = L.shape[1]
    if not rotate:
        D = np.eye(r)
    elif blocks is None:
        D = rotation(L)
    else:
        D = block_rotation(L, blocks)
    return D * sign_flips(L @ D, anchors)[None, :]


def order_by(scores: Sequence[float]) -> np.ndarray:
    """Factor indices by decreasing ``scores`` (ties keep their order)."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def permutation(order: Sequence[int], signs: Optional[Sequence[float]] = None) -> np.ndarray:
    """The ``D`` that reorders factors (reported factor ``j`` is factor ``order[j]``), with optional signs.

    ``(F @ D)[:, j] == signs[j] * F[:, order[j]]``.
    """
    order = np.asarray(order, dtype=np.int64)
    r = len(order)
    if sorted(order.tolist()) != list(range(r)):
        raise ValueError(f"order must be a permutation of 0..{r - 1}, got {order.tolist()}.")
    D = np.zeros((r, r))
    D[order, np.arange(r)] = 1.0 if signs is None else np.asarray(signs, dtype=np.float64)
    return D


def resolve_anchors(
    sign_anchors: Optional[Mapping[Any, Any]], factor_names: Sequence[Any], series_names: Sequence[Any]
) -> dict:
    """``sign_anchors={factor_name: series_name}`` as ``{factor index: series index}``.

    Raises
    ------
    ConfigurationError
        For a mapping of another type, or an unknown factor or series name.
    """
    if sign_anchors is None:
        return {}
    if not isinstance(sign_anchors, Mapping):
        raise ConfigurationError(
            f"sign_anchors must be None or a dict {{factor_name: series_name}}, got {type(sign_anchors).__name__}."
        )
    factors, series = list(factor_names), list(series_names)
    unknown_f = [f for f in sign_anchors if f not in factors]
    unknown_s = [s for s in sign_anchors.values() if s not in series]
    if unknown_f or unknown_s:
        parts = []
        if unknown_f:
            parts.append(f"unknown factors {unknown_f} (the factors are {factors})")
        if unknown_s:
            parts.append(f"unknown series {unknown_s} (the series are {series})")
        raise ConfigurationError("sign_anchors names " + " and ".join(parts) + ".")
    return {factors.index(f): series.index(s) for f, s in sign_anchors.items()}


# ---------------------------------------------------------------------------
# Applying D
# ---------------------------------------------------------------------------


def apply_factors(factors: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Reported factors ``f~_t = D' f_t``, row-wise: ``F D`` for ``F`` of shape (T, r)."""
    return np.asarray(factors, dtype=np.float64) @ D


def apply_cov(cov: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``D' P D`` for one covariance (r, r) or a stack (T, r, r)."""
    P = np.asarray(cov, dtype=np.float64)
    return D.T @ P @ D if P.ndim == 2 else np.einsum("ji,tjk,kl->til", D, P, D)


def apply_std(cov: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Standard deviations of the reported factors: ``sqrt(diag(D' P D))`` for (r, r) or (T, r, r)."""
    rotated = apply_cov(cov, D)
    diag = np.diagonal(rotated, axis1=-2, axis2=-1)
    return np.sqrt(np.clip(diag, 0.0, None))


def apply_loadings(loadings: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Reported loadings ``L D`` (N, r)."""
    return np.asarray(loadings, dtype=np.float64) @ D


def apply_transition(transition: np.ndarray, D: np.ndarray) -> np.ndarray:
    """``D' A_l D`` for one transition matrix (r, r) or one per lag (p, r, r)."""
    return apply_cov(transition, D)


def contributions(loadings: np.ndarray, factors: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Per-factor contributions in reported coordinates, shape (T, r, N).

    ``out[t, j, i] = sum_l (L_l D)[i, j] * (F_l D)[t, j]``: the part of series
    ``i`` at time ``t`` due to reported factor ``j``. Summed over ``j`` it is
    ``sum_l L_l f_{t,l}``, the common component, whatever ``D``.

    Parameters
    ----------
    loadings : ndarray, shape (N, r) or (p, N, r)
        The loadings, or one matrix per lag ``l``.
    factors : ndarray, shape (T, r) or (p, T, r)
        The factors, or the factor states at each lag (``f_{t-l}`` for lag ``l``).
    D : ndarray, shape (r, r)
    """
    L = np.asarray(loadings, dtype=np.float64)
    F = np.asarray(factors, dtype=np.float64)
    if L.ndim == 2:
        L, F = L[None], F[None]
    if L.shape[0] != F.shape[0]:
        raise ValueError(f"loadings have {L.shape[0]} lags but factors have {F.shape[0]}.")
    return np.einsum("ltj,lij->tji", F @ D, L @ D)

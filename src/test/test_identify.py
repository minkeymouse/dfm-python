"""Factor identification for reporting, ``dfm_python._identify`` (spec section 4.10)."""

import numpy as np
import pytest

from dfm_python import _identify as ident
from dfm_python.errors import ConfigurationError


def _model(seed=0, T=50, N=6, r=3):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((N, r))
    F = rng.standard_normal((T, r))
    A = 0.3 * rng.standard_normal((r, r))
    Q = np.cov(rng.standard_normal((100, r)), rowvar=False)
    return L, F, A, Q


def test_rotation_is_orthogonal_and_sorts_the_eigenvalues():
    L, _, _, _ = _model()
    D = ident.rotation(L)
    np.testing.assert_allclose(D.T @ D, np.eye(3), atol=1e-12)
    rotated = L @ D
    gram = rotated.T @ rotated
    np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-10)  # orthogonal columns
    eigenvalues = np.diag(gram)
    assert np.all(np.diff(eigenvalues) <= 0)
    np.testing.assert_allclose(eigenvalues, np.sort(np.linalg.eigvalsh(L.T @ L))[::-1], rtol=1e-12)


def test_rotation_leaves_the_common_component_invariant():
    L, F, A, Q = _model()
    D = ident.identify(L)
    common = F @ L.T
    np.testing.assert_allclose(ident.apply_factors(F, D) @ ident.apply_loadings(L, D).T, common, atol=1e-12)
    # C x for a state space with lags: the per-lag images and the contributions add up the same way
    L_lags = np.stack([L, 0.5 * L])
    F_lags = np.stack([F, np.roll(F, 1, axis=0)])
    parts = ident.contributions(L_lags, F_lags, D)
    assert parts.shape == (50, 3, 6)
    np.testing.assert_allclose(parts.sum(axis=1), F @ L.T + 0.5 * np.roll(F, 1, axis=0) @ L.T, atol=1e-12)
    # the transition and covariances transform consistently: Var(D'f) = D' Q D, D'f_t = (D' A D) D'f_{t-1}
    At = ident.apply_transition(A, D)
    np.testing.assert_allclose(ident.apply_factors(F, D)[:-1] @ At.T, (F[:-1] @ A.T) @ D, atol=1e-12)
    np.testing.assert_allclose(ident.apply_cov(Q, D), D.T @ Q @ D, atol=1e-14)
    stack = np.stack([Q, 2 * Q])
    np.testing.assert_allclose(ident.apply_cov(stack, D)[1], 2 * D.T @ Q @ D, atol=1e-12)
    np.testing.assert_allclose(ident.apply_std(stack, D)[0], np.sqrt(np.diag(D.T @ Q @ D)), atol=1e-14)
    per_lag = np.stack([A, A @ A])
    np.testing.assert_allclose(ident.apply_transition(per_lag, D)[1], D.T @ A @ A @ D, atol=1e-12)


def test_default_sign_rule_makes_column_sums_non_negative():
    L, _, _, _ = _model(seed=3)
    for flip in ([1, -1, 1], [-1, -1, -1]):
        D = ident.identify(L * np.array(flip), rotate=False)
        reported = L * np.array(flip) @ D
        assert np.all(reported.sum(axis=0) >= 0)
    # the result does not depend on the input signs
    a = ident.identify(L)
    b = ident.identify(L * np.array([-1, 1, -1]))
    np.testing.assert_allclose(L @ a, (L * np.array([-1, 1, -1])) @ b, atol=1e-12)


def test_anchors_fix_the_sign_of_one_loading():
    L, _, _, _ = _model(seed=4)
    reported = L @ ident.identify(L, anchors={0: 2, 2: 5})
    assert reported[2, 0] >= 0 and reported[5, 2] >= 0
    # the reported loadings do not depend on the estimated signs
    np.testing.assert_allclose(-L @ ident.identify(-L, anchors={0: 2, 2: 5}), reported, atol=1e-12)


def test_a_near_zero_anchor_falls_back_to_the_default_rule():
    L = np.array([[1e-10, 1.0], [-2.0, 1.0], [-1.0, -0.5]])
    signs = ident.sign_flips(L, anchors={0: 0})
    assert signs[0] == -1.0  # column sum -3 < 0: flipped by the default rule, the anchor is ignored
    assert ident.sign_flips(np.array([[2e-8], [-1.0]]), anchors={0: 0})[0] == 1.0  # |2e-8| >= 1e-8: the anchor wins


def test_block_rotation_is_block_diagonal():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((8, 4))
    D = ident.block_rotation(L, [[0, 1, 2], [3]])
    assert np.all(D[:3, 3] == 0) and np.all(D[3, :3] == 0) and D[3, 3] == 1.0
    np.testing.assert_allclose(D[:3, :3], ident.rotation(L[:, :3]), atol=1e-14)
    signed = ident.identify(L, blocks=[[0, 1, 2], [3]])
    assert np.all(signed[:3, 3] == 0)
    assert np.all((L @ signed).sum(axis=0) >= 0)
    with pytest.raises(ValueError, match="cover"):
        ident.block_rotation(L, [[0, 1], [3]])


def test_permutation_reorders_factors_with_signs():
    F = np.arange(12.0).reshape(4, 3)
    order = ident.order_by([0.5, 2.0, 1.0])
    assert list(order) == [1, 2, 0]
    D = ident.permutation(order, signs=[1.0, -1.0, 1.0])
    reported = ident.apply_factors(F, D)
    np.testing.assert_array_equal(reported, np.column_stack([F[:, 1], -F[:, 2], F[:, 0]]))
    np.testing.assert_allclose(D.T @ D, np.eye(3))
    assert list(ident.order_by([1.0, 1.0, 2.0])) == [2, 0, 1]  # ties keep their order
    with pytest.raises(ValueError, match="permutation"):
        ident.permutation([0, 0, 1])


def test_resolve_anchors():
    factors, series = ["factor_0", "factor_1"], ["gdp", "ip", "cpi"]
    assert ident.resolve_anchors(None, factors, series) == {}
    assert ident.resolve_anchors({"factor_1": "cpi"}, factors, series) == {1: 2}
    with pytest.raises(ConfigurationError, match=r"unknown factors \['factor_9'\]"):
        ident.resolve_anchors({"factor_9": "gdp"}, factors, series)
    with pytest.raises(ConfigurationError, match=r"unknown series \['oil'\]"):
        ident.resolve_anchors({"factor_0": "oil"}, factors, series)
    with pytest.raises(ConfigurationError, match="dict"):
        ident.resolve_anchors(["factor_0"], factors, series)

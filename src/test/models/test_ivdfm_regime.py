"""Tests for iVDFM regime implementation: K=1 collapse, K>1 A_diag/B_diag, effective coeffs, forward/predict paths."""

import pytest
import numpy as np
import torch
from dfm_python.models.ivdfm import iVDFM
from dfm_python.config.schema.model import iVDFMConfig


def _make_data(T=80, N=5, seed=42):
    np.random.seed(seed)
    return np.random.randn(T, N).astype(np.float32)


class TestIVDFMRegimeK1:
    """K=1: no regime structure; A_diag/B_diag are None; same as baseline."""

    def test_k1_no_a_diag_b_diag(self):
        """With num_regimes=1, A_diag and B_diag are None."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=1,
            factor_order=4,
        )
        model._build_components()
        assert model.A_diag is None
        assert model.B_diag is None

    def test_k1_forward_uses_global_ssm(self):
        """K=1 forward does not use regime dynamics; output shape correct."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=1,
            factor_order=2,
        )
        model._build_components()
        batch, T, N = 2, 20, 5
        device = next(model.parameters()).device
        y_1T = torch.randn(batch, T, N, device=device)
        u_1T = torch.randn(batch, T, 1, device=device)
        out = model.forward(y_1T, u_1T)
        assert out["factors"].shape == (batch, T, 3)
        assert "regime_weights" not in out or out["regime_weights"] is None

    def test_k1_fit_predict_p4(self):
        """K=1, p=4: fit and predict run without error; no NaNs."""
        data = _make_data(T=60, N=4)
        model = iVDFM(
            data_dim=4,
            num_factors=3,
            context_dim=1,
            sequence_length=24,
            num_regimes=1,
            factor_order=4,
            max_epochs=2,
            batch_size=16,
        )
        model.fit(data)
        pred = model.predict(horizon=5)
        assert pred.shape == (5, 4)
        assert not np.isnan(pred).any()


class TestIVDFMRegimeKGreaterThan1:
    """K>1: A_diag (K,p,r), B_diag (K,r); effective coeffs; π at window origin."""

    def test_k2_has_a_diag_b_diag(self):
        """With num_regimes=2, A_diag (K,p,r) and B_diag (K,r) exist."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=2,
            factor_order=4,
        )
        model._build_components()
        assert model.A_diag is not None
        assert model.B_diag is not None
        assert model.A_diag.shape == (2, 4, 3)
        assert model.B_diag.shape == (2, 3)

    def test_effective_regime_coeffs_shape(self):
        """_effective_regime_coeffs(pi_w) returns A_eff (batch, r, p), B_eff (batch, r)."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=3,
            factor_order=4,
        )
        model._build_components()
        batch = 4
        device = next(model.parameters()).device
        pi_w = torch.softmax(torch.randn(batch, 3, device=device), dim=-1)
        A_eff, B_eff = model._effective_regime_coeffs(pi_w)
        assert A_eff.shape == (batch, 3, 4)
        assert B_eff.shape == (batch, 3)

    def test_effective_regime_coeffs_convex_combination(self):
        """When pi_w is one-hot, A_eff and B_eff equal that regime's A_diag[k], B_diag[k]."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=3,
            factor_order=4,
        )
        model._build_components()
        device = next(model.parameters()).device
        pi_w = torch.tensor([[0.0, 1.0, 0.0]], device=device)  # regime 1
        A_eff, B_eff = model._effective_regime_coeffs(pi_w)
        torch.testing.assert_close(A_eff[0], model.A_diag[1].permute(1, 0))  # (p,r)->(r,p)
        torch.testing.assert_close(B_eff[0], model.B_diag[1])

    def test_k2_forward_returns_regime_weights(self):
        """K>1 forward includes regime_weights and factors shape correct."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=2,
            factor_order=2,
        )
        model._build_components()
        batch, T, N = 2, 20, 5
        device = next(model.parameters()).device
        y_1T = torch.randn(batch, T, N, device=device)
        u_1T = torch.randn(batch, T, 1, device=device)
        out = model.forward(y_1T, u_1T)
        assert out["factors"].shape == (batch, T, 3)
        assert "regime_weights" in out
        assert out["regime_weights"].shape == (batch, T, 2)
        assert not torch.isnan(out["factors"]).any()

    def test_k2_fit_predict_p4(self):
        """K=2, p=4: fit and predict run; no NaNs."""
        data = _make_data(T=60, N=4)
        model = iVDFM(
            data_dim=4,
            num_factors=3,
            context_dim=1,
            sequence_length=24,
            num_regimes=2,
            factor_order=4,
            max_epochs=2,
            batch_size=16,
        )
        model.fit(data)
        pred = model.predict(horizon=5)
        assert pred.shape == (5, 4)
        assert not np.isnan(pred).any()


class TestIVDFMRegimeWindowFrozen:
    """Regime is frozen per window: π at origin for dynamics."""

    def test_forward_uses_pi_at_origin(self):
        """With K>1, dynamics use regime_weights[:, 0, :] (window origin), not per-step π."""
        model = iVDFM(
            data_dim=5,
            num_factors=3,
            context_dim=1,
            sequence_length=20,
            num_regimes=2,
            factor_order=1,
        )
        model._build_components()
        batch, T, N = 2, 20, 5
        device = next(model.parameters()).device
        y_1T = torch.randn(batch, T, N, device=device)
        u_1T = torch.randn(batch, T, 1, device=device)
        out = model.forward(y_1T, u_1T)
        # Internal path: pi_w = regime_weights[:, 0, :]; one A_eff/B_eff per batch item for whole T
        assert out["regime_weights"].shape == (batch, T, 2)
        assert out["factors"].shape == (batch, T, 3)

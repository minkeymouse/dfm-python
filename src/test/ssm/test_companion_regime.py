"""Tests for iVDFMCompanionSSM regime support: forward_with_coeffs, forward_closed_loop_with_coeffs, get_impulse_response_with_coeffs."""

import pytest
import torch
from dfm_python.ssm.companion import iVDFMCompanionSSM


class TestCompanionRegime:
    """Test regime AR(p) path: per-batch coefficients, window-frozen dynamics."""

    @pytest.fixture
    def ssm_p1(self, device="cpu"):
        """Companion SSM with p=1."""
        return iVDFMCompanionSSM(
            latent_dim=3,
            factor_order=1,
            device=device,
        )

    @pytest.fixture
    def ssm_p4(self, device="cpu"):
        """Companion SSM with p=4."""
        return iVDFMCompanionSSM(
            latent_dim=3,
            factor_order=4,
            device=device,
        )

    def test_forward_with_coeffs_p1_shape(self, ssm_p1):
        """forward_with_coeffs(eta, f0, ar_coeffs, B_diag) returns (batch, T, r) for p=1."""
        batch, T, r = 4, 20, 3
        eta = torch.randn(batch, T, r)
        ar_coeffs = torch.randn(batch, r, 1)  # (batch, r, p) with p=1
        B_diag = torch.randn(batch, r)
        out = ssm_p1.forward_with_coeffs(eta, None, ar_coeffs, B_diag)
        assert out.shape == (batch, T, r)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_with_coeffs_p1_matches_global_when_uniform(self, ssm_p1):
        """When ar_coeffs and B_diag are broadcast from SSM params, forward_with_coeffs matches forward (p=1)."""
        batch, T, r = 2, 15, 3
        eta = torch.randn(batch, T, r)
        # Use SSM's own params as single set of coeffs, broadcast to batch
        ar_batch = ssm_p1.A.unsqueeze(0).unsqueeze(2).expand(batch, r, 1)  # (batch, r, 1)
        B_batch = ssm_p1.B.unsqueeze(0).expand(batch, -1)
        f0 = ssm_p1.f0.unsqueeze(0).expand(batch, -1)
        out_coeffs = ssm_p1.forward_with_coeffs(eta, f0, ar_batch, B_batch)
        out_global = ssm_p1.forward(eta, f0)
        torch.testing.assert_close(out_coeffs, out_global)

    def test_get_impulse_response_with_coeffs_p1_shape(self, ssm_p1):
        """get_impulse_response_with_coeffs(length, ar_coeffs, B_diag) returns (r, length) for p=1."""
        length = 10
        ar_coeffs = torch.randn(3, 1)  # (r, p)
        B_diag = torch.randn(3)
        H = ssm_p1.get_impulse_response_with_coeffs(length, ar_coeffs, B_diag)
        assert H.shape == (3, length)
        assert not torch.isnan(H).any()

    def test_get_impulse_response_with_coeffs_p4_shape(self, ssm_p4):
        """get_impulse_response_with_coeffs returns (r, length) for p=4."""
        length = 10
        ar_coeffs = torch.randn(3, 4) * 0.1  # (r, p) stable
        B_diag = torch.randn(3)
        H = ssm_p4.get_impulse_response_with_coeffs(length, ar_coeffs, B_diag)
        assert H.shape == (3, length)
        assert not torch.isnan(H).any()

    def test_forward_with_coeffs_p4_shape(self, ssm_p4):
        """forward_with_coeffs returns (batch, T, r) for p=4."""
        batch, T, r = 2, 20, 3
        eta = torch.randn(batch, T, r)
        ar_coeffs = torch.randn(batch, r, 4) * 0.1
        B_diag = torch.randn(batch, r)
        out = ssm_p4.forward_with_coeffs(eta, None, ar_coeffs, B_diag)
        assert out.shape == (batch, T, r)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_closed_loop_with_coeffs_shape(self, ssm_p1):
        """forward_closed_loop_with_coeffs returns (batch, horizon, r)."""
        batch, horizon, r = 2, 8, 3
        f_current = torch.randn(batch, r)
        eta_future = torch.randn(batch, horizon, r)
        ar_coeffs = torch.randn(batch, r, 1)
        B_diag = torch.randn(batch, r)
        out = ssm_p1.forward_closed_loop_with_coeffs(f_current, eta_future, ar_coeffs, B_diag)
        assert out.shape == (batch, horizon, r)
        assert not torch.isnan(out).any()

    def test_forward_closed_loop_with_coeffs_matches_forward_slice(self, ssm_p1):
        """closed_loop_with_coeffs matches forward_with_coeffs on [0, eta_future] then slice [1:]."""
        batch, horizon, r = 2, 6, 3
        f_current = torch.randn(batch, r)
        eta_future = torch.randn(batch, horizon, r)
        ar_coeffs = torch.randn(batch, r, 1)
        B_diag = torch.randn(batch, r)
        eta_padded = torch.cat([torch.zeros(batch, 1, r), eta_future], dim=1)
        full = ssm_p1.forward_with_coeffs(eta_padded, f_current, ar_coeffs, B_diag)
        expected = full[:, 1:, :]
        out = ssm_p1.forward_closed_loop_with_coeffs(f_current, eta_future, ar_coeffs, B_diag)
        torch.testing.assert_close(out, expected)

    def test_forward_with_coeffs_grouped_matches_loop_p4(self, ssm_p4):
        """Grouped path (group_id = arange(batch)) matches loop path (group_id=None) for p=4."""
        torch.manual_seed(42)
        device = next(ssm_p4.parameters()).device
        batch, T, r = 4, 20, 3
        eta = torch.randn(batch, T, r, device=device)
        ar_coeffs = torch.randn(batch, r, 4, device=device) * 0.1
        B_diag = torch.randn(batch, r, device=device)
        f0 = torch.randn(batch, r, device=device)
        out_loop = ssm_p4._forward_with_coeffs_loop(eta, f0, ar_coeffs, B_diag)
        group_id = torch.arange(batch, device=device, dtype=torch.long)
        out_grouped = ssm_p4._forward_with_coeffs_grouped(eta, f0, ar_coeffs, B_diag, group_id)
        torch.testing.assert_close(out_grouped, out_loop, atol=1e-5, rtol=1e-5)

    def test_forward_with_coeffs_same_group_batched_p4(self, ssm_p4):
        """When all windows share same group and same coeffs, grouped path matches loop."""
        torch.manual_seed(43)
        device = next(ssm_p4.parameters()).device
        batch, T, r = 4, 16, 3
        ar_rep = torch.randn(r, 4, device=device) * 0.1
        B_rep = torch.randn(r, device=device)
        ar_coeffs = ar_rep.unsqueeze(0).expand(batch, -1, -1)
        B_diag = B_rep.unsqueeze(0).expand(batch, -1)
        eta = torch.randn(batch, T, r, device=device)
        f0 = torch.randn(batch, r, device=device)
        out_loop = ssm_p4._forward_with_coeffs_loop(eta, f0, ar_coeffs, B_diag)
        group_id = torch.zeros(batch, device=device, dtype=torch.long)
        out_grouped = ssm_p4._forward_with_coeffs_grouped(eta, f0, ar_coeffs, B_diag, group_id)
        torch.testing.assert_close(out_grouped, out_loop, atol=1e-5, rtol=1e-5)

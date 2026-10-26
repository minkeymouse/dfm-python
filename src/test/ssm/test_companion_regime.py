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

    def test_forward_closed_loop_with_coeffs_is_the_scan_from_f_current(self, ssm_p1):
        """For AR(1) closed_loop_with_coeffs is the scan of eta_future from f_current (the state before step 1)."""
        batch, horizon, r = 2, 6, 3
        f_current = torch.randn(batch, r)
        eta_future = torch.randn(batch, horizon, r)
        ar_coeffs = torch.randn(batch, r, 1)
        B_diag = torch.randn(batch, r)
        expected = ssm_p1.forward_with_coeffs(eta_future, f_current, ar_coeffs, B_diag)
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


class TestCompanionNorm:
    """The L1 normalization of the AR coefficient rows."""

    def test_all_zero_row_is_left_unnormalized(self):
        ssm = iVDFMCompanionSSM(latent_dim=3, factor_order=3, device="cpu")
        x = torch.tensor([[0.5, -0.2, 0.1], [0.0, 0.0, 0.0], [0.3, 0.3, -0.4]], requires_grad=True)
        out = ssm.norm(x)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[1], torch.zeros(3), rtol=0, atol=0)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_nonzero_rows_are_normalized_exactly_as_before(self):
        ssm = iVDFMCompanionSSM(latent_dim=3, factor_order=3, device="cpu")
        x = torch.tensor([[0.5, -0.2, 0.1], [0.0, 0.0, 0.0], [0.3, 0.3, -0.4]])
        out = ssm.norm(x)
        for i in (0, 2):
            expected = x[i] / torch.linalg.norm(x[i], ord=1)
            torch.testing.assert_close(out[i], expected, rtol=0, atol=0)
        # without a zero row the result is the plain row-wise division, bit for bit
        y = x[[0, 2]]
        torch.testing.assert_close(ssm.norm(y), y / torch.linalg.norm(y, ord=1, dim=-1, keepdim=True),
                                   rtol=0, atol=0)

    def test_padded_factor_ar_coefficients_give_a_finite_impulse_response(self):
        ssm = iVDFMCompanionSSM(latent_dim=3, factor_order=3, device="cpu")
        with torch.no_grad():
            ssm.ar_coeffs[2] = 0.0
        assert torch.isfinite(ssm.get_impulse_response(12, use_cache=False)).all()


@pytest.mark.parametrize("with_coeffs", [False, True])
def test_ar1_closed_loop_steps_once_per_horizon(with_coeffs):
    """Step h of the AR(1) rollout is A^h f_t plus the innovations (it was A^(h+1) f_t)."""
    from dfm_python.ssm.companion import iVDFMCompanionSSM

    torch.manual_seed(0)
    ssm = iVDFMCompanionSSM(latent_dim=3, factor_order=1, device="cpu")
    f = torch.randn(2, 3)
    eta = torch.randn(2, 4, 3)
    with torch.no_grad():
        if with_coeffs:
            ar = torch.randn(2, 3, 1)
            B = torch.randn(2, 3).abs() + 0.1
            out = ssm.forward_closed_loop_with_coeffs(f, eta, ar, B)
            A, Bn = torch.sigmoid(ar[:, :, 0]), ssm._normed(B)
        else:
            out = ssm.forward_closed_loop(f, eta, horizon=4)
            A, Bn = torch.sigmoid(ssm.A), ssm._normed(ssm.B)
    expected, state = [], f
    for h in range(4):
        state = A * state + Bn * eta[:, h]
        expected.append(state)
    torch.testing.assert_close(out, torch.stack(expected, dim=1))


def test_ar2_closed_loop_steps_once_per_horizon():
    """The companion path (p > 1) already stepped once per horizon: step 1 is a_1 f_t + a_2 f_{t-1}=0 + B eta."""
    from dfm_python.ssm.companion import iVDFMCompanionSSM

    torch.manual_seed(1)
    ssm = iVDFMCompanionSSM(latent_dim=2, factor_order=2, device="cpu")
    f = torch.randn(1, 2)
    eta = torch.zeros(1, 1, 2)
    with torch.no_grad():
        out = ssm.forward_closed_loop(f, eta, horizon=1)
        a = ssm._normed(ssm.ar_coeffs)
    torch.testing.assert_close(out[0, 0], a[:, 0] * f[0])


def test_ar2_impulse_response_follows_the_ar_recursion():
    """H_0 = b, H_1 = a1 b, H_k = a1 H_{k-1} + a2 H_{k-2} per factor (the companion matrix had no dynamics)."""
    from dfm_python.ssm.companion import iVDFMCompanionSSM

    torch.manual_seed(2)
    ssm = iVDFMCompanionSSM(latent_dim=2, factor_order=2, device="cpu")
    with torch.no_grad():
        H = ssm.get_impulse_response(6, use_cache=False)
        a, b = ssm._normed(ssm.ar_coeffs), ssm._normed(ssm.B)
    expected = [b, a[:, 0] * b]
    for _ in range(4):
        expected.append(a[:, 0] * expected[-1] + a[:, 1] * expected[-2])
    torch.testing.assert_close(H, torch.stack(expected, dim=1))


def test_rollout_from_a_full_state_uses_the_lags():
    from dfm_python.ssm.companion import iVDFMCompanionSSM

    torch.manual_seed(3)
    ssm = iVDFMCompanionSSM(latent_dim=2, factor_order=2, device="cpu")
    f_t, f_lag = torch.randn(1, 2), torch.randn(1, 2)
    state = torch.stack([f_t, f_lag], dim=-1).reshape(1, 4)  # [f_t, f_{t-1}] per factor
    with torch.no_grad():
        out = ssm.forward_closed_loop(state, torch.zeros(1, 2, 2), horizon=2)
        a = ssm._normed(ssm.ar_coeffs)
    step1 = a[:, 0] * f_t[0] + a[:, 1] * f_lag[0]
    torch.testing.assert_close(out[0, 0], step1)
    torch.testing.assert_close(out[0, 1], a[:, 0] * step1 + a[:, 1] * f_t[0])


def test_c_is_a_fixed_selector():
    from dfm_python.ssm.companion import iVDFMCompanionSSM

    ssm = iVDFMCompanionSSM(latent_dim=3, factor_order=2, device="cpu")
    assert "C" not in dict(ssm.named_parameters()) and "C" in ssm.state_dict()
    torch.testing.assert_close(ssm.C, torch.eye(3).repeat_interleave(2, dim=1) * torch.tensor([1.0, 0.0]).repeat(3))

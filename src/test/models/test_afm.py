"""Tests for the Attention Factor Model (models.afm).

Covers the residual-construction math (attention weights, closed-form loadings,
residual projection identity), the LongConv filter, the three paper model
classes (attention+longconv, pca+longconv, pca+ou), the training loop, and
save/load. Small synthetic panels only — this is a from-the-paper unit test, not
an empirical reproduction.
"""

import numpy as np
import pytest
import torch

from dfm_python.models.afm import (AFM, AttentionFactors, PCAFactors,
                                   LongConv1d, residuals, ridge_loadings,
                                   avellaneda_lee_weights)
from dfm_python.dataset.afm_dataset import AFMDataset
from dfm_python.config.schema.model import AFMConfig
from dfm_python.config.schema.results import AFMResult
from dfm_python.utils.errors import ModelNotTrainedError, ConfigurationError


def _make_data(T=60, N=6, M=8, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, N)) * 0.01
    characteristics = rng.standard_normal((T, N, M))
    return returns, characteristics


class TestAttentionFactors:
    def test_factor_weight_shapes_and_normalization(self):
        block = AttentionFactors(n_char=8, n_factors=3, embed_dim=16)
        X = torch.randn(4, 6, 8)                      # (B, N, M)
        omega_f, beta_t = block(X)
        assert omega_f.shape == (4, 3, 6)             # (B, K, N)
        assert beta_t.shape == (4, 6, 3)              # (B, N, K)
        # softmax over assets: each factor row sums to 1
        row_sums = omega_f.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_residual_projection_identity(self):
        block = AttentionFactors(n_char=8, n_factors=3, embed_dim=16)
        X = torch.randn(6, 8)
        R = torch.randn(6)
        omega_f, beta_t = block(X)
        eps_fn = residuals(omega_f, beta_t, R)
        # omega_eps R = (I - beta^T omega_f) R must equal R - beta^T (omega_f R)
        I = torch.eye(6)
        omega_eps = I - beta_t @ omega_f
        eps_proj = omega_eps @ R
        assert torch.allclose(eps_fn, eps_proj, atol=1e-5)

    def test_ridge_loadings_shape(self):
        omega_f = torch.randn(2, 4, 7)               # (B, K, N)
        beta_t = ridge_loadings(omega_f, ridge=1e-2)
        assert beta_t.shape == (2, 7, 4)             # (B, N, K)


class TestPCAFactors:
    def test_pca_shapes(self):
        pca = PCAFactors(n_factors=3)
        Rwin = torch.randn(2, 40, 6)                 # (B, L, N)
        omega_f, beta_t = pca(Rwin)
        assert omega_f.shape == (2, 3, 6)
        assert beta_t.shape == (2, 6, 3)


class TestLongConv:
    def test_output_shape_and_history_use(self):
        conv = LongConv1d(seq_len=10, n_kernels=8)
        u = torch.randn(4, 6, 10)                     # (B, N, s)
        w = conv(u)
        assert w.shape == (4, 6)
        # a change in the history changes the output (uses the history)
        u2 = u.clone(); u2[..., 3] += 1.0
        assert not torch.allclose(conv(u), conv(u2))

    def test_wrong_length_raises(self):
        conv = LongConv1d(seq_len=10, n_kernels=8)
        with pytest.raises(ValueError):
            conv(torch.randn(4, 6, 9))


class TestAFM:
    def _dataset(self, T=60, N=6, M=8, seed=0):
        returns, characteristics = _make_data(T, N, M, seed)
        return AFMDataset(returns=returns, characteristics=characteristics)

    def test_initialization(self):
        ds = self._dataset()
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=2)
        assert model.n_char == 8
        assert model.n_factors == 3
        assert model.factor_model == "attention"

    def test_forward_shapes(self):
        ds = self._dataset()
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=2)
        R = torch.randn(2, 60, 6)
        X = torch.randn(2, 60, 6, 8)
        out = model.forward(R, X)
        assert out["residuals"].shape == (2, 60, 6)
        assert out["book_returns"].shape == (2, 50)   # W - hist_len
        assert out["weights"].shape == (2, 50, 6)
        assert torch.isfinite(out["loss"])

    def test_predict_before_fit_raises(self):
        ds = self._dataset()
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=2)
        with pytest.raises(ModelNotTrainedError):
            model.predict()

    def test_full_workflow_attention(self):
        ds = self._dataset()
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=3, seed=0)
        state = model.fit()
        assert state.num_iter >= 1
        out = model.predict()
        assert out["residuals"].shape[-1] == 6
        result = model.get_result()
        assert isinstance(result, AFMResult)
        assert result.sharpe is not None
        assert result.factor_model == "attention"

    def test_training_reduces_loss(self):
        ds = self._dataset(seed=1)
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=1, seed=0)
        R, X = model._batch_from_dataset()
        loss0 = float(model.forward(R, X)["loss"].detach())
        model.max_epochs = 40
        model.fit()
        loss1 = float(model.forward(R, X)["loss"].detach())
        assert loss1 <= loss0 + 1e-6                  # objective did not worsen

    def test_pca_longconv(self):
        returns, _ = _make_data()
        ds = AFMDataset(returns=returns)              # no characteristics
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=3,
                    factor_model="pca", trading="longconv", seed=0)
        model.fit()
        assert model.get_result().factor_model == "pca"

    def test_pca_ou_baseline(self):
        returns, _ = _make_data()
        ds = AFMDataset(returns=returns)
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=5,
                    factor_model="pca", trading="ou")
        state = model.fit()                           # no learnable params -> single pass
        assert state.num_iter == 1
        assert model.get_result().trading == "ou"

    def test_save_load_roundtrip(self, tmp_path):
        ds = self._dataset()
        model = AFM(dataset=ds, n_factors=3, hist_len=10, max_epochs=2, seed=0)
        model.fit()
        path = tmp_path / "afm.pt"
        model.save(path)
        loaded = AFM.load(path)
        assert loaded.n_factors == 3
        assert loaded.factor_model == "attention"
        assert loaded.training_state is not None


class TestAFMConfig:
    def test_from_dict(self):
        cfg = AFMConfig.from_dict({"clock": "d", "num_factors": 5,
                                   "factor_model": "pca", "hist_len": 15})
        assert cfg.num_factors == 5
        assert cfg.factor_model == "pca"
        assert cfg.hist_len == 15
        assert cfg.turnover_cost == 5e-4              # default preserved

    def test_invalid_factor_model_raises(self):
        with pytest.raises(ConfigurationError):
            AFMConfig(factor_model="bogus")

    def test_invalid_trading_raises(self):
        with pytest.raises(ConfigurationError):
            AFMConfig(trading="bogus")

    def test_config_drives_model(self):
        cfg = AFMConfig.from_dict({"num_factors": 4, "hist_len": 8,
                                   "embed_dim": 12, "max_epochs": 2})
        returns, characteristics = _make_data(T=40, N=5, M=7)
        ds = AFMDataset(returns=returns, characteristics=characteristics)
        model = AFM(dataset=ds, config=cfg)
        assert model.n_factors == 4
        assert model.hist_len == 8
        model.fit()
        assert model.get_result().num_factors == 4

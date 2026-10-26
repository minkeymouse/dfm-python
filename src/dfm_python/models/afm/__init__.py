"""Attention Factor Model (AFM) for statistical arbitrage.

Epstein et al. (2025), "Attention Factors for Statistical Arbitrage": learned
conditional factors via cross-sectional attention, a LongConv residual trading
filter, and an after-cost Sharpe + explained-variance objective, with PCA and
Avellaneda--Lee benchmarks. ``AFM`` is the public estimator; the training code
is the private engine ``_engine.AFMEngine``. The factor models, the filter and
the benchmarks use torch and are imported on first access (they need the
``deep`` extra).
"""

from ...utils.optional import lazy_torch_exports
from .estimator import AFM

__all__ = [
    "AFM",
    "AttentionFactors",
    "PCAFactors",
    "LongConv1d",
    "residuals",
    "ridge_loadings",
    "avellaneda_lee_weights",
    "ou_sscore",
]

__getattr__ = lazy_torch_exports(__name__, {
    "AttentionFactors": ".factors",
    "PCAFactors": ".factors",
    "residuals": ".factors",
    "ridge_loadings": ".factors",
    "LongConv1d": ".longconv",
    "avellaneda_lee_weights": ".baselines",
    "ou_sscore": ".baselines",
})

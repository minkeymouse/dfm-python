"""Tests for factor extraction and diagnostics."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import warnings

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import dfm_python as dfm
from dfm_python.config import Params


def test_factor_not_near_zero():
    """Test that factors are not near-zero after training."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    data_file = base_dir / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Test data files not found")
    
    params = Params(max_iter=10, threshold=1e-4)
    dfm.from_spec(spec_file, params=params)
    dfm.load_data(data_file, sample_start='2015-01-01', sample_end='2022-12-31')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfm.train()
    
    result = dfm.get_result()
    assert result is not None, "Training failed"
    
    Z = result.Z
    Q = result.Q
    C = result.C
    
    # Check first factor (Block_Global)
    if Z.shape[1] > 0:
        factor0 = Z[:, 0]
        assert np.std(factor0) > 1e-10, "Factor 0 should have non-zero variance"
        assert np.abs(np.mean(factor0)) < 1e6 or np.std(factor0) > 1e-10, \
            "Factor 0 should not be essentially zero"
    
    # Check Q[0,0] (Block_Global innovation variance)
    if Q.shape[0] > 0:
        assert Q[0, 0] >= 1e-8, f"Q[0,0] should be >= 1e-8, got {Q[0, 0]:.10e}"
    
    # Check loadings on first factor
    if C.shape[1] > 0:
        loadings_factor0 = C[:, 0]
        max_loading = np.abs(loadings_factor0).max()
        assert max_loading > 1e-6, \
            f"Loadings on factor 0 should not all be near-zero, max={max_loading:.10e}"


def test_factor_innovation_variance():
    """Test that innovation variances are above minimum threshold."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    data_file = base_dir / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Test data files not found")
    
    params = Params(max_iter=5, threshold=1e-3)
    dfm.from_spec(spec_file, params=params)
    dfm.load_data(data_file, sample_start='2020-01-01', sample_end='2022-12-31')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfm.train()
    
    result = dfm.get_result()
    assert result is not None
    
    Q = result.Q
    Q_diag = np.diag(Q)
    
    # All diagonal elements should be above minimum
    assert np.all(Q_diag >= 1e-8), \
        f"Q diagonal should be >= 1e-8, min={Q_diag.min():.10e}"


def test_factor_loadings_nonzero():
    """Test that factor loadings are not all zero."""
    base_dir = project_root
    spec_file = base_dir / 'data' / 'sample_spec.csv'
    data_file = base_dir / 'data' / 'sample_data.csv'
    
    if not spec_file.exists() or not data_file.exists():
        pytest.skip("Test data files not found")
    
    params = Params(max_iter=5, threshold=1e-3)
    dfm.from_spec(spec_file, params=params)
    dfm.load_data(data_file, sample_start='2020-01-01', sample_end='2022-12-31')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfm.train()
    
    result = dfm.get_result()
    assert result is not None
    
    C = result.C
    
    # Should have non-zero loadings
    assert not np.all(C == 0), "Loading matrix should not be all zeros"
    assert np.sum(C != 0) > 0, "Should have at least some non-zero loadings"
    
    # Per-series max loadings should not all be zero
    max_loadings = np.abs(C).max(axis=1)
    assert np.sum(max_loadings > 1e-6) > 0, \
        "At least some series should have meaningful loadings"


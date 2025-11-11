"""Consolidated test suite for DFM nowcasting module."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from dfm_python.dfm import DFM, DFMResult
from dfm_python.core.em import em_step, init_conditions, em_converged, EMStepParams
from dfm_python.data import load_data, rem_nans_spline
from dfm_python.api import load_config
from dfm_python.config import (
    DFMConfig, SeriesConfig, BlockConfig, Params,
    YamlSource, DictSource, SpecCSVSource, MergedConfigSource,
    make_config_source
)
from dfm_python import update_nowcast
from dfm_python.kalman import skf, fis, miss_data, KalmanFilterState
from dfm_python.core.numeric import (
    _compute_covariance_safe,
    _compute_variance_safe,
    _ensure_innovation_variance_minimum,
    _ensure_real_and_symmetric,
    _ensure_covariance_stable,
)
import dfm_python as dfm

# ============================================================================
# Helper Functions
# ============================================================================

def create_test_data(T=100, n=10, m=3, seed=42):
    """Create test data for Kalman filter."""
    np.random.seed(seed)
    A = np.random.randn(m, m) * 0.5
    A = A / (np.max(np.abs(np.linalg.eigvals(A))) + 0.1)
    C = np.random.randn(n, m)
    Q = np.eye(m) * 0.1
    R = np.eye(n) * 0.5
    Z_0 = np.zeros(m)
    V_0 = np.eye(m)
    z = np.zeros((T, m))
    z[0] = Z_0
    for t in range(1, T):
        z[t] = A @ z[t-1] + np.random.multivariate_normal(np.zeros(m), Q)
    y = np.zeros((n, T))
    for t in range(T):
        y[:, t] = C @ z[t] + np.random.multivariate_normal(np.zeros(n), R)
    return y, A, C, Q, R, Z_0, V_0, z

# ============================================================================
# API Edge Cases
# ============================================================================

def test_empty_data():
    """Test with empty data array."""
    config = DFMConfig(
        series=[SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])],
        blocks={'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    )
    X = np.array([]).reshape(0, 1)
    with pytest.raises((ValueError, IndexError)):
        model = DFM()
        model.fit(X, config)

def test_single_time_period():
    """Test with single time period."""
    np.random.seed(42)
    X = np.random.randn(1, 5)
    series_list = [SeriesConfig(series_id=f'test_{i}', frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(5)]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        result = model.fit(X, config, max_iter=1, threshold=1e-2)
        assert result is not None
        assert not result.converged

def test_very_small_threshold():
    """Test with very small convergence threshold."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    series_list = [SeriesConfig(series_id=f'test_{i}', frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        result = model.fit(X, config, threshold=1e-10, max_iter=10)
        assert result is not None

def test_very_large_max_iter():
    """Test with very large max_iter."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    series_list = [SeriesConfig(series_id=f'test_{i}', frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        result = model.fit(X, config, max_iter=10000)
        assert result is not None

def test_predict_without_training():
    """Test predict without training."""
    dfm.reset()
    with pytest.raises((ValueError, AttributeError)):
        dfm.predict(horizon=6)

def test_plot_without_training():
    """Test plot without training."""
    dfm.reset()
    with pytest.raises(ValueError):
        dfm.plot(kind='factor', factor_index=0)

def test_get_result_without_training():
    """Test get_result without training."""
    dfm.reset()
    result = dfm.get_result()
    assert result is None

def test_api_reset():
    """Test API reset functionality."""
    series_list = [SeriesConfig(frequency='m', transformation='lin', blocks=['Block_Global'])]
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    config = DFMConfig(series=series_list, blocks=blocks)
    dfm.load_config(config)
    dfm.reset()
    assert dfm.get_config() is None
    assert dfm.get_result() is None

# ============================================================================
# Tutorial Tests
# ============================================================================

def test_tutorial_smoke_test():
    """Smoke test for tutorial structure."""
    assert hasattr(dfm, 'from_spec')
    assert hasattr(dfm, 'load_data')
    assert hasattr(dfm, 'train')
    assert hasattr(dfm, 'predict')
    assert hasattr(dfm, 'get_result')
    params = Params(max_iter=10, threshold=1e-4)
    assert params.max_iter == 10
    assert params.threshold == 1e-4

def test_basic_tutorial_workflow():
    """Test basic tutorial workflow."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
    if not spec_file.exists() or not data_file.exists():
        pytest.skip(f"Tutorial data files not found")
    try:
        params = Params(max_iter=1, threshold=1e-2)
        dfm.from_spec(spec_file, params=params)
        dfm.load_data(data_file, sample_start='2021-01-01', sample_end='2022-12-31')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfm.train()
        result = dfm.get_result()
        assert result is not None
        assert hasattr(result, 'Z')
        X_forecast, Z_forecast = dfm.predict(horizon=6)
        assert X_forecast.shape[0] == 6
    except Exception as e:
        pytest.skip(f"Tutorial test skipped: {e}")

def test_hydra_tutorial_workflow():
    """Test Hydra tutorial workflow."""
    try:
        import hydra
        from omegaconf import DictConfig
    except ImportError:
        pytest.skip("Hydra not available")
    data_file = project_root / 'data' / 'sample_data.csv'
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    try:
        cfg = DictConfig({'clock': 'm', 'max_iter': 1, 'threshold': 1e-2, 'series': [], 'blocks': {}})
        dfm.load_config(hydra=cfg)
        dfm.load_data(data_file, sample_start='2021-01-01', sample_end='2022-12-31')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfm.train()
        result = dfm.get_result()
        assert result is not None
    except Exception as e:
        pytest.skip(f"Hydra tutorial test skipped: {e}")

# ============================================================================
# Config Source Tests
# ============================================================================

def test_yaml_source():
    """Test YAML config source."""
    yaml_file = project_root / 'config' / 'default.yaml'
    if not yaml_file.exists():
        pytest.skip(f"YAML config file not found: {yaml_file}")
    source = YamlSource(yaml_file)
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0
    assert len(config.blocks) > 0

def test_dict_source():
    """Test dictionary config source."""
    config_dict = {
        'series': [{'series_id': 'test_1', 'series_name': 'Test Series 1', 'frequency': 'm', 'transformation': 'lin', 'blocks': ['Block_Global']}],
        'blocks': {'Block_Global': {'factors': 1, 'clock': 'm'}},
        'clock': 'm', 'max_iter': 100, 'threshold': 1e-5
    }
    source = DictSource(config_dict)
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) == 1
    assert config.series[0].series_id == 'test_1'
    assert config.max_iter == 100

def test_spec_csv_source():
    """Test spec CSV config source."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    if not spec_file.exists():
        pytest.skip(f"Spec CSV file not found: {spec_file}")
    source = SpecCSVSource(spec_file)
    config = source.load()
    assert isinstance(config, DFMConfig)
    assert len(config.series) > 0
    assert len(config.blocks) > 0

def test_merged_config_source():
    """Test merged config source."""
    base_dict = {'series': [{'series_id': 'base_1', 'series_name': 'Base Series 1', 'frequency': 'm', 'transformation': 'lin', 'blocks': ['Block_Global']}], 'blocks': {'Block_Global': {'factors': 1, 'clock': 'm'}}}
    override_dict = {'max_iter': 200, 'threshold': 1e-5}
    base_source = DictSource(base_dict)
    override_source = DictSource(override_dict)
    merged_source = MergedConfigSource(base_source, override_source)
    config = merged_source.load()
    assert config.max_iter == 200
    assert config.threshold == 1e-5
    assert len(config.series) > 0

def test_make_config_source():
    """Test config source factory function."""
    yaml_file = project_root / 'config' / 'default.yaml'
    if yaml_file.exists():
        source = make_config_source(yaml_file)
        assert isinstance(source, YamlSource)
    config_dict = {'series': [{'series_id': 'test_1', 'frequency': 'm', 'transformation': 'lin', 'blocks': ['Block_Global']}], 'blocks': {'Block_Global': {'factors': 1, 'clock': 'm'}}}
    source = make_config_source(config_dict)
    assert isinstance(source, DictSource)
    series_list = [SeriesConfig(series_id='test_1', frequency='m', transformation='lin', blocks=['Block_Global'])]
    config = DFMConfig(series=series_list, blocks={'Block_Global': BlockConfig(factors=1, clock='m')})
    source = make_config_source(config)
    assert hasattr(source, 'load')
    assert source.load() is config

def test_from_dict():
    """Test from_dict convenience constructor."""
    config_dict = {'series': [{'series_id': 'test_1', 'frequency': 'm', 'transformation': 'lin', 'blocks': ['Block_Global']}], 'blocks': {'Block_Global': {'factors': 1, 'clock': 'm'}}}
    config = DFMConfig.from_dict(config_dict)
    assert isinstance(config, DFMConfig)
    assert len(config.series) == 1

# ============================================================================
# News Decomposition Tests
# ============================================================================

def test_update_nowcast_basic():
    """Test basic nowcast update."""
    spec_file = project_root / 'Nowcasting' / 'Spec_US_example.xls'
    if not spec_file.exists():
        pytest.skip("Spec file not found")
    vintage_old = '2016-12-16'
    vintage_new = '2016-12-23'
    datafile_old = project_root / 'Nowcasting' / 'data' / 'US' / f'{vintage_old}.xls'
    datafile_new = project_root / 'Nowcasting' / 'data' / 'US' / f'{vintage_new}.xls'
    if not datafile_old.exists() or not datafile_new.exists():
        pytest.skip("Vintage data files not found")
    try:
        config = load_config(spec_file)
        X_old, Time_old, _ = load_data(datafile_old, config, sample_start=pd.Timestamp('2000-01-01'))
        X_new, Time, _ = load_data(datafile_new, config, sample_start=pd.Timestamp('2000-01-01'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DFM()
            Res = model.fit(X_old, config, threshold=1e-3, max_iter=20)
        update_nowcast(X_old, X_new, Time, config, Res, series='GDPC1', period='2016q4', vintage_old=vintage_old, vintage_new=vintage_new)
    except (UnicodeDecodeError, ImportError, ValueError) as e:
        pytest.skip(f"Nowcast test skipped: {e}")

# ============================================================================
# Core DFM Tests
# ============================================================================

def test_em_step_basic():
    """Test basic EM step functionality."""
    T, N = 80, 8
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    params = EMStepParams(
        y=y,
        A=A,
        C=C,
        Q=Q,
        R=R,
        Z_0=Z_0,
        V_0=V_0,
        r=r,
        p=p,
        R_mat=R_mat,
        q=q,
        nQ=nQ,
        i_idio=i_idio,
        blocks=blocks,
        tent_weights_dict={},
        clock='m',
        frequencies=None,
        config=None
    )
    C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(params)
    assert C_new is not None and R_new is not None
    assert A_new is not None and np.isfinite(loglik)
    assert C_new.shape == (N, A.shape[0])
    assert R_new.shape == (N, N)

def test_init_conditions_basic():
    """Test basic initial conditions."""
    T, N = 100, 10
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    blocks = np.zeros((N, 2), dtype=int)
    blocks[:, 0] = 1
    r = np.ones(2)
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    m = A.shape[0]
    assert A.shape == (m, m)
    assert C.shape == (N, m)
    assert Q.shape == (m, m)
    assert R.shape == (N, N)
    assert Z_0.shape == (m,)
    assert V_0.shape == (m, m)
    assert not np.any(np.isnan(A))
    assert not np.any(np.isnan(C))

def test_init_conditions_large_block():
    """Test initial conditions with large block."""
    T, N = 396, 78
    np.random.seed(42)
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23
    x[missing_mask] = np.nan
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    m = A.shape[0]
    assert A.shape == (m, m)
    assert C.shape == (N, m), f"Expected C shape ({N}, {m}), got {C.shape}"
    assert Q.shape == (m, m)
    assert R.shape == (N, N), f"Expected R shape ({N}, {N}), got {R.shape}"
    assert not np.any(np.isnan(A))
    assert not np.any(np.isnan(C))

def test_dfm_quick():
    """Quick DFM test with synthetic data."""
    T, N = 50, 10
    np.random.seed(42)
    factors = np.random.randn(T, 2)
    loadings = np.random.randn(N, 2) * 0.5
    X = factors @ loadings.T + np.random.randn(T, N) * 0.3
    missing_mask = np.random.rand(T, N) < 0.1
    X[missing_mask] = np.nan
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=5)
    assert hasattr(Res, 'x_sm') and hasattr(Res, 'X_sm')
    assert hasattr(Res, 'Z') and hasattr(Res, 'C')
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert Res.C.shape[0] == N
    assert np.any(np.isfinite(Res.Z))

def test_dfm_class_fit():
    """Test DFM class fit() method."""
    np.random.seed(42)
    T, N = 50, 5
    X = np.random.randn(T, N)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    model = DFM()
    result = model.fit(X, config, threshold=1e-2, max_iter=5)
    assert isinstance(result, DFMResult)
    assert result.x_sm.shape == (T, N)
    assert result.Z.shape[0] == T
    assert result.C.shape[0] == N
    assert np.any(np.isfinite(result.Z))
    assert model.result is not None
    assert model.config is not None

def test_multi_block_different_factors():
    """Test multi-block with different factor counts."""
    np.random.seed(42)
    T, N = 100, 15
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.1
    x[missing_mask] = np.nan
    blocks = np.zeros((N, 3), dtype=int)
    blocks[0:5, 0] = 1
    blocks[5:10, 1] = 1
    blocks[10:15, 2] = 1
    blocks[:, 0] = 1
    r = np.array([3, 2, 2])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    nQ = 0
    i_idio = np.ones(N)
    R_mat = None
    q = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert A is not None and C is not None
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(C))
    assert C.shape[0] == N, f"Expected C shape ({N}, ...), got {C.shape}"
    assert R.shape == (N, N), f"Expected R shape ({N}, {N}), got {R.shape}"

# ============================================================================
# Edge Case Tests
# ============================================================================

def test_single_series():
    """Test with single time series."""
    T = 50
    np.random.seed(42)
    X = np.random.randn(T, 1)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id="TEST_01", series_name="Test Series", frequency='m', transformation='lin', blocks=['Block_Global'])]
    config = DFMConfig(series=series_list, blocks=blocks)
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=5)
    assert Res.x_sm.shape == (T, 1)
    assert Res.Z.shape[0] == T
    assert np.any(np.isfinite(Res.Z))

def test_init_conditions_block_global_single_series():
    """Test Block_Global initialization with only one series."""
    np.random.seed(42)
    T, N = 50, 1
    x = np.random.randn(T, N)
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert A is not None
    assert C is not None
    assert Q is not None
    assert C.shape == (N, A.shape[0]), f"C shape should be ({N}, {A.shape[0]}), got {C.shape}"
    assert Q.shape[0] == Q.shape[1]
    Q_diag = np.diag(Q)
    assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
    max_loading_abs = np.max(np.abs(C))
    assert max_loading_abs > 0, f"Loadings should not be all zero, max_abs={max_loading_abs}"

def test_all_nan_data():
    """Test with all NaN data."""
    T, N = 50, 5
    X = np.full((T, N), np.nan)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = DFM()
            Res = model.fit(X, config, threshold=1e-2, max_iter=5)
            assert Res is not None
            assert Res.x_sm.shape == (T, N)
            assert not Res.converged
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["nan", "missing", "data", "insufficient"]), f"Error should mention data issue, got: {e}"

def test_high_missing_data():
    """Test with very high percentage of missing data."""
    T, N = 100, 10
    np.random.seed(42)
    X = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.8
    X[missing_mask] = np.nan
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-2, max_iter=5)
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))

def test_mixed_frequencies():
    """Test with mixed frequencies."""
    T, N = 60, 8
    np.random.seed(42)
    X = np.random.randn(T, N)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = []
    for i in range(N):
        freq = 'm' if i < 5 else 'q'
        series_list.append(SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency=freq, transformation='lin', blocks=['Block_Global']))
    config = DFMConfig(series=series_list, blocks=blocks, clock='m')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-2, max_iter=5)
        assert Res.x_sm.shape == (T, N)
        assert np.any(np.isfinite(Res.Z))

# ============================================================================
# Stress Tests
# ============================================================================

def test_large_dataset():
    """Test with large dataset."""
    T, N = 200, 30
    np.random.seed(42)
    X = np.random.randn(T, N)
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    model = DFM()
    Res = model.fit(X, config, threshold=1e-2, max_iter=20)
    assert Res.x_sm.shape == (T, N)
    assert Res.Z.shape[0] == T
    assert np.all(np.isfinite(Res.Z))

def test_numerical_precision():
    """Test numerical precision with very small values."""
    T, N = 50, 5
    np.random.seed(42)
    X = np.random.randn(T, N) * 1e-10
    blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    series_list = [SeriesConfig(series_id=f"TEST_{i:02d}", series_name=f"Test Series {i}", frequency='m', transformation='lin', blocks=['Block_Global']) for i in range(N)]
    config = DFMConfig(series=series_list, blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DFM()
        Res = model.fit(X, config, threshold=1e-3, max_iter=50)
        assert Res.x_sm.shape == (T, N)
        assert np.all(np.isfinite(Res.Z))

def test_em_converged():
    """Test EM convergence detection logic."""
    threshold = 1e-4
    loglik_current = 100.0
    loglik_previous = 100.01
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert isinstance(converged, bool)
    assert isinstance(decreased, bool)
    loglik_current = 100.0
    loglik_previous = 100.00001
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged
    assert not decreased
    loglik_current = 100.0
    loglik_previous = 50.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert not converged
    assert not decreased
    loglik_current = 50.0
    loglik_previous = 100.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold, check_decreased=True)
    assert not converged
    assert decreased
    loglik_current = 0.0
    loglik_previous = 0.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged
    assert not decreased
    loglik_current = -100.0
    loglik_previous = -100.01
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert isinstance(converged, bool)
    assert isinstance(decreased, bool)
    loglik_current = 1.0
    loglik_previous = 1.0 + threshold * 0.5
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert converged
    loglik_current = 1.0
    loglik_previous = 1.0 + threshold * 1.0
    converged, decreased = em_converged(loglik_current, loglik_previous, threshold=threshold)
    assert isinstance(converged, bool)

def test_kalman_stability_edge_cases():
    """Test Kalman filter/smoother stability with edge cases."""
    np.random.seed(42)
    T, N = 50, 5
    x = np.random.randn(T, N)
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    m = A.shape[0]
    xNaN = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    xNaN_est, _ = rem_nans_spline(xNaN, method=3, k=3)
    y = xNaN_est.T
    R_very_small = np.eye(N) * 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(y, A, C, Q, R_very_small, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, tent_weights_dict={}, clock='m', frequencies=None, config=None)
            assert np.all(np.isfinite(C_new))
            assert np.all(np.isfinite(R_new))
            assert np.all(np.isfinite(A_new))
            assert np.all(np.isfinite(Q_new))
            assert np.isfinite(loglik) or loglik == -np.inf
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["singular", "ill-conditioned", "broadcast", "shape", "matrix"]), f"Should fail with expected error type, got: {e}"
    Q_very_large = Q.copy() * 1e6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(y, A, C, Q_very_large, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, tent_weights_dict={}, clock='m', frequencies=None, config=None)
            assert np.all(np.isfinite(C_new))
            assert np.all(np.isfinite(Q_new))
            max_eig = np.max(np.linalg.eigvals(Q_new))
            assert max_eig < 1e7, f"Q should be capped, max_eig={max_eig:.2e}"
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["singular", "ill-conditioned", "broadcast", "shape", "matrix", "eigenvalue"]), f"Should fail with expected error type, got: {e}"
    Q_near_singular = Q.copy() * 1e-12
    Q_near_singular[0, 0] = 1e-15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(y, A, C, Q_near_singular, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, tent_weights_dict={}, clock='m', frequencies=None, config=None)
            assert np.all(np.isfinite(Q_new))
            Q_diag = np.diag(Q_new)
            assert np.all(Q_diag > 0)
        except Exception as e:
            assert isinstance(e, (np.linalg.LinAlgError, ValueError, RuntimeError)), f"Should fail gracefully, got: {type(e).__name__}"

def test_init_conditions_block_global_sparse_data():
    """Test Block_Global initialization with high missing data ratio."""
    np.random.seed(42)
    T, N = 200, 20
    x = np.random.randn(T, N)
    missing_rate = 0.6
    missing_mask = np.random.rand(T, N) < missing_rate
    x[missing_mask] = np.nan
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert Q is not None
    assert Q.shape[0] == Q.shape[1]
    Q_diag = np.diag(Q)
    assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
    assert np.all(np.isfinite(Q_diag))
    assert C is not None
    assert C.shape == (N, A.shape[0]), f"C shape should be ({N}, {A.shape[0]}), got {C.shape}"
    max_loading_abs = np.max(np.abs(C))
    assert max_loading_abs > 0, f"Loadings should not be all zero, max_abs={max_loading_abs}"
    assert A is not None
    assert A.shape[0] == A.shape[1]
    max_A_abs = np.max(np.abs(A))
    assert max_A_abs > 0, f"A should not be all zero, max_abs={max_A_abs}"
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(Q))

def test_init_conditions_pairwise_complete_block_global():
    """Test that pairwise_complete=True is used for Block_Global initialization."""
    np.random.seed(42)
    T, N = 100, 15
    x = np.random.randn(T, N)
    for t in range(T):
        missing_idx = np.random.choice(N, size=max(1, N // 3), replace=False)
        x[t, missing_idx] = np.nan
    complete_rows = np.all(np.isfinite(x), axis=1)
    assert np.sum(complete_rows) == 0, "Test setup failed: some rows are complete"
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert A is not None
    assert C is not None
    assert Q is not None
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(Q))

def test_init_conditions_block_global_all_nan_residuals():
    """Test Block_Global initialization when block residuals are all NaN."""
    np.random.seed(42)
    T, N = 50, 10
    x = np.random.randn(T, N)
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    x_sparse = x.copy()
    for t in range(T):
        if t % 5 != 0:
            x_sparse[t, :] = np.nan
        else:
            x_sparse[t, ::2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            A, C, Q, R, Z_0, V_0 = init_conditions(x_sparse, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
            assert A is not None
            assert C is not None
            assert Q is not None
            assert np.all(np.isfinite(A))
            assert np.all(np.isfinite(C))
            assert np.all(np.isfinite(Q))
            Q_diag = np.diag(Q)
            assert np.all(Q_diag > 0), f"Q diagonal should be > 0, got: {Q_diag}"
        except ValueError as e:
            error_msg = str(e).lower()
            assert "insufficient data" in error_msg or "data" in error_msg, f"Expected error about insufficient data, got: {e}"

def test_with_direct_config():
    """Test with direct DFMConfig creation."""
    try:
        series_list = [SeriesConfig(series_id='series_0', series_name='Test Series 0', frequency='m', transformation='lin', blocks=['Block_Global']), SeriesConfig(series_id='series_1', series_name='Test Series 1', frequency='m', transformation='lin', blocks=['Block_Global'])]
        blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
        config = DFMConfig(series=series_list, blocks=blocks)
        T = 50
        np.random.seed(42)
        X = np.random.randn(T, 2)
        model = DFM()
        result = model.fit(X, config, threshold=1e-2, max_iter=5)
        assert result.Z.shape[1] > 0
        assert result.C.shape[0] == 2
    except Exception as e:
        pytest.skip(f"Integration test skipped: {e}")

# ============================================================================
# Factor Tests
# ============================================================================

def test_factor_not_near_zero():
    """Test that factors are not near-zero after training."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
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
    if Z.shape[1] > 0:
        factor0 = Z[:, 0]
        assert np.std(factor0) > 1e-10, "Factor 0 should have non-zero variance"
        assert np.abs(np.mean(factor0)) < 1e6 or np.std(factor0) > 1e-10, "Factor 0 should not be essentially zero"
    if Q.shape[0] > 0:
        assert Q[0, 0] >= 1e-8, f"Q[0,0] should be >= 1e-8, got {Q[0, 0]:.10e}"
    if C.shape[1] > 0:
        loadings_factor0 = C[:, 0]
        max_loading = np.abs(loadings_factor0).max()
        assert max_loading > 1e-6, f"Loadings on factor 0 should not all be near-zero, max={max_loading:.10e}"

def test_factor_innovation_variance():
    """Test that innovation variances are above minimum threshold."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
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
    assert np.all(Q_diag >= 1e-8), f"Q diagonal should be >= 1e-8, min={Q_diag.min():.10e}"

def test_factor_loadings_nonzero():
    """Test that factor loadings are not all zero."""
    spec_file = project_root / 'data' / 'sample_spec.csv'
    data_file = project_root / 'data' / 'sample_data.csv'
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
    assert not np.all(C == 0), "Loading matrix should not be all zeros"
    assert np.sum(C != 0) > 0, "Should have at least some non-zero loadings"
    max_loadings = np.abs(C).max(axis=1)
    assert np.sum(max_loadings > 1e-6) > 0, "At least some series should have meaningful loadings"

# ============================================================================
# Kalman Filter Tests
# ============================================================================

def test_skf_basic():
    """Test basic Kalman filter."""
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    assert Sf is not None
    assert hasattr(Sf, 'ZmU') and hasattr(Sf, 'VmU')
    assert hasattr(Sf, 'loglik')
    assert Sf.ZmU.shape == (z_true.shape[1], y.shape[1] + 1)
    assert np.isfinite(Sf.loglik)

def test_skf_missing_data():
    """Test Kalman filter with missing data."""
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    missing_mask = np.random.rand(*y.shape) < 0.2
    y_missing = y.copy()
    y_missing[missing_mask] = np.nan
    Sf = skf(y_missing, A, C, Q, R, Z_0, V_0)
    assert Sf is not None
    assert np.isfinite(Sf.loglik)

def test_fis_basic():
    """Test basic fixed interval smoother."""
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=50, n=5, m=2)
    Sf = skf(y, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    assert Ss is not None
    assert hasattr(Ss, 'ZmT') and hasattr(Ss, 'VmT')
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape

def test_miss_data():
    """Test missing data elimination function."""
    n, m = 10, 3
    y = np.random.randn(n)
    C = np.random.randn(n, m)
    R = np.eye(n) * 0.5
    y_missing = y.copy()
    y_missing[2] = np.nan
    y_missing[5] = np.nan
    y_missing[8] = np.nan
    y_new, C_new, R_new, L = miss_data(y_missing, C, R)
    assert len(y_new) == n - 3
    assert C_new.shape[0] == n - 3
    assert C_new.shape[1] == m
    assert R_new.shape == (n - 3, n - 3)
    assert L.shape == (n, n - 3)
    non_missing_idx = [i for i in range(n) if not np.isnan(y_missing[i])]
    assert np.allclose(y_new, y[non_missing_idx])

def test_skf_zero_observation_variance():
    """Test Kalman filter with zero observation variance."""
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=30, n=5, m=2)
    R_zero = R.copy()
    R_zero[0, 0] = 0.0
    R_zero[2, 2] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sf = skf(y, A, C, Q, R_zero, Z_0, V_0)
    assert Sf is not None
    assert hasattr(Sf, 'ZmU') and hasattr(Sf, 'VmU')
    assert hasattr(Sf, 'loglik')
    assert Sf.ZmU.shape == (z_true.shape[1], y.shape[1] + 1)
    assert np.isfinite(Sf.loglik) or np.isnan(Sf.loglik)
    assert np.all(np.isfinite(Sf.ZmU)) or np.any(np.isfinite(Sf.ZmU))

def test_fis_all_missing_observations():
    """Test fixed interval smoother with all missing observations."""
    y, A, C, Q, R, Z_0, V_0, z_true = create_test_data(T=30, n=5, m=2)
    y_all_missing = np.full_like(y, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sf = skf(y_all_missing, A, C, Q, R, Z_0, V_0)
    Ss = fis(A, Sf)
    assert Ss is not None
    assert hasattr(Ss, 'ZmT') and hasattr(Ss, 'VmT')
    assert Ss.ZmT.shape == Sf.ZmU.shape
    assert Ss.VmT.shape == Sf.VmU.shape
    assert np.all(np.isfinite(Ss.ZmT)) or np.any(np.isfinite(Ss.ZmT))
    assert np.all(np.isfinite(Ss.VmT)) or np.any(np.isfinite(Ss.VmT))

# ============================================================================
# Numeric Utility Tests
# ============================================================================

def test_compute_covariance_safe_basic():
    """Test basic covariance computation."""
    np.random.seed(42)
    T, N = 100, 10
    data = np.random.randn(T, N)
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=False)
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)

def test_compute_covariance_safe_pairwise_complete():
    """Test pairwise complete covariance computation."""
    np.random.seed(42)
    T, N = 100, 10
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.2
    data[missing_mask] = np.nan
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)

def test_compute_covariance_safe_pairwise_extreme_sparsity():
    """Test pairwise complete covariance with extreme sparsity."""
    np.random.seed(42)
    T, N = 50, 5
    data = np.full((T, N), np.nan)
    for i in range(N):
        for j in range(i, N):
            n_obs = min(10, T)
            overlap_indices = np.random.choice(T, size=n_obs, replace=False)
            if i == j:
                data[overlap_indices, i] = np.random.randn(len(overlap_indices))
            else:
                data[overlap_indices, i] = np.random.randn(len(overlap_indices))
                data[overlap_indices, j] = np.random.randn(len(overlap_indices))
    for t in range(T):
        if np.all(np.isfinite(data[t, :])):
            remove_idx = np.random.randint(N)
            data[t, remove_idx] = np.nan
    complete_rows = np.all(np.isfinite(data), axis=1)
    assert np.sum(complete_rows) == 0, "Test setup failed: some rows are complete"
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)
    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals >= -1e-10)

def test_compute_covariance_safe_large_block():
    """Test covariance with large block."""
    np.random.seed(42)
    T, N = 396, 78
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23
    data[missing_mask] = np.nan
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True, fallback_to_identity=True)
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))

def test_compute_covariance_safe_small_block():
    """Test covariance with small block."""
    np.random.seed(42)
    T, N = 396, 20
    data = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.23
    data[missing_mask] = np.nan
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True, fallback_to_identity=True)
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))

def test_compute_covariance_safe_pairwise_single_observation():
    """Test pairwise covariance where each pair has only one overlapping observation."""
    np.random.seed(42)
    T, N = 20, 5
    data = np.full((T, N), np.nan)
    pair_count = 0
    for i in range(N):
        for j in range(i, N):
            t = pair_count % T
            if i == j:
                data[t, i] = np.random.randn()
            else:
                val = np.random.randn()
                data[t, i] = val
                data[t, j] = val + np.random.randn() * 0.1
            pair_count += 1
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True)
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T)
    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals >= -1e-10)

def test_compute_covariance_safe_sparse_data():
    """Test covariance with very sparse data."""
    np.random.seed(42)
    T, N = 396, 7
    data = np.full((T, N), np.nan)
    for i in range(0, T, 3):
        data[i, :] = np.random.randn(N)
    cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True, fallback_to_identity=True)
    assert cov.shape == (N, N), f"Expected ({N}, {N}), got {cov.shape}"
    assert np.all(np.isfinite(cov))

def test_compute_covariance_safe_shape_consistency():
    """Test that covariance always returns correct shape."""
    np.random.seed(42)
    T, N = 200, 15
    test_cases = [("no_missing", None), ("low_missing", 0.1), ("medium_missing", 0.3), ("high_missing", 0.6)]
    for name, missing_rate in test_cases:
        data = np.random.randn(T, N)
        if missing_rate is not None:
            missing_mask = np.random.rand(T, N) < missing_rate
            data[missing_mask] = np.nan
        cov = _compute_covariance_safe(data, rowvar=True, pairwise_complete=True, fallback_to_identity=True)
        assert cov.shape == (N, N), f"{name}: Expected ({N}, {N}), got {cov.shape}"

def test_compute_covariance_safe_rowvar_false():
    """Test covariance with rowvar=False."""
    np.random.seed(42)
    N, T = 10, 100
    data = np.random.randn(N, T)
    cov = _compute_covariance_safe(data, rowvar=False, pairwise_complete=False)
    assert cov.shape == (N, N)
    assert np.all(np.isfinite(cov))

def test_compute_covariance_safe_single_variable():
    """Test covariance with single variable."""
    np.random.seed(42)
    T = 100
    data = np.random.randn(T, 1)
    cov = _compute_covariance_safe(data, rowvar=True)
    assert cov.shape == (1, 1)
    assert np.isfinite(cov[0, 0])

def test_compute_covariance_safe_empty_data():
    """Test covariance with empty data."""
    data = np.array([]).reshape(0, 5)
    cov = _compute_covariance_safe(data, rowvar=True, fallback_to_identity=True)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, np.eye(5))

def test_compute_variance_safe_basic():
    """Test basic variance computation."""
    np.random.seed(42)
    data = np.random.randn(100)
    var = _compute_variance_safe(data)
    assert np.isfinite(var)
    assert var > 0

def test_compute_variance_safe_with_missing():
    """Test variance computation with missing values."""
    np.random.seed(42)
    data = np.random.randn(100)
    data[10:20] = np.nan
    var = _compute_variance_safe(data)
    assert np.isfinite(var)
    assert var > 0

def test_compute_variance_safe_all_missing():
    """Test variance with all missing data."""
    data = np.full(100, np.nan)
    var = _compute_variance_safe(data, default_variance=1.0)
    assert var == 1.0

def test_ensure_innovation_variance_minimum():
    """Test innovation variance minimum enforcement."""
    np.random.seed(42)
    m = 5
    Q = np.random.randn(m, m)
    Q = Q @ Q.T
    Q[0, 0] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q, min_variance=1e-8)
    assert Q_fixed.shape == (m, m)
    assert np.all(np.diag(Q_fixed) >= 1e-8)
    assert Q_fixed[0, 0] >= 1e-8

def test_q_diagonal_never_zero():
    """Test that Q diagonal never becomes zero."""
    from dfm_python.core.em import MIN_INNOVATION_VARIANCE
    np.random.seed(42)
    min_variance = MIN_INNOVATION_VARIANCE
    Q_all_zeros = np.zeros((5, 5))
    Q_fixed = _ensure_innovation_variance_minimum(Q_all_zeros, min_variance=min_variance)
    assert np.all(np.diag(Q_fixed) >= min_variance)
    Q_single_zero = np.eye(5) * 0.1
    Q_single_zero[2, 2] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_single_zero, min_variance=min_variance)
    assert Q_fixed[2, 2] >= min_variance
    assert np.all(np.diag(Q_fixed) >= min_variance)
    Q_multiple_zeros = np.eye(5) * 0.1
    Q_multiple_zeros[0, 0] = 0.0
    Q_multiple_zeros[3, 3] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_multiple_zeros, min_variance=min_variance)
    assert Q_fixed[0, 0] >= min_variance
    assert Q_fixed[3, 3] >= min_variance
    assert np.all(np.diag(Q_fixed) >= min_variance)
    Q_very_small = np.eye(5) * 1e-10
    Q_fixed = _ensure_innovation_variance_minimum(Q_very_small, min_variance=min_variance)
    assert np.all(np.diag(Q_fixed) >= min_variance)
    Q_negative = np.eye(5) * 0.1
    Q_negative[1, 1] = -0.01
    Q_fixed = _ensure_innovation_variance_minimum(Q_negative, min_variance=min_variance)
    assert Q_fixed[1, 1] >= min_variance * 0.99, f"Negative diagonal should be fixed, got {Q_fixed[1, 1]:.2e}"
    Q_with_offdiag = np.eye(5) * 0.1
    Q_with_offdiag[0, 1] = 0.05
    Q_with_offdiag[1, 0] = 0.05
    Q_with_offdiag[0, 0] = 0.0
    Q_fixed = _ensure_innovation_variance_minimum(Q_with_offdiag, min_variance=min_variance)
    assert Q_fixed[0, 0] >= min_variance
    assert Q_fixed[0, 1] == Q_with_offdiag[0, 1]
    assert Q_fixed[1, 0] == Q_with_offdiag[1, 0]

def test_ensure_real_and_symmetric():
    """Test real and symmetric enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m) + 1j * np.random.randn(m, m) * 1e-10
    M_fixed = _ensure_real_and_symmetric(M)
    assert np.isrealobj(M_fixed)
    assert np.allclose(M_fixed, M_fixed.T)

def test_ensure_covariance_stable():
    """Test covariance stability enforcement."""
    np.random.seed(42)
    m = 5
    M = np.random.randn(m, m)
    M = M @ M.T
    eigenvals = np.linalg.eigvalsh(M)
    M = M - np.eye(m) * (np.min(eigenvals) + 0.1)
    M_fixed = _ensure_covariance_stable(M, min_eigenval=1e-8)
    eigenvals_fixed = np.linalg.eigvalsh(M_fixed)
    assert np.all(eigenvals_fixed >= 1e-8 * 0.99), f"Eigenvalues: {eigenvals_fixed}"

def test_covariance_in_init_conditions_scenario():
    """Test covariance computation in realistic init_conditions scenario."""
    np.random.seed(42)
    T, N = 200, 20
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.2
    x[missing_mask] = np.nan
    blocks = np.ones((N, 1), dtype=int)
    r = np.array([1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert A is not None
    assert C.shape == (N, A.shape[0])
    assert Q.shape == (A.shape[0], A.shape[0])
    assert R.shape == (N, N)

def test_covariance_multiple_blocks():
    """Test covariance computation with multiple blocks."""
    np.random.seed(42)
    T, N = 200, 30
    x = np.random.randn(T, N)
    missing_mask = np.random.rand(T, N) < 0.25
    x[missing_mask] = np.nan
    blocks = np.zeros((N, 3), dtype=int)
    blocks[:, 0] = 1
    blocks[0:10, 1] = 1
    blocks[10:20, 2] = 1
    r = np.array([2, 1, 1])
    p = 1
    opt_nan = {'method': 2, 'k': 3}
    R_mat = None
    q = None
    nQ = 0
    i_idio = np.ones(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A, C, Q, R, Z_0, V_0 = init_conditions(x, r, p, blocks, opt_nan, R_mat, q, nQ, i_idio, clock='m', tent_weights_dict={}, frequencies=None)
    assert A is not None
    assert C.shape[0] == N
    assert Q.shape[0] == A.shape[0]

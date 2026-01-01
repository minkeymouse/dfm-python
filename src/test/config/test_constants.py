"""Tests for config.constants module."""

import pytest
from dfm_python.config.constants import (
    DEFAULT_CLOCK_FREQUENCY,
    PERIODS_PER_YEAR,
    DEFAULT_BLOCK_NAME,
    DEFAULT_PROGRESS_LOG_INTERVAL,
    DEFAULT_VARIANCE_FALLBACK,
    DEFAULT_TENT_KERNEL_SIZE,
    MIN_STD,
    MIN_FACTOR_VARIANCE,
    DEFAULT_CLEAN_NAN,
    CHOLESKY_LOG_DET_FACTOR,
    SYMMETRY_AVERAGE_FACTOR,
    DEFAULT_KDFM_AR_ORDER,
    DEFAULT_KDFM_MA_ORDER,
    DEFAULT_MAX_EPOCHS,
    MIN_TIME_STEPS,
    MIN_VARIABLES,
    MIN_DDFM_TIME_STEPS,
    MIN_DDFM_DATASET_SIZE_WARNING,
    MIN_ITER_FOR_DELTA_COMPUTATION,
    MIN_EPS_SHAPE_FOR_IDIO,
)


class TestConstants:
    """Test suite for configuration constants."""
    
    def test_default_clock_frequency(self):
        """Test default clock frequency constant."""
        assert DEFAULT_CLOCK_FREQUENCY is not None
        assert isinstance(DEFAULT_CLOCK_FREQUENCY, str)
    
    def test_periods_per_year(self):
        """Test periods per year mapping."""
        assert isinstance(PERIODS_PER_YEAR, dict)
        # Check for common frequencies (lowercase keys: 'm', 'q', 'a')
        assert 'q' in PERIODS_PER_YEAR
        assert 'a' in PERIODS_PER_YEAR
        assert 'm' in PERIODS_PER_YEAR
    
    def test_default_block_name(self):
        """Test default block name constant."""
        assert DEFAULT_BLOCK_NAME is not None
        assert isinstance(DEFAULT_BLOCK_NAME, str)
    
    def test_default_progress_log_interval(self):
        """Test DEFAULT_PROGRESS_LOG_INTERVAL constant."""
        assert DEFAULT_PROGRESS_LOG_INTERVAL is not None
        assert isinstance(DEFAULT_PROGRESS_LOG_INTERVAL, int)
        assert DEFAULT_PROGRESS_LOG_INTERVAL == 5
    
    def test_default_variance_fallback(self):
        """Test DEFAULT_VARIANCE_FALLBACK constant."""
        assert DEFAULT_VARIANCE_FALLBACK is not None
        assert isinstance(DEFAULT_VARIANCE_FALLBACK, float)
        assert DEFAULT_VARIANCE_FALLBACK == 1.0
    
    def test_default_tent_kernel_size(self):
        """Test DEFAULT_TENT_KERNEL_SIZE constant."""
        assert DEFAULT_TENT_KERNEL_SIZE is not None
        assert isinstance(DEFAULT_TENT_KERNEL_SIZE, int)
        assert DEFAULT_TENT_KERNEL_SIZE == 5
    
    def test_min_std(self):
        """Test MIN_STD constant."""
        assert MIN_STD is not None
        assert isinstance(MIN_STD, float)
        assert MIN_STD == 1e-8
    
    def test_min_factor_variance(self):
        """Test MIN_FACTOR_VARIANCE constant."""
        assert MIN_FACTOR_VARIANCE is not None
        assert isinstance(MIN_FACTOR_VARIANCE, float)
        assert MIN_FACTOR_VARIANCE == 1e-10
    
    def test_default_clean_nan(self):
        """Test DEFAULT_CLEAN_NAN constant."""
        assert DEFAULT_CLEAN_NAN is not None
        assert isinstance(DEFAULT_CLEAN_NAN, float)
        assert DEFAULT_CLEAN_NAN == 0.0
    
    def test_cholesky_log_det_factor(self):
        """Test CHOLESKY_LOG_DET_FACTOR constant."""
        assert CHOLESKY_LOG_DET_FACTOR is not None
        assert isinstance(CHOLESKY_LOG_DET_FACTOR, float)
        assert CHOLESKY_LOG_DET_FACTOR == 2.0
    
    def test_symmetry_average_factor(self):
        """Test SYMMETRY_AVERAGE_FACTOR constant."""
        assert SYMMETRY_AVERAGE_FACTOR is not None
        assert isinstance(SYMMETRY_AVERAGE_FACTOR, float)
        assert SYMMETRY_AVERAGE_FACTOR == 0.5
    
    def test_default_kdfm_ar_order(self):
        """Test DEFAULT_KDFM_AR_ORDER constant."""
        assert DEFAULT_KDFM_AR_ORDER is not None
        assert isinstance(DEFAULT_KDFM_AR_ORDER, int)
        assert DEFAULT_KDFM_AR_ORDER == 1
    
    def test_default_kdfm_ma_order(self):
        """Test DEFAULT_KDFM_MA_ORDER constant."""
        assert DEFAULT_KDFM_MA_ORDER is not None
        assert isinstance(DEFAULT_KDFM_MA_ORDER, int)
        assert DEFAULT_KDFM_MA_ORDER == 0
    
    def test_default_max_epochs(self):
        """Test DEFAULT_MAX_EPOCHS constant."""
        assert DEFAULT_MAX_EPOCHS is not None
        assert isinstance(DEFAULT_MAX_EPOCHS, int)
        assert DEFAULT_MAX_EPOCHS == 100
    
    def test_min_time_steps(self):
        """Test MIN_TIME_STEPS constant."""
        assert MIN_TIME_STEPS is not None
        assert isinstance(MIN_TIME_STEPS, int)
        assert MIN_TIME_STEPS == 1
    
    def test_min_variables(self):
        """Test MIN_VARIABLES constant."""
        assert MIN_VARIABLES is not None
        assert isinstance(MIN_VARIABLES, int)
        assert MIN_VARIABLES == 1
    
    def test_min_ddfm_time_steps(self):
        """Test MIN_DDFM_TIME_STEPS constant."""
        assert MIN_DDFM_TIME_STEPS is not None
        assert isinstance(MIN_DDFM_TIME_STEPS, int)
        assert MIN_DDFM_TIME_STEPS == 2
        # Verify it's different from general MIN_TIME_STEPS
        assert MIN_DDFM_TIME_STEPS > MIN_TIME_STEPS
    
    def test_min_ddfm_dataset_size_warning(self):
        """Test MIN_DDFM_DATASET_SIZE_WARNING constant."""
        assert MIN_DDFM_DATASET_SIZE_WARNING is not None
        assert isinstance(MIN_DDFM_DATASET_SIZE_WARNING, int)
        assert MIN_DDFM_DATASET_SIZE_WARNING == 10
        # Verify it's greater than MIN_DDFM_TIME_STEPS (warning threshold should be higher than minimum requirement)
        assert MIN_DDFM_DATASET_SIZE_WARNING > MIN_DDFM_TIME_STEPS
    
    def test_min_iter_for_delta_computation(self):
        """Test MIN_ITER_FOR_DELTA_COMPUTATION constant."""
        assert MIN_ITER_FOR_DELTA_COMPUTATION is not None
        assert isinstance(MIN_ITER_FOR_DELTA_COMPUTATION, int)
        assert MIN_ITER_FOR_DELTA_COMPUTATION == 1
        # Verify it's used in trainer/ddfm.py for iter_count comparison
        # The constant is used in conditions: iter_count > MIN_ITER_FOR_DELTA_COMPUTATION
    
    def test_min_eps_shape_for_idio(self):
        """Test MIN_EPS_SHAPE_FOR_IDIO constant."""
        assert MIN_EPS_SHAPE_FOR_IDIO is not None
        assert isinstance(MIN_EPS_SHAPE_FOR_IDIO, int)
        assert MIN_EPS_SHAPE_FOR_IDIO == 1
        # Verify it's used in trainer/ddfm.py for eps.shape[0] comparison
        # The constant is used in condition: eps.shape[0] > MIN_EPS_SHAPE_FOR_IDIO


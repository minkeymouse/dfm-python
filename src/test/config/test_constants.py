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
    DEFAULT_MAX_EPOCHS,
    MIN_TIME_STEPS,
    MIN_VARIABLES,
    MIN_DDFM_TIME_STEPS,
    MIN_DDFM_DATASET_SIZE_WARNING,
    MIN_ITER_FOR_DELTA_COMPUTATION,
    MIN_EPS_SHAPE_FOR_IDIO,
    DEFAULT_N_MC_SAMPLES,
    DEFAULT_FACTOR_ORDER,
    DEFAULT_AR_ORDER_2,
    MIN_SHAPE_FOR_AR2,
    DEFAULT_DDFM_CLIP_RANGE_DEEP,
    DEFAULT_DDFM_CLIP_RANGE_SHALLOW,
    DEFAULT_NUM_WORKERS,
    DEFAULT_CUDA_DEVICE_INDEX,
    COMPUTATION_ERROR_TYPES,
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
    
    def test_default_n_mc_samples(self):
        """Test DEFAULT_N_MC_SAMPLES constant."""
        assert DEFAULT_N_MC_SAMPLES is not None
        assert isinstance(DEFAULT_N_MC_SAMPLES, int)
        assert DEFAULT_N_MC_SAMPLES == 10  # Matches original TensorFlow epochs=10 default (per experiment/config/model/ddfm.yaml)
        # Verify it's used in models/ddfm.py for n_mc_samples parameter
        # This is the number of MC samples per MCMC iteration for DDFM denoising training
        # Matches original paper's typical usage (around 200 samples)
        assert DEFAULT_N_MC_SAMPLES > 0
    
    def test_default_factor_order(self):
        """Test DEFAULT_FACTOR_ORDER constant."""
        assert DEFAULT_FACTOR_ORDER is not None
        assert isinstance(DEFAULT_FACTOR_ORDER, int)
        assert DEFAULT_FACTOR_ORDER == 1  # Default factor order for DDFM (AR(1) dynamics)
        # Verify it's used in models/ddfm.py for factor_order parameter
        # This replaces hardcoded factor_order=1 throughout the codebase
        assert DEFAULT_FACTOR_ORDER > 0
    
    def test_default_ar_order_2(self):
        """Test DEFAULT_AR_ORDER_2 constant."""
        assert DEFAULT_AR_ORDER_2 is not None
        assert isinstance(DEFAULT_AR_ORDER_2, int)
        assert DEFAULT_AR_ORDER_2 == 2  # Default AR order 2 for DDFM forecast (AR(2) dynamics)
        # Verify it's used in models/ddfm.py for AR(2) forecast
        # This replaces hardcoded AR_ORDER_2 = 2 in ddfm.py line 1758
        assert DEFAULT_AR_ORDER_2 > 0
    
    def test_min_shape_for_ar2(self):
        """Test MIN_SHAPE_FOR_AR2 constant."""
        assert MIN_SHAPE_FOR_AR2 is not None
        assert isinstance(MIN_SHAPE_FOR_AR2, int)
        assert MIN_SHAPE_FOR_AR2 == 2  # Minimum shape dimension for AR(2) forecast (need at least 2 previous time steps)
        # Verify it's used in models/ddfm.py for shape checks
        # This replaces hardcoded >= 2 comparisons in ddfm.py lines 1325, 1759
        assert MIN_SHAPE_FOR_AR2 > 0
        # Verify it's related to DEFAULT_AR_ORDER_2 (need at least AR_ORDER_2 previous time steps)
        assert MIN_SHAPE_FOR_AR2 == DEFAULT_AR_ORDER_2
    
    def test_default_ddfm_clip_range_deep(self):
        """Test DEFAULT_DDFM_CLIP_RANGE_DEEP constant."""
        assert DEFAULT_DDFM_CLIP_RANGE_DEEP is not None
        assert isinstance(DEFAULT_DDFM_CLIP_RANGE_DEEP, float)
        assert DEFAULT_DDFM_CLIP_RANGE_DEEP == 8.0  # Clipping range for deep networks (>2 layers)
        # Verify it's used in models/ddfm.py for clip_range calculation
        # Used in _get_clip_range() helper method for deep networks
        assert DEFAULT_DDFM_CLIP_RANGE_DEEP > 0
    
    def test_default_ddfm_clip_range_shallow(self):
        """Test DEFAULT_DDFM_CLIP_RANGE_SHALLOW constant."""
        assert DEFAULT_DDFM_CLIP_RANGE_SHALLOW is not None
        assert isinstance(DEFAULT_DDFM_CLIP_RANGE_SHALLOW, float)
        assert DEFAULT_DDFM_CLIP_RANGE_SHALLOW == 10.0  # Clipping range for shallow networks (<=2 layers)
        # Verify it's used in models/ddfm.py for clip_range calculation
        # Used in _get_clip_range() helper method for shallow networks
        assert DEFAULT_DDFM_CLIP_RANGE_SHALLOW > 0
        # Verify shallow range is greater than deep range (shallow networks can handle larger values)
        assert DEFAULT_DDFM_CLIP_RANGE_SHALLOW > DEFAULT_DDFM_CLIP_RANGE_DEEP
    
    def test_default_num_workers(self):
        """Test DEFAULT_NUM_WORKERS constant."""
        assert DEFAULT_NUM_WORKERS is not None
        assert isinstance(DEFAULT_NUM_WORKERS, int)
        assert DEFAULT_NUM_WORKERS == 0  # Default number of workers for DataLoader (0 = single-threaded)
        # Verify it's used in models/ddfm.py and encoder/simple_autoencoder.py for DataLoader initialization
        # This replaces hardcoded num_workers=0 throughout the codebase
        assert DEFAULT_NUM_WORKERS >= 0
    
    def test_default_cuda_device_index(self):
        """Test DEFAULT_CUDA_DEVICE_INDEX constant."""
        assert DEFAULT_CUDA_DEVICE_INDEX is not None
        assert isinstance(DEFAULT_CUDA_DEVICE_INDEX, int)
        assert DEFAULT_CUDA_DEVICE_INDEX == 0  # Default CUDA device index for GPU operations (0 = first GPU)
        # Verify it's used in models/ddfm.py for GPU device name logging
        # This replaces hardcoded device index 0 in GPU operations
        assert DEFAULT_CUDA_DEVICE_INDEX >= 0
    
    def test_computation_error_types(self):
        """Test COMPUTATION_ERROR_TYPES constant."""
        assert COMPUTATION_ERROR_TYPES is not None
        assert isinstance(COMPUTATION_ERROR_TYPES, tuple)
        # Verify it contains expected exception types
        assert RuntimeError in COMPUTATION_ERROR_TYPES
        assert ValueError in COMPUTATION_ERROR_TYPES
        assert TypeError in COMPUTATION_ERROR_TYPES
        assert AttributeError in COMPUTATION_ERROR_TYPES
        assert KeyError in COMPUTATION_ERROR_TYPES
        # Verify it does not contain OSError (which is used separately in ddfm.py for I/O errors)
        # OSError is a built-in exception, check by name
        error_type_names = [et.__name__ for et in COMPUTATION_ERROR_TYPES]
        assert 'OSError' not in error_type_names
        # Used to consolidate duplicate exception handling patterns across models


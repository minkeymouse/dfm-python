"""Common constants used across the dfm-python package.

This module centralizes numeric constants, thresholds, and default values
to reduce hardcoded values and improve maintainability.
"""

from typing import Dict, Tuple
from datetime import datetime
import numpy as np

# ============================================================================
# Convergence and Tolerance Constants
# ============================================================================

# Default convergence thresholds
DEFAULT_CONVERGENCE_THRESHOLD = 1e-4  # EM algorithm convergence (general)
DEFAULT_EM_THRESHOLD = 1e-5  # EM algorithm convergence threshold (DFM-specific)
DEFAULT_TOLERANCE = 0.0005  # MCMC/denoising convergence
DEFAULT_MIN_DELTA = 1e-6  # Minimum change for improvement

# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Minimum eigenvalues and variances
MIN_EIGENVALUE = 1e-8  # Minimum eigenvalue for positive definite matrices
MIN_DIAGONAL_VARIANCE = 1e-8  # Minimum variance for diagonal elements
MIN_OBSERVATION_NOISE = 1e-4  # Minimum observation noise for measurement error (used in EM updates)
MIN_FACTOR_VARIANCE = 1e-10  # Minimum variance for factors
MIN_STD = 1e-8  # Minimum standard deviation
DEFAULT_VARIANCE_FALLBACK = 1.0  # Default variance fallback value for numerical stability

# Maximum eigenvalues
MAX_EIGENVALUE = 1e6  # Maximum eigenvalue cap

# Eigenvalue stability thresholds
DEFAULT_EIGENVALUE_MAX_MAGNITUDE = 1.0  # Default maximum eigenvalue magnitude for stability checks
DEFAULT_EIGENVALUE_WARN_THRESHOLD = 0.99  # Default warning threshold for near-unstable eigenvalues

# Matrix cleaning defaults
DEFAULT_CLEAN_NAN = 0.0  # Default value for NaN replacement in clean_matrix
DEFAULT_CLEAN_INF = MAX_EIGENVALUE  # Default value for Inf replacement in clean_matrix (uses MAX_EIGENVALUE)

# Identity matrix defaults
DEFAULT_IDENTITY_SCALE = 1.0  # Default scale for create_scaled_identity(n, 1.0)
DEFAULT_ZERO_VALUE = 0.0  # Default zero value for explicit zero assignments
DEFAULT_XAVIER_GAIN = 1.0  # Default gain for Xavier initialization
DEFAULT_OUTPUT_LAYER_GAIN = 0.1  # Default gain for output layer Xavier initialization (smaller for stability)

# Companion matrix initialization defaults
DEFAULT_INIT_SCALE = 0.01  # Default initialization scale for companion matrix B and C matrices
DEFAULT_KERNEL_INIT_SCALE = 0.1  # Default initialization scale for companion matrix coefficient matrices

# Regularization scales
DEFAULT_REGULARIZATION_SCALE = 1e-5  # Default ridge regularization scale
DEFAULT_REGULARIZATION = 1e-6  # Default regularization value

# Clipping thresholds
DEFAULT_CLIP_THRESHOLD = 10.0  # Default clipping threshold (in standard deviations)
DEFAULT_DATA_CLIP_THRESHOLD = 100.0  # Default data clipping threshold

# ============================================================================
# Training Defaults
# ============================================================================

# Iteration and epoch defaults
DEFAULT_MAX_ITER = 100  # Default maximum EM iterations (general)
DEFAULT_EM_MAX_ITER = 5000  # Default maximum EM iterations (DFM-specific)
DEFAULT_MAX_EPOCHS = 100  # Default maximum training epochs
DEFAULT_MAX_MCMC_ITER = 200  # Default maximum MCMC iterations

# Batch size defaults
DEFAULT_BATCH_SIZE = 32  # Default batch size for neural networks
DEFAULT_DDFM_BATCH_SIZE = 100  # Default batch size for DDFM

# Learning rate defaults
DEFAULT_LEARNING_RATE = 0.001  # Default learning rate
DEFAULT_DDFM_LEARNING_RATE = 0.005  # Default learning rate for DDFM

# Gradient clipping
DEFAULT_GRAD_CLIP_VAL = 1.0  # Default gradient clipping value

# Weight decay defaults
DEFAULT_WEIGHT_DECAY = 0.0  # Default weight decay (L2 regularization)

# Learning rate decay
DEFAULT_LR_DECAY_RATE = 0.96  # Default exponential decay rate for learning rate

# Loss function defaults
DEFAULT_HUBER_DELTA = 1.0  # Default delta parameter for Huber loss
HUBER_QUADRATIC_COEFF = 0.5  # Quadratic coefficient for Huber loss (0.5 * a^2 term)

# Matrix computation defaults
SYMMETRY_AVERAGE_FACTOR = 0.5  # Averaging factor for symmetric matrix computation (0.5 * (M + M.T))

# Data clipping defaults
DEFAULT_DDFM_CLIP_RANGE_DEEP = 8.0  # Clipping range for deep networks (>2 layers)
DEFAULT_DDFM_CLIP_RANGE_SHALLOW = 10.0  # Clipping range for shallow networks (<=2 layers)

# Numerical stability for division
DEFAULT_EPSILON = 1e-8  # Default epsilon for division operations to prevent division by zero

# Random seed defaults
DEFAULT_SEED = 3  # Default random seed for reproducibility

# Structural identification defaults
DEFAULT_STRUCTURAL_REG_WEIGHT = 0.1  # Default weight for structural regularization loss
DEFAULT_STRUCTURAL_INIT_SCALE = 0.1  # Default initialization scale for structural matrices
DEFAULT_STRUCTURAL_DIAG_SCALE = 1.0  # Default diagonal scale for structural matrices (FIXED Iteration 7: was 0.1, caused near-singular S)
DEFAULT_CHOLESKY_EPS = 1e-6  # Default epsilon for Cholesky decomposition stability
CHOLESKY_LOG_DET_FACTOR = 2.0  # Factor for log determinant computation from Cholesky decomposition (log det = 2.0 * sum(log(diag(L))))

# ============================================================================
# Network Architecture Defaults
# ============================================================================

# Encoder layer defaults
DEFAULT_ENCODER_LAYERS = [64, 32]  # Default encoder layer sizes

# ============================================================================
# Data Processing Defaults
# ============================================================================

# Missing data handling
DEFAULT_NAN_METHOD = 2  # Default missing data method
DEFAULT_NAN_K = 3  # Default spline interpolation order

# Default date for synthetic time indices
DEFAULT_START_DATE = datetime(2000, 1, 1)

# Default window size for DDFM
DEFAULT_WINDOW_SIZE = 100
DEFAULT_TENT_KERNEL_SIZE = 5  # Default tent kernel size for slower-frequency series aggregation
# Note: DEFAULT_BATCH_SIZE is defined above (line 52) as 32 for general neural networks
# Use DEFAULT_DDFM_BATCH_SIZE (100) for DDFM-specific batch size

# Warning/display limits
MAX_WARNING_ITEMS = 5  # Maximum number of items to show in warning messages
MAX_ERROR_ITEMS = 20  # Maximum number of items to show in error message details

# Minimum observations
DEFAULT_MIN_OBS = 5  # Default minimum observations for estimation
DEFAULT_MIN_OBS_IDIO = 5  # Default minimum observations for idio estimation
DEFAULT_MIN_OBS_VAR = 7  # Minimum observations for VAR estimation (order + 5)

# Dimension validation bounds
MIN_TIME_STEPS = 1  # Minimum number of time steps (T) required for data
MIN_VARIABLES = 1  # Minimum number of variables (N) required for data
MIN_DDFM_TIME_STEPS = 2  # Minimum number of time steps (T) required for DDFM training (DDFM-specific, different from general MIN_TIME_STEPS=1)
MIN_DDFM_DATASET_SIZE_WARNING = 10  # Minimum dataset size (T) below which DDFM denoising training warns about potential instability
MIN_ITER_FOR_DELTA_COMPUTATION = 1  # Minimum iteration count required before computing delta (MSE change) in DDFM denoising training
MIN_EPS_SHAPE_FOR_IDIO = 1  # Minimum eps shape dimension required for idiosyncratic component processing in DDFM

# Idiosyncratic component defaults
DEFAULT_IDIO_STD = 0.1  # Default idiosyncratic standard deviation (when estimation fails)
DEFAULT_IDIO_RHO0 = 0.1  # Default initial AR coefficient for idiosyncratic components
DEFAULT_AR_COEF = 0.5  # Default AR coefficient for initialization (conservative, used in DDFM)
DEFAULT_PROCESS_NOISE = 0.1  # Default process noise for initialization
DEFAULT_TRANSITION_COEF = 0.9  # Default transition coefficient for DFM initialization

# Standardization defaults removed - now using sklearn scalers directly

# VAR stability and clipping
VAR_STABILITY_THRESHOLD = 0.99  # Maximum eigenvalue for VAR stability
AR_CLIP_MIN = -0.99  # Minimum AR coefficient clipping value
AR_CLIP_MAX = 0.99  # Maximum AR coefficient clipping value
MIN_Q_FLOOR = 0.01  # Minimum floor for innovation covariance Q

# EM algorithm specific constants
DEFAULT_SLOWER_FREQ_AR_COEF = 0.1  # AR coefficient for slower-frequency idiosyncratic components
DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR = 19.0  # Variance denominator for slower-frequency series
DEFAULT_EXTREME_FORECAST_THRESHOLD = 50.0  # Threshold for detecting extreme forecasts
DEFAULT_MAX_VARIANCE = 1e4  # Maximum variance cap

# Correlation and validation thresholds
PERFECT_CORR_THRESHOLD = 0.999  # Threshold for detecting perfect correlation between factors
HIGH_CORR_THRESHOLD = 0.9  # Threshold for high correlation warnings
DEFAULT_DAMPING_FACTOR = 0.5  # Default damping factor for parameter updates (used in utils/misc.py)

# Numerical thresholds
MIN_CONDITION_NUMBER = 1e-12  # Minimum value for condition number calculations
MAX_CONDITION_NUMBER = 1e8  # Maximum condition number threshold for regularization (ill-conditioned matrix threshold)

# ============================================================================
# Display and Logging Defaults
# ============================================================================

DEFAULT_DISP = 10  # Default display interval for progress
DEFAULT_LOG_INTERVAL = 10  # Default logging interval divisor (log every num_epochs // DEFAULT_LOG_INTERVAL epochs)
DEFAULT_PROGRESS_LOG_INTERVAL = 5  # Default progress logging interval for EM algorithm

# ============================================================================
# Precision Defaults
# ============================================================================

DEFAULT_DTYPE = np.float32  # Default numpy dtype for arrays

# PyTorch dtype (matches DEFAULT_DTYPE)
try:
    import torch
    DEFAULT_TORCH_DTYPE = torch.float32  # Default PyTorch dtype for tensors
except ImportError:
    DEFAULT_TORCH_DTYPE = None  # PyTorch not available

# ============================================================================
# IRF (Impulse Response Function) Defaults
# ============================================================================

DEFAULT_IRF_HORIZON = 20  # Default horizon for IRF computation

# ============================================================================
# Forecast Defaults
# ============================================================================

DEFAULT_FORECAST_HORIZON = 6  # Default horizon for forecast computation

# ============================================================================
# Error Handling Constants
# ============================================================================

# Common exception types for computation error handling
# Used to consolidate duplicate exception handling patterns across models
COMPUTATION_ERROR_TYPES = (RuntimeError, ValueError, TypeError, AttributeError, KeyError)

# ============================================================================
# KDFM Defaults
# ============================================================================

DEFAULT_KDFM_AR_ORDER = 1  # Default AR order (VAR lag order p) for KDFM
DEFAULT_KDFM_MA_ORDER = 0  # Default MA order (MA lag order q) for KDFM (0 = pure VAR)

# ============================================================================
# Tutorial Defaults
# ============================================================================

TUTORIAL_MAX_PERIODS = 100  # Default maximum periods for tutorial data (reduced for faster execution)
TUTORIAL_MAX_EPOCHS = 10  # Default maximum epochs for tutorial training (reduced for faster execution)

# ============================================================================
# Matrix Type Constants
# ============================================================================

MATRIX_TYPE_GENERAL = 'general'
MATRIX_TYPE_COVARIANCE = 'covariance'
MATRIX_TYPE_DIAGONAL = 'diagonal'
MATRIX_TYPE_LOADING = 'loading'

# ============================================================================
# Log-Determinant Constants
# ============================================================================

MAX_LOG_DETERMINANT = 700.0  # Maximum log-determinant before overflow (exp(700) is near float64 max)

# ============================================================================
# Default Frequency Constants
# ============================================================================

DEFAULT_CLOCK_FREQUENCY = 'm'  # Default clock frequency (monthly)
DEFAULT_HIERARCHY_VALUE = 3  # Default hierarchy value (monthly = 3)

# Block structure defaults
DEFAULT_BLOCK_NAME = 'Block_0'  # Default block name for DFM blocks

# Periods per year for each frequency
PERIODS_PER_YEAR: Dict[str, int] = {
    'd': 365,   # Daily (approximate)
    'w': 52,    # Weekly (approximate)
    'm': 12,    # Monthly
    'q': 4,     # Quarterly
    'sa': 2,    # Semi-annual
    'a': 1      # Annual
}

# Valid frequency codes
VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes - REMOVED: transformations are handled by preprocessing pipeline, not in core package

# ============================================================================
# Frequency Hierarchy and Tent Kernel Constants
# ============================================================================

# Frequency hierarchy (from highest to lowest frequency)
# Used to determine which frequencies are slower/faster than the clock
FREQUENCY_HIERARCHY: Dict[str, int] = {
    'd': 1,   # Daily (highest frequency)
    'w': 2,   # Weekly
    'm': 3,   # Monthly
    'q': 4,   # Quarterly
    'sa': 5,  # Semi-annual
    'a': 6    # Annual (lowest frequency)
}

# Maximum tent kernel size (number of periods)
# For frequency gaps larger than this, the missing data approach is used instead
MAX_TENT_SIZE: int = 12

# Deterministic tent weights lookup for supported frequency pairs
# Format: (slower_freq, faster_freq) -> tent_weights_array
# These weights define how slower-frequency series aggregate clock-frequency factors
TENT_WEIGHTS_LOOKUP: Dict[Tuple[str, str], np.ndarray] = {
    ('q', 'm'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: quarterly -> monthly
    ('sa', 'm'): np.array([1, 2, 3, 4, 3, 2, 1]),             # 7 periods: semi-annual -> monthly
    ('a', 'm'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: annual -> monthly
    ('m', 'w'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: monthly -> weekly
    ('q', 'w'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: quarterly -> weekly
    ('sa', 'w'): np.array([1, 2, 3, 4, 3, 2, 1]),             # 7 periods: semi-annual -> weekly
    ('a', 'w'): np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),       # 9 periods: annual -> weekly
    ('sa', 'q'): np.array([1, 2, 1]),                         # 3 periods: semi-annual -> quarterly
    ('a', 'q'): np.array([1, 2, 3, 2, 1]),                    # 5 periods: annual -> quarterly
    ('a', 'sa'): np.array([1, 2, 1]),                         # 3 periods: annual -> semi-annual
}

# ============================================================================
# Export all constants
# ============================================================================

__all__ = [
    # Convergence
    'DEFAULT_CONVERGENCE_THRESHOLD',
    'DEFAULT_EM_THRESHOLD',
    'DEFAULT_TOLERANCE',
    'DEFAULT_MIN_DELTA',
    'DEFAULT_EM_MAX_ITER',
    # Numerical stability
    'MIN_EIGENVALUE',
    'MIN_DIAGONAL_VARIANCE',
    'MIN_OBSERVATION_NOISE',
    'MIN_FACTOR_VARIANCE',
    'MIN_STD',
    'MAX_EIGENVALUE',
    'DEFAULT_EIGENVALUE_MAX_MAGNITUDE',
    'DEFAULT_EIGENVALUE_WARN_THRESHOLD',
    'DEFAULT_CLEAN_NAN',
    'DEFAULT_CLEAN_INF',
    'DEFAULT_IDENTITY_SCALE',
    'DEFAULT_ZERO_VALUE',
    'DEFAULT_XAVIER_GAIN',
    'DEFAULT_OUTPUT_LAYER_GAIN',
    'DEFAULT_INIT_SCALE',
    'DEFAULT_KERNEL_INIT_SCALE',
    'DEFAULT_REGULARIZATION_SCALE',
    'DEFAULT_REGULARIZATION',
    'DEFAULT_CLIP_THRESHOLD',
    'DEFAULT_DATA_CLIP_THRESHOLD',
    'MIN_CONDITION_NUMBER',
    'MAX_CONDITION_NUMBER',
    # Training
    'DEFAULT_MAX_ITER',
    'DEFAULT_MAX_EPOCHS',
    'DEFAULT_MAX_MCMC_ITER',
    'DEFAULT_BATCH_SIZE',  # General neural network default (32)
    'DEFAULT_DDFM_BATCH_SIZE',  # DDFM-specific default (100)
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_DDFM_LEARNING_RATE',
    'DEFAULT_GRAD_CLIP_VAL',
    'DEFAULT_WEIGHT_DECAY',
    'DEFAULT_LR_DECAY_RATE',
    'DEFAULT_HUBER_DELTA',
    'HUBER_QUADRATIC_COEFF',
    'DEFAULT_DDFM_CLIP_RANGE_DEEP',
    'DEFAULT_DDFM_CLIP_RANGE_SHALLOW',
    'DEFAULT_EPSILON',
    'DEFAULT_SEED',
    # Structural identification
    'DEFAULT_STRUCTURAL_REG_WEIGHT',
    'DEFAULT_STRUCTURAL_INIT_SCALE',
    'DEFAULT_STRUCTURAL_DIAG_SCALE',
    'DEFAULT_CHOLESKY_EPS',
    # Architecture
    'DEFAULT_ENCODER_LAYERS',
    # Data processing
    'DEFAULT_NAN_METHOD',
    'DEFAULT_NAN_K',
    'DEFAULT_START_DATE',
    'DEFAULT_WINDOW_SIZE',
    'MAX_WARNING_ITEMS',
    'DEFAULT_MIN_OBS',
    'DEFAULT_MIN_OBS_IDIO',
    'DEFAULT_MIN_OBS_VAR',
    # Dimension validation
    'MIN_TIME_STEPS',
    'MIN_VARIABLES',
    'MIN_DDFM_TIME_STEPS',
    'MIN_DDFM_DATASET_SIZE_WARNING',
    'MIN_ITER_FOR_DELTA_COMPUTATION',
    'MIN_EPS_SHAPE_FOR_IDIO',
    'DEFAULT_IDIO_STD',
    'DEFAULT_IDIO_RHO0',
    'DEFAULT_AR_COEF',
    'DEFAULT_PROCESS_NOISE',
    'VAR_STABILITY_THRESHOLD',
    'AR_CLIP_MIN',
    'AR_CLIP_MAX',
    'MIN_Q_FLOOR',
    'MIN_CONDITION_NUMBER',
    # EM algorithm
    'DEFAULT_SLOWER_FREQ_AR_COEF',
    'DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR',
    'DEFAULT_EXTREME_FORECAST_THRESHOLD',
    'DEFAULT_MAX_VARIANCE',
    # Correlation and validation
    'PERFECT_CORR_THRESHOLD',
    'HIGH_CORR_THRESHOLD',
    'DEFAULT_DAMPING_FACTOR',
    # Display
    'DEFAULT_DISP',
    'DEFAULT_LOG_INTERVAL',
    'DEFAULT_PROGRESS_LOG_INTERVAL',
    'DEFAULT_TENT_KERNEL_SIZE',
    'MAX_ERROR_ITEMS',
    # IRF
    'DEFAULT_IRF_HORIZON',
    # Forecast
    'DEFAULT_FORECAST_HORIZON',
    # Error handling
    'COMPUTATION_ERROR_TYPES',
    # KDFM defaults
    'DEFAULT_KDFM_AR_ORDER',
    'DEFAULT_KDFM_MA_ORDER',
    # Tutorial defaults
    'TUTORIAL_MAX_PERIODS',
    'TUTORIAL_MAX_EPOCHS',
    # Matrix types
    'MATRIX_TYPE_GENERAL',
    'MATRIX_TYPE_COVARIANCE',
    'MATRIX_TYPE_DIAGONAL',
    'MATRIX_TYPE_LOADING',
    # Log-determinant
    'MAX_LOG_DETERMINANT',
    'CHOLESKY_LOG_DET_FACTOR',
    # Matrix computation
    'SYMMETRY_AVERAGE_FACTOR',
    # Default frequency
    'DEFAULT_CLOCK_FREQUENCY',
    'DEFAULT_HIERARCHY_VALUE',
    # Block structure
    'DEFAULT_BLOCK_NAME',
    # Periods per year
    'PERIODS_PER_YEAR',
    # Frequency validation
    'VALID_FREQUENCIES',
    # Frequency hierarchy and tent kernels
    'FREQUENCY_HIERARCHY',
    'MAX_TENT_SIZE',
    'TENT_WEIGHTS_LOOKUP',
]


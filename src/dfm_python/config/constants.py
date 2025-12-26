"""Common constants used across the dfm-python package.

This module centralizes numeric constants, thresholds, and default values
to reduce hardcoded values and improve maintainability.
"""

# ============================================================================
# Convergence and Tolerance Constants
# ============================================================================

# Default convergence thresholds
DEFAULT_CONVERGENCE_THRESHOLD = 1e-4  # EM algorithm convergence
DEFAULT_TOLERANCE = 0.0005  # MCMC/denoising convergence
DEFAULT_MIN_DELTA = 1e-6  # Minimum change for improvement

# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Minimum eigenvalues and variances
MIN_EIGENVALUE = 1e-8  # Minimum eigenvalue for positive definite matrices
MIN_DIAGONAL_VARIANCE = 1e-8  # Minimum variance for diagonal elements
MIN_FACTOR_VARIANCE = 1e-10  # Minimum variance for factors
MIN_STD = 1e-8  # Minimum standard deviation

# Maximum eigenvalues
MAX_EIGENVALUE = 1e6  # Maximum eigenvalue cap

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
DEFAULT_MAX_ITER = 100  # Default maximum EM iterations
DEFAULT_MAX_EPOCHS = 100  # Default maximum training epochs
DEFAULT_MAX_MCMC_ITER = 200  # Default maximum MCMC iterations
DEFAULT_EPOCHS_PER_ITER = 10  # Default epochs per MCMC iteration

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

# Data clipping defaults
DEFAULT_DDFM_CLIP_RANGE_DEEP = 8.0  # Clipping range for deep networks (>2 layers)
DEFAULT_DDFM_CLIP_RANGE_SHALLOW = 10.0  # Clipping range for shallow networks (<=2 layers)

# Numerical stability for division
DEFAULT_EPSILON = 1e-8  # Default epsilon for division operations to prevent division by zero

# Structural identification defaults
DEFAULT_STRUCTURAL_REG_WEIGHT = 0.1  # Default weight for structural regularization loss
DEFAULT_STRUCTURAL_INIT_SCALE = 0.1  # Default initialization scale for structural matrices
DEFAULT_STRUCTURAL_DIAG_SCALE = 0.1  # Default diagonal scale for structural matrices
DEFAULT_CHOLESKY_EPS = 1e-6  # Default epsilon for Cholesky decomposition stability

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

# Minimum observations
DEFAULT_MIN_OBS = 5  # Default minimum observations for estimation
DEFAULT_MIN_OBS_IDIO = 5  # Default minimum observations for idio estimation
DEFAULT_MIN_OBS_VAR = 7  # Minimum observations for VAR estimation (order + 5)

# ============================================================================
# Display and Logging Defaults
# ============================================================================

DEFAULT_DISP = 10  # Default display interval for progress

# ============================================================================
# Precision Defaults
# ============================================================================

DEFAULT_PRECISION = 32  # Default training precision

# ============================================================================
# IRF (Impulse Response Function) Defaults
# ============================================================================

DEFAULT_IRF_HORIZON = 20  # Default horizon for IRF computation

# ============================================================================
# Export all constants
# ============================================================================

__all__ = [
    # Convergence
    'DEFAULT_CONVERGENCE_THRESHOLD',
    'DEFAULT_TOLERANCE',
    'DEFAULT_MIN_DELTA',
    # Numerical stability
    'MIN_EIGENVALUE',
    'MIN_DIAGONAL_VARIANCE',
    'MIN_FACTOR_VARIANCE',
    'MIN_STD',
    'MAX_EIGENVALUE',
    'DEFAULT_REGULARIZATION_SCALE',
    'DEFAULT_REGULARIZATION',
    'DEFAULT_CLIP_THRESHOLD',
    'DEFAULT_DATA_CLIP_THRESHOLD',
    # Training
    'DEFAULT_MAX_ITER',
    'DEFAULT_MAX_EPOCHS',
    'DEFAULT_MAX_MCMC_ITER',
    'DEFAULT_EPOCHS_PER_ITER',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DDFM_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_DDFM_LEARNING_RATE',
    'DEFAULT_GRAD_CLIP_VAL',
    'DEFAULT_WEIGHT_DECAY',
    'DEFAULT_LR_DECAY_RATE',
    'DEFAULT_HUBER_DELTA',
    'DEFAULT_DDFM_CLIP_RANGE_DEEP',
    'DEFAULT_DDFM_CLIP_RANGE_SHALLOW',
    'DEFAULT_EPSILON',
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
    'DEFAULT_MIN_OBS',
    'DEFAULT_MIN_OBS_IDIO',
    'DEFAULT_MIN_OBS_VAR',
    # Display
    'DEFAULT_DISP',
    # Precision
    'DEFAULT_PRECISION',
    # IRF
    'DEFAULT_IRF_HORIZON',
]


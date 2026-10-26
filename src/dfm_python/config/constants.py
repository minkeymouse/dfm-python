"""Numeric constants, thresholds and default values shared across the package."""

from typing import Dict
from datetime import datetime
import numpy as np

# Convergence
DEFAULT_CONVERGENCE_THRESHOLD = 1e-4  # EM (general)
DEFAULT_EM_THRESHOLD = 1e-5  # EM (DFM)
DEFAULT_TOLERANCE = 0.0005  # MCMC/denoising
DEFAULT_MIN_DELTA = 1e-6  # minimum change counted as an improvement

# Numerical stability
MIN_EIGENVALUE = 1e-6  # floor for positive definite matrices
MIN_DIAGONAL_VARIANCE = 1e-8
MIN_OBSERVATION_NOISE = 1e-4  # floor on measurement error in EM updates
DEFAULT_DDFM_OBSERVATION_NOISE = 1e-15  # DDFM state space (matches the TensorFlow reference)
MIN_FACTOR_VARIANCE = 1e-10
MIN_STD = 1e-8
MIN_STD_FOR_SCALE_CHECK = 1e-10  # more lenient than MIN_STD, for scale ratios
DEFAULT_VARIANCE_FALLBACK = 1.0
MAX_EIGENVALUE = 1e3  # cap; with MIN_EIGENVALUE the condition number stays at 1e9

# Time budgets of the EM steps (hard limits on pathological runs)
MAX_ESTEP_SECONDS = 600.0
MAX_MSTEP_SECONDS = 600.0
MAX_EM_ITERATION_SECONDS = 1200.0
MAX_FIRST_ITERATION_SECONDS = 600.0

# Numerical health thresholds (fail fast on pathological regimes); the
# initialization (first iteration) limits are more lenient.
MAX_CONDITION_NUMBER_SMOOTHER = 1e8
MAX_CONDITION_NUMBER_INIT = 1e10
MAX_STABILIZATION_AMOUNT = 1e-3  # diagonal loading before error
MAX_STABILIZATION_AMOUNT_INIT = 5e-3
MAX_STD_UNSCALED = 100.0  # above this the data look unpreprocessed
MIN_STD_SCALED = 1e-4  # below this a scaled series is degenerate
MAX_FALLBACKS_PER_ITERATION = 0  # 0 = fail on any fallback

# Eigenvalue stability
DEFAULT_EIGENVALUE_MAX_MAGNITUDE = 1.0
DEFAULT_EIGENVALUE_WARN_THRESHOLD = 0.99

# clean_matrix replacements
DEFAULT_CLEAN_NAN = 0.0
DEFAULT_CLEAN_INF = MAX_EIGENVALUE

DEFAULT_IDENTITY_SCALE = 1.0  # create_scaled_identity(n, 1.0)
DEFAULT_ZERO_VALUE = 0.0
# Xavier gains: GlorotNormal (gain 1) for every layer, including the output
# layer, as in the TensorFlow DDFM; a smaller output gain made the encoder
# weights too small and the factors near zero.
DEFAULT_XAVIER_GAIN = 1.0
DEFAULT_OUTPUT_LAYER_GAIN = 1.0

# Companion matrix initialization
DEFAULT_INIT_SCALE = 0.01  # B and C matrices
DEFAULT_KERNEL_INIT_SCALE = 0.1  # coefficient matrices

# Regularization
DEFAULT_REGULARIZATION_SCALE = 1e-5  # ridge
DEFAULT_REGULARIZATION = 1e-6

# Clipping (in standard deviations)
DEFAULT_CLIP_THRESHOLD = 10.0
DEFAULT_DATA_CLIP_THRESHOLD = 100.0

# DDFM scale alignment checks (prediction std / data std, and standardized data)
DEFAULT_SCALE_RATIO_MAX = 10.0
DEFAULT_SCALE_RATIO_MIN = 0.1
DEFAULT_STANDARDIZATION_MEAN_THRESHOLD = 0.1  # |mean| of standardized data
DEFAULT_STANDARDIZATION_STD_MIN = 0.1
DEFAULT_STANDARDIZATION_STD_MAX = 10.0
DEFAULT_STANDARDIZED_TARGET_STD = 1.0

# DDFM variance collapse diagnostics
DEFAULT_VARIANCE_COLLAPSE_THRESHOLD = 0.1  # prediction std below this is a collapse
DEFAULT_FACTOR_COLLAPSE_THRESHOLD = 0.01  # |factor mean| below this is a collapse
DEFAULT_BATCHNORM_SUPPRESSION_THRESHOLD = 0.01  # BatchNorm running_var below this suppresses the signal
DEFAULT_TIMESTEP_COLLAPSE_THRESHOLD = 0.1  # per-time-step std
DEFAULT_TIMESTEP_COLLAPSE_RATIO_THRESHOLD = 0.5  # share of collapsed time steps that counts as widespread
DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MIN = 0.1  # for warning messages
DEFAULT_EXPECTED_FACTOR_MAGNITUDE_MAX = 1.0

# DDFM reproduction targets (diagnostics only; exchange_rate dataset, TensorFlow reference)
DEFAULT_TARGET_PREDICTION_STD = 1.0
DEFAULT_VARIANCE_COLLAPSE_STD = 0.03  # observed collapse std vs target 1.0
DEFAULT_TARGET_CONVERGENCE_ITERATIONS = 10
DEFAULT_TARGET_DDFM_LOSS = 0.56
DEFAULT_DDFM_LOSS_MULTIPLIER = 1.8  # observed loss / target loss

# Logging format precisions (digits after the point)
DEFAULT_SCALE_LOG_PRECISION = 6
DEFAULT_TIME_LOG_PRECISION = 2
DEFAULT_LOSS_LOG_PRECISION = 6

SECONDS_PER_MINUTE = 60

# Iterations and epochs
DEFAULT_MAX_ITER = 100  # EM (general)
DEFAULT_EM_MAX_ITER = 5000  # EM (DFM)
DEFAULT_MAX_EPOCHS = 100
DEFAULT_MAX_MCMC_ITER = 200
DEFAULT_N_MC_SAMPLES = 10  # MC samples per MCMC iteration (TensorFlow DDFM epochs=10)
DEFAULT_FACTOR_ORDER = 1  # DDFM factors follow an AR(1)
DEFAULT_AR_ORDER_2 = 2  # DDFM forecast AR(2)

# Batch sizes
DEFAULT_BATCH_SIZE = 32
DEFAULT_DDFM_WINDOW_SIZE = 100  # time-step batch size
DEFAULT_DDFM_BATCH_SIZE = 100  # TensorFlow DDFM batch_size

DEFAULT_MIN_TARGET_INTERPOLATE_RATIO = 0.3  # DDFM skips interpolation above this missing ratio

# DDFM sequential MC training: one epoch per MC sample
DEFAULT_MCMC_EPOCHS = 1

# DDFM placeholder dataset (replaced after on_train_start)
DEFAULT_PLACEHOLDER_MC_SAMPLES = 1
DEFAULT_PLACEHOLDER_DATA_SHAPE_T = 100
DEFAULT_PLACEHOLDER_DATA_SHAPE_N = 10
DEFAULT_PLACEHOLDER_SEED = 42
DEFAULT_NUM_WORKERS = 0  # DataLoader, single-threaded
DEFAULT_CUDA_DEVICE_INDEX = 0

# Learning rates
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DDFM_LEARNING_RATE = 0.005

# Autoencoder training (TensorFlow pattern: one silent epoch per autoencoder.fit() call)
DEFAULT_AUTOENCODER_FIT_EPOCHS = 1
DEFAULT_AUTOENCODER_FIT_VERBOSE = 0

DEFAULT_GRAD_CLIP_VAL = 1.0
DEFAULT_WEIGHT_DECAY = 0.0  # L2
DEFAULT_LR_DECAY_RATE = 0.96  # exponential

VALID_OPTIMIZERS = {'Adam', 'AdamW', 'SGD'}

# Adam
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPS = 1e-7  # TensorFlow default (PyTorch's is 1e-8), for DDFM compatibility

# Batch normalization (TensorFlow/Keras defaults)
DEFAULT_BATCH_NORM_MOMENTUM = 0.99
DEFAULT_BATCH_NORM_EPS = 1e-3

DEFAULT_RANDOM_SEED_MAX = 2**31  # 32-bit signed integer range

# Huber loss
DEFAULT_HUBER_DELTA = 1.0
HUBER_QUADRATIC_COEFF = 0.5  # the 0.5 * a^2 term

SYMMETRY_AVERAGE_FACTOR = 0.5  # 0.5 * (M + M.T)

# DDFM data clipping range by depth
DEFAULT_DDFM_CLIP_RANGE_DEEP = 8.0  # more than 2 layers
DEFAULT_DDFM_CLIP_RANGE_SHALLOW = 10.0

DEFAULT_EPSILON = 1e-8  # division guard

DEFAULT_SEED = 3

# Structural identification
DEFAULT_STRUCTURAL_REG_WEIGHT = 0.1
DEFAULT_STRUCTURAL_INIT_SCALE = 0.1
DEFAULT_STRUCTURAL_DIAG_SCALE = 1.0  # 0.1 made S near-singular
DEFAULT_CHOLESKY_EPS = 1e-6
CHOLESKY_LOG_DET_FACTOR = 2.0  # log det = 2 * sum(log(diag(L)))

# DDFM network architecture
DEFAULT_ENCODER_LAYERS = [16, 4]
DEFAULT_NUM_FACTORS = 3
DEFAULT_ACTIVATION = 'relu'
DEFAULT_DECODER = 'linear'
DEFAULT_USE_BATCH_NORM = True

# Missing data handling
DEFAULT_NAN_METHOD = 2
DEFAULT_NAN_K = 3  # spline interpolation order

DEFAULT_START_DATE = datetime(2000, 1, 1)  # synthetic time indices

DEFAULT_WINDOW_SIZE = 100  # DDFM
DEFAULT_TENT_KERNEL_SIZE = 5  # slower-frequency aggregation
DEFAULT_TENT_KERNEL_REGULARIZATION_MULTIPLIER = 100.0  # tent kernel series are ill-conditioned

# Items shown in warning and error messages
MAX_WARNING_ITEMS = 5
MAX_ERROR_ITEMS = 20

# Minimum observations
DEFAULT_MIN_OBS = 5
DEFAULT_MIN_OBS_IDIO = 5
DEFAULT_MIN_OBS_VAR = 7  # order + 5
DEFAULT_MIN_OBS_PRETRAIN = 50  # DDFM pre-training without interpolation
DEFAULT_MULT_EPOCH_PRETRAIN = 1  # DDFM pre-training epoch multiplier
DEFAULT_PRETRAIN_EPOCHS = 200  # DDFM pre-training (the paper's value)

# Dimension bounds
MIN_TIME_STEPS = 1
MIN_VARIABLES = 1
MIN_DDFM_TIME_STEPS = 2
MIN_DDFM_DATASET_SIZE_WARNING = 10  # below this DDFM warns about instability
MIN_ITER_FOR_DELTA_COMPUTATION = 1  # DDFM computes the MSE delta after this many iterations
DEFAULT_MIN_ITER_FOR_CONVERGENCE_CHECK = 2  # DDFM checks convergence from iteration 3 on
MIN_EPS_SHAPE_FOR_IDIO = 1  # DDFM idiosyncratic processing needs eps.shape[0] above this
MIN_SHAPE_FOR_AR2 = 2  # an AR(2) forecast needs two previous steps

# Idiosyncratic components and initialization
DEFAULT_IDIO_STD = 0.1  # when estimation fails
DEFAULT_IDIO_RHO0 = 0.1  # initial AR coefficient
DEFAULT_AR_COEF = 0.5  # DDFM
DEFAULT_PROCESS_NOISE = 0.1
DEFAULT_TRANSITION_COEF = 0.9  # DFM

# VAR stability and clipping
VAR_STABILITY_THRESHOLD = 0.99  # maximum eigenvalue
AR_CLIP_MIN = -0.99
AR_CLIP_MAX = 0.99
MIN_Q_FLOOR = 0.01  # innovation covariance

# EM
DEFAULT_SLOWER_FREQ_AR_COEF = 0.1  # slower-frequency idiosyncratic components
DEFAULT_SLOWER_FREQ_VARIANCE_DENOMINATOR = 19.0
DEFAULT_EXTREME_FORECAST_THRESHOLD = 50.0
DEFAULT_MAX_VARIANCE = 1e4

# Correlation and validation
PERFECT_CORR_THRESHOLD = 0.999  # between factors
HIGH_CORR_THRESHOLD = 0.9
DEFAULT_DAMPING_FACTOR = 0.5  # parameter updates (utils/misc.py)

# Condition numbers
MIN_CONDITION_NUMBER = 1e-12
MAX_CONDITION_NUMBER = 1e8  # ill-conditioned above this; regularize

# Display and logging
DEFAULT_DISP = 10  # progress display interval
DEFAULT_LOG_INTERVAL = 10  # log every num_epochs // DEFAULT_LOG_INTERVAL epochs
DEFAULT_PROGRESS_LOG_INTERVAL = 5  # EM

# Precision
DEFAULT_DTYPE = np.float64
# DEFAULT_TORCH_DTYPE (torch.float32) is resolved by __getattr__ below: torch is optional.

DEFAULT_IRF_HORIZON = 20
DEFAULT_FORECAST_HORIZON = 6

# Exception types the models catch around computations
COMPUTATION_ERROR_TYPES = (RuntimeError, ValueError, TypeError, AttributeError, KeyError)

# iVDFM dimensions
DEFAULT_IVDFM_LATENT_DIM = 3
DEFAULT_IVDFM_AUX_DIM = 1

# iVDFM regimes (K=1 is the baseline iVDFM without regime structure)
DEFAULT_IVDFM_NUM_REGIMES = 1
DEFAULT_IVDFM_REGIME_TEMPERATURE = 1.0  # softmax temperature of pi; below 1 sharpens (local commitment)
DEFAULT_IVDFM_MIXING = False  # global post-factor mixing f -> M f (identity init) before decoding

# iVDFM networks
DEFAULT_IVDFM_ENCODER_HIDDEN_DIM = 200
DEFAULT_IVDFM_ENCODER_N_HIDDEN_LAYERS = 2
DEFAULT_IVDFM_DECODER_TYPE = "mlp"  # 'linear', 'residual' or 'mlp'
DEFAULT_IVDFM_DECODER_HIDDEN_DIM = 200
DEFAULT_IVDFM_DECODER_N_HIDDEN_LAYERS = 2
DEFAULT_IVDFM_PRIOR_HIDDEN_DIM = 100
DEFAULT_IVDFM_PRIOR_N_HIDDEN_LAYERS = 1

# iVDFM dynamics and distributions
DEFAULT_IVDFM_FACTOR_ORDER = 1
DEFAULT_IVDFM_INNOVATION_DIST = 'laplace'
DEFAULT_IVDFM_DECODER_VAR = 0.01
DEFAULT_IVDFM_ACTIVATION = 'lrelu'
DEFAULT_IVDFM_SLOPE = 0.1  # leaky ReLU

# iVDFM training
DEFAULT_IVDFM_BATCH_SIZE = 32
DEFAULT_IVDFM_MAX_EPOCHS = 100
DEFAULT_IVDFM_AUX_VARIABLE_TYPE = 'time'
DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY = 0.0
DEFAULT_IVDFM_OPTIMIZER_MOMENTUM = 0.9  # SGD

# iVDFM schedulers
DEFAULT_IVDFM_SCHEDULER_TYPE = 'step'  # 'step', 'plateau', 'cosine', 'exponential' or None
DEFAULT_IVDFM_SCHEDULER_STEP_SIZE = None  # StepLR; None = max_epochs // 3
DEFAULT_IVDFM_SCHEDULER_GAMMA = 0.5  # StepLR/ExponentialLR
DEFAULT_IVDFM_SCHEDULER_PATIENCE = 10  # ReduceLROnPlateau
DEFAULT_IVDFM_SCHEDULER_FACTOR = 0.1
DEFAULT_IVDFM_SCHEDULER_MIN_LR = 0.0
DEFAULT_IVDFM_EARLY_STOP_PATIENCE = 20  # epochs without improvement

# Tutorials (reduced for speed)
TUTORIAL_MAX_PERIODS = 100
TUTORIAL_MAX_EPOCHS = 10

# Matrix types
MATRIX_TYPE_GENERAL = 'general'
MATRIX_TYPE_COVARIANCE = 'covariance'
MATRIX_TYPE_DIAGONAL = 'diagonal'
MATRIX_TYPE_LOADING = 'loading'

MAX_LOG_DETERMINANT = 700.0  # exp(700) is near the float64 maximum

# Frequencies
DEFAULT_CLOCK_FREQUENCY = 'm'
DEFAULT_HIERARCHY_VALUE = 3  # FREQUENCY_HIERARCHY of the default clock

DEFAULT_BLOCK_NAME = 'Block_0'

PERIODS_PER_YEAR: Dict[str, int] = {
    'd': 365,
    'w': 52,
    'm': 12,
    'q': 4,
    'sa': 2,
    'a': 1
}

VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Highest to lowest frequency; decides which frequencies are slower than the clock.
FREQUENCY_HIERARCHY: Dict[str, int] = {
    'd': 1,
    'w': 2,
    'm': 3,
    'q': 4,
    'sa': 5,
    'a': 6
}

# Largest tent kernel (periods); wider frequency gaps use the missing-data approach.
MAX_TENT_SIZE: int = 12

# AFM (Attention Factor Model)
DEFAULT_AFM_NUM_FACTORS = 5          # K latent factors
DEFAULT_AFM_FACTOR_MODEL = 'attention'  # 'attention' | 'pca'
DEFAULT_AFM_TRADING = 'longconv'     # 'longconv' | 'ou'
DEFAULT_AFM_EMBED_DIM = 32           # characteristic embedding dimension d
DEFAULT_AFM_HIST_LEN = 20            # residual-history length s for the filter
DEFAULT_AFM_N_KERNELS = 32           # number of long convolutions
DEFAULT_AFM_RIDGE = 1e-2             # ridge penalty for closed-form loadings
DEFAULT_AFM_SQUASH_LAMBDA = 1e-3     # LongConv Squash kernel regularizer strength
DEFAULT_AFM_PCA_WINDOW = 252         # trailing window for the PCA benchmark
DEFAULT_AFM_REESTIM = 21             # PCA re-estimation stride
DEFAULT_AFM_LAMBDA_VAR = 0.1         # explained-variance regularizer weight
DEFAULT_AFM_TURNOVER_COST = 5e-4     # 5bps per unit turnover (paper)
DEFAULT_AFM_SHORT_COST = 1e-4        # 1bp shorting cost (paper)
DEFAULT_AFM_MAX_EPOCHS = 50          # training epochs

__all__ = [
    # AFM
    'DEFAULT_AFM_NUM_FACTORS', 'DEFAULT_AFM_FACTOR_MODEL', 'DEFAULT_AFM_TRADING',
    'DEFAULT_AFM_EMBED_DIM', 'DEFAULT_AFM_HIST_LEN', 'DEFAULT_AFM_N_KERNELS',
    'DEFAULT_AFM_RIDGE', 'DEFAULT_AFM_SQUASH_LAMBDA', 'DEFAULT_AFM_PCA_WINDOW',
    'DEFAULT_AFM_REESTIM', 'DEFAULT_AFM_LAMBDA_VAR', 'DEFAULT_AFM_TURNOVER_COST',
    'DEFAULT_AFM_SHORT_COST', 'DEFAULT_AFM_MAX_EPOCHS',
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
    'MIN_STD_FOR_SCALE_CHECK',
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
    'DEFAULT_SCALE_RATIO_MAX',
    'DEFAULT_SCALE_RATIO_MIN',
    'DEFAULT_STANDARDIZATION_MEAN_THRESHOLD',
    'DEFAULT_STANDARDIZATION_STD_MIN',
    'DEFAULT_STANDARDIZATION_STD_MAX',
    'DEFAULT_STANDARDIZED_TARGET_STD',
    'DEFAULT_TARGET_PREDICTION_STD',
    'DEFAULT_VARIANCE_COLLAPSE_STD',
    'DEFAULT_TARGET_CONVERGENCE_ITERATIONS',
    'DEFAULT_TARGET_DDFM_LOSS',
    'DEFAULT_DDFM_LOSS_MULTIPLIER',
    'DEFAULT_SCALE_LOG_PRECISION',
    'DEFAULT_TIME_LOG_PRECISION',
    'DEFAULT_LOSS_LOG_PRECISION',
    'SECONDS_PER_MINUTE',
    'MIN_CONDITION_NUMBER',
    'MAX_CONDITION_NUMBER',
    # Training
    'DEFAULT_MAX_ITER',
    'DEFAULT_MAX_EPOCHS',
    'DEFAULT_MAX_MCMC_ITER',
    'DEFAULT_N_MC_SAMPLES',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_DDFM_WINDOW_SIZE',
    'DEFAULT_DDFM_BATCH_SIZE',
    'DEFAULT_PLACEHOLDER_MC_SAMPLES',
    'DEFAULT_PLACEHOLDER_DATA_SHAPE_T',
    'DEFAULT_PLACEHOLDER_DATA_SHAPE_N',
    'DEFAULT_PLACEHOLDER_SEED',
    'DEFAULT_NUM_WORKERS',
    'DEFAULT_CUDA_DEVICE_INDEX',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_DDFM_LEARNING_RATE',
    'DEFAULT_AUTOENCODER_FIT_EPOCHS',
    'DEFAULT_AUTOENCODER_FIT_VERBOSE',
    'DEFAULT_GRAD_CLIP_VAL',
    'DEFAULT_WEIGHT_DECAY',
    'DEFAULT_LR_DECAY_RATE',
    'VALID_OPTIMIZERS',
    'DEFAULT_ADAM_BETA1',
    'DEFAULT_ADAM_BETA2',
    'DEFAULT_ADAM_EPS',
    'DEFAULT_BATCH_NORM_MOMENTUM',
    'DEFAULT_BATCH_NORM_EPS',
    'DEFAULT_RANDOM_SEED_MAX',
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
    'DEFAULT_FACTOR_ORDER',
    'DEFAULT_AR_ORDER_2',
    'DEFAULT_ENCODER_LAYERS',
    'DEFAULT_NUM_FACTORS',
    'DEFAULT_ACTIVATION',
    'DEFAULT_DECODER',
    'DEFAULT_USE_BATCH_NORM',
    # Data processing
    'DEFAULT_NAN_METHOD',
    'DEFAULT_NAN_K',
    'DEFAULT_START_DATE',
    'DEFAULT_WINDOW_SIZE',
    'MAX_WARNING_ITEMS',
    'DEFAULT_MIN_OBS',
    'DEFAULT_MIN_OBS_IDIO',
    'DEFAULT_MIN_OBS_VAR',
    'DEFAULT_MIN_OBS_PRETRAIN',
    'DEFAULT_MULT_EPOCH_PRETRAIN',
    'DEFAULT_PRETRAIN_EPOCHS',
    # Dimension validation
    'MIN_TIME_STEPS',
    'MIN_VARIABLES',
    'MIN_SHAPE_FOR_AR2',
    'MIN_DDFM_TIME_STEPS',
    'MIN_DDFM_DATASET_SIZE_WARNING',
    'MIN_ITER_FOR_DELTA_COMPUTATION',
    'DEFAULT_MIN_ITER_FOR_CONVERGENCE_CHECK',
    'MIN_EPS_SHAPE_FOR_IDIO',
    'DEFAULT_IDIO_STD',
    'DEFAULT_IDIO_RHO0',
    'DEFAULT_AR_COEF',
    'DEFAULT_PROCESS_NOISE',
    'VAR_STABILITY_THRESHOLD',
    'AR_CLIP_MIN',
    'AR_CLIP_MAX',
    'MIN_Q_FLOOR',
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
    'DEFAULT_TENT_KERNEL_REGULARIZATION_MULTIPLIER',
    'MAX_ERROR_ITEMS',
    # IRF
    'DEFAULT_IRF_HORIZON',
    # Forecast
    'DEFAULT_FORECAST_HORIZON',
    # Error handling
    'COMPUTATION_ERROR_TYPES',
    # iVDFM defaults
    'DEFAULT_IVDFM_LATENT_DIM',
    'DEFAULT_IVDFM_AUX_DIM',
    'DEFAULT_IVDFM_ENCODER_HIDDEN_DIM',
    'DEFAULT_IVDFM_ENCODER_N_HIDDEN_LAYERS',
    'DEFAULT_IVDFM_NUM_REGIMES',
    'DEFAULT_IVDFM_DECODER_TYPE',
    'DEFAULT_IVDFM_DECODER_HIDDEN_DIM',
    'DEFAULT_IVDFM_DECODER_N_HIDDEN_LAYERS',
    'DEFAULT_IVDFM_PRIOR_HIDDEN_DIM',
    'DEFAULT_IVDFM_PRIOR_N_HIDDEN_LAYERS',
    'DEFAULT_IVDFM_FACTOR_ORDER',
    'DEFAULT_IVDFM_INNOVATION_DIST',
    'DEFAULT_IVDFM_DECODER_VAR',
    'DEFAULT_IVDFM_ACTIVATION',
    'DEFAULT_IVDFM_SLOPE',
    'DEFAULT_IVDFM_BATCH_SIZE',
    'DEFAULT_IVDFM_MAX_EPOCHS',
    'DEFAULT_IVDFM_AUX_VARIABLE_TYPE',
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
    # Performance and numerical health guardrails
    'MAX_ESTEP_SECONDS',
    'MAX_MSTEP_SECONDS',
    'MAX_EM_ITERATION_SECONDS',
    'MAX_FIRST_ITERATION_SECONDS',
    'MAX_CONDITION_NUMBER_SMOOTHER',
    'MAX_CONDITION_NUMBER_INIT',
    'MAX_STABILIZATION_AMOUNT',
    'MAX_STABILIZATION_AMOUNT_INIT',
    'MAX_STD_UNSCALED',
    'MIN_STD_SCALED',
    'MAX_FALLBACKS_PER_ITERATION',
]


def __getattr__(name: str):
    # The torch models' dtype; resolving it imports torch, which only they need.
    if name == "DEFAULT_TORCH_DTYPE":
        import torch

        return torch.float32
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

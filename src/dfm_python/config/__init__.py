"""Configuration subpackage for DFM.

This subpackage provides:
- Schema (DFMConfig, etc.) in schema.py
- IO (ConfigSource, YamlSource, etc.) in adapter.py

Note: Series are specified via frequency dict mapping column names to frequencies.
"""

from .schema import (
    BaseModelConfig, DFMConfig, DDFMConfig,
    BaseResult, DFMResult, DDFMResult,
    DFMStateSpaceParams, DDFMStateSpaceParams,
)
# DEFAULT_BLOCK_NAME is imported lazily where needed to avoid circular imports
from .constants import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_TOLERANCE,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLOCK_FREQUENCY,
    MIN_EIGENVALUE,
    MIN_STD,
)
from .schema.model import validate_frequency
from .adapter import (
    ConfigSource,
    YamlSource,
    DictSource,
    make_config_source,
    detect_config_type,
)
# Re-export types for convenience
from .types import (
    ArrayLike,
    FloatArray,
    IntArray,
    BoolArray,
    OptionalArray,
    OptionalTensor,
    OptionalArrayLike,
    FactorState,
    ObservationState,
    ForecastResult,
    CoefficientMatrix,
    CovarianceMatrix,
    SeriesID,
    Frequency,
    DatasetName,
    ModelName,
    Batch,
    Loss,
    Optimizer,
    ResultDict,
    ForecastDict,
    MetricsDict,
    CheckpointDict,
    ConfigDict,
    Device,
    Shape,
    Shape2D,
    Shape3D,
    Shape4D,
    ForecastHorizon,
    IRFHorizon,
    LagOrder,
    NumFactors,
    NumVars,
    PathLike,
    ValidationIssue,
    ValidationResult,
    Tensor,
    is_numpy_array,
    is_torch_tensor,
    is_array_like,
    get_array_shape,
    to_numpy,
    to_tensor,
)
# Import lazily to avoid circular dependencies
try:
    from ..numeric.builder import compute_idio_lengths
except ImportError:
    compute_idio_lengths = None

try:
    from ..numeric.tent import get_tent_weights, get_agg_structure, group_by_freq
except ImportError:
    get_tent_weights = None
    get_agg_structure = None
    group_by_freq = None
from .constants import FREQUENCY_HIERARCHY, PERIODS_PER_YEAR

# Simple utility function
def get_periods_per_year(frequency: str) -> int:
    """Get number of periods per year for a given frequency."""
    return PERIODS_PER_YEAR.get(frequency, PERIODS_PER_YEAR.get(DEFAULT_CLOCK_FREQUENCY, 12))

__all__ = [
    # Base classes
    'BaseModelConfig', 'BaseResult',
    # 'DEFAULT_BLOCK_NAME',  # Removed to avoid circular import - import directly from functional.dfm_block
    # Model-specific configs (from schema.py)
    'DFMConfig', 'DDFMConfig',
    # State-space parameters
    'DFMStateSpaceParams', 'DDFMStateSpaceParams',
    # Model-specific results (from results.py)
    'DFMResult', 'DDFMResult',
    # Utilities
    'validate_frequency',
    # IO
    'ConfigSource', 'YamlSource', 'DictSource',
    'make_config_source',
    'detect_config_type',
    # Frequency and aggregation utilities
    'FREQUENCY_HIERARCHY',
    'PERIODS_PER_YEAR',
    'get_periods_per_year',
    'compute_idio_lengths',
    'get_tent_weights',
    'get_agg_structure',
    'group_by_freq',
    # Type definitions (from types.py)
    'ArrayLike',
    'FloatArray',
    'IntArray',
    'BoolArray',
    'OptionalArray',
    'OptionalTensor',
    'OptionalArrayLike',
    'FactorState',
    'ObservationState',
    'ForecastResult',
    'CoefficientMatrix',
    'CovarianceMatrix',
    'SeriesID',
    'Frequency',
    'DatasetName',
    'ModelName',
    'Batch',
    'Loss',
    'Optimizer',
    'ResultDict',
    'ForecastDict',
    'MetricsDict',
    'CheckpointDict',
    'ConfigDict',
    'Device',
    'Shape',
    'Shape2D',
    'Shape3D',
    'Shape4D',
    'ForecastHorizon',
    'IRFHorizon',
    'LagOrder',
    'NumFactors',
    'NumVars',
    'PathLike',
    'ValidationIssue',
    'ValidationResult',
    'Tensor',
    'is_numpy_array',
    'is_torch_tensor',
    'is_array_like',
    'get_array_shape',
    'to_numpy',
    'to_tensor',
]


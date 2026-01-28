"""Configuration schema for DFM models.

This module provides model-specific configuration dataclasses:
- BaseModelConfig: Base class with shared model structure (series, clock, data handling)
- DFMConfig(BaseModelConfig): Linear DFM with EM algorithm parameters and block structure
- DDFMConfig(BaseModelConfig): Deep DFM with neural network training parameters (no blocks)
- KDFMConfig(BaseModelConfig): Kernelized DFM with VARMA parameters

The configuration hierarchy:
- BaseModelConfig: Model structure (series, clock, data handling) - NO blocks
- DFMConfig: Adds blocks structure and EM algorithm parameters (max_iter, threshold, regularization)
- DDFMConfig: Adds neural network parameters (epochs, learning_rate, encoder_layers) - NO blocks
- KDFMConfig: Adds VARMA parameters (ar_order, ma_order, structural_method) - NO blocks

Note: Series are specified via frequency dict mapping column names to frequencies. Result classes are in schema/results.py

Blocks are DFM-specific and defined as Dict[str, Dict[str, Any]] where each block is a dict with:
- num_factors: int (number of factors)
- series: List[str] (list of series names/column names in this block)

For loading configurations from files (YAML) or other sources,
see the config.adapter module which provides source adapters.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    try:
        from sklearn.preprocessing import StandardScaler, RobustScaler
        ScalerType = Union[StandardScaler, RobustScaler, Any]
    except ImportError:
        ScalerType = Any
else:
    ScalerType = Any

# Import ConfigurationError and DataError lazily to avoid circular imports
# They are only used in methods, not at module level
from ..constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DDFM_WINDOW_SIZE,
    DEFAULT_GRAD_CLIP_VAL,
    DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY,
    DEFAULT_IVDFM_OPTIMIZER_MOMENTUM,
    DEFAULT_IVDFM_SCHEDULER_TYPE,
    DEFAULT_IVDFM_SCHEDULER_STEP_SIZE,
    DEFAULT_IVDFM_SCHEDULER_GAMMA,
    DEFAULT_IVDFM_SCHEDULER_PATIENCE,
    DEFAULT_IVDFM_SCHEDULER_FACTOR,
    DEFAULT_IVDFM_SCHEDULER_MIN_LR,
    DEFAULT_REGULARIZATION_SCALE,
    DEFAULT_STRUCTURAL_REG_WEIGHT,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_EM_THRESHOLD,
    DEFAULT_EM_MAX_ITER,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_MCMC_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_DATA_CLIP_THRESHOLD,
    DEFAULT_MIN_OBS_IDIO,
    DEFAULT_DISP,
    DEFAULT_IDIO_RHO0,
    AR_CLIP_MIN,
    AR_CLIP_MAX,
    MIN_EIGENVALUE,
    MAX_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_KDFM_AR_ORDER,
    DEFAULT_KDFM_MA_ORDER,
    FREQUENCY_HIERARCHY,
    DEFAULT_HIERARCHY_VALUE,
    DEFAULT_IVDFM_SEQUENCE_LENGTH,
    DEFAULT_IVDFM_LATENT_DIM,
    DEFAULT_IVDFM_AUX_DIM,
    DEFAULT_IVDFM_ENCODER_HIDDEN_DIM,
    DEFAULT_IVDFM_ENCODER_N_LAYERS,
    DEFAULT_IVDFM_DECODER_HIDDEN_DIM,
    DEFAULT_IVDFM_DECODER_N_LAYERS,
    DEFAULT_IVDFM_PRIOR_HIDDEN_DIM,
    DEFAULT_IVDFM_PRIOR_N_LAYERS,
    DEFAULT_IVDFM_FACTOR_ORDER,
    DEFAULT_IVDFM_INNOVATION_DIST,
    DEFAULT_IVDFM_DECODER_VAR,
    DEFAULT_IVDFM_ACTIVATION,
    DEFAULT_IVDFM_SLOPE,
    DEFAULT_IVDFM_BATCH_SIZE,
    DEFAULT_IVDFM_MAX_EPOCHS,
    DEFAULT_IVDFM_AUX_VARIABLE_TYPE,
)



# ============================================================================
# Base Model Configuration
# ============================================================================

@dataclass
class BaseModelConfig:
    """Base configuration class with shared model structure.
    
    This base class contains the model structure that is common to all
    factor models (DFM, DDFM, KDFM):
    - Series definitions (via frequency dict mapping column names to frequencies)
    - Clock frequency (required, base frequency for latent factors)
    - Data preprocessing (missing data handling)
    
    Series Configuration:
    - Provide `frequency` dict in one of two formats:
      1. Grouped format: {'w': [series1, series2, ...], 'm': [series3, ...]} (recommended for large configs)
      2. Individual format: {'series1': 'w', 'series2': 'm', ...} (backward compatible)
    - If `frequency` is None, all columns will use `clock` frequency
    - If a column is missing from `frequency` dict, it will use `clock` frequency
    - When data is loaded, missing columns in `frequency` dict are automatically added with `clock` frequency
    
    Note: Blocks are DFM-specific and are NOT included in BaseModelConfig.
    DFMConfig adds block structure, while DDFMConfig and KDFMConfig do not use blocks.
    
    Subclasses (DFMConfig, DDFMConfig, KDFMConfig) add model-specific training parameters.
    
    Examples
    --------
    >>> # With grouped frequency mapping (recommended for large configs)
    >>> config = DFMConfig(
    ...     frequency={'q': ['gdp'], 'm': ['unemployment', 'interest_rate']},
    ...     clock='m',
    ...     blocks={...}
    ... )
    >>> 
    >>> # With individual frequency mapping (backward compatible)
    >>> config = DFMConfig(
    ...     frequency={'gdp': 'q', 'unemployment': 'm', 'interest_rate': 'm'},
    ...     clock='m',
    ...     blocks={...}
    ... )
    >>> 
    >>> # Without frequency (all use clock)
    >>> config = DFMConfig(
    ...     frequency=None,  # or omit it
    ...     clock='m',
    ...     blocks={...}
    ... )
    >>> # Series will be built from data columns using clock='m' when data is loaded
    """
    # ========================================================================
    # Model Structure (WHAT - defines the model)
    # ========================================================================
    frequency: Optional[Dict[str, str]] = None  # Optional: Maps column names to frequencies {'column_name': 'frequency'}
    # If None, all series use clock frequency (data is assumed aligned with clock)
    
    # ========================================================================
    # Shared Data Handling Parameters
    # ========================================================================
    clock: str = 'm'  # Required: Base frequency for latent factors (global clock): 'd', 'w', 'm', 'q', 'sa', 'a' (defaults to 'm' for monthly)
    target_scaler: Optional[ScalerType] = None  # Fitted sklearn scaler instance (StandardScaler, RobustScaler, etc.) for target series only. Must be a fitted scaler object (call .fit() on target data first). Pass scaler object directly, not string. Feature series are assumed to be manually preprocessed. If None, target series are assumed to be already in the desired scale.
    scaler: Optional[str] = None  # Scaler type as string: 'standard', 'robust', 'minmax', 'maxabs', 'quantile', or null. Applied to targets only (not context/auxiliary variables). For iVDFM, this is the preferred way to specify scaler (like DDFM pattern).
    
    def __post_init__(self):
        """Validate basic model structure.
        
        This method performs basic validation of the model configuration:
        - Validates clock frequency
        - Validates frequency dict if provided
        
        Raises
        ------
        ValueError
            If any validation check fails, with a descriptive error message
            indicating what needs to be fixed.
        """
        from ...config.adapter import _raise_config_error, _is_dict_like
        
        # Validate global clock (required)
        self.clock = validate_frequency(self.clock)
        
        # Validate frequency dict if provided
        if self.frequency is not None:
            if not _is_dict_like(self.frequency):
                _raise_config_error(
                    f"frequency must be a dict mapping column names to frequencies, got {type(self.frequency)}"
                )
            
            # Empty frequency dict is allowed (will be filled from columns later with clock frequency)
            
            # Validate all frequencies in the dict
            for col_name, freq in self.frequency.items():
                if not isinstance(col_name, str):
                    _raise_config_error(f"frequency dict keys must be strings (column names), got {type(col_name)}")
                validate_frequency(freq)
    
    def get_frequencies(self, columns: Optional[List[str]] = None) -> List[str]:
        """Get frequencies. Auto-creates dict from columns if None, defaults to clock for missing."""
        if columns is not None:
            # Auto-create frequency dict if None
            if self.frequency is None:
                self.frequency = {col: self.clock for col in columns}
            # Return frequencies, defaulting to clock for missing columns
            return [self.frequency.get(col, self.clock) for col in columns]
        
        # No columns provided - return from existing dict
        if self.frequency is None:
            return []
        return list(self.frequency.values())
    
    def get_series_ids(self, columns: Optional[List[str]] = None) -> List[str]:
        """Get series IDs. Auto-creates frequency dict from columns if None."""
        if columns is not None:
            # Auto-create frequency dict if None
            if self.frequency is None:
                self.frequency = {col: self.clock for col in columns}
            return columns
        
        # No columns provided - return from existing dict
        if self.frequency is None:
            return []
        return list(self.frequency.keys())
    
    @classmethod
    def _extract_base(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shared base parameters from config dict."""
        from ...config.adapter import _convert_series_to_frequency_dict
        
        base_params = {
            'clock': data.get('clock', DEFAULT_CLOCK_FREQUENCY),
            'target_scaler': data.get('target_scaler', None),
        }
        
        # Handle frequency dict (new API) or legacy series list/dict
        from ...config.adapter import _extract_frequency_dict
        frequency_dict = _extract_frequency_dict(data, base_params['clock'])
        if frequency_dict is not None:
            base_params['frequency'] = frequency_dict
        
        return base_params
    
    @classmethod
    def _extract_params(cls, data: Dict[str, Any], param_map: Dict[str, Any]) -> Dict[str, Any]:
        """Generic parameter extraction helper.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Source data dictionary
        param_map : Dict[str, Any]
            Mapping of parameter names to default values
            
        Returns
        -------
        Dict[str, Any]
            Extracted parameters with defaults applied
        """
        return {key: data.get(key, default) for key, default in param_map.items()}


# ============================================================================
# Model-Specific Configuration Classes
# ============================================================================
# BaseModelConfig is imported from base.py - no duplicate definition needed


@dataclass
class DFMConfig(BaseModelConfig):
    """Linear DFM configuration - EM algorithm parameters and block structure.
    
    This configuration class extends BaseModelConfig with parameters specific
    to linear Dynamic Factor Models trained using the Expectation-Maximization
    (EM) algorithm. DFM uses block structure to organize factors (global + sector-specific).
    
    The configuration can be built from:
    - Main settings (estimation parameters) from config/default.yaml
    - Series definitions via frequency dict (column names -> frequencies)
    - Block definitions from config/blocks/default.yaml
    """
    # ========================================================================
    # Block Structure (DFM-specific)
    # ========================================================================
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Block configurations: {"block_name": {"num_factors": int, "series": [str]}}
    block_names: List[str] = field(init=False)  # Block names in order (derived from blocks dict)
    factors_per_block: List[int] = field(init=False)  # Number of factors per block (derived from blocks)
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # Internal cache
    
    # ========================================================================
    # EM Algorithm Parameters (HOW - controls the algorithm)
    # ========================================================================
    # Note: ar_lag removed - factors always use AR(1) dynamics (simplified)
    threshold: float = DEFAULT_EM_THRESHOLD  # EM convergence threshold
    max_iter: int = DEFAULT_EM_MAX_ITER  # Maximum EM iterations
    
    # ========================================================================
    # Numerical Stability Parameters (transparent and configurable)
    # ========================================================================
    # AR Coefficient Clipping: If provided (not None), automatically enables clipping and always warns
    ar_clip: Optional[Dict[str, float]] = None  # {"min": float, "max": float} - AR coefficient clipping bounds. If None, no clipping. If provided, clipping enabled and warnings always shown.
    
    # Data Value Clipping: If provided (not None), automatically enables clipping and always warns
    data_clip: Optional[float] = None  # Clip values beyond this many standard deviations. If None, no clipping. If provided, clipping enabled and warnings always shown.
    
    # Regularization: If provided (not None), automatically enables regularization and always warns
    regularization: Optional[Dict[str, float]] = None  # {"scale": float, "min_eigenvalue": float, "max_eigenvalue": float} - Regularization parameters. If None, no regularization. If provided, regularization enabled and warnings always shown.
    
    # Damped Updates: If provided (not None), automatically enables damping and always warns
    damping_factor: Optional[float] = None  # Damping factor (0.8 = 80% new, 20% old). If None, no damping. If provided, damping enabled and warnings always shown.
    
    # Idiosyncratic Component Parameters
    idio_rho0: float = DEFAULT_IDIO_RHO0  # Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var: float = MIN_DIAGONAL_VARIANCE  # Minimum variance for idiosyncratic innovation covariance (defaults to MIN_DIAGONAL_VARIANCE)
    # Note: augment_idio and augment_idio_slow are INTERNALLY auto-detected (not configurable)
    # - If all series use clock frequency: augment_idio=False, augment_idio_slow=False
    # - If mixed frequencies detected: augment_idio=True, augment_idio_slow=True (required for tent kernel)
    # These are properties, not config fields - automatically determined from frequency configuration
    
    # Tent Kernel Weights (for mixed-frequency aggregation)
    tent_weights: Optional[Union[Dict[str, List[float]], Dict[str, np.ndarray]]] = None  # Required for mixed-frequency data: tent weights for frequency pairs. Format: {'freq_pair': [weights]} or {'freq': [weights]}. Example: {'m:w': [1, 2, 1]} or {'m': [1, 2, 1]}. Must be specified in config for all slower-frequency pairs.
    
    def __post_init__(self):
        """Validate blocks structure and derive block properties."""
        super().__post_init__()
        
        from ...config.adapter import _raise_config_error
        from ..constants import FREQUENCY_HIERARCHY, DEFAULT_HIERARCHY_VALUE
        
        if not self.blocks:
            _raise_config_error("DFM configuration must contain at least one block.")
        
        # Derive block_names and factors_per_block
        block_names_list = list(self.blocks.keys())
        object.__setattr__(self, 'block_names', block_names_list)
        object.__setattr__(self, 'factors_per_block', 
                         [self.blocks[name].get('num_factors', 1) for name in self.block_names])
        
        # Validate blocks
        for block_name, block_cfg in self.blocks.items():
            num_factors = block_cfg.get('num_factors', 1)
            series_list = block_cfg.get('series', [])
            
            from ...config.adapter import _raise_config_error
            if num_factors < 1:
                _raise_config_error(f"Block '{block_name}' must have num_factors >= 1, got {num_factors}")
            
            if not isinstance(series_list, list):
                _raise_config_error(f"Block '{block_name}' must have 'series' as a list, got {type(series_list)}")
            
            # Validate series exist in frequency dict if available
            if self.frequency is not None:
                for series_name in series_list:
                    if series_name not in self.frequency:
                        # Auto-add missing series with clock frequency
                        self.frequency[series_name] = self.clock
        
        from ...config.adapter import _raise_config_error
        if any(f < 1 for f in self.factors_per_block):
            _raise_config_error("factors_per_block must contain positive integers >= 1")
        
        # Auto-detect mixed frequencies and set augment_idio/augment_idio_slow
        # If frequency dict exists, check if all frequencies match clock
        if self.frequency is not None and len(self.frequency) > 0:
            frequencies = list(self.frequency.values())
            clock_hierarchy = FREQUENCY_HIERARCHY.get(self.clock, DEFAULT_HIERARCHY_VALUE)
            is_mixed_freq = any(
                FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) != clock_hierarchy
                for freq in frequencies
            )
            # Auto-set augment_idio and augment_idio_slow based on frequency detection
            # Store as internal attributes (not in __init__ signature)
            object.__setattr__(self, '_augment_idio', is_mixed_freq)
            object.__setattr__(self, '_augment_idio_slow', is_mixed_freq)
        else:
            # No frequency info yet - will be auto-detected when data is loaded
            object.__setattr__(self, '_augment_idio', False)
            object.__setattr__(self, '_augment_idio_slow', False)
    
    @property
    def augment_idio(self) -> bool:
        """Auto-detected: True if mixed frequencies detected, False if single frequency."""
        return getattr(self, '_augment_idio', False)
    
    @property
    def augment_idio_slow(self) -> bool:
        """Auto-detected: True if mixed frequencies detected, False if single frequency."""
        return getattr(self, '_augment_idio_slow', False)
    
    def _update_idio_flags_from_frequencies(self, frequencies: List[str]) -> None:
        """Update augment_idio flags based on detected frequencies (called when data is loaded)."""
        from ..constants import FREQUENCY_HIERARCHY, DEFAULT_HIERARCHY_VALUE
        clock_hierarchy = FREQUENCY_HIERARCHY.get(self.clock, DEFAULT_HIERARCHY_VALUE)
        is_mixed_freq = any(
            FREQUENCY_HIERARCHY.get(freq, DEFAULT_HIERARCHY_VALUE) != clock_hierarchy
            for freq in frequencies
        )
        object.__setattr__(self, '_augment_idio', is_mixed_freq)
        object.__setattr__(self, '_augment_idio_slow', is_mixed_freq)
    
    def to_em_config(self) -> 'EMConfig':
        """Create EMConfig from DFMConfig consolidated parameters.
        
        Automatically passes all numerical stability parameters (regularization, ar_clip,
        damping_factor, data_clip) from DFMConfig to EMConfig for use in EM algorithm.
        """
        from ...functional.em import EMConfig
        from ..constants import DEFAULT_REGULARIZATION, VAR_STABILITY_THRESHOLD
        
        # Extract regularization parameters directly from dict
        if self.regularization is not None and isinstance(self.regularization, dict):
            reg_scale = self.regularization.get('scale', DEFAULT_REGULARIZATION_SCALE)
            min_eigenval = self.regularization.get('min_eigenvalue', MIN_EIGENVALUE)
            max_eigenval = self.regularization.get('max_eigenvalue', VAR_STABILITY_THRESHOLD)
        else:
            reg_scale = DEFAULT_REGULARIZATION
            min_eigenval = MIN_EIGENVALUE
            max_eigenval = VAR_STABILITY_THRESHOLD
        
        return EMConfig(
            regularization=reg_scale,
            min_norm=min_eigenval,
            max_eigenval=max_eigenval,
            # Pass numerical stability parameters from DFMConfig
            ar_clip=self.ar_clip,
            damping_factor=self.damping_factor,
            data_clip=self.data_clip,
            # Other parameters use defaults from EMConfig
        )
    
    def get_blocks_array(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """Get blocks as numpy array (N x B) where N is number of series and B is number of blocks.
        
        Returns 1 if series is in block, 0 otherwise.
        """
        if self._cached_blocks is None:
            # Auto-create frequency dict if needed
            if self.frequency is None:
                if columns is None:
                    from ...config.adapter import _raise_config_error
                    _raise_config_error("frequency dict or columns required")
                self.frequency = {col: self.clock for col in columns}
            
            series_ids = list(self.frequency.keys()) if columns is None else columns
            
            # Build blocks array from block series lists (N x B matrix)
            # Default global block behavior:
            # If config omitted blocks.series (empty list) and we have runtime columns,
            # interpret a single empty Block_Global as "all series in this block".
            block_series_sets: Dict[str, set] = {}
            default_global_applied = False
            for name in self.block_names:
                series_list = self.blocks[name].get('series', [])
                if (
                    columns is not None
                    and (series_list is None or len(series_list) == 0)
                    and len(self.block_names) == 1
                    and name == "Block_Global"
                ):
                    block_series_sets[name] = set(series_ids)
                    default_global_applied = True
                else:
                    block_series_sets[name] = set(series_list or [])

            blocks_list = [
                [1 if series_id in block_series_sets[name] else 0 for name in self.block_names]
                for series_id in series_ids
            ]
            
            self._cached_blocks = np.array(blocks_list, dtype=int)

            # Quick sanity on membership counts for the first block (if any)
        return self._cached_blocks
    
    @classmethod
    def _extract_dfm_params(cls, data: Dict[str, Any], base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract DFM-specific parameters from config dict.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Config dictionary
        base_params : Dict[str, Any], optional
            Pre-extracted base parameters. If provided, avoids duplicate extraction.
            If None, extracts base params internally.
        """
        if base_params is None:
            base_params = cls._extract_base(data)
        
        # Extract consolidated parameters (new format only)
        dfm_params = {
            'threshold': data.get('threshold', DEFAULT_EM_THRESHOLD),
            'max_iter': data.get('max_iter', DEFAULT_EM_MAX_ITER),
            'idio_rho0': data.get('idio_rho0', DEFAULT_IDIO_RHO0),
            'idio_min_var': data.get('idio_min_var', MIN_DIAGONAL_VARIANCE),
            'ar_clip': data.get('ar_clip', None),
            'data_clip': data.get('data_clip', None),
            'regularization': data.get('regularization', None),
            'damping_factor': data.get('damping_factor', None),
            'tent_weights': data.get('tent_weights', None),
        }
        
        result = base_params.copy()
        result.update(dfm_params)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig']:
        """Create DFMConfig or DDFMConfig from dictionary.
        
        Expected format: {'frequency': {'column_name': 'frequency'}, 'blocks': {...}, ...}
        
        Also accepts estimation parameters: threshold, max_iter, etc.
        """
        from ...config.adapter import detect_config_type, MODEL_TYPE_DDFM, _normalize_blocks_dict
        from ...utils.errors import ConfigurationError
        
        # Extract base params (handles frequency conversion from series if needed)
        base_params = cls._extract_base(data)
        
        # Determine config type
        config_type = detect_config_type(data)
        
        if config_type == MODEL_TYPE_DDFM:
            return DDFMConfig(**base_params, **DDFMConfig._extract_ddfm(data))
        
        # Handle blocks for DFM
        from ...config.adapter import _raise_config_error, _is_dict_like
        blocks_dict = data.get('blocks', {})
        if not blocks_dict:
            # Default behavior: single global block.
            # This enables minimal configs like:
            #   clock: d
            #   num_factors: 2
            # and lets the model bind series names from the dataset at runtime.
            default_num_factors = int(data.get('num_factors', 1) or 1)
            if default_num_factors < 1:
                _raise_config_error(f"num_factors must be >= 1, got {default_num_factors}")
            blocks_dict = {
                "Block_Global": {
                    "num_factors": default_num_factors,
                    "series": [],  # to be bound from dataset columns
                }
            }
        if not _is_dict_like(blocks_dict):
            _raise_config_error(f"blocks must be a dict, got {type(blocks_dict)}")
        
        blocks_dict_normalized = _normalize_blocks_dict(blocks_dict)
        # Pass base_params to _extract_dfm_params to avoid duplicate extraction
        dfm_params = DFMConfig._extract_dfm_params(data, base_params=base_params)
        return DFMConfig(blocks=blocks_dict_normalized, **dfm_params)


@dataclass
class DDFMConfig(BaseModelConfig):
    """Deep Dynamic Factor Model configuration - neural network training parameters.
    
    This configuration class extends BaseModelConfig with parameters specific
    to Deep Dynamic Factor Models trained using neural networks (autoencoders).
    
    Note: DDFM does NOT use block structure. Use num_factors directly to specify
    the number of factors. Blocks are DFM-specific and not needed for DDFM.
    
    The configuration can be built from:
    - Main settings (training parameters) from config/default.yaml
    - Series definitions via frequency dict (column names -> frequencies)
    """
    # ========================================================================
    # Neural Network Training Hyper Parameters
    # ========================================================================
    encoder_layers: Optional[List[int]] = None  # Hidden layer dimensions for encoder (default: [64, 32])
    num_factors: Optional[int] = None  # Number of factors (inferred from config if None)
    activation: str = 'relu'  # Activation function ('tanh', 'relu', 'sigmoid', default: 'relu' to match original DDFM)
    use_batch_norm: bool = True  # Use batch normalization in encoder (default: True)
    learning_rate: float = 0.001  # Learning rate for Adam optimizer (default: 0.001)
    n_mc_samples: int = 10  # Number of MC samples per MCMC iteration (default: 10, matching original TensorFlow epochs=10 default, per experiment/config/model/ddfm.yaml)
    window_size: int = 100  # Window size (time-step batch size) for training (default: 100 to match original DDFM)
    # Note: factor_order removed - factors always use AR(1) dynamics (simplified)
    use_idiosyncratic: bool = True  # Model idio components with AR(1) dynamics (default: True)
    min_obs_idio: int = 5  # Minimum observations for idio AR(1) estimation (default: 5)
    
    # Additional training parameters
    max_epoch: int = DEFAULT_MAX_MCMC_ITER  # Maximum number of epochs (MCMC iterations). One epoch = one MCMC iteration (MC sampling → training → convergence check)
    tolerance: float = DEFAULT_TOLERANCE  # Convergence tolerance for MCMC iterations
    disp: int = 10  # Display frequency for training progress
    seed: Optional[int] = None  # Random seed for reproducibility
    lags_input: int = 0  # Number of lags of inputs on encoder (default 0, matching original TensorFlow DDFM)
    
    # Interpolation parameters (for handling missing values)
    interpolation_method: str = 'linear'  # Interpolation method: 'linear', 'spline', 'cubic', etc. (default: 'linear' for stability)
    interpolation_limit: Optional[int] = 10  # Maximum consecutive NaNs to interpolate (default: 10, prevents extreme extrapolation)
    interpolation_limit_direction: str = 'both'  # Direction to fill: 'forward', 'backward', or 'both' (default: 'both')
    
    
    # ========================================================================
    # Factory Methods
    # ========================================================================
    
    @classmethod
    def _extract_ddfm(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DDFM-specific parameters from config dict."""
        # Don't extract base params here - they're already in base_params from from_dict
        from ..constants import DEFAULT_N_MC_SAMPLES
        ddfm_params = cls._extract_params(data, {
            'encoder_layers': None,
            'num_factors': None,
            'activation': 'relu',
            'use_batch_norm': True,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'epochs': DEFAULT_N_MC_SAMPLES,  # Backward compatibility: map 'epochs' to n_mc_samples
            'n_mc_samples': DEFAULT_N_MC_SAMPLES,  # Preferred name: number of MC samples per MCMC iteration
            'window_size': DEFAULT_DDFM_WINDOW_SIZE,  # Window size (time-step batch size) for training
            'use_idiosyncratic': True,
            'min_obs_idio': DEFAULT_MIN_OBS_IDIO,
            'max_epoch': DEFAULT_MAX_MCMC_ITER,  # Maximum epochs (MCMC iterations)
            'tolerance': DEFAULT_TOLERANCE,
            'disp': DEFAULT_DISP,
            'seed': None,
            'lags_input': 0,  # Number of lags (default 0, matching original TensorFlow)
            'interpolation_method': 'linear',  # Interpolation method (default: 'linear' for stability)
            'interpolation_limit': 10,  # Maximum consecutive NaNs to interpolate (default: 10)
            'interpolation_limit_direction': 'both',  # Direction to fill (default: 'both')
        })
        # Map 'epochs' from config to 'n_mc_samples' for clarity (backward compatibility)
        # Always remove 'epochs' if present (even if n_mc_samples is also present)
        if 'epochs' in ddfm_params:
            if 'n_mc_samples' not in ddfm_params:
                ddfm_params['n_mc_samples'] = ddfm_params['epochs']
            ddfm_params.pop('epochs')  # Always remove 'epochs' to avoid passing it to constructor
        # Map 'batch_size' from config to 'window_size' for backward compatibility
        if 'batch_size' in ddfm_params:
            if 'window_size' not in ddfm_params:
                ddfm_params['window_size'] = ddfm_params['batch_size']
            ddfm_params.pop('batch_size')  # Always remove 'batch_size' to avoid passing it to constructor
        # Only accept 'max_epoch' parameter (no backward compatibility)
        # Remove any old parameter names if present
        ddfm_params.pop('max_iter', None)
        ddfm_params.pop('max_iterations', None)
        ddfm_params.pop('max_mc_iter', None)
        return ddfm_params
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DDFMConfig':
        """Create DDFMConfig from dictionary (delegates to DFMConfig.from_dict for type detection)."""
        result = DFMConfig.from_dict(data)
        if isinstance(result, DDFMConfig):
            return result
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(
            "Expected DDFMConfig but got DFMConfig",
            details=f"Result type: {type(result).__name__}, expected: DDFMConfig"
        )


@dataclass
class iVDFMConfig(BaseModelConfig):
    """Identifiable Variational Dynamic Factor Model configuration.
    
    Uses 'scaler' (string) pattern like DDFM, not 'target_scaler' (instance).
    Scaler is applied to targets only, not to context/auxiliary variables.
    """
    """Identifiable Variational Dynamic Factor Model configuration.
    
    This configuration class extends BaseModelConfig with parameters specific
    to iVDFM models trained using variational inference with identifiable
    innovation priors.
    
    Note: iVDFM does NOT use block structure. Use num_factors directly to specify
    the number of factors.
    
    The configuration can be built from:
    - Main settings (training parameters) from config files
    - Series definitions via frequency dict (column names -> frequencies)
    """
    # ========================================================================
    # Model Structure
    # ========================================================================
    num_factors: Optional[int] = None  # Number of factors (inferred from config if None)
    sequence_length: int = DEFAULT_IVDFM_SEQUENCE_LENGTH  # Sequence length for training
    context: Optional[Union[List[str], List[int]]] = None  # Column names (DataFrame) or indices (array) for context variables. If None, context_dim is used to generate time-based context.
    context_dim: int = DEFAULT_IVDFM_AUX_DIM  # Dimension of context. Used when context is None (generates time-based context) or when context is multivariate.
    
    # ========================================================================
    # Neural Network Architecture
    # ========================================================================
    encoder_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_ENCODER_HIDDEN_DIM  # Encoder architecture
    encoder_n_layers: int = DEFAULT_IVDFM_ENCODER_N_LAYERS  # Number of encoder layers
    decoder_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_DECODER_HIDDEN_DIM  # Decoder architecture
    decoder_n_layers: int = DEFAULT_IVDFM_DECODER_N_LAYERS  # Number of decoder layers
    prior_hidden_dim: Union[int, List[int]] = DEFAULT_IVDFM_PRIOR_HIDDEN_DIM  # Prior network architecture
    prior_n_layers: int = DEFAULT_IVDFM_PRIOR_N_LAYERS  # Number of prior network layers
    activation: str = DEFAULT_IVDFM_ACTIVATION  # Activation function
    slope: float = DEFAULT_IVDFM_SLOPE  # Leaky ReLU slope
    
    # ========================================================================
    # Dynamics and Distribution Parameters
    # ========================================================================
    factor_order: int = DEFAULT_IVDFM_FACTOR_ORDER  # AR order for factors
    innovation_distribution: str = DEFAULT_IVDFM_INNOVATION_DIST  # Innovation distribution type
    decoder_var: float = DEFAULT_IVDFM_DECODER_VAR  # Decoder variance
    
    # ========================================================================
    # Training Parameters
    # ========================================================================
    learning_rate: float = DEFAULT_LEARNING_RATE  # Learning rate
    optimizer: str = 'Adam'  # Optimizer type
    optimizer_weight_decay: float = DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY  # Weight decay (L2 regularization)
    optimizer_momentum: float = DEFAULT_IVDFM_OPTIMIZER_MOMENTUM  # Momentum for SGD
    batch_size: int = DEFAULT_IVDFM_BATCH_SIZE  # Batch size
    max_epochs: int = DEFAULT_IVDFM_MAX_EPOCHS  # Maximum epochs
    tolerance: float = DEFAULT_TOLERANCE  # Convergence tolerance
    seed: Optional[int] = None  # Random seed
    
    # ========================================================================
    # Scheduler Parameters
    # ========================================================================
    scheduler_type: Optional[str] = DEFAULT_IVDFM_SCHEDULER_TYPE  # Scheduler type: 'step', 'plateau', 'cosine', 'exponential', None
    scheduler_step_size: Optional[int] = DEFAULT_IVDFM_SCHEDULER_STEP_SIZE  # Step size for StepLR (None = auto: max_epochs // 3)
    scheduler_gamma: float = DEFAULT_IVDFM_SCHEDULER_GAMMA  # Gamma for StepLR/ExponentialLR
    scheduler_patience: int = DEFAULT_IVDFM_SCHEDULER_PATIENCE  # Patience for ReduceLROnPlateau
    scheduler_factor: float = DEFAULT_IVDFM_SCHEDULER_FACTOR  # Factor for ReduceLROnPlateau
    scheduler_min_lr: float = DEFAULT_IVDFM_SCHEDULER_MIN_LR  # Min learning rate for ReduceLROnPlateau
    
    def __post_init__(self):
        """Validate iVDFM configuration."""
        super().__post_init__()  # Validate base config
        
        from ...utils.errors import ConfigurationError
        
        # Validate sequence_length
        if self.sequence_length < 1:
            raise ConfigurationError(
                f"sequence_length must be >= 1, got {self.sequence_length}"
            )
        
        # Validate context_dim
        if self.context_dim < 1:
            raise ConfigurationError(
                f"context_dim must be >= 1, got {self.context_dim}"
            )
        
        # Validate num_factors if provided
        if self.num_factors is not None and self.num_factors < 1:
            raise ConfigurationError(
                f"num_factors must be >= 1, got {self.num_factors}"
            )
        
        # Validate factor_order
        if self.factor_order < 1:
            raise ConfigurationError(
                f"factor_order must be >= 1, got {self.factor_order}"
            )
        
        # Validate innovation_distribution
        valid_dists = {'laplace', 'gaussian', 'student_t', 'gamma', 'beta', 'exponential'}
        if self.innovation_distribution not in valid_dists:
            raise ConfigurationError(
                f"innovation_distribution must be one of {valid_dists}, got '{self.innovation_distribution}'"
            )
        
        # Validate activation
        valid_activations = {'relu', 'lrelu', 'tanh', 'sigmoid'}
        if self.activation not in valid_activations:
            raise ConfigurationError(
                f"activation must be one of {valid_activations}, got '{self.activation}'"
            )
        
        # Validate optimizer
        from ..constants import VALID_OPTIMIZERS
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ConfigurationError(
                f"optimizer must be one of {VALID_OPTIMIZERS}, got '{self.optimizer}'"
            )
        
        # Validate scheduler_type
        valid_schedulers = {'step', 'plateau', 'cosine', 'exponential', None}
        if self.scheduler_type not in valid_schedulers:
            raise ConfigurationError(
                f"scheduler_type must be one of {valid_schedulers}, got '{self.scheduler_type}'"
            )
        
        # Validate scheduler parameters
        if self.scheduler_gamma <= 0 or self.scheduler_gamma > 1:
            raise ConfigurationError(
                f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}"
            )
        
        if self.scheduler_patience < 0:
            raise ConfigurationError(
                f"scheduler_patience must be >= 0, got {self.scheduler_patience}"
            )
        
        if self.scheduler_factor <= 0 or self.scheduler_factor > 1:
            raise ConfigurationError(
                f"scheduler_factor must be in (0, 1], got {self.scheduler_factor}"
            )
        
        if self.scheduler_min_lr < 0:
            raise ConfigurationError(
                f"scheduler_min_lr must be >= 0, got {self.scheduler_min_lr}"
            )
    
    @classmethod
    def _extract_ivdfm(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract iVDFM-specific parameters from config dict."""
        from ..constants import (
            DEFAULT_IVDFM_SEQUENCE_LENGTH,
            DEFAULT_IVDFM_LATENT_DIM,
            DEFAULT_IVDFM_AUX_DIM,
            DEFAULT_IVDFM_ENCODER_HIDDEN_DIM,
            DEFAULT_IVDFM_ENCODER_N_LAYERS,
            DEFAULT_IVDFM_DECODER_HIDDEN_DIM,
            DEFAULT_IVDFM_DECODER_N_LAYERS,
            DEFAULT_IVDFM_PRIOR_HIDDEN_DIM,
            DEFAULT_IVDFM_PRIOR_N_LAYERS,
            DEFAULT_IVDFM_FACTOR_ORDER,
            DEFAULT_IVDFM_INNOVATION_DIST,
            DEFAULT_IVDFM_DECODER_VAR,
            DEFAULT_IVDFM_ACTIVATION,
            DEFAULT_IVDFM_SLOPE,
            DEFAULT_IVDFM_BATCH_SIZE,
            DEFAULT_IVDFM_MAX_EPOCHS,
            DEFAULT_IVDFM_AUX_VARIABLE_TYPE,
            DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY,
            DEFAULT_IVDFM_OPTIMIZER_MOMENTUM,
            DEFAULT_IVDFM_SCHEDULER_TYPE,
            DEFAULT_IVDFM_SCHEDULER_STEP_SIZE,
            DEFAULT_IVDFM_SCHEDULER_GAMMA,
            DEFAULT_IVDFM_SCHEDULER_PATIENCE,
            DEFAULT_IVDFM_SCHEDULER_FACTOR,
            DEFAULT_IVDFM_SCHEDULER_MIN_LR,
        )
        
        ivdfm_params = cls._extract_params(data, {
            'num_factors': None,
            'sequence_length': DEFAULT_IVDFM_SEQUENCE_LENGTH,
            'context': None,
            'context_dim': DEFAULT_IVDFM_AUX_DIM,
            'encoder_hidden_dim': DEFAULT_IVDFM_ENCODER_HIDDEN_DIM,
            'encoder_n_layers': DEFAULT_IVDFM_ENCODER_N_LAYERS,
            'decoder_hidden_dim': DEFAULT_IVDFM_DECODER_HIDDEN_DIM,
            'decoder_n_layers': DEFAULT_IVDFM_DECODER_N_LAYERS,
            'prior_hidden_dim': DEFAULT_IVDFM_PRIOR_HIDDEN_DIM,
            'prior_n_layers': DEFAULT_IVDFM_PRIOR_N_LAYERS,
            'activation': DEFAULT_IVDFM_ACTIVATION,
            'slope': DEFAULT_IVDFM_SLOPE,
            'factor_order': DEFAULT_IVDFM_FACTOR_ORDER,
            'innovation_distribution': DEFAULT_IVDFM_INNOVATION_DIST,
            'decoder_var': DEFAULT_IVDFM_DECODER_VAR,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'optimizer': 'Adam',
            'optimizer_weight_decay': DEFAULT_IVDFM_OPTIMIZER_WEIGHT_DECAY,
            'optimizer_momentum': DEFAULT_IVDFM_OPTIMIZER_MOMENTUM,
            'batch_size': DEFAULT_IVDFM_BATCH_SIZE,
            'max_epochs': DEFAULT_IVDFM_MAX_EPOCHS,
            'tolerance': DEFAULT_TOLERANCE,
            'seed': None,
            'scheduler_type': DEFAULT_IVDFM_SCHEDULER_TYPE,
            'scheduler_step_size': DEFAULT_IVDFM_SCHEDULER_STEP_SIZE,
            'scheduler_gamma': DEFAULT_IVDFM_SCHEDULER_GAMMA,
            'scheduler_patience': DEFAULT_IVDFM_SCHEDULER_PATIENCE,
            'scheduler_factor': DEFAULT_IVDFM_SCHEDULER_FACTOR,
            'scheduler_min_lr': DEFAULT_IVDFM_SCHEDULER_MIN_LR,
            'scaler': None,  # Scaler string: 'standard', 'robust', 'minmax', 'maxabs', 'quantile', or null
        })
        
        return ivdfm_params
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'iVDFMConfig':
        """Create iVDFMConfig from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        iVDFMConfig
            iVDFM configuration instance
        """
        # Extract base params
        base_params = cls._extract_base(data)
        
        # Extract iVDFM-specific params
        ivdfm_params = cls._extract_ivdfm(data)
        
        # Combine and create config
        return cls(**base_params, **ivdfm_params)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_frequency(frequency: str) -> str:
    """Validate frequency code.
    
    Parameters
    ----------
    frequency : str
        Frequency code to validate
        
    Returns
    -------
    str
        Validated frequency code
        
    Raises
    ------
    ConfigurationError
        If frequency is not in VALID_FREQUENCIES
    """
    from ..constants import VALID_FREQUENCIES
    from ...utils.errors import ConfigurationError
    
    if not isinstance(frequency, str):
        raise ConfigurationError(
            f"Frequency must be a string, got {type(frequency).__name__}: {frequency}"
        )
    
    if frequency not in VALID_FREQUENCIES:
        raise ConfigurationError(
            f"Invalid frequency: '{frequency}'. Must be one of {VALID_FREQUENCIES}. "
            f"Common frequencies: 'd' (daily), 'w' (weekly), 'm' (monthly), "
            f"'q' (quarterly), 'sa' (semi-annual), 'a' (annual)."
        )
    
    return frequency


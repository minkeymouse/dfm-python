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
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
    DEFAULT_CLOCK_FREQUENCY,
    DEFAULT_KDFM_AR_ORDER,
    DEFAULT_KDFM_MA_ORDER,
    FREQUENCY_HIERARCHY,
    DEFAULT_HIERARCHY_VALUE,
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
    # Note: nan_method and nan_k are internal constants (DEFAULT_NAN_METHOD, DEFAULT_NAN_K) used during initialization only
    
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
    
    # Idiosyncratic Component Parameters (auto-detected from frequencies)
    idio_rho0: float = DEFAULT_IDIO_RHO0  # Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var: float = MIN_DIAGONAL_VARIANCE  # Minimum variance for idiosyncratic innovation covariance (defaults to MIN_DIAGONAL_VARIANCE)
    # Note: augment_idio and augment_idio_slow are auto-detected from frequency configuration
    # - If all series use clock frequency: augment_idio=False, augment_idio_slow=False
    # - If mixed frequencies detected: augment_idio=True, augment_idio_slow=True (auto-enabled)
    
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
        """Create EMConfig from DFMConfig consolidated parameters."""
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
            block_series_sets = {
                name: set(self.blocks[name].get('series', []))
                for name in self.block_names
            }
            blocks_list = [
                [1 if series_id in block_series_sets[name] else 0 for name in self.block_names]
                for series_id in series_ids
            ]
            
            self._cached_blocks = np.array(blocks_list, dtype=int)
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
        }
        
        result = base_params.copy()
        result.update(dfm_params)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig', 'KDFMConfig']:
        """Create DFMConfig, DDFMConfig, or KDFMConfig from dictionary.
        
        Expected format: {'frequency': {'column_name': 'frequency'}, 'blocks': {...}, ...}
        
        Also accepts estimation parameters: threshold, max_iter, etc.
        """
        from ...config.adapter import detect_config_type, MODEL_TYPE_DDFM, MODEL_TYPE_KDFM, _normalize_blocks_dict
        from ...utils.errors import ConfigurationError
        
        # Extract base params (handles frequency conversion from series if needed)
        base_params = cls._extract_base(data)
        
        # Determine config type
        config_type = detect_config_type(data)
        
        if config_type == MODEL_TYPE_DDFM:
            return DDFMConfig(**base_params, **DDFMConfig._extract_ddfm(data))
        
        if config_type == MODEL_TYPE_KDFM:
            return KDFMConfig(**base_params, **KDFMConfig._extract_kdfm(data))
        
        # Handle blocks for DFM
        from ...config.adapter import _raise_config_error, _is_dict_like
        blocks_dict = data.get('blocks', {})
        if not blocks_dict:
            _raise_config_error("blocks dict is required for DFM config")
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
class KDFMConfig(BaseModelConfig):
    """KDFM configuration dataclass.
    
    This dataclass contains all configuration parameters for the KDFM model.
    It inherits from BaseModelConfig and adds KDFM-specific parameters.
    
    Note: KDFM does not use blocks structure (unlike DFM). Only frequency dict is needed.
    """
    # VARMA parameters
    ar_order: int = DEFAULT_KDFM_AR_ORDER  # VAR order p
    ma_order: int = DEFAULT_KDFM_MA_ORDER  # MA order q (0 = pure VAR)
    
    # Structural identification
    structural_method: str = 'cholesky'  # 'cholesky', 'full', 'low_rank'
    structural_rank: Optional[int] = None  # For low-rank parameterization
    
    # Training parameters (use constants for defaults)
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = DEFAULT_REGULARIZATION_SCALE
    grad_clip_val: float = DEFAULT_GRAD_CLIP_VAL
    
    # Regularization
    structural_reg_weight: float = DEFAULT_STRUCTURAL_REG_WEIGHT  # Weight for structural loss
    use_regularization: bool = True
    regularization_scale: float = DEFAULT_REGULARIZATION_SCALE
    
    # ========================================================================
    # Factory Methods
    # ========================================================================
    
    @classmethod
    def _extract_kdfm(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract KDFM-specific parameters from config dict."""
        # Don't extract base params here - they're already in base_params from from_dict
        kdfm_params = cls._extract_params(data, {
            'ar_order': DEFAULT_KDFM_AR_ORDER,
            'ma_order': DEFAULT_KDFM_MA_ORDER,
            'structural_method': 'cholesky',
            'structural_rank': None,
            'learning_rate': DEFAULT_LEARNING_RATE,
            'max_epochs': DEFAULT_MAX_EPOCHS,
            'batch_size': DEFAULT_BATCH_SIZE,
            'weight_decay': DEFAULT_REGULARIZATION_SCALE,
            'grad_clip_val': DEFAULT_GRAD_CLIP_VAL,
            'structural_reg_weight': DEFAULT_STRUCTURAL_REG_WEIGHT,
            'use_regularization': True,
            'regularization_scale': DEFAULT_REGULARIZATION_SCALE,
        })
        return kdfm_params
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KDFMConfig':
        """Create KDFMConfig from dictionary (delegates to DFMConfig.from_dict for type detection)."""
        result = DFMConfig.from_dict(data)
        if isinstance(result, KDFMConfig):
            return result
        from ..utils.errors import ConfigurationError
        raise ConfigurationError(
            f"Expected KDFMConfig but got {type(result).__name__}",
            details=f"Result type: {type(result).__name__}, expected: KDFMConfig"
        )


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


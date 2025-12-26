"""Configuration schema for DFM models.

This module provides model-specific configuration dataclasses:
- DFMConfig(BaseModelConfig): Linear DFM with EM algorithm parameters and block structure
- DDFMConfig(BaseModelConfig): Deep DFM with neural network training parameters (no blocks)
- KDFMConfig(BaseModelConfig): Kernelized DFM with VARMA parameters

Base classes (BaseModelConfig, SeriesConfig) are in config/base.py
Utility functions are in config/utils.py

The configuration hierarchy:
- BaseModelConfig (base.py): Model structure (series, clock, data handling) - NO blocks
- DFMConfig: Adds blocks structure and EM algorithm parameters (max_iter, threshold, regularization)
- DDFMConfig: Adds neural network parameters (epochs, learning_rate, encoder_layers) - NO blocks
- KDFMConfig: Adds VARMA parameters (ar_order, ma_order, structural_method) - NO blocks

Blocks are DFM-specific and defined as Dict[str, Dict[str, Any]] where each block is a dict with:
- factors: int (number of factors)
- ar_lag: int (AR lag order)
- clock: str (block clock frequency)

For loading configurations from files (YAML) or other sources,
see the config.adapter module which provides source adapters.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

# Import base classes and utilities
from .base import BaseModelConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from .utils import parse_series_list, parse_blocks_dict, infer_blocks, detect_config_type
from .constants import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DDFM_BATCH_SIZE,
    DEFAULT_GRAD_CLIP_VAL,
    DEFAULT_REGULARIZATION_SCALE,
    DEFAULT_STRUCTURAL_REG_WEIGHT,
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_MCMC_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_DATA_CLIP_THRESHOLD,
    MIN_EIGENVALUE,
    MAX_EIGENVALUE,
    MIN_DIAGONAL_VARIANCE,
    DEFAULT_NAN_METHOD,
    DEFAULT_NAN_K,
)


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
    - Series definitions from config/series/default.yaml or CSV
    - Block definitions from config/blocks/default.yaml
    """
    # ========================================================================
    # Block Structure (DFM-specific)
    # ========================================================================
    blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Block configurations (block_name -> {factors, ar_lag, clock, notes})
    block_names: List[str] = field(init=False)  # Block names in order (derived from blocks dict)
    factors_per_block: List[int] = field(init=False)  # Number of factors per block (derived from blocks)
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # Internal cache
    
    # ========================================================================
    # EM Algorithm Parameters (HOW - controls the algorithm)
    # ========================================================================
    ar_lag: int = 1  # Number of lags in AR transition equation (lookback window). Must be 1 or 2 (maximum supported order is VAR(2))
    threshold: float = 1e-5  # EM convergence threshold
    max_iter: int = 5000  # Maximum EM iterations
    
    # ========================================================================
    # Numerical Stability Parameters (transparent and configurable)
    # ========================================================================
    # AR Coefficient Clipping
    clip_ar_coefficients: bool = True  # Enable AR coefficient clipping for stationarity
    ar_clip_min: float = -0.99  # Minimum AR coefficient (must be > -1 for stationarity)
    ar_clip_max: float = 0.99   # Maximum AR coefficient (must be < 1 for stationarity)
    warn_on_ar_clip: bool = True  # Warn when AR coefficients are clipped (indicates near-unit root)
    
    # Data Value Clipping
    clip_data_values: bool = True  # Enable clipping of extreme data values
    data_clip_threshold: float = 100.0  # Clip values beyond this many standard deviations
    warn_on_data_clip: bool = True  # Warn when data values are clipped (indicates outliers)
    
    # Regularization
    use_regularization: bool = True  # Enable regularization for numerical stability
    regularization_scale: float = 1e-5  # Scale factor for ridge regularization (relative to trace, default 1e-5)
    min_eigenvalue: float = 1e-8  # Minimum eigenvalue for positive definite matrices
    max_eigenvalue: float = 1e6   # Maximum eigenvalue cap to prevent explosion
    warn_on_regularization: bool = True  # Warn when regularization is applied
    
    # Damped Updates
    use_damped_updates: bool = True  # Enable damped updates when likelihood decreases
    damping_factor: float = 0.8  # Damping factor (0.8 = 80% new, 20% old)
    warn_on_damped_update: bool = True  # Warn when damped updates are used
    
    # Idiosyncratic Component Augmentation
    augment_idio: bool = True  # Enable state augmentation with idiosyncratic components (default: True)
    augment_idio_slow: bool = True  # Enable tent-length chains for slower-frequency series (default: True)
    idio_rho0: float = 0.1  # Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var: float = 1e-8  # Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
    
    def __post_init__(self):
        """Validate blocks structure and consistency for DFM.
        
        This method performs comprehensive validation of the DFM configuration:
        - Derives block_names and factors_per_block from blocks dict
        - Ensures at least one series is specified
        - Validates block structure consistency across all series
        - Ensures all series load on the global block
        - Validates block clock constraints (series frequency <= block clock)
        - Validates factor dimensions match block structure
        - Validates clock frequency
        
        Raises
        ------
        ValueError
            If any validation check fails, with a descriptive error message
            indicating what needs to be fixed.
        """
        # Call parent __post_init__ first (validates series and clock)
        super().__post_init__()
        
        # Import frequency hierarchy for validation
        from .utils import FREQUENCY_HIERARCHY
        
        if not self.blocks:
            raise ValueError(
                "DFM configuration must contain at least one block. "
                "Please add block definitions to your configuration."
            )
        
        # Derive block_names and factors_per_block from blocks dict
        # Ensure global block (first block) is present
        block_names_list = list(self.blocks.keys())
        global_block_name = None
        
        # Try to find first block (use first block as default)
        if block_names_list:
            global_block_name = block_names_list[0]
        elif DEFAULT_BLOCK_NAME in self.blocks:
            global_block_name = DEFAULT_BLOCK_NAME
        elif block_names_list:
            # Use first block as global if Block_Global not found
            global_block_name = block_names_list[0]
        
        if global_block_name is None:
            raise ValueError(
                "DFM configuration must include at least one block. "
                "The first block serves as the global/common factor that all series load on."
            )
        
        # Build ordered list: global block first, then others
        other_blocks = [name for name in block_names_list if name != global_block_name]
        object.__setattr__(self, 'block_names', [global_block_name] + other_blocks)
        object.__setattr__(self, 'factors_per_block', 
                         [self.blocks[name].get('factors', 1) for name in self.block_names])
        
        # Validate global clock
        global_clock_hierarchy = FREQUENCY_HIERARCHY.get(self.clock, 3)
        
        # Validate block clocks (must be >= global clock)
        for block_name, block_cfg in self.blocks.items():
            block_clock = block_cfg.get('clock', self.clock)
            from .utils import validate_frequency
            block_clock = validate_frequency(block_clock)
            block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_clock, 3)
            if block_clock_hierarchy < global_clock_hierarchy:
                raise ValueError(
                    f"Block '{block_name}' has clock '{block_clock}' which is faster than "
                    f"global clock '{self.clock}'. Block clocks must be >= global clock. "
                    f"Suggested fix: change block '{block_name}' clock to '{self.clock}' or slower, "
                    f"or set global clock to '{block_clock}' or faster."
                )
            # Validate block properties
            factors = block_cfg.get('factors', 1)
            ar_lag = block_cfg.get('ar_lag', 1)
            if factors < 1:
                raise ValueError(
                    f"Block '{block_name}' validation failed: must have at least 1 factor, got {factors}. "
                    f"Please set factors >= 1 for block '{block_name}'."
                )
            if ar_lag < 1:
                raise ValueError(
                    f"Block '{block_name}' validation failed: AR lag must be at least 1, got {ar_lag}. "
                    f"Please set ar_lag >= 1 for block '{block_name}'."
                )
            if ar_lag > 2:
                raise ValueError(
                    f"Block '{block_name}' validation failed: AR lag must be at most 2 (maximum supported VAR order is VAR(2)), got {ar_lag}. "
                    f"Please set ar_lag to 1 (VAR(1)) or 2 (VAR(2)) for block '{block_name}'."
                )
        
        # Auto-generate series_id if not provided
        # Note: Blocks are defined in DFMConfig, not in SeriesConfig.
        # Series-to-block mapping is handled separately (e.g., via series_to_blocks dict or default to global block).
        for i, s in enumerate(self.series):
            if s.series_id is None:
                object.__setattr__(s, 'series_id', f"series_{i}")
        
        # Validate block clock constraints: series frequency <= block clock
        # All series load on all blocks by default (can be customized via series_to_blocks mapping)
        for i, s in enumerate(self.series):
            series_freq_hierarchy = FREQUENCY_HIERARCHY.get(s.frequency, 3)
            
            # Check against all blocks (series can load on multiple blocks)
            for block_name, block_cfg in self.blocks.items():
                block_clock = block_cfg.get('clock', self.clock)
                block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_clock, 3)
                
                # Series frequency must be <= block clock (slower or equal)
                if series_freq_hierarchy < block_clock_hierarchy:
                    # Suggest valid frequencies for the series
                    valid_freqs = [freq for freq, hier in FREQUENCY_HIERARCHY.items() 
                                  if hier >= block_clock_hierarchy]
                    valid_freqs_str = ', '.join(sorted(valid_freqs))
                    raise ValueError(
                        f"Series '{s.series_id}' has frequency '{s.frequency}' which is faster than "
                        f"block '{block_name}' clock '{block_clock}'. "
                        f"Series in a block must have frequency <= block clock. "
                        f"Suggested fix: change series frequency to one of [{valid_freqs_str}] "
                        f"(slower or equal to block clock '{block_clock}'), "
                        f"or set block clock to '{s.frequency}' or faster."
                    )
        
        # Validate factors_per_block
        if any(f < 1 for f in self.factors_per_block):
            invalid_blocks = [i for i, f in enumerate(self.factors_per_block) if f < 1]
            raise ValueError(
                f"factors_per_block must contain positive integers (>= 1). "
                f"Invalid values found at block indices {invalid_blocks}: "
                f"{[self.factors_per_block[i] for i in invalid_blocks]}. "
                f"Each block must have at least one factor."
            )
    
    def get_blocks_array(self) -> np.ndarray:
        """Get blocks as numpy array (snake_case - recommended, cached).
        
        Note: SeriesConfig no longer contains blocks information.
        By default, all series load on all blocks (all 1s).
        Custom series-to-block mapping can be provided via series_to_blocks attribute.
        """
        if self._cached_blocks is None:
            n_series = len(self.series)
            n_blocks = len(self.block_names)
            # Default: all series load on all blocks (all 1s)
            # Custom mapping can be provided via series_to_blocks attribute if needed
            if hasattr(self, 'series_to_blocks') and self.series_to_blocks is not None:
                blocks_list = []
                for s in self.series:
                    series_blocks = self.series_to_blocks.get(s.series_id, [1] * n_blocks)
                    blocks_list.append(series_blocks)
                self._cached_blocks = np.array(blocks_list, dtype=int)
            else:
                # Default: all series load on all blocks
                self._cached_blocks = np.ones((n_series, n_blocks), dtype=int)
        return self._cached_blocks
    
    # ========================================================================
    # Factory Methods
    # ========================================================================
    
    @classmethod
    def _extract_base(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shared base parameters from config dict."""
        return {
            'nan_method': data.get('nan_method', 2),
            'nan_k': data.get('nan_k', 3),
            'clock': data.get('clock', 'm'),
            'scaler': data.get('scaler', 'standard'),  # Unified scaler for all series (default: 'standard')
        }
    
    @classmethod
    def _extract_dfm_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DFM-specific parameters from config dict."""
        base_params = cls._extract_base(data)
        base_params.update({
            'ar_lag': data.get('ar_lag', 1),
            'threshold': data.get('threshold', 1e-5),
            'max_iter': data.get('max_iter', 5000),
            # Numerical stability parameters
            'clip_ar_coefficients': data.get('clip_ar_coefficients', True),
            'ar_clip_min': data.get('ar_clip_min', -0.99),
            'ar_clip_max': data.get('ar_clip_max', 0.99),
            'warn_on_ar_clip': data.get('warn_on_ar_clip', True),
            'clip_data_values': data.get('clip_data_values', True),
            'data_clip_threshold': data.get('data_clip_threshold', DEFAULT_DATA_CLIP_THRESHOLD),
            'warn_on_data_clip': data.get('warn_on_data_clip', True),
            'use_regularization': data.get('use_regularization', True),
            'regularization_scale': data.get('regularization_scale', 1e-5),
            'min_eigenvalue': data.get('min_eigenvalue', 1e-8),
            'max_eigenvalue': data.get('max_eigenvalue', 1e6),
            'warn_on_regularization': data.get('warn_on_regularization', True),
            'use_damped_updates': data.get('use_damped_updates', True),
            'damping_factor': data.get('damping_factor', 0.8),
            'warn_on_damped_update': data.get('warn_on_damped_update', True),
            # Idiosyncratic component augmentation
            'augment_idio': data.get('augment_idio', True),
            'augment_idio_slow': data.get('augment_idio_slow', True),
            'idio_rho0': data.get('idio_rho0', 0.1),
            'idio_min_var': data.get('idio_min_var', MIN_DIAGONAL_VARIANCE),
        })
        return base_params


@dataclass
class DDFMConfig(BaseModelConfig):
    """Deep Dynamic Factor Model configuration - neural network training parameters.
    
    This configuration class extends BaseModelConfig with parameters specific
    to Deep Dynamic Factor Models trained using neural networks (autoencoders).
    
    Note: DDFM does NOT use block structure. Use num_factors directly to specify
    the number of factors. Blocks are DFM-specific and not needed for DDFM.
    
    The configuration can be built from:
    - Main settings (training parameters) from config/default.yaml
    - Series definitions from config/series/default.yaml or CSV
    """
    # ========================================================================
    # Neural Network Training Parameters
    # ========================================================================
    encoder_layers: Optional[List[int]] = None  # Hidden layer dimensions for encoder (default: [64, 32])
    num_factors: Optional[int] = None  # Number of factors (inferred from config if None)
    activation: str = 'relu'  # Activation function ('tanh', 'relu', 'sigmoid', default: 'relu' to match original DDFM)
    use_batch_norm: bool = True  # Use batch normalization in encoder (default: True)
    learning_rate: float = 0.001  # Learning rate for Adam optimizer (default: 0.001)
    epochs: int = 100  # Number of training epochs (default: 100)
    batch_size: int = 100  # Batch size for training (default: 100 to match original DDFM)
    factor_order: int = 1  # VAR lag order for factor dynamics. Must be 1 or 2 (maximum supported order is VAR(2), default: 1)
    use_idiosyncratic: bool = True  # Model idio components with AR(1) dynamics (default: True)
    min_obs_idio: int = 5  # Minimum observations for idio AR(1) estimation (default: 5)
    
    # Additional training parameters
    max_iter: int = DEFAULT_MAX_MCMC_ITER  # Maximum MCMC iterations for iterative factor extraction
    tolerance: float = DEFAULT_TOLERANCE  # Convergence tolerance for MCMC iterations
    disp: int = 10  # Display frequency for training progress
    seed: Optional[int] = None  # Random seed for reproducibility
    
    
    # ========================================================================
    # Factory Methods (shared base methods)
    # ========================================================================
    
    @classmethod
    def _extract_base(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract shared base parameters from config dict (delegates to DFMConfig)."""
        return DFMConfig._extract_base(data)
    
    @classmethod
    def _extract_dfm_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DFM-specific parameters from config dict (delegates to DFMConfig)."""
        return DFMConfig._extract_dfm_params(data)
    
    @classmethod
    def _extract_ddfm(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DDFM-specific parameters from config dict."""
        base_params = cls._extract_base(data)
        # Handle both direct keys and ddfm_ prefix format
        base_params.update({
            'encoder_layers': data.get('encoder_layers') or data.get('ddfm_encoder_layers', None),
            'num_factors': data.get('num_factors') or data.get('ddfm_num_factors', None),
            'activation': data.get('activation') or data.get('ddfm_activation', 'relu'),
            'use_batch_norm': data.get('use_batch_norm', data.get('ddfm_use_batch_norm', True)),
            'learning_rate': data.get('learning_rate', data.get('ddfm_learning_rate', 0.001)),
            'epochs': data.get('epochs', data.get('ddfm_epochs', 100)),
            'batch_size': data.get('batch_size', data.get('ddfm_batch_size', 100)),
            'factor_order': data.get('factor_order', data.get('ddfm_factor_order', 1)),
            'use_idiosyncratic': data.get('use_idiosyncratic', data.get('ddfm_use_idiosyncratic', True)),
            'min_obs_idio': data.get('min_obs_idio', data.get('ddfm_min_obs_idio', 5)),
            'max_iter': data.get('max_iter', DEFAULT_MAX_MCMC_ITER),
            'tolerance': data.get('tolerance', DEFAULT_TOLERANCE),
            'disp': data.get('disp', 10),
            'seed': data.get('seed', None),
        })
        return base_params
    
    @classmethod
    def _from_hydra_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig']:
        """Convert Hydra format (series as dict) to new format."""
        # Get block_names first (required for series processing)
        blocks_dict = data.get('blocks', {})
        if isinstance(blocks_dict, dict) and blocks_dict:
            block_names = list(blocks_dict.keys())
            factors_per_block = [
                blocks_dict[bn].get('factors', 1) if isinstance(blocks_dict[bn], dict) else blocks_dict[bn]
                for bn in block_names
            ]
        else:
            block_names = data.get('block_names', [])
            factors_per_block = data.get('factors_per_block', None)
        
        # Note: SeriesConfig no longer contains blocks information.
        # Blocks must be defined in DFMConfig, not inferred from series.
        # If block_names is still empty, create default block
        if not block_names:
            block_names = [DEFAULT_BLOCK_NAME]
            factors_per_block = [1]
            blocks_dict = {DEFAULT_BLOCK_NAME: {'factors': 1, 'ar_lag': 1, 'clock': data.get('clock', 'm')}}
        
        # Parse series dict: {series_id: {frequency: ..., ...}}
        # Note: transformation is handled by preprocessing pipeline, not in SeriesConfig
        # Note: blocks are defined in DFMConfig, not in SeriesConfig
        series_list = []
        for series_id, series_cfg in data['series'].items():
            if isinstance(series_cfg, dict):
                series_list.append(SeriesConfig(
                    series_id=series_id,
                    series_name=series_cfg.get('series_name', series_id),
                    frequency=series_cfg.get('frequency', 'm'),
                    # transformation removed - handled by preprocessing pipeline
                    # blocks removed - defined in DFMConfig
                    units=series_cfg.get('units', None),  # Optional, for display only
                    release_date=series_cfg.get('release_date', None)  # Optional, for nowcasting
                ))
        
        # Convert blocks_dict to dict of block properties
        blocks_dict_final = {}
        if isinstance(blocks_dict, dict) and blocks_dict:
            # Already have blocks dict from input
            for block_name, block_data in blocks_dict.items():
                if isinstance(block_data, dict):
                    blocks_dict_final[block_name] = {
                        'factors': block_data.get('factors', 1),
                        'ar_lag': block_data.get('ar_lag', 1),
                        'clock': block_data.get('clock', 'm'),
                        'notes': block_data.get('notes', None)
                    }
                else:
                    blocks_dict_final[block_name] = {'factors': 1, 'ar_lag': 1, 'clock': 'm'}
        elif block_names:
            # Create blocks dict from block_names (fallback)
            for i, block_name in enumerate(block_names):
                factors = factors_per_block[i] if factors_per_block and i < len(factors_per_block) else 1
                blocks_dict_final[block_name] = {'factors': factors, 'ar_lag': 1, 'clock': 'm'}
        else:
            # Default: create default block if no blocks specified
            blocks_dict_final[DEFAULT_BLOCK_NAME] = {'factors': 1, 'ar_lag': 1, 'clock': 'm'}
        
        # Determine config type using helper function
        config_type = detect_config_type(data)
        
        if config_type == 'ddfm':
            # DDFM does not use block structure - no blocks needed
            return DDFMConfig(
                series=series_list,
                **DDFMConfig._extract_ddfm(data)
            )
        else:
            return DFMConfig(
                series=series_list,
                blocks=blocks_dict_final,
                **DDFMConfig._extract_dfm_params(data)
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig']:
        """Create DFMConfig from dictionary.
        
        Handles multiple formats:
        1. New format (list): {'series': [{'series_id': ..., ...}], 'block_names': [...]}
        2. New format (Hydra): {'series': {'series_id': {...}}, 'blocks': {'block_name': {'factors': N}}}
        
        Also accepts estimation parameters: ar_lag, threshold, max_iter, nan_method, nan_k
        """
        # New Hydra format: series is a dict
        if 'series' in data and isinstance(data['series'], dict):
            return cls._from_hydra_dict(data)
        
        # New format with series list
        if 'series' in data and isinstance(data['series'], list):
            # Parse series list using helper
            series_list = parse_series_list(data['series'])
            
            # Handle blocks: dict of block properties
            if 'blocks' in data:
                blocks_data = data['blocks']
                if isinstance(blocks_data, dict):
                    blocks_dict = parse_blocks_dict(blocks_data)
                else:
                    raise ValueError(f"blocks must be a dict, got {type(blocks_data)}")
            else:
                # If no blocks provided, infer from series using helper
                blocks_dict = infer_blocks(series_list, data)
            
            # Determine config type using helper function
            config_type = detect_config_type(data)
            
            if config_type == 'ddfm':
                # DDFM does not use block structure - no blocks needed
                return DDFMConfig(
                    series=series_list,
                    **DDFMConfig._extract_ddfm(data)
                )
            else:
                return DFMConfig(
                    series=series_list,
                    blocks=blocks_dict,
                    **DFMConfig._extract_dfm_params(data)
                )
        
        # Direct instantiation (shouldn't happen often, but handle it)
        # Try to determine type from instance
        if isinstance(cls, type) and issubclass(cls, DDFMConfig):
            return cls(**data)
        elif isinstance(cls, type) and issubclass(cls, DFMConfig):
            return cls(**data)
        else:
            # Default to DFMConfig
            return DFMConfig(**data)

    @classmethod
    def from_hydra(cls, cfg: Any) -> Union['DFMConfig', 'DDFMConfig']:
        """Create config from a Hydra DictConfig or plain dict.
        
        Parameters
        ----------
        cfg : DictConfig | dict
            Hydra DictConfig (or dict) that contains the composed configuration.
        
        Returns
        -------
        DFMConfig or DDFMConfig
            Validated configuration instance (type determined automatically).
        """
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
            if isinstance(cfg, DictConfig):
                cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            # OmegaConf not available or not a DictConfig; assume dict
            pass
        if not isinstance(cfg, dict):
            raise TypeError("from_hydra expects a DictConfig or dict.")
        # Use DFMConfig.from_dict which handles type detection (defined on DFMConfig, not BaseModelConfig)
        return DFMConfig.from_dict(cfg)

@dataclass
class KDFMConfig(BaseModelConfig):
    """KDFM configuration dataclass.
    
    This dataclass contains all configuration parameters for the KDFM model.
    It inherits from BaseModelConfig and adds KDFM-specific parameters.
    """
    # VARMA parameters
    ar_order: int = 1  # VAR order p
    ma_order: int = 0  # MA order q (0 = pure VAR)
    
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

# Add factory methods to DFMConfig class
def _dfm_from_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig']:
    """Create DFMConfig or DDFMConfig from dictionary (auto-detects type)."""
    # Handle Hydra format (series as dict)
    if 'series' in data and isinstance(data['series'], dict):
        # Use shared _from_hydra_dict which has detection logic
        return DDFMConfig._from_hydra_dict(data)
    
    # Handle list format - use detection logic to determine config type
    if 'series' in data and isinstance(data['series'], list):
        # Parse series list using helper
        series_list = parse_series_list(data['series'])
        
        # Handle blocks using helpers
        if 'blocks' in data:
            blocks_data = data['blocks']
            if isinstance(blocks_data, dict):
                blocks_dict = parse_blocks_dict(blocks_data)
            else:
                raise ValueError(f"blocks must be a dict, got {type(blocks_data)}")
        else:
            # Infer blocks from series using helper
            blocks_dict = infer_blocks(series_list, data)
        
        # Determine config type using helper function
        config_type = detect_config_type(data)
        
        if config_type == 'ddfm':
            # DDFM does not use blocks - don't pass blocks parameter
            return DDFMConfig(
                series=series_list,
                **DDFMConfig._extract_ddfm(data)
            )
        else:
            return DFMConfig(
                series=series_list,
                blocks=blocks_dict,
                **DFMConfig._extract_dfm_params(data)
            )
    
    # Direct instantiation - try to detect type using helper function
    config_type = detect_config_type(data)
    
    if config_type == 'ddfm':
        return DDFMConfig(**data)
    else:
        return DFMConfig(**data)

def _from_hydra(cls, cfg: Any) -> Union['DFMConfig', 'DDFMConfig']:
    """Create config from Hydra DictConfig (auto-detects DFM/DDFM)."""
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if not isinstance(cfg, dict):
        raise TypeError("from_hydra expects a DictConfig or dict.")
    return cls.from_dict(cfg)

DFMConfig.from_dict = classmethod(_dfm_from_dict)
DFMConfig.from_hydra = classmethod(_from_hydra)

# Add factory methods to DDFMConfig class
def _ddfm_from_dict(cls, data: Dict[str, Any]) -> Union['DFMConfig', 'DDFMConfig']:
    """Create DDFMConfig or DFMConfig from dictionary (auto-detects type)."""
    # Handle Hydra format (series as dict)
    if 'series' in data and isinstance(data['series'], dict):
        # Use shared _from_hydra_dict which has detection logic
        return DDFMConfig._from_hydra_dict(data)
    
    # Handle list format - use detection logic to determine config type
    if 'series' in data and isinstance(data['series'], list):
        # Parse series list using helper
        series_list = parse_series_list(data['series'])
        
        # Handle blocks using helpers
        if 'blocks' in data:
            blocks_data = data['blocks']
            if isinstance(blocks_data, dict):
                blocks_dict = parse_blocks_dict(blocks_data)
            else:
                raise ValueError(f"blocks must be a dict, got {type(blocks_data)}")
        else:
            # Infer blocks from series using helper
            blocks_dict = infer_blocks(series_list, data)
        
        # Determine config type using helper function
        config_type = detect_config_type(data)
        
        if config_type == 'kdfm':
            # KDFM does not use blocks
            return KDFMConfig(
                series=series_list,
                **{k: v for k, v in data.items() if k not in ['series', 'blocks']}
            )
        elif config_type == 'ddfm':
            # DDFM does not use block structure - no blocks needed
            return DDFMConfig(
                series=series_list,
                **DDFMConfig._extract_ddfm(data)
            )
        else:
            # DFM uses blocks
            return DFMConfig(
                series=series_list,
                blocks=blocks_dict,
                **DFMConfig._extract_dfm_params(data)
            )
    
    # Direct instantiation - try to detect type using helper function
    config_type = detect_config_type(data)
    
    if config_type == 'kdfm':
        # KDFM does not use blocks
        data_clean = {k: v for k, v in data.items() if k != 'blocks'}
        return KDFMConfig(**data_clean)
    elif config_type == 'ddfm':
        # DDFM does not use block structure - remove blocks from data if present
        data_clean = {k: v for k, v in data.items() if k != 'blocks'}
        return DDFMConfig(**data_clean)
    else:
        return DFMConfig(**data)

DDFMConfig.from_dict = classmethod(_ddfm_from_dict)
DDFMConfig.from_hydra = classmethod(_from_hydra)



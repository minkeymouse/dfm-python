"""Configuration models for DFM nowcasting using OmegaConf and Hydra.

This module defines the configuration structure for Dynamic Factor Models,
providing a unified interface for specifying:
- Series definitions (frequency, transformation, block loadings)
- Block structure and factor dimensions
- Estimation parameters (EM algorithm settings, missing data handling)
- Clock frequency for mixed-frequency synchronization

The configuration system supports both YAML and CSV formats, with validation
and type checking to ensure model specifications are correct before estimation.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import warnings
from dataclasses import dataclass, field

try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    ConfigStore = None
    DictConfig = None
    OmegaConf = None

# Valid frequency codes
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes
_VALID_TRANSFORMATIONS = {
    'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 
    'cch', 'cca', 'log'
}

# Transformation to readable units mapping
_TRANSFORM_UNITS_MAP = {
    'lin': 'Levels (No Transformation)',
    'chg': 'Change (Difference)',
    'ch1': 'Year over Year Change (Difference)',
    'pch': 'Percent Change',
    'pc1': 'Year over Year Percent Change',
    'pca': 'Percent Change (Annual Rate)',
    'cch': 'Continuously Compounded Rate of Change',
    'cca': 'Continuously Compounded Annual Rate of Change',
    'log': 'Natural Log'
}


def validate_frequency(frequency: str) -> str:
    """Validate frequency code."""
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {_VALID_FREQUENCIES}")
    return frequency


def validate_transformation(transformation: str) -> str:
    """Validate transformation code."""
    if transformation not in _VALID_TRANSFORMATIONS:
        warnings.warn(f"Unknown transformation code: {transformation}. Will use untransformed data.")
    return transformation


@dataclass
class SeriesConfig:
    """Configuration for a single time series.
    
    This is a generic DFM configuration - no API or database-specific fields.
    For API/database integration, implement adapters in your application layer.
    """
    # Required fields (no defaults)
    frequency: str
    transformation: str
    blocks: List[int]
    # Optional fields (with defaults - must come after required fields)
    series_id: Optional[str] = None  # Auto-generated if None: "series_0", "series_1", etc.
    series_name: Optional[str] = None  # Optional metadata for display
    units: str = ""  # Optional metadata
    category: str = ""  # Optional metadata
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self.frequency = validate_frequency(self.frequency)
        self.transformation = validate_transformation(self.transformation)
        # Auto-generate series_id if not provided
        if self.series_id is None:
            # Will be set when SeriesConfig is created in a list
            pass
        # Auto-generate series_name if not provided
        if self.series_name is None and self.series_id:
            self.series_name = self.series_id


@dataclass
class DFMConfig:
    """Unified DFM configuration - model structure + estimation parameters.
    
    This is the single configuration class for the DFM module, combining:
    - Model structure (what series, blocks, factors)
    - Estimation parameters (how to run EM algorithm)
    """
    # ========================================================================
    # Model Structure (WHAT - defines the model)
    # ========================================================================
    series: List[SeriesConfig]  # Series specifications
    block_names: List[str]  # Block names (e.g., ["Global", "Consumption", "Investment"])
    factors_per_block: Optional[List[int]] = None  # Number of factors per block. If None, defaults to 1 per block
    
    # ========================================================================
    # Estimation Parameters (HOW - controls the algorithm)
    # ========================================================================
    ar_lag: int = 1  # Number of lags in AR transition equation (lookback window)
    threshold: float = 1e-5  # EM convergence threshold
    max_iter: int = 5000  # Maximum EM iterations
    nan_method: int = 2  # Missing data handling method (1-5)
    nan_k: int = 3  # Spline parameter for NaN interpolation
    clock: str = 'm'  # Base frequency for nowcasting (global clock): 'd', 'w', 'm', 'q', 'sa', 'a' (defaults to 'm' for monthly)
    
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
    regularization_scale: float = 1e-6  # Scale factor for ridge regularization (relative to trace)
    min_eigenvalue: float = 1e-8  # Minimum eigenvalue for positive definite matrices
    max_eigenvalue: float = 1e6   # Maximum eigenvalue cap to prevent explosion
    warn_on_regularization: bool = True  # Warn when regularization is applied
    
    # Damped Updates
    use_damped_updates: bool = True  # Enable damped updates when likelihood decreases
    damping_factor: float = 0.8  # Damping factor (0.8 = 80% new, 20% old)
    warn_on_damped_update: bool = True  # Warn when damped updates are used
    
    # ========================================================================
    # Internal cache (not user-configurable)
    # ========================================================================
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate blocks structure and consistency.
        
        This method performs comprehensive validation of the DFM configuration:
        - Ensures at least one series is specified
        - Validates block structure consistency across all series
        - Ensures all series load on the global block
        - Validates factor dimensions match block structure
        - Validates clock frequency
        
        Raises
        ------
        ValueError
            If any validation check fails, with a descriptive error message
            indicating what needs to be fixed.
        """
        if not self.series:
            raise ValueError(
                "DFM configuration must contain at least one series. "
                "Please add series definitions to your configuration."
            )
        
        # Auto-generate series_id if not provided
        for i, s in enumerate(self.series):
            if s.series_id is None:
                s.series_id = f"series_{i}"
            if s.series_name is None:
                s.series_name = s.series_id
        
        # Extract blocks matrix
        n_series = len(self.series)
        n_blocks = len(self.block_names)
        
        # Check all series have same number of blocks
        for i, s in enumerate(self.series):
            if len(s.blocks) != n_blocks:
                raise ValueError(
                    f"Series {i} ('{s.series_id}') has {len(s.blocks)} block loadings, "
                    f"but expected {n_blocks} (from block_names: {self.block_names}). "
                    f"Each series must specify a loading (0 or 1) for each block."
                )
        
        # Check first column (global block) is all 1s
        for i, s in enumerate(self.series):
            if s.blocks[0] != 1:
                raise ValueError(
                    f"Series {i} ('{s.series_id}') must load on the global block "
                    f"(first block '{self.block_names[0]}'). "
                    f"All series must have blocks[0] = 1. "
                    f"Current value: {s.blocks[0]}"
                )
        
        # Validate factors_per_block if provided
        if self.factors_per_block is not None:
            if len(self.factors_per_block) != n_blocks:
                raise ValueError(
                    f"factors_per_block length ({len(self.factors_per_block)}) must match "
                    f"number of blocks ({n_blocks}). "
                    f"Block names: {self.block_names}. "
                    f"Please provide one factor count per block."
                )
            if any(f < 1 for f in self.factors_per_block):
                invalid_blocks = [i for i, f in enumerate(self.factors_per_block) if f < 1]
                raise ValueError(
                    f"factors_per_block must contain positive integers (>= 1). "
                    f"Invalid values found at block indices {invalid_blocks}: "
                    f"{[self.factors_per_block[i] for i in invalid_blocks]}. "
                    f"Each block must have at least one factor."
                )
        
        # Validate clock
        self.clock = validate_frequency(self.clock)
    
    # Convenience properties for backward compatibility
    @property
    def SeriesID(self) -> List[str]:
        """Backward compatibility: SeriesID property."""
        return [s.series_id if s.series_id is not None else f"series_{i}" 
                for i, s in enumerate(self.series)]
    
    @property
    def SeriesName(self) -> List[str]:
        """Backward compatibility: SeriesName property."""
        return [s.series_name if s.series_name is not None else (s.series_id or f"series_{i}")
                for i, s in enumerate(self.series)]
    
    @property
    def Frequency(self) -> List[str]:
        """Backward compatibility: Frequency property."""
        return [s.frequency for s in self.series]
    
    @property
    def Units(self) -> List[str]:
        """Backward compatibility: Units property."""
        return [s.units for s in self.series]
    
    @property
    def Transformation(self) -> List[str]:
        """Backward compatibility: Transformation property."""
        return [s.transformation for s in self.series]
    
    @property
    def Category(self) -> List[str]:
        """Backward compatibility: Category property."""
        return [s.category for s in self.series]
    
    @property
    def Blocks(self) -> np.ndarray:
        """Backward compatibility: Blocks property as numpy array (cached)."""
        if self._cached_blocks is None:
            blocks_list = [s.blocks for s in self.series]
            self._cached_blocks = np.array(blocks_list, dtype=int)
        return self._cached_blocks
    
    @property
    def BlockNames(self) -> List[str]:
        """Backward compatibility: BlockNames property (alias for block_names)."""
        return self.block_names
    
    @property
    def UnitsTransformed(self) -> List[str]:
        """Backward compatibility: UnitsTransformed property."""
        return [_TRANSFORM_UNITS_MAP.get(t, t) for t in self.Transformation]
    
    @classmethod
    def _from_legacy_dict(cls, data: Dict[str, Any]) -> 'DFMConfig':
        """Convert legacy format (separate lists) to new format (series list)."""
        series_list = []
        n = len(data.get('SeriesID', data.get('series_id', [])))
        
        # Handle Blocks - can be numpy array or list of lists
        blocks_data = data.get('Blocks', data.get('blocks', []))
        if isinstance(blocks_data, np.ndarray):
            blocks_data = blocks_data.tolist()
        elif not isinstance(blocks_data, list):
            blocks_data = []
        
        # Helper to get list value with index fallback
        def get_list_value(key: str, index: int, default=None):
            """Get value from list, handling both camelCase and snake_case keys."""
            val = data.get(key, data.get(key.lower(), default))
            if isinstance(val, list) and index < len(val):
                return val[index]
            return default
        
        for i in range(n):
            # Extract blocks for this series
            if blocks_data and i < len(blocks_data):
                if isinstance(blocks_data[i], (list, np.ndarray)):
                    series_blocks = list(blocks_data[i]) if isinstance(blocks_data[i], np.ndarray) else blocks_data[i]
                else:
                    series_blocks = [blocks_data[i]]
            else:
                series_blocks = []
            
            freq_val = get_list_value('Frequency', i, 'm')
            trans_val = get_list_value('Transformation', i, 'lin')
            units_val = get_list_value('Units', i, '')
            cat_val = get_list_value('Category', i, '')
            series_list.append(SeriesConfig(
                frequency=str(freq_val) if freq_val is not None else 'm',
                transformation=str(trans_val) if trans_val is not None else 'lin',
                blocks=series_blocks,
                series_id=get_list_value('SeriesID', i, None),
                series_name=get_list_value('SeriesName', i, None),
                units=str(units_val) if units_val is not None else '',
                category=str(cat_val) if cat_val is not None else ''
            ))
        
        # Extract estimation parameters if provided
        return cls(
                series=series_list,
                block_names=data.get('BlockNames', data.get('block_names', [])),
                factors_per_block=data.get('factors_per_block', None),
                # Estimation parameters
                ar_lag=data.get('ar_lag', 1),
                threshold=data.get('threshold', 1e-5),
                max_iter=data.get('max_iter', 5000),
                nan_method=data.get('nan_method', 2),
                nan_k=data.get('nan_k', 3),
                clock=data.get('clock', 'm'),
                # Numerical stability parameters
                clip_ar_coefficients=data.get('clip_ar_coefficients', True),
                ar_clip_min=data.get('ar_clip_min', -0.99),
                ar_clip_max=data.get('ar_clip_max', 0.99),
                warn_on_ar_clip=data.get('warn_on_ar_clip', True),
                clip_data_values=data.get('clip_data_values', True),
                data_clip_threshold=data.get('data_clip_threshold', 100.0),
                warn_on_data_clip=data.get('warn_on_data_clip', True),
                use_regularization=data.get('use_regularization', True),
                regularization_scale=data.get('regularization_scale', 1e-6),
                min_eigenvalue=data.get('min_eigenvalue', 1e-8),
                max_eigenvalue=data.get('max_eigenvalue', 1e6),
                warn_on_regularization=data.get('warn_on_regularization', True),
                use_damped_updates=data.get('use_damped_updates', True),
                damping_factor=data.get('damping_factor', 0.8),
                warn_on_damped_update=data.get('warn_on_damped_update', True)
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DFMConfig':
        """Create DFMConfig from dictionary.
        
        Handles multiple formats:
        1. Legacy format: {'SeriesID': [...], 'Frequency': [...], 'Blocks': [[...]], ...}
        2. New format (list): {'series': [{'series_id': ..., ...}], 'block_names': [...]}
        3. New format (Hydra): {'series': {'series_id': {...}}, 'blocks': {'block_name': {'factors': N}}}
        
        Also accepts estimation parameters: ar_lag, threshold, max_iter, nan_method, nan_k
        """
        # Detect legacy format (has SeriesID or series_id as lists)
        if 'SeriesID' in data or ('series_id' in data and isinstance(data.get('series_id'), list)):
            return cls._from_legacy_dict(data)
        
        # New Hydra format: series is a dict (series_id -> config), blocks is a dict (block_name -> {factors: N})
        if 'series' in data and isinstance(data['series'], dict):
            # First, parse blocks dict to get block_names (needed for conversion)
            blocks_dict = data.get('blocks', {})
            if isinstance(blocks_dict, dict):
                block_names = list(blocks_dict.keys())
                factors_per_block = [
                    blocks_dict[bn].get('factors', 1) if isinstance(blocks_dict[bn], dict) else blocks_dict[bn]
                    for bn in block_names
                ]
            else:
                block_names = data.get('block_names', [])
                factors_per_block = data.get('factors_per_block', None)
            
            # Parse series dict: {series_id: {frequency: ..., blocks: [block_names], ...}}
            series_list = []
            for series_id, series_cfg in data['series'].items():
                if isinstance(series_cfg, dict):
                    # Convert block names to binary array (0/1 for each block)
                    series_blocks_names = series_cfg.get('blocks', [])
                    if isinstance(series_blocks_names, list) and series_blocks_names:
                        # Create binary array: 1 if series loads on block, 0 otherwise
                        # First block (Global) must always be 1
                        series_blocks = [0] * len(block_names)
                        series_blocks[0] = 1  # Global block is always 1
                        for block_name in series_blocks_names:
                            if block_name in block_names:
                                block_idx = block_names.index(block_name)
                                series_blocks[block_idx] = 1
                    else:
                        # Default: only global block
                        series_blocks = [1] + [0] * (len(block_names) - 1)
                    
                    series_list.append(SeriesConfig(
                        series_id=series_id,
                        series_name=series_cfg.get('series_name', series_id),
                        frequency=series_cfg.get('frequency', 'm'),
                        transformation=series_cfg.get('transformation', 'lin'),
                        blocks=series_blocks,
                        units=series_cfg.get('units', ''),
                        category=series_cfg.get('category', '')
                    ))
            
            # Extract estimation parameters if provided
            return cls(
                series=series_list,
                block_names=block_names,
                factors_per_block=factors_per_block if factors_per_block else None,
                # Estimation parameters
                ar_lag=data.get('ar_lag', 1),
                threshold=data.get('threshold', 1e-5),
                max_iter=data.get('max_iter', 5000),
                nan_method=data.get('nan_method', 2),
                nan_k=data.get('nan_k', 3),
                clock=data.get('clock', 'm'),
                # Numerical stability parameters
                clip_ar_coefficients=data.get('clip_ar_coefficients', True),
                ar_clip_min=data.get('ar_clip_min', -0.99),
                ar_clip_max=data.get('ar_clip_max', 0.99),
                warn_on_ar_clip=data.get('warn_on_ar_clip', True),
                clip_data_values=data.get('clip_data_values', True),
                data_clip_threshold=data.get('data_clip_threshold', 100.0),
                warn_on_data_clip=data.get('warn_on_data_clip', True),
                use_regularization=data.get('use_regularization', True),
                regularization_scale=data.get('regularization_scale', 1e-6),
                min_eigenvalue=data.get('min_eigenvalue', 1e-8),
                max_eigenvalue=data.get('max_eigenvalue', 1e6),
                warn_on_regularization=data.get('warn_on_regularization', True),
                use_damped_updates=data.get('use_damped_updates', True),
                damping_factor=data.get('damping_factor', 0.8),
                warn_on_damped_update=data.get('warn_on_damped_update', True)
            )
        
        # New format with series list
        if 'series' in data and isinstance(data['series'], list):
            series_list = [
                SeriesConfig(**s) if isinstance(s, dict) else s 
                for s in data['series']
            ]
            # Extract estimation parameters if provided
            return cls(
                series=series_list,
                block_names=data.get('block_names', []),
                factors_per_block=data.get('factors_per_block', None),
                # Estimation parameters
                ar_lag=data.get('ar_lag', 1),
                threshold=data.get('threshold', 1e-5),
                max_iter=data.get('max_iter', 5000),
                nan_method=data.get('nan_method', 2),
                nan_k=data.get('nan_k', 3),
                clock=data.get('clock', 'm'),
                # Numerical stability parameters
                clip_ar_coefficients=data.get('clip_ar_coefficients', True),
                ar_clip_min=data.get('ar_clip_min', -0.99),
                ar_clip_max=data.get('ar_clip_max', 0.99),
                warn_on_ar_clip=data.get('warn_on_ar_clip', True),
                clip_data_values=data.get('clip_data_values', True),
                data_clip_threshold=data.get('data_clip_threshold', 100.0),
                warn_on_data_clip=data.get('warn_on_data_clip', True),
                use_regularization=data.get('use_regularization', True),
                regularization_scale=data.get('regularization_scale', 1e-6),
                min_eigenvalue=data.get('min_eigenvalue', 1e-8),
                max_eigenvalue=data.get('max_eigenvalue', 1e6),
                warn_on_regularization=data.get('warn_on_regularization', True),
                use_damped_updates=data.get('use_damped_updates', True),
                damping_factor=data.get('damping_factor', 0.8),
                warn_on_damped_update=data.get('warn_on_damped_update', True)
            )
        
        # Direct instantiation (shouldn't happen often, but handle it)
        return cls(**data)


# DataConfig removed - data loading is application-specific, not part of generic DFM module
# Use adapters for data loading configuration

@dataclass
class AppConfig:
    """Root application configuration (for backward compatibility only).
    
    Note: DataConfig has been removed from generic DFM module.
    Data loading configuration should be handled by adapters/application layer.
    """
    model: DFMConfig  # Renamed from ModelConfig to DFMConfig
    # data: DataConfig  # Removed - not part of generic DFM module
    dfm: Optional[DFMConfig] = None  # Deprecated - use model config directly


# Register with Hydra ConfigStore following the Structured Config schema pattern
# This enables validation of YAML config files while keeping our full dataclass
# with @property methods for runtime use.
# 
# Pattern: Schema validation (from Hydra docs)
# - YAML files extend schemas via defaults list
# - Schemas provide type checking and validation
# - Runtime uses full ModelConfig with @property methods
if HYDRA_AVAILABLE and ConfigStore is not None:
    try:
        cs = ConfigStore.instance()
        if cs is None:
            raise RuntimeError("ConfigStore.instance() returned None")
        # Create schema versions without @property methods for Hydra validation
        # These match our dataclass structure exactly for schema validation.
        # We'll still use the full ModelConfig/DataConfig/DFMConfig classes with
        # @property methods at runtime via from_dict() conversion.
        from dataclasses import dataclass as schema_dataclass
        
        @schema_dataclass
        class SeriesConfigSchema:
            """Schema for SeriesConfig validation in Hydra."""
            series_id: str
            series_name: str
            frequency: str
            units: str
            transformation: str
            category: str
            blocks: List[int]
        
        @schema_dataclass
        class DFMConfigSchema:
            """Schema for unified DFMConfig validation in Hydra."""
            # Model structure
            series: List[SeriesConfigSchema]
            block_names: List[str]
            factors_per_block: Optional[List[int]] = None
            # Estimation parameters
            ar_lag: int = 1
            threshold: float = 1e-5
            max_iter: int = 5000
            nan_method: int = 2
            nan_k: int = 3
            clock: str = 'm'
        
        # Register schemas in config groups (following Hydra docs pattern)
        # These can be referenced in YAML defaults lists for validation
        # Format: defaults: [base_dfm_config, _self_]
        cs.store(group="dfm", name="base_dfm_config", node=DFMConfigSchema)
        # Also register as "model" for backward compatibility
        cs.store(group="model", name="base_model_config", node=DFMConfigSchema)
        
        # Also register standalone for direct use
        cs.store(name="dfm_config_schema", node=DFMConfigSchema)
        # Backward compatibility alias
        cs.store(name="model_config_schema", node=DFMConfigSchema)
        
    except Exception as e:
        # If registration fails, continue without schema validation
        # Configs will still work via from_dict() in scripts
        warnings.warn(f"Could not register Hydra structured config schemas: {e}. "
                     f"Configs will still work via from_dict() but without schema validation.")

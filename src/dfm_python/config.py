"""Configuration models and factory methods for DFM nowcasting.

This module provides the core configuration system for Dynamic Factor Models:
- Configuration dataclasses (DFMConfig, SeriesConfig, BlockConfig, Params)
- Factory methods for creating DFMConfig from dictionaries and Hydra configs

The configuration dataclasses define:
- Model structure (series, blocks, factors)
- Estimation parameters (EM algorithm settings)
- Numerical stability controls (regularization, clipping, damping)

Factory methods support:
- Dictionary configurations (legacy format, new format, Hydra format)
- Hydra DictConfig objects (via from_hydra())

For loading configurations from files (YAML, Spec CSV) or other sources,
see the config_sources module which provides source adapters.

Note: Source adapter classes (YamlSource, DictSource, etc.) are re-exported
from config_sources for backward compatibility.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

# Validation functions are defined at the end of this file

# Default global block name (can be overridden in config)
DEFAULT_GLOBAL_BLOCK_NAME = 'Block_Global'

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


@dataclass
class BlockConfig:
    """Configuration for a single factor block.
    
    Each block represents a group of related time series that share common
    latent factors. Blocks can have their own clock frequency, which must be
    >= the global clock frequency.
    
    Attributes
    ----------
    factors : int
        Number of latent factors in this block (typically 1)
    ar_lag : int
        Autoregressive lag order for the block-level factor (typically 1)
    clock : str
        Block-level clock frequency. Must be >= global clock.
        Series in this block must have frequency <= block clock.
    notes : str, optional
        Optional comments/description for the block
    """
    factors: int = 1
    ar_lag: int = 1
    clock: str = 'm'
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate block configuration."""
        self.clock = validate_frequency(self.clock)
        if self.factors < 1:
            raise ValueError(f"Block must have at least 1 factor, got {self.factors}")
        if self.ar_lag < 1:
            raise ValueError(f"AR lag must be at least 1, got {self.ar_lag}")


@dataclass
class SeriesConfig:
    """Configuration for a single time series.
    
    This is a generic DFM configuration - no API or database-specific fields.
    For API/database integration, implement adapters in your application layer.
    
    Attributes
    ----------
    frequency : str
        Series frequency: 'm' (monthly), 'q' (quarterly), 'sa' (semi-annual), 'a' (annual)
    transformation : str
        Transformation code: 'lin', 'pch', 'pca', etc.
    blocks : List[str] or List[int]
        Block names (strings) or block indices (ints) this series loads on.
        Must include the global block (first block).
    series_id : str, optional
        Unique identifier (auto-generated if None)
    series_name : str, optional
        Human-readable name (defaults to series_id if None)
    units : str
        Units of measurement
    category : str
        Category/group name
    """
    # Required fields (no defaults)
    frequency: str
    transformation: str
    blocks: Union[List[str], List[int]]  # Can be block names (str) or indices (int)
    # Optional fields (with defaults - must come after required fields)
    series_id: Optional[str] = None  # Auto-generated if None: "series_0", "series_1", etc.
    series_name: Optional[str] = None  # Optional metadata for display
    units: str = ""  # Optional metadata
    category: str = ""  # Optional metadata
    aggregate: Optional[str] = None  # Aggregation method (deprecated: higher frequencies than clock are not supported)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        self.frequency = validate_frequency(self.frequency)
        self.transformation = validate_transformation(self.transformation)
        # Auto-generate series_name if not provided
        if self.series_name is None and self.series_id:
            self.series_name = self.series_id
    
    def to_block_indices(self, block_names: List[str]) -> List[int]:
        """Convert block names to indices.
        
        Parameters
        ----------
        block_names : List[str]
            List of block names in order
            
        Returns
        -------
        List[int]
            Block indices (0 or 1) for each block
        """
        if not self.blocks:
            raise ValueError(f"Series {self.series_id} has no blocks specified")
        
        # If already integers, validate and return
        if isinstance(self.blocks[0], int):
            if len(self.blocks) != len(block_names):
                raise ValueError(
                    f"Series {self.series_id} has {len(self.blocks)} block indices "
                    f"but {len(block_names)} blocks defined"
                )
            return list(self.blocks)
        
        # Convert block names to indices
        # Normalize block names (handle both underscore and hyphen variants)
        block_names_normalized = {name.replace('-', '_'): name for name in block_names}
        blocks_normalized = {name.replace('-', '_'): name for name in self.blocks}
        
        block_indices = [0] * len(block_names)
        for block_name in self.blocks:
            block_name_norm = block_name.replace('-', '_')
            if block_name_norm not in block_names_normalized:
                # Try exact match first
                if block_name in block_names:
                    block_indices[block_names.index(block_name)] = 1
                else:
                    raise ValueError(
                        f"Series {self.series_id} references block '{block_name}' "
                        f"which is not in block_names: {block_names}. "
                        f"Note: Block names are case-sensitive and must match exactly."
                    )
            else:
                # Use normalized name to find index
                actual_block_name = block_names_normalized[block_name_norm]
                block_indices[block_names.index(actual_block_name)] = 1
        
        return block_indices


@dataclass
class Params:
    """Global estimation parameters for DFM (main settings only).
    
    This dataclass contains only the global knobs that control the EM algorithm
    and numerical stability. It does NOT include series or block definitions.
    Use this with spec CSV files to provide main settings while the spec CSV
    defines all series and their block memberships.
    
    All parameters have defaults matching the standard YAML configuration.
    
    Attributes
    ----------
    ar_lag : int
        AR lag for factor transition equation (typically 1)
    threshold : float
        EM convergence threshold (default: 1e-5)
    max_iter : int
        Maximum EM iterations (default: 5000)
    nan_method : int
        Missing data handling method (1-5, default: 2 = spline interpolation).
        This preprocessing step is followed by Kalman Filter-based missing data
        handling during DFM estimation, following standard practice in state-space
        models (Mariano & Murasawa 2003, Harvey 1989).
    nan_k : int
        Spline parameter for NaN interpolation (default: 3 for cubic spline)
    clock : str
        Base frequency for all latent factors (default: 'm' for monthly)
    clip_ar_coefficients : bool
        Enable AR coefficient clipping for stationarity (default: True)
    ar_clip_min : float
        Minimum AR coefficient (default: -0.99)
    ar_clip_max : float
        Maximum AR coefficient (default: 0.99)
    warn_on_ar_clip : bool
        Warn when AR coefficients are clipped (default: True)
    clip_data_values : bool
        Enable clipping of extreme data values (default: True)
    data_clip_threshold : float
        Clip values beyond this many standard deviations (default: 100.0)
    warn_on_data_clip : bool
        Warn when data values are clipped (default: True)
    use_regularization : bool
        Enable regularization for numerical stability (default: True)
    regularization_scale : float
        Scale factor for ridge regularization (default: 1e-5)
    min_eigenvalue : float
        Minimum eigenvalue for positive definite matrices (default: 1e-8)
    max_eigenvalue : float
        Maximum eigenvalue cap (default: 1e6)
    warn_on_regularization : bool
        Warn when regularization is applied (default: True)
    use_damped_updates : bool
        Enable damped updates when likelihood decreases (default: True)
    damping_factor : float
        Damping factor: 0.8 = 80% new, 20% old (default: 0.8)
    warn_on_damped_update : bool
        Warn when damped updates are used (default: True)
    augment_idio : bool
        Enable state augmentation with idiosyncratic components (default: True)
    augment_idio_slow : bool
        Enable tent-length chains for slower-frequency series (default: True)
    idio_rho0 : float
        Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var : float
        Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
    
    Examples
    --------
    >>> from dfm_python.config import Params
    >>> params = Params(max_iter=100, threshold=1e-4)
    >>> dfm.from_spec('data/spec.csv', params=params)
    """
    # Estimation parameters
    ar_lag: int = 1
    threshold: float = 1e-5
    max_iter: int = 5000
    nan_method: int = 2
    nan_k: int = 3
    clock: str = 'm'
    
    # Numerical stability - AR clipping
    clip_ar_coefficients: bool = True
    ar_clip_min: float = -0.99
    ar_clip_max: float = 0.99
    warn_on_ar_clip: bool = True
    
    # Numerical stability - Data clipping
    clip_data_values: bool = True
    data_clip_threshold: float = 100.0
    warn_on_data_clip: bool = True
    
    # Numerical stability - Regularization
    use_regularization: bool = True
    regularization_scale: float = 1e-5
    min_eigenvalue: float = 1e-8
    max_eigenvalue: float = 1e6
    warn_on_regularization: bool = True
    
    # Numerical stability - Damped updates
    use_damped_updates: bool = True
    damping_factor: float = 0.8
    warn_on_damped_update: bool = True
    
    # Idiosyncratic component augmentation
    augment_idio: bool = True  # Enable state augmentation with idiosyncratic components (default: True)
    augment_idio_slow: bool = True  # Enable tent-length chains for slower-frequency series (default: True)
    idio_rho0: float = 0.1  # Initial AR coefficient for idiosyncratic components (default: 0.1)
    idio_min_var: float = 1e-8  # Minimum variance for idiosyncratic innovation covariance (default: 1e-8)
    
    def __post_init__(self):
        """Validate parameters."""
        self.clock = validate_frequency(self.clock)
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {self.max_iter}")
        if self.ar_clip_min >= self.ar_clip_max:
            raise ValueError(f"ar_clip_min ({self.ar_clip_min}) must be < ar_clip_max ({self.ar_clip_max})")
        if not (-1 < self.ar_clip_min < 1):
            raise ValueError(f"ar_clip_min must be in (-1, 1), got {self.ar_clip_min}")
        if not (-1 < self.ar_clip_max < 1):
            raise ValueError(f"ar_clip_max must be in (-1, 1), got {self.ar_clip_max}")


@dataclass
class DFMConfig:
    """Unified DFM configuration - model structure + estimation parameters.
    
    This is the single configuration class for the DFM module, combining:
    - Model structure (what series, blocks, factors)
    - Estimation parameters (how to run EM algorithm)
    - Numerical stability controls (regularization, damping, clipping)
    
    The configuration can be built from:
    - Main settings (estimation parameters) from config/default.yaml
    - Series definitions from config/series/default.yaml or CSV
    - Block definitions from config/blocks/default.yaml
    """
    # ========================================================================
    # Model Structure (WHAT - defines the model)
    # ========================================================================
    series: List[SeriesConfig]  # Series specifications
    blocks: Dict[str, BlockConfig]  # Block configurations (block_name -> BlockConfig)
    block_names: List[str] = field(init=False)  # Block names in order (derived from blocks dict)
    factors_per_block: List[int] = field(init=False)  # Number of factors per block (derived from blocks)
    
    # ========================================================================
    # Estimation Parameters (HOW - controls the algorithm)
    # ========================================================================
    ar_lag: int = 1  # Number of lags in AR transition equation (lookback window)
    threshold: float = 1e-5  # EM convergence threshold
    max_iter: int = 5000  # Maximum EM iterations
    nan_method: int = 2  # Missing data handling method (1-5). Preprocessing step before Kalman Filter-based handling
    nan_k: int = 3  # Spline parameter for NaN interpolation (cubic spline)
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
    
    # ========================================================================
    # Internal cache (not user-configurable)
    # ========================================================================
    _cached_blocks: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Validate blocks structure and consistency.
        
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
        # Import frequency hierarchy for validation
        from .utils import FREQUENCY_HIERARCHY
        
        if not self.series:
            raise ValueError(
                "DFM configuration must contain at least one series. "
                "Please add series definitions to your configuration."
            )
        
        if not self.blocks:
            raise ValueError(
                "DFM configuration must contain at least one block. "
                "Please add block definitions to your configuration."
            )
        
        # Derive block_names and factors_per_block from blocks dict
        # Ensure global block (first block) is present
        # Find the global block (first block in order or named 'Block_Global')
        block_names_list = list(self.blocks.keys())
        global_block_name = None
        
        # Try to find Block_Global first (convention)
        if DEFAULT_GLOBAL_BLOCK_NAME in self.blocks:
            global_block_name = DEFAULT_GLOBAL_BLOCK_NAME
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
                         [self.blocks[name].factors for name in self.block_names])
        
        # Validate global clock
        self.clock = validate_frequency(self.clock)
        global_clock_hierarchy = FREQUENCY_HIERARCHY.get(self.clock, 3)
        
        # Validate block clocks (must be >= global clock)
        for block_name, block_cfg in self.blocks.items():
            block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_cfg.clock, 3)
            if block_clock_hierarchy < global_clock_hierarchy:
                raise ValueError(
                    f"Block '{block_name}' has clock '{block_cfg.clock}' which is faster than "
                    f"global clock '{self.clock}'. Block clocks must be >= global clock. "
                    f"Suggested fix: change block '{block_name}' clock to '{self.clock}' or slower, "
                    f"or set global clock to '{block_cfg.clock}' or faster."
                )
        
        # Auto-generate series_id if not provided and convert blocks to indices
        n_blocks = len(self.block_names)
        for i, s in enumerate(self.series):
            if s.series_id is None:
                s.series_id = f"series_{i}"
            if s.series_name is None:
                s.series_name = s.series_id
            
            # Convert block names to indices if needed
            if isinstance(s.blocks, list) and len(s.blocks) > 0:
                if isinstance(s.blocks[0], str):
                    # Convert block names to indices
                    block_indices = s.to_block_indices(self.block_names)
                    object.__setattr__(s, 'blocks', block_indices)
        
        # Validate all series have correct number of blocks
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
        
        # Validate block clock constraints: series frequency <= block clock
        for i, s in enumerate(self.series):
            series_freq_hierarchy = FREQUENCY_HIERARCHY.get(s.frequency, 3)
            
            # Find which blocks this series loads on
            for block_idx, loads_on_block in enumerate(s.blocks):
                if loads_on_block == 1:
                    block_name = self.block_names[block_idx]
                    block_cfg = self.blocks[block_name]
                    block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_cfg.clock, 3)
                    
                    # Series frequency must be <= block clock (slower or equal)
                    if series_freq_hierarchy < block_clock_hierarchy:
                        # Suggest valid frequencies for the series
                        valid_freqs = [freq for freq, hier in FREQUENCY_HIERARCHY.items() 
                                      if hier >= block_clock_hierarchy]
                        valid_freqs_str = ', '.join(sorted(valid_freqs))
                        raise ValueError(
                            f"Series '{s.series_id}' has frequency '{s.frequency}' which is faster than "
                            f"block '{block_name}' clock '{block_cfg.clock}'. "
                            f"Series in a block must have frequency <= block clock. "
                            f"Suggested fix: change series frequency to one of [{valid_freqs_str}] "
                            f"(slower or equal to block clock '{block_cfg.clock}'), "
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
    
    # ========================================================================
    # Helper Methods (snake_case - recommended)
    # ========================================================================
    
    def get_series_ids(self) -> List[str]:
        """Get list of series IDs (snake_case - recommended)."""
        return [s.series_id if s.series_id is not None else f"series_{i}" 
                for i, s in enumerate(self.series)]
    
    def get_series_names(self) -> List[str]:
        """Get list of series names (snake_case - recommended)."""
        return [s.series_name if s.series_name is not None else (s.series_id or f"series_{i}")
                for i, s in enumerate(self.series)]
    
    def get_frequencies(self) -> List[str]:
        """Get list of frequencies (snake_case - recommended)."""
        return [s.frequency for s in self.series]
    
    def get_blocks_array(self) -> np.ndarray:
        """Get blocks as numpy array (snake_case - recommended, cached)."""
        if self._cached_blocks is None:
            blocks_list = [s.blocks for s in self.series]
            self._cached_blocks = np.array(blocks_list, dtype=int)
        return self._cached_blocks
    
    def validate_and_report(self) -> Dict[str, Any]:
        """Validate configuration and return structured report with issues and suggestions.
        
        This method performs validation checks without raising exceptions, returning
        a structured report that can be used for debugging and user guidance.
        
        Returns
        -------
        Dict[str, Any]
            Report dictionary with keys:
            - 'valid': bool - Whether configuration is valid
            - 'errors': List[str] - List of error messages
            - 'warnings': List[str] - List of warning messages
            - 'suggestions': List[str] - List of actionable suggestions
            
        Examples
        --------
        >>> config = DFMConfig(...)
        >>> report = config.validate_and_report()
        >>> if not report['valid']:
        ...     print("Errors:", report['errors'])
        ...     print("Suggestions:", report['suggestions'])
        """
        from .utils import FREQUENCY_HIERARCHY
        
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for empty series
        if not self.series:
            report['valid'] = False
            report['errors'].append("DFM configuration must contain at least one series.")
            report['suggestions'].append("Add series definitions to your configuration.")
            return report
        
        # Check for empty blocks
        if not self.blocks:
            report['valid'] = False
            report['errors'].append("DFM configuration must contain at least one block.")
            report['suggestions'].append("Add block definitions to your configuration.")
            return report
        
        # Check frequency constraints
        global_clock_hierarchy = FREQUENCY_HIERARCHY.get(self.clock, 3)
        for i, s in enumerate(self.series):
            series_freq_hierarchy = FREQUENCY_HIERARCHY.get(s.frequency, 3)
            
            for block_idx, loads_on_block in enumerate(s.blocks):
                if loads_on_block == 1:
                    if block_idx < len(self.block_names):
                        block_name = self.block_names[block_idx]
                        block_cfg = self.blocks[block_name]
                        block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_cfg.clock, 3)
                        
                        if series_freq_hierarchy < block_clock_hierarchy:
                            valid_freqs = [freq for freq, hier in FREQUENCY_HIERARCHY.items() 
                                          if hier >= block_clock_hierarchy]
                            valid_freqs_str = ', '.join(sorted(valid_freqs))
                            report['valid'] = False
                            report['errors'].append(
                                f"Series '{s.series_id}' has frequency '{s.frequency}' which is faster than "
                                f"block '{block_name}' clock '{block_cfg.clock}'."
                            )
                            report['suggestions'].append(
                                f"For series '{s.series_id}': change frequency to one of [{valid_freqs_str}], "
                                f"or set block '{block_name}' clock to '{s.frequency}' or faster."
                            )
        
        # Check block clock constraints
        for block_name, block_cfg in self.blocks.items():
            block_clock_hierarchy = FREQUENCY_HIERARCHY.get(block_cfg.clock, 3)
            if block_clock_hierarchy < global_clock_hierarchy:
                report['valid'] = False
                report['errors'].append(
                    f"Block '{block_name}' has clock '{block_cfg.clock}' which is faster than "
                    f"global clock '{self.clock}'."
                )
                report['suggestions'].append(
                    f"Change block '{block_name}' clock to '{self.clock}' or slower, "
                    f"or set global clock to '{block_cfg.clock}' or faster."
                )
        
        # Check factors_per_block
        if any(f < 1 for f in self.factors_per_block):
            invalid_blocks = [i for i, f in enumerate(self.factors_per_block) if f < 1]
            report['valid'] = False
            report['errors'].append(
                f"factors_per_block must contain positive integers (>= 1). "
                f"Invalid values found at block indices {invalid_blocks}."
            )
            report['suggestions'].append(
                f"Set factors_per_block[{invalid_blocks[0]}] to at least 1 for block '{self.block_names[invalid_blocks[0]]}'."
            )
        
        return report
    
    # Note: Legacy PascalCase properties were removed to keep the API clean and generic.
    # Use the snake_case helper methods above.
    
    # ========================================================================
    # Factory Methods
    # ========================================================================
    
    @classmethod
    def _extract_estimation_params(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract estimation parameters from dictionary (helper to reduce duplication)."""
        return {
            'ar_lag': data.get('ar_lag', 1),
            'threshold': data.get('threshold', 1e-5),
            'max_iter': data.get('max_iter', 5000),
            'nan_method': data.get('nan_method', 2),
            'nan_k': data.get('nan_k', 3),
            'clock': data.get('clock', 'm'),
            # Numerical stability parameters
            'clip_ar_coefficients': data.get('clip_ar_coefficients', True),
            'ar_clip_min': data.get('ar_clip_min', -0.99),
            'ar_clip_max': data.get('ar_clip_max', 0.99),
            'warn_on_ar_clip': data.get('warn_on_ar_clip', True),
            'clip_data_values': data.get('clip_data_values', True),
            'data_clip_threshold': data.get('data_clip_threshold', 100.0),
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
            'idio_min_var': data.get('idio_min_var', 1e-8)
        }
    
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
            
            series_list.append(SeriesConfig(
                frequency=str(get_list_value('Frequency', i, 'm')),
                transformation=str(get_list_value('Transformation', i, 'lin')),
                blocks=series_blocks,
                series_id=get_list_value('SeriesID', i, None),
                series_name=get_list_value('SeriesName', i, None),
                units=str(get_list_value('Units', i, '')),
                category=str(get_list_value('Category', i, '')),
                aggregate=get_list_value('Aggregate', i, None)
            ))
        
        # Convert block_names to blocks dict
        block_names = data.get('BlockNames', data.get('block_names', []))
        factors_per_block = data.get('factors_per_block', None)
        
        blocks_dict = {}
        if block_names:
            for i, block_name in enumerate(block_names):
                factors = factors_per_block[i] if factors_per_block and i < len(factors_per_block) else 1
                blocks_dict[block_name] = BlockConfig(factors=factors, clock='m')
        else:
            # Default: create Block_Global if no blocks specified
            blocks_dict[DEFAULT_GLOBAL_BLOCK_NAME] = BlockConfig(factors=1, clock='m')
        
        return cls(
            series=series_list,
            blocks=blocks_dict,
            **cls._extract_estimation_params(data)
        )
    
    @classmethod
    def _from_hydra_dict(cls, data: Dict[str, Any]) -> 'DFMConfig':
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
        
        # If block_names is still empty, try to infer from series blocks
        if not block_names and 'series' in data:
            # Collect all unique block names from series
            all_blocks = set()
            for series_cfg in data['series'].values():
                if isinstance(series_cfg, dict):
                    series_blocks = series_cfg.get('blocks', [])
                    if isinstance(series_blocks, list):
                        all_blocks.update(series_blocks)
            if all_blocks:
                # Ensure global block is first (prefer DEFAULT_GLOBAL_BLOCK_NAME)
                block_names = []
                if DEFAULT_GLOBAL_BLOCK_NAME in all_blocks:
                    block_names.append(DEFAULT_GLOBAL_BLOCK_NAME)
                    all_blocks.remove(DEFAULT_GLOBAL_BLOCK_NAME)
                elif all_blocks:
                    # Use first block as global
                    first_block = sorted(all_blocks)[0]
                    block_names.append(first_block)
                    all_blocks.remove(first_block)
                block_names.extend(sorted(all_blocks))
                factors_per_block = [1] * len(block_names)
                # Create default blocks dict for later use
                blocks_dict = {name: {'factors': 1, 'clock': 'm'} for name in block_names}
        
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
                    category=series_cfg.get('category', ''),
                    aggregate=series_cfg.get('aggregate', None)  # Deprecated
                ))
        
        # Convert blocks_dict to BlockConfig dict
        blocks_dict_final = {}
        if isinstance(blocks_dict, dict) and blocks_dict:
            # Already have blocks dict from input
            for block_name, block_data in blocks_dict.items():
                if isinstance(block_data, dict):
                    blocks_dict_final[block_name] = BlockConfig(
                        factors=block_data.get('factors', 1),
                        ar_lag=block_data.get('ar_lag', 1),
                        clock=block_data.get('clock', 'm')
                    )
                else:
                    blocks_dict_final[block_name] = BlockConfig(factors=1, clock='m')
        elif block_names:
            # Create blocks dict from block_names (fallback)
            for i, block_name in enumerate(block_names):
                factors = factors_per_block[i] if factors_per_block and i < len(factors_per_block) else 1
                blocks_dict_final[block_name] = BlockConfig(factors=factors, clock='m')
        else:
            # Default: create Block_Global if no blocks specified
            blocks_dict_final[DEFAULT_GLOBAL_BLOCK_NAME] = BlockConfig(factors=1, clock='m')
        
        return cls(
            series=series_list,
            blocks=blocks_dict_final,
            **cls._extract_estimation_params(data)
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
        
        # New Hydra format: series is a dict
        if 'series' in data and isinstance(data['series'], dict):
            return cls._from_hydra_dict(data)
        
        # New format with series list
        if 'series' in data and isinstance(data['series'], list):
            series_list = [
                SeriesConfig(**s) if isinstance(s, dict) else s
                for s in data['series']
            ]
            # Handle blocks: can be dict of BlockConfig or dict with BlockConfig-like dicts
            blocks_dict = {}
            if 'blocks' in data:
                blocks_data = data['blocks']
                if isinstance(blocks_data, dict):
                    for block_name, block_cfg in blocks_data.items():
                        if isinstance(block_cfg, BlockConfig):
                            blocks_dict[block_name] = block_cfg
                        elif isinstance(block_cfg, dict):
                            blocks_dict[block_name] = BlockConfig(**block_cfg)
                        else:
                            raise ValueError(f"Invalid block config for {block_name}: {block_cfg}")
                else:
                    raise ValueError(f"blocks must be a dict, got {type(blocks_data)}")
            else:
                # If no blocks provided, infer from series
                # Find all unique block names/indices from series
                block_names_set = set()
                for s in series_list:
                    if isinstance(s, dict):
                        blocks = s.get('blocks', [])
                    else:
                        blocks = s.blocks
                    if isinstance(blocks[0], str):
                        block_names_set.update(blocks)
                    else:
                        # If indices, we need block_names to map
                        if 'block_names' in data:
                            block_names_list = data['block_names']
                            for idx in blocks:
                                if idx < len(block_names_list):
                                    block_names_set.add(block_names_list[idx])
                # Create default blocks
                for block_name in sorted(block_names_set):
                    blocks_dict[block_name] = BlockConfig(factors=1, clock=data.get('clock', 'm'))
            
            return cls(
                series=series_list,
                blocks=blocks_dict,
                **cls._extract_estimation_params(data)
            )
        
        # Direct instantiation (shouldn't happen often, but handle it)
        return cls(**data)

    @classmethod
    def from_hydra(cls, cfg: Any) -> 'DFMConfig':
        """Create DFMConfig from a Hydra DictConfig or plain dict.
        
        Parameters
        ----------
        cfg : DictConfig | dict
            Hydra DictConfig (or dict) that contains the composed configuration.
        
        Returns
        -------
        DFMConfig
            Validated configuration instance.
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
        return cls.from_dict(cfg)


# ============================================================================
# Configuration Source Adapters
# ============================================================================

# Import ConfigSource classes from separate module
from .config_sources import (
    ConfigSource,
    YamlSource,
    DictSource,
    SpecCSVSource,
    HydraSource,
    MergedConfigSource,
    make_config_source,
)

# Re-export for backward compatibility
__all__ = [
    'DFMConfig', 'SeriesConfig', 'BlockConfig', 'Params',
    'DEFAULT_GLOBAL_BLOCK_NAME',
    'ConfigSource', 'YamlSource', 'DictSource', 'SpecCSVSource',
    'HydraSource', 'MergedConfigSource', 'make_config_source',
]

# Legacy class definitions removed - now in config_sources.py
# For backward compatibility, classes are imported and re-exported above


# ============================================================================
# Configuration validation functions
# ============================================================================

# Valid frequency codes
_VALID_FREQUENCIES = {'d', 'w', 'm', 'q', 'sa', 'a'}

# Valid transformation codes
_VALID_TRANSFORMATIONS = {
    'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 
    'cch', 'cca', 'log'
}


def validate_frequency(frequency: str) -> str:
    """Validate frequency code.
    
    Parameters
    ----------
    frequency : str
        Frequency code to validate
        
    Returns
    -------
    str
        Validated frequency code (same as input if valid)
        
    Raises
    ------
    ValueError
        If frequency is not in the set of valid frequencies
        
    Examples
    --------
    >>> validate_frequency('m')
    'm'
    >>> validate_frequency('invalid')
    ValueError: Invalid frequency: invalid. Must be one of {'d', 'w', 'm', 'q', 'sa', 'a'}
    """
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(f"Invalid frequency: {frequency}. Must be one of {_VALID_FREQUENCIES}")
    return frequency


def validate_transformation(transformation: str) -> str:
    """Validate transformation code.
    
    Parameters
    ----------
    transformation : str
        Transformation code to validate
        
    Returns
    -------
    str
        Validated transformation code (same as input, even if unknown)
        
    Notes
    -----
    Unknown transformation codes trigger a warning but are not rejected,
    allowing for extensibility. The code will be used as-is, and the
    transformation logic should handle unknown codes appropriately.
    
    Examples
    --------
    >>> validate_transformation('lin')
    'lin'
    >>> validate_transformation('unknown')  # Issues warning but returns value
    'unknown'
    """
    import warnings
    if transformation not in _VALID_TRANSFORMATIONS:
        warnings.warn(f"Unknown transformation code: {transformation}. Will use untransformed data.")
    return transformation

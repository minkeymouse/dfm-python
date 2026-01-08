"""Configuration source adapters for DFM.

This module provides adapters for loading DFMConfig from various sources:
- YAML files (with Hydra/OmegaConf support)
- Dictionary configurations
- Hydra DictConfig objects
- Merged configurations from multiple sources

All adapters implement the ConfigSource protocol and return DFMConfig objects.
"""

from typing import Protocol, Optional, Dict, Any, Union, TYPE_CHECKING, NoReturn
from pathlib import Path
from dataclasses import is_dataclass, asdict

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from .schema.model import DDFMConfig
else:
    try:
        from .schema.model import DDFMConfig
    except ImportError:
        DDFMConfig = None  # Fallback if not available

from .schema.model import DFMConfig
from .constants import DEFAULT_CLOCK_FREQUENCY

# Error helper to reduce repetitive imports
def _get_config_error():
    """Get ConfigurationError (lazy import to avoid circular dependencies)."""
    from dfm_python.utils.errors import ConfigurationError
    return ConfigurationError


def _raise_config_error(message: str, details: Optional[str] = None) -> NoReturn:
    """Raise ConfigurationError with message and optional details.
    
    Parameters
    ----------
    message : str
        Error message
    details : str, optional
        Additional error details
        
    Raises
    ------
    ConfigurationError
        Always raises (NoReturn)
    """
    ConfigurationError = _get_config_error()
    if details:
        raise ConfigurationError(f"{message}\nDetails: {details}")
    else:
        raise ConfigurationError(message)


# ============================================================================
# Helper Functions for Config Parsing
# ============================================================================

def _is_dict_like(obj: Any) -> bool:
    """Check if object is dict-like (dict, OrderedDict, etc.).
    
    Parameters
    ----------
    obj : Any
        Object to check
        
    Returns
    -------
    bool
        True if object is dict-like, False otherwise
    """
    return isinstance(obj, dict) or hasattr(obj, 'keys') and hasattr(obj, '__getitem__')


def _convert_series_to_frequency_dict(
    series_data: Any,
    default_clock: str = DEFAULT_CLOCK_FREQUENCY
) -> Dict[str, str]:
    """Convert series data (dict or list) to frequency dict.
    
    Parameters
    ----------
    series_data : dict, list, or None
        Series data in various formats:
        - dict: {series_id: {frequency: ...}} or {series_id: frequency_string}
        - list: [{'series_id': ..., 'frequency': ...}, ...]
    default_clock : str
        Default clock frequency if not specified
        
    Returns
    -------
    Dict[str, str]
        Frequency dict mapping series_id to frequency
    """
    result = {}
    
    if _is_dict_like(series_data):
        for series_id, item in series_data.items():
            if _is_dict_like(item):
                freq = item.get('frequency', default_clock)
            elif isinstance(item, str):
                freq = item  # frequency as string
            else:
                freq = default_clock
            result[series_id] = freq
    elif isinstance(series_data, list):
        for item in series_data:
            if _is_dict_like(item):
                series_id = item.get('series_id', f"series_{len(result)}")
                result[series_id] = item.get('frequency', default_clock)
    
    return result


def _extract_frequency_dict(
    data: Dict[str, Any],
    clock: str = DEFAULT_CLOCK_FREQUENCY
) -> Optional[Dict[str, str]]:
    """Extract frequency dict from data, handling multiple formats.
    
    Supports:
    1. Grouped format: {'w': [series1, series2, ...], 'm': [series3, ...]}
    2. Individual format: {'series1': 'w', 'series2': 'm', ...}
    3. Legacy format: {'series': [{'series_id': ..., 'frequency': ...}, ...]}
    
    Parameters
    ----------
    data : Dict[str, Any]
        Configuration data dictionary
    clock : str, default DEFAULT_CLOCK_FREQUENCY
        Default clock frequency if not specified
        
    Returns
    -------
    Optional[Dict[str, str]]
        Frequency dict mapping column names to frequencies, or None if not found
    """
    if 'frequency' not in data:
        # Legacy: check for 'series' key
        if 'series' in data:
            return _convert_series_to_frequency_dict(data['series'], clock)
        return None
    
    freq_data = data['frequency']
    
    # Check if it's in grouped format: {'w': [...], 'm': [...]}
    if _is_dict_like(freq_data):
        from .constants import VALID_FREQUENCIES
        
        # Check if all keys are valid frequency codes (grouped format)
        keys_are_frequencies = all(
            isinstance(k, str) and k in VALID_FREQUENCIES 
            for k in freq_data.keys()
        )
        
        # Check if values are lists (grouped format)
        values_are_lists = all(
            isinstance(v, (list, tuple)) for v in freq_data.values()
        )
        
        if keys_are_frequencies and values_are_lists:
            # Grouped format: convert to individual format
            result = {}
            for freq, series_list in freq_data.items():
                if not isinstance(series_list, (list, tuple)):
                    _raise_config_error(
                        f"Frequency '{freq}' must map to a list of series names, got {type(series_list)}"
                    )
                for series_name in series_list:
                    if not isinstance(series_name, str):
                        _raise_config_error(
                            f"Series names must be strings, got {type(series_name)} in frequency '{freq}'"
                        )
                    if series_name in result:
                        _raise_config_error(
                            f"Series '{series_name}' appears in multiple frequency groups"
                        )
                    result[series_name] = freq
            return result
        else:
            # Individual format: already in correct format, validate structure
            result = {}
            for series_name, freq in freq_data.items():
                if not isinstance(series_name, str):
                    _raise_config_error(
                        f"Frequency dict keys must be strings (series names), got {type(series_name)}"
                    )
                if not isinstance(freq, str):
                    _raise_config_error(
                        f"Frequency dict values must be strings (frequency codes), got {type(freq)} for series '{series_name}'"
                    )
                result[series_name] = freq
            return result
    
    return None


def _normalize_blocks_dict(blocks_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize blocks dict to format: {"block_name": {"num_factors": int, "series": [str]}}.
    
    Parameters
    ----------
    blocks_dict : Dict[str, Any]
        Blocks dict (may contain legacy 'factors' key)
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Normalized blocks dict with 'num_factors' and 'series' keys
        
    Raises
    ------
    ConfigurationError
        If block config is invalid
    """
    ConfigurationError = _get_config_error()
    
    normalized = {}
    for block_name, block_cfg in blocks_dict.items():
        if not isinstance(block_cfg, dict):
            _raise_config_error(f"Block '{block_name}' must be a dict")
        normalized[block_name] = {
            'num_factors': block_cfg.get('num_factors', block_cfg.get('factors', 1)),
            'series': block_cfg.get('series', []) if isinstance(block_cfg.get('series'), list) else []
        }
    return normalized




class ConfigSource(Protocol):
    """Protocol for configuration sources.
    
    Any object that implements a `load()` method returning a DFMConfig
    can be used as a configuration source.
    """
    def load(self) -> DFMConfig:
        """Load and return a DFMConfig object."""
        ...


class YamlSource:
    """Load configuration from a YAML file."""
    def __init__(self, yaml_path: Union[str, Path]):
        """Initialize YAML source.
        
        Parameters
        ----------
        yaml_path : str or Path
            Path to YAML configuration file
        """
        self.yaml_path = Path(yaml_path)
    
    def load(self) -> Union[DFMConfig, 'DDFMConfig']:
        """Load configuration from YAML file.
        
        This method loads a configuration from a YAML file, automatically detecting
        the config type (DFM, DDFM, or KDFM) based on the presence of model-specific
        parameters.
        
        Returns
        -------
        DFMConfig or DDFMConfig
            Configuration object. Type is automatically detected based on config content.
            Returns DDFMConfig if DDFM-specific parameters are present, otherwise DFMConfig.
            
        Raises
        ------
        ConfigurationError
            If configuration file does not exist or cannot be loaded
        ImportError
            If omegaconf is not installed (required for YAML loading)
        ValueError
            If configuration content is invalid or cannot be parsed
            
        Examples
        --------
        >>> from pathlib import Path
        >>> source = YamlSource(Path('config/dfm_config.yaml'))
        >>> config = source.load()
        >>> assert isinstance(config, DFMConfig)
        """
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")
        
        configfile = Path(self.yaml_path)
        if not configfile.exists():
            _raise_config_error(
                f"Configuration file not found: {configfile}. "
                f"Please check the file path and ensure the file exists.",
                details=f"Absolute path: {configfile.absolute()}"
            )
        
        cfg = OmegaConf.load(configfile)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Extract main settings (estimation parameters)
        # Exclude structural/config keys, user metadata, and internal constants
        _EXCLUDED_KEYS = {
            'defaults', '_target_', '_recursive_', '_convert_', 
            'series', 'blocks', 'data', 'output', 'description', 'name', 'target', 'model',
            'nan_method', 'nan_k'  # Internal constants, not user-configurable
        }
        main_settings = {k: v for k, v in cfg_dict.items() if k not in _EXCLUDED_KEYS}
        
        # Merge model config if present (for DDFM parameters)
        model_config = cfg_dict.get('model')
        if isinstance(model_config, dict):
            main_settings.update({k: v for k, v in model_config.items() if k not in _EXCLUDED_KEYS})
        
        # Load frequency dict from main config
        frequency_dict = _extract_frequency_dict(
            cfg_dict, main_settings.get('clock', DEFAULT_CLOCK_FREQUENCY)
        )
        
        # Load blocks from main config
        blocks_dict = cfg_dict.get('blocks', {})
        if not _is_dict_like(blocks_dict):
            blocks_dict = {}
        
        # Blocks dict is required for DFM - will be validated in DFMConfig.__post_init__
        if not blocks_dict:
            _raise_config_error("blocks dict is required for DFM config")
        
        # Normalize blocks to new format
        blocks_dict = _normalize_blocks_dict(blocks_dict)
        
        # Build config dict - from_dict() handles type detection automatically
        config_dict = {
            'blocks': blocks_dict,
            **main_settings
        }
        if frequency_dict is not None:
            config_dict['frequency'] = frequency_dict
        return DFMConfig.from_dict(config_dict)


class DictSource:
    """Load configuration from a dictionary or Hydra DictConfig.
    
    Supports multiple dict formats:
    - New format: {'frequency': {'series_id': 'freq', ...}, 'blocks': {...}}
    - Legacy format: {'series': [{'series_id': ..., 'frequency': ...}], ...}
    - Hydra format: {'series': {'series_id': {...}}, 'blocks': {...}}
    - Hydra DictConfig: Automatically converts to dict
    """
    def __init__(self, mapping: Union[Dict[str, Any], 'DictConfig']):  # type: ignore
        """Initialize dictionary source.
        
        Parameters
        ----------
        mapping : dict or DictConfig
            Dictionary containing configuration data, or Hydra DictConfig object
        """
        # Handle DictConfig conversion
        if _is_dict_like(mapping):
            try:
                from omegaconf import DictConfig, OmegaConf
                if isinstance(mapping, DictConfig):
                    self.mapping = OmegaConf.to_container(mapping, resolve=True)
                else:
                    self.mapping = mapping
            except (ImportError, TypeError):
                # OmegaConf not available or not a DictConfig; assume dict
                self.mapping = mapping
        else:
            _raise_config_error(f"mapping must be a dict or DictConfig, got {type(mapping).__name__}")
    
    def load(self) -> DFMConfig:
        """Load configuration from dictionary.
        
        If the dictionary is partial (e.g., only max_iter, threshold),
        it will be merged with a minimal default config.
        """
        # Check if this is a partial config (missing frequency/blocks)
        has_frequency = 'frequency' in self.mapping and self.mapping['frequency']
        has_blocks = 'blocks' in self.mapping and self.mapping['blocks']
        
        if not has_frequency and not has_blocks:
            # This is a partial config - create a minimal default and merge
            from .constants import DEFAULT_CLOCK_FREQUENCY, DEFAULT_EM_MAX_ITER, DEFAULT_EM_THRESHOLD
            minimal_default = {
                'blocks': {},
                'clock': DEFAULT_CLOCK_FREQUENCY,
                'max_iter': DEFAULT_EM_MAX_ITER,
                'threshold': DEFAULT_EM_THRESHOLD
            }
            # Merge: mapping takes precedence
            merged = {**minimal_default, **self.mapping}
            return DFMConfig.from_dict(merged)
        
        return DFMConfig.from_dict(self.mapping)



def make_config_source(
    source: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Union[Dict[str, Any], Any]] = None,
    hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,  # type: ignore
) -> ConfigSource:
    """Create a ConfigSource adapter from various input formats.
    
    This factory function automatically selects the appropriate adapter
    based on the input type or explicit keyword arguments.
    
    Parameters
    ----------
    source : str, Path, dict, or ConfigSource, optional
        Configuration source. If a ConfigSource, returned as-is.
        If str/Path, treated as YAML file path.
        If dict, treated as dictionary config.
    yaml : str or Path, optional
        Explicit YAML file path
    mapping : dict, optional
        Explicit dictionary config
    hydra : DictConfig or dict, optional
        Explicit Hydra config
        
    Returns
    -------
    ConfigSource
        Appropriate adapter for the input
        
    Examples
    --------
    >>> # From YAML file
    >>> source = make_config_source('config/default.yaml')
    >>> 
    >>> # From dictionary
    >>> source = make_config_source({'frequency': {...}, 'clock': 'm'})
    >>> 
    >>> # Explicit keyword arguments
    >>> source = make_config_source(yaml='config/default.yaml')
    """
    # Check for explicit keyword arguments (only one allowed)
    explicit_kwargs = [k for k, v in [('yaml', yaml), ('mapping', mapping), ('hydra', hydra)] if v is not None]
    if len(explicit_kwargs) > 1:
        _raise_config_error(
            f"Only one of yaml, mapping, or hydra can be specified. "
            f"Got: {', '.join(explicit_kwargs)}."
        )
    
    # Helper: coerce to dict
    def _coerce_to_mapping(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        _raise_config_error(f"Unsupported mapping type {type(obj)}. Expected dict or dataclass.")
    
    # Handle explicit keyword arguments
    if yaml is not None:
        return YamlSource(yaml)
    if mapping is not None:
        return DictSource(_coerce_to_mapping(mapping))
    if hydra is not None:
        return DictSource(hydra)  # DictSource now handles DictConfig directly
    
    # Infer from source argument
    if source is None:
        _raise_config_error(
            "No configuration source provided. "
            "Specify source, yaml, mapping, or hydra."
        )
    
    # If already a ConfigSource, return as-is
    if hasattr(source, 'load') and callable(getattr(source, 'load')):
        return source  # type: ignore
    
    # Infer type from source
    # Check for DictConfig (before dict check, as DictConfig is a dict subclass)
    if _is_dict_like(source):
        try:
            from omegaconf import DictConfig
            if isinstance(source, DictConfig):
                return DictSource(source)
        except (ImportError, TypeError):
            pass
        return DictSource(source)
    
    if isinstance(source, DFMConfig):
        # DFMConfig can be used directly - return a simple callable wrapper
        class _ConfigWrapper:
            def __init__(self, cfg: DFMConfig):
                self._cfg = cfg
            def load(self) -> DFMConfig:
                return self._cfg
        return _ConfigWrapper(source)
    
    if isinstance(source, (str, Path)):
        return YamlSource(Path(source))
    
    if is_dataclass(source):
        return DictSource(asdict(source))
    
    _raise_config_error(f"Unsupported source type: {type(source)}. Expected str, Path, dict, ConfigSource, or DFMConfig.")




# ============================================================================
# Configuration Parsing Utilities
# ============================================================================

# Model type constants (exported for use in schema/model.py)
MODEL_TYPE_KDFM = 'kdfm'
MODEL_TYPE_DDFM = 'ddfm'
MODEL_TYPE_DFM = 'dfm'

# Model type detection patterns
_KDFM_TYPE_ALIASES = {'kdfm', 'kernelized'}
_DDFM_TYPE_ALIASES = {'ddfm', 'deep'}
_KDFM_PARAMS = {'ar_order', 'ma_order', 'structural_method', 'structural_reg_weight'}
_DDFM_PARAMS = {'encoder_layers', 'epochs', 'learning_rate', 'batch_size'}

def detect_config_type(data: Dict[str, Any]) -> str:
    """Detect config type (DFM, DDFM, or KDFM) from data dictionary.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Configuration data dictionary
        
    Returns
    -------
    str
        'kdfm', 'ddfm', or 'dfm'
        
    Raises
    ------
    ConfigurationError
        If data is not a dictionary
    """
    ConfigurationError = _get_config_error()
    
    if not _is_dict_like(data):
        _raise_config_error(f"data must be a dictionary, got {type(data).__name__}")
    
    model_type = data.get('model_type', '').lower()
    
    # Check explicit model type
    if model_type in _KDFM_TYPE_ALIASES:
        return MODEL_TYPE_KDFM
    if model_type in _DDFM_TYPE_ALIASES:
        return MODEL_TYPE_DDFM
    if model_type == MODEL_TYPE_DFM:
        return MODEL_TYPE_DFM
    
    # Check for model-specific parameters
    keys = set(data.keys())
    if keys & _KDFM_PARAMS:
        return MODEL_TYPE_KDFM
    if any(k.startswith('ddfm_') for k in keys) or (keys & _DDFM_PARAMS):
        return MODEL_TYPE_DDFM
    
    return MODEL_TYPE_DFM


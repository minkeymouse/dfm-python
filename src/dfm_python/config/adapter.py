"""Configuration source adapters for DFM nowcasting.

This module provides adapters for loading DFMConfig from various sources:
- YAML files (with Hydra/OmegaConf support)
- Dictionary configurations
- Hydra DictConfig objects
- Merged configurations from multiple sources

All adapters implement the ConfigSource protocol and return DFMConfig objects.
"""

import warnings
from typing import Protocol, Optional, Dict, Any, Union, Tuple, List, TYPE_CHECKING
from pathlib import Path
from dataclasses import is_dataclass, asdict

from .schema.model import DFMConfig
from .schema.series import SeriesConfig
# DEFAULT_BLOCK_NAME imported lazily where needed to avoid circular imports
from ..logger import get_logger
# Import ConfigurationError and DataError lazily to avoid circular imports
# They are only used in methods, not at module level

try:
    from .schema.model import DDFMConfig
except ImportError:
    DDFMConfig = None  # Fallback if not available

if TYPE_CHECKING:
    from .schema.model import DDFMConfig

_logger = get_logger(__name__)


def _load_config_defaults(
    cfg: Any,
    root_config_dir: Path,
    config_type: str
) -> Optional[Dict[str, Any]]:
    """Load config from defaults or default.yaml file.
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        Main config object from OmegaConf
    root_config_dir : Path
        Root config directory containing defaults/
    config_type : str
        Type of config: 'series' or 'blocks'
        
    Returns
    -------
    dict or None
        Loaded config dict or None if not found
        
    Raises
    ------
    ConfigurationError
        If config file exists but cannot be loaded or parsed
    """
    from omegaconf import OmegaConf
    
    # Try loading from defaults
    if 'defaults' in cfg:
        for default_item in cfg.defaults:
            default_dict = OmegaConf.to_container(default_item, resolve=False) if hasattr(default_item, 'keys') else default_item
            if isinstance(default_dict, dict) and config_type in default_dict:
                config_path = root_config_dir / config_type / f'{default_dict[config_type]}.yaml'
                if config_path.exists():
                    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    
    # Try default.yaml
    config_path = root_config_dir / config_type / 'default.yaml'
    if config_path.exists():
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    
    return None


class ConfigSource(Protocol):
    """Protocol for configuration sources.
    
    Any object that implements a `load()` method returning a DFMConfig
    can be used as a configuration source.
    """
    def load(self) -> DFMConfig:
        """Load and return a DFMConfig object."""
        ...


class YamlSource:
    """Load configuration from a YAML file.
    
    Supports Hydra-style configs with defaults for series and blocks.
    """
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
        parameters. Supports Hydra-style configs with defaults for series and blocks.
        
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
            from ...utils.errors import ConfigurationError
            raise ConfigurationError(
                f"Configuration file not found: {configfile}. "
                f"Please check the file path and ensure the file exists.",
                details=f"Absolute path: {configfile.absolute()}"
            )
        
        # Find root config directory (contains series/ or blocks/ subdirectories)
        config_dir = configfile.parent
        root_config_dir = config_dir
        while root_config_dir.parent != root_config_dir:
            if (root_config_dir / 'series').exists() or (root_config_dir / 'blocks').exists():
                break
            root_config_dir = root_config_dir.parent
        
        cfg = OmegaConf.load(configfile)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Extract main settings (estimation parameters)
        excluded_keys = {
            'defaults', '_target_', '_recursive_', '_convert_', 
            'series', 'blocks', 'data', 'output', 'description', 'name', 'target', 'model'
        }
        main_settings = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}
        
        # Merge model config if present (for DDFM parameters)
        if 'model' in cfg_dict and isinstance(cfg_dict['model'], dict):
            for key, value in cfg_dict['model'].items():
                if key not in excluded_keys:
                    main_settings[key] = value
        
        # Load series: try defaults first, then main config
        series_list = []
        series_dict = _load_config_defaults(cfg, root_config_dir, 'series')
        
        if series_dict:
            # Convert dict format to list format for parse_series_list
            series_list_data = []
            for series_id, series_data in series_dict.items():
                if isinstance(series_data, dict):
                    # Clean and add series_id
                    series_data_clean = {k: v for k, v in series_data.items() 
                                       if k not in ['transformation', 'blocks']}
                    series_data_clean['series_id'] = series_id
                    # Handle release_date
                    release_date = series_data_clean.get('release', series_data_clean.get('release_date'))
                    if release_date is not None:
                        try:
                            series_data_clean['release_date'] = int(release_date)
                        except (ValueError, TypeError):
                            pass
                    series_list_data.append(series_data_clean)
            series_list = parse_series_list(series_list_data)
        
        # If no series from defaults, get from main config
        if not series_list and 'series' in cfg_dict:
            series_data = cfg_dict['series']
            if isinstance(series_data, list):
                series_list = parse_series_list(series_data)
            elif isinstance(series_data, dict):
                # Convert dict to list format
                series_list_data = []
                for series_id, series_item in series_data.items():
                    if isinstance(series_item, dict):
                        series_item['series_id'] = series_id
                        series_list_data.append(series_item)
                    elif isinstance(series_item, str):
                        series_list_data.append({'series_id': series_id, 'frequency': 'm'})
                series_list = parse_series_list(series_list_data)
        
        # Load blocks: try defaults first, then main config
        blocks_dict = {}
        blocks_dict_raw = _load_config_defaults(cfg, root_config_dir, 'blocks')
        if blocks_dict_raw:
            blocks_dict.update(blocks_dict_raw)
        
        if 'blocks' in cfg_dict and isinstance(cfg_dict['blocks'], dict):
            blocks_dict.update(cfg_dict['blocks'])
        
        # If no blocks, create default single block
        if not blocks_dict:
            from ..functional.dfm_block import DEFAULT_BLOCK_NAME
            default_clock = main_settings.get('clock', DEFAULT_CLOCK_FREQUENCY)
            default_ar_lag = main_settings.get('ar_lag', 1)
            blocks_dict[DEFAULT_BLOCK_NAME] = {
                'factors': 1,
                'ar_lag': default_ar_lag,
                'clock': default_clock
            }
        
        # Build config dict - from_dict() handles type detection automatically
        config_dict = {
            'series': series_list,
            'blocks': blocks_dict,
            **main_settings
        }
        return DFMConfig.from_dict(config_dict)


class DictSource:
    """Load configuration from a dictionary.
    
    Supports multiple dict formats:
    - New format: {'series': [{'series_id': ..., ...}], ...}
    - New format (list): {'series': [{'series_id': ..., ...}], ...}
    - Hydra format: {'series': {'series_id': {...}}, 'blocks': {...}}
    """
    def __init__(self, mapping: Dict[str, Any]):
        """Initialize dictionary source.
        
        Parameters
        ----------
        mapping : dict
            Dictionary containing configuration data
        """
        self.mapping = mapping
    
    def load(self) -> DFMConfig:
        """Load configuration from dictionary.
        
        If the dictionary is partial (e.g., only max_iter, threshold),
        it will be merged with a minimal default config.
        """
        # Check if this is a partial config (missing series or blocks)
        has_series = 'series' in self.mapping and self.mapping['series']
        has_blocks = 'blocks' in self.mapping and self.mapping['blocks']
        
        if not has_series or not has_blocks:
            # This is a partial config - create a minimal default and merge
            minimal_default = {
                'series': [],
                'blocks': {},
                'clock': 'm',
                'max_iter': 5000,
                'threshold': 1e-5
            }
            # Merge: mapping takes precedence
            merged = {**minimal_default, **self.mapping}
            return DFMConfig.from_dict(merged)
        
        return DFMConfig.from_dict(self.mapping)


class HydraSource:
    """Load configuration from a Hydra DictConfig or dict.
    
    This adapter handles Hydra's composed configuration objects,
    converting them to DFMConfig format.
    """
    def __init__(self, cfg: Union[Dict[str, Any], 'DictConfig']):
        """Initialize Hydra source.
        
        Parameters
        ----------
        cfg : DictConfig or dict
            Hydra configuration object or dictionary in Hydra format
        """
        self.cfg = cfg
    
    def load(self) -> DFMConfig:
        """Load configuration from Hydra DictConfig/dict."""
        return DFMConfig.from_hydra(self.cfg)


class MergedConfigSource:
    """Merge multiple configuration sources.
    
    This allows combining configurations from different sources,
    e.g., base YAML config + series from another YAML or dict.
    
    The merge strategy:
    - Base config provides main settings (threshold, max_iter, clock, blocks)
    - Override config provides series definitions (replaces base series)
    - Block definitions are merged (override takes precedence)
    """
    def __init__(self, base: ConfigSource, override: ConfigSource):
        """Initialize merged config source.
        
        Parameters
        ----------
        base : ConfigSource
            Base configuration (provides main settings)
        override : ConfigSource
            Override configuration (provides series/block overrides)
        """
        self.base = base
        self.override = override
    
    def load(self) -> DFMConfig:
        """Load and merge configurations."""
        from dataclasses import asdict
        
        base_cfg = self.base.load()
        override_cfg = self.override.load()
        
        # Merge: override takes precedence
        base_dict = asdict(base_cfg)
        override_dict = asdict(override_cfg)
        
        # Merge blocks and series explicitly
        merged_blocks = {**base_dict['blocks'], **override_dict['blocks']}
        merged_series = override_dict['series'] if override_dict['series'] else base_dict['series']
        
        # Merge all other fields
        merged_dict = {**base_dict, **override_dict}
        merged_dict['blocks'] = merged_blocks
        merged_dict['series'] = merged_series
        
        return DFMConfig.from_dict(merged_dict)




def make_config_source(
    source: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Union[Dict[str, Any], Any]] = None,
    hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
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
    >>> source = make_config_source({'series': [...], 'clock': 'm'})
    >>> 
    >>> # Explicit keyword arguments
    >>> source = make_config_source(yaml='config/default.yaml')
    >>> 
    >>> # Merge YAML base + dict override
    >>> base = make_config_source(yaml='config/default.yaml')
    >>> override = make_config_source(mapping={'series': [...]})
    >>> merged = MergedConfigSource(base, override)
    """
    # Check for explicit keyword arguments (only one allowed)
    explicit_kwargs = [k for k, v in [('yaml', yaml), ('mapping', mapping), ('hydra', hydra)] if v is not None]
    if len(explicit_kwargs) > 1:
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(
            f"Only one of yaml, mapping, or hydra can be specified. "
            f"Got: {', '.join(explicit_kwargs)}. "
            f"For merging configs, use MergedConfigSource."
        )
    
    # Helper: coerce to dict
    def _coerce_to_mapping(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(f"Unsupported mapping type {type(obj)}. Expected dict or dataclass.")
    
    # Handle explicit keyword arguments
    if yaml is not None:
        return YamlSource(yaml)
    if mapping is not None:
        return DictSource(_coerce_to_mapping(mapping))
    if hydra is not None:
        return HydraSource(hydra)
    
    # Infer from source argument
    if source is None:
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(
            "No configuration source provided. "
            "Specify source, yaml, mapping, or hydra."
        )
    
    # If already a ConfigSource, return as-is
    if hasattr(source, 'load') and callable(getattr(source, 'load')):
        return source  # type: ignore
    
    # Infer type from source
    if isinstance(source, DFMConfig):
        # Wrap DFMConfig in a simple adapter
        class DFMConfigAdapter:
            def __init__(self, cfg: DFMConfig):
                self._cfg = cfg
            def load(self) -> DFMConfig:
                return self._cfg
        return DFMConfigAdapter(source)
    
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return YamlSource(path)
        elif suffix == '.csv':
            from ...utils.errors import ConfigurationError
            raise ConfigurationError(
                "Direct CSV configs are no longer supported. "
                "Please use YAML configuration files instead."
            )
        else:
            # Default to YAML if extension unclear
            return YamlSource(path)
    
    if isinstance(source, dict):
        return DictSource(source)
    
    # Try to coerce dataclass to dict
    if is_dataclass(source):
        return DictSource(asdict(source))
    
    from ...utils.errors import ConfigurationError
    raise ConfigurationError(f"Unsupported source type: {type(source)}. Expected str, Path, dict, ConfigSource, or DFMConfig.")




# ============================================================================
# Configuration Parsing Utilities
# ============================================================================

def parse_series_list(
    series_data: List[Union[Dict[str, Any], 'SeriesConfig']]
) -> List['SeriesConfig']:
    """Parse series from list format.
    
    Parameters
    ----------
    series_data : List[Union[Dict, SeriesConfig]]
        List of series configurations (dicts or SeriesConfig instances)
        
    Returns
    -------
    List[SeriesConfig]
        List of SeriesConfig instances
        
    Raises
    ------
    ConfigurationError
        If series_data is not a list or contains invalid entries
    """
    if not isinstance(series_data, list):
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(
            f"series_data must be a list, got {type(series_data).__name__}"
        )
    return [
        SeriesConfig(**s) if isinstance(s, dict) else s
        for s in series_data
    ]


def detect_config_type(data: Dict[str, Any]) -> str:
    """Detect config type (DFM, DDFM, or KDFM) from data dictionary.
    
    This helper function provides a single source of truth for config type detection.
    It checks for model-specific parameters or explicit model_type specification.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Configuration data dictionary
        
    Returns
    -------
    str
        'kdfm' if KDFM config detected, 'ddfm' if DDFM config detected, 'dfm' otherwise
        
    Detection Logic:
    - Checks if model_type is 'kdfm', 'ddfm', 'deep', or 'dfm'
    - Checks for KDFM-specific parameters: 'ar_order', 'ma_order', 'structural_method'
    - Checks for DDFM-specific parameters:
      - Keys starting with 'ddfm_'
      - Keys: 'encoder_layers', 'epochs', 'learning_rate', 'batch_size'
    - Returns appropriate type if any condition is met
        
    Raises
    ------
    ConfigurationError
        If data is not a dictionary
    """
    if not isinstance(data, dict):
        from ...utils.errors import ConfigurationError
        raise ConfigurationError(
            f"data must be a dictionary, got {type(data).__name__}"
        )
    model_type = data.get('model_type', '').lower()
    
    # Check for explicit model type
    if model_type in ('kdfm', 'kernelized'):
        return 'kdfm'
    if model_type in ('ddfm', 'deep'):
        return 'ddfm'
    if model_type == 'dfm':
        return 'dfm'
    
    # Check for KDFM-specific parameters
    has_kdfm_params = any(
        key in ['ar_order', 'ma_order', 'structural_method', 'structural_reg_weight']
        for key in data.keys()
    )
    if has_kdfm_params:
        return 'kdfm'
    
    # Check for DDFM-specific parameters
    has_ddfm_params = any(
        key.startswith('ddfm_') or 
        key in ['encoder_layers', 'epochs', 'learning_rate', 'batch_size']
        for key in data.keys()
    )
    if has_ddfm_params:
        return 'ddfm'
    
    # Default to DFM
    return 'dfm'


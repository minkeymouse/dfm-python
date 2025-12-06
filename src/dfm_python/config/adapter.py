"""Configuration source adapters for DFM nowcasting.

This module provides adapters for loading DFMConfig from various sources:
- YAML files (with Hydra/OmegaConf support)
- Dictionary configurations
- Hydra DictConfig objects
- Merged configurations from multiple sources

All adapters implement the ConfigSource protocol and return DFMConfig objects.
"""

import warnings
from typing import Protocol, Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
from pathlib import Path
from dataclasses import is_dataclass, asdict

from .schema import DFMConfig, SeriesConfig, DEFAULT_BLOCK_NAME
from ..logger import get_logger

try:
    from .schema import DDFMConfig
except ImportError:
    DDFMConfig = None  # Fallback if not available

if TYPE_CHECKING:
    from .schema import DDFMConfig

_logger = get_logger(__name__)


def _load_config_defaults(cfg, root_config_dir, config_type: str) -> Optional[dict]:
    """Load config from defaults or direct path (helper for series/blocks loading).
    
    Parameters
    ----------
    cfg : OmegaConf.DictConfig
        Main config object
    root_config_dir : Path
        Root config directory (contains series/ and blocks/ subdirectories)
    config_type : str
        Type of config to load: 'series' or 'blocks'
        
    Returns
    -------
    Optional[dict]
        Loaded config dict or None if not found
    """
    from omegaconf import OmegaConf
    
    config_dict = None
    config_loaded = False
    
    # Try loading from defaults
    if 'defaults' in cfg:
        defaults_list = cfg.defaults
        for default_item in defaults_list:
            default_dict = OmegaConf.to_container(default_item, resolve=False) if hasattr(default_item, 'keys') else default_item
            
            # Handle dict format: {'series': 'default'} or {'blocks': 'default'}
            if isinstance(default_dict, dict) and config_type in default_dict:
                config_name = default_dict[config_type]
                config_path = root_config_dir / config_type / f'{config_name}.yaml'
                if config_path.exists():
                    config_cfg = OmegaConf.load(config_path)
                    config_dict = OmegaConf.to_container(config_cfg, resolve=True)
                    config_loaded = True
                    break
    
    # If not loaded from defaults, try direct path
    if not config_loaded:
        config_path = root_config_dir / config_type / 'default.yaml'
        if config_path.exists():
            config_cfg = OmegaConf.load(config_path)
            config_dict = OmegaConf.to_container(config_cfg, resolve=True)
            config_loaded = True
    
    return config_dict if config_loaded else None


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
        
        Returns
        -------
        DFMConfig or DDFMConfig
            Configuration object. Type is automatically detected based on config content.
            Returns DDFMConfig if DDFM-specific parameters are present, otherwise DFMConfig.
        """
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")
        
        configfile = Path(self.yaml_path)
        if not configfile.exists():
            raise FileNotFoundError(f"Configuration file not found: {configfile}")
        
        config_dir = configfile.parent
        # If config file is in a subdirectory (e.g., experiment/), find the root config directory
        # Look for series/ or blocks/ directories to identify config root
        root_config_dir = config_dir
        while root_config_dir.parent != root_config_dir:  # Not at filesystem root
            if (root_config_dir / 'series').exists() or (root_config_dir / 'blocks').exists():
                break
            root_config_dir = root_config_dir.parent
        
        cfg = OmegaConf.load(configfile)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Check for model type from defaults and load model config if needed
        model_type = None
        model_config_dict = {}
        
        # Check defaults to find model config file
        if 'defaults' in cfg:
            defaults_list = cfg.defaults
            for default_item in defaults_list:
                default_dict = OmegaConf.to_container(default_item, resolve=False) if hasattr(default_item, 'keys') else default_item
                if isinstance(default_dict, dict):
                    # Check for 'override /model' or '/model' keys
                    model_key = None
                    if 'override /model' in default_dict:
                        model_key = 'override /model'
                        model_value = default_dict['override /model']
                    elif '/model' in default_dict:
                        model_key = '/model'
                        model_value = default_dict['/model']
                    
                    if model_key and model_value:
                        if model_value in ('ddfm', 'deep'):
                            model_type = 'ddfm'
                        # Load model config file
                        model_config_path = root_config_dir / 'model' / f'{model_value}.yaml'
                        if model_config_path.exists():
                            model_cfg = OmegaConf.load(model_config_path)
                            model_config_dict = OmegaConf.to_container(model_cfg, resolve=True)
                        break
        
        # Also check if model key exists in resolved config
        if not model_config_dict and 'model' in cfg_dict:
            model_value = cfg_dict['model']
            if isinstance(model_value, dict):
                model_config_dict = model_value
                # Check for DDFM-specific parameters
                if any(key.startswith('ddfm_') or key in ['encoder_layers', 'epochs', 'learning_rate', 'batch_size'] 
                       for key in model_value.keys()):
                    model_type = 'ddfm'
                elif 'model_type' in model_value:
                    model_type = model_value['model_type'].lower()
        
        # Extract main settings (estimation parameters)
        excluded_keys = {
            'defaults', '_target_', '_recursive_', '_convert_', 
            'series', 'blocks', 'data', 'output', 'description', 'name', 'target', 'model'
        }
        main_settings = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}
        
        # Merge model config parameters into main_settings (for DDFM detection)
        if model_config_dict:
            for key, value in model_config_dict.items():
                if key not in excluded_keys:
                    main_settings[key] = value
        
        # Add model_type to main_settings if detected
        if model_type:
            main_settings['model_type'] = model_type
        
        # Load series from config/series/default.yaml
        series_list = []
        series_dict = _load_config_defaults(cfg, root_config_dir, 'series')
        series_loaded = series_dict is not None
        
        # Convert series dict to SeriesConfig objects
        if series_loaded and series_dict is not None:
            for series_id, series_data in series_dict.items():
                if isinstance(series_data, dict):
                    # Parse release_date if available
                    release_date = series_data.get('release', series_data.get('release_date', None))
                    if release_date is not None:
                        try:
                            release_date = int(release_date)
                        except (ValueError, TypeError):
                            release_date = None
                    
                    series_list.append(SeriesConfig(
                        series_id=series_id,
                        series_name=series_data.get('series_name', series_id),
                        frequency=series_data.get('frequency', 'm'),
                        transformation=series_data.get('transformation', 'lin'),
                        blocks=series_data.get('blocks', []),
                        release_date=release_date
                    ))
        
        # If no series loaded from separate files, try to get from main config
        if not series_loaded and 'series' in cfg_dict:
            series_data = cfg_dict['series']
            if isinstance(series_data, list):
                for series_item in series_data:
                    if isinstance(series_item, str):
                        # Series is just a string ID - create minimal config
                        series_list.append(SeriesConfig(series_id=series_item, frequency='m', transformation='lin', blocks=[1]))
                    elif isinstance(series_item, dict):
                        series_list.append(SeriesConfig(**series_item))
            elif isinstance(series_data, dict):
                for series_id, series_item in series_data.items():
                    if isinstance(series_item, dict):
                        series_item['series_id'] = series_id
                        series_list.append(SeriesConfig(**series_item))
                    elif isinstance(series_item, str):
                        # Series value is just a string - create minimal config
                        series_list.append(SeriesConfig(series_id=series_id, frequency='m', transformation='lin', blocks=[1]))
        
        # Load block properties from model config (factors, ar_lag, clock)
        # Model config defines all blocks and their properties
        blocks_properties_from_config = {}
        
        if 'blocks' in cfg_dict:
            blocks_data = cfg_dict['blocks']
            if isinstance(blocks_data, dict):
                blocks_properties_from_config = blocks_data
        
        # Also try loading from config/blocks/default.yaml
        blocks_dict_raw = _load_config_defaults(cfg, root_config_dir, 'blocks')
        if blocks_dict_raw is not None:
            blocks_properties_from_config.update(blocks_dict_raw)
        
        # Derive blocks from series configs (new pattern: blocks discovered from series)
        # Collect all unique block names from series
        block_names_from_series = set()
        for series in series_list:
            if hasattr(series, 'blocks') and series.blocks:
                # Handle both list of strings and list of ints
                if isinstance(series.blocks, list):
                    for block in series.blocks:
                        if isinstance(block, str):
                            block_names_from_series.add(block)
        
        # If no blocks specified in series, use default single block
        # Use first block from model config if available, otherwise create default
        if not block_names_from_series:
            # Check if model config defines blocks
            if blocks_properties_from_config:
                # Use first block from model config
                first_block_name = list(blocks_properties_from_config.keys())[0]
                block_names_from_series.add(first_block_name)
            else:
                # Create default generic block
                block_names_from_series.add(DEFAULT_BLOCK_NAME)
        
        # Create block dicts for each block
        # Priority: 1) Model config properties, 2) Defaults from main_settings
        blocks_dict = {}
        default_clock = main_settings.get('clock', 'm')
        default_ar_lag = main_settings.get('ar_lag', 1)
        
        # First, add all blocks from model config (these define the block structure)
        for block_name, block_props in blocks_properties_from_config.items():
            if isinstance(block_props, dict):
                blocks_dict[block_name] = {
                    'factors': block_props.get('factors', 1),
                    'ar_lag': block_props.get('ar_lag', default_ar_lag),
                    'clock': block_props.get('clock', default_clock),
                    'notes': block_props.get('notes', None)
                }
            else:
                # If it's already a dict, use it
                blocks_dict[block_name] = block_props if isinstance(block_props, dict) else {
                    'factors': 1,
                    'ar_lag': default_ar_lag,
                    'clock': default_clock
                }
        
        # Then, ensure all blocks referenced by series exist
        for block_name in block_names_from_series:
            if block_name not in blocks_dict:
                # Create block with defaults if not in model config
                blocks_dict[block_name] = {
                    'factors': 1,
                    'ar_lag': default_ar_lag,
                    'clock': default_clock
                }
        
        # Convert block names to list for from_dict() if series use integer block indices
        # Check if series use integer indices instead of block names
        block_names_list = None
        if series_list:
            first_series_blocks = series_list[0].blocks if hasattr(series_list[0], 'blocks') else []
            if first_series_blocks and isinstance(first_series_blocks[0], int):
                # Series use integer indices, need block_names for from_dict()
                block_names_list = list(blocks_dict.keys())
        
        # Build dictionary for from_dict() which handles type detection
        # Include all main_settings (which may contain DDFM parameters from model config)
        config_dict = {
            'series': series_list,
            'blocks': blocks_dict,
            **main_settings
        }
        
        # Add block_names if series use integer indices
        if block_names_list is not None:
            config_dict['block_names'] = block_names_list
        
        # Use from_dict() which now correctly detects DDFM configs
        # The detection bug has been fixed in schema.py
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
        from dataclasses import fields
        
        base_cfg = self.base.load()
        
        # Check if override is a partial config (DictSource with partial dict)
        override_is_partial = False
        if isinstance(self.override, DictSource):
            has_series = 'series' in self.override.mapping and self.override.mapping['series']
            has_blocks = 'blocks' in self.override.mapping and self.override.mapping['blocks']
            override_is_partial = not (has_series and has_blocks)
        
        if override_is_partial:
            # Handle partial override: merge fields directly without loading full DFMConfig
            override_dict = self.override.mapping
            override_cfg = base_cfg  # Use base as template
        else:
            override_cfg = self.override.load()
        
        # Merge blocks: override takes precedence
        if override_is_partial and 'blocks' in override_dict:
            # Merge block dicts
            merged_blocks = {**base_cfg.blocks, **override_dict['blocks']}
        else:
            merged_blocks = {**base_cfg.blocks, **override_cfg.blocks}
        
        # Use override's series if provided and non-empty, otherwise use base's series
        if override_is_partial:
            if 'series' in override_dict and override_dict['series']:
                merged_series = override_dict['series']
            else:
                merged_series = base_cfg.series
        else:
            merged_series = override_cfg.series if (override_cfg.series and len(override_cfg.series) > 0) else base_cfg.series
        
        # Get all config fields (excluding derived/computed fields)
        excluded_fields = {'series', 'blocks', 'block_names', 'factors_per_block', '_cached_blocks'}
        base_settings = {
            field.name: getattr(base_cfg, field.name)
            for field in fields(DFMConfig)
            if field.name not in excluded_fields
        }
        
        # Override settings from override_cfg or override_dict
        if override_is_partial:
            override_settings = {
                field.name: override_dict.get(field.name, getattr(base_cfg, field.name))
                for field in fields(DFMConfig)
                if field.name not in excluded_fields
            }
        else:
            override_settings = {
                field.name: getattr(override_cfg, field.name)
                for field in fields(DFMConfig)
                if field.name not in excluded_fields
                and hasattr(override_cfg, field.name)
            }
        
        # Merge settings: base + override (override takes precedence)
        merged_settings = {**base_settings, **override_settings}
        
        # Create merged config: merged settings + merged series + merged blocks
        return DFMConfig(
            series=merged_series,
            blocks=merged_blocks,
            **merged_settings
        )




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
        raise ValueError(
            f"Only one of yaml, mapping, or hydra can be specified. "
            f"Got: {', '.join(explicit_kwargs)}. "
            f"For merging configs, use MergedConfigSource."
        )
    
    # Helper: coerce arbitrary object with attributes into dict
    def _coerce_to_mapping(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            try:
                return dict(vars(obj))
            except Exception:
                pass
        raise TypeError(
            f"Config loading failed: unsupported mapping type {type(obj)}. "
            f"Please provide a dict, dataclass instance, or an object with attributes."
        )
    
    # Handle explicit keyword arguments
    if yaml is not None:
        return YamlSource(yaml)
    if mapping is not None:
        return DictSource(_coerce_to_mapping(mapping))
    if hydra is not None:
        return HydraSource(hydra)
    
    # Infer from source argument
    if source is None:
        raise ValueError(
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
            raise ValueError(
                "Direct CSV configs are no longer supported. "
                "Please use YAML configuration files instead."
            )
        else:
            # Default to YAML if extension unclear
            return YamlSource(path)
    
    if isinstance(source, dict):
        return DictSource(source)
    # Accept objects that can be coerced into dict (dataclass or attribute bag)
    try:
        coerced = _coerce_to_mapping(source)
        return DictSource(coerced)
    except Exception:
        pass
    
    raise TypeError(
        f"Unsupported source type: {type(source)}. "
        f"Expected str, Path, dict, ConfigSource, or DFMConfig."
    )




# ============================================================================
# Hydra ConfigStore Registration (optional - only if Hydra is available)
# ============================================================================

try:
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    ConfigStore = None

if HYDRA_AVAILABLE and ConfigStore is not None:
    try:
        cs = ConfigStore.instance()
        if cs is not None:
            from dataclasses import dataclass as schema_dataclass
            from typing import List as ListType
            
            @schema_dataclass
            class SeriesConfigSchema:
                """Schema for SeriesConfig validation in Hydra."""
                series_id: str
                series_name: str
                frequency: str
                transformation: str
                blocks: ListType[int]
                units: Optional[str] = None  # Optional, for display only
            
            @schema_dataclass
            class DFMConfigSchema:
                """Schema for unified DFMConfig validation in Hydra."""
                series: ListType[SeriesConfigSchema]
                block_names: ListType[str]
                factors_per_block: Optional[ListType[int]] = None
                ar_lag: int = 1
                threshold: float = 1e-5
                max_iter: int = 5000
                nan_method: int = 2
                nan_k: int = 3
                clock: str = 'm'
            
            # Register schemas
            cs.store(group="dfm", name="base_dfm_config", node=DFMConfigSchema)
            cs.store(group="model", name="base_model_config", node=DFMConfigSchema)
            cs.store(name="dfm_config_schema", node=DFMConfigSchema)
            cs.store(name="model_config_schema", node=DFMConfigSchema)
            
    except Exception as e:
        warnings.warn(f"Could not register Hydra structured config schemas: {e}. "
                     f"Configs will still work via from_dict() but without schema validation.")


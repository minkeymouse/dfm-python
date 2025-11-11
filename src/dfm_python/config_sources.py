"""Configuration source adapters for DFM nowcasting.

This module provides adapters for loading DFMConfig from various sources:
- YAML files (with Hydra/OmegaConf support)
- Dictionary configurations
- Spec CSV files (series definitions)
- Hydra DictConfig objects
- Merged configurations from multiple sources

All adapters implement the ConfigSource protocol and return DFMConfig objects.
"""

from typing import Protocol, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import is_dataclass, asdict

from .config import DFMConfig, SeriesConfig, BlockConfig, DEFAULT_GLOBAL_BLOCK_NAME


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
    
    def load(self) -> DFMConfig:
        """Load configuration from YAML file."""
        try:
            from omegaconf import OmegaConf
        except ImportError:
            raise ImportError("omegaconf is required for YAML config loading. Install with: pip install omegaconf")
        
        configfile = Path(self.yaml_path)
        if not configfile.exists():
            raise FileNotFoundError(f"Configuration file not found: {configfile}")
        
        config_dir = configfile.parent
        cfg = OmegaConf.load(configfile)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Extract main settings (estimation parameters)
        excluded_keys = {'defaults', '_target_', '_recursive_', '_convert_'}
        main_settings = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}
        
        # Load series from config/series/default.yaml
        series_list = []
        series_loaded = False
        series_dict = None
        
        if 'defaults' in cfg:
            defaults_list = cfg.defaults
            for default_item in defaults_list:
                default_dict = OmegaConf.to_container(default_item, resolve=False) if hasattr(default_item, 'keys') else default_item
                
                # Handle dict format: {'series': 'default'}
                if isinstance(default_dict, dict) and 'series' in default_dict:
                    series_config_name = default_dict['series']
                    series_config_path = config_dir / 'series' / f'{series_config_name}.yaml'
                    if series_config_path.exists():
                        series_cfg = OmegaConf.load(series_config_path)
                        series_dict = OmegaConf.to_container(series_cfg, resolve=True)
                        series_loaded = True
                        break
        
        # If not loaded from defaults, try direct path
        if not series_loaded:
            series_config_path = config_dir / 'series' / 'default.yaml'
            if series_config_path.exists():
                series_cfg = OmegaConf.load(series_config_path)
                series_dict = OmegaConf.to_container(series_cfg, resolve=True)
                series_loaded = True
        
        # Convert series dict to SeriesConfig objects
        if series_loaded and series_dict is not None:
            for series_id, series_data in series_dict.items():
                if isinstance(series_data, dict):
                    series_list.append(SeriesConfig(
                        series_id=series_id,
                        series_name=series_data.get('series_name', series_id),
                        frequency=series_data.get('frequency', 'm'),
                        transformation=series_data.get('transformation', 'lin'),
                        category=series_data.get('category', ''),
                        units=series_data.get('units', ''),
                        blocks=series_data.get('blocks', [])
                    ))
        
        # If no series loaded from separate files, try to get from main config
        if not series_loaded and 'series' in cfg_dict:
            series_data = cfg_dict['series']
            if isinstance(series_data, list):
                for series_item in series_data:
                    series_list.append(SeriesConfig(**series_item))
            elif isinstance(series_data, dict):
                for series_id, series_item in series_data.items():
                    if isinstance(series_item, dict):
                        series_item['series_id'] = series_id
                        series_list.append(SeriesConfig(**series_item))
        
        # Load blocks from config/blocks/default.yaml
        blocks_dict = {}
        blocks_loaded = False
        blocks_dict_raw = None
        
        if 'defaults' in cfg:
            defaults_list = cfg.defaults
            for default_item in defaults_list:
                default_dict = OmegaConf.to_container(default_item, resolve=False) if hasattr(default_item, 'keys') else default_item
                
                # Handle dict format: {'blocks': 'default'}
                if isinstance(default_dict, dict) and 'blocks' in default_dict:
                    blocks_config_name = default_dict['blocks']
                    blocks_config_path = config_dir / 'blocks' / f'{blocks_config_name}.yaml'
                    if blocks_config_path.exists():
                        blocks_cfg = OmegaConf.load(blocks_config_path)
                        blocks_dict_raw = OmegaConf.to_container(blocks_cfg, resolve=True)
                        blocks_loaded = True
                        break
        
        # If not loaded from defaults, try direct path
        if not blocks_loaded:
            blocks_config_path = config_dir / 'blocks' / 'default.yaml'
            if blocks_config_path.exists():
                blocks_cfg = OmegaConf.load(blocks_config_path)
                blocks_dict_raw = OmegaConf.to_container(blocks_cfg, resolve=True)
                blocks_loaded = True
        
        # Convert blocks dict to BlockConfig objects
        if blocks_loaded and blocks_dict_raw is not None:
            for block_name, block_data in blocks_dict_raw.items():
                if isinstance(block_data, dict):
                    blocks_dict[block_name] = BlockConfig(
                        factors=block_data.get('factors', 1),
                        ar_lag=block_data.get('ar_lag', 1),
                        clock=block_data.get('clock', 'm'),
                        notes=block_data.get('notes', None)
                    )
        
        # If no blocks loaded from separate files, try to get from main config
        if not blocks_loaded and 'blocks' in cfg_dict:
            blocks_data = cfg_dict['blocks']
            if isinstance(blocks_data, dict):
                for block_name, block_item in blocks_data.items():
                    if isinstance(block_item, dict):
                        blocks_dict[block_name] = BlockConfig(**block_item)
        
        # Ensure at least one block exists
        if not blocks_dict:
            blocks_dict[DEFAULT_GLOBAL_BLOCK_NAME] = BlockConfig(factors=1, clock=main_settings.get('clock', 'm'))
        
        return DFMConfig(
            series=series_list,
            blocks=blocks_dict,
            **main_settings
        )


class DictSource:
    """Load configuration from a dictionary.
    
    Supports multiple dict formats:
    - Legacy format: {'SeriesID': [...], 'Frequency': [...], ...}
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


class SpecCSVSource:
    """Load series definitions from a spec CSV file.
    
    This source reads series definitions and block memberships from a CSV.
    It constructs a valid DFMConfig with sensible defaults for other
    parameters (e.g., clock='m', threshold=1e-5, max_iter=5000, factors=1).
    
    For custom main settings (threshold, max_iter, clock, per-block factors),
    you can still merge with a base config via MergedConfigSource or
    dfm.load_config(base=..., override=spec_path).
    """
    def __init__(self, spec_path: Union[str, Path]):
        """Initialize spec CSV source.
        
        Parameters
        ----------
        spec_path : str or Path
            Path to spec CSV file
        """
        self.spec_path = Path(spec_path)
    
    def load(self) -> DFMConfig:
        """Load series definitions from spec CSV."""
        import pandas as pd
        from .data import _load_config_from_dataframe
        
        specfile = Path(self.spec_path)
        if not specfile.exists():
            raise FileNotFoundError(f"Spec file not found: {specfile}")
        
        try:
            df_full = pd.read_csv(specfile)
        except Exception as e:
            raise ValueError(f"Failed to read spec file {specfile}: {e}")
        
        return _load_config_from_dataframe(df_full)


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
    e.g., base YAML config + series from Spec CSV.
    
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
    spec: Optional[Union[str, Path]] = None,
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
    spec : str or Path, optional
        Explicit spec CSV file path
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
    >>> source = make_config_source(spec='data/spec.csv')
    >>> 
    >>> # Merge YAML base + Spec CSV series
    >>> base = make_config_source(yaml='config/default.yaml')
    >>> override = make_config_source(spec='data/spec.csv')
    >>> merged = MergedConfigSource(base, override)
    """
    # Check for explicit keyword arguments (only one allowed)
    explicit_kwargs = [k for k, v in [('yaml', yaml), ('mapping', mapping), ('spec', spec), ('hydra', hydra)] if v is not None]
    if len(explicit_kwargs) > 1:
        raise ValueError(
            f"Only one of yaml, mapping, spec, or hydra can be specified. "
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
            f"Unsupported mapping type: {type(obj)}. "
            f"Provide a dict, dataclass instance, or an object with attributes."
        )
    
    # Handle explicit keyword arguments
    if yaml is not None:
        return YamlSource(yaml)
    if mapping is not None:
        return DictSource(_coerce_to_mapping(mapping))
    if spec is not None:
        return SpecCSVSource(spec)
    if hydra is not None:
        return HydraSource(hydra)
    
    # Infer from source argument
    if source is None:
        raise ValueError(
            "No configuration source provided. "
            "Specify source, yaml, mapping, spec, or hydra."
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
            return SpecCSVSource(path)
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
                units: str
                transformation: str
                category: str
                blocks: ListType[int]
            
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
        import warnings
        warnings.warn(f"Could not register Hydra structured config schemas: {e}. "
                     f"Configs will still work via from_dict() but without schema validation.")


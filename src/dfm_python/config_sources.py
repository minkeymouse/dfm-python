"""Configuration source adapters for DFM nowcasting.

This module provides adapters for loading DFMConfig from various sources:
- YAML files (with Hydra/OmegaConf support)
- Dictionary configurations
- Spec CSV files (series definitions)
- Hydra DictConfig objects
- Merged configurations from multiple sources

All adapters implement the ConfigSource protocol and return DFMConfig objects.
"""

import warnings
from typing import Protocol, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from dataclasses import is_dataclass, asdict

from .config import DFMConfig, SeriesConfig, BlockConfig, DEFAULT_GLOBAL_BLOCK_NAME
import polars as pl


def _load_config_from_defaults(cfg, root_config_dir, config_type: str) -> Optional[dict]:
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
        # If config file is in a subdirectory (e.g., experiment/), find the root config directory
        # Look for series/ or blocks/ directories to identify config root
        root_config_dir = config_dir
        while root_config_dir.parent != root_config_dir:  # Not at filesystem root
            if (root_config_dir / 'series').exists() or (root_config_dir / 'blocks').exists():
                break
            root_config_dir = root_config_dir.parent
        
        cfg = OmegaConf.load(configfile)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Extract main settings (estimation parameters)
        excluded_keys = {'defaults', '_target_', '_recursive_', '_convert_'}
        main_settings = {k: v for k, v in cfg_dict.items() if k not in excluded_keys}
        
        # Load series from config/series/default.yaml
        series_list = []
        series_dict = _load_config_from_defaults(cfg, root_config_dir, 'series')
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
                    series_list.append(SeriesConfig(**series_item))
            elif isinstance(series_data, dict):
                for series_id, series_item in series_data.items():
                    if isinstance(series_item, dict):
                        series_item['series_id'] = series_id
                        series_list.append(SeriesConfig(**series_item))
        
        # Load blocks from config/blocks/default.yaml
        blocks_dict = {}
        blocks_dict_raw = _load_config_from_defaults(cfg, root_config_dir, 'blocks')
        blocks_loaded = blocks_dict_raw is not None
        
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


def _write_series_blocks_yaml(
    config: DFMConfig,
    output_dir: Path,
    series_filename: str,
    blocks_filename: str
) -> Tuple[Path, Path]:
    """Write series and block YAML files for a given configuration."""
    series_dir = output_dir / 'series'
    series_dir.mkdir(parents=True, exist_ok=True)
    series_path = series_dir / f'{series_filename}.yaml'
    
    blocks_dir = output_dir / 'blocks'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = blocks_dir / f'{blocks_filename}.yaml'
    
    series_dict = {}
    block_names = config.block_names
    
    for series in config.series:
        if isinstance(series.blocks, list) and len(series.blocks) > 0:
            if isinstance(series.blocks[0], int):
                blocks_names = [block_names[i] for i, val in enumerate(series.blocks) if val != 0]
            else:
                blocks_names = list(series.blocks)
        else:
            blocks_names = ['Block_Global']
        
        series_entry = {
            'series_name': series.series_name,
            'frequency': series.frequency,
            'transformation': series.transformation,
            'blocks': blocks_names
        }
        if series.release_date is not None:
            series_entry['release'] = series.release_date
        series_dict[series.series_id] = series_entry
    
    blocks_dict = {}
    for block_name, block_config in config.blocks.items():
        block_entry = {
            'factors': block_config.factors,
            'ar_lag': block_config.ar_lag,
            'clock': block_config.clock
        }
        if block_config.notes:
            block_entry['notes'] = block_config.notes
        blocks_dict[block_name] = block_entry
    
    def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
        try:
            import yaml  # type: ignore
            yaml_kwargs = {'default_flow_style': False, 'sort_keys': False, 'allow_unicode': True}
            import io
            stream = io.StringIO()
            yaml.dump(payload, stream, **yaml_kwargs)
            yaml_content = stream.getvalue()
            lines = yaml_content.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines):
                if line and not line.startswith(' ') and ':' in line and not line.strip().startswith('#'):
                    if i > 0 and formatted_lines and formatted_lines[-1].strip():
                        formatted_lines.append('')
                formatted_lines.append(line)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_lines))
        except ImportError:
            try:
                from omegaconf import OmegaConf  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Either PyYAML or omegaconf is required for YAML generation. "
                    "Install with: pip install pyyaml or pip install omegaconf"
                ) from exc
            cfg = OmegaConf.create(payload)
            OmegaConf.save(cfg, path)
    
    _write_yaml(series_path, series_dict)
    _write_yaml(blocks_path, blocks_dict)
    
    return series_path, blocks_path


def from_spec(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    series_filename: Optional[str] = None,
    blocks_filename: Optional[str] = None
) -> Tuple[Path, Path]:
    """Convert spec CSV file to YAML configuration files."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if output_dir is None:
        output_dir = csv_path.parent.parent / 'config'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_basename = csv_path.stem
    series_filename = series_filename or csv_basename
    blocks_filename = blocks_filename or csv_basename
    
    try:
        df = pl.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")
    
    config = _load_config_from_dataframe(df)
    return _write_series_blocks_yaml(config, output_dir, series_filename, blocks_filename)
    # Prepare series YAML content
    series_dir = output_dir / 'series'
    series_dir.mkdir(parents=True, exist_ok=True)
    series_path = series_dir / f'{series_filename}.yaml'
    
    series_dict = {}
    # Get block names in order (from config.block_names)
    block_names = config.block_names
    
    for series in config.series:
        # Convert blocks from indices to block names if needed
        if isinstance(series.blocks, list) and len(series.blocks) > 0:
            if isinstance(series.blocks[0], int):
                # Blocks are indices - convert to block names
                blocks_names = [block_names[i] for i, val in enumerate(series.blocks) if val != 0]
            else:
                # Blocks are already names
                blocks_names = list(series.blocks)
        else:
            blocks_names = ['Block_Global']  # Default
        
        series_entry = {
            'series_name': series.series_name,
            'frequency': series.frequency,
            'transformation': series.transformation,
            'blocks': blocks_names
        }
        if series.release_date is not None:
            series_entry['release'] = series.release_date
        series_dict[series.series_id] = series_entry
    
    # Prepare blocks YAML content
    blocks_dir = output_dir / 'blocks'
    blocks_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = blocks_dir / f'{blocks_filename}.yaml'
    
    blocks_dict = {}
    for block_name, block_config in config.blocks.items():
        block_entry = {
            'factors': block_config.factors,
            'ar_lag': block_config.ar_lag,
            'clock': block_config.clock
        }
        if block_config.notes:
            block_entry['notes'] = block_config.notes
        blocks_dict[block_name] = block_entry
    
    # Write YAML files
    try:
        try:
            import yaml
            yaml_lib = yaml
            yaml_kwargs = {'default_flow_style': False, 'sort_keys': False, 'allow_unicode': True}
        except ImportError:
            try:
                from omegaconf import OmegaConf
                yaml_lib = OmegaConf
                yaml_kwargs = {}
            except ImportError:
                raise ImportError(
                    "Either PyYAML or omegaconf is required for YAML generation. "
                    "Install with: pip install pyyaml or pip install omegaconf"
                )
        
        # Write series YAML
        if hasattr(yaml_lib, 'dump'):
            # PyYAML - add blank lines between series for better readability
            import io
            stream = io.StringIO()
            yaml_lib.dump(series_dict, stream, **yaml_kwargs)
            yaml_content = stream.getvalue()
            
            # Add blank line before each series key (except the first one)
            lines = yaml_content.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines):
                # If line starts a new series (starts with a key and colon, not indented)
                if line and not line.startswith(' ') and ':' in line and not line.strip().startswith('#'):
                    # Add blank line before this line (except for the first series)
                    if i > 0 and formatted_lines and formatted_lines[-1].strip():
                        formatted_lines.append('')
                formatted_lines.append(line)
            
            with open(series_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_lines))
        else:
            # OmegaConf
            series_cfg = OmegaConf.create(series_dict)
            OmegaConf.save(series_cfg, series_path)
        
        # Write blocks YAML
        if hasattr(yaml_lib, 'dump'):
            # PyYAML - add blank lines between blocks for better readability
            import io
            stream = io.StringIO()
            yaml_lib.dump(blocks_dict, stream, **yaml_kwargs)
            yaml_content = stream.getvalue()
            
            # Add blank line before each block key (except the first one)
            lines = yaml_content.split('\n')
            formatted_lines = []
            for i, line in enumerate(lines):
                # If line starts a new block (starts with a key and colon, not indented)
                if line and not line.startswith(' ') and ':' in line and not line.strip().startswith('#'):
                    # Add blank line before this line (except for the first block)
                    if i > 0 and formatted_lines and formatted_lines[-1].strip():
                        formatted_lines.append('')
                formatted_lines.append(line)
            
            with open(blocks_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_lines))
        else:
            # OmegaConf
            blocks_cfg = OmegaConf.create(blocks_dict)
            OmegaConf.save(blocks_cfg, blocks_path)
            
    except Exception as e:
        raise RuntimeError(f"Failed to write YAML files: {e}")
    
    return series_path, blocks_path


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
        raise ValueError(
            "Direct spec CSV loading has been removed. "
            "Use dfm_python.api.from_spec() or from_spec_df() to convert the CSV "
            "to YAML files, then load via YAML/Hydra."
        )
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
            raise ValueError(
                "Direct CSV configs are no longer supported. "
                "Use dfm_python.api.from_spec() or from_spec_df() to convert the CSV "
                "to YAML files, then load the YAML configuration."
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
# Internal helper: Load config from DataFrame
# ============================================================================

def _load_config_from_dataframe(df: pl.DataFrame) -> DFMConfig:
    """Load configuration from DataFrame (internal helper).
    
    This function converts a DataFrame with series specifications into a DFMConfig.
    
    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame with columns: series_id, series_name, frequency, transformation, blocks
        Optional columns: Release (release date)
        Block_* columns: Binary indicators for block membership
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    series_list = []
    block_names_set = set()
    
    # Convert polars DataFrame to dict of rows for easier access
    df_dict = df.to_dict(as_series=False)
    
    # First, try to find Block_* columns (CSV format)
    block_cols = [col for col in df.columns 
                 if (col.startswith('Block_') or col.startswith('Block-')) 
                 and col not in ['blocks']]
    
    # Iterate over rows
    num_rows = len(df)
    for i in range(num_rows):
        # Parse blocks from Block_* columns or 'blocks' column
        blocks = []
        
        if block_cols:
            # CSV format: Block_Global, Block_Consumption, etc. columns with 1/0 values
            for block_col in block_cols:
                if block_col in df_dict:
                    block_value = df_dict[block_col][i]
                    # Handle various types: int, float, numpy int/float, bool, string "1"/"0"
                    # Check numpy types first (numpy.int64, numpy.float64, etc.)
                    if hasattr(block_value, 'item'):
                        # numpy scalar - convert to Python native type
                        block_value = block_value.item()
                    
                    if isinstance(block_value, (int, float)) and block_value != 0:
                        blocks.append(block_col)
                    elif isinstance(block_value, bool) and block_value:
                        blocks.append(block_col)
                    elif isinstance(block_value, str) and block_value.strip() in ['1', 'True', 'true']:
                        blocks.append(block_col)
        else:
            # Fallback: try 'blocks' column (comma-separated string or list)
            if 'blocks' in df_dict:
                blocks_str = df_dict['blocks'][i]
                if isinstance(blocks_str, str):
                    blocks = [b.strip() for b in blocks_str.split(',')]
                elif isinstance(blocks_str, list):
                    blocks = blocks_str
                else:
                    blocks = ['Block_Global']
            else:
                blocks = ['Block_Global']
        
        # Ensure at least Block_Global
        if not blocks:
            blocks = ['Block_Global']
        
        # Track block names
        for block in blocks:
            block_names_set.add(block)
        
        # Parse release_date if available
        release_date = None
        if 'Release' in df_dict:
            release_date = df_dict['Release'][i]
            if release_date is not None:
                try:
                    release_date = int(release_date)
                except (ValueError, TypeError):
                    release_date = None
        
        series_id = df_dict.get('series_id', [f"series_{j}" for j in range(num_rows)])[i]
        series_name = df_dict.get('series_name', df_dict.get('series_id', [f"Series {j}" for j in range(num_rows)]))[i]
        
        series_list.append(SeriesConfig(
            series_id=series_id,
            series_name=series_name,
            frequency=df_dict.get('frequency', ['m'] * num_rows)[i],
            transformation=df_dict.get('transformation', ['lin'] * num_rows)[i],
            blocks=blocks,
            release_date=release_date
        ))
    
    # Create default blocks if none specified
    block_names = sorted(block_names_set) if block_names_set else ['Block_Global']
    blocks = {}
    for block_name in block_names:
        blocks[block_name] = BlockConfig(factors=1, ar_lag=1, clock='m')
    
    # block_names is derived automatically in DFMConfig.__post_init__ from blocks dict
    return DFMConfig(series=series_list, blocks=blocks)


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


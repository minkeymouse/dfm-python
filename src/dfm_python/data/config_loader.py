"""Configuration loading utilities for DFM estimation.

This module provides functions for loading DFM configuration from various
sources, including YAML files and spec CSV files.
"""

from pathlib import Path
from typing import Union
import pandas as pd

from ..config import DFMConfig, SeriesConfig, BlockConfig, YamlSource


def load_config_from_yaml(configfile: Union[str, Path]) -> DFMConfig:
    """Load model configuration from YAML file.
    
    This function loads the config structure:
    - Main settings (estimation parameters) from config/default.yaml
    - Series definitions from config/series/default.yaml
    - Block definitions from config/blocks/default.yaml
    
    Parameters
    ----------
    configfile : str or Path
        Path to main YAML configuration file (e.g., config/default.yaml)
        
    Returns
    -------
    DFMConfig
        Model configuration (dataclass with validation)
        
    Raises
    ------
    FileNotFoundError
        If configfile or referenced files do not exist
    ImportError
        If omegaconf is not available
    ValueError
        If configuration is invalid
    """
    return YamlSource(configfile).load()


def _load_config_from_dataframe(df: pd.DataFrame) -> DFMConfig:
    """Load configuration from DataFrame (internal helper).
    
    This function converts a DataFrame with series specifications into a DFMConfig.
    It's used internally by SpecCSVSource and other config loaders.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: series_id, series_name, frequency, transformation, blocks
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    series_list = []
    block_names_set = set()
    
    for _, row in df.iterrows():
        # Parse blocks (can be comma-separated string or list)
        blocks_str = row.get('blocks', 'Block_Global')
        if isinstance(blocks_str, str):
            blocks = [b.strip() for b in blocks_str.split(',')]
        else:
            blocks = blocks_str if isinstance(blocks_str, list) else ['Block_Global']
        
        # Track block names
        for block in blocks:
            block_names_set.add(block)
        
        series_list.append(SeriesConfig(
            series_id=row.get('series_id', f"series_{len(series_list)}"),
            series_name=row.get('series_name', row.get('series_id', f"Series {len(series_list)}")),
            frequency=row.get('frequency', 'm'),
            transformation=row.get('transformation', 'lin'),
            blocks=blocks
        ))
    
    # Create default blocks if none specified
    block_names = sorted(block_names_set) if block_names_set else ['Block_Global']
    blocks = {}
    for block_name in block_names:
        blocks[block_name] = BlockConfig(factors=1, ar_lag=1, clock='m')
    
    return DFMConfig(series=series_list, blocks=blocks, block_names=block_names)


def load_config_from_spec(specfile: Union[str, Path]) -> DFMConfig:
    """Load configuration from spec CSV file.
    
    Parameters
    ----------
    specfile : str or Path
        Path to spec CSV file
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    specfile = Path(specfile)
    if not specfile.exists():
        raise FileNotFoundError(f"Spec file not found: {specfile}")
    
    df = pd.read_csv(specfile)
    return _load_config_from_dataframe(df)


def load_config(configfile: Union[str, Path, DFMConfig]) -> DFMConfig:
    """Load configuration from file or return DFMConfig directly.
    
    This is a convenience function that handles both file paths and DFMConfig objects.
    For more control, use the specific load_config_from_* functions or ConfigSource classes.
    
    Parameters
    ----------
    configfile : str, Path, or DFMConfig
        Path to config file (YAML or spec CSV) or DFMConfig object
        
    Returns
    -------
    DFMConfig
        Model configuration object
    """
    if isinstance(configfile, DFMConfig):
        return configfile
    
    configfile = Path(configfile)
    if not configfile.exists():
        raise FileNotFoundError(f"Config file not found: {configfile}")
    
    # Try YAML first, then spec CSV
    if configfile.suffix.lower() in ['.yaml', '.yml']:
        return load_config_from_yaml(configfile)
    elif configfile.suffix.lower() == '.csv':
        return load_config_from_spec(configfile)
    else:
        # Default to YAML
        return load_config_from_yaml(configfile)

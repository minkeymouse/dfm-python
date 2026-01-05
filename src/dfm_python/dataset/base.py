"""Base dataset class for Factor Models.

This module provides the base dataset class that all factor model datasets inherit from.
Contains common functionality for config loading and target series handling.
"""

from typing import Optional, Union, List, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ..config import DFMConfig

from ..logger import get_logger
from ..config import DFMConfig, YamlSource
from ..utils.misc import get_config_attr
from ..utils.errors import ConfigurationError

_logger = get_logger(__name__)


def load_config(config: Optional[DFMConfig], config_path: Optional[Union[str, Path]]) -> DFMConfig:
    """Load configuration from config object or config_path.
    
    Parameters
    ----------
    config : DFMConfig, optional
        Configuration object
    config_path : str or Path, optional
        Path to configuration file
        
    Returns
    -------
    DFMConfig
        Loaded configuration object
        
    Raises
    ------
    ConfigurationError
        If both config and config_path are None
    """
    if config is None and config_path is not None:
        source = YamlSource(config_path)
        config = source.load()
    
    if config is None:
        raise ConfigurationError(
            "Dataset initialization failed: either config or config_path must be provided. "
            "Please provide a DFMConfig object or a path to a configuration file.",
            details="Both config and config_path are None. One must be provided."
        )
    
    return config


def normalize_target_series(target_series: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalize target_series to a list.
    
    Parameters
    ----------
    target_series : str, List[str], or None
        Target series specification
        
    Returns
    -------
    List[str]
        Normalized list of target series (empty list if None)
    """
    if target_series is None:
        return []
    elif isinstance(target_series, str):
        return [target_series]
    else:
        return list(target_series)


class BaseFactorModelDataset:
    """Base dataset class for all factor model datasets.
    
    Provides common functionality for config loading and target series handling.
    """
    
    def __init__(
        self,
        config: Optional['DFMConfig'] = None,
        config_path: Optional[Union[str, Path]] = None,
        target_series: Optional[Union[str, List[str]]] = None,
    ):
        """Initialize base dataset with common attributes.
        
        Parameters
        ----------
        config : DFMConfig, optional
            Model configuration object
        config_path : str or Path, optional
            Path to configuration file
        target_series : str or List[str], optional
            Target series column names
        """
        # Load config
        self.config = load_config(config, config_path)
        
        # Normalize target_series and extract target_scaler
        self.target_series = normalize_target_series(target_series)
        self.target_scaler = get_config_attr(self.config, 'target_scaler', None)

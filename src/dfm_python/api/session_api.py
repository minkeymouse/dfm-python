"""Module-level session API and convenience functions for DFM.

This module provides singleton instances and module-level convenience functions
for both DFM and DDFM, as well as factory functions and constructors.
"""

from typing import Optional, Union, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import asdict
import polars as pl
import logging

from .model_api import DFM, DDFM
from ..config import DFMConfig, Params, ConfigSource
from ..config.io import _load_config_from_dataframe, _write_series_blocks_yaml

_logger = logging.getLogger(__name__)


def _dump_yaml_to_file(path: Path, payload: Dict[str, Any]) -> None:
    """Helper function to dump YAML file."""
    try:
        import yaml  # type: ignore
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
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


# Create singleton instances for module-level usage
_dfm_instance = DFM()
_ddfm_instance = DDFM()


# Module-level convenience functions that delegate to the singleton (DFM)
def load_config(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DFM:
    """Load configuration (module-level convenience function)."""
    return _dfm_instance.load_config(
        source=source,
        yaml=yaml,
        mapping=mapping,
        hydra=hydra,
        base=base,
        override=override,
    )


def load_data(data_path: Optional[Union[str, Path]] = None,
               data: Optional[Any] = None,
               **kwargs) -> DFM:
    """Load data (module-level convenience function)."""
    return _dfm_instance.load_data(data_path=data_path, data=data, **kwargs)


def train(threshold: Optional[float] = None,
          max_iter: Optional[int] = None,
          **kwargs) -> DFM:
    """Train the model (module-level convenience function)."""
    return _dfm_instance.train(threshold=threshold, max_iter=max_iter, **kwargs)


def predict(horizon: Optional[int] = None, **kwargs):
    """Forecast using the trained model (module-level convenience function)."""
    return _dfm_instance.predict(horizon=horizon, **kwargs)


def plot(**kwargs):
    """Plot common visualizations (module-level convenience function)."""
    return _dfm_instance.plot(**kwargs)


def load_pickle(path: Union[str, Path], **kwargs) -> DFM:
    """Load a saved model payload (module-level convenience function)."""
    return _dfm_instance.load_pickle(path, **kwargs)


def reset() -> DFM:
    """Reset state (module-level convenience function)."""
    return _dfm_instance.reset()


# DDFM module-level convenience functions
def load_config_ddfm(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DDFM:
    """Load configuration for DDFM (module-level convenience function)."""
    return _ddfm_instance.load_config(
        source=source,
        yaml=yaml,
        mapping=mapping,
        hydra=hydra,
        base=base,
        override=override,
    )


def load_data_ddfm(data_path: Optional[Union[str, Path]] = None,
                   data: Optional[Any] = None,
                   **kwargs) -> DDFM:
    """Load data for DDFM (module-level convenience function)."""
    return _ddfm_instance.load_data(data_path=data_path, data=data, **kwargs)


def train_ddfm(
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    encoder_layers: Optional[List[int]] = None,
    num_factors: Optional[int] = None,
    factor_order: int = 1,
    use_idiosyncratic: bool = True,
    activation: str = 'tanh',
    use_batch_norm: bool = True,
    min_obs_idio: int = 5,
    **kwargs
) -> DDFM:
    """Train DDFM (Deep Dynamic Factor Model) using autoencoder.
    
    Convenience function for training DDFM model using the module-level DDFM instance.
    This is equivalent to:
    >>> ddfm = DDFM(encoder_layers=[64, 32], ...)
    >>> ddfm.load_config('config/default.yaml')
    >>> ddfm.load_data('data/sample_data.csv')
    >>> ddfm.train(epochs=100, ...)
    
    Parameters
    ----------
    epochs : int, optional
        Number of training epochs for autoencoder. Default: 100
    batch_size : int, optional
        Batch size for training. Default: 32
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default: 0.001
    encoder_layers : List[int], optional
        Hidden layer dimensions for encoder. Default: [64, 32]
    num_factors : int, optional
        Number of factors. If None, inferred from config.
    factor_order : int, default 1
        VAR lag order for factor dynamics (1 or 2). Default: 1
    use_idiosyncratic : bool, default True
        Whether to model idiosyncratic components with AR(1) dynamics.
    activation : str, default 'tanh'
        Activation function ('tanh', 'relu', 'sigmoid').
    use_batch_norm : bool, default True
        Whether to use batch normalization in encoder.
    min_obs_idio : int, default 5
        Minimum observations for idio AR(1) estimation.
    **kwargs
        Additional parameters passed to model.fit().
        
    Returns
    -------
    DDFM
        The DDFM instance with trained model.
        
    Examples
    --------
    >>> import dfm_python as dfm
    >>> dfm.load_config_ddfm('config/default.yaml')
    >>> dfm.load_data_ddfm('data/sample_data.csv')
    >>> dfm.train_ddfm(epochs=100, encoder_layers=[64, 32], num_factors=2)
    """
    # Update DDFM instance parameters if provided
    # Only recreate if parameters actually differ from defaults
    needs_reinit = (
        encoder_layers is not None or
        num_factors is not None or
        activation != 'tanh' or
        use_batch_norm is not True or
        factor_order != 1 or
        use_idiosyncratic is not True or
        min_obs_idio != 5
    )
    
    if needs_reinit:
        # Create new instance with updated parameters
        global _ddfm_instance
        _ddfm_instance = DDFM(
            encoder_layers=encoder_layers,
            num_factors=num_factors,
            activation=activation,
            use_batch_norm=use_batch_norm,
            factor_order=factor_order,
            use_idiosyncratic=use_idiosyncratic,
            min_obs_idio=min_obs_idio,
        )
    
    return _ddfm_instance.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )


def predict_ddfm(horizon: Optional[int] = None, **kwargs):
    """Forecast using the trained DDFM model (module-level convenience function)."""
    return _ddfm_instance.predict(horizon=horizon, **kwargs)


def plot_ddfm(**kwargs):
    """Plot common visualizations for DDFM (module-level convenience function)."""
    return _ddfm_instance.plot(**kwargs)


def reset_ddfm() -> DDFM:
    """Reset DDFM state (module-level convenience function)."""
    return _ddfm_instance.reset()


def create_model(model_type: str = 'dfm', **kwargs):
    """Create a factor model instance.
    
    Factory function to create different types of factor models.
    
    Parameters
    ----------
    model_type : str
        Type of model to create. Options:
        - 'dfm' or 'linear': Linear Dynamic Factor Model (default)
        - 'ddfm' or 'deep': Deep Dynamic Factor Model (requires PyTorch)
    **kwargs
        Additional arguments passed to model constructor.
        
        For DDFM, these include:
        - encoder_layers: List[int] - Hidden layer dimensions (default: [64, 32])
        - num_factors: int - Number of factors (inferred from config if None)
        - activation: str - Activation function ('tanh', 'relu', 'sigmoid', default: 'tanh')
        - use_batch_norm: bool - Use batch normalization (default: True)
        - learning_rate: float - Learning rate for Adam (default: 0.001)
        - epochs: int - Number of training epochs (default: 100)
        - batch_size: int - Batch size (default: 32)
        - factor_order: int - VAR lag order for factors (1 or 2, default: 1)
        - use_idiosyncratic: bool - Model idio components with AR(1) (default: True)
        - min_obs_idio: int - Min observations for idio AR(1) (default: 5)
        
    Returns
    -------
    BaseFactorModel
        Model instance (DFM or DDFM)
        
    Examples
    --------
    >>> # Create linear DFM
    >>> model = create_model('dfm')
    >>> 
    >>> # Create DDFM with custom encoder
    >>> model = create_model('ddfm', encoder_layers=[64, 32], num_factors=2)
    >>> 
    >>> # Create DDFM with VAR(2) and idio modeling
    >>> model = create_model('ddfm', factor_order=2, use_idiosyncratic=True)
    """
    model_type = model_type.lower()
    
    if model_type in ('dfm', 'linear'):
        from ..models.dfm import DFMLinear
        return DFMLinear(**kwargs)
    elif model_type in ('ddfm', 'deep'):
        try:
            from ..models.ddfm import DDFM
            return DDFM(**kwargs)
        except ImportError:
            raise ImportError(
                "DDFM requires PyTorch. Install with: pip install dfm-python[deep]"
            )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'dfm', 'linear', 'ddfm', 'deep'"
        )


# Convenience constructors for cleaner API
def from_yaml(yaml_path: Union[str, Path]) -> DFM:
    """Load configuration from YAML file (convenience constructor)."""
    return _dfm_instance.load_config(yaml=yaml_path)


def from_spec(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    series_filename: Optional[str] = None,
    blocks_filename: Optional[str] = None
) -> Tuple[Path, Path]:
    """Convert spec CSV file to YAML configuration files.
    
    This function reads a spec CSV file and generates two YAML files:
    - config/series/{basename}.yaml - series definitions
    - config/blocks/{basename}.yaml - block definitions
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the spec CSV file
    output_dir : str or Path, optional
        Output directory for YAML files. Defaults to config/ directory relative to CSV.
    series_filename : str, optional
        Custom filename for series YAML (without .yaml extension).
        Defaults to CSV basename.
    blocks_filename : str, optional
        Custom filename for blocks YAML (without .yaml extension).
        Defaults to CSV basename.
        
    Returns
    -------
    Tuple[Path, Path]
        Paths to generated series YAML and blocks YAML files
        
    Examples
    --------
    >>> series_path, blocks_path = from_spec('data/sample_spec.csv')
    >>> # Creates config/series/sample_spec.yaml and config/blocks/sample_spec.yaml
    """
    from ..config.io import from_spec as _from_spec
    return _from_spec(csv_path, output_dir, series_filename, blocks_filename)


def from_spec_df(
    spec_df: Union[pl.DataFrame, Any],
    params: Optional[Params] = None,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    config_name: Optional[str] = None
) -> DFM:
    """Convert spec DataFrame to YAML files and load via YAML/Hydra."""
    if params is None:
        params = Params()
    
    if not isinstance(spec_df, pl.DataFrame):
        raise TypeError(f"spec_df must be polars DataFrame, got {type(spec_df)}")
    
    config = _load_config_from_dataframe(spec_df)
    
    if output_dir is None:
        output_dir = Path('config') / 'generated'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = uuid.uuid4().hex[:6]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = config_name or f'spec_{timestamp}_{suffix}'
    series_filename = f'{base_name}_series'
    blocks_filename = f'{base_name}_blocks'
    
    series_path, blocks_path = _write_series_blocks_yaml(
        config,
        output_dir,
        series_filename,
        blocks_filename
    )
    
    main_config_path = output_dir / f'{base_name}.yaml'
    params_dict = {k: v for k, v in asdict(params).items() if v is not None}
    main_payload: Dict[str, Any] = {
        'defaults': [
            {'series': series_filename},
            {'blocks': blocks_filename},
            '_self_'
        ]
    }
    main_payload.update(params_dict)
    
    _dump_yaml_to_file(main_config_path, main_payload)
    
    _logger.info("✓ Spec DataFrame converted to YAML:")
    _logger.info(f"  - Series YAML: {series_path}")
    _logger.info(f"  - Blocks YAML: {blocks_path}")
    _logger.info(f"  - Main config : {main_config_path}")
    
    _dfm_instance.load_config(yaml=main_config_path)
    return _dfm_instance


def from_dict(mapping: Dict[str, Any]) -> DFM:
    """Load configuration from dictionary (convenience constructor)."""
    return _dfm_instance.load_config(mapping=mapping)


"""High-level model API classes for Dynamic Factor Models.

This module contains the DFMBase, DFM, and DDFM classes that provide
object-oriented interfaces for DFM estimation.
"""

import os
import logging
import pickle
from ..core.helpers import get_logger
from typing import Optional, Union, Dict, Any, Tuple, List, Sequence
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl

from ..config import (
    DFMConfig, Params,
    make_config_source,
    ConfigSource,
    MergedConfigSource,
)
from ..dataloader.loader import load_data as _load_data, DataView
from ..models.base import BaseFactorModel
from ..models.dfm import DFMLinear
# DFMCore is now an alias for DFMLinear (backward compatibility)
DFMCore = DFMLinear
from ..core.results import DFMResult
from ..nowcasting import Nowcast
from ..core.helpers import (
    safe_get_method,
    safe_get_attr,
    get_clock_frequency,
    _validate_config_loaded,
    _validate_data_loaded,
    _validate_result_loaded,
)
# Helper functions and dataset utilities are now in this file
from ..core.time import TimeIndex
# Helper functions (convert_time_index_to_list, generate_dataset, get_state, etc.) 
# are defined later in this file (starting around line 509)

# ============================================================================
# High-level API Classes (DFMBase, DFM, DDFM)
# ============================================================================

_logger = get_logger(__name__)


class DFMBase(BaseFactorModel):
    """Base class for high-level DFM API.
    
    This class provides a unified interface for loading configuration, data,
    training, and prediction. It delegates to low-level model implementations
    (DFMLinear for linear DFM, DDFM for deep DFM).
    """
    
    def __init__(self):
        """Initialize DFMBase instance."""
        super().__init__()
        self._model_impl: Optional[BaseFactorModel] = None
        self._original_data: Optional[np.ndarray] = None
        self._data_frame: Optional[pl.DataFrame] = None
        self._nowcast: Optional[Nowcast] = None
    
    @property
    def original_data(self) -> Optional[np.ndarray]:
        """Get original (untransformed) data matrix."""
        return self._original_data
    
    @property
    def nowcast(self) -> Nowcast:
        """Get nowcasting manager instance."""
        if self._nowcast is None:
            _validate_config_loaded(self._config)
            _validate_data_loaded(self._data)
            _validate_result_loaded(self._result)
            self._nowcast = Nowcast(
                config=self._config,
                data=self._data,
                time=self._time,
                result=self._result
            )
        return self._nowcast
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Dict[str, Any]] = None,
        hydra: Optional[Union[Dict[str, Any], Any]] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    ) -> 'DFMBase':
        """Load configuration from various sources."""
        config_source = make_config_source(
            source=source,
            yaml=yaml,
            mapping=mapping,
            hydra=hydra,
            base=base,
            override=override,
        )
        self._config = config_source.load()
        return self
    
    def load_data(
        self,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Any] = None,
        **kwargs
    ) -> 'DFMBase':
        """Load data from file or array."""
        _validate_config_loaded(self._config)
        
        if data_path is not None:
            self._data, self._time, self._original_data = _load_data(
                data_path, self._config, **kwargs
            )
        elif data is not None:
            if isinstance(data, pl.DataFrame):
                self._data_frame = data
                # Convert DataFrame to numpy array
                self._data = data.to_numpy()
            else:
                self._data = np.asarray(data)
            # Generate time index if not provided
            if self._time is None:
                from ..core.time import datetime_range, clock_to_datetime_freq
                clock = get_clock_frequency(self._config, 'm')
                datetime_freq = clock_to_datetime_freq(clock)
                from datetime import datetime
                start_date = datetime(2000, 1, 1)
                self._time = TimeIndex(datetime_range(start=start_date, periods=len(self._data), freq=datetime_freq))
            self._original_data = self._data.copy()
        else:
            raise ValueError("Either data_path or data must be provided")
        
        return self
    
    def train(self, **kwargs) -> 'DFMBase':
        """Train the model."""
        _validate_config_loaded(self._config)
        _validate_data_loaded(self._data)
        
        if self._model_impl is None:
            raise ValueError("Model implementation not set. Use DFM() or DDFM() instead of DFMBase()")
        
        self._result = self._model_impl.fit(self._data, self._config, **kwargs)
        return self
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ):
        """Forecast future values."""
        _validate_result_loaded(self._result)
        return self._model_impl.predict(
            horizon=horizon,
            return_series=return_series,
            return_factors=return_factors
        )
    
    def plot(self, **kwargs):
        """Plot common visualizations."""
        _validate_result_loaded(self._result)
        # Plotting functionality would go here
        _logger.info("Plot functionality not yet implemented")
        return self
    
    def reset(self) -> 'DFMBase':
        """Reset state."""
        self._config = None
        self._data = None
        self._time = None
        self._result = None
        self._original_data = None
        self._data_frame = None
        self._nowcast = None
        return self
    
    def load_pickle(self, path: Union[str, Path], **kwargs) -> 'DFMBase':
        """Load a saved model payload."""
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        # Restore state from payload
        self._config = payload.get('config')
        self._data = payload.get('data')
        self._time = payload.get('time')
        self._result = payload.get('result')
        self._original_data = payload.get('original_data')
        return self
    
    def generate_dataset(
        self,
        target_series: str,
        periods: List[datetime],
        backward: int = 0,
        forward: int = 0,
        dataview: Optional[DataView] = None
    ) -> Dict[str, Any]:
        """Generate dataset for DFM evaluation."""
        return generate_dataset(
            self, target_series, periods, backward, forward, dataview
        )
    
    def get_state(
        self,
        t: Union[int, datetime],
        target_series: str,
        lookback: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get DFM state at time t."""
        return get_state(self, t, target_series, lookback)


class DFM(DFMBase):
    """High-level API for Linear Dynamic Factor Model."""
    
    def __init__(self):
        """Initialize DFM instance."""
        super().__init__()
        self._model_impl = DFMLinear()
    
    def train(
        self,
        threshold: Optional[float] = None,
        max_iter: Optional[int] = None,
        **kwargs
    ) -> 'DFM':
        """Train the linear DFM model."""
        _validate_config_loaded(self._config)
        _validate_data_loaded(self._data)
        
        self._result = self._model_impl.fit(
            self._data,
            self._config,
            threshold=threshold,
            max_iter=max_iter,
            **kwargs
        )
        return self


class DDFM(DFMBase):
    """High-level API for Deep Dynamic Factor Model."""
    
    def __init__(
        self,
        encoder_layers: Optional[List[int]] = None,
        num_factors: Optional[int] = None,
        activation: str = 'tanh',
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        factor_order: int = 1,
        use_idiosyncratic: bool = True,
        min_obs_idio: int = 5,
        **kwargs
    ):
        """Initialize DDFM instance."""
        super().__init__()
        try:
            from ..models.ddfm import DDFM as DDFMModel
            self._model_impl = DDFMModel(
                encoder_layers=encoder_layers,
                num_factors=num_factors,
                activation=activation,
                use_batch_norm=use_batch_norm,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                factor_order=factor_order,
                use_idiosyncratic=use_idiosyncratic,
                min_obs_idio=min_obs_idio,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "DDFM requires PyTorch. Install with: pip install dfm-python[deep]"
            )
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ) -> 'DDFM':
        """Train the DDFM model."""
        _validate_config_loaded(self._config)
        _validate_data_loaded(self._data)
        
        train_kwargs = {}
        if epochs is not None:
            train_kwargs['epochs'] = epochs
        if batch_size is not None:
            train_kwargs['batch_size'] = batch_size
        if learning_rate is not None:
            train_kwargs['learning_rate'] = learning_rate
        train_kwargs.update(kwargs)
        
        self._result = self._model_impl.fit(
            self._data,
            self._config,
            **train_kwargs
        )
        return self


# ============================================================================
# Module-level convenience functions (singleton-based API)
# ============================================================================
"""Module-level convenience functions for DFM API.

This module provides singleton-based convenience functions for backward compatibility.
For new code, prefer using explicit instances: model = DFM() instead of
using the singleton-based functions.
"""

from typing import Optional, Union, Dict, Any, Tuple, List, TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    from omegaconf import DictConfig

from ..config import (
    DFMConfig,
    Params,
    ConfigSource,
)

# Create singleton instances for module-level usage (backward compatibility)
# Note: For new code, prefer creating explicit instances: model = DFM()
# These are created lazily to avoid issues with abstract methods
_dfm_instance: Optional[DFM] = None
_ddfm_instance: Optional[DDFM] = None


def _get_dfm_instance() -> DFM:
    """Get or create DFM singleton instance."""
    global _dfm_instance
    if _dfm_instance is None:
        _dfm_instance = DFM()
    return _dfm_instance


def _get_ddfm_instance() -> DDFM:
    """Get or create DDFM singleton instance."""
    global _ddfm_instance
    if _ddfm_instance is None:
        _ddfm_instance = DDFM()
    return _ddfm_instance


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


# Module-level convenience functions that delegate to the singleton (DFM)
def load_config(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Union[Dict[str, Any], Any]] = None,  # DictConfig when TYPE_CHECKING
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DFM:
    """Load configuration (module-level convenience function)."""
    return _get_dfm_instance().load_config(
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
    return _get_dfm_instance().load_data(data_path=data_path, data=data, **kwargs)


def train(threshold: Optional[float] = None,
          max_iter: Optional[int] = None,
          **kwargs) -> DFM:
    """Train the model (module-level convenience function)."""
    return _get_dfm_instance().train(threshold=threshold, max_iter=max_iter, **kwargs)


def predict(
    horizon: Optional[int] = None,
    *,
    return_series: bool = True,
    return_factors: bool = True
):
    """Forecast using the trained model (module-level convenience function)."""
    return _get_dfm_instance().predict(
        horizon=horizon,
        return_series=return_series,
        return_factors=return_factors
    )


def plot(**kwargs):
    """Plot common visualizations (module-level convenience function)."""
    return _get_dfm_instance().plot(**kwargs)


def load_pickle(path: Union[str, Path], **kwargs) -> DFM:
    """Load a saved model payload (module-level convenience function)."""
    return _get_dfm_instance().load_pickle(path, **kwargs)


def reset() -> DFM:
    """Reset state (module-level convenience function)."""
    return _get_dfm_instance().reset()


# DDFM module-level convenience functions
def load_config_ddfm(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Union[Dict[str, Any], Any]] = None,  # DictConfig when TYPE_CHECKING
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DDFM:
    """Load configuration for DDFM (module-level convenience function)."""
    return _get_ddfm_instance().load_config(
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
    return _get_ddfm_instance().load_data(data_path=data_path, data=data, **kwargs)


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
    
    return _get_ddfm_instance().train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )


def predict_ddfm(
    horizon: Optional[int] = None,
    *,
    return_series: bool = True,
    return_factors: bool = True
):
    """Forecast using the trained DDFM model (module-level convenience function)."""
    return _get_ddfm_instance().predict(
        horizon=horizon,
        return_series=return_series,
        return_factors=return_factors
    )


def plot_ddfm(**kwargs):
    """Plot common visualizations for DDFM (module-level convenience function)."""
    return _get_ddfm_instance().plot(**kwargs)


def reset_ddfm() -> DDFM:
    """Reset DDFM state (module-level convenience function)."""
    return _get_ddfm_instance().reset()


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
    from ..models.base import BaseFactorModel
    
    model_type = model_type.lower()
    
    if model_type in ('dfm', 'linear'):
        from ..models.dfm import DFMLinear
        return DFMLinear(**kwargs)
    elif model_type in ('ddfm', 'deep'):
        try:
            from ..models.ddfm import DDFM as DDFMModel
            return DDFMModel(**kwargs)
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
    return _get_dfm_instance().load_config(yaml=yaml_path)


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
    spec_df: Any,  # polars.DataFrame
    params: Optional[Params] = None,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    config_name: Optional[str] = None
) -> DFM:
    """Convert spec DataFrame to YAML files and load via YAML/Hydra."""
    from ..config.io import _load_config_from_dataframe, _write_series_blocks_yaml
    from dataclasses import asdict
    import uuid
    from datetime import datetime
    import polars as pl
    
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
    
    logging.info("✓ Spec DataFrame converted to YAML:")
    logging.info(f"  - Series YAML: {series_path}")
    logging.info(f"  - Blocks YAML: {blocks_path}")
    logging.info(f"  - Main config : {main_config_path}")
    
    _get_dfm_instance().load_config(yaml=main_config_path)
    return _get_dfm_instance()


def from_dict(mapping: Dict[str, Any]) -> DFM:
    """Load configuration from dictionary (convenience constructor)."""
    return _get_dfm_instance().load_config(mapping=mapping)



# Utilities from model_api_utils.py
"""Model API utility functions.

This module contains helper functions extracted from model_api.py
to keep the main file under 1000 lines.
"""

# Helper functions and dataset utilities (merged from helpers.py and dataset_utils.py)
# ============================================================================

def convert_time_index_to_list(time_index: Optional[Any]) -> List[Any]:
    """Convert time index to list format for plotting/display."""
    if time_index is None:
        return []
    if isinstance(time_index, TimeIndex):
        return time_index.to_list()
    if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
        return list(time_index)
    try:
        return [time_index[i] for i in range(len(time_index))]
    except (TypeError, AttributeError):
        return []


def extract_features(model_instance: Any, X_view: np.ndarray, Time_view: Any, period: datetime) -> np.ndarray:
    """Extract features from data view for model evaluation."""
    if model_instance._result is not None and hasattr(model_instance._result, 'Z'):
        latest_factors = model_instance._result.Z[-1, :] if model_instance._result.Z.shape[0] > 0 else np.zeros(model_instance._result.Z.shape[1])
    else:
        latest_factors = np.array([])
    if X_view.shape[0] > 0:
        mean_residual = np.nanmean(X_view[-1, :]) if X_view.shape[0] > 0 else 0.0
    else:
        mean_residual = 0.0
    features = np.concatenate([latest_factors, [mean_residual]])
    return features


def construct_feature_vector(factors: np.ndarray, residuals: np.ndarray, news_summary: Dict[str, Any]) -> np.ndarray:
    """Construct feature vector from model state."""
    feature_parts = [
        factors.flatten(),
        residuals.flatten(),
        np.array([news_summary.get('total_impact', 0.0)]),
        np.array([news_summary.get('revision_impact', 0.0)]),
        np.array([news_summary.get('release_impact', 0.0)])
    ]
    features = np.concatenate([part for part in feature_parts if part.size > 0])
    return features


def generate_dataset(
    model_instance: Any, target_series: str, periods: List[datetime],
    backward: int = 0, forward: int = 0, dataview: Optional[DataView] = None
) -> Dict[str, Any]:
    """Generate dataset for DFM evaluation and model training."""
    from ..core.helpers import find_series_index
    from ..core.time import find_time_index
    
    i_series = find_series_index(model_instance._config, target_series)
    X_features, y_baseline, y_actual, metadata, backward_results = [], [], [], [], []
    
    if dataview is not None:
        dataview_factory = dataview
    else:
        dataview_factory = DataView.from_arrays(
            X=model_instance._data, Time=model_instance._time,
            Z=model_instance._original_data, config=model_instance._config,
            X_frame=model_instance._data_frame
        )
    if dataview_factory.config is None:
        dataview_factory.config = model_instance._config
    
    for period in periods:
        view_obj = dataview_factory.with_view_date(period)
        X_view, Time_view, _ = view_obj.materialize()
        
        if backward > 0:
            nowcasts, data_view_dates = [], []
            for weeks_back in range(backward, -1, -1):
                data_view_date = period - timedelta(weeks=weeks_back)
                view_past = dataview_factory.with_view_date(data_view_date)
                X_view_past, Time_view_past, _ = view_past.materialize()
                nowcast_val = model_instance.nowcast(
                    target_series=target_series,
                    view_date=view_past.view_date or data_view_date,
                    target_period=period
                )
                nowcasts.append(nowcast_val)
                data_view_dates.append(view_past.view_date or data_view_date)
            baseline_nowcast = nowcasts[-1]
            backward_results.append({
                'nowcasts': np.array(nowcasts),
                'data_view_dates': data_view_dates,
                'target_date': period
            })
        else:
            baseline_nowcast = model_instance.nowcast(
                target_series=target_series,
                view_date=view_obj.view_date or period,
                target_period=period
            )
        
        y_baseline.append(baseline_nowcast)
        t_idx = find_time_index(model_instance._time, period)
        actual_val = np.nan
        if t_idx is not None and t_idx < model_instance._data.shape[0] and i_series < model_instance._data.shape[1]:
            actual_val = model_instance._data[t_idx, i_series]
        y_actual.append(actual_val)
        features = extract_features(model_instance, X_view, Time_view, period)
        X_features.append(features)
        metadata.append({'period': period, 'target_series': target_series})
    
    return {
        'X': np.array(X_features),
        'y_baseline': np.array(y_baseline),
        'y_actual': np.array(y_actual),
        'y_target': np.array(y_actual) - np.array(y_baseline),
        'metadata': metadata,
        'backward_results': backward_results if backward > 0 else []
    }


def get_state(model_instance: Any, t: Union[int, datetime], target_series: str, lookback: Optional[int] = None) -> Dict[str, Any]:
    """Get DFM state at time t for downstream model input."""
    from ..core.structure import get_periods_per_year
    from ..core.helpers import get_clock_frequency, find_series_index
    from ..core.time import find_time_index, convert_to_timestamp
    from ..dataloader.loader import create_data_view
    
    if lookback is None:
        clock = get_clock_frequency(model_instance._config, 'm')
        lookback = get_periods_per_year(clock)
    
    t = convert_to_timestamp(t, model_instance._time, None)
    i_series = find_series_index(model_instance._config, target_series)
    
    X_view, Time_view, _ = create_data_view(
        X=model_instance._data, Time=model_instance._time,
        Z=model_instance._original_data, config=model_instance._config, view_date=t
    )
    
    baseline_nowcast = model_instance.nowcast(target_series=target_series, view_date=t, target_period=None)
    
    baseline_forecast, actual_history, residuals, factors_history = [], [], [], []
    t_idx = find_time_index(model_instance._time, t)
    if t_idx is None:
        raise ValueError(f"Time {t} not found in model_instance._time")
    
    for i in range(max(0, t_idx - lookback + 1), t_idx + 1):
        if i < model_instance._data.shape[0]:
            forecast_val = baseline_nowcast
            baseline_forecast.append(forecast_val)
            actual_val = model_instance._data[i, i_series] if i_series < model_instance._data.shape[1] else np.nan
            actual_history.append(actual_val)
            residual = actual_val - forecast_val if not np.isnan(actual_val) else np.nan
            residuals.append(residual)
            if model_instance._result is not None and hasattr(model_instance._result, 'Z') and i < model_instance._result.Z.shape[0]:
                factors_history.append(model_instance._result.Z[i, :])
            else:
                factors_history.append(np.zeros(model_instance._result.Z.shape[1]) if model_instance._result is not None else np.array([]))
    
    while len(baseline_forecast) < lookback:
        baseline_forecast.insert(0, np.nan)
        actual_history.insert(0, np.nan)
        residuals.insert(0, np.nan)
        factors_history.insert(0, np.zeros(factors_history[0].shape) if factors_history else np.array([]))
    
    if model_instance._result is not None and hasattr(model_instance._result, 'Z') and t_idx < model_instance._result.Z.shape[0]:
        factors = model_instance._result.Z[t_idx, :]
    else:
        factors = np.zeros(model_instance._result.Z.shape[1]) if model_instance._result is not None else np.array([])
    
    news_summary = {'total_impact': 0.0, 'top_contributors': [], 'revision_impact': 0.0, 'release_impact': 0.0}
    features = construct_feature_vector(factors=factors, residuals=np.array(residuals), news_summary=news_summary)
    
    n_missing = np.sum(np.isnan(X_view[-1, :])) if X_view.shape[0] > 0 else 0
    n_available = X_view.shape[1] - n_missing
    
    return {
        'baseline_nowcast': baseline_nowcast,
        'baseline_forecast': np.array(baseline_forecast),
        'actual_history': np.array(actual_history),
        'residuals': np.array(residuals),
        'factors': factors,
        'factors_history': np.array(factors_history),
        'news_summary': news_summary,
        'features': features,
        'metadata': {
            't': t_idx, 'date': t, 'target_series': target_series,
            'data_availability': {'n_missing': int(n_missing), 'n_available': int(n_available), 'missing_series': []}
        }
    }

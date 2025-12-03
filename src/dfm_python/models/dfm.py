"""Linear Dynamic Factor Model (DFM) implementation.

This module contains the linear DFM implementation using EM algorithm.
It inherits from BaseFactorModel to provide a consistent interface.
"""

import numpy as np
import polars as pl
import torch
from typing import Optional, Tuple, Union, Any, List, TYPE_CHECKING
from datetime import datetime
from ..logger import get_logger

from .base import BaseFactorModel
from ..config import DFMConfig, SeriesConfig, BlockConfig, validate_frequency
from ..config.results import DFMResult, DFMParams
from ..config.params import FitParams

if TYPE_CHECKING:
    from ..lightning import DFMDataModule

_logger = get_logger(__name__)


class DFMLinear(BaseFactorModel):
    """Linear Dynamic Factor Model using EM algorithm.
    
    This class implements the standard linear DFM with EM estimation.
    It inherits from BaseFactorModel to provide a consistent interface
    with other factor models (e.g., DDFM).
    
    The model assumes:
    - Linear observation equation: y_t = C Z_t + e_t
    - Linear factor dynamics: Z_t = A Z_{t-1} + v_t
    - Gaussian innovations
    
    Parameters are estimated via Expectation-Maximization (EM) algorithm.
    
    Note: This class consolidates the functionality previously split between
    DFMCore and DFMLinear. For backward compatibility, DFMCore is available
    as an alias to this class.
    """
    
    def fit(
        self,
        data_module: 'DFMDataModule',
        config: Optional[DFMConfig] = None,
        *,
        fit_params: Optional[FitParams] = None,
        **kwargs
    ) -> DFMResult:
        """Fit the linear DFM model using EM algorithm.
        
        This method performs the complete EM workflow:
        1. Initialization via PCA and OLS
        2. EM iterations until convergence
        3. Final Kalman smoothing
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        config : DFMConfig, optional
            Unified DFM configuration object. If None, uses config from data_module.
        fit_params : FitParams, optional
            Fit parameters object. If None, parameters are extracted from kwargs or config.
        **kwargs
            Additional parameters that override config values. Merged into fit_params if provided.
            
        Returns
        -------
        DFMResult
            Estimation results including parameters, factors, and diagnostics.
        """
        from ..lightning import DFMDataModule
        
        if not isinstance(data_module, DFMDataModule):
            raise TypeError(f"data_module must be DFMDataModule, got {type(data_module)}")
        
        # Ensure DataModule is set up
        if data_module.data_processed is None:
            data_module.setup()
        
        # Use config from data_module if not provided
        if config is None:
            config = data_module.config
        
        # Extract fit parameters from kwargs or use provided FitParams
        fit_params = self._prepare_fit_params(fit_params, **kwargs)
        
        # Get processed data and standardization params from DataModule
        X_torch = data_module.get_processed_data()
        Mx, Wx = data_module.get_standardization_params()
        
        # Handle case where standardization params might be None
        # (if transformer doesn't include StandardScaler)
        if Mx is None or Wx is None:
            # Use zeros/ones as defaults (no standardization)
            N = X_torch.shape[1]
            Mx = np.zeros(N, dtype=np.float32)
            Wx = np.ones(N, dtype=np.float32)
        
        # Get training parameters
        num_factors, training_params = self._extract_training_params(config, fit_params)
        
        # Train model using Lightning
        result = self._train_with_lightning(X_torch, config, num_factors, training_params, Mx, Wx)
        
        # Store results
        self._result = result
        return result
    
    def _prepare_fit_params(self, fit_params: Optional[FitParams], **kwargs) -> FitParams:
        """Prepare fit parameters from kwargs or provided FitParams."""
        if fit_params is None:
            return FitParams.from_kwargs(**kwargs)
        # Merge kwargs into fit_params
        fit_dict = fit_params.to_dict()
        fit_dict.update({k: v for k, v in kwargs.items() if v is not None})
        return FitParams.from_kwargs(**fit_dict)
    
    def _extract_training_params(
        self, config: DFMConfig, fit_params: FitParams
    ) -> Tuple[int, Dict[str, Any]]:
        """Extract training parameters from config and fit_params."""
        # Get parameters for Lightning module
        threshold_val = fit_params.threshold if fit_params.threshold is not None else getattr(config, 'threshold', 1e-4)
        max_iter_val = fit_params.max_iter if fit_params.max_iter is not None else getattr(config, 'max_iter', 100)
        nan_method_val = fit_params.nan_method if fit_params.nan_method is not None else getattr(config, 'nan_method', 2)
        nan_k_val = fit_params.nan_k if fit_params.nan_k is not None else getattr(config, 'nan_k', 3)
        
        # Determine number of factors
        if hasattr(config, 'factors_per_block') and config.factors_per_block:
            num_factors = int(np.sum(config.factors_per_block))
        else:
            blocks = config.get_blocks_array()
            if blocks.shape[1] > 0:
                num_factors = int(np.sum(blocks[:, 0]))
            else:
                num_factors = 1
        
        training_params = {
            'threshold': threshold_val,
            'max_iter': max_iter_val,
            'nan_method': nan_method_val,
            'nan_k': nan_k_val
        }
        
        return num_factors, training_params
    
    def _train_with_lightning(
        self,
        X_torch: torch.Tensor,
        config: DFMConfig,
        num_factors: int,
        training_params: Dict[str, Any],
        Mx: np.ndarray,
        Wx: np.ndarray
    ) -> DFMResult:
        """Train model using PyTorch Lightning."""
        from ..lightning.dfm_module import DFMLightningModule
        
        lightning_module = DFMLightningModule(
            config=config,
            num_factors=num_factors,
            **training_params
        )
        
        # Run EM algorithm
        lightning_module.fit_em(X_torch, Mx=Mx, Wx=Wx)
        
        # Extract results
        return lightning_module.get_result()
    
    def _create_default_config(
        self,
        X: np.ndarray,
        time_index: Optional[Union[List[datetime], np.ndarray, pl.Series]] = None,
        series_ids: Optional[List[str]] = None,
        num_factors: int = 1,
        clock: Optional[str] = None,
        transformation: str = 'lin'
    ) -> DFMConfig:
        """Create default configuration with smart defaults.
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix (T x N)
        time_index : list of datetime, np.ndarray, or pl.Series, optional
            Time index for frequency inference
        series_ids : list of str, optional
            Series identifiers (auto-generated if None)
        num_factors : int, default 1
            Number of factors in global block
        clock : str, optional
            Clock frequency (inferred from time_index if None)
        transformation : str, default 'lin'
            Transformation code for all series
            
        Returns
        -------
        DFMConfig
            Configuration with smart defaults
        """
        T, N = X.shape
        
        # Auto-generate series_ids if not provided
        if series_ids is None:
            series_ids = [f"series_{i}" for i in range(N)]
        
        # Infer clock frequency if not provided
        if clock is None:
            clock = self._infer_frequency(time_index)
        
        # Validate clock
        clock = validate_frequency(clock)
        
        # Create series configs (all series in global block)
        series_configs = []
        for i, series_id in enumerate(series_ids):
            series_configs.append(
                SeriesConfig(
                    frequency=clock,  # All series use clock frequency
                    transformation=transformation,
                    blocks=['Block_Global'],  # All series load on global block
                    series_id=series_id,
                    series_name=series_id
                )
            )
        
        # Create block config (single global block)
        block_configs = {
            'Block_Global': BlockConfig(
                factors=num_factors,
                ar_lag=1,
                clock=clock
            )
        }
        
        # Create DFMConfig with defaults
        config = DFMConfig(
            series=series_configs,
            blocks=block_configs,
            clock=clock,
            ar_lag=1,
            threshold=1e-5,
            max_iter=5000,
            nan_method=2,
            nan_k=3
        )
        
        return config
    
    def _infer_frequency(
        self,
        time_index: Optional[Union[List[datetime], np.ndarray, pl.Series]] = None
    ) -> str:
        """Infer frequency from time index.
        
        Parameters
        ----------
        time_index : list of datetime, np.ndarray, or pl.Series, optional
            Time index to analyze
            
        Returns
        -------
        str
            Inferred frequency code ('d', 'w', 'm', 'q', 'sa', 'a')
            Defaults to 'm' if inference fails or time_index is None.
        """
        if time_index is None or len(time_index) < 2:
            _logger.warning("Cannot infer frequency: time_index is None or too short. Defaulting to 'm' (monthly).")
            return 'm'
        
        # Convert to list of datetime if needed
        if isinstance(time_index, pl.Series):
            time_list = time_index.to_list()
        elif isinstance(time_index, np.ndarray):
            time_list = time_index.tolist()
        else:
            time_list = list(time_index)
        
        # Ensure all elements are datetime
        from ..utils.time import parse_timestamp
        try:
            time_list = [parse_timestamp(t) for t in time_list]
        except (ValueError, TypeError):
            _logger.warning("Cannot parse time_index. Defaulting to 'm' (monthly).")
            return 'm'
        
        if len(time_list) < 2:
            return 'm'
        
        # Calculate median time difference
        diffs = []
        for i in range(1, len(time_list)):
            diff = (time_list[i] - time_list[i-1]).total_seconds()
            diffs.append(diff)
        
        if not diffs:
            return 'm'
        
        median_diff_seconds = np.median(diffs)
        median_diff_days = median_diff_seconds / (24 * 3600)
        
        # Infer frequency based on median difference
        if median_diff_days <= 1.5:
            return 'd'  # Daily
        elif median_diff_days <= 5:
            return 'w'  # Weekly
        elif median_diff_days <= 35:
            return 'm'  # Monthly
        elif median_diff_days <= 100:
            return 'q'  # Quarterly
        elif median_diff_days <= 200:
            return 'sa'  # Semi-annual
        else:
            return 'a'  # Annual
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values using the fitted model.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 year
            of periods based on clock frequency.
        return_series : bool, optional
            Whether to return forecasted series (default: True)
        return_factors : bool, optional
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            If both return_series and return_factors are True:
                (X_forecast, Z_forecast) tuple
            If only return_series is True:
                X_forecast (horizon x N)
            If only return_factors is True:
                Z_forecast (horizon x m)
        """
        if self._result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Default horizon: 1 year of periods based on clock frequency
        if horizon is None:
            from ..config.structure import get_periods_per_year
            from ..utils.helpers import get_clock_frequency
            if self._config is None:
                clock = 'm'  # Default to monthly
            else:
                clock = get_clock_frequency(self._config, 'm')
            horizon = get_periods_per_year(clock)
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        # Extract model parameters
        A = self._result.A
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]
        
        # Deterministic forecast: iteratively apply transition matrix A
        Z_forecast = np.zeros((horizon, Z_last.shape[0]))
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Transform factors to observed series: X = Z @ C^T, then denormalize
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast


# Backward compatibility: DFMCore is an alias for DFMLinear
DFMCore = DFMLinear


# ============================================================================
# High-level API Classes
# ============================================================================

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, TYPE_CHECKING

from ..config import (
    DFMConfig, Params,
    make_config_source,
    ConfigSource,
    MergedConfigSource,
)
from ..transformations.utils import read_data as _load_data
from ..nowcast.dataview import DataView
from ..utils.helpers import (
    safe_get_method,
    safe_get_attr,
    get_clock_frequency,
    _validate_config_loaded,
    _validate_result_loaded,
)
from ..utils.time import TimeIndex

if TYPE_CHECKING:
    from omegaconf import DictConfig


class DFM(BaseFactorModel):
    """High-level API for Linear Dynamic Factor Model.
    
    This class provides a unified interface for loading configuration, data,
    training, and prediction. It uses DFMLinear internally for the actual
    model implementation.
    
    Example:
        >>> from dfm_python.lightning import DFMDataModule
        >>> from sktime.transformations.compose import ColumnTransformer
        >>> 
        >>> model = DFM()
        >>> model.load_config('config.yaml')
        >>> 
        >>> # Create transformer (user must provide)
        >>> transformer = ColumnTransformer([...])  # User-defined
        >>> 
        >>> # Create DataModule
        >>> data_module = DFMDataModule(config=model.config, transformer=transformer, data_path='data.csv')
        >>> data_module.setup()
        >>> 
        >>> # Train
        >>> model.train(data_module, max_iter=100)
        >>> Xf, Zf = model.predict(horizon=6)
    """
    
    def __init__(self):
        """Initialize DFM instance."""
        super().__init__()
        self._model_impl = DFMLinear()
        self._data_module: Optional['DFMDataModule'] = None
        self._nowcast: Optional['Nowcast'] = None
    
    @property
    def nowcast(self) -> 'Nowcast':
        """Get nowcasting manager instance."""
        if self._nowcast is None:
            _validate_config_loaded(self._config)
            if self._data_module is None:
                raise ValueError("DataModule must be provided via train() before accessing nowcast")
            _validate_result_loaded(self._result)
            from ..nowcast import Nowcast
            self._nowcast = Nowcast(model=self, data_module=self._data_module)
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
    ) -> 'DFM':
        """Load configuration from various sources."""
        # Handle base and override merging
        if base is not None or override is not None:
            if base is None:
                raise ValueError("base must be provided when override is specified")
            base_source = make_config_source(source=base)
            override_source = make_config_source(source=override) if override is not None else None
            if override_source is not None:
                config_source = MergedConfigSource(base_source, override_source)
            else:
                config_source = base_source
        else:
            config_source = make_config_source(
                source=source,
                yaml=yaml,
                mapping=mapping,
                hydra=hydra,
            )
        self._config = config_source.load()
        return self
    
    
    def fit(self, data_module: 'DFMDataModule', config: DFMConfig, **kwargs) -> DFMResult:
        """Fit the linear DFM model (implements abstract method from BaseFactorModel)."""
        self._config = config
        self._data_module = data_module
        self._result = self._model_impl.fit(data_module, config, **kwargs)
        return self._result
    
    def train(
        self,
        data_module: 'DFMDataModule',
        fit_params: Optional[FitParams] = None,
        **kwargs
    ) -> 'DFM':
        """Train the linear DFM model.
        
        Parameters
        ----------
        data_module : DFMDataModule
            DataModule containing preprocessed data. Must have setup() called.
        fit_params : FitParams, optional
            Fit parameters object
        **kwargs
            Additional parameters
        """
        from ..lightning import DFMDataModule
        _validate_config_loaded(self._config)
        
        if not isinstance(data_module, DFMDataModule):
            raise TypeError(f"data_module must be DFMDataModule, got {type(data_module)}")
        
        # Ensure DataModule is set up
        if data_module.data_processed is None:
            data_module.setup()
        
        self._data_module = data_module
        self._result = self._model_impl.fit(
            data_module,
            self._config,
            fit_params=fit_params,
            **kwargs
        )
        return self
    
    def predict(
        self,
        horizon: Optional[int] = None,
        *,
        return_series: bool = True,
        return_factors: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Forecast future values (implements abstract method from BaseFactorModel)."""
        _validate_result_loaded(self._result)
        return self._model_impl.predict(
            horizon=horizon,
            return_series=return_series,
            return_factors=return_factors
        )
    
    def plot(self, **kwargs) -> 'DFM':
        """Plot common visualizations."""
        _validate_result_loaded(self._result)
        _logger.info("Plot functionality not yet implemented")
        return self
    
    def reset(self) -> 'DFM':
        """Reset model state."""
        self._config = None
        self._data_module = None
        self._result = None
        self._nowcast = None
        return self
    
    def load_pickle(self, path: Union[str, Path], **kwargs) -> 'DFM':
        """Load a saved model from pickle file.
        
        Note: DataModule is not saved in pickle. Users must create a new DataModule
        and call train() with it after loading the model.
        """
        import pickle
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        self._config = payload.get('config')
        self._result = payload.get('result')
        # Note: data_module is not loaded - users must provide it via train()
        return self


# ============================================================================
# Module-level convenience functions (instance-based API)
# ============================================================================



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


# Module-level convenience functions (create new instances)
def load_config(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    hydra: Optional[Union[Dict[str, Any], Any]] = None,
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DFM:
    """Load configuration (module-level convenience function).
    
    Creates a new DFM instance and loads configuration.
    For stateful usage, create a DFM() instance directly.
    """
    model = DFM()
    return model.load_config(
        source=source,
        yaml=yaml,
        mapping=mapping,
        hydra=hydra,
        base=base,
        override=override,
    )


def train(data_module: 'DFMDataModule', fit_params: Optional[FitParams] = None, **kwargs) -> DFM:
    """Train the model (module-level convenience function).
    
    Note: This creates a new instance. For stateful usage, create a DFM() instance directly.
    """
    from ..config.params import FitParams
    if fit_params is None:
        fit_params = FitParams.from_kwargs(**kwargs)
    model = DFM()
    return model.train(data_module, fit_params=fit_params)


def predict(
    horizon: Optional[int] = None,
    *,
    return_series: bool = True,
    return_factors: bool = True
):
    """Forecast using the trained model (module-level convenience function)."""
    model = DFM()
    return model.predict(
        horizon=horizon,
        return_series=return_series,
        return_factors=return_factors
    )


def plot(**kwargs):
    """Plot common visualizations (module-level convenience function)."""
    model = DFM()
    return model.plot(**kwargs)


def load_pickle(path: Union[str, Path], **kwargs) -> DFM:
    """Load a saved model payload (module-level convenience function)."""
    model = DFM()
    return model.load_pickle(path, **kwargs)


def reset() -> DFM:
    """Reset state (module-level convenience function)."""
    return DFM()  # Reset = new instance


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
        
    Returns
    -------
    BaseFactorModel
        Model instance (DFMLinear or DDFMModel)
    """
    model_type = model_type.lower()
    
    if model_type in ('dfm', 'linear'):
        return DFMLinear(**kwargs)
    elif model_type in ('ddfm', 'deep'):
        try:
            from .ddfm import DDFMModel
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
    model = DFM()
    return model.load_config(yaml=yaml_path)


def from_spec(
    csv_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    series_filename: Optional[str] = None,
    blocks_filename: Optional[str] = None
) -> Tuple[Path, Path]:
    """Convert spec CSV file to YAML configuration files."""
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
    
    import logging
    logging.info("✓ Spec DataFrame converted to YAML:")
    logging.info(f"  - Series YAML: {series_path}")
    logging.info(f"  - Blocks YAML: {blocks_path}")
    logging.info(f"  - Main config : {main_config_path}")
    
    model = DFM()
    model.load_config(yaml=main_config_path)
    return model


def from_dict(mapping: Dict[str, Any]) -> DFM:
    """Load configuration from dictionary (convenience constructor)."""
    model = DFM()
    return model.load_config(mapping=mapping)


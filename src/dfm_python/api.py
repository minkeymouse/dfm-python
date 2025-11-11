"""High-level API for Dynamic Factor Model (convenience layer).

This module provides an object-oriented interface for DFM estimation
on top of the core `DFM` in `dfm.py`. Use this for a simpler, more
intuitive workflow:

Example:
    >>> import dfm_python as dfm
    >>> dfm.load_config('config/default.yaml')
    >>> dfm.load_data('data/sample_data.csv')
    >>> dfm.train(max_iter=1)
    >>> # Forecast and plot
    >>> Xf, Zf = dfm.predict(horizon=6)
    >>> dfm.plot(kind='factor', factor_index=0, forecast_horizon=6, save_path='outputs/factor_forecast.png')
    >>> # Access results
    >>> result = dfm.get_result()
    >>> factors = result.Z
"""

from typing import Optional, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from .config import DFMConfig, Params
from .data import load_data as _load_data
from .config import (
    make_config_source,
    ConfigSource,
    MergedConfigSource,
)
from .dfm import DFM as _DFMCore, DFMResult
from .core.helpers import safe_get_method, safe_get_attr


class DFM(_DFMCore):
    """High-level API for Dynamic Factor Model estimation.
    
    This class provides a simple, object-oriented interface for DFM operations.
    It maintains state (config, data, results) and provides convenient methods
    for loading configuration, data, training, prediction, and plotting.
    
    Example:
        >>> import dfm_python as dfm
        >>> dfm.load_config('config/default.yaml')
        >>> dfm.load_data('data/sample_data.csv')
        >>> dfm.train(max_iter=1)
        >>> Xf, Zf = dfm.predict(horizon=6)
        >>> dfm.plot(kind='factor', factor_index=0, forecast_horizon=6, save_path='outputs/factor_forecast.png')
        >>> print(dfm.get_result().converged)
        >>> factors = dfm.get_result().Z
    """
    
    def __init__(self):
        """Initialize DFM instance with empty state."""
        super().__init__()
        self._time: Optional[np.ndarray] = None
        self._original_data: Optional[np.ndarray] = None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get current data matrix (T x N)."""
        return self._data
    
    @property
    def time(self) -> Optional[np.ndarray]:
        """Get time index for data."""
        return self._time
    
    @property
    def original_data(self) -> Optional[np.ndarray]:
        """Get original (untransformed) data matrix."""
        return self._original_data
    
    def load_config(
        self,
        source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource, Any]] = None,
        *,
        yaml: Optional[Union[str, Path]] = None,
        mapping: Optional[Union[Dict[str, Any], Any]] = None,
        spec: Optional[Union[str, Path]] = None,
        hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        config: Optional[Any] = None,
    ) -> 'DFM':
        """Load configuration from various sources.
        
        Unified interface for YAML files, dictionaries, spec CSV, or Hydra configs.
        Supports merging configurations (e.g., YAML base + Spec CSV series).
        """
        # Handle direct DFMConfig
        if isinstance(source, DFMConfig):
            self._config = source
            return self
        
        # Support (mapping + spec) convenience: treat as base+override
        if (mapping is not None or config is not None) and spec is not None:
            base_source = make_config_source(mapping=mapping if mapping is not None else config)
            override_source = make_config_source(spec=spec)
            merged_source = MergedConfigSource(base_source, override_source)
            self._config = merged_source.load()
            return self
        
        # Handle merging
        if base is not None or override is not None:
            if base is None or override is None:
                raise ValueError("Both 'base' and 'override' must be provided for merging.")
            base_source = make_config_source(base)
            override_source = make_config_source(override)
            merged_source = MergedConfigSource(base_source, override_source)
            self._config = merged_source.load()
            return self
        
        # Create appropriate source adapter
        source_adapter = make_config_source(
            source=source,
            yaml=yaml,
            mapping=mapping if mapping is not None else config,
            spec=spec,
            hydra=hydra,
        )
        
        self._config = source_adapter.load()
        return self
    
    def load_data(self, 
                  data_path: Optional[Union[str, Path]] = None,
                  data: Optional[np.ndarray] = None,
                  **kwargs) -> 'DFM':
        """Load data from file or use provided array."""
        if self._config is None:
            raise ValueError("Configuration must be loaded before loading data. "
                           "Call load_config() first.")
        
        if data_path is not None:
            self._data, self._time, self._original_data = _load_data(
                data_path, self._config, **kwargs
            )
        elif data is not None:
            # If data is provided directly, we still need a time index
            self._data = data
            if 'time' in kwargs:
                self._time = kwargs['time']
            else:
                # Generate default monthly-end index
                self._time = pd.date_range(
                    start='2000-01-01', 
                    periods=len(data), 
                    freq='ME'
                )
            self._original_data = data
        else:
            raise ValueError("Either data_path or data must be provided.")
        
        return self
    
    def train(self, 
              threshold: Optional[float] = None,
              max_iter: Optional[int] = None,
              **kwargs) -> 'DFM':
        """Train the DFM model using EM algorithm."""
        if self._config is None:
            raise ValueError("Configuration must be loaded before training. "
                           "Call load_config() first.")
        if self._data is None:
            raise ValueError("Data must be loaded before training. "
                           "Call load_data() first.")
        
        self._result = self.fit(
            self._data,
            self._config,
            threshold=threshold,
            max_iter=max_iter,
            **kwargs
        )
        # Attach metadata for convenient OOP access
        if self._time is not None:
            self._result.time_index = self._time
        series_ids = safe_get_method(self._config, 'get_series_ids')
        if series_ids is not None:
            self._result.series_ids = series_ids
        block_names = safe_get_attr(self._config, 'block_names')
        if block_names is not None:
            self._result.block_names = block_names
        return self
    
    def reset(self) -> 'DFM':
        """Reset all state (config, data, results)."""
        super().__init__()
        self._time = None
        self._original_data = None
        return self
    
    # ---------------------------------------------------------------------
    # Inference API
    # ---------------------------------------------------------------------
    def predict(self, horizon: int = 12, *, return_series: bool = True, return_factors: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Forecast series and/or factors for a given horizon using the trained model.
        
        Parameters
        ----------
        horizon : int
            Number of periods ahead to forecast (default: 12)
        return_series : bool
            Whether to return forecasted series (default: True)
        return_factors : bool
            Whether to return forecasted factors (default: True)
            
        Returns
        -------
        If both return_series and return_factors are True:
            Tuple[np.ndarray, np.ndarray]
                (X_forecast, Z_forecast) where X_forecast is (horizon x N) and Z_forecast is (horizon x m)
        If only return_series is True:
            np.ndarray
                X_forecast (horizon x N)
        If only return_factors is True:
            np.ndarray
                Z_forecast (horizon x m)
        """
        if self._result is None:
            raise ValueError("Model must be trained before calling predict(). Call train() first.")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        A = self._result.A
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]
        
        # Forecast latent factors: Z_{t+h} = A^h Z_t (deterministic)
        Z_forecast = np.zeros((horizon, Z_last.shape[0]))
        Z_forecast[0, :] = A @ Z_last
        for h in range(1, horizon):
            Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        
        # Forecast observed standardized series then denormalize
        X_forecast_std = Z_forecast @ C.T
        X_forecast = X_forecast_std * Wx + Mx
        
        if return_series and return_factors:
            return X_forecast, Z_forecast
        if return_series:
            return X_forecast
        return Z_forecast
    
    def plot(self, *, kind: str = 'factor', factor_index: int = 0, forecast_horizon: Optional[int] = None,
             save_path: Optional[Union[str, Path]] = None, show: bool = False):
        """Quick plotting utility for common visualizations."""
        if self._result is None:
            raise ValueError("No results to plot. Train the model first.")
        try:
            import matplotlib.pyplot as plt  # Local import to avoid hard dependency at import time
        except Exception as e:
            raise RuntimeError(f"matplotlib is required for plotting: {e}")
        
        Z = self._result.Z
        if factor_index < 0 or factor_index >= Z.shape[1]:
            raise ValueError(f"factor_index out of range: {factor_index} (num factors: {Z.shape[1]})")
        
        # Build time axis
        if self._time is not None:
            time_hist = pd.to_datetime(self._time)
        else:
            time_hist = pd.date_range(start='2000-01-01', periods=Z.shape[0], freq='ME')
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_hist, Z[:, factor_index], 'b-', linewidth=2, label='Historical factor')
        
        if forecast_horizon is not None and forecast_horizon > 0:
            Z_fore = self.predict(forecast_horizon, return_series=False, return_factors=True)
            # Build forecast dates
            last_date = time_hist.iloc[-1] if hasattr(time_hist, 'iloc') else time_hist[-1]
            # Use same monthly-end spacing; if freq missing, fallback to monthly-end
            try:
                forecast_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset('ME'),
                                               periods=forecast_horizon, freq='ME')
            except Exception:
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=30),
                                               periods=forecast_horizon, freq='ME')
            plt.plot(forecast_dates, Z_fore[:, factor_index], 'r--', linewidth=2, label='Forecast factor')
            plt.axvline(x=last_date, color='gray', linestyle=':', linewidth=1, label='Forecast start')
        
        title_block = None
        if hasattr(self._result, 'block_names') and self._result.block_names:
            title_block = self._result.block_names[0] if factor_index == 0 else None
        plt.title(f"Factor {factor_index}" + (f" ({title_block})" if title_block else ""))
        plt.xlabel("Date")
        plt.ylabel("Factor value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        out_path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            out_path = save_path
        if show:
            plt.show()
        else:
            plt.close()
        return out_path


# Create a singleton instance for module-level usage
_dfm_instance = DFM()


# Module-level convenience functions that delegate to the singleton
def load_config(
    source: Optional[Union[str, Path, Dict[str, Any], DFMConfig, ConfigSource]] = None,
    *,
    yaml: Optional[Union[str, Path]] = None,
    mapping: Optional[Dict[str, Any]] = None,
    spec: Optional[Union[str, Path]] = None,
    hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
    base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
    override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
) -> DFM:
    """Load configuration (module-level convenience function)."""
    return _dfm_instance.load_config(
        source=source,
        yaml=yaml,
        mapping=mapping,
        spec=spec,
        hydra=hydra,
        base=base,
        override=override,
    )


def load_data(data_path: Optional[Union[str, Path]] = None,
               data: Optional[np.ndarray] = None,
               **kwargs) -> DFM:
    """Load data (module-level convenience function)."""
    return _dfm_instance.load_data(data_path=data_path, data=data, **kwargs)


def train(threshold: Optional[float] = None,
          max_iter: Optional[int] = None,
          **kwargs) -> DFM:
    """Train the model (module-level convenience function)."""
    return _dfm_instance.train(threshold=threshold, max_iter=max_iter, **kwargs)


def predict(horizon: int = 12, **kwargs):
    """Forecast using the trained model (module-level convenience function)."""
    return _dfm_instance.predict(horizon=horizon, **kwargs)


def plot(**kwargs):
    """Plot common visualizations (module-level convenience function)."""
    return _dfm_instance.plot(**kwargs)


def reset() -> DFM:
    """Reset state (module-level convenience function)."""
    return _dfm_instance.reset()


# Convenience constructors for cleaner API
def from_yaml(yaml_path: Union[str, Path]) -> DFM:
    """Load configuration from YAML file (convenience constructor)."""
    return _dfm_instance.load_config(yaml=yaml_path)


def from_spec(spec_path: Union[str, Path], params: Optional[Params] = None) -> DFM:
    """Load configuration from spec CSV file (convenience constructor)."""
    from .config import SpecCSVSource
    
    if params is None:
        params = Params()
    
    # Load config from spec CSV using ConfigSource pattern
    spec_source = SpecCSVSource(spec_path)
    config = spec_source.load()
    
    # Override estimation parameters from params
    for key, value in params.__dict__.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    _dfm_instance._config = config
    return _dfm_instance


def from_spec_df(spec_df: pd.DataFrame, params: Optional[Params] = None) -> DFM:
    """Load configuration from spec DataFrame (convenience constructor)."""
    from .data import _load_config_from_dataframe
    
    if params is None:
        params = Params()
    
    # Load config from DataFrame (creates DFMConfig with series/blocks from spec)
    config = _load_config_from_dataframe(spec_df)
    
    # Override estimation parameters from params
    for key, value in params.__dict__.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    _dfm_instance._config = config
    return _dfm_instance


def from_dict(mapping: Dict[str, Any]) -> DFM:
    """Load configuration from dictionary (convenience constructor)."""
    return _dfm_instance.load_config(mapping=mapping)


# Expose singleton instance for direct access
# Users can access: dfm.config, dfm.data, dfm.result, etc.
__all__ = ['DFM', 'load_config', 'load_data', 'train', 'predict', 'plot', 'reset', 
           'from_yaml', 'from_spec', 'from_spec_df', 'from_dict']



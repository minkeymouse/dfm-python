"""High-level model API classes for Dynamic Factor Models.

This module contains the DFMBase, DFM, and DDFM classes that provide
object-oriented interfaces for DFM estimation.
"""

import os
import pickle
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
from ..dataloader.loader import load_data as _load_data, DataView, create_data_view
from ..models.dfm import DFMLinear
# DFMCore is now an alias for DFMLinear (backward compatibility)
DFMCore = DFMLinear
from ..core.results import DFMResult
from ..nowcasting import Nowcast
from ..core.helpers import (
    safe_get_method,
    safe_get_attr,
    find_series_index,
    get_clock_frequency,
    _validate_config_loaded,
    _validate_data_loaded,
    _validate_result_loaded,
)
from ..core.time import (
    TimeIndex,
    datetime_range,
    get_next_period_end,
    clock_to_datetime_freq,
    get_latest_time,
    convert_to_timestamp,
    find_time_index,
    extract_last_date,
)


def _convert_time_index_to_list(time_index: Optional[Any]) -> List[Any]:
    """Convert time index to list format for plotting/display.
    
    Parameters
    ----------
    time_index : TimeIndex, list, or compatible
        Time index to convert
        
    Returns
    -------
    List[Any]
        List representation of time index
    """
    if time_index is None:
        return []
    if isinstance(time_index, TimeIndex):
        return time_index.to_list()
    if hasattr(time_index, '__iter__') and not isinstance(time_index, (str, bytes)):
        return list(time_index)
    # Fallback: try to index
    try:
        return [time_index[i] for i in range(len(time_index))]
    except (TypeError, AttributeError):
        return []


class DFMBase(DFMLinear):
    """Base class for high-level DFM API classes.
    
    This class contains shared functionality for both linear DFM and DDFM
    high-level API classes, including configuration loading, data loading,
    prediction, plotting, and nowcasting.
    """
    
    def __init__(self):
        """Initialize base DFM instance with empty state."""
        super().__init__()
        self._time: Optional[TimeIndex] = None
        self._original_data: Optional[np.ndarray] = None
        self._data_frame: Optional[pl.DataFrame] = None
        self._nowcast_instance: Optional[Nowcast] = None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get current data matrix (T x N)."""
        return self._data
    
    @property
    def data_frame(self) -> Optional[pl.DataFrame]:
        """Get current data as polars DataFrame."""
        return self._data_frame
    
    @property
    def time(self) -> Optional[TimeIndex]:
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
        hydra: Optional[Union[Dict[str, Any], 'DictConfig']] = None,
        base: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        override: Optional[Union[str, Path, Dict[str, Any], ConfigSource]] = None,
        config: Optional[Any] = None,
    ) -> 'DFMBase':
        """Load configuration from various sources.
        
        Unified interface for YAML files, dictionaries, spec CSV, or Hydra configs.
        Supports merging configurations (e.g., YAML base + Spec CSV series).
        """
        # Handle direct DFMConfig
        if isinstance(source, DFMConfig):
            self._config = source
            self._invalidate_nowcast_cache()
            return self
        
        # Handle merging
        if base is not None or override is not None:
            if base is None or override is None:
                raise ValueError("Both 'base' and 'override' must be provided for merging.")
            base_source = make_config_source(base)
            override_source = make_config_source(override)
            merged_source = MergedConfigSource(base_source, override_source)
            self._config = merged_source.load()
            self._invalidate_nowcast_cache()
            return self
        
        # Create appropriate source adapter
        source_adapter = make_config_source(
            source=source,
            yaml=yaml,
            mapping=mapping if mapping is not None else config,
            spec=None,
            hydra=hydra,
        )
        
        self._config = source_adapter.load()
        self._invalidate_nowcast_cache()
        return self
    
    def load_data(self, 
                  data_path: Optional[Union[str, Path]] = None,
                  data: Optional[np.ndarray] = None,
                  **kwargs) -> 'DFMBase':
        """Load data from file or use provided array."""
        _validate_config_loaded(self._config, "config")
        
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
                # Generate default time index based on clock frequency
                clock = get_clock_frequency(self._config, 'm')
                datetime_freq = clock_to_datetime_freq(clock)
                # Use current date as starting point, or extract from data if available
                if self._time is not None:
                    start_date = get_latest_time(self._time)
                else:
                    # Default to reasonable starting point
                    start_date = datetime(2000, 1, 1)
                self._time = TimeIndex(datetime_range(start=start_date, periods=len(data), freq=datetime_freq))
            self._original_data = data
        else:
            raise ValueError("Either data_path or data must be provided.")
        
        self._refresh_data_frame()
        self._invalidate_nowcast_cache()
        return self
    
    def load_pickle(
        self,
        path: Union[str, Path],
        *,
        load_data_path: Optional[Union[str, Path]] = None,
        data: Optional[np.ndarray] = None,
        time_index: Optional[Union[TimeIndex, Sequence[Any], np.ndarray]] = None,
        original_data: Optional[np.ndarray] = None,
    ) -> 'DFMBase':
        """Load a previously saved DFMResult (and optional data) for fast inference.
        
        Parameters
        ----------
        path : str or Path
            Path to pickle file produced by ``DFMResult.save`` or the auto-save payload.
        load_data_path : str or Path, optional
            Optional data file to load via ``load_data`` after restoring the model.
        data : np.ndarray, optional
            Provide data directly (alternative to ``load_data_path``). If provided,
            ``time_index`` should also be supplied (or will be generated if possible).
        time_index : TimeIndex or sequence, optional
            Time index corresponding to ``data``. If omitted, falls back to
            ``result.time_index`` when available.
        original_data : np.ndarray, optional
            Optional original (untransformed) data matrix. Useful when passing
            standardized arrays to ``data`` but wanting to preserve originals.
        
        Returns
        -------
        DFMBase
            The current instance with restored state.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {path}")
        
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        
        result: Optional[DFMResult] = None
        config: Optional[DFMConfig] = None
        
        if isinstance(payload, dict) and 'result' in payload:
            result = payload.get('result')
            config = payload.get('config')
        elif isinstance(payload, DFMResult):
            result = payload
        else:
            raise ValueError(
                "Unsupported pickle payload. Expected DFMResult or "
                "dict with 'result' (and optional 'config')."
            )
        
        if result is None:
            raise ValueError("Pickle payload does not contain a DFMResult.")
        
        if config is not None:
            self._config = config
        elif self._config is None:
            raise ValueError(
                "Configuration missing in pickle payload. Load configuration "
                "before calling load_pickle or save with config included."
            )
        
        # Ensure data/time are available (optional but recommended for nowcast/backtest)
        if load_data_path is not None:
            self.load_data(data_path=load_data_path)
        elif data is not None:
            load_kwargs: Dict[str, Any] = {}
            if time_index is not None:
                load_kwargs['time'] = time_index if isinstance(time_index, TimeIndex) else TimeIndex(time_index)
            self.load_data(data=data, **load_kwargs)
            if original_data is not None:
                self._original_data = original_data
        else:
            if time_index is not None:
                self._time = time_index if isinstance(time_index, TimeIndex) else TimeIndex(time_index)
            elif getattr(result, 'time_index', None) is not None:
                stored_time = result.time_index
                self._time = stored_time if isinstance(stored_time, TimeIndex) else TimeIndex(stored_time)
            if original_data is not None:
                self._original_data = original_data
        
        self._result = result
        if self._time is not None:
            try:
                self._result.time_index = self._time
            except AttributeError:
                pass
        
        if self._config is not None:
            series_ids = safe_get_method(self._config, 'get_series_ids')
            if series_ids is not None:
                self._result.series_ids = series_ids
            block_names = safe_get_attr(self._config, 'block_names')
            if block_names is not None:
                self._result.block_names = block_names
        
        self._refresh_data_frame()
        self._invalidate_nowcast_cache()
        return self
    
    def _invalidate_nowcast_cache(self) -> None:
        """Invalidate cached Nowcast instance.
        
        Called when model state changes (training, data loading, config changes)
        to ensure Nowcast instance uses latest data.
        """
        self._nowcast_instance = None
    
    def _refresh_data_frame(self) -> None:
        """Sync polars representation of data for faster view creation."""
        if self._data is None:
            self._data_frame = None
            return
        try:
            series_ids = safe_get_method(self._config, 'get_series_ids')
        except Exception:
            series_ids = None
        if series_ids is None:
            schema = [f'series_{i}' for i in range(self._data.shape[1])]
        else:
            schema = list(series_ids[:self._data.shape[1]])
        self._data_frame = pl.DataFrame(self._data, schema=schema)
    
    def reset(self) -> 'DFMBase':
        """Reset all state (config, data, results)."""
        super().__init__()
        self._time = None
        self._original_data = None
        self._data_frame = None
        self._invalidate_nowcast_cache()
        return self
    
    def _auto_save_result(self) -> None:
        """Persist the latest DFMResult next to Hydra outputs (if available)."""
        if self._result is None or self._config is None:
            return
        
        run_dir_env = os.environ.get('HYDRA_RUN_DIR')
        if run_dir_env:
            base_dir = Path(run_dir_env)
        else:
            base_dir = Path('outputs') / 'latest'
        base_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = base_dir / 'model_result.pkl'
        payload = {
            'result': self._result,
            'config': self._config,
            'timestamp': datetime.now().isoformat(),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)
        
        _logger.info(f"✓ Model result auto-saved to {save_path}")
    
    def predict(self, horizon: Optional[int] = None, *, return_series: bool = True, return_factors: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Forecast series and/or factors for a given horizon using the trained model.
        
        This method performs deterministic forecasting by iteratively applying the
        transition matrix A to the last known factor state. This is distinct from
        nowcast() which uses Kalman filtering to estimate current/recent values
        considering data availability constraints.
        
        Note: This method uses a deterministic forecast approach (A matrix iteration).
        For nowcasting with data availability constraints, use model.nowcast() which
        leverages the shared forecast core (_forecast_core) with Kalman filtering.
        
        Parameters
        ----------
        horizon : int, optional
            Number of periods ahead to forecast. If None, defaults to 1 period based on clock frequency.
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
        _validate_result_loaded(self._result)
        
        # Default horizon: 1 period based on clock frequency (generic)
        if horizon is None:
            from ..core.structure import get_periods_per_year
            clock = get_clock_frequency(self._config, 'm')
            # Default to 1 year worth of periods based on clock frequency
            horizon = get_periods_per_year(clock)
        
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        
        # Extract model parameters
        A = self._result.A
        C = self._result.C
        Wx = self._result.Wx
        Mx = self._result.Mx
        Z_last = self._result.Z[-1, :]
        p = self._result.p  # VAR order
        
        # Deterministic forecast: handle VAR(1) and VAR(2)
        if p == 1:
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            Z_forecast[0, :] = A @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A @ Z_forecast[h - 1, :]
        elif p == 2:
            # VAR(2): need last two factor values
            if self._result.Z.shape[0] < 2:
                # Fallback to VAR(1) if not enough history
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                A1 = A[:, :Z_last.shape[0]]
                Z_forecast[0, :] = A1 @ Z_last
                for h in range(1, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
            else:
                Z_prev = self._result.Z[-2, :]  # f_{t-2}
                A1 = A[:, :Z_last.shape[0]]
                A2 = A[:, Z_last.shape[0]:]
                Z_forecast = np.zeros((horizon, Z_last.shape[0]))
                Z_forecast[0, :] = A1 @ Z_last + A2 @ Z_prev
                if horizon > 1:
                    Z_forecast[1, :] = A1 @ Z_forecast[0, :] + A2 @ Z_last
                for h in range(2, horizon):
                    Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :] + A2 @ Z_forecast[h - 2, :]
        else:
            # Fallback to VAR(1) behavior
            Z_forecast = np.zeros((horizon, Z_last.shape[0]))
            A1 = A[:, :Z_last.shape[0]] if A.shape[1] > Z_last.shape[0] else A
            Z_forecast[0, :] = A1 @ Z_last
            for h in range(1, horizon):
                Z_forecast[h, :] = A1 @ Z_forecast[h - 1, :]
        
        # Transform factors to observed series: X = Z @ C^T, then denormalize
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
        _validate_result_loaded(self._result)
        try:
            import matplotlib.pyplot as plt  # Local import to avoid hard dependency at import time
        except Exception as e:
            raise RuntimeError(f"matplotlib is required for plotting: {e}")
        
        Z = self._result.Z
        if factor_index < 0 or factor_index >= Z.shape[1]:
            raise ValueError(f"factor_index out of range: {factor_index} (num factors: {Z.shape[1]})")
        
        # Build time axis
        if self._time is not None:
            time_hist = _convert_time_index_to_list(self._time)
        else:
            # Generate default time index based on clock frequency
            clock = get_clock_frequency(self._config, 'm')
            datetime_freq = clock_to_datetime_freq(clock)
            start_date = datetime(2000, 1, 1)
            time_hist = datetime_range(start=start_date, periods=Z.shape[0], freq=datetime_freq)
        
        plt.figure(figsize=(12, 4))
        plt.plot(time_hist, Z[:, factor_index], 'b-', linewidth=2, label='Historical factor')
        
        if forecast_horizon is not None and forecast_horizon > 0:
            Z_fore = self.predict(forecast_horizon, return_series=False, return_factors=True)
            # Build forecast dates (generic extraction)
            last_date = extract_last_date(time_hist)
            
            # Calculate next period end (generic based on clock frequency)
            clock = get_clock_frequency(self._config, 'm')
            next_period = get_next_period_end(last_date, clock)
            # Map clock to datetime frequency (use shared mapping)
            datetime_freq = clock_to_datetime_freq(clock)
            forecast_dates = datetime_range(start=next_period, periods=forecast_horizon, freq=datetime_freq)
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
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @property
    def nowcast(self) -> Nowcast:
        """Get Nowcast manager instance (cached).
        
        Returns
        -------
        Nowcast
            Nowcast manager instance for this model (cached and reused)
            
        Examples
        --------
        >>> model = DFM()
        >>> model.load_config('config/default.yaml')
        >>> model.load_data('data/sample_data.csv')
        >>> model.train()
        >>> 
        >>> # Get Nowcast instance and calculate nowcast
        >>> value = model.nowcast('gdp', view_date='2024-01-15', target_period='2024Q1')
        >>> 
        >>> # Reuse instance for news decomposition (same instance, cache preserved)
        >>> news = model.nowcast.decompose('gdp', '2024Q1', '2024-01-15', '2024-02-15')
        """
        # Validate model state
        _validate_result_loaded(self._result)
        _validate_data_loaded(self._data, "data")
        if self._time is None:
            raise ValueError("Time index must be loaded. Call load_data() first.")
        _validate_config_loaded(self._config, "config")
        
        # Return cached instance or create new one
        if self._nowcast_instance is None:
            self._nowcast_instance = Nowcast(self)
        return self._nowcast_instance
    
    def generate_dataset(
        self,
        target_series: str,
        periods: List[datetime],
        backward: int = 0,
        forward: int = 0,
        dataview: Optional[DataView] = None
    ) -> Dict[str, Any]:
        """Generate dataset for DFM evaluation and model training.
        
        Parameters
        ----------
        target_series : str
            Target series ID
        periods : List[datetime]
            List of periods to evaluate
        backward : int, default 0
            Number of clock steps for backward data views
        forward : int, default 0
            Number of clock steps for forward forecast (not used in evaluation)
        dataview : DataView, optional
            Base data view descriptor to materialize views from custom sources.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'X': np.ndarray (n_samples, n_features) - feature vectors
            - 'y_baseline': np.ndarray (n_samples,) - DFM baseline predictions
            - 'y_actual': np.ndarray (n_samples,) - actual values
            - 'y_target': np.ndarray (n_samples,) - target for model (y_actual - y_baseline)
            - 'metadata': List[Dict] - metadata for each sample
            - 'backward_results': List[Dict] - backward results for each period (if backward > 0)
        """
        _validate_result_loaded(self._result)
        _validate_data_loaded(self._data, "data")
        if self._time is None:
            raise ValueError("Time index must be loaded. Call load_data() first.")
        _validate_config_loaded(self._config, "config")
        
        # Get target series index using helper function
        i_series = find_series_index(self._config, target_series)
        
        # Initialize result arrays
        n_samples = len(periods)
        X_features = []  # Will be converted to array after feature extraction
        y_baseline = []
        y_actual = []
        metadata = []
        backward_results = []
        
        if dataview is not None:
            dataview_factory = dataview
        else:
            dataview_factory = DataView.from_arrays(
                X=self._data,
                Time=self._time,
                Z=self._original_data,
                config=self._config,
                X_frame=self._data_frame
            )
        if dataview_factory.config is None:
            dataview_factory.config = self._config
        
        # For each period
        for period in periods:
            # Create data view at this period
            view_obj = dataview_factory.with_view_date(period)
            X_view, Time_view, _ = view_obj.materialize()
            
            # Calculate nowcast (baseline)
            if backward > 0:
                # Use backward's last value as baseline
                # Generate backward data views first
                nowcasts = []
                data_view_dates = []
                
                for weeks_back in range(backward, -1, -1):
                    data_view_date = period - timedelta(weeks=weeks_back)
                    
                    view_past = dataview_factory.with_view_date(data_view_date)
                    X_view_past, Time_view_past, _ = view_past.materialize()
                    
                    # Use Nowcast instance for calculation
                    nowcast_val = self.nowcast(
                        target_series=target_series,
                        view_date=view_past.view_date or data_view_date,
                        target_period=period
                    )
                    
                    nowcasts.append(nowcast_val)
                    data_view_dates.append(view_past.view_date or data_view_date)
                
                baseline_nowcast = nowcasts[-1]  # Last (most recent) nowcast
                
                backward_results.append({
                    'nowcasts': np.array(nowcasts),
                    'data_view_dates': data_view_dates,
                    'target_date': period
                })
            else:
                # Direct nowcast calculation
                baseline_nowcast = self.nowcast(
                    target_series=target_series,
                    view_date=view_obj.view_date or period,
                    target_period=period
                )
            
            y_baseline.append(baseline_nowcast)
            
            # Get actual value (if available) using helper function
            
            t_idx = find_time_index(self._time, period)
            actual_val = np.nan
            if t_idx is not None and t_idx < self._data.shape[0] and i_series < self._data.shape[1]:
                actual_val = self._data[t_idx, i_series]
            
            y_actual.append(actual_val)
            
            # Extract features for model evaluation
            features = self._extract_features(X_view, Time_view, period)
            X_features.append(features)
            
            # Metadata
            metadata.append({
                'period': period,
                'target_series': target_series
            })
        
        # Convert to arrays
        X_features = np.array(X_features)
        y_baseline = np.array(y_baseline)
        y_actual = np.array(y_actual)
        y_target = y_actual - y_baseline  # Model target: prediction error
        
        return {
            'X': X_features,
            'y_baseline': y_baseline,
            'y_actual': y_actual,
            'y_target': y_target,
            'metadata': metadata,
            'backward_results': backward_results if backward > 0 else []
        }
    
    def _extract_features(
        self,
        X_view: np.ndarray,
        Time_view: Any,
        period: datetime
    ) -> np.ndarray:
        """Extract features from data view for model evaluation.
        
        Returns features including factors and residuals.
        """
        # Get latest factors from result
        if self._result is not None and hasattr(self._result, 'Z'):
            latest_factors = self._result.Z[-1, :] if self._result.Z.shape[0] > 0 else np.zeros(self._result.Z.shape[1])
        else:
            latest_factors = np.array([])
        
        # Calculate mean residuals (simplified)
        if X_view.shape[0] > 0:
            mean_residual = np.nanmean(X_view[-1, :]) if X_view.shape[0] > 0 else 0.0
        else:
            mean_residual = 0.0
        
        # Combine features
        features = np.concatenate([latest_factors, [mean_residual]])
        
        return features
    
    def get_state(
        self,
        t: Union[int, datetime],
        target_series: str,
        lookback: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get DFM state at time t for downstream model input.
        
        Parameters
        ----------
        t : int or datetime
            Time point
        target_series : str
            Target series ID
        lookback : int, optional
            Number of periods to look back for history.
            If None, defaults to 1 year worth of periods based on clock frequency
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'baseline_nowcast': float - current DFM nowcast
            - 'baseline_forecast': np.ndarray (lookback,) - past DFM forecasts
            - 'actual_history': np.ndarray (lookback,) - past actual values
            - 'residuals': np.ndarray (lookback,) - past prediction errors
            - 'factors': np.ndarray (n_factors,) - current factors
            - 'factors_history': np.ndarray (lookback, n_factors) - past factors
            - 'news_summary': Dict - news decomposition summary
            - 'features': np.ndarray (n_features,) - flattened feature vector
            - 'metadata': Dict - metadata
        """
        _validate_result_loaded(self._result)
        _validate_data_loaded(self._data, "data")
        if self._time is None:
            raise ValueError("Time index must be loaded. Call load_data() first.")
        _validate_config_loaded(self._config, "config")
        
        # Set default lookback based on clock frequency if not provided
        if lookback is None:
            from ..core.structure import get_periods_per_year
            clock = get_clock_frequency(self._config, 'm')
            lookback = get_periods_per_year(clock)
        
        # Convert t to datetime using helper function
        t = convert_to_timestamp(t, self._time, None)
        
        # Get target series index using helper function
        i_series = find_series_index(self._config, target_series)
        
        # Get current nowcast
        X_view, Time_view, _ = create_data_view(
            X=self._data,
            Time=self._time,
            Z=self._original_data,
            config=self._config,
            view_date=t
        )
        
        # Use Nowcast instance for calculation
        baseline_nowcast = self.nowcast(
            target_series=target_series,
            view_date=t,
            target_period=None
        )
        
        # Extract history
        baseline_forecast = []
        actual_history = []
        residuals = []
        factors_history = []
        
        # Find time index for t using helper function
        
        t_idx = find_time_index(self._time, t)
        if t_idx is None:
            raise ValueError(f"Time {t} not found in self._time")
        
        # Get lookback periods
        for i in range(max(0, t_idx - lookback + 1), t_idx + 1):
            if i < self._data.shape[0]:
                # Get forecast using current model state
                forecast_val = baseline_nowcast
                baseline_forecast.append(forecast_val)
                
                # Get actual
                actual_val = self._data[i, i_series] if i_series < self._data.shape[1] else np.nan
                actual_history.append(actual_val)
                
                # Calculate residual
                residual = actual_val - forecast_val if not np.isnan(actual_val) else np.nan
                residuals.append(residual)
                
                # Get factors
                if self._result is not None and hasattr(self._result, 'Z') and i < self._result.Z.shape[0]:
                    factors_history.append(self._result.Z[i, :])
                else:
                    factors_history.append(np.zeros(self._result.Z.shape[1]) if self._result is not None else np.array([]))
        
        # Pad if needed
        while len(baseline_forecast) < lookback:
            baseline_forecast.insert(0, np.nan)
            actual_history.insert(0, np.nan)
            residuals.insert(0, np.nan)
            factors_history.insert(0, np.zeros(factors_history[0].shape) if factors_history else np.array([]))
        
        # Get current factors
        if self._result is not None and hasattr(self._result, 'Z') and t_idx < self._result.Z.shape[0]:
            factors = self._result.Z[t_idx, :]
        else:
            factors = np.zeros(self._result.Z.shape[1]) if self._result is not None else np.array([])
        
        # News summary (basic implementation)
        news_summary = {
            'total_impact': 0.0,
            'top_contributors': [],
            'revision_impact': 0.0,
            'release_impact': 0.0
        }
        
        # Construct feature vector
        features = self._construct_feature_vector(
            factors=factors,
            residuals=np.array(residuals),
            news_summary=news_summary
        )
        
        # Data availability
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
                't': t_idx,
                'date': t,
                'target_series': target_series,
                'data_availability': {
                    'n_missing': int(n_missing),
                    'n_available': int(n_available),
                    'missing_series': []  # Would need to identify which series are missing
                }
            }
        }
    
    def _construct_feature_vector(
        self,
        factors: np.ndarray,
        residuals: np.ndarray,
        news_summary: Dict[str, Any]
    ) -> np.ndarray:
        """Construct feature vector from model state.
        
        Returns flattened feature vector combining factors, residuals, and news.
        """
        # Combine features
        feature_parts = [
            factors.flatten(),
            residuals.flatten(),
            np.array([news_summary.get('total_impact', 0.0)]),
            np.array([news_summary.get('revision_impact', 0.0)]),
            np.array([news_summary.get('release_impact', 0.0)])
        ]
        
        features = np.concatenate([part for part in feature_parts if part.size > 0])
        
        return features
    
    def get_result(self) -> Optional[DFMResult]:
        """Get the training result."""
        return self._result
    
    @property
    def config(self) -> Optional[DFMConfig]:
        """Get current configuration."""
        return self._config


class DFM(DFMBase):
    """High-level API for Linear Dynamic Factor Model estimation.
    
    This class provides a simple, object-oriented interface for linear DFM operations.
    It maintains state (config, data, results) and provides convenient methods
    for loading configuration, data, training, prediction, and plotting.
    
    This class uses the EM algorithm for linear DFM estimation. For Deep DFM,
    use the separate `DDFM` class.
    
    Example:
        >>> import dfm_python as dfm
        >>> model = dfm.DFM()
        >>> model.load_config('config/default.yaml')
        >>> model.load_data('data/sample_data.csv')
        >>> model.train(max_iter=100)
        >>> Xf, Zf = model.predict(horizon=6)
        >>> model.plot(kind='factor', factor_index=0, forecast_horizon=6, save_path='outputs/factor_forecast.png')
        >>> print(model.get_result().converged)
        >>> factors = model.get_result().Z
    """
    
    def __init__(self):
        """Initialize DFM instance for linear DFM only."""
        super().__init__()
    
    def train(self, 
              threshold: Optional[float] = None,
              max_iter: Optional[int] = None,
              **kwargs) -> 'DFM':
        """Train the linear DFM model using EM algorithm.
        
        Parameters
        ----------
        threshold : float, optional
            EM convergence threshold. If None, uses config.threshold.
        max_iter : int, optional
            Maximum EM iterations. If None, uses config.max_iter.
        **kwargs
            Additional parameters that override config values:
            - ar_lag: AR lag for factors
            - nan_method: Missing data handling method
            - And other DFM-specific parameters
            
        Returns
        -------
        DFM
            Self for method chaining.
        """
        from ..core.helpers import _validate_config_loaded, _validate_data_loaded
        
        _validate_config_loaded(self._config, "config")
        _validate_data_loaded(self._data, "data")
        
        # Use parent class fit() method (linear DFM from DFMCore)
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
        self._auto_save_result()
        self._invalidate_nowcast_cache()
        return self


class DDFM(DFMBase):
    """High-level API for Deep Dynamic Factor Model estimation.
    
    This class provides a simple, object-oriented interface for DDFM operations.
    It maintains state (config, data, results) and provides convenient methods
    for loading configuration, data, training, prediction, and plotting.
    
    This class uses an autoencoder (nonlinear encoder + linear decoder) for factor
    extraction, followed by Kalman filtering. For linear DFM, use the separate
    `DFM` class.
    
    Example:
        >>> import dfm_python as dfm
        >>> model = dfm.DDFM(encoder_layers=[64, 32], num_factors=2)
        >>> model.load_config('config/default.yaml')
        >>> model.load_data('data/sample_data.csv')
        >>> model.train(epochs=100, batch_size=32)
        >>> Xf, Zf = model.predict(horizon=6)
        >>> model.plot(kind='factor', factor_index=0, forecast_horizon=6)
        >>> factors = model.get_result().Z
    """
    
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
        """Initialize DDFM high-level API instance.
        
        Parameters
        ----------
        encoder_layers : List[int], optional
            Hidden layer dimensions for encoder. Default: [64, 32]
        num_factors : int, optional
            Number of factors. If None, inferred from config during fit.
        activation : str
            Activation function ('tanh', 'relu', 'sigmoid'). Default: 'tanh'
        use_batch_norm : bool
            Whether to use batch normalization in encoder. Default: True
        learning_rate : float
            Learning rate for Adam optimizer. Default: 0.001
        epochs : int
            Number of training epochs. Default: 100
        batch_size : int
            Batch size for training. Default: 32
        factor_order : int
            VAR lag order for factor dynamics (1 or 2). Default: 1
        use_idiosyncratic : bool
            Whether to model idiosyncratic components with AR(1) dynamics. Default: True
        min_obs_idio : int
            Minimum number of observations required for idio AR(1) estimation. Default: 5
        **kwargs
            Additional arguments (ignored for now, for future compatibility)
        """
        super().__init__()
        self._ddfm_model: Optional[Any] = None  # Will store models.ddfm.DDFM instance
        self._ddfm_kwargs = {
            'encoder_layers': encoder_layers,
            'num_factors': num_factors,
            'activation': activation,
            'use_batch_norm': use_batch_norm,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'factor_order': factor_order,
            'use_idiosyncratic': use_idiosyncratic,
            'min_obs_idio': min_obs_idio,
        }
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        factor_order: Optional[int] = None,
        use_idiosyncratic: Optional[bool] = None,
        **kwargs
    ) -> 'DDFM':
        """Train the DDFM model using autoencoder.
        
        Parameters
        ----------
        epochs : int, optional
            Number of training epochs. Overrides value from __init__ or config.
        batch_size : int, optional
            Batch size. Overrides value from __init__ or config.
        learning_rate : float, optional
            Learning rate. Overrides value from __init__ or config.
        factor_order : int, optional
            VAR lag order (1 or 2). Overrides value from __init__ or config.
        use_idiosyncratic : bool, optional
            Whether to model idio components. Overrides value from __init__ or config.
        **kwargs
            Additional parameters passed to model.fit():
            - encoder_layers: Override encoder architecture
            - num_factors: Override number of factors
            - activation: Override activation function
            - use_batch_norm: Override batch normalization
            - min_obs_idio: Override minimum observations for idio
        
        Returns
        -------
        DDFM
            Self for method chaining.
        """
        from ..core.helpers import _validate_config_loaded, _validate_data_loaded
        
        _validate_config_loaded(self._config, "config")
        _validate_data_loaded(self._data, "data")
        
        # Create or update DDFM model instance
        if self._ddfm_model is None:
            try:
                from ..models.ddfm import DDFM as DDFMModel
            except ImportError:
                raise ImportError(
                    "DDFM requires PyTorch. Install with: pip install dfm-python[deep]"
                )
            
            # Build kwargs for DDFM model
            ddfm_kwargs = self._ddfm_kwargs.copy()
            
            # Override from train() parameters
            if epochs is not None:
                ddfm_kwargs['epochs'] = epochs
            if batch_size is not None:
                ddfm_kwargs['batch_size'] = batch_size
            if learning_rate is not None:
                ddfm_kwargs['learning_rate'] = learning_rate
            if factor_order is not None:
                ddfm_kwargs['factor_order'] = factor_order
            if use_idiosyncratic is not None:
                ddfm_kwargs['use_idiosyncratic'] = use_idiosyncratic
            
            # Override from config if available
            for param in ['encoder_layers', 'num_factors', 'activation', 'use_batch_norm',
                         'learning_rate', 'epochs', 'batch_size', 'factor_order',
                         'use_idiosyncratic', 'min_obs_idio']:
                if hasattr(self._config, f'ddfm_{param}'):
                    config_val = getattr(self._config, f'ddfm_{param}')
                    if config_val is not None and param not in kwargs:
                        ddfm_kwargs[param] = config_val
            
            # Override from kwargs
            ddfm_kwargs.update(kwargs)
            
            self._ddfm_model = DDFMModel(**ddfm_kwargs)
        
        # Fit DDFM model
        self._result = self._ddfm_model.fit(
            self._data,
            self._config,
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
        self._auto_save_result()
        self._invalidate_nowcast_cache()
        return self


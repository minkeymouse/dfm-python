"""Sktime transformers for nowcasting operations.

This module provides sktime-compatible transformers for nowcasting workflows,
including publication lag masking and news decomposition.
"""

from typing import Optional, Union, Any
from datetime import datetime
import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

from ..config import DFMConfig
from ..utils.time import TimeIndex, parse_timestamp, to_python_datetime
from ..utils.data import create_data_view
from ..utils.helpers import get_series_ids


class PublicationLagMasker(BaseTransformer):
    """Transformer that masks data based on publication lags.
    
    This transformer applies publication lag logic to create data views at
    specific dates, using SeriesConfig.release_date to determine availability.
    It leverages the existing create_data_view() function internally.
    
    Parameters
    ----------
    config : DFMConfig
        Model configuration containing series release date information
    view_date : datetime or str
        Date when data snapshot is taken (data available at this date)
    time_index : TimeIndex or array-like, optional
        Time index for the data. If None, inferred from input data.
    
    Examples
    --------
    >>> from dfm_python.nowcast.transformers import PublicationLagMasker
    >>> from datetime import datetime
    >>> 
    >>> masker = PublicationLagMasker(
    ...     config=model.config,
    ...     view_date=datetime(2024, 1, 15)
    ... )
    >>> 
    >>> # Fit and transform
    >>> X_masked = masker.fit_transform(X)
    >>> 
    >>> # Use in pipeline
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> pipeline = ForecastingPipeline([
    ...     ('mask', PublicationLagMasker(config=model.config, view_date='2024-01-15')),
    ...     ('forecaster', DFMForecaster(config_path='config/dfm.yaml'))
    ... ])
    """
    
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "univariate-only": False,
        "requires-y": False,
        "enforce_index_type": None,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,  # Masking is not invertible
    }
    
    def __init__(
        self,
        config: DFMConfig,
        view_date: Union[datetime, str],
        time_index: Optional[Union[TimeIndex, list, np.ndarray]] = None
    ):
        super().__init__()
        
        self.config = config
        self.view_date = view_date
        self.time_index = time_index
        self._fitted_time_index = None
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit the transformer (no-op, but stores time index).
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data (T × N) with datetime index
        y : pd.DataFrame, optional
            Target series (not used)
            
        Returns
        -------
        self
        """
        # Store time index for transform
        if self.time_index is None:
            # Infer from X index
            if isinstance(X.index, pd.DatetimeIndex):
                self._fitted_time_index = [t.to_pydatetime() for t in X.index]
            else:
                # Try to parse index
                self._fitted_time_index = [parse_timestamp(str(t)) for t in X.index]
        else:
            # Use provided time index
            if isinstance(self.time_index, TimeIndex):
                self._fitted_time_index = [to_python_datetime(t) for t in self.time_index]
            else:
                self._fitted_time_index = list(self.time_index)
        
        return self
    
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform data by applying publication lag masking.
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data (T × N) with datetime index
        y : pd.DataFrame, optional
            Target series (not used)
            
        Returns
        -------
        pd.DataFrame
            Masked data with same index and columns as X
        """
        # Parse view_date if string
        if isinstance(self.view_date, str):
            view_date = parse_timestamp(self.view_date)
        else:
            view_date = self.view_date
        
        # Convert X to numpy
        X_array = X.values
        
        # Get time index
        if self._fitted_time_index is not None:
            time_index = self._fitted_time_index
        elif isinstance(X.index, pd.DatetimeIndex):
            time_index = [t.to_pydatetime() for t in X.index]
        else:
            time_index = [parse_timestamp(str(t)) for t in X.index]
        
        # Convert to TimeIndex if needed
        from ..utils.time import TimeIndex
        time_index_obj = TimeIndex(time_index)
        
        # Use create_data_view to get masked data
        X_view, Time_view, _ = create_data_view(
            X=X_array,
            Time=time_index_obj,
            config=self.config,
            view_date=view_date,
            X_frame=X
        )
        
        # Convert back to DataFrame with same index and columns
        X_masked = pd.DataFrame(
            X_view,
            index=X.index,
            columns=X.columns
        )
        
        return X_masked
    
    def _inverse_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Inverse transform (returns original data).
        
        Since masking is not invertible, this returns the input unchanged.
        This is required for sktime pipeline compatibility.
        
        Parameters
        ----------
        X : pd.DataFrame
            Masked data
        y : pd.DataFrame, optional
            Target series (not used)
            
        Returns
        -------
        pd.DataFrame
            Returns X unchanged (masking is not invertible)
        """
        return X


class NewsDecompositionTransformer(BaseTransformer):
    """Transformer that computes news decomposition between two data views.
    
    This transformer computes news decomposition (forecast update attribution)
    between two view dates, using the existing Nowcast.decompose() method.
    
    Parameters
    ----------
    nowcast_manager : Any
        Nowcast manager instance (from model.nowcast property)
    target_series : str
        Target series ID to nowcast
    target_period : datetime or str
        Target period for nowcast
    view_date_old : datetime or str
        Older data view date (baseline)
    view_date_new : datetime or str
        Newer data view date (contains additional data releases)
    
    Examples
    --------
    >>> from dfm_python.nowcast.transformers import NewsDecompositionTransformer
    >>> 
    >>> transformer = NewsDecompositionTransformer(
    ...     nowcast_manager=model.nowcast,
    ...     target_series='gdp',
    ...     target_period='2024Q1',
    ...     view_date_old='2024-01-15',
    ...     view_date_new='2024-02-15'
    ... )
    >>> 
    >>> # Transform returns news decomposition results
    >>> news_result = transformer.fit_transform(X_old, X_new)
    """
    
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "univariate-only": False,
        "requires-y": True,  # Needs both old and new views
        "enforce_index_type": None,
        "fit_is_empty": True,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": True,
    }
    
    def __init__(
        self,
        nowcast_manager: Any,
        target_series: str,
        target_period: Union[datetime, str],
        view_date_old: Union[datetime, str],
        view_date_new: Union[datetime, str]
    ):
        super().__init__()
        
        self.nowcast_manager = nowcast_manager
        self.target_series = target_series
        self.target_period = target_period
        self.view_date_old = view_date_old
        self.view_date_new = view_date_new
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fit the transformer (no-op).
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series data (not used, kept for API compatibility)
        y : pd.DataFrame, optional
            Target series (not used)
            
        Returns
        -------
        self
        """
        return self
    
    def _transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform by computing news decomposition.
        
        Parameters
        ----------
        X : pd.DataFrame
            New data view (X_new)
        y : pd.DataFrame, optional
            Old data view (X_old) - passed as y for sktime compatibility
            
        Returns
        -------
        pd.DataFrame
            News decomposition results as DataFrame
        """
        # Get news decomposition
        news_result = self.nowcast_manager.decompose(
            self.target_series,
            self.target_period,
            self.view_date_old,
            self.view_date_new,
            return_dict=False
        )
        
        # Convert to DataFrame for sktime compatibility
        from .helpers import NewsDecompResult
        
        if isinstance(news_result, NewsDecompResult):
            # Create DataFrame with news decomposition summary
            result_df = pd.DataFrame({
                'y_old': [news_result.y_old],
                'y_new': [news_result.y_new],
                'change': [news_result.change],
                'total_news': [news_result.change],  # Alias
            })
            
            # Add top contributors as additional columns
            if news_result.top_contributors:
                for i, (series_id, impact) in enumerate(news_result.top_contributors[:5]):
                    result_df[f'top_contributor_{i+1}'] = [series_id]
                    result_df[f'top_impact_{i+1}'] = [impact]
            
            return result_df
        else:
            # Fallback if dict format
            result_df = pd.DataFrame([news_result])
            return result_df


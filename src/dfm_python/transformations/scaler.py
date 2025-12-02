"""DFM Scaler using sktime transformers with Polars support.

This module provides the DFMScaler class that combines series-specific
transformations and global standardization into a single pipeline.
"""

import numpy as np
import polars as pl
from typing import Optional

from ..config import DFMConfig
from .sktime import (
    check_sktime_available,
    ColumnTransformer,
    TransformerPipeline,
    LogTransformer,
    Differencer,
    FunctionTransformer,
    StandardScaler,
)
from .transformers import (
    FREQ_TO_LAG_YOY,
    FREQ_TO_LAG_STEP,
    get_periods_per_year,
    get_annual_factor,
    identity_transform,
    log_transform,
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
)


class DFMScaler:
    """DFM Scaler using sktime transformers with Polars support.
    
    This class handles:
    1. Series-specific transformations (lin, log, chg, ch1, pch, pc1, pca, cch, cca)
    2. Frequency-aware transformations (based on series frequency)
    3. Global standardization (mean=0, std=1 for entire dataset)
    
    Uses sktime's ColumnTransformer with Polars support for efficient processing.
    The scaler accepts Polars DataFrames as input and returns Polars DataFrames,
    avoiding unnecessary pandas conversions.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration containing series transformations and frequencies
        
    Attributes
    ----------
    Mx : np.ndarray
        Mean values used for standardization (N,)
    Wx : np.ndarray
        Standard deviation values used for standardization (N,)
    
    Examples
    --------
    >>> from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
    >>> from dfm_python.transformations import DFMScaler
    >>> import polars as pl
    >>> 
    >>> # Create config
    >>> series = [
    ...     SeriesConfig(series_id='series_0', frequency='m', transformation='lin', blocks=['Block_Global']),
    ...     SeriesConfig(series_id='series_1', frequency='m', transformation='log', blocks=['Block_Global'])
    ... ]
    >>> blocks = {'Block_Global': BlockConfig(factors=1, ar_lag=1, clock='m')}
    >>> config = DFMConfig(series=series, blocks=blocks)
    >>> 
    >>> # Create scaler
    >>> scaler = DFMScaler(config)
    >>> 
    >>> # Transform data (Polars DataFrame)
    >>> X = pl.DataFrame({
    ...     'series_0': [1.0, 2.0, 3.0, 4.0],
    ...     'series_1': [10.0, 20.0, 30.0, 40.0]
    ... })
    >>> X_transformed = scaler.fit_transform(X)
    >>> 
    >>> # Access standardization parameters
    >>> mean = scaler.Mx  # Mean values
    >>> std = scaler.Wx   # Standard deviation values
    
    Notes
    -----
    - Requires sktime>=0.27.0 (install with: pip install dfm-python[transform])
    - All transformations are frequency-aware (monthly, quarterly, etc.)
    - Standardization is applied globally after transformations
    - Polars DataFrames are preserved throughout (no pandas conversion)
    """
    
    def __init__(self, config: DFMConfig):
        """Initialize scaler from DFMConfig.
        
        Parameters
        ----------
        config : DFMConfig
            DFM configuration containing series transformations and frequencies
        """
        check_sktime_available()
        
        self.config = config
        self.transformations = [s.transformation for s in config.series]
        self.frequencies = [s.frequency for s in config.series]
        self.series_ids = config.get_series_ids()
        
        # Each series별로 TransformerPipeline 구성
        transformers = []
        for i, (trans, freq) in enumerate(zip(self.transformations, self.frequencies)):
            trafo_pipeline = self._create_transformation_pipeline(trans, freq)
            transformers.append((f'series_{i}', trafo_pipeline, i))
        
        # ColumnTransformer 구성 (Polars 지원!)
        self.column_transformer = ColumnTransformer(transformers)
        
        # 전체 pipeline: ColumnTransformer → StandardScaler
        self.pipeline = TransformerPipeline([
            ("transform", self.column_transformer),
            ("scaler", StandardScaler())
        ])
        
        # Polars 출력 설정
        self.pipeline.set_output(transform="polars")
        
        # Standardization 파라미터 (fit 후 설정됨)
        self._Mx: Optional[np.ndarray] = None
        self._Wx: Optional[np.ndarray] = None
    
    def _create_transformation_pipeline(self, trans: str, freq: str) -> TransformerPipeline:
        """Create transformation pipeline for a single series.
        
        Parameters
        ----------
        trans : str
            Transformation code: 'lin', 'log', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca'
        freq : str
            Frequency code: 'm', 'q', 'sa', 'a'
            
        Returns
        -------
        TransformerPipeline
            Configured pipeline for this transformation
        """
        if trans == 'lin':
            # Linear: Identity transformation
            return TransformerPipeline([
                FunctionTransformer(func=identity_transform)
            ])
        elif trans == 'log':
            # Log transformation
            return TransformerPipeline([
                FunctionTransformer(func=log_transform)
            ])
        elif trans == 'chg':
            # First difference
            lag = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                Differencer(lags=[lag])
            ])
        elif trans == 'ch1':
            # Year-over-year difference
            lag = FREQ_TO_LAG_YOY.get(freq, 1)
            return TransformerPipeline([
                Differencer(lags=[lag])
            ])
        elif trans == 'pch':
            # Percent change
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                make_pch_transformer(step)
            ])
        elif trans == 'pc1':
            # Year-over-year percent change
            year_step = FREQ_TO_LAG_YOY.get(freq, 12)
            return TransformerPipeline([
                make_pc1_transformer(year_step)
            ])
        elif trans == 'pca':
            # Percent change annualized
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            return TransformerPipeline([
                make_pca_transformer(step, annual_factor)
            ])
        elif trans == 'cch':
            # Continuously compounded rate
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                make_cch_transformer(step)
            ])
        elif trans == 'cca':
            # Continuously compounded annual rate
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            return TransformerPipeline([
                make_cca_transformer(step, annual_factor)
            ])
        else:
            # Default: Identity
            return TransformerPipeline([
                FunctionTransformer(func=identity_transform)
            ])
    
    def fit(self, X: pl.DataFrame, y=None) -> 'DFMScaler':
        """Fit the scaler to data.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input data (T x N), where T is time periods and N is number of series.
            Column names should match series_ids from config.
        y : optional
            Not used, present for sklearn compatibility
        
        Returns
        -------
        self : DFMScaler
            Fitted scaler instance
        """
        # Ensure column order matches series_ids
        if list(X.columns) != self.series_ids:
            X = X.select(self.series_ids)
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # Extract StandardScaler parameters
        self.scaler = self.pipeline.steps[1][1]  # StandardScaler
        self._Mx = self.scaler.mean_
        self._Wx = self.scaler.scale_
        
        return self
    
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Transform data using fitted scaler.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input data (T x N)
        
        Returns
        -------
        X_transformed : pl.DataFrame
            Transformed and standardized data
        """
        # Ensure column order matches series_ids
        if list(X.columns) != self.series_ids:
            X = X.select(self.series_ids)
        
        return self.pipeline.transform(X, y)
    
    def fit_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Fit and transform in one step.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input data (T x N)
        y : optional
            Not used, present for sklearn compatibility
        
        Returns
        -------
        X_transformed : pl.DataFrame
            Transformed and standardized data
        """
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Inverse transform (for forecasting/backtesting).
        
        Note: Some transformations (pch, pc1, etc.) are not fully invertible
        due to division by zero handling. Returns best-effort reconstruction.
        
        Parameters
        ----------
        X : pl.DataFrame
            Transformed data (T x N)
        y : optional
            Not used, present for sklearn compatibility
        
        Returns
        -------
        X_original : pl.DataFrame
            Inverse transformed data (best-effort)
        """
        return self.pipeline.inverse_transform(X, y)
    
    @property
    def Mx(self) -> np.ndarray:
        """Mean values used for standardization (N,)
        
        Returns
        -------
        np.ndarray
            Mean values for each series
            
        Raises
        ------
        ValueError
            If scaler has not been fitted yet
        """
        if self._Mx is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self._Mx
    
    @property
    def Wx(self) -> np.ndarray:
        """Standard deviation values used for standardization (N,)
        
        Returns
        -------
        np.ndarray
            Standard deviation values for each series
            
        Raises
        ------
        ValueError
            If scaler has not been fitted yet
        """
        if self._Wx is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self._Wx


"""Custom transformation functions for DFM.

This module provides transformation functions that match the logic
from the legacy _transform_series() function, implemented as
pickleable functions for use with sktime's FunctionTransformer.
"""

import numpy as np
from typing import Callable

# Frequency → lag mappings
FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1}
FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12}


def get_periods_per_year(freq: str) -> int:
    """Get number of periods per year for a frequency.
    
    Parameters
    ----------
    freq : str
        Frequency code: 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    int
        Number of periods per year (12 for monthly, 4 for quarterly, etc.)
    """
    return FREQ_TO_LAG_YOY.get(freq, 12)


def get_annual_factor(freq: str, step: int) -> float:
    """Get annualization factor for a frequency and step.
    
    Parameters
    ----------
    freq : str
        Frequency code: 'm', 'q', 'sa', 'a'
    step : int
        Number of base periods per observation
        
    Returns
    -------
    float
        Annualization factor (periods_per_year / step)
    """
    periods_per_year = get_periods_per_year(freq)
    return periods_per_year / step


# Transformation functions

def pch_transform(X, step: int = 1) -> np.ndarray:
    """Percent change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
        
    Returns
    -------
    np.ndarray
        Transformed series (percent change)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = 100.0 * (X[step:] - X[:-step]) / np.abs(X[:-step] + 1e-10)
    return result


def pc1_transform(X, year_step: int = 12) -> np.ndarray:
    """Year-over-year percent change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    year_step : int, default 12
        Number of periods for year-over-year (12 for monthly, 4 for quarterly)
        
    Returns
    -------
    np.ndarray
        Transformed series (year-over-year percent change)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > year_step:
        result[year_step:] = 100.0 * (X[year_step:] - X[:-year_step]) / np.abs(X[:-year_step] + 1e-10)
    return result


def pca_transform(X, step: int = 1, annual_factor: float = 12.0) -> np.ndarray:
    """Percent change annualized transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    np.ndarray
        Transformed series (percent change annualized)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = annual_factor * 100.0 * (X[step:] - X[:-step]) / np.abs(X[:-step] + 1e-10)
    return result


def cch_transform(X, step: int = 1) -> np.ndarray:
    """Continuously compounded rate of change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
        
    Returns
    -------
    np.ndarray
        Transformed series (continuously compounded rate)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = 100.0 * (
            np.log(np.abs(X[step:]) + 1e-10) - 
            np.log(np.abs(X[:-step]) + 1e-10)
        )
    return result


def cca_transform(X, step: int = 1, annual_factor: float = 12.0) -> np.ndarray:
    """Continuously compounded annual rate of change transformation.
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
    step : int, default 1
        Number of periods to difference
    annual_factor : float, default 12.0
        Annualization factor (periods_per_year / step)
        
    Returns
    -------
    np.ndarray
        Transformed series (continuously compounded annual rate)
    """
    X = np.asarray(X).flatten()
    T = len(X)
    result = np.full(T, np.nan)
    if T > step:
        result[step:] = annual_factor * 100.0 * (
            np.log(np.abs(X[step:]) + 1e-10) - 
            np.log(np.abs(X[:-step]) + 1e-10)
        )
    return result


def log_transform(X) -> np.ndarray:
    """Log transformation (with absolute value + epsilon).
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
        
    Returns
    -------
    np.ndarray
        Transformed series (log of absolute value)
    """
    X = np.asarray(X).flatten()
    return np.log(np.abs(X) + 1e-10)


def identity_transform(X) -> np.ndarray:
    """Identity transformation (no-op).
    
    Parameters
    ----------
    X : array-like
        Input time series (1D or 2D)
        
    Returns
    -------
    np.ndarray
        Unchanged input
    """
    return np.asarray(X).flatten()


# Factory functions (pickleable closures)

def make_pch_transformer(step: int):
    """Create pch transformer with step parameter.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for percent change
    """
    from .sktime import FunctionTransformer, check_sktime_available
    check_sktime_available()
    
    def transform_func(X):
        return pch_transform(X, step=step)
    return FunctionTransformer(func=transform_func)


def make_pc1_transformer(year_step: int):
    """Create pc1 transformer with year_step parameter.
    
    Parameters
    ----------
    year_step : int
        Number of periods for year-over-year
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for year-over-year percent change
    """
    from .sktime import FunctionTransformer, check_sktime_available
    check_sktime_available()
    
    def transform_func(X):
        return pc1_transform(X, year_step=year_step)
    return FunctionTransformer(func=transform_func)


def make_pca_transformer(step: int, annual_factor: float):
    """Create pca transformer with step and annual_factor parameters.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
    annual_factor : float
        Annualization factor
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for percent change annualized
    """
    from .sktime import FunctionTransformer, check_sktime_available
    check_sktime_available()
    
    def transform_func(X):
        return pca_transform(X, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=transform_func)


def make_cch_transformer(step: int):
    """Create cch transformer with step parameter.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for continuously compounded rate
    """
    from .sktime import FunctionTransformer, check_sktime_available
    check_sktime_available()
    
    def transform_func(X):
        return cch_transform(X, step=step)
    return FunctionTransformer(func=transform_func)


def make_cca_transformer(step: int, annual_factor: float):
    """Create cca transformer with step and annual_factor parameters.
    
    Parameters
    ----------
    step : int
        Number of periods to difference
    annual_factor : float
        Annualization factor
        
    Returns
    -------
    FunctionTransformer
        Configured FunctionTransformer for continuously compounded annual rate
    """
    from .sktime import FunctionTransformer, check_sktime_available
    check_sktime_available()
    
    def transform_func(X):
        return cca_transform(X, step=step, annual_factor=annual_factor)
    return FunctionTransformer(func=transform_func)


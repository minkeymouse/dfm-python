"""Nowcasting utility functions.

This module contains helper functions for nowcasting operations:
- Frequency and date calculations
- Forecast horizon configuration
- Data transformations
- Configuration validation
- News decomposition utilities
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime, timedelta
import logging

from ..config import DFMConfig

# TimeIndex is in utils.time, not core.time
from ..utils.time import TimeIndex
from ..utils.helpers import (
    get_series_ids,
    get_frequencies,
    safe_get_attr,
)
from ..config.utils import FREQUENCY_HIERARCHY, get_periods_per_year
from ..utils.time import clock_to_datetime_freq
from ..logger import get_logger

_logger = get_logger(__name__)


def get_higher_frequency(clock: str) -> Optional[str]:
    """Get frequency one step faster than clock.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    str or None
        Frequency one step faster than clock, or None if no higher frequency available
        
    Examples
    --------
    >>> get_higher_frequency('m')
    'w'
    >>> get_higher_frequency('q')
    'm'
    >>> get_higher_frequency('d')
    None
    """
    clock_h = FREQUENCY_HIERARCHY.get(clock, 3)
    target_h = clock_h - 1
    
    if target_h < 1:
        return None  # No higher frequency available (clock is already fastest)
    
    # Find frequency with target hierarchy
    for freq, h in FREQUENCY_HIERARCHY.items():
        if h == target_h:
            return freq
    
    return None  # No higher frequency found


def calc_backward_date(
    target_date: datetime,
    step: int,
    freq: str
) -> datetime:
    """Calculate backward date with accurate calendar handling.
    
    Parameters
    ----------
    target_date : datetime
        Target date to go backward from
    step : int
        Number of steps to go backward
    freq : str
        Frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
        
    Returns
    -------
    datetime
        Calculated backward date
        
    Examples
    --------
    >>> from datetime import datetime
    >>> calc_backward_date(datetime(2024, 3, 15), 1, 'm')
    datetime(2024, 2, 15)
    >>> calc_backward_date(datetime(2024, 3, 15), 2, 'q')
    datetime(2023, 9, 15)
    """
    try:
        from dateutil.relativedelta import relativedelta
        use_relativedelta = True
    except ImportError:
        use_relativedelta = False
        relativedelta = None  # type: ignore
        _logger.debug("dateutil.relativedelta not available, using timedelta approximation")
    
    if freq == 'd':
        return target_date - timedelta(days=step)
    elif freq == 'w':
        return target_date - timedelta(weeks=step)
    elif freq == 'm':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step)
        else:
            # Approximate: 30 days per month
            return target_date - timedelta(days=step * 30)
    elif freq == 'q':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 3)
        else:
            # Approximate: 90 days per quarter
            return target_date - timedelta(days=step * 90)
    elif freq == 'sa':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(months=step * 6)
        else:
            # Approximate: 180 days per semi-annual
            return target_date - timedelta(days=step * 180)
    elif freq == 'a':
        if use_relativedelta and relativedelta is not None:
            return target_date - relativedelta(years=step)
        else:
            # Approximate: 365 days per year
            return target_date - timedelta(days=step * 365)
    else:
        # Fallback for unknown frequencies
        _logger.warning(f"Unknown frequency '{freq}', using 30-day approximation")
        return target_date - timedelta(days=step * 30)


def get_forecast_horizon(clock: str, horizon: Optional[int] = None) -> Tuple[int, str]:
    """Get forecast horizon configuration based on clock frequency.
    
    Parameters
    ----------
    clock : str
        Clock frequency code: 'd', 'w', 'm', 'q', 'sa', 'a'
    horizon : int, optional
        Number of periods for forecast horizon. If None, defaults to 1 timestep.
        
    Returns
    -------
    Tuple[int, str]
        (horizon_periods, datetime_freq) where:
        - horizon_periods: Number of periods to forecast
        - datetime_freq: Frequency string for datetime_range() ('D', 'W', 'ME', 'QE', 'YE')
        
    Examples
    --------
    >>> get_forecast_horizon('m', 3)
    (3, 'ME')
    >>> get_forecast_horizon('q')
    (1, 'QE')
    """
    if horizon is None:
        horizon = 1  # Default: 1 timestep based on clock frequency
    
    # Map clock frequency to datetime frequency string
    datetime_freq = clock_to_datetime_freq(clock)
    
    # For semi-annual, we need 6 months per period
    if clock == 'sa' and horizon > 0:
        horizon = horizon * 6  # Convert to months
    
    return horizon, datetime_freq


def check_config(saved_config: Any, current_config: DFMConfig) -> None:
    """Check if saved config is consistent with current config.
    
    Parameters
    ----------
    saved_config : Any
        Saved configuration object (may be DFMConfig or dict-like)
    current_config : DFMConfig
        Current configuration object
        
    Notes
    -----
    - Issues a warning if configs differ significantly
    - Does not raise exceptions (allows computation to continue)
    
    Examples
    --------
    >>> check_config(saved_config, current_config)
    # May issue warnings if configs differ
    """
    try:
        # Basic checks
        if hasattr(saved_config, 'series') and hasattr(current_config, 'series'):
            if len(saved_config.series) != len(current_config.series):
                _logger.warning(
                    f"Config mismatch: saved config has {len(saved_config.series)} series, "
                    f"current config has {len(current_config.series)} series"
                )
        
        if hasattr(saved_config, 'block_names') and hasattr(current_config, 'block_names'):
            if saved_config.block_names != current_config.block_names:
                _logger.warning(
                    f"Config mismatch: block names differ. "
                    f"Saved: {saved_config.block_names}, Current: {current_config.block_names}"
                )
    except Exception as e:
        _logger.debug(f"Config consistency check failed (non-critical): {str(e)}")
        # If comparison fails, continue anyway


def extract_news(
    singlenews: np.ndarray,
    weight: np.ndarray,
    series_ids: List[str],
    top_n: int = 5
) -> Dict[str, Any]:
    """Extract summary statistics from news decomposition.
    
    Parameters
    ----------
    singlenews : np.ndarray
        News contributions (N,) or (N, n_targets)
    weight : np.ndarray
        Weights (N,) or (N, n_targets)
    series_ids : List[str]
        Series IDs
    top_n : int, default 5
        Number of top contributors to include
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'total_impact', 'top_contributors', etc.
        
    Examples
    --------
    >>> summary = extract_news(singlenews, weight, series_ids, top_n=10)
    >>> print(summary['top_contributors'])
    """
    # Handle both 1D and 2D arrays
    if singlenews.ndim == 1:
        news_contributions = singlenews
        weights = weight
    else:
        # If 2D, use first target (column 0)
        news_contributions = singlenews[:, 0]
        weights = weight[:, 0] if weight.ndim > 1 else weight
    
    # Calculate total impact
    total_impact = np.nansum(news_contributions)
    
    # Get top contributors
    abs_contributions = np.abs(news_contributions)
    top_indices = np.argsort(abs_contributions)[::-1][:top_n]
    
    # Build list of top contributors
    top_contributors = []
    for idx in top_indices:
        if idx < len(series_ids):
            top_contributors.append({
                'series_id': series_ids[idx],
                'contribution': float(news_contributions[idx]),
                'weight': float(weights[idx]) if idx < len(weights) else 0.0
            })
    
    return {
        'total_impact': float(total_impact),
        'top_contributors': top_contributors,
        # Note: revision_impact and release_impact are placeholders for future implementation.
        # revision_impact: Measures impact of data revisions (currently returns total_impact).
        # release_impact: Measures impact of new data releases (currently returns 0.0).
        # These features are not critical for core nowcasting functionality and can be
        # implemented incrementally when needed. See STATUS.md for tracking.
        'revision_impact': float(total_impact),  # Placeholder: returns total_impact
        'release_impact': 0.0  # Placeholder: returns 0.0
    }
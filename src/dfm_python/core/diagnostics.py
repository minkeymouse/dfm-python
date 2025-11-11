"""Diagnostic functions for DFM estimation results."""

from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from ..config import DFMConfig

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..dfm import DFMResult
else:
    # Avoid circular import at runtime
    DFMResult = Any


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Calculate Root Mean Squared Error (RMSE) between actual and predicted values.
    
    RMSE is calculated as: sqrt(mean((actual - predicted)^2))
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values (T × N) or (T,) array
    predicted : np.ndarray
        Predicted/forecasted values (T × N) or (T,) array, same shape as actual
    mask : np.ndarray, optional
        Boolean mask (T × N) or (T,) indicating which values to include in calculation.
        If None, only non-NaN values are used.
        
    Returns
    -------
    rmse_overall : float
        Overall RMSE averaged across all series and time periods
    rmse_per_series : np.ndarray
        RMSE for each series (N,) or scalar if 1D input
        
    Notes
    -----
    - Returns NaN for overall RMSE if no valid observations exist
    - Returns NaN for individual series if that series has no valid observations
    - Mask parameter allows selective calculation (e.g., exclude certain time periods)
    - Used in diagnostics and model evaluation
    - Automatically handles missing data by excluding NaN values
    """
    # Ensure arrays are the same shape
    if actual.shape != predicted.shape:
        raise ValueError(f"actual and predicted must have same shape, got {actual.shape} and {predicted.shape}")
    
    # Create mask for valid values
    if mask is None:
        # Use non-NaN values in both actual and predicted
        mask = np.isfinite(actual) & np.isfinite(predicted)
    else:
        # Combine user mask with finite check
        mask = mask & np.isfinite(actual) & np.isfinite(predicted)
    
    # Calculate squared errors
    errors_sq = (actual - predicted) ** 2
    
    # Handle 1D case (single series)
    if actual.ndim == 1:
        if np.sum(mask) == 0:
            return np.nan, np.array([np.nan])
        rmse_series = np.sqrt(np.mean(errors_sq[mask]))
        return rmse_series, np.array([rmse_series])
    
    # Handle 2D case (multiple series)
    T, N = actual.shape
    
    # Calculate RMSE per series
    rmse_per_series = np.zeros(N)
    for i in range(N):
        series_mask = mask[:, i]
        if np.sum(series_mask) > 0:
            rmse_per_series[i] = np.sqrt(np.mean(errors_sq[series_mask, i]))
        else:
            rmse_per_series[i] = np.nan
    
    # Calculate overall RMSE (average across all valid observations)
    if np.any(mask):
        rmse_overall = np.sqrt(np.mean(errors_sq[mask]))
    else:
        rmse_overall = np.nan
    
    return rmse_overall, rmse_per_series


def _display_dfm_tables(Res: DFMResult, config: DFMConfig, nQ: int) -> None:
    """Display DFM estimation output tables.
    
    Displays formatted tables for factor loadings, AR coefficients, and
    idiosyncratic components. Uses pandas DataFrame formatting if available,
    otherwise falls back to shape information.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results containing C, A, Q, p, r
    config : DFMConfig
        Configuration object with series and block information
    nQ : int
        Number of slower-frequency series (for mixed-frequency models)
        
    Notes
    -----
    - Only displays if logging level is INFO or higher
    - Tables include: same-frequency loadings, slower-frequency loadings,
      factor AR coefficients, and idiosyncratic AR coefficients
    - Automatically handles missing pandas dependency
    """
    if not _logger.isEnabledFor(logging.INFO):
        return
    
    series_ids = config.get_series_ids()
    series_names = config.get_series_names()
    block_names = config.block_names
    n_same_freq = len(series_ids) - nQ
    nLags = max(Res.p, 5)
    nFactors = int(np.sum(Res.r))
    
    try:
        _logger.info('\n\n\n')
        # Factor Loadings for Same-Frequency Series
        _logger.info('Factor Loadings for Same-Frequency Series')
        C_same_freq = Res.C[:n_same_freq, ::5][:, :nFactors]
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame(C_same_freq, 
                                index=[name.replace(' ', '_') for name in series_names[:n_same_freq]],
                                columns=block_names[:nFactors] if len(block_names) >= nFactors else [f'Block{i+1}' for i in range(nFactors)])
                _logger.info(f'\n{df.to_string()}')
            except Exception as e:
                _logger.debug(f'Failed to format same-frequency loadings table: {e}')
                _logger.info(f'Same-frequency loadings shape: {C_same_freq.shape}')
        else:
            _logger.info(f'Same-frequency loadings shape: {C_same_freq.shape}')
        
        _logger.info('\n\n\n')
        # Slower-Frequency Loadings Sample (First Factor)
        _logger.info('Slower-Frequency Loadings Sample (First Factor)')
        C_slower_freq = Res.C[-nQ:, :5]
        if PANDAS_AVAILABLE:
            try:
                n_lags = min(5, C_slower_freq.shape[1])
                lag_cols = [f'factor1_lag{i}' for i in range(n_lags)]
                df = pd.DataFrame(C_slower_freq,
                                index=[name.replace(' ', '_') for name in series_names[-nQ:]],
                                columns=lag_cols)
                _logger.info(f'\n{df.to_string()}')
            except Exception as e:
                _logger.debug(f'Failed to format slower-frequency loadings table: {e}')
                _logger.info(f'Slower-frequency loadings shape: {C_slower_freq.shape}')
        else:
            _logger.info(f'Slower-frequency loadings shape: {C_slower_freq.shape}')
        
        _logger.info('\n\n\n')
        # Autoregressive Coefficients on Factors
        _logger.info('Autoregressive Coefficients on Factors')
        A_terms = np.diag(Res.A)
        Q_terms = np.diag(Res.Q)
        A_terms_factors = A_terms[::5][:nFactors]
        Q_terms_factors = Q_terms[::5][:nFactors]
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame({
                    'AR_Coefficient': A_terms_factors,
                    'Variance_Residual': Q_terms_factors
                }, index=[name.replace(' ', '_') for name in block_names[:nFactors]])
                _logger.info(f'\n{df.to_string()}')
            except Exception as e:
                _logger.debug(f'Failed to format AR coefficients table: {e}')
                _logger.info(f'Factor AR coefficients: {A_terms_factors}')
        else:
            _logger.info(f'Factor AR coefficients: {A_terms_factors}')
        
        _logger.info('\n\n\n')
        # Autoregressive Coefficients on Idiosyncratic Component
        _logger.info('Autoregressive Coefficients on Idiosyncratic Component')
        rp1 = nFactors * 5
        same_freq_idx = np.arange(rp1, rp1 + n_same_freq)
        slower_freq_idx = np.arange(rp1 + n_same_freq, len(A_terms), 5)
        combined_idx = np.concatenate([same_freq_idx, slower_freq_idx])
        combined_idx = combined_idx[combined_idx < len(A_terms)]
        A_idio = A_terms[combined_idx]
        Q_idio = Q_terms[combined_idx]
        if PANDAS_AVAILABLE:
            try:
                series_names_list = []
                for idx in combined_idx:
                    if idx < rp1 + n_same_freq:
                        series_idx = idx - rp1
                        if series_idx < n_same_freq:
                            series_names_list.append(series_names[series_idx].replace(' ', '_'))
                    else:
                        slower_idx = (idx - (rp1 + n_same_freq)) // 5
                        if slower_idx < nQ:
                            series_names_list.append(series_names[n_same_freq + slower_idx].replace(' ', '_'))
                df = pd.DataFrame({
                    'AR_Coefficient': A_idio[:len(series_names_list)],
                    'Variance_Residual': Q_idio[:len(series_names_list)]
                }, index=series_names_list)
                _logger.info(f'\n{df.to_string()}')
            except Exception as e:
                _logger.debug(f'Failed to format idiosyncratic AR coefficients table: {e}')
                _logger.info(f'Idiosyncratic AR coefficients (first 10): {A_idio[:min(10, len(A_idio))]}')
                if len(A_idio) > 10:
                    _logger.info(f'... (total {len(A_idio)} coefficients)')
        else:
            _logger.info(f'Idiosyncratic AR coefficients (first 10): {A_idio[:min(10, len(A_idio))]}')
            if len(A_idio) > 10:
                _logger.info(f'... (total {len(A_idio)} coefficients)')
        
        _logger.info('\n\n\n')
        # Model Fit Statistics (RMSE)
        if Res.rmse is not None and not np.isnan(Res.rmse):
            _logger.info('Model Fit Statistics')
            _logger.info(f'  Overall RMSE (original scale): {Res.rmse:.6f}')
            if Res.rmse_std is not None and not np.isnan(Res.rmse_std):
                _logger.info(f'  Overall RMSE (standardized scale): {Res.rmse_std:.6f}')
            if Res.rmse_per_series is not None and len(Res.rmse_per_series) > 0:
                _logger.info('\n  RMSE per Series (Original Scale):')
                try:
                    for i, (name, rmse_val) in enumerate(zip(series_names, Res.rmse_per_series)):
                        if not np.isnan(rmse_val):
                            mean_val = Res.Mx[i] if i < len(Res.Mx) else np.nan
                            if not np.isnan(mean_val) and abs(mean_val) > 1e-6:
                                pct = 100.0 * rmse_val / abs(mean_val)
                                _logger.info(f'    {name:40s}: {rmse_val:.6f} ({pct:.2f}% of mean)')
                            else:
                                _logger.info(f'    {name:40s}: {rmse_val:.6f}')
                except Exception as e:
                    _logger.debug(f'Failed to format RMSE per series: {e}')
                    for i, rmse_val in enumerate(Res.rmse_per_series):
                        if not np.isnan(rmse_val):
                            _logger.info(f'    Series {i:3d}: {rmse_val:.6f}')
            if Res.rmse_std_per_series is not None and len(Res.rmse_std_per_series) > 0:
                _logger.info('\n  RMSE per Series (Standardized Scale):')
                try:
                    for i, (name, rmse_std_val) in enumerate(zip(series_names, Res.rmse_std_per_series)):
                        if not np.isnan(rmse_std_val):
                            _logger.info(f'    {name:40s}: {rmse_std_val:.6f} std dev')
                except Exception as e:
                    _logger.debug(f'Failed to format RMSE std per series: {e}')
                    for i, rmse_std_val in enumerate(Res.rmse_std_per_series):
                        if not np.isnan(rmse_std_val):
                            _logger.info(f'    Series {i:3d}: {rmse_std_val:.6f} std dev')
            if Res.rmse_per_series is not None and Res.Mx is not None:
                _logger.info('\n  Diagnostic Warnings:')
                try:
                    warnings_count = 0
                    for i, (name, rmse_val) in enumerate(zip(series_names, Res.rmse_per_series)):
                        if not np.isnan(rmse_val) and i < len(Res.Mx):
                            mean_val = Res.Mx[i]
                            std_val = Res.Wx[i] if i < len(Res.Wx) else np.nan
                            if not np.isnan(mean_val) and abs(mean_val) > 1e-6:
                                pct_of_mean = 100.0 * rmse_val / abs(mean_val)
                                if pct_of_mean > 50.0 or (not np.isnan(std_val) and rmse_val > 10.0 * std_val):
                                    warnings_count += 1
                                    if warnings_count <= 5:
                                        _logger.warning(f'    ⚠ {name:40s}: RMSE is {pct_of_mean:.1f}% of mean')
                                        if not np.isnan(std_val):
                                            _logger.warning(f'      (RMSE={rmse_val:.2e}, Mean={mean_val:.2e}, Std={std_val:.2e})')
                    if warnings_count > 5:
                        _logger.warning(f'    ... and {warnings_count - 5} more series with high RMSE')
                except Exception as e:
                    _logger.debug(f'Failed to format diagnostic warnings: {e}')
            _logger.info('\n\n\n')
    except Exception as e:
        _logger.debug(f'Failed to display DFM tables: {e}')


def diagnose_series(Res: DFMResult, config: DFMConfig, series_name: Optional[str] = None, 
                    series_idx: Optional[int] = None) -> Dict[str, Any]:
    """Diagnose model fit issues for a specific series.
    
    Computes diagnostic statistics including RMSE, loading magnitudes,
    and standardization information for a single series.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results containing C, x_sm, x, and other outputs
    config : DFMConfig
        Configuration object with series information
    series_name : str, optional
        Name of series to diagnose (case-insensitive matching)
    series_idx : int, optional
        Index of series to diagnose (0-based)
        
    Returns
    -------
    dict
        Dictionary containing diagnostic information:
        - 'series_name': Name of the series
        - 'series_idx': Index of the series
        - 'rmse_original': RMSE on original scale
        - 'rmse_standardized': RMSE on standardized scale
        - 'rmse_pct_of_mean': RMSE as percentage of mean
        - 'rmse_in_std_devs': RMSE in standard deviations
        - 'mean': Mean of original series
        - 'std': Standard deviation of original series
        - 'max_loading_abs': Maximum absolute loading value
        - 'loading_norm': L2 norm of loading vector
        
    Raises
    ------
    ValueError
        If neither series_name nor series_idx is provided, or if
        series_name is not found or series_idx is out of range
        
    Notes
    -----
    - Either series_name or series_idx must be provided
    - Series name matching is case-insensitive
    - RMSE values may be None if insufficient data is available
    """
    if series_name is not None:
        try:
            series_names = config.get_series_names() if config.series else []
            if series_name in series_names:
                series_idx = series_names.index(series_name)
            else:
                series_idx = next((i for i, name in enumerate(series_names) 
                                 if name.lower() == series_name.lower()), None)
                if series_idx is None:
                    raise ValueError(f"Series '{series_name}' not found in configuration")
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Cannot find series '{series_name}': {e}")
    if series_idx is None:
        raise ValueError("Must provide either series_name or series_idx")
    if series_idx < 0 or series_idx >= Res.C.shape[0]:
        raise ValueError(f"Series index {series_idx} out of range [0, {Res.C.shape[0]})")
    try:
        series_names = config.get_series_names() if config.series else []
        name = series_names[series_idx] if series_idx < len(series_names) else f"Series_{series_idx}"
    except (AttributeError, IndexError, KeyError):
        name = f"Series_{series_idx}"
    rmse_original = None
    rmse_standardized = None
    if Res.rmse_per_series is not None and series_idx < len(Res.rmse_per_series):
        rmse_original = Res.rmse_per_series[series_idx]
    if Res.rmse_std_per_series is not None and series_idx < len(Res.rmse_std_per_series):
        rmse_standardized = Res.rmse_std_per_series[series_idx]
    mean_val = Res.Mx[series_idx] if series_idx < len(Res.Mx) else np.nan
    std_val = Res.Wx[series_idx] if series_idx < len(Res.Wx) else np.nan
    rmse_pct_of_mean = 100.0 * rmse_original / abs(mean_val) if (rmse_original is not None and not np.isnan(mean_val) and abs(mean_val) > 1e-6) else None
    rmse_in_std_devs = rmse_original / std_val if (rmse_original is not None and not np.isnan(std_val) and std_val > 1e-6) else None
    factor_loadings = Res.C[series_idx, :] if series_idx < Res.C.shape[0] else np.array([])
    max_loading = np.max(np.abs(factor_loadings)) if len(factor_loadings) > 0 else np.nan
    loading_sum_sq = np.sum(factor_loadings ** 2) if len(factor_loadings) > 0 else np.nan
    return {
        'series_name': name,
        'series_idx': series_idx,
        'rmse_original': rmse_original,
        'rmse_standardized': rmse_standardized,
        'mean': mean_val,
        'std': std_val,
        'rmse_pct_of_mean': rmse_pct_of_mean,
        'rmse_in_std_devs': rmse_in_std_devs,
        'factor_loadings': factor_loadings,
        'max_loading': max_loading,
        'loading_sum_sq': loading_sum_sq,
        'reconstruction_error_mean': None,
        'reconstruction_error_std': None
    }


def print_series_diagnosis(Res: DFMResult, config: DFMConfig, 
                          series_name: Optional[str] = None, 
                          series_idx: Optional[int] = None) -> None:
    """Print a formatted diagnosis report for a specific series.
    
    Prints a user-friendly diagnostic report including RMSE statistics,
    standardization values, and loading information.
    
    Parameters
    ----------
    Res : DFMResult
        DFM estimation results
    config : DFMConfig
        Configuration object with series information
    series_name : str, optional
        Name of series to diagnose (case-insensitive matching)
    series_idx : int, optional
        Index of series to diagnose (0-based)
        
    Notes
    -----
    - This function uses print() for user-facing output, as it's intended
      to be called directly by users who want to see diagnostic information
    - Calls diagnose_series() internally to compute statistics
    - Prints formatted report with section headers and clear labels
    """
    diag = diagnose_series(Res, config, series_name=series_name, series_idx=series_idx)
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT: {diag['series_name']}")
    print(f"{'='*70}\n")
    print("RMSE Statistics:")
    print(f"  Original scale:     {diag['rmse_original']:.6e}" if diag['rmse_original'] is not None else "  Original scale:     N/A")
    print(f"  Standardized scale: {diag['rmse_standardized']:.6f} std dev" if diag['rmse_standardized'] is not None else "  Standardized scale: N/A")
    if diag['rmse_pct_of_mean'] is not None:
        print(f"  As % of mean:       {diag['rmse_pct_of_mean']:.2f}%")
    if diag['rmse_in_std_devs'] is not None:
        print(f"  In std deviations: {diag['rmse_in_std_devs']:.2f}x")
    print("\nStandardization Values:")
    print(f"  Mean:  {diag['mean']:.6e}" if not np.isnan(diag['mean']) else "  Mean:  N/A")
    print(f"  Std:   {diag['std']:.6e}" if not np.isnan(diag['std']) else "  Std:   N/A")
    print("\nFactor Loadings:")
    if len(diag['factor_loadings']) > 0:
        print(f"  Number of loadings: {len(diag['factor_loadings'])}")
        print(f"  Max absolute:       {diag['max_loading']:.6f}" if not np.isnan(diag['max_loading']) else "  Max absolute:       N/A")
        print(f"  Sum of squares:     {diag['loading_sum_sq']:.6f}" if not np.isnan(diag['loading_sum_sq']) else "  Sum of squares:     N/A")
        abs_loadings = np.abs(diag['factor_loadings'])
        top_indices = np.argsort(abs_loadings)[-5:][::-1]
        print(f"  Top 5 loadings:")
        for idx in top_indices:
            print(f"    Factor {idx:3d}: {diag['factor_loadings'][idx]:8.4f}")
    else:
        print("  No loadings available")
    print(f"\n{'='*70}\n")

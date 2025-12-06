"""Scaling transformer utilities for DFM/DDFM.

This module provides utilities for creating unified scaling transformers
(StandardScaler for all series) from configuration files or direct usage.

Features:
- Unified scaling (StandardScaler for all series) - recommended for factor models
- Direct sklearn scaler usage (no TabularToSeriesAdaptor needed per sktime docs)
- Integration with sktime TransformerPipeline
- Class pattern for custom scaling strategies (optional)

**Recommended**: Use unified scaling (StandardScaler for all series) as it ensures
all series contribute proportionally to factor extraction without scale-driven dominance.
"""

from typing import Optional, Union, List, Dict, Any, Protocol, Callable
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..config import DFMConfig, SeriesConfig
from ..logger import get_logger

_logger = get_logger(__name__)


def _check_sktime():
    """Check if sktime is available and raise ImportError if not."""
    try:
        import sktime
        return True
    except ImportError:
        raise ImportError(
            "Scaling utilities require sktime. Install with: pip install sktime"
        )


def _check_sklearn():
    """Check if sklearn is available and raise ImportError if not."""
    try:
        import sklearn
        return True
    except ImportError:
        raise ImportError(
            "Scaling utilities require scikit-learn. Install with: pip install scikit-learn"
        )


# ============================================================================
# Scaling Strategy Protocol (Class Pattern)
# ============================================================================

class ScalingStrategy(Protocol):
    """Protocol for custom scaling strategies (optional - unified scaling recommended).
    
    **Note**: Unified scaling (StandardScaler for all series) is recommended for
    factor models. This protocol is provided for advanced use cases only.
    
    Users can implement this protocol to define custom scaling logic, though
    the default unified scaling approach is typically preferred.
    
    Example (advanced use case):
        class MyScalingStrategy:
            def get_scaler(self, series: SeriesConfig, series_index: int, 
                          column_name: str) -> Optional[str]:
                # Custom logic (unified scaling is typically preferred)
                if 'volatility' in column_name.lower():
                    return 'robust'
                elif 'return' in column_name.lower():
                    return None  # No scaling
                else:
                    return 'standard'  # Default: unified scaling
    """
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Determine scaler type for a series.
        
        Parameters
        ----------
        series : SeriesConfig
            Series configuration
        series_index : int
            Index of series in config (0-based)
        column_name : str
            Column name in data (typically series_id)
            
        Returns
        -------
        Optional[str]
            Scaler type: 'standard', 'robust', 'minmax', 'maxabs', 'quantile', 
            or None (no scaling). Returns None if series should not be scaled.
        """
        ...


# ============================================================================
# Default Scaling Strategies
# ============================================================================

class DefaultScalingStrategy:
    """Default scaling strategy: unified scaling (StandardScaler for all series).
    
    This is the recommended approach for factor models as it ensures all series
    contribute proportionally to factor extraction without scale-driven dominance.
    """
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Return 'standard' for unified scaling (all series use StandardScaler)."""
        return 'standard'  # Unified scaling: StandardScaler for all series


class NoScalingStrategy:
    """Strategy that applies no scaling to any series."""
    
    def get_scaler(self, series: SeriesConfig, series_index: int, 
                   column_name: str) -> Optional[str]:
        """Return None (no scaling) for all series."""
        return None


# ============================================================================
# Scaler Factory
# ============================================================================

def _create_scaler_transformer(scaler_type: Optional[str]) -> Any:
    """Create a raw sklearn scaler for unified scaling.
    
    Returns a raw sklearn scaler that can be used directly in TransformerPipeline.
    Per sktime docs, sklearn transformers are automatically applied per series instance
    when used in TransformerPipeline.
    
    Parameters
    ----------
    scaler_type : Optional[str]
        Scaler type: 'standard' (recommended), 'robust', 'minmax', 'maxabs', 'quantile', 
        or None (passthrough)
        
    Returns
    -------
    Any
        Raw sklearn scaler (StandardScaler, RobustScaler, etc.) that can be used
        directly in TransformerPipeline, or FunctionTransformer for passthrough.
        
    Notes
    -----
    Per sktime documentation, sklearn transformers can be used directly in
    TransformerPipeline without TabularToSeriesAdaptor:
    - "If applied to Series, sklearn transformers are applied by series instance"
    - Example: `StandardScaler() * SummaryTransformer()` works directly
    - Unified scaling (same scaler for all series) is recommended for factor models
    """
    _check_sklearn()
    
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, 
        MaxAbsScaler, QuantileTransformer
    )
    
    if scaler_type is None or scaler_type == 'none':
        # Passthrough: no scaling
        return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    
    scaler_type_lower = scaler_type.lower()
    
    if scaler_type_lower == 'standard':
        return StandardScaler()
    elif scaler_type_lower == 'robust':
        return RobustScaler()
    elif scaler_type_lower == 'minmax':
        return MinMaxScaler()
    elif scaler_type_lower == 'maxabs':
        return MaxAbsScaler()
    elif scaler_type_lower == 'quantile':
        return QuantileTransformer(output_distribution='normal')
    else:
        _logger.warning(
            f"Unknown scaler type '{scaler_type}', using StandardScaler as fallback"
        )
        return StandardScaler()


# ============================================================================
# Main API: Create Scaling Transformer from Config
# ============================================================================

def create_scaling_transformer_from_config(
    config: DFMConfig,
    strategy: Optional[ScalingStrategy] = None,
    column_names: Optional[List[str]] = None
) -> Any:
    """Create a unified scaling transformer from model-level scaler configuration.
    
    This function returns a raw sklearn scaler (StandardScaler, RobustScaler, etc.)
    that can be used directly in TransformerPipeline. Per sktime docs, sklearn 
    transformers are automatically applied per series instance when used in 
    TransformerPipeline.
    
    **Unified scaling**: The scaler type is now specified at the model level
    (in model config YAML) rather than per-series. This ensures all series use
    the same scaling method, which is the recommended approach for factor models
    as it ensures all series contribute proportionally to factor extraction.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration containing model-level scaler setting (config.scaler).
        The scaler type is read from config.scaler (default: 'standard').
    strategy : Optional[ScalingStrategy], default None
        Custom scaling strategy. If None, uses the scaler type from config.scaler.
    column_names : Optional[List[str]], default None
        Column names in data (used for validation). If None, uses
        config.get_series_ids().
        
    Returns
    -------
    Any
        Raw sklearn scaler (StandardScaler, RobustScaler, etc.) that can be used
        directly in TransformerPipeline. Per sktime docs, sklearn transformers
        are applied per series instance automatically when used in TransformerPipeline.
        
    Examples
    --------
    **Unified scaling (recommended for factor models)**:
    
    ```python
    from dfm_python import DFMConfig
    from dfm_python.lightning.scaling import create_scaling_transformer_from_config
    from sktime.transformations.compose import TransformerPipeline
    from sktime.transformations.series.impute import Imputer
    
    config = DFMConfig.from_hydra(hydra_cfg)
    
    # Returns raw sklearn StandardScaler (unified scaling for all series)
    scaler = create_scaling_transformer_from_config(config)
    
    # Use in pipeline - per sktime docs, sklearn transformers work directly
    # Applied per series instance automatically
    pipeline = TransformerPipeline([
        ("impute_ffill", Imputer(method="ffill")),
        ("impute_bfill", Imputer(method="bfill")),
        ("scaler", scaler),  # Unified StandardScaler for all series
    ])
    ```
    
    **Direct usage (simplest)**:
    
    ```python
    from sktime.transformations.compose import TransformerPipeline
    from sktime.transformations.series.impute import Imputer
    from sklearn.preprocessing import StandardScaler
    
    # Per sktime docs: sklearn transformers work directly in TransformerPipeline
    # Applied per series instance automatically (unified scaling)
    pipeline = TransformerPipeline([
        ("impute_ffill", Imputer(method="ffill")),
        ("impute_bfill", Imputer(method="bfill")),
        ("scaler", StandardScaler()),  # Unified scaling - no wrapper needed!
    ])
    ```
    
    **Note**: Unified scaling (StandardScaler for all series) is recommended for
    factor models as it ensures all series contribute proportionally to factor
    extraction without scale-driven dominance.
    """
    _check_sklearn()
    
    # Get scaler type from model-level config (config.scaler)
    # Unified scaling: all series use the same scaler specified at model level
    if hasattr(config, 'scaler') and config.scaler is not None:
        scaler_type = config.scaler
    elif strategy is not None:
        # Fallback to strategy if config.scaler not available
        if len(config.series) > 0:
            first_series = config.series[0]
            first_column_name = config.get_series_ids()[0] if config.series else "series_0"
            scaler_type = strategy.get_scaler(first_series, 0, first_column_name)
        else:
            scaler_type = 'standard'
    else:
        # Default to 'standard' for unified scaling
        scaler_type = 'standard'
    
    # Return raw sklearn scaler (works directly in TransformerPipeline per sktime docs)
    # Per sktime: sklearn transformers are applied per series instance automatically
    if scaler_type is None or scaler_type == 'null':
        # No scaling - return passthrough
        from sktime.transformations.series.func_transform import FunctionTransformer
        return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    else:
        # Unified scaling - return raw sklearn scaler (no TabularToSeriesAdaptor)
        return _create_scaler_transformer(scaler_type)


# ============================================================================
# Convenience Function: Simple Uniform Scaling (No ColumnEnsembleTransformer/ColumnTransformer)
# ============================================================================

def create_uniform_scaling_transformer(
    scaler_type: str = 'standard'
) -> Any:
    """Create a unified scaling transformer (StandardScaler for all series).
    
    This function returns a raw sklearn scaler that can be used directly
    in TransformerPipeline without TabularToSeriesAdaptor, per sktime docs.
    When used in TransformerPipeline, sklearn transformers are automatically
    applied per series instance.
    
    **Recommended for factor models**: Unified scaling ensures all series
    contribute proportionally to factor extraction.
    
    Parameters
    ----------
    scaler_type : str, default 'standard'
        Scaler type: 'standard' (recommended), 'robust', 'minmax', 'maxabs', 'quantile'
        
    Returns
    -------
    Any
        Raw sklearn scaler (StandardScaler, RobustScaler, etc.) that can be
        used directly in TransformerPipeline
        
    Examples
    --------
    **Unified scaling (recommended)**:
    
    ```python
    from sktime.transformations.compose import TransformerPipeline
    from sktime.transformations.series.impute import Imputer
    from sklearn.preprocessing import StandardScaler
    
    # Per sktime docs: sklearn transformers work directly
    # Applied per series instance automatically (unified scaling)
    pipeline = TransformerPipeline([
        ("impute_ffill", Imputer(method="ffill")),
        ("impute_bfill", Imputer(method="bfill")),
        ("scaler", StandardScaler()),  # Unified scaling - no wrapper needed!
    ])
    ```
    
    **Using convenience function**:
    
    ```python
    from dfm_python.lightning.scaling import create_uniform_scaling_transformer
    
    scaler = create_uniform_scaling_transformer('standard')
    pipeline = TransformerPipeline([
        ("impute_ffill", Imputer(method="ffill")),
        ("scaler", scaler),  # Unified StandardScaler
    ])
    ```
    """
    _check_sklearn()
    
    return _create_scaler_transformer(scaler_type)


# ============================================================================
# Convenience Function: Create Full Pipeline with Scaling
# ============================================================================

def create_preprocessing_pipeline_with_scaling(
    config: DFMConfig,
    imputation_steps: Optional[List[Any]] = None,
    feature_engineering: Optional[Any] = None,
    scaling_strategy: Optional[ScalingStrategy] = None,
    column_names: Optional[List[str]] = None
) -> Any:
    """Create a complete preprocessing pipeline with automatic per-series scaling.
    
    This is a convenience function that combines imputation, feature engineering,
    and per-series scaling into a single pipeline.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration
    imputation_steps : Optional[List[Any]], default None
        List of imputation transformers (e.g., [Imputer(method="ffill"), ...]).
        If None, no imputation steps are added.
    feature_engineering : Optional[Any], default None
        Feature engineering transformer (e.g., WindowSummarizer).
        If None, no feature engineering is added.
    scaling_strategy : Optional[ScalingStrategy], default None
        Custom scaling strategy. If None, uses DefaultScalingStrategy.
    column_names : Optional[List[str]], default None
        Column names in data. If None, uses config.get_series_ids().
        
    Returns
    -------
    Any
        sktime TransformerPipeline with imputation, feature engineering, and scaling
        
    Examples
    --------
    ```python
    from sktime.transformations.series.impute import Imputer
    from sktime.transformations.series.summarize import WindowSummarizer
    from dfm_python.lightning.scaling import create_preprocessing_pipeline_with_scaling
    
    pipeline = create_preprocessing_pipeline_with_scaling(
        config=config,
        imputation_steps=[
            Imputer(method="ffill"),
            Imputer(method="bfill"),
        ],
        feature_engineering=WindowSummarizer(
            lag_feature={"lag": [1, 2, 3], "mean": [[5, 10]]}
        ),
        scaling_strategy=None  # Auto-detect from config
    )
    ```
    """
    _check_sktime()
    
    from sktime.transformations.compose import TransformerPipeline
    
    steps = []
    
    # Add imputation steps
    if imputation_steps:
        for i, step in enumerate(imputation_steps):
            steps.append((f"impute_{i}", step))
    
    # Add feature engineering
    if feature_engineering is not None:
        steps.append(("feature_engineering", feature_engineering))
    
    # Add unified scaling (StandardScaler for all series)
    scaling_transformer = create_scaling_transformer_from_config(
        config, strategy=scaling_strategy, column_names=column_names
    )
    steps.append(("scaling", scaling_transformer))
    
    return TransformerPipeline(steps)


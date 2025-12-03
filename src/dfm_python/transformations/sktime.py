"""Optional dependency handling for sktime transformers.

This module provides safe imports of sktime transformers with proper
error handling for when sktime is not installed.
"""

try:
    from sktime.transformations.compose import (
        ColumnTransformer,
        TransformerPipeline
    )
    from sktime.transformations.series.log import LogTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import StandardScaler
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    ColumnTransformer = None
    TransformerPipeline = None
    LogTransformer = None
    Differencer = None
    FunctionTransformer = None
    StandardScaler = None


def check_sktime_available():
    """Check if sktime is available and raise ImportError if not.
    
    Raises
    ------
    ImportError
        If sktime is not installed, with helpful installation message.
    """
    if not HAS_SKTIME:
        raise ImportError(
            "sktime is required for sktime transformers. "
            "Install it with: pip install dfm-python[transform]"
        )


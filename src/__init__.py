"""Application-specific utilities and wrappers.

Note: The core DFM module is now available as the 'dfm-python' package on PyPI.
Import it directly: from dfm_python import DFMConfig, dfm, load_config

This src/ package now only contains application-specific utilities.
"""

__version__ = "0.1.0"

# Only export utilities - DFM module is now in dfm_python package
from .utils import summarize

__all__ = [
    'summarize',
]


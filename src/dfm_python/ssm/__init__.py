"""State-space models.

- ``DFMKalmanFilter``: Kalman filter and Durbin-Koopman smoother of the DFM
  E-step (per-element missing data, exact log-likelihood).
- ``BaseSSM``: base class of the iVDFM SSMs (innovations to factors).
- ``iVDFMCompanionSSM``: block-diagonal companion-form SSM of iVDFM.

The iVDFM SSMs are torch modules, imported on first access (they need the
``deep`` extra).
"""

from ..utils.optional import lazy_torch_exports
from .kalman import DFMKalmanFilter

__all__ = [
    'DFMKalmanFilter',
    'BaseSSM',
    'iVDFMCompanionSSM',
]

__getattr__ = lazy_torch_exports(__name__, {'BaseSSM': '.base', 'iVDFMCompanionSSM': '.companion'})

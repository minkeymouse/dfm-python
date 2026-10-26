"""Identifiable Variational Dynamic Factor Model (iVDFM).

``iVDFM`` is the public estimator; the training code is the private engine
``_engine.iVDFMEngine``. The network components are torch modules, imported
on first access (they need the ``deep`` extra).
"""

from ...utils.optional import lazy_torch_exports
from .estimator import iVDFM

__all__ = [
    'iVDFM',
    'iVDFMInnovationEncoder',
    'iVDFMDecoder',
    'iVDFMPriorNetwork',
    'RegimeNet',
]

__getattr__ = lazy_torch_exports(__name__, {
    'iVDFMInnovationEncoder': '.encoder',
    'iVDFMDecoder': '.decoder',
    'iVDFMPriorNetwork': '.prior',
    'RegimeNet': '...layer.regime',
})

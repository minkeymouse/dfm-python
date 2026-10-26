"""Deep Dynamic Factor Model (DDFM) and its components.

``DDFM`` is the public estimator; the training code lives in the private
engine ``_engine.DDFMEngine``. The network components are torch modules,
imported on first access (they need the ``deep`` extra).
"""

from ...utils.optional import lazy_torch_exports
from .estimator import DDFM

__all__ = [
    'DDFM',
    'DDFMEncoder',
    'DDFMLinearDecoder',
    'DDFMMLPDecoder',
    'SimpleAutoencoder',
    'extract_decoder_params',
]

__getattr__ = lazy_torch_exports(__name__, {
    'DDFMEncoder': '.encoder',
    'SimpleAutoencoder': '.encoder',
    'extract_decoder_params': '.encoder',
    'DDFMLinearDecoder': '.decoder',
    'DDFMMLPDecoder': '.decoder',
})

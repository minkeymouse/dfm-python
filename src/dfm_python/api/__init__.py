"""High-level API subpackage for Dynamic Factor Models.

This subpackage provides:
- Model API classes (DFMBase, DFM, DDFM) in model_api.py
- Session API (singleton instances and module-level functions) in session_api.py
"""

from .model_api import DFMBase, DFM, DDFM
from .session_api import (
    _dfm_instance,
    _ddfm_instance,
    load_config,
    load_data,
    train,
    predict,
    plot,
    load_pickle,
    reset,
    load_config_ddfm,
    load_data_ddfm,
    train_ddfm,
    predict_ddfm,
    plot_ddfm,
    reset_ddfm,
    create_model,
    from_yaml,
    from_spec,
    from_spec_df,
    from_dict,
)

__all__ = [
    # High-level API classes
    'DFMBase', 'DFM', 'DDFM',
    # Singleton instances (internal use)
    '_dfm_instance', '_ddfm_instance',
    # Module-level convenience functions (DFM)
    'load_config', 'load_data', 'load_pickle', 'train', 'predict', 'plot', 'reset',
    # Module-level convenience functions (DDFM)
    'load_config_ddfm', 'load_data_ddfm', 'train_ddfm', 'predict_ddfm', 'plot_ddfm', 'reset_ddfm',
    # Factory and convenience constructors
    'create_model', 'from_yaml', 'from_spec', 'from_spec_df', 'from_dict',
]


"""PyTorch Lightning DataModule for KDFM training.

This module provides KDFMDataModule for KDFM models.
KDFM uses the same data format as DFM/DDFM, so we can reuse DFMDataModule.
"""

from .dfm_dm import DFMDataModule
from ..logger import get_logger

_logger = get_logger(__name__)


class KDFMDataModule(DFMDataModule):
    """PyTorch Lightning DataModule for KDFM training.
    
    KDFM uses the same data format as DFM/DDFM, so this class
    reuses DFMDataModule functionality. No special preprocessing needed.
    
    **Usage Pattern**:
    - Same as DFMDataModule
    - Data can contain NaN values - KDFM will handle them
    - Users handle preprocessing (imputation, scaling) before passing data
    
    Parameters
    ----------
    config : KDFMConfig or DFMConfig
        Model configuration object (KDFMConfig inherits from BaseModelConfig)
    pipeline : Any, optional
        sktime-compatible preprocessing pipeline (same as DFMDataModule)
    data_path : str or Path, optional
        Path to data file (CSV)
    data : np.ndarray or pd.DataFrame, optional
        Data array or DataFrame
    preprocessed : bool, default False
        Whether data is already preprocessed
    time_index : TimeIndex, optional
        Time index for the data
    time_index_column : str or list of str, optional
        Column name(s) in DataFrame to use as time index
    batch_size : int, optional
        Batch size for DataLoader
    num_workers : int, default 0
        Number of worker processes for DataLoader
    val_split : float, optional
        Validation split ratio (0.0 to 1.0)
    """
    
    # KDFMDataModule reuses all functionality from DFMDataModule
    # No additional methods needed - same data format and handling


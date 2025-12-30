"""PyTorch Lightning DataModule for KDFM training.

This module provides KDFMDataModule for KDFM models.
KDFM uses the same data format as DFM/DDFM, but needs Lightning DataModule
interface with Dataset/DataLoader for Lightning training compatibility.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
import pytorch_lightning as lightning_pl

from .base import BaseDataModule
from .dfm_dm import DFMDataModule
from ..dataset.dataset import KDFMDataset
from ..dataset.dataloader import create_kdfm_dataloader
from ..logger import get_logger

_logger = get_logger(__name__)


class KDFMDataModule(BaseDataModule, lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for KDFM training.
    
    KDFM uses the same data format as DFM/DDFM, but this class extends
    DFMDataModule to provide Lightning DataModule interface with Dataset/DataLoader
    for Lightning training compatibility.
    
    **Usage Pattern**:
    - Same as DFMDataModule for data loading/preprocessing
    - Adds Lightning DataModule interface (train_dataloader, val_dataloader)
    - Uses KDFMDataset for Lightning compatibility
    - Data can contain NaN values - KDFM will handle them
    - Users handle preprocessing (imputation, scaling) before passing data
    
    Parameters
    ----------
    config : KDFMConfig or DFMConfig
        Model configuration object (KDFMConfig inherits from BaseModelConfig)
    scaler : Any, optional
        Optional scaler for extracting Mx/Wx statistics (same as DFMDataModule).
        Can be a scaler instance (e.g., StandardScaler, RobustScaler) or None.
    data_path : str or Path, optional
        Path to data file (CSV)
    data : np.ndarray or pd.DataFrame, optional
        Preprocessed data array or DataFrame. Data must be preprocessed before passing.
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
    
    def __init__(
        self,
        config=None,
        config_path=None,
        scaler=None,
        data_path=None,
        data=None,
        time_index=None,
        time_index_column=None,
        batch_size=None,
        num_workers=0,
        val_split=None,
        **kwargs
    ):
        # Initialize LightningDataModule first (no arguments)
        lightning_pl.LightningDataModule.__init__(self)
        # Initialize BaseDataModule
        BaseDataModule.__init__(
            self,
            config=config,
            config_path=config_path,
            data_path=data_path,
            data=data,
            time_index=time_index,
            time_index_column=time_index_column,
            **kwargs
        )
        
        # Create internal DFMDataModule for preprocessing (composition)
        # Extract scaler from pipeline if provided (for backward compatibility)
        # Note: pipeline parameter is deprecated, use scaler directly
        scaler_to_use = scaler
        if scaler_to_use is None and 'pipeline' in kwargs:
            # Backward compatibility: try to extract scaler from pipeline
            try:
                from ..dataset.process import _get_scaler
                scaler_to_use = _get_scaler(kwargs['pipeline'])
            except (AttributeError, ImportError):
                pass
        
        self._dfm_dm = DFMDataModule(
            config=config,
            config_path=config_path,
            data_path=data_path,
            data=data,
            scaler=scaler_to_use,  # Use scaler instead of pipeline
            time_index=time_index,
            time_index_column=time_index_column,
            **kwargs
        )
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset: Optional[KDFMDataset] = None
        self.val_dataset: Optional[KDFMDataset] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and prepare data, create datasets.
        
        This method uses internal DFMDataModule for data loading/preprocessing,
        then creates KDFMDataset objects for Lightning training.
        """
        # Use internal DFMDataModule for preprocessing
        self._dfm_dm.setup()
        
        # Get processed data from internal DFMDataModule
        data_processed = self._dfm_dm.get_processed_data()
        
        # Copy Mx, Wx from internal DFMDataModule
        self.Mx = self._dfm_dm.Mx
        self.Wx = self._dfm_dm.Wx
        
        # Convert to torch tensor if needed
        from ..config.constants import DEFAULT_TORCH_DTYPE
        if not isinstance(data_processed, torch.Tensor):
            data_processed = torch.tensor(data_processed, dtype=DEFAULT_TORCH_DTYPE)
        
        # Create train/val splits if requested
        if self.val_split is not None and 0 < self.val_split < 1:
            T = data_processed.shape[0]
            split_idx = int(T * (1 - self.val_split))
            
            train_data = data_processed[:split_idx, :]
            val_data = data_processed[split_idx:, :]
            
            # Create KDFM datasets
            self.train_dataset = KDFMDataset(train_data)
            self.val_dataset = KDFMDataset(val_data)
        else:
            # Use all data for training
            self.train_dataset = KDFMDataset(data_processed)
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule train_dataloader failed: setup() must be called before train_dataloader(). "
                "Please call dm.setup() first to load and preprocess data."
            )
        
        return create_kdfm_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data array."""
        if self.train_dataset is None:
            raise RuntimeError("DataModule setup() must be called before get_processed_data()")
        # Get data from train_dataset
        return self.train_dataset.data
    
    def get_initialization_params(self) -> Dict[str, Any]:
        """Get initialization parameters for DFM model (delegates to internal DFMDataModule)."""
        return self._dfm_dm.get_initialization_params()
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create DataLoader for validation."""
        if self.val_dataset is None:
            return None
        
        return create_kdfm_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )


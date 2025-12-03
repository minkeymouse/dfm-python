"""PyTorch Lightning DataModule for DFM training.

This module provides LightningDataModule implementations for loading and
preprocessing data for Dynamic Factor Model training.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import polars as pl
from typing import Optional, Union, Tuple, Any
from pathlib import Path
import pytorch_lightning as lightning_pl

from ..config import DFMConfig
from ..data.utils import load_data as _load_data
from ..data.dataset import DFMDataset, DDFMDataset
from ..data.dataloader import create_dfm_dataloader, create_ddfm_dataloader
from ..utils.time import TimeIndex
from ..logger import get_logger

_logger = get_logger(__name__)


def _check_sktime_available():
    """Check if sktime is available and raise ImportError if not."""
    try:
        import sktime
        return True
    except ImportError:
        raise ImportError(
            "sktime is required for sktime transformers. "
            "Install it with: pip install sktime"
        )


class DFMDataModule(lightning_pl.LightningDataModule):
    """PyTorch Lightning DataModule for DFM training.
    
    This DataModule handles:
    1. Data loading from file or array
    2. Preprocessing using user-provided sktime transformer
    3. Creating DataLoaders for training
    4. Optional train/val splits
    
    Users must provide a sktime-compatible transformer (e.g., ColumnTransformer,
    TransformerPipeline) that handles transformations and standardization.
    The transformer should support Polars DataFrames via set_output(transform="polars").
    
    For linear DFM, this uses DFMDataset which returns full sequences.
    For DDFM, use DDFMDataModule which uses DDFMDataset with windowing.
    
    Parameters
    ----------
    config : DFMConfig
        DFM configuration object
    transformer : Any
        Sktime-compatible transformer (ColumnTransformer, TransformerPipeline, etc.)
        Must implement fit_transform() and support Polars output.
        Required parameter - users must provide their own transformer.
    data_path : str or Path, optional
        Path to data file (CSV). If None, data must be provided in setup().
    data : np.ndarray or pl.DataFrame, optional
        Data array or DataFrame. If None, data_path must be provided.
    time_index : TimeIndex, optional
        Time index for the data
    batch_size : int, optional
        Batch size for DataLoader. For DFM, typically 1 (full sequence).
    num_workers : int, default 0
        Number of worker processes for DataLoader
    val_split : float, optional
        Validation split ratio (0.0 to 1.0). If None, no validation split.
    """
    
    def __init__(
        self,
        config: DFMConfig,
        transformer: Any,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[Union[np.ndarray, pl.DataFrame]] = None,
        time_index: Optional[TimeIndex] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        val_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        _check_sktime_available()
        
        if transformer is None:
            raise ValueError(
                "transformer is required. Users must provide a sktime-compatible transformer "
                "(e.g., ColumnTransformer, TransformerPipeline). "
                "See documentation for examples of creating transformers from config."
            )
        
        self.config = config
        self.transformer = transformer
        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data
        self.time_index = time_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Will be set in setup()
        self.train_dataset: Optional[DFMDataset] = None
        self.val_dataset: Optional[DFMDataset] = None
        self.Mx: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.data_processed: Optional[torch.Tensor] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load and preprocess data.
        
        This method is called by Lightning to set up the data module.
        It loads data, applies user-provided transformer, and creates train/val datasets.
        """
        # Load data if not already provided
        if self.data is None:
            if self.data_path is None:
                raise ValueError("Either data_path or data must be provided")
            
            # Load data from file
            # Note: load_data returns (X, Time, Z) where X and Z are both raw data
            X, Time, Z = _load_data(
                self.data_path,
                self.config,
            )
            # Use raw data (X or Z, they're the same)
            self.data = X
            self.time_index = Time
        
        # Convert to Polars DataFrame if needed
        if isinstance(self.data, np.ndarray):
            series_ids = self.config.get_series_ids()
            X_df = pl.DataFrame(self.data, schema=series_ids)
        elif isinstance(self.data, pl.DataFrame):
            X_df = self.data
        else:
            raise TypeError(f"Unsupported data type: {type(self.data)}")
        
        # Try to set Polars output if transformer supports it
        try:
            if hasattr(self.transformer, 'set_output'):
                self.transformer.set_output(transform="polars")
        except (AttributeError, ValueError):
            # Transformer doesn't support set_output or Polars output
            # Will convert to pandas/numpy as needed
            pass
        
        # Apply user-provided transformer
        try:
            X_transformed = self.transformer.fit_transform(X_df)
        except Exception as e:
            raise ValueError(
                f"Transformer failed to fit_transform data: {e}. "
                f"Ensure transformer is sktime-compatible and supports Polars DataFrames."
            ) from e
        
        # Convert to Polars DataFrame if not already
        if not isinstance(X_transformed, pl.DataFrame):
            # Try to convert from pandas or numpy
            if hasattr(X_transformed, 'to_polars'):
                X_transformed = X_transformed.to_polars()
            elif hasattr(X_transformed, 'values'):
                # Pandas DataFrame or numpy array
                X_transformed = pl.DataFrame(X_transformed.values)
            else:
                # Numpy array
                X_transformed = pl.DataFrame(X_transformed)
        
        # Try to extract standardization parameters if transformer includes StandardScaler
        # This is optional - some transformers may not have standardization
        # Mx and Wx are already initialized in __init__
        
        # Try to extract standardization parameters if transformer includes StandardScaler
        # This is optional - some transformers may not have standardization
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Check if transformer is a pipeline with StandardScaler
            if hasattr(self.transformer, 'steps'):
                for name, step in self.transformer.steps:
                    if isinstance(step, StandardScaler):
                        if hasattr(step, 'mean_') and hasattr(step, 'scale_'):
                            mean_val = step.mean_
                            scale_val = step.scale_
                            # Convert to numpy array if needed
                            if not isinstance(mean_val, np.ndarray):
                                mean_val = np.asarray(mean_val)
                            if not isinstance(scale_val, np.ndarray):
                                scale_val = np.asarray(scale_val)
                            self.Mx = mean_val
                            self.Wx = scale_val
                            break
            elif hasattr(self.transformer, 'transformers'):
                # ColumnTransformer - check each transformer
                for name, trans, cols in self.transformer.transformers:
                    if isinstance(trans, StandardScaler):
                        if hasattr(trans, 'mean_') and hasattr(trans, 'scale_'):
                            mean_val = trans.mean_
                            scale_val = trans.scale_
                            # Convert to numpy array if needed
                            if not isinstance(mean_val, np.ndarray):
                                mean_val = np.asarray(mean_val)
                            if not isinstance(scale_val, np.ndarray):
                                scale_val = np.asarray(scale_val)
                            self.Mx = mean_val
                            self.Wx = scale_val
                            break
        except (AttributeError, ImportError):
            # StandardScaler not found or not accessible
            pass
        
        # Convert to numpy then to torch tensor
        X_processed_np = X_transformed.to_numpy()
        self.data_processed = torch.tensor(X_processed_np, dtype=torch.float32)
        
        # Create train/val splits if requested
        if self.val_split is not None and 0 < self.val_split < 1:
            T = self.data_processed.shape[0]
            split_idx = int(T * (1 - self.val_split))
            
            train_data = self.data_processed[:split_idx, :]
            val_data = self.data_processed[split_idx:, :]
            
            # For linear DFM, use full sequences (no windowing)
            self.train_dataset = DFMDataset(train_data)
            self.val_dataset = DFMDataset(val_data)
        else:
            # Use all data for training
            self.train_dataset = DFMDataset(self.data_processed)
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        
        return create_dfm_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create DataLoader for validation."""
        if self.val_dataset is None:
            return None
        
        return create_dfm_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_standardization_params(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get standardization parameters (Mx, Wx) if available.
        
        Returns
        -------
        Mx : np.ndarray or None
            Mean values (N,) if transformer includes StandardScaler, None otherwise
        Wx : np.ndarray or None
            Standard deviation values (N,) if transformer includes StandardScaler, None otherwise
        """
        if self.data_processed is None:
            raise RuntimeError("setup() must be called before get_standardization_params()")
        return self.Mx, self.Wx
    
    def get_transformer(self) -> Any:
        """Get the transformer used for preprocessing.
        
        Returns
        -------
        transformer : Any
            The sktime transformer provided by the user
        """
        return self.transformer
    
    def get_processed_data(self) -> torch.Tensor:
        """Get processed data tensor.
        
        Returns
        -------
        data : torch.Tensor
            Processed data (T x N)
        """
        if self.data_processed is None:
            raise RuntimeError("setup() must be called before get_processed_data()")
        return self.data_processed


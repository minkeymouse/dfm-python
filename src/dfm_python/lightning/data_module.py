"""PyTorch Lightning DataModule for DFM training.

This module provides LightningDataModule implementations for loading and
preprocessing data for Dynamic Factor Model training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from typing import Optional, Union, Tuple, List, Any
from pathlib import Path
import pytorch_lightning as lightning_pl

from ..config import DFMConfig
from ..transformations.utils import load_data as _load_data
from ..transformations.sktime import check_sktime_available
from ..utils.time import TimeIndex
from ..logger import get_logger

_logger = get_logger(__name__)


class DFMDataset(Dataset):
    """PyTorch Dataset for DFM time series data.
    
    This dataset handles time series data for DFM training. For standard
    DFM training, the entire sequence is used. For batch training (DDFM),
    sequences are split into windows.
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor (T x N) where T is time periods and N is number of series
    window_size : int, optional
        Window size for creating sequences. If None, uses full sequence.
    stride : int, default 1
        Stride for windowing. Default 1 means overlapping windows.
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        window_size: Optional[int] = None,
        stride: int = 1
    ):
        self.data = data
        self.T, self.N = data.shape
        self.window_size = window_size if window_size is not None else self.T
        self.stride = stride
        
        # Compute number of samples
        if self.window_size >= self.T:
            self.n_samples = 1
        else:
            self.n_samples = (self.T - self.window_size) // stride + 1
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.
        
        Returns
        -------
        x : torch.Tensor
            Input data (window_size x N)
        target : torch.Tensor
            Target data (same as x for autoencoder/reconstruction)
        """
        if self.window_size >= self.T:
            # Return full sequence
            x = self.data
        else:
            # Return window
            start_idx = idx * self.stride
            end_idx = start_idx + self.window_size
            x = self.data[start_idx:end_idx, :]
        
        # For autoencoder/reconstruction tasks, target is same as input
        target = x.clone()
        
        return x, target


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
    batch_size : int, default 32
        Batch size for DataLoader
    window_size : int, optional
        Window size for sequence batching. If None, uses full sequence.
    stride : int, default 1
        Stride for windowing
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
        batch_size: int = 32,
        window_size: Optional[int] = None,
        stride: int = 1,
        num_workers: int = 0,
        val_split: Optional[float] = None,
        **kwargs
    ):
        super().__init__()
        check_sktime_available()
        
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
        self.window_size = window_size
        self.stride = stride
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
        
        # Check if transformer has StandardScaler in its pipeline
        if hasattr(self.transformer, 'steps') or hasattr(self.transformer, 'transformers'):
            # Try to find StandardScaler in pipeline
            try:
                from sklearn.preprocessing import StandardScaler
                from ..transformations.sktime import StandardScaler as SktimeStandardScaler
                
                # Check if transformer is a pipeline with StandardScaler
                if hasattr(self.transformer, 'steps'):
                    for name, step in self.transformer.steps:
                        if isinstance(step, (StandardScaler, SktimeStandardScaler)):
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
                        if isinstance(trans, (StandardScaler, SktimeStandardScaler)):
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
            
            self.train_dataset = DFMDataset(
                train_data,
                window_size=self.window_size,
                stride=self.stride
            )
            self.val_dataset = DFMDataset(
                val_data,
                window_size=self.window_size,
                stride=self.stride
            )
        else:
            # Use all data for training
            self.train_dataset = DFMDataset(
                self.data_processed,
                window_size=self.window_size,
                stride=self.stride
            )
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before train_dataloader()")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create DataLoader for validation."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for validation
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
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


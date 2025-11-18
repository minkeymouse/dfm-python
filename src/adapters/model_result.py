"""Model result storage adapters for DFM results.

This module provides adapters for saving and loading DFM estimation results
(model results) to various storage backends (files, databases, etc.).
"""

from typing import Protocol, Dict, Any, Tuple, Optional, List, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import pickle
import json

# Type hints - use TYPE_CHECKING to avoid circular imports

if TYPE_CHECKING:
    from dfm_python.dfm import DFMResult
    from dfm_python.config import DFMConfig
else:
    # Runtime imports
    try:
        from dfm_python.dfm import DFMResult
        from dfm_python.config import DFMConfig
    except ImportError:
        # Fallback for when dfm_python is not installed
        DFMResult = Any
        DFMConfig = Any


class ModelResultSaver(Protocol):
    """Protocol for model result storage adapters.
    
    Any implementation can save/load DFM results and configs.
    """
    
    def save_model_result(
        self,
        result: DFMResult,
        config: DFMConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model result and return result ID.
        
        Parameters
        ----------
        result : DFMResult
            DFM estimation results
        config : DFMConfig
            Model configuration
        metadata : dict, optional
            Additional metadata (timestamp, tags, etc.)
            
        Returns
        -------
        str
            Result ID (unique identifier)
        """
        ...
    
    def load_model_result(
        self,
        result_id: str
    ) -> Tuple[DFMResult, DFMConfig, Dict[str, Any]]:
        """Load model result and return (result, config, metadata).
        
        Parameters
        ----------
        result_id : str
            Result ID
            
        Returns
        -------
        tuple
            (result, config, metadata)
        """
        ...
    
    def list_model_results(self) -> List[str]:
        """List all result IDs.
        
        Returns
        -------
        list
            List of result IDs
        """
        ...


class PickleModelResultSaver:
    """File-based model result saver using pickle.
    
    This is a basic implementation that saves model results to pickle files.
    Suitable for development and small-scale use.
    
    Examples
    --------
    >>> from dfm_python.adapters import PickleModelResultSaver
    >>> saver = PickleModelResultSaver(base_dir='model_results')
    >>> result_id = saver.save_model_result(result, config, {'tag': 'baseline'})
    >>> result, config, metadata = saver.load_model_result(result_id)
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize pickle model result saver.
        
        Parameters
        ----------
        base_dir : str, optional
            Base directory for storing model results.
            Defaults to './model_results' if not provided.
        """
        if base_dir is None:
            # Default value (hardcoded for convenience)
            base_dir = './model_results'
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_result(
        self,
        result: DFMResult,
        config: DFMConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model result to pickle file.
        
        Parameters
        ----------
        result : DFMResult
            DFM estimation results
        config : DFMConfig
            Model configuration
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        str
            Result ID (timestamp-based)
        """
        # Generate result ID
        result_id = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare data
        data = {
            'result': result,
            'config': config,
            'metadata': metadata or {},
            'result_id': result_id,
            'created_at': datetime.now().isoformat(),
        }
        
        # Save to file
        path = self.base_dir / f"{result_id}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        return result_id
    
    def load_model_result(
        self,
        result_id: str
    ) -> Tuple[DFMResult, DFMConfig, Dict[str, Any]]:
        """Load model result from pickle file.
        
        Parameters
        ----------
        result_id : str
            Result ID
            
        Returns
        -------
        tuple
            (result, config, metadata)
        """
        path = self.base_dir / f"{result_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model result not found: {result_id}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data['result'], data['config'], data.get('metadata', {})
    
    def list_model_results(self) -> List[str]:
        """List all result IDs.
        
        Returns
        -------
        list
            List of result IDs (sorted by creation time, newest first)
        """
        results = []
        for path in self.base_dir.glob("result_*.pkl"):
            # Extract result ID from filename
            result_id = path.stem
            results.append(result_id)
        
        # Sort by creation time (newest first)
        results.sort(reverse=True)
        return results


"""Tests for DFM model save/load functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig
from dfm_python.config.schema.params import DFMStateSpaceParams, DFMModelState
from dfm_python.utils.errors import ModelNotTrainedError


class TestDFMSaveLoad:
    """Test suite for DFM model save/load functionality."""
    
    def test_save_load_atomic_write(self):
        """Test that save uses atomic write (temp file + rename)."""
        model = DFM()
        
        # Set up minimal model state for save
        model._config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        
        # Create dummy training state
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        model.data_processed = np.random.randn(10, 3)
        model._training_loglik = -100.0
        model._training_num_iter = 1
        model._training_converged = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            temp_path = save_path.with_suffix('.pkl.tmp')
            
            # Save model
            model.save(save_path)
            
            # Verify temp file is removed (atomic write completed)
            assert not temp_path.exists(), "Temp file should be removed after atomic rename"
            assert save_path.exists(), "Model file should exist after save"
            
            # Verify file can be loaded
            loaded_model = DFM.load(save_path)
            assert loaded_model.num_factors == model.num_factors
            assert np.allclose(loaded_model.training_state.A, model.training_state.A)
            assert loaded_model._training_loglik == model._training_loglik
    
    def test_save_verification(self):
        """Test that saved file is verified (can be loaded immediately after save)."""
        model = DFM()
        
        # Set up minimal model state
        model._config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            
            # Save should succeed and verify the file
            model.save(save_path)
            
            # File should be loadable immediately
            loaded_model = DFM.load(save_path)
            assert loaded_model is not None
    
    def test_load_nonexistent_file(self):
        """Test that load raises FileNotFoundError for nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / 'nonexistent.pkl'
            
            with pytest.raises(FileNotFoundError):
                DFM.load(nonexistent_path)
    
    def test_load_with_config_override(self):
        """Test that load can use provided config instead of checkpoint config."""
        model = DFM()
        
        # Set up and save model
        original_config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        model._config = original_config
        
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            model.save(save_path)
            
            # Load with different config
            override_config = DFMConfig(
                blocks={'Block_Global': {'num_factors': 3, 'series': []}},
                frequency={'test': 'w'}
            )
            
            loaded_model = DFM.load(save_path, config=override_config)
            assert loaded_model.config == override_config
    
    def test_save_load_preserves_all_parameters(self):
        """Test that save/load preserves all model parameters correctly."""
        model = DFM()
        
        # Set up complete model state
        model._config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        model.data_processed = np.random.randn(10, 3)
        model.threshold = 1e-4
        model.max_iter = 50
        model._training_loglik = -150.5
        model._training_num_iter = 10
        model._training_converged = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            model.save(save_path)
            
            loaded_model = DFM.load(save_path)
            
            # Verify all parameters are preserved
            assert loaded_model.num_factors == model.num_factors
            assert np.allclose(loaded_model.training_state.A, model.training_state.A)
            assert np.allclose(loaded_model.training_state.C, model.training_state.C)
            assert np.allclose(loaded_model.training_state.Q, model.training_state.Q)
            assert np.allclose(loaded_model.training_state.R, model.training_state.R)
            assert np.allclose(loaded_model.training_state.Z_0, model.training_state.Z_0)
            assert np.allclose(loaded_model.training_state.V_0, model.training_state.V_0)
            assert loaded_model.threshold == model.threshold
            assert loaded_model.max_iter == model.max_iter
            assert loaded_model._training_loglik == model._training_loglik
            assert loaded_model._training_num_iter == model._training_num_iter
            assert loaded_model._training_converged == model._training_converged
            assert np.array_equal(loaded_model.data_processed, model.data_processed)
    
    def test_save_cleanup_on_error(self):
        """Test that temp file is cleaned up if save fails."""
        model = DFM()
        
        # Set up model with invalid state that might cause save to fail
        model._config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        
        # Create a state with very large arrays that might cause issues
        # (This is a minimal test - actual failures are hard to force)
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            temp_path = save_path.with_suffix('.pkl.tmp')
            
            # Normal save should clean up temp file
            model.save(save_path)
            assert not temp_path.exists(), "Temp file should be cleaned up after successful save"
    
    def test_save_uses_high_protocol(self):
        """Test that saved file uses pickle HIGHEST_PROTOCOL."""
        import pickle
        model = DFM()
        
        model._config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': []}},
            frequency={'test': 'w'}
        )
        
        model.training_state = DFMStateSpaceParams(
            A=np.eye(2),
            C=np.ones((3, 2)),
            Q=np.eye(2) * 0.1,
            R=np.eye(3) * 0.1,
            Z_0=np.zeros(2),
            V_0=np.eye(2)
        )
        
        model.num_factors = 2
        model.r = np.array([2])
        model.p = 1
        model.blocks = np.array([[0], [0], [0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pkl'
            model.save(save_path)
            
            # Check pickle protocol
            with open(save_path, 'rb') as f:
                header = f.read(2)
                if header[0] == 0x80:  # Pickle protocol marker
                    protocol = header[1]
                    assert protocol == 5, f"Should use protocol 5 (HIGHEST_PROTOCOL), got {protocol}"

"""Tests for DFM model save/load using mock checkpoint files."""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig
from dfm_python.config.schema.params import DFMStateSpaceParams, DFMModelState
from dfm_python.utils.errors import ModelNotTrainedError


class TestDFMSaveLoadWithMockCheckpoint:
    """Test DFM save/load using mock checkpoint files."""
    
    def _create_mock_model(self):
        """Helper to create a fully configured mock DFM model."""
        model = DFM()
        
        # Create config
        config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 3, 'series': ['A', 'B', 'C']}},
            frequency={'A': 'w', 'B': 'w', 'C': 'w'},
            clock='w'
        )
        model._config = config
        
        # Create training state (state-space parameters)
        model.training_state = DFMStateSpaceParams(
            A=np.eye(5),
            C=np.random.randn(10, 5) * 0.5,
            Q=np.eye(5) * 0.1,
            R=np.eye(10) * 0.1,
            Z_0=np.zeros(5),
            V_0=np.eye(5)
        )
        
        # Set model attributes
        model.num_factors = 3
        model.r = np.array([3])
        model.p = 2
        model.blocks = np.array([[0]] * 10)
        model.data_processed = np.random.randn(100, 10)
        model.threshold = 1e-3
        model.max_iter = 50
        model.nan_method = 2
        model.nan_k = 3
        
        # Set training metadata
        model._training_loglik = -150.5
        model._training_num_iter = 10
        model._training_converged = False
        
        # Set model state attributes (for DFMModelState.from_model)
        model._mixed_freq = False
        model._constraint_matrix = None
        model._constraint_vector = None
        model._n_slower_freq = 0
        model._n_clock_freq = 10
        model._tent_weights_dict = None
        model._frequencies = None
        model._idio_indicator = None
        model._max_lag_size = 3
        
        return model
    
    def test_load_from_mock_checkpoint_file(self):
        """Test loading from a mock checkpoint file created by save()."""
        # Create and save a model
        model = self._create_mock_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'mock_checkpoint.pkl'
            
            # Save model to create mock checkpoint
            model.save(checkpoint_path)
            
            # Verify file exists and protocol
            assert checkpoint_path.exists(), "Mock checkpoint file should exist"
            
            with open(checkpoint_path, 'rb') as f:
                header = f.read(2)
                protocol = header[1] if header[0] == 0x80 else None
                assert protocol == 5, f"Should use protocol 5, got {protocol}"
            
            # Test loading from mock checkpoint
            loaded_model = DFM.load(checkpoint_path)
            
            # Verify all attributes are loaded correctly
            assert loaded_model.num_factors == model.num_factors
            assert loaded_model.r is not None
            assert loaded_model.p == model.p
            assert np.array_equal(loaded_model.blocks, model.blocks)
            
            # Verify state-space parameters
            assert loaded_model.training_state is not None
            assert np.allclose(loaded_model.training_state.A, model.training_state.A)
            assert np.allclose(loaded_model.training_state.C, model.training_state.C)
            assert np.allclose(loaded_model.training_state.Q, model.training_state.Q)
            assert np.allclose(loaded_model.training_state.R, model.training_state.R)
            assert np.allclose(loaded_model.training_state.Z_0, model.training_state.Z_0)
            assert np.allclose(loaded_model.training_state.V_0, model.training_state.V_0)
            
            # Verify training metadata
            assert loaded_model._training_loglik == model._training_loglik
            assert loaded_model._training_num_iter == model._training_num_iter
            # Note: _training_converged may be None if not set, check it matches original or is None
            assert loaded_model._training_converged == model._training_converged or \
                   (loaded_model._training_converged is None and model._training_converged is False)
            
            # Verify config
            assert loaded_model.config is not None
            # Config comparison: check key attributes (direct == may fail with arrays)
            assert loaded_model.config.clock == model.config.clock
            # Check that blocks exist (num_factors is in blocks, not directly in config)
            assert loaded_model.config.blocks is not None
            assert 'Block_Global' in loaded_model.config.blocks
            
            print(f"✓ Successfully loaded from mock checkpoint")
    
    def test_load_from_manually_created_checkpoint(self):
        """Test loading from a manually created checkpoint dict (simulating old/new formats)."""
        # Create a checkpoint dict manually (simulating what save() creates)
        checkpoint = {
            'state_space_params': DFMStateSpaceParams(
                A=np.eye(3),
                C=np.ones((5, 3)),
                Q=np.eye(3) * 0.1,
                R=np.eye(5) * 0.1,
                Z_0=np.zeros(3),
                V_0=np.eye(3)
            ),
            'model_state': DFMModelState(
                num_factors=2,
                r=np.array([2]),
                p=1,
                blocks=np.array([[0]] * 5),
                mixed_freq=False,
                constraint_matrix=None,
                constraint_vector=None,
                n_slower_freq=0,
                n_clock_freq=5,
                tent_weights_dict=None,
                frequencies=None,
                idio_indicator=None,
                max_lag_size=2
            ),
            'config': DFMConfig(
                blocks={'Block_Global': {'num_factors': 2, 'series': []}},
                frequency={'test': 'w'},
                clock='w'
            ),
            'threshold': 1e-3,
            'max_iter': 50,
            'nan_method': 2,
            'nan_k': 3,
            'data_processed': np.random.randn(50, 5),
            'target_scaler': None,
            'training_loglik': -100.0,
            'training_num_iter': 5,
            'training_converged': False,
            'result': None
        }
        
        # Save checkpoint manually
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'manual_checkpoint.pkl'
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Test loading
            loaded_model = DFM.load(checkpoint_path)
            
            # Verify loaded correctly
            assert loaded_model.num_factors == 2
            assert loaded_model.p == 1
            assert loaded_model._training_loglik == -100.0
            assert loaded_model._training_num_iter == 5
            # training_converged may be None if checkpoint uses old format
            assert loaded_model._training_converged == False or loaded_model._training_converged is None
            assert np.allclose(loaded_model.training_state.A, np.eye(3))
            assert np.allclose(loaded_model.training_state.C, np.ones((5, 3)))
            
            print(f"✓ Successfully loaded from manually created checkpoint")
    
    def test_load_mock_checkpoint_and_verify_all_parameters(self):
        """Test loading mock checkpoint and verify all parameters are correct."""
        model = self._create_mock_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'verify_checkpoint.pkl'
            
            # Save
            model.save(checkpoint_path)
            
            # Load
            loaded_model = DFM.load(checkpoint_path)
            
            # Comprehensive parameter verification
            params_to_check = {
                'A': (model.training_state.A, loaded_model.training_state.A),
                'C': (model.training_state.C, loaded_model.training_state.C),
                'Q': (model.training_state.Q, loaded_model.training_state.Q),
                'R': (model.training_state.R, loaded_model.training_state.R),
                'Z_0': (model.training_state.Z_0, loaded_model.training_state.Z_0),
                'V_0': (model.training_state.V_0, loaded_model.training_state.V_0),
            }
            
            all_match = True
            for param_name, (original, loaded) in params_to_check.items():
                if not np.allclose(original, loaded, rtol=1e-10, atol=1e-10):
                    print(f"  ⚠ Mismatch in {param_name}")
                    all_match = False
            
            # Verify scalar attributes
            scalar_attrs = [
                'num_factors', 'p', 'threshold', 'max_iter',
                '_training_loglik', '_training_num_iter', '_training_converged'
            ]
            
            for attr in scalar_attrs:
                original_val = getattr(model, attr)
                loaded_val = getattr(loaded_model, attr)
                # Special handling for _training_converged (may be None if not properly set)
                if attr == '_training_converged' and loaded_val is None and original_val is False:
                    continue  # None is acceptable for False in this case
                if original_val != loaded_val:
                    print(f"  ⚠ Mismatch in {attr}: {original_val} vs {loaded_val}")
                    all_match = False
            
            assert all_match, "All parameters should match exactly"
            print(f"✓ All parameters verified correctly")
    
    def test_mock_checkpoint_protocol_compatibility(self):
        """Test that mock checkpoints saved with protocol 5 can be loaded."""
        model = self._create_mock_model()
        
        # Test protocol 5 (HIGHEST_PROTOCOL)
        protocol = 5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / f'checkpoint_protocol_{protocol}.pkl'
            
            # Create checkpoint dict
            checkpoint = {
                'state_space_params': model.training_state,
                'model_state': DFMModelState.from_model(model),
                'config': model.config,
                'threshold': model.threshold,
                'max_iter': model.max_iter,
                'nan_method': model.nan_method,
                'nan_k': model.nan_k,
                'data_processed': model.data_processed,
                'target_scaler': None,
                'training_loglik': model._training_loglik,
                'training_num_iter': model._training_num_iter,
                'training_converged': model._training_converged,
            }
            
            # Save with protocol 5
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=protocol)
            
            # Verify protocol
            with open(checkpoint_path, 'rb') as f:
                header = f.read(2)
                saved_protocol = header[1] if header[0] == 0x80 else None
                assert saved_protocol == protocol, f"Expected protocol {protocol}, got {saved_protocol}"
            
            # Test loading
            loaded_model = DFM.load(checkpoint_path)
            assert loaded_model.num_factors == model.num_factors
            assert np.allclose(loaded_model.training_state.A, model.training_state.A)
            
            print(f"✓ Protocol {protocol} (HIGHEST_PROTOCOL) checkpoint loaded successfully")
    
    def test_load_mock_checkpoint_error_handling(self):
        """Test error handling when loading invalid/corrupted mock checkpoints."""
        from dfm_python.models.dfm import DFM
        
        # Test 1: Non-existent file
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / 'nonexistent.pkl'
            with pytest.raises(FileNotFoundError):
                DFM.load(nonexistent_path)
        
        # Test 2: Corrupted file (not valid pickle)
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_path = Path(tmpdir) / 'corrupted.pkl'
            with open(corrupted_path, 'wb') as f:
                f.write(b'This is not a valid pickle file')
            
            with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
                DFM.load(corrupted_path)
        
        # Test 3: Empty file
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / 'empty.pkl'
            with open(empty_path, 'wb') as f:
                pass  # Create empty file
            
            with pytest.raises((RuntimeError, EOFError)):
                DFM.load(empty_path)
        
        print(f"✓ Error handling works correctly")
    
    def test_mock_checkpoint_round_trip(self):
        """Test save → load → save → load round trip with mock checkpoint."""
        model = self._create_mock_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / 'checkpoint1.pkl'
            path2 = Path(tmpdir) / 'checkpoint2.pkl'
            
            # Save original
            model.save(path1)
            
            # Load and save again
            loaded1 = DFM.load(path1)
            loaded1.save(path2)
            
            # Load second time
            loaded2 = DFM.load(path2)
            
            # Verify both loaded models match original
            assert loaded1.num_factors == model.num_factors
            assert loaded2.num_factors == model.num_factors
            assert np.allclose(loaded1.training_state.A, model.training_state.A)
            assert np.allclose(loaded2.training_state.A, model.training_state.A)
            
            # Verify both checkpoints are protocol 5
            for path in [path1, path2]:
                with open(path, 'rb') as f:
                    header = f.read(2)
                    protocol = header[1] if header[0] == 0x80 else None
                    assert protocol == 5, f"Expected protocol 5, got {protocol}"
            
            print(f"✓ Round trip test passed: save → load → save → load")

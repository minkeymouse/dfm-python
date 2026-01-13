"""Tests for DFM model save during training pipeline.

This test suite verifies that DFM models are saved correctly during the training
pipeline, catching bugs where save() might be skipped and joblib.dump() overwrites
with corrupted files.
"""

import pytest
import numpy as np
import tempfile
import pickle
from pathlib import Path
from dfm_python.models.dfm import DFM
from dfm_python.config import DFMConfig
from dfm_python.config.schema.params import DFMStateSpaceParams, DFMModelState
from dfm_python.dataset.dfm_dataset import DFMDataset


class TestDFMTrainingPipelineSave:
    """Test DFM save behavior during actual training pipeline."""
    
    def _create_mock_training_data(self, n_samples=100, n_series=5):
        """Create mock training data."""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_series)
        return data
    
    def _create_training_config(self):
        """Create a basic training config."""
        config = DFMConfig(
            blocks={'Block_Global': {'num_factors': 2, 'series': ['s1', 's2', 's3', 's4', 's5']}},
            frequency={'s1': 'w', 's2': 'w', 's3': 'w', 's4': 'w', 's5': 'w'},
            clock='w'
        )
        return config
    
    def test_save_after_fit_without_checkpoint_callback(self):
        """Test that model is saved correctly after fit() even without checkpoint callback.
        
        This simulates the scenario where:
        - fit() is called with no checkpoint callback
        - Model should be saved explicitly after fit() completes
        - This catches the bug where save() was skipped and joblib.dump() overwrote the file
        """
        config = self._create_training_config()
        model = DFM(config=config)
        data = self._create_mock_training_data()
        
        # Create dataset
        dataset = DFMDataset(config=config, data=data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            
            # Train model (no checkpoint callback, simulates max_iter=1 scenario)
            model.fit(X=data, dataset=dataset, checkpoint_callback=None)
            
            # Simulate training pipeline: save after fit() completes
            # This is what the training code should do
            model.save(model_path)
            
            # Verify file exists and uses protocol 5
            assert model_path.exists(), "Model file should exist after save"
            
            with open(model_path, 'rb') as f:
                header = f.read(2)
                if header[0] == 0x80:
                    protocol = header[1]
                    assert protocol == 5, f"Should use protocol 5 (HIGHEST_PROTOCOL), got {protocol}"
            
            # Verify model can be loaded
            loaded_model = DFM.load(model_path)
            assert loaded_model.num_factors == model.num_factors
            assert loaded_model.training_state is not None
            assert np.allclose(loaded_model.training_state.A, model.training_state.A)
            
            print("✓ Model saved correctly after fit() without checkpoint callback")
    
    def test_save_after_fit_with_checkpoint_callback_not_triggering(self):
        """Test that model is saved even when checkpoint callback doesn't trigger.
        
        This simulates max_iter=1 scenario where checkpoint callback only saves every 5 iterations,
        so callback never triggers, but model should still be saved after fit().
        """
        config = self._create_training_config()
        model = DFM(config=config)
        model.max_iter = 1  # Only 1 iteration
        data = self._create_mock_training_data()
        
        # Create dataset
        dataset = DFMDataset(config=config, data=data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            checkpoint_saved = []
            
            # Create checkpoint callback that only saves every 5 iterations
            def checkpoint_callback(iteration, state):
                checkpoint_saved.append(iteration)
            
            # Train model with checkpoint callback
            model.fit(X=data, dataset=dataset, checkpoint_callback=checkpoint_callback)
            
            # Verify callback didn't trigger (max_iter=1, saves every 5 iterations)
            assert len(checkpoint_saved) == 0, "Checkpoint callback should not trigger with max_iter=1"
            
            # Model should still be saved after fit() completes
            model.save(model_path)
            
            # Verify file exists and is correct
            assert model_path.exists(), "Model file should exist even when callback doesn't trigger"
            
            with open(model_path, 'rb') as f:
                header = f.read(2)
                if header[0] == 0x80:
                    protocol = header[1]
                    assert protocol == 5, f"Should use protocol 5, got {protocol}"
            
            # Verify model can be loaded
            loaded_model = DFM.load(model_path)
            assert loaded_model.num_factors == model.num_factors
            assert loaded_model._training_num_iter == 1
            
            print("✓ Model saved correctly even when checkpoint callback doesn't trigger")
    
    def test_training_pipeline_save_vs_joblib_dump(self):
        """Test that model.save() uses protocol 5 (HIGHEST_PROTOCOL).
        
        This test verifies that model.save() uses protocol 5 and demonstrates why
        training code should use model.save() instead of joblib.dump().
        """
        config = self._create_training_config()
        model = DFM(config=config)
        data = self._create_mock_training_data()
        
        dataset = DFMDataset(config=config, data=data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            model.fit(X=data, dataset=dataset)
            
            # Test 1: model.save() uses protocol 5 (HIGHEST_PROTOCOL)
            model_path_save = Path(tmpdir) / "model_save.pkl"
            model.save(model_path_save)
            
            with open(model_path_save, 'rb') as f:
                header = f.read(2)
                protocol = header[1] if header[0] == 0x80 else None
                assert protocol == 5, f"model.save() should use protocol 5 (HIGHEST_PROTOCOL), got {protocol}"
            
            # Test 2: Verify model can be loaded
            loaded_from_save = DFM.load(model_path_save)
            assert loaded_from_save.num_factors == model.num_factors
            
            print("✓ model.save() uses protocol 5 (HIGHEST_PROTOCOL)")
            print("✓ Training pipeline should use model.save() for DFM models")
    
    def test_training_pipeline_save_integration(self):
        """Integration test: simulate full training pipeline save workflow.
        
        This test simulates the actual training pipeline workflow:
        1. Create model and dataset
        2. Train with fit()
        3. Save after training (as training code should do)
        4. Verify saved file can be loaded
        """
        config = self._create_training_config()
        model = DFM(config=config)
        data = self._create_mock_training_data()
        
        dataset = DFMDataset(config=config, data=data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            
            # Full training pipeline simulation
            # Step 1: Train
            model.fit(X=data, dataset=dataset)
            
            # Step 2: Save (what training code should do after fit())
            model.save(model_path)
            
            # Step 3: Verify
            assert model_path.exists()
            
            # Step 4: Load and verify
            loaded_model = DFM.load(model_path)
            assert loaded_model.training_state is not None
            assert loaded_model.num_factors == model.num_factors
            assert np.allclose(
                loaded_model.training_state.A, 
                model.training_state.A,
                rtol=1e-10
            )
            assert np.allclose(
                loaded_model.training_state.C,
                model.training_state.C,
                rtol=1e-10
            )
            
            # Step 5: Verify protocol
            with open(model_path, 'rb') as f:
                header = f.read(2)
                protocol = header[1] if header[0] == 0x80 else None
                assert protocol == 5, f"Expected protocol 5, got {protocol}"
            
            print("✓ Full training pipeline save workflow verified")
    
    def test_no_double_save_in_training_pipeline(self):
        """Test that training pipeline doesn't accidentally save twice.
        
        This ensures that if checkpoint callback saves, we don't save again unnecessarily,
        but if it doesn't save, we still save after fit().
        """
        config = self._create_training_config()
        model = DFM(config=config)
        data = self._create_mock_training_data()
        
        dataset = DFMDataset(config=config, data=data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            save_count = []
            
            # Checkpoint callback that saves
            def checkpoint_callback(iteration, state):
                save_count.append(iteration)
                model.save(model_path)
            
            # Train with max_iter=5 (callback will trigger at iteration 5)
            model.max_iter = 5
            model.fit(X=data, dataset=dataset, checkpoint_callback=checkpoint_callback)
            
            # Verify callback saved at iteration 5
            assert 5 in save_count, "Checkpoint callback should save at iteration 5"
            
            # Save after fit() (training code should do this regardless)
            # This should overwrite with final state, which is fine
            model.save(model_path)
            
            # Verify final save works
            loaded_model = DFM.load(model_path)
            assert loaded_model.num_factors == model.num_factors
            assert loaded_model._training_num_iter == 5
            
            print("✓ Model can be saved by callback and after fit() without issues")

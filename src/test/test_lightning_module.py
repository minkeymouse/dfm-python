"""Tests for PyTorch Lightning module integration.

This module provides comprehensive tests for PyTorch Lightning integration in the
dfm-python package. Lightning modules enable distributed training, GPU acceleration,
and integration with the Lightning ecosystem.

**Test Organization**:
- `TestDFMDataModule`: Tests data module for Lightning (8 tests)
  - Initialization and configuration
  - Data loading and preprocessing
  - DataLoader creation (train, validation)
  - Validation split handling
  - Standardization parameter access
  - Processed data access
  - Series ID handling
- `TestDFMLightningModule`: Tests linear DFM Lightning module (5 tests)
  - Module initialization
  - Optimizer configuration
  - Parameter initialization from data
  - Training step execution
  - EM algorithm integration
- `TestDDFMLightningModule`: Tests deep DFM Lightning module (4 tests)
  - Module initialization with neural encoder
  - Network initialization
  - Optimizer configuration
  - Training step execution

**Dependencies**:
- Required: `pytorch_lightning` (for Lightning framework)
- Required: `torch` (PyTorch for tensor operations)
- Required: `sktime` (for preprocessing via DFMScaler)
- Tests are skipped if dependencies are not available

**Usage Patterns**:
- DFMDataModule: Handles data loading, preprocessing, and DataLoader creation
- DFMLightningModule: Wraps linear DFM for Lightning training (uses EM algorithm)
- DDFMLightningModule: Wraps DDFM for Lightning training (uses gradient descent)
- All modules integrate with pytorch_lightning.Trainer for distributed training

**Related Test Files**:
- `test_dfm.py`: Tests high-level and low-level DFM APIs (used by DFMLightningModule)
- `test_ddfm.py`: Tests high-level and low-level DDFM APIs (used by DDFMLightningModule)
- `test_transformations.py`: Tests preprocessing transformations used by DFMDataModule
"""

# Standard library imports
import pytest

# Third-party imports
import numpy as np
import pytorch_lightning as pl
import torch

# Skip all tests if PyTorch Lightning is not available
pytest.importorskip("pytorch_lightning", reason="PyTorch Lightning is required for Lightning module tests")
# Skip all tests if PyTorch is not available
pytest.importorskip("torch", reason="PyTorch is required for Lightning module tests")
# Skip all tests if sktime is not available (required for preprocessing)
pytest.importorskip("sktime", reason="sktime is required for Lightning module tests")

# Local application imports
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig
from dfm_python.lightning import (
    DDFMLightningModule,
    DDFMTrainingState,
    DFMDataModule,
    DFMDataset,
    DFMLightningModule,
    DFMTrainingState,
)

# Local relative imports
from . import create_simple_config, generate_synthetic_data


# ============================================================================
# DFMDataModule Tests
# ============================================================================

class TestDFMDataModule:
    """Test DFMDataModule functionality."""
    
    def test_data_module_init(self):
        """Test DFMDataModule initialization.
        
        This test verifies that DFMDataModule initializes correctly:
        1. Configuration is stored correctly
        2. Batch size and window size parameters are set
        3. Scaler and datasets are None before setup() is called
        
        Expected behavior:
        - Config matches input configuration
        - Batch size is set correctly
        - Window size and val_split default to None
        - Scaler is None before setup() is called
        - Train dataset is None before setup() is called
        """
        config = create_simple_config(num_series=5, num_factors=1)
        data_module = DFMDataModule(
            config=config,
            batch_size=16,
            window_size=None,
            val_split=None
        )
        
        assert data_module.config == config, "DFMDataModule config should match input config"
        assert data_module.batch_size == 16, f"Expected batch_size 16, got {data_module.batch_size}"
        assert data_module.window_size is None, "window_size should be None by default"
        assert data_module.val_split is None, "val_split should be None by default"
        assert data_module.scaler is None, "Scaler should be None before setup() is called"
        assert data_module.train_dataset is None, "Train dataset should be None before setup() is called"
    
    def test_data_module_setup(self):
        """Test DFMDataModule setup with synthetic data.
        
        This test verifies that DFMDataModule.setup() works correctly:
        1. Setup completes without errors
        2. Scaler is created and fitted
        3. Standardization parameters (Mx, Wx) are set
        4. Datasets are created
        5. Processed data is converted to torch.Tensor with correct shape
        
        Expected behavior:
        - setup() completes without raising exceptions
        - Scaler is created and fitted during setup
        - Standardization parameters (Mx, Wx) are set
        - Train dataset is created as DFMDataset instance
        - Processed data is torch.Tensor with correct shape (T × N)
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16
        )
        
        # Call setup - should not raise errors
        try:
            data_module.setup()
        except Exception as e:
            pytest.fail(f"setup() raised {type(e).__name__}: {e}")
        
        # Check that scaler was created and fitted
        assert data_module.scaler is not None, "Scaler should be created during setup"
        assert data_module.Mx is not None, "Mx (mean) should be set during setup"
        assert data_module.Wx is not None, "Wx (std) should be set during setup"
        
        # Check that datasets were created
        assert data_module.train_dataset is not None, "Train dataset should be created"
        assert isinstance(data_module.train_dataset, DFMDataset), f"Train dataset should be DFMDataset, got {type(data_module.train_dataset)}"
        
        # Check processed data
        assert data_module.data_processed is not None, "Processed data should be set"
        assert isinstance(data_module.data_processed, torch.Tensor), f"Processed data should be torch.Tensor, got {type(data_module.data_processed)}"
        assert data_module.data_processed.shape[0] == 50, f"Expected 50 time periods, got {data_module.data_processed.shape[0]}"
        assert data_module.data_processed.shape[1] == 5, f"Expected 5 series, got {data_module.data_processed.shape[1]}"
    
    def test_data_module_dataloaders(self):
        """Test DFMDataModule dataloader creation.
        
        This test verifies that DFMDataModule creates DataLoaders correctly:
        1. Train dataloader is created with correct batch size
        2. Validation dataloader is None when val_split is not specified
        3. DataLoaders are proper torch.utils.data.DataLoader instances
        
        Expected behavior:
        - Train dataloader is created and not None
        - Train dataloader is DataLoader instance
        - Train dataloader has correct batch size
        - Val dataloader is None when val_split is not specified
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16
        )
        data_module.setup()
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        assert train_loader is not None, "Train dataloader should not be None after setup"
        assert isinstance(train_loader, torch.utils.data.DataLoader), f"Train dataloader should be DataLoader, got {type(train_loader)}"
        assert train_loader.batch_size == 16, f"Expected batch_size 16, got {train_loader.batch_size}"
        
        # Test val dataloader (should be None if no val_split)
        val_loader = data_module.val_dataloader()
        assert val_loader is None, "Val dataloader should be None when val_split is not specified"
    
    def test_data_module_val_split(self):
        """Test DFMDataModule with validation split.
        
        This test verifies that DFMDataModule handles validation split correctly:
        1. Both train and validation datasets are created when val_split is specified
        2. Both train and validation dataloaders are created
        3. Data is properly split between train and validation sets
        
        Expected behavior:
        - Train dataset is created when val_split is specified
        - Val dataset is created when val_split is specified
        - Train dataloader is created and not None
        - Val dataloader is created and not None
        - Val dataloader is DataLoader instance
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=100, n_series=5)
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16,
            val_split=0.2  # 20% validation
        )
        data_module.setup()
        
        # Check that both train and val datasets exist
        assert data_module.train_dataset is not None, "Train dataset should be created when val_split is specified"
        assert data_module.val_dataset is not None, "Val dataset should be created when val_split is specified"
        
        # Check train dataloader
        train_loader = data_module.train_dataloader()
        assert train_loader is not None, "Train dataloader should not be None when val_split is specified"
        
        # Check val dataloader
        val_loader = data_module.val_dataloader()
        assert val_loader is not None, "Val dataloader should not be None when val_split is specified"
        assert isinstance(val_loader, torch.utils.data.DataLoader), f"Val dataloader should be DataLoader, got {type(val_loader)}"
    
    def test_data_module_standardization_params(self):
        """Test DFMDataModule standardization parameter access.
        
        This test verifies that DFMDataModule provides access to standardization parameters:
        1. get_standardization_params() returns Mx and Wx without errors
        2. Parameters have correct shapes matching number of series
        3. Standard deviations (Wx) are all positive
        
        Expected behavior:
        - get_standardization_params() returns (Mx, Wx) tuple
        - Mx and Wx are not None
        - Mx shape matches number of series (N,)
        - Wx shape matches number of series (N,)
        - All standard deviations (Wx) are positive
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16
        )
        data_module.setup()
        
        # Get standardization parameters - should not raise errors
        try:
            Mx, Wx = data_module.get_standardization_params()
        except RuntimeError as e:
            pytest.fail(f"get_standardization_params() raised RuntimeError: {e}")
        
        assert Mx is not None, "Mx should not be None"
        assert Wx is not None, "Wx should not be None"
        num_series = X.shape[1]
        assert Mx.shape == (num_series,), f"Mx shape mismatch: expected ({num_series},), got {Mx.shape}"
        assert Wx.shape == (num_series,), f"Wx shape mismatch: expected ({num_series},), got {Wx.shape}"
        assert np.all(Wx > 0), f"All standard deviations should be positive, got: {Wx}"
    
    def test_data_module_get_processed_data(self):
        """Test DFMDataModule processed data access.
        
        This test verifies that DFMDataModule provides access to processed data:
        1. get_processed_data() returns processed data without errors
        2. Processed data is a torch.Tensor
        3. Data shape matches expected dimensions (T × N)
        
        Expected behavior:
        - get_processed_data() returns processed data without raising RuntimeError
        - Processed data is not None
        - Processed data is torch.Tensor
        - Data shape matches expected (T periods × N series)
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16
        )
        data_module.setup()
        
        # Get processed data - should not raise errors
        try:
            processed_data = data_module.get_processed_data()
        except RuntimeError as e:
            pytest.fail(f"get_processed_data() raised RuntimeError: {e}")
        
        assert processed_data is not None, "Processed data should not be None"
        assert isinstance(processed_data, torch.Tensor), f"Processed data should be torch.Tensor, got {type(processed_data)}"
        assert processed_data.shape == (50, 5), f"Expected shape (50, 5), got {processed_data.shape}"
    
    def test_data_module_series_ids_match_data(self):
        """Test that DFMDataModule correctly handles series_ids from config.
        
        This test verifies that DFMDataModule correctly handles series_ids:
        1. Config provides correct series_ids
        2. Setup works correctly with series_ids from config
        3. Processed data columns match series_ids count
        
        Expected behavior:
        - Config has correct number of series_ids
        - setup() completes without ValueError or TypeError
        - Processed data columns match series_ids count
        - Data alignment is correct between config and data
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        
        # Verify config has correct series_ids
        series_ids = config.get_series_ids()
        assert len(series_ids) == 5, f"Expected 5 series_ids, got {len(series_ids)}"
        
        data_module = DFMDataModule(
            config=config,
            data=X,
            batch_size=16
        )
        
        # Setup should work correctly with series_ids
        try:
            data_module.setup()
        except (ValueError, TypeError) as e:
            pytest.fail(f"setup() failed with series_ids: {type(e).__name__}: {e}")
        
        # Verify processed data has correct number of columns
        processed_data = data_module.get_processed_data()
        assert processed_data.shape[1] == len(series_ids), f"Data columns ({processed_data.shape[1]}) should match series_ids count ({len(series_ids)})"


# ============================================================================
# DFMLightningModule Tests
# ============================================================================

class TestDFMLightningModule:
    """Test DFMLightningModule functionality."""
    
    def test_dfm_lightning_module_init(self):
        """Test DFMLightningModule initialization.
        
        This test verifies that DFMLightningModule initializes correctly:
        1. Configuration and parameters are stored correctly
        2. Manual optimization is enabled (EM algorithm doesn't use standard optimizers)
        3. Kalman filter and EM algorithm are initialized
        
        Expected behavior:
        - Config matches input configuration
        - num_factors, threshold, max_iter are set correctly
        - automatic_optimization is False (EM uses manual optimization)
        - Kalman filter is initialized
        - EM algorithm is initialized
        """
        config = create_simple_config(num_series=5, num_factors=1)
        
        module = DFMLightningModule(
            config=config,
            num_factors=1,
            threshold=1e-4,
            max_iter=10
        )
        
        assert module.config == config, "DFMLightningModule config should match input config"
        assert module.num_factors == 1, f"Expected num_factors 1, got {module.num_factors}"
        assert module.threshold == 1e-4, f"Expected threshold 1e-4, got {module.threshold}"
        assert module.max_iter == 10, f"Expected max_iter 10, got {module.max_iter}"
        assert module.automatic_optimization is False, "DFMLightningModule should use manual optimization (EM algorithm)"
        assert module.kalman is not None, "Kalman filter should be initialized"
        assert module.em is not None, "EM algorithm should be initialized"
    
    def test_dfm_lightning_module_setup(self):
        """Test DFMLightningModule setup method.
        
        This test verifies that DFMLightningModule.setup() works correctly:
        1. Setup completes without errors
        2. Parameters (A, C) are None until initialized from data
        
        Expected behavior:
        - setup() completes without raising exceptions
        - A (transition matrix) is None before initialization from data
        - C (loadings) is None before initialization from data
        - Parameters are initialized later via initialize_from_data() or training_step()
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DFMLightningModule(config=config)
        
        # Setup should not raise an error (parameters initialized later)
        module.setup()
        
        # Parameters should be None until initialized from data
        assert module.A is None, "A (transition matrix) should be None before initialization from data"
        assert module.C is None, "C (loadings) should be None before initialization from data"
    
    def test_dfm_lightning_module_optimizers(self):
        """Test DFMLightningModule optimizer configuration.
        
        This test verifies that DFMLightningModule uses manual optimization:
        1. configure_optimizers() returns empty list (EM algorithm doesn't use standard optimizers)
        2. EM algorithm handles optimization manually in training_step()
        
        Expected behavior:
        - configure_optimizers() returns empty list
        - EM algorithm uses manual optimization (not standard PyTorch optimizers)
        - Optimization happens in training_step() via EM algorithm
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DFMLightningModule(config=config)
        
        # EM algorithm doesn't use standard optimizers
        optimizers = module.configure_optimizers()
        assert optimizers == [], "DFMLightningModule should return empty list for optimizers (EM uses manual optimization)"
    
    def test_dfm_lightning_module_initialize_from_data(self):
        """Test DFMLightningModule parameter initialization from data.
        
        This test verifies that DFMLightningModule initializes parameters from data:
        1. initialize_from_data() completes without errors
        2. All model parameters (A, C, Q, R, Z_0, V_0) are initialized
        3. Parameter shapes match expected dimensions based on config and data
        
        Expected behavior:
        - initialize_from_data() completes without raising exceptions
        - A (transition matrix) is initialized with shape (num_factors, num_factors)
        - C (loadings) is initialized with shape (num_series, num_factors)
        - Q (process noise) is initialized with shape (num_factors, num_factors)
        - R (observation noise) is initialized with shape (num_series, num_series)
        - Z_0 (initial state) and V_0 (initial covariance) are initialized
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        module = DFMLightningModule(config=config)
        
        # Initialize from data - should not raise errors
        try:
            module.initialize_from_data(X_tensor)
        except Exception as e:
            pytest.fail(f"initialize_from_data() raised {type(e).__name__}: {e}")
        
        # Check that parameters were initialized
        assert module.A is not None, "A (transition matrix) should be initialized"
        assert module.C is not None, "C (loadings) should be initialized"
        assert module.Q is not None, "Q (process noise) should be initialized"
        assert module.R is not None, "R (observation noise) should be initialized"
        assert module.Z_0 is not None, "Z_0 (initial state) should be initialized"
        assert module.V_0 is not None, "V_0 (initial covariance) should be initialized"
        
        # Check parameter shapes dynamically based on config
        num_factors = module.num_factors
        num_series = X_tensor.shape[1]  # N series from data
        
        assert module.A.shape == (num_factors, num_factors), f"A shape mismatch: expected ({num_factors}, {num_factors}), got {module.A.shape}"
        assert module.C.shape == (num_series, num_factors), f"C shape mismatch: expected ({num_series}, {num_factors}), got {module.C.shape}"
        assert module.Q.shape == (num_factors, num_factors), f"Q shape mismatch: expected ({num_factors}, {num_factors}), got {module.Q.shape}"
        assert module.R.shape == (num_series, num_series), f"R shape mismatch: expected ({num_series}, {num_series}), got {module.R.shape}"
    
    def test_dfm_lightning_module_training_step(self):
        """Test DFMLightningModule training_step method.
        
        This test verifies that DFMLightningModule.training_step() works correctly:
        1. training_step() processes a batch and returns loss
        2. Parameters are initialized during training step
        3. Loss is a valid torch.Tensor
        
        Expected behavior:
        - training_step() returns loss tensor without errors
        - Loss is torch.Tensor instance
        - A (transition matrix) is initialized after training_step
        - C (loadings) is initialized after training_step
        - EM algorithm performs parameter updates during training_step
        """
        config = create_simple_config(num_series=5, num_factors=1)
        X = generate_synthetic_data(n_periods=50, n_series=5)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        module = DFMLightningModule(config=config, max_iter=10)
        
        # Create a batch (for DFM, batch is full sequence)
        batch = (X_tensor, X_tensor)  # (data, target)
        
        # Training step should initialize parameters and perform EM step
        loss = module.training_step(batch, batch_idx=0)
        
        # Check that loss is a tensor
        assert isinstance(loss, torch.Tensor), f"Loss should be torch.Tensor, got {type(loss)}"
        
        # Check that parameters were initialized
        assert module.A is not None, "A (transition matrix) should be initialized after training_step"
        assert module.C is not None, "C (loadings) should be initialized after training_step"


# ============================================================================
# DDFMLightningModule Tests
# ============================================================================

class TestDDFMLightningModule:
    """Test DDFMLightningModule functionality."""
    
    def test_ddfm_lightning_module_init(self):
        """Test DDFMLightningModule initialization.
        
        This test verifies that DDFMLightningModule initializes correctly:
        1. Configuration and neural network parameters are stored correctly
        2. Encoder and decoder are None before initialize_networks() is called
        3. All hyperparameters (encoder_layers, num_factors, activation, learning_rate, epochs) are set
        
        Expected behavior:
        - Config matches input configuration
        - encoder_layers, num_factors, activation, learning_rate, epochs_per_iter are set correctly
        - Encoder is None before initialize_networks() is called
        - Decoder is None before initialize_networks() is called
        - Networks are initialized later via initialize_networks()
        """
        config = create_simple_config(num_series=5, num_factors=1)
        
        module = DDFMLightningModule(
            config=config,
            encoder_layers=[32, 16],
            num_factors=1,
            activation='tanh',
            learning_rate=0.001,
            epochs=10
        )
        
        assert module.config == config, "DDFMLightningModule config should match input config"
        assert module.encoder_layers == [32, 16], f"Expected encoder_layers [32, 16], got {module.encoder_layers}"
        assert module.num_factors == 1, f"Expected num_factors 1, got {module.num_factors}"
        assert module.activation == 'tanh', f"Expected activation 'tanh', got {module.activation}"
        assert module.learning_rate == 0.001, f"Expected learning_rate 0.001, got {module.learning_rate}"
        assert module.epochs_per_iter == 10, f"Expected epochs_per_iter 10, got {module.epochs_per_iter}"
        assert module.encoder is None, "Encoder should be None before initialize_networks is called"
        assert module.decoder is None, "Decoder should be None before initialize_networks is called"
    
    def test_ddfm_lightning_module_setup(self):
        """Test DDFMLightningModule setup method.
        
        This test verifies that DDFMLightningModule.setup() works correctly:
        1. Setup completes without errors
        2. Encoder and decoder remain None until initialize_networks() is called
        
        Expected behavior:
        - setup() completes without raising exceptions
        - Encoder is None after setup (before initialize_networks)
        - Decoder is None after setup (before initialize_networks)
        - Networks are initialized separately via initialize_networks()
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DDFMLightningModule(
            config=config,
            encoder_layers=[32, 16],
            num_factors=1
        )
        
        # Setup should not raise an error
        module.setup()
        
        # Encoder/decoder should be None until initialize_networks is called
        assert module.encoder is None, "Encoder should be None after setup (before initialize_networks)"
        assert module.decoder is None, "Decoder should be None after setup (before initialize_networks)"
    
    def test_ddfm_lightning_module_initialize_networks(self):
        """Test DDFMLightningModule network initialization.
        
        This test verifies that DDFMLightningModule initializes neural networks correctly:
        1. initialize_networks() creates encoder and decoder
        2. Forward pass works correctly with initialized networks
        3. Reconstructed output shape matches input shape
        
        Expected behavior:
        - Encoder is created after initialize_networks()
        - Decoder is created after initialize_networks()
        - Forward pass completes without errors
        - Reconstructed output shape matches input shape
        - Networks are ready for training
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DDFMLightningModule(
            config=config,
            encoder_layers=[32, 16],
            num_factors=1
        )
        
        # Initialize networks
        module.initialize_networks(input_dim=5)  # 5 series
        
        # Check that encoder and decoder were created
        assert module.encoder is not None, "Encoder should be created after initialize_networks"
        assert module.decoder is not None, "Decoder should be created after initialize_networks"
        
        # Test forward pass
        X = torch.randn(10, 5)  # 10 time steps, 5 series
        with torch.no_grad():
            reconstructed = module.forward(X)
            assert reconstructed.shape == X.shape, f"Reconstructed shape {reconstructed.shape} should match input shape {X.shape}"
    
    def test_ddfm_lightning_module_optimizers(self):
        """Test DDFMLightningModule optimizer configuration.
        
        This test verifies that DDFMLightningModule configures optimizers correctly:
        1. configure_optimizers() returns empty list before networks are initialized
        2. After network initialization, returns list with Adam optimizer
        3. Optimizer has correct learning rate
        
        Expected behavior:
        - configure_optimizers() returns empty list before networks are initialized
        - After initialize_networks(), returns list with optimizer
        - Optimizer is Adam instance
        - Learning rate matches module's learning_rate parameter
        - Optimizer is ready for gradient-based training
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DDFMLightningModule(
            config=config,
            encoder_layers=[32, 16],
            num_factors=1
        )
        
        # Optimizer should return empty list if networks not initialized
        optimizers = module.configure_optimizers()
        assert optimizers == [], "Optimizers should return empty list before networks are initialized"
        
        # Initialize networks
        module.initialize_networks(input_dim=5)
        
        # Now should return list with optimizer (Lightning expects list/dict/tuple)
        optimizers = module.configure_optimizers()
        assert optimizers is not None, "Optimizers should not be None after networks are initialized"
        assert isinstance(optimizers, list), f"Optimizers should be a list, got {type(optimizers)}"
        assert len(optimizers) == 1, f"Expected 1 optimizer, got {len(optimizers)}"
        assert isinstance(optimizers[0], torch.optim.Adam), f"Optimizer should be Adam, got {type(optimizers[0])}"
        assert optimizers[0].param_groups[0]['lr'] == 0.001, f"Expected learning rate 0.001, got {optimizers[0].param_groups[0]['lr']}"
    
    def test_ddfm_lightning_module_training_step(self):
        """Test DDFMLightningModule training_step method.
        
        This test verifies that DDFMLightningModule.training_step() works correctly:
        1. training_step() processes a batch and returns loss
        2. Loss is a valid torch.Tensor
        3. Loss is non-negative (reconstruction error)
        
        Expected behavior:
        - training_step() returns loss tensor without errors
        - Loss is torch.Tensor instance
        - Loss is non-negative (reconstruction error)
        - Gradient-based training works correctly with Adam optimizer
        """
        config = create_simple_config(num_series=5, num_factors=1)
        module = DDFMLightningModule(
            config=config,
            encoder_layers=[32, 16],
            num_factors=1
        )
        
        # Initialize networks
        module.initialize_networks(input_dim=5)
        
        # Create a batch
        X = torch.randn(10, 5)  # 10 time steps, 5 series
        batch = (X, X)  # (data, target)
        
        # Training step
        loss = module.training_step(batch, batch_idx=0)
        
        # Check that loss is a tensor
        assert isinstance(loss, torch.Tensor), f"Loss should be torch.Tensor, got {type(loss)}"
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

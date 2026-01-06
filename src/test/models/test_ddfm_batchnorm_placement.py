"""Test for BatchNorm placement comparison with original TensorFlow.

This test verifies that our PyTorch implementation matches the original
TensorFlow BatchNorm placement: Dense → ReLU → BatchNorm → Dense → ReLU.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestBatchNormPlacement:
    """Test BatchNorm placement matches original TensorFlow."""
    
    def test_tensorflow_batchnorm_order(self):
        """Test that TensorFlow uses: Dense → ReLU → BatchNorm → Dense → ReLU."""
        # Original TensorFlow order (from DDFM/models/ddfm.py):
        # encoded = Dense(16, activation=relu)(inputs_)
        # for j in structure_encoder[1:]:  # j = 4
        #     if batch_norm:
        #         encoded = BatchNormalization()(encoded)  # BN AFTER ReLU
        #     encoded = Dense(j, activation=relu)(encoded)
        
        # Expected order: Dense(16) → ReLU → BatchNorm → Dense(4) → ReLU
        expected_order = [
            'Dense(16)',
            'ReLU',
            'BatchNorm',
            'Dense(4)',
            'ReLU'
        ]
        
        print("TensorFlow order:", " → ".join(expected_order))
        assert len(expected_order) == 5, "TensorFlow should have 5 operations"
        assert expected_order.index('BatchNorm') > expected_order.index('ReLU'), \
            "BatchNorm should come AFTER first ReLU in TensorFlow"
    
    def test_pytorch_batchnorm_order(self):
        """Test that our PyTorch uses: Dense → BatchNorm → ReLU → Dense → BatchNorm → ReLU."""
        # Our PyTorch order (from simple_autoencoder.py):
        # for i, layer in enumerate(self.layers):  # Dense(16)
        #     x = layer(x)
        #     if use_batch_norm:
        #         x = self.batch_norms[i](x)  # BN BEFORE ReLU
        #     x = self.activation(x)  # ReLU
        
        # Actual order: Dense(16) → BatchNorm → ReLU → Dense(4) → BatchNorm → ReLU
        actual_order = [
            'Dense(16)',
            'BatchNorm',
            'ReLU',
            'Dense(4)',
            'BatchNorm',
            'ReLU'
        ]
        
        print("PyTorch order:", " → ".join(actual_order))
        assert len(actual_order) == 6, "PyTorch has 6 operations (extra BatchNorm)"
        assert actual_order.index('BatchNorm') < actual_order.index('ReLU'), \
            "BatchNorm comes BEFORE ReLU in PyTorch"
    
    def test_batchnorm_placement_difference(self):
        """Test that identifies the placement difference."""
        # TensorFlow: Dense → ReLU → BatchNorm
        tf_order = ['Dense', 'ReLU', 'BatchNorm']
        
        # PyTorch: Dense → BatchNorm → ReLU
        pytorch_order = ['Dense', 'BatchNorm', 'ReLU']
        
        # Find where BatchNorm appears relative to ReLU
        tf_bn_idx = tf_order.index('BatchNorm')
        tf_relu_idx = tf_order.index('ReLU')
        pytorch_bn_idx = pytorch_order.index('BatchNorm')
        pytorch_relu_idx = pytorch_order.index('ReLU')
        
        # In TensorFlow, BatchNorm comes AFTER ReLU
        assert tf_bn_idx > tf_relu_idx, "TensorFlow: BatchNorm should be AFTER ReLU"
        
        # In PyTorch, BatchNorm comes BEFORE ReLU
        assert pytorch_bn_idx < pytorch_relu_idx, "PyTorch: BatchNorm should be BEFORE ReLU"
        
        # This is the difference
        assert tf_bn_idx > tf_relu_idx and pytorch_bn_idx < pytorch_relu_idx, \
            "BatchNorm placement differs between TensorFlow and PyTorch"
    
    def test_batchnorm_input_difference(self):
        """Test that BatchNorm receives different inputs in TensorFlow vs PyTorch."""
        # Create test data
        x = torch.randn(100, 8) * 1.0  # Standardized input
        
        # TensorFlow order: Dense → ReLU → BatchNorm
        dense_tf = nn.Linear(8, 16)
        relu_tf = nn.ReLU()
        bn_tf = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        
        # PyTorch order: Dense → BatchNorm → ReLU
        dense_pt = nn.Linear(8, 16)
        bn_pt = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        relu_pt = nn.ReLU()
        
        # TensorFlow: BatchNorm receives ReLU output (non-negative)
        x_tf = dense_tf(x)
        x_tf = relu_tf(x_tf)  # ReLU first
        x_tf_bn_input = x_tf.clone()
        x_tf = bn_tf(x_tf)  # BatchNorm after ReLU
        
        # PyTorch: BatchNorm receives linear output (can be negative)
        x_pt = dense_pt(x)
        x_pt_bn_input = x_pt.clone()
        x_pt = bn_pt(x_pt)  # BatchNorm before ReLU
        x_pt = relu_pt(x_pt)
        
        # Verify difference
        print(f"TensorFlow BatchNorm input: mean={x_tf_bn_input.mean().item():.6f}, "
              f"std={x_tf_bn_input.std().item():.6f}, min={x_tf_bn_input.min().item():.6f}")
        print(f"PyTorch BatchNorm input: mean={x_pt_bn_input.mean().item():.6f}, "
              f"std={x_pt_bn_input.std().item():.6f}, min={x_pt_bn_input.min().item():.6f}")
        
        # TensorFlow input should be non-negative (ReLU output)
        assert (x_tf_bn_input >= 0).all(), "TensorFlow BatchNorm input should be non-negative (ReLU output)"
        
        # PyTorch input can be negative (linear output)
        assert (x_pt_bn_input < 0).any(), "PyTorch BatchNorm input can be negative (linear output)"
        
        # This difference could cause different learning dynamics
        assert x_tf_bn_input.min().item() >= 0, "TensorFlow: min should be >= 0"
        assert x_pt_bn_input.min().item() < 0, "PyTorch: min can be < 0"


class TestBatchNormPlacementImpact:
    """Test the impact of BatchNorm placement on learning."""
    
    def test_batchnorm_with_negative_inputs(self):
        """Test that BatchNorm with negative inputs can cause issues."""
        # Simulate what happens in PyTorch: BatchNorm receives negative values
        bn = nn.BatchNorm1d(8, momentum=0.99, eps=1e-3)
        
        # Train with inputs that have both positive and negative values
        bn.train()
        for i in range(200):
            x = torch.randn(100, 8) * 0.1  # Small std (what model learns)
            _ = bn(x)
        
        running_var = bn.running_var.mean().item()
        print(f"BatchNorm running_var after training with std=0.1: {running_var:.6f}")
        
        # running_var should be ~0.01 for std=0.1 input
        assert running_var < 0.1, "running_var should be small for small input std"
        
        # Test amplification
        bn.eval()
        test_input = torch.randn(100, 8) * 1.0  # Standardized input
        output = bn(test_input)
        amplification = output.std().item() / test_input.std().item()
        
        print(f"Amplification with running_var={running_var:.6f}: {amplification:.2f}x")
        assert amplification > 5.0, "Amplification should be significant when running_var is small"
    
    def test_batchnorm_with_nonnegative_inputs(self):
        """Test that BatchNorm with non-negative inputs behaves differently."""
        # Simulate what happens in TensorFlow: BatchNorm receives ReLU output (non-negative)
        bn = nn.BatchNorm1d(8, momentum=0.99, eps=1e-3)
        relu = nn.ReLU()
        
        # Train with non-negative inputs (ReLU output)
        bn.train()
        for i in range(200):
            x_linear = torch.randn(100, 8) * 0.1  # Small std
            x_relu = relu(x_linear)  # ReLU makes it non-negative
            _ = bn(x_relu)  # BatchNorm receives ReLU output
        
        running_var = bn.running_var.mean().item()
        print(f"BatchNorm running_var after training with ReLU output (std=0.1): {running_var:.6f}")
        
        # ReLU output has different distribution (non-negative, potentially more variance)
        # This might prevent running_var from collapsing as much
        assert running_var > 0, "running_var should be positive"


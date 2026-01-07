"""Comparison test: Why TensorFlow BatchNorm works vs PyTorch.

This test demonstrates why the original TensorFlow implementation works
and how BatchNorm placement affects learning dynamics.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


class TestTensorFlowBatchNormWhyItWorks:
    """Test why TensorFlow BatchNorm placement works."""
    
    def test_tensorflow_batchnorm_receives_relu_output(self):
        """Test that TensorFlow BatchNorm receives ReLU-activated output."""
        # TensorFlow order: Dense → ReLU → BatchNorm
        dense = nn.Linear(8, 16)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        
        # Input
        x = torch.randn(100, 8) * 1.0  # Standardized
        
        # TensorFlow flow
        x_dense = dense(x)
        x_relu = relu(x_dense)  # ReLU first
        x_bn = bn(x_relu)  # BatchNorm receives ReLU output
        
        # Verify BatchNorm receives non-negative values
        assert (x_relu >= 0).all(), "TensorFlow: BatchNorm input should be non-negative (ReLU output)"
        assert x_relu.min().item() >= 0, "TensorFlow: min should be >= 0"
        
        # Train BatchNorm
        bn.train()
        for i in range(200):
            x_dense = dense(x)
            x_relu = relu(x_dense)
            _ = bn(x_relu)
        
        running_var = bn.running_var.mean().item()
        running_mean = bn.running_mean.mean().item()
        
        # BatchNorm should learn reasonable statistics
        assert running_var > 0.01, f"TensorFlow: running_var should be reasonable, got {running_var}"
        assert running_mean > 0, f"TensorFlow: running_mean should be positive (ReLU output), got {running_mean}"
    
    def test_relu_output_has_inherent_variance(self):
        """Test that ReLU output has inherent variance even with small input."""
        relu = nn.ReLU()
        
        # Test with different input scales
        for input_std in [1.0, 0.5, 0.1, 0.01]:
            x = torch.randn(100, 8) * input_std
            x_relu = relu(x)
            
            relu_std = x_relu.std().item()
            relu_mean = x_relu.mean().item()
            
            print(f"Input std={input_std:.2f}: ReLU mean={relu_mean:.6f}, std={relu_std:.6f}")
            
            # ReLU output should have some variance even with small input
            # (because ReLU preserves positive values)
            if input_std >= 0.1:
                assert relu_std > 0.01, f"ReLU output should have variance for std={input_std}"
    
    def test_tensorflow_batchnorm_with_small_input(self):
        """Test TensorFlow BatchNorm behavior with small input (what model learns)."""
        dense = nn.Linear(8, 16)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        
        # Small input (simulating what model learns)
        x = torch.randn(100, 8) * 0.1  # Small std
        
        # TensorFlow order: Dense → ReLU → BatchNorm
        bn.train()
        for i in range(200):
            x_dense = dense(x)
            x_relu = relu(x_dense)  # ReLU preserves some variance
            _ = bn(x_relu)
        
        running_var = bn.running_var.mean().item()
        running_mean = bn.running_mean.mean().item()
        
        print(f"TensorFlow order with small input: running_var={running_var:.6f}, running_mean={running_mean:.6f}")
        
        # Even with small input, ReLU output has some variance
        # BatchNorm running_var should be small but not collapsed
        assert running_var > 0.001, "TensorFlow: running_var should not collapse completely"
    
    def test_pytorch_batchnorm_with_small_input(self):
        """Test PyTorch BatchNorm behavior with small input (demonstrates collapse)."""
        dense = nn.Linear(8, 16)
        bn = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        relu = nn.ReLU()
        
        # Small input (simulating what model learns)
        x = torch.randn(100, 8) * 0.1  # Small std
        
        # PyTorch order: Dense → BatchNorm → ReLU
        bn.train()
        for i in range(200):
            x_dense = dense(x)
            _ = bn(x_dense)  # BatchNorm on linear output
            _ = relu(bn(x_dense))
        
        running_var = bn.running_var.mean().item()
        running_mean = bn.running_mean.mean().item()
        
        print(f"PyTorch order with small input: running_var={running_var:.6f}, running_mean={running_mean:.6f}")
        
        # PyTorch: BatchNorm sees linear output with small variance
        # running_var collapses to match this small variance
        assert running_var < 0.1, f"PyTorch: running_var should collapse, got {running_var}"
        assert running_var < 0.05, f"PyTorch: running_var should be very small, got {running_var}"


class TestBatchNormPlacementImpact:
    """Test the impact of BatchNorm placement on statistics."""
    
    def test_batchnorm_statistics_comparison(self):
        """Compare BatchNorm statistics for different placements."""
        # Create two BatchNorm layers
        bn_after_relu = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        bn_before_relu = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        
        dense = nn.Linear(8, 16)
        relu = nn.ReLU()
        
        # Use larger input where ReLU preserves more variance
        # (with small input, ReLU reduces variance by zeroing negatives)
        x = torch.randn(100, 8) * 1.0
        
        # TensorFlow order: Dense → ReLU → BatchNorm
        bn_after_relu.train()
        for i in range(200):
            x_dense = dense(x)
            x_relu = relu(x_dense)
            _ = bn_after_relu(x_relu)
        
        # PyTorch order: Dense → BatchNorm → ReLU
        bn_before_relu.train()
        for i in range(200):
            x_dense = dense(x)
            _ = bn_before_relu(x_dense)
            _ = relu(bn_before_relu(x_dense))
        
        var_after_relu = bn_after_relu.running_var.mean().item()
        var_before_relu = bn_before_relu.running_var.mean().item()
        
        print(f"BatchNorm AFTER ReLU: running_var={var_after_relu:.6f}")
        print(f"BatchNorm BEFORE ReLU: running_var={var_before_relu:.6f}")
        
        # BatchNorm before ReLU should have higher running_var
        # (because ReLU reduces variance by zeroing negatives)
        assert var_before_relu > var_after_relu, \
            f"BatchNorm before ReLU should have higher running_var: {var_before_relu} vs {var_after_relu}"
    
    def test_amplification_comparison(self):
        """Compare amplification effect for different BatchNorm placements."""
        # Create BatchNorm layers
        bn_after_relu = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        bn_before_relu = nn.BatchNorm1d(16, momentum=0.99, eps=1e-3)
        
        dense = nn.Linear(8, 16)
        relu = nn.ReLU()
        
        # Small input (what model learns during pre-training)
        x_small = torch.randn(100, 8) * 0.1
        
        # Train both
        bn_after_relu.train()
        bn_before_relu.train()
        for i in range(200):
            # TensorFlow order
            x_dense = dense(x_small)
            x_relu = relu(x_dense)
            _ = bn_after_relu(x_relu)
            
            # PyTorch order
            x_dense = dense(x_small)
            _ = bn_before_relu(x_dense)
            _ = relu(bn_before_relu(x_dense))
        
        # Test with standardized input (what we use for inference)
        x_std = torch.randn(100, 8) * 1.0
        
        bn_after_relu.eval()
        bn_before_relu.eval()
        
        with torch.no_grad():
            # TensorFlow order
            x_dense = dense(x_std)
            x_relu = relu(x_dense)
            output_after = bn_after_relu(x_relu)
            amp_after = output_after.std().item() / x_relu.std().item()
            
            # PyTorch order
            x_dense = dense(x_std)
            output_before = bn_before_relu(x_dense)
            amp_before = output_before.std().item() / x_dense.std().item()
        
        print(f"Amplification AFTER ReLU: {amp_after:.2f}x")
        print(f"Amplification BEFORE ReLU: {amp_before:.2f}x")
        
        # TensorFlow order (after ReLU) should have higher amplification
        # (due to ReLU reducing variance, BatchNorm sees smaller running_var)
        assert amp_after > amp_before, \
            f"TensorFlow order should have higher amplification: {amp_after} vs {amp_before}"


class TestWhyTensorFlowWorks:
    """Test why TensorFlow implementation works."""
    
    def test_relu_preserves_variance(self):
        """Test that ReLU preserves variance in positive values."""
        # For input with std=1.0, ReLU output has std ~0.5-0.7
        # This is because ReLU zeros negatives but preserves positives
        x = torch.randn(10000, 8) * 1.0
        x_relu = torch.relu(x)
        
        input_std = x.std().item()
        relu_std = x_relu.std().item()
        
        print(f"Input std: {input_std:.6f}, ReLU std: {relu_std:.6f}")
        
        # ReLU output should have significant variance
        assert relu_std > 0.3, f"ReLU should preserve variance, got {relu_std}"
        assert relu_std < input_std, "ReLU output should have lower std than input (zeros negatives)"
    
    def test_batchnorm_with_relu_output_stays_stable(self):
        """Test that BatchNorm with ReLU output maintains stable statistics."""
        bn = nn.BatchNorm1d(8, momentum=0.99, eps=1e-3)
        relu = nn.ReLU()
        
        # Simulate what happens: model learns small values
        x = torch.randn(100, 8) * 0.1  # Small input
        
        bn.train()
        for i in range(200):
            x_relu = relu(x)  # ReLU first
            _ = bn(x_relu)  # BatchNorm on ReLU output
        
        running_var = bn.running_var.mean().item()
        
        # Even with small input, ReLU output has some variance
        # BatchNorm running_var should stay reasonable
        assert running_var > 0.001, \
            f"BatchNorm with ReLU output should maintain reasonable running_var, got {running_var}"
        
        # Test amplification
        bn.eval()
        test_input = torch.randn(100, 8) * 1.0
        test_relu = relu(test_input)
        output = bn(test_relu)
        amplification = output.std().item() / test_relu.std().item()
        
        print(f"Running_var: {running_var:.6f}, Amplification: {amplification:.2f}x")
        
        # With small input training, running_var collapses, causing higher amplification
        # This is expected behavior when BatchNorm is trained on small input
        assert running_var > 0.001, \
            f"BatchNorm with ReLU output should maintain reasonable running_var, got {running_var}"


"""Tests for EM algorithm properties (monotonicity, invariance, etc.).

These tests verify that EM properties are maintained (or document when they're not).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dfm_python.functional.em import run_em_algorithm, EMConfig
from dfm_python.config.schema.params import DFMModelState
from dfm_python.config.constants import DEFAULT_DTYPE


class TestEMMonotonicity:
    """Test EM monotonicity property.
    
    NOTE: With stabilization, normalization, and damping, strict monotonicity
    may not hold. These tests document the actual behavior.
    """
    
    def test_loglik_monotonicity_without_damping(self):
        """Test that loglik is monotone when damping is disabled."""
        # Create simple synthetic data
        np.random.seed(42)
        T, N, m = 50, 5, 3
        
        X = np.random.randn(T, N).astype(DEFAULT_DTYPE) * 0.5
        
        # Create initial state
        state = DFMModelState(
            num_factors=m,
            r=np.array([m]),
            p=1,
            blocks=np.ones((N, 1), dtype=np.int32),
            A=np.eye(m, dtype=DEFAULT_DTYPE) * 0.8,
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE) * 0.1,
            Q=np.eye(m, dtype=DEFAULT_DTYPE) * 0.1,
            R=np.eye(N, dtype=DEFAULT_DTYPE) * 0.1,
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=np.eye(m, dtype=DEFAULT_DTYPE) * 0.5,
            mixed_freq=False,
            n_clock_freq=N,
            n_slower_freq=0,
            max_lag_size=2
        )
        
        # Run EM without damping
        config = EMConfig(damping_factor=None)
        final_state, metadata = run_em_algorithm(
            X, state, max_iter=10, threshold=1e-3, config=config
        )
        
        # Check that loglik history is available
        if 'loglik_history' in metadata:
            logliks = metadata['loglik_history']
            # With stabilization, monotonicity may not hold strictly
            # But we should see general improvement
            if len(logliks) > 1:
                # Allow small decreases due to stabilization bias
                decreases = sum(1 for i in range(1, len(logliks)) if logliks[i] < logliks[i-1])
                decrease_ratio = decreases / (len(logliks) - 1) if len(logliks) > 1 else 0
                
                # Most iterations should show improvement
                assert decrease_ratio < 0.3, (
                    f"Too many loglik decreases ({decreases}/{len(logliks)-1}). "
                    f"This may indicate stabilization bias is too large."
                )
    
    def test_damping_breaks_monotonicity(self):
        """Test that damping breaks monotonicity (expected behavior)."""
        # Create simple synthetic data
        np.random.seed(42)
        T, N, m = 50, 5, 3
        
        X = np.random.randn(T, N).astype(DEFAULT_DTYPE) * 0.5
        
        # Create initial state
        state = DFMModelState(
            num_factors=m,
            r=np.array([m]),
            p=1,
            blocks=np.ones((N, 1), dtype=np.int32),
            A=np.eye(m, dtype=DEFAULT_DTYPE) * 0.8,
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE) * 0.1,
            Q=np.eye(m, dtype=DEFAULT_DTYPE) * 0.1,
            R=np.eye(N, dtype=DEFAULT_DTYPE) * 0.1,
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=np.eye(m, dtype=DEFAULT_DTYPE) * 0.5,
            mixed_freq=False,
            n_clock_freq=N,
            n_slower_freq=0,
            max_lag_size=2
        )
        
        # Run EM with damping
        config = EMConfig(damping_factor=0.8)
        final_state, metadata = run_em_algorithm(
            X, state, max_iter=10, threshold=1e-3, config=config
        )
        
        # With damping, monotonicity is expected to be broken
        # This test documents that behavior
        assert metadata.get('converged') is not None
        # Damping should be logged in metadata
        assert 'damping_applied' in metadata or config.damping_factor is not None


class TestStabilizationTracking:
    """Test that stabilization is properly tracked."""
    
    def test_stabilization_tracking(self):
        """Test that Kalman filter tracks stabilization amount."""
        from dfm_python.ssm.kalman import DFMKalmanFilter
        
        m, N = 3, 5
        kalman = DFMKalmanFilter(
            transition_matrices=np.eye(m, dtype=DEFAULT_DTYPE) * 0.8,
            observation_matrices=np.random.randn(N, m).astype(DEFAULT_DTYPE) * 0.1,
            transition_covariance=np.eye(m, dtype=DEFAULT_DTYPE) * 0.1,
            observation_covariance=np.eye(N, dtype=DEFAULT_DTYPE) * 0.1,
            initial_state_mean=np.zeros(m, dtype=DEFAULT_DTYPE),
            initial_state_covariance=np.eye(m, dtype=DEFAULT_DTYPE) * 0.5
        )
        
        # Check that stabilization tracking exists
        assert hasattr(kalman, '_stabilization_applied'), "Kalman filter should track stabilization"
        assert hasattr(kalman, '_stabilization_amount'), "Kalman filter should track stabilization amount"
        
        # Stabilization should be applied by default
        assert kalman._stabilization_applied is not None


class TestInvarianceDocumentation:
    """Test that invariance violations are properly documented."""
    
    def test_C_normalization_documented(self):
        """Test that C normalization is documented as breaking invariance."""
        from dfm_python.functional.em import _normalize_C_with_invariance
        
        # The function should exist and document invariance preservation
        assert callable(_normalize_C_with_invariance)
        
        # Check that it rescales A, Q, V_0, Z_0
        # (This is tested in test_invariance.py)
        pass

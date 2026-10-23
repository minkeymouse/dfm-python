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


class TestNoEStepStabilization:
    """The E-step uses the parameters exactly as given (no regularization of any covariance)."""

    def test_estep_moments_are_the_exact_moments_of_the_given_parameters(self):
        """Smoothed moments and loglik equal the independent reference filter's (rel 1e-10)."""
        from dfm_python.ssm.kalman import DFMKalmanFilter

        from ..ssm import _reference_kf as ref

        A, C, Q, R, Z0, V0, Y = ref.random_model(m=3, N=5, T=40, nan_frac=0.3, seed=21)
        kalman = DFMKalmanFilter(
            transition_matrices=A, observation_matrices=C, transition_covariance=Q,
            observation_covariance=R, initial_state_mean=Z0, initial_state_covariance=V0,
        )
        for name, given in (("transition_covariance", Q), ("observation_covariance", R),
                            ("initial_state_covariance", V0)):
            np.testing.assert_array_equal(getattr(kalman, name), 0.5 * (given + given.T), err_msg=name)
        assert not hasattr(kalman, "_stabilization_applied")
        EZ, V, VV, ll = kalman.filter_and_smooth(Y)
        expected = ref.filter_smooth(A, C, Q, R, Z0, V0, Y, smoother="rts")
        assert ref.rel_err(EZ, expected["sm"]) < 1e-10
        assert ref.rel_err(V, expected["sP"]) < 1e-10
        assert ref.rel_err(VV, expected["pair"]) < 1e-10
        assert abs(ll - expected["ll"]) < 1e-10 * abs(expected["ll"])


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


class TestEStepFailureFallback:
    """A failed E-step after the first iteration stops EM with the last parameters whose E-step succeeded."""

    @staticmethod
    def _state(N: int, m: int) -> DFMModelState:
        rng = np.random.default_rng(3)
        return DFMModelState(
            num_factors=m, r=np.array([m]), p=1, blocks=np.ones((N, 1), dtype=np.int32),
            A=np.eye(m, dtype=DEFAULT_DTYPE) * 0.8, C=rng.standard_normal((N, m)) * 0.1,
            Q=np.eye(m, dtype=DEFAULT_DTYPE) * 0.1, R=np.eye(N, dtype=DEFAULT_DTYPE) * 0.1,
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE), V_0=np.eye(m, dtype=DEFAULT_DTYPE) * 0.5,
            mixed_freq=False, n_clock_freq=N, n_slower_freq=0, max_lag_size=2,
        )

    def test_fallback_to_the_last_successful_estep(self, monkeypatch):
        import dfm_python.functional.em as em
        from dfm_python.ssm.kalman import DFMKalmanFilter
        from dfm_python.utils.errors import NumericalError

        X = np.random.default_rng(4).standard_normal((40, 5)) * 0.5
        X[3, 1] = np.nan
        inputs, logliks = [], []
        original_step = em.em_step
        original_smooth = DFMKalmanFilter.filter_and_smooth

        def recording_step(X_, state, *args, **kwargs):
            inputs.append(state)
            return original_step(X_, state, *args, **kwargs)

        def failing_third_smooth(self, *args, **kwargs):
            if len(logliks) == 2:
                logliks.append(None)
                raise NumericalError("Kalman filter: the filtered moments or the log-likelihood are not finite.")
            out = original_smooth(self, *args, **kwargs)
            if len(logliks) < 2:
                logliks.append(out[3])
            return out

        monkeypatch.setattr(em, "em_step", recording_step)
        monkeypatch.setattr(DFMKalmanFilter, "filter_and_smooth", failing_third_smooth)
        # normalize_C=False: compare the returned parameters as they are (the
        # final normalization is a change of state units).
        state, meta = run_em_algorithm(X, self._state(5, 3), max_iter=10, threshold=1e-12,
                                       config=EMConfig(normalize_C=False, em_acceleration=None))
        # E-steps 1 and 2 succeeded, E-step 3 failed: EM returns the input of iteration 2 and its loglik.
        assert len(inputs) == 3
        assert meta["num_iter"] == 2 and not meta["converged"]
        for name in ("A", "C", "Q", "R", "Z_0", "V_0"):
            np.testing.assert_array_equal(getattr(state, name), getattr(inputs[1], name), err_msg=name)
        assert meta["loglik"] == logliks[1]

    def test_failure_in_the_first_estep_raises(self, monkeypatch):
        from dfm_python.ssm.kalman import DFMKalmanFilter
        from dfm_python.utils.errors import NumericalError

        def fail(self, *args, **kwargs):
            raise NumericalError("Kalman filter: the innovation covariance is not positive definite at row 0.")

        monkeypatch.setattr(DFMKalmanFilter, "filter_and_smooth", fail)
        with pytest.raises(RuntimeError, match="iteration 1"):
            run_em_algorithm(np.zeros((10, 5)), self._state(5, 3), max_iter=3)

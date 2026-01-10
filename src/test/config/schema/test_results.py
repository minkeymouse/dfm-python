"""Tests for config.schema.results module."""

import pytest
import numpy as np
from dfm_python.config.schema.results import (
    BaseResult,
    DFMResult,
    DDFMResult,
)
from dfm_python.numeric.stability import create_scaled_identity
from dfm_python.config.constants import DEFAULT_IDENTITY_SCALE, DEFAULT_DTYPE


class TestBaseResult:
    """Test suite for BaseResult."""
    
    def test_base_result_initialization(self):
        """Test BaseResult can be initialized."""
        T, N, m = 10, 3, 2
        result = BaseResult(
            x_sm=np.random.randn(T, N).astype(DEFAULT_DTYPE),
            Z=np.random.randn(T, m).astype(DEFAULT_DTYPE),
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE),
            R=create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            A=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Q=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            r=np.array([m], dtype=np.int32),
            p=1
        )
        assert result is not None
        assert result.num_series() == N
        assert result.num_state() == m
    
    def test_base_result_summary(self):
        """Test BaseResult summary method."""
        T, N, m = 10, 3, 2
        result = BaseResult(
            x_sm=np.random.randn(T, N).astype(DEFAULT_DTYPE),
            Z=np.random.randn(T, m).astype(DEFAULT_DTYPE),
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE),
            R=create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            A=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Q=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            r=np.array([m], dtype=np.int32),
            p=1,
            converged=True,
            num_iter=10,
            loglik=-100.0
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Model Summary" in summary
        assert f"Series: {N}" in summary
        assert f"Factors: {m}" in summary
        assert f"Time periods: {T}" in summary


class TestDFMResult:
    """Test suite for DFMResult."""
    
    def test_dfm_result_initialization(self):
        """Test DFMResult can be initialized."""
        T, N, m = 10, 3, 2
        result = DFMResult(
            x_sm=np.random.randn(T, N).astype(DEFAULT_DTYPE),
            Z=np.random.randn(T, m).astype(DEFAULT_DTYPE),
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE),
            R=create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            A=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Q=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            r=np.array([m], dtype=np.int32),
            p=1
        )
        assert result is not None
        assert isinstance(result, DFMResult)
        assert isinstance(result, BaseResult)
    
    def test_dfm_result_summary(self):
        """Test DFMResult summary method."""
        T, N, m = 10, 3, 2
        result = DFMResult(
            x_sm=np.random.randn(T, N).astype(DEFAULT_DTYPE),
            Z=np.random.randn(T, m).astype(DEFAULT_DTYPE),
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE),
            R=create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            A=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Q=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            r=np.array([m], dtype=np.int32),
            p=1,
            converged=True,
            num_iter=5,
            loglik=-123.45
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "DFM Model Summary" in summary
        assert f"Series: {N}" in summary
        assert f"Factors: {m}" in summary
        assert f"Time periods: {T}" in summary
        assert "Converged: True" in summary
        assert "Iterations: 5" in summary
        assert "-123.4500" in summary or "-123.45" in summary


class TestDDFMResult:
    """Test suite for DDFMResult."""
    
    def test_ddfm_result_summary(self):
        """Test DDFMResult summary method includes neural network info."""
        T, N, m = 10, 3, 2
        result = DDFMResult(
            x_sm=np.random.randn(T, N).astype(DEFAULT_DTYPE),
            Z=np.random.randn(T, m).astype(DEFAULT_DTYPE),
            C=np.random.randn(N, m).astype(DEFAULT_DTYPE),
            R=create_scaled_identity(N, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            A=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Q=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            Z_0=np.zeros(m, dtype=DEFAULT_DTYPE),
            V_0=create_scaled_identity(m, DEFAULT_IDENTITY_SCALE, dtype=DEFAULT_DTYPE),
            r=np.array([m], dtype=np.int32),
            p=1,
            converged=True,
            num_iter=20,
            loglik=-200.0,
            training_loss=0.001,
            encoder_layers=[64, 32]
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "DDFM Model Summary" in summary
        assert "Neural Network Training:" in summary
        assert "Final training loss: 0.0010" in summary
        assert "[64, 32]" in summary or "64, 32" in summary


#!/usr/bin/env python3
"""Test script to verify Q cap and C normalization fixes."""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

from dfm_python import DFM
from dfm_python.config import DFMConfig, SeriesConfig, BlockConfig, Params

print("=" * 70)
print("Q Cap and C Normalization Fix Test")
print("=" * 70)
print()

# Generate test data
np.random.seed(42)
T = 100
N = 10
num_factors = 3

# Generate factors
factors = np.zeros((T + 1, num_factors))
factors[0, :] = 0.0
ar_coeffs = np.array([0.8, 0.7, 0.6])
innovation_std = 0.5

for t in range(T):
    factors[t + 1, :] = ar_coeffs * factors[t, :] + np.random.randn(num_factors) * innovation_std

# Generate data
X = np.zeros((T, N))
loadings = np.random.randn(N, num_factors) * 0.5 + 1.0
common_part_all = factors[1:T+1, :] @ loadings.T

idio_std = 0.3
idio_rho = 0.5
idio_innovations = np.random.randn(T, N) * idio_std
idio = np.zeros((T, N))
idio[0, :] = idio_innovations[0, :]
for t in range(1, T):
    idio[t, :] = idio_rho * idio[t-1, :] + idio_innovations[t, :]

X = common_part_all + idio

# Add some missing values
missing_mask = np.random.rand(T, N) < 0.05
X[missing_mask] = np.nan

print(f"Data shape: {X.shape}")
print(f"Missing ratio: {np.isnan(X).sum() / X.size * 100:.2f}%")
print()

# Configure model
series = [
    SeriesConfig(
        series_id=f'series_{i}',
        frequency='m',
        transformation='lin',
        blocks=[1]
    )
    for i in range(N)
]

blocks = {
    'Block_Global': BlockConfig(
        factors=3,
        ar_lag=1,
        clock='m'
    )
}

config = DFMConfig(
    series=series,
    blocks=blocks,
    max_iter=50,
    threshold=1e-4
)

# Tighten stabilization for the test (attributes read via safe_get_attr)
setattr(config, 'max_innovation_variance', 0.5)      # Q cap
setattr(config, 'max_observation_variance', 2.0)     # R cap (stricter)
setattr(config, 'damping_factor', 0.6)               # stronger damping
setattr(config, 'use_damped_updates', True)

# Fit model
print("Fitting model...")
model = DFM()
result = model.fit(X, config)

print()
print("=" * 70)
print("Results Analysis")
print("=" * 70)
print()

# Check Q values
Q = result.Q
Q_diag = np.diag(Q)
num_factors_actual = int(np.sum(result.r))
Q_factor_diag = Q_diag[:num_factors_actual]

print("1. Q (Innovation Covariance) Analysis:")
print(f"   Factor Q values:")
for i in range(num_factors_actual):
    print(f"     Factor {i+1}: {Q_factor_diag[i]:.6f}")
print()
print(f"   Q statistics:")
print(f"     Min: {np.min(Q_factor_diag):.6f}")
print(f"     Max: {np.max(Q_factor_diag):.6f}")
print(f"     Mean: {np.mean(Q_factor_diag):.6f}")
print()

# Check Q floor and cap
Q_min = 0.01
Q_max = 1.0
q_floor_ok = np.all(Q_factor_diag >= Q_min)
q_cap_ok = np.all(Q_factor_diag <= Q_max)

print(f"   Q Floor check (>= {Q_min}): {'✅ PASS' if q_floor_ok else '❌ FAIL'}")
print(f"   Q Cap check (<= {Q_max}): {'✅ PASS' if q_cap_ok else '❌ FAIL'}")
if not q_floor_ok:
    print(f"     ⚠️ Some Q values below floor: {Q_factor_diag[Q_factor_diag < Q_min]}")
if not q_cap_ok:
    print(f"     ⚠️ Some Q values above cap: {Q_factor_diag[Q_factor_diag > Q_max]}")
print()

# Check R values
R = result.R
R_diag = np.diag(R)

print("2. R (Observation Error Covariance) Analysis:")
print(f"   R statistics:")
print(f"     Min: {np.min(R_diag):.6f}")
print(f"     Max: {np.max(R_diag):.6f}")
print(f"     Mean: {np.mean(R_diag):.6f}")
print()

# Check if R is reasonable (should be < 100 for standardized data)
R_reasonable = np.all(R_diag < 100)
print(f"   R Reasonable check (< 100): {'✅ PASS' if R_reasonable else '❌ FAIL'}")
if not R_reasonable:
    print(f"     ⚠️ Some R values too large: {R_diag[R_diag >= 100]}")
print()

# Check C matrix norms
C = result.C
C_factor = C[:, :num_factors_actual]

print("3. C (Loading Matrix) Analysis:")
norms = []
for j in range(num_factors_actual):
    norm = np.linalg.norm(C_factor[:, j])
    norms.append(norm)
    print(f"   Factor {j+1} loading norm: {norm:.6f}")

print()
print(f"   C norm statistics:")
print(f"     Min: {np.min(norms):.6f}")
print(f"     Max: {np.max(norms):.6f}")
print(f"     Mean: {np.mean(norms):.6f}")
print()

# Check if C norms are reasonable (should be close to 1 if normalized)
C_norm_reasonable = np.all(np.abs(np.array(norms) - 1.0) < 0.5)
print(f"   C Normalized check (≈ 1.0): {'✅ PASS' if C_norm_reasonable else '❌ FAIL'}")
print()

# Check explained variance
print("4. Explained Variance Analysis:")
X_var = np.nanvar(X, axis=0)
explained_var = 1 - R_diag / X_var
print(f"   Explained variance per series:")
for i in range(min(5, N)):
    print(f"     Series {i}: {explained_var[i]*100:.2f}%")
print()
print(f"   Explained variance statistics:")
print(f"     Min: {np.min(explained_var)*100:.2f}%")
print(f"     Max: {np.max(explained_var)*100:.2f}%")
print(f"     Mean: {np.mean(explained_var)*100:.2f}%")
print()

explained_positive = np.all(explained_var > 0)
print(f"   Explained variance positive: {'✅ PASS' if explained_positive else '❌ FAIL'}")
if not explained_positive:
    print(f"     ⚠️ Some series have negative explained variance: {explained_var[explained_var <= 0]}")
print()

# Overall assessment
print("=" * 70)
print("Overall Assessment")
print("=" * 70)
print()

all_checks = [
    ("Q Floor", q_floor_ok),
    ("Q Cap", q_cap_ok),
    ("R Reasonable", R_reasonable),
    ("C Normalized", C_norm_reasonable),
    ("Explained Variance Positive", explained_positive)
]

passed = sum(1 for _, ok in all_checks if ok)
total = len(all_checks)

print(f"Tests passed: {passed}/{total}")
print()

if passed == total:
    print("✅ All checks passed! Fixes are working correctly.")
else:
    print("⚠️ Some checks failed. Review the output above.")
    for name, ok in all_checks:
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")

print()


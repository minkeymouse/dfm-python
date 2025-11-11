"""Test script to diagnose factor near-zero issue."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

import dfm_python as dfm
from dfm_python.config import Params

print("="*70)
print("Factor Near-Zero Issue Diagnostic Test")
print("="*70)

# Load actual data
base_dir = project_root
spec_file = base_dir / 'data' / 'sample_spec.csv'
data_file = base_dir / 'data' / 'sample_data.csv'

if not spec_file.exists() or not data_file.exists():
    print(f"ERROR: Data files not found!")
    print(f"  spec_file: {spec_file}")
    print(f"  data_file: {data_file}")
    sys.exit(1)

print(f"\n1. Loading data...")
print(f"   Spec file: {spec_file}")
print(f"   Data file: {data_file}")

# Load config and data
params = Params(max_iter=10, threshold=1e-4)
dfm.from_spec(spec_file, params=params)
dfm.load_data(data_file, sample_start='2015-01-01', sample_end='2022-12-31')

config = dfm.get_config()
print(f"   ✓ Loaded {len(config.series)} series")
print(f"   ✓ Data shape: {dfm._data.shape if hasattr(dfm, '_data') else 'N/A'}")

print(f"\n2. Training model...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dfm.train()

result = dfm.get_result()
if result is None:
    print("   ✗ Training failed!")
    sys.exit(1)

print(f"   ✓ Training completed")
print(f"   ✓ Converged: {result.converged}")
print(f"   ✓ Iterations: {result.num_iter}")

print(f"\n3. Analyzing factors (Z)...")
Z = result.Z
print(f"   Z shape: {Z.shape}")

# Check first few factors (should include Block_Global)
num_factors_to_check = min(5, Z.shape[1])
for i in range(num_factors_to_check):
    factor = Z[:, i]
    print(f"\n   Factor {i+1}:")
    print(f"     Mean: {np.mean(factor):.6f}")
    print(f"     Std: {np.std(factor):.6f}")
    print(f"     Min: {np.min(factor):.6f}, Max: {np.max(factor):.6f}")
    print(f"     Recent (last 3): {factor[-3:]}")
    
    # Check if essentially zero
    if np.abs(np.mean(factor)) < 1e-10 and np.std(factor) < 1e-10:
        print(f"     ⚠ WARNING: Factor {i+1} is essentially zero!")

print(f"\n4. Analyzing Q matrix (innovation covariance)...")
Q = result.Q
print(f"   Q shape: {Q.shape}")
print(f"   Q diagonal (first 10):")
for i in range(min(10, Q.shape[0])):
    print(f"     Q[{i},{i}] = {Q[i,i]:.10e}")

# Check Q[0,0] specifically (Block_Global innovation variance)
if Q.shape[0] > 0:
    print(f"\n   Q[0,0] (Block_Global innovation variance): {Q[0,0]:.10e}")
    if Q[0,0] < 1e-8:
        print(f"     ⚠ WARNING: Q[0,0] is too small! This prevents factor evolution.")
    else:
        print(f"     ✓ Q[0,0] is above minimum threshold")

print(f"\n5. Analyzing C matrix (loadings)...")
C = result.C
print(f"   C shape: {C.shape}")
print(f"   C[:, 0] (loadings on Factor 0 / Block_Global):")
loadings_on_factor0 = C[:, 0]
print(f"     Non-zero count: {np.sum(loadings_on_factor0 != 0)}")
print(f"     Mean: {np.mean(loadings_on_factor0):.6f}")
print(f"     Std: {np.std(loadings_on_factor0):.6f}")
print(f"     Min: {np.min(loadings_on_factor0):.6f}, Max: {np.max(loadings_on_factor0):.6f}")
print(f"     Top 5 absolute values: {np.sort(np.abs(loadings_on_factor0))[-5:][::-1]}")

if np.all(np.abs(loadings_on_factor0) < 1e-6):
    print(f"     ⚠ WARNING: All loadings on Factor 0 are essentially zero!")

print(f"\n6. Checking A matrix (factor dynamics)...")
A = result.A
print(f"   A shape: {A.shape}")
if A.shape[0] > 0:
    print(f"   A[0,0] (Factor 0 AR coefficient): {A[0,0]:.10e}")
    print(f"   A eigenvalues (first 5): {np.linalg.eigvals(A)[:5]}")

print(f"\n7. Summary and diagnosis...")
issues = []

if Q.shape[0] > 0 and Q[0,0] < 1e-8:
    issues.append("Q[0,0] is too small - factor cannot evolve")

if np.all(np.abs(Z[:, 0]) < 1e-10):
    issues.append("Factor 0 (Block_Global) is essentially zero")

if np.all(np.abs(loadings_on_factor0) < 1e-6):
    issues.append("All loadings on Factor 0 are essentially zero")

if A.shape[0] > 0 and np.abs(A[0,0]) < 1e-10:
    issues.append("A[0,0] (AR coefficient) is too small")

if issues:
    print(f"   ⚠ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"     {i}. {issue}")
else:
    print(f"   ✓ No obvious issues found")

print(f"\n" + "="*70)
print("Diagnostic complete")
print("="*70)


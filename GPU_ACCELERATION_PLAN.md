# GPU Acceleration & Lightning Implementation Plan

## Overview

This document outlines the recommended approach for:
1. **GPU Acceleration**: Ensuring all operations stay on GPU
2. **Lightning Implementation**: Proper integration with PyTorch Lightning

## Recommendation Summary

### Architecture: Hybrid Approach
- **LightningModule structure** for device management, logging, module composition
- **Custom execution loop** (`fit_em()`) for natural EM algorithm flow
- **Optional Trainer support** for validation/inference workflows

### GPU Acceleration Strategy
- Remove all CPU transfers
- Create PyTorch version of `rem_nans_spline`
- Vectorize loop-based operations
- Ensure all tensors stay on GPU

---

## Part 1: GPU Acceleration Fixes

### Fix 1: Remove CPU Transfers in `initialize_from_data()`

**Current (BAD):**
```python
# dfm_module.py lines 147-148
blocks_np = self.blocks.cpu().numpy()  # ❌ CPU transfer
r_np = self.r.cpu().numpy()            # ❌ CPU transfer
A, C, Q, R, Z_0, V_0 = self.em.initialize_parameters(
    X,
    r=torch.tensor(r_np, dtype=X.dtype, device=X.device),  # ❌ Creates new tensor
    blocks=torch.tensor(blocks_np, dtype=X.dtype, device=X.device)  # ❌ Creates new tensor
)
```

**Fixed (GOOD):**
```python
# Keep tensors on GPU - no CPU conversion
A, C, Q, R, Z_0, V_0 = self.em.initialize_parameters(
    X,
    r=self.r.to(X.device),  # ✅ Direct device move
    p=self.p,
    blocks=self.blocks.to(X.device),  # ✅ Direct device move
    ...
)
```

### Fix 2: Create PyTorch Version of `rem_nans_spline()`

**Current (BAD):**
```python
# em.py line 315
x_np = x.cpu().numpy()  # ❌ Forces CPU transfer
x_clean_np, _ = rem_nans_spline(x_np, ...)
x_clean = torch.tensor(x_clean_np, device=device, dtype=dtype)  # ❌ Back to GPU
```

**Solution:**
Create `rem_nans_spline_torch()` in `utils/data.py` that:
- Uses `torch.nn.functional.interpolate()` or `torch.linspace()` for splines
- Uses `torch.conv1d()` for moving average filtering
- Stays entirely on GPU

### Fix 3: Vectorize M-Step Operations

**Current (SLOW):**
```python
# em.py lines 159-161
XTX_A = torch.sum(torch.stack([torch.outer(X_A[t, :], X_A[t, :]) for t in range(T - 1)]), dim=0)
```

**Optimized (FAST):**
```python
# Vectorized: compute all outer products at once
XTX_A = torch.sum(X_A[:, :, None] * X_A[:, None, :], dim=0)
```

**Current (SLOW):**
```python
# em.py lines 140-141
for t in range(T):
    EZZ[t, :, :] = Vsmooth[:, :, t + 1] + torch.outer(EZ[t, :], EZ[t, :])
```

**Optimized (FAST):**
```python
# Vectorized: batch outer products
EZZ = Vsmooth[:, :, 1:].permute(2, 0, 1) + torch.bmm(EZ[:, :, None], EZ[:, None, :])
```

---

## Part 2: Lightning Implementation

### Recommended Architecture

#### Option A: Hybrid (RECOMMENDED) ✅

**Structure:**
- Use `LightningModule` for structure and infrastructure
- Use `fit_em()` as primary training interface
- Keep `training_step()` optional (for Trainer-based workflows)

**Benefits:**
- Natural fit for EM algorithm
- Leverages Lightning's device management
- Leverages Lightning's logging
- Flexible: can use Trainer for validation/inference

**Implementation:**
```python
class DFMLightningModule(pl.LightningModule):
    def __init__(self, config, ...):
        super().__init__()
        # Compose modules
        self.kalman = KalmanFilter(...)
        self.em = EMAlgorithm(kalman=self.kalman, ...)
        # Lightning automatically handles device placement
    
    def fit_em(self, X: torch.Tensor, ...) -> DFMTrainingState:
        """Primary training interface - custom loop."""
        # Ensure data is on same device as model
        X = X.to(self.device)  # Lightning provides self.device
        
        # Initialize
        self.initialize_from_data(X)
        
        # EM loop with Lightning logging
        for iteration in range(self.max_iter):
            loglik = self._em_iteration(X)
            
            # Use Lightning's logging
            self.log('train/loglik', loglik, on_step=True)
            self.log('train/em_iteration', iteration)
            
            # Convergence check
            if self._check_convergence(loglik, previous_loglik):
                break
                
        return self.get_result()
    
    def _em_iteration(self, X):
        """Single EM iteration."""
        # E-step: Kalman smoothing
        zsmooth, Vsmooth, _, loglik = self.kalman(...)
        
        # M-step: Parameter updates
        C_new, R_new, A_new, Q_new, Z_0_new, V_0_new = self.em(...)
        
        # Update parameters (no gradients)
        with torch.no_grad():
            self.A.data = A_new
            self.C.data = C_new
            # ...
            
        return loglik
```

**Usage:**
```python
# Create model
model = DFMLightningModule(config)
model = model.to('cuda')  # Lightning handles device

# Train with custom loop (RECOMMENDED)
result = model.fit_em(X_torch, Mx, Wx)

# Optional: Use Trainer for validation/inference
trainer = pl.Trainer(accelerator='gpu', devices=1)
trainer.validate(model, val_dataloader)
```

#### Option B: Full Trainer Integration (NOT RECOMMENDED) ❌

**Why not:**
- EM doesn't fit epoch/batch paradigm
- Awkward: one "epoch" = one EM iteration
- Requires DataLoader even for full sequence
- Convergence checking in `on_train_epoch_end()` is unnatural

**Only use Trainer for:**
- Validation during EM (if you have validation data)
- Inference/prediction workflows
- Distributed training (if needed)

---

## Implementation Checklist

### GPU Acceleration
- [ ] Fix `initialize_from_data()` - remove CPU transfers
- [ ] Create `rem_nans_spline_torch()` - PyTorch version
- [ ] Vectorize M-step loops in `em.py`
- [ ] Ensure all tensor creation uses `device=device`
- [ ] Test GPU memory usage for large problems

### Lightning Implementation
- [ ] Keep `fit_em()` as primary interface
- [ ] Enhance `fit_em()` with Lightning logging
- [ ] Use `self.device` from Lightning
- [ ] Use `self.log()` for metrics
- [ ] Make `training_step()` optional/deprecated
- [ ] Document usage patterns

### Testing
- [ ] Test GPU acceleration (compare CPU vs GPU speed)
- [ ] Test device placement (CPU, single GPU, multi-GPU)
- [ ] Test memory usage for large problems
- [ ] Verify numerical equivalence (CPU vs GPU)

---

## Performance Expectations

### GPU Speedup
- **Small problems** (T < 1k, N < 100): 2-5x speedup
- **Medium problems** (T ~ 5k, N ~ 200): 10-20x speedup
- **Large problems** (T > 10k, N > 500): 20-50x speedup

### Memory Considerations
- GPU memory: ~4x CPU memory for same problem
- Use `torch.float32` (not `float64`) for GPU
- Consider gradient checkpointing for very large problems

---

## Code Changes Required

### 1. `lightning/dfm_module.py`
- Fix `initialize_from_data()` - remove CPU transfers
- Enhance `fit_em()` with Lightning logging
- Add `_em_iteration()` helper method

### 2. `ssm/em.py`
- Fix `initialize_parameters()` - use PyTorch `rem_nans_spline`
- Vectorize M-step loops

### 3. `utils/data.py`
- Add `rem_nans_spline_torch()` function

### 4. Documentation
- Update usage examples
- Document GPU requirements
- Document device management

---

## Migration Path

1. **Phase 1**: Fix GPU transfers (critical)
2. **Phase 2**: Create PyTorch `rem_nans_spline` (critical)
3. **Phase 3**: Vectorize operations (optimization)
4. **Phase 4**: Enhance Lightning integration (polish)


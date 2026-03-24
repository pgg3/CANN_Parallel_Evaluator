## Normalization Pattern Guide

Normalization operators combine reduction (mean/variance) with elementwise operations. They can be implemented with manual Vector ops or high-level APIs.

### Manual Vector Implementation (Recommended for Custom Norms)
```
1. mean = ReduceSum(x) / N
2. diff = x - mean           → Adds(diff, x, -mean, n)
3. var = ReduceSum(diff²) / N
4. normalize = diff / sqrt(var + eps)
5. output = normalize * gamma + beta
```

### High-Level APIs (When Available)
```cpp
// LayerNorm — normalizes over the last dimension(s)
#include "lib/normalization/layernorm.h"
LayerNorm(output, mean, variance, input, gamma, beta, epsilon, tiling);

// RMSNorm — root mean square normalization (no mean subtraction)
#include "lib/normalization/rmsnorm.h"
RMSNorm(output, input, gamma, epsilon, tiling);

// BatchNorm — normalizes over the batch dimension
#include "lib/normalization/batchnorm.h"
BatchNorm(output, input, runningMean, runningVar, gamma, beta, epsilon, tiling);
```

### Variant Handling
- **LayerNorm**: Reduce over last D dims, shape [N, ..., D] → mean/var per sample
- **RMSNorm**: Like LayerNorm but skip mean subtraction; rms = sqrt(mean(x²))
- **BatchNorm**: Reduce over batch dim (dim 0), per-channel statistics
- **InstanceNorm**: Reduce over spatial dims (H, W), per-sample per-channel
- **GroupNorm**: Split channels into groups, reduce over spatial + group
- **L1/L2/Frobenius Norm**: Reduction + rescale. When a single row exceeds UB, requires **two-pass tiling** (see below)

### Reduce Axis by Variant
Different norms reduce over different dimensions — this is the critical difference:
```
LayerNorm:    reduce over last dim (features).     Stats per sample.
RMSNorm:     reduce over last dim (no mean step).  Stats per sample.
BatchNorm:   reduce over dim 0 + spatial dims.     Stats per channel. (inference: use running stats)
InstanceNorm: reduce over spatial dims only.        Stats per sample per channel.
GroupNorm:   reduce over spatial + (C/G) channels.  Stats per sample per group.
L1/L2/Frobenius: reduce over target dim → rescale.  Two-pass tiling when row > UB.
```
> **BatchNorm inference**: skip reduction entirely — use `runningMean` and `runningVar` as-is.

### Buffer Layout
- Input queue (`VECIN`), output queue (`VECOUT`)
- Work buffer (`TBuf<VECCALC>`) for reduction workspace
- Intermediate buffers for mean, variance if computing manually
- High-level APIs manage internal buffers but may need workspace

### Key Points
- High-level APIs (LayerNorm, RMSNorm) require tiling from host — use corresponding tiling helpers
- For simple norms (L1, L2, Frobenius): use ReduceSum + Sqrt/Rsqrt, pure Vector ops
- **When row > UB (e.g. L2 norm on dim=65535)**: two-pass tiling required — see below
- Epsilon must be added before Rsqrt to avoid division by zero

### Performance: Two-Pass Bandwidth-Bound Norms (L1/L2/Frobenius)

When a single row exceeds UB capacity, norm computation requires two passes over GM:
- **Pass 1 (Reduce):** loop over tiles, accumulate squared sum (L2) or abs sum (L1) into a scalar → compute invNorm via Rsqrt
- **Pass 2 (Normalize):** loop over tiles again, multiply each element by invNorm, write out

**Key optimizations:**
1. **Maximize tile size to minimize DMA calls** — for bandwidth-bound kernels (fast vector ops, slow DMA), fewer large tiles always beats many small tiles. Use 2× `TBuf<VECCALC>` (data + work) instead of TQue to minimize buffer overhead: `tileLen = UB_SIZE / (2 * sizeof(float))` ≈ 22528 per tile. This is 2× larger than TQue with BUFFER_NUM=2.
2. **In-place squaring in Pass 1** — `Mul(data, data, data, len)` squares data in-place. Safe because Pass 2 re-reads from GM.
3. **ReduceSum buffer rule** — `ReduceSum(dst, src, workLocal, len)`: `dst` and `workLocal` **MUST be different tensors**. Example: `ReduceSum(dataBuf, dataBuf, workBuf, len)` is valid (dst=data, work=work). But `ReduceSum(workBuf, dataBuf, workBuf, len)` is **INVALID** (dst=work=workBuf, silent garbage).
4. **Multi-core division** — divide rows across cores via `GetBlockIdx()`. Use maximum valid `BLOCK_DIM` when total rows are large to maximize parallelism.

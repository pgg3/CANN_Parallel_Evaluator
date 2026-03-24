## Pooling Pattern Guide

Pooling operators apply a sliding window reduction (max or average) over spatial dimensions. AscendC has NO direct `MaxPool()` or `AvgPool()` API — you must implement the sliding window logic manually using Vector ops.

### Implementation Strategy
1. **TilingFunc** computes output spatial dimensions:
   ```
   H_out = (H + 2*pad - kH) / stride + 1
   W_out = (W + 2*pad - kW) / stride + 1
   ```
2. **Kernel** loops over output spatial positions:
   - For each output position (oh, ow), identify the input window
   - Load the window elements into UB
   - Apply reduction: `ReduceMax` (max pooling) or `ReduceSum` + scale (avg pooling)
   - Write result to output

### Max Pooling
```cpp
// For each output position:
//   1. Gather window elements into a contiguous buffer
//   2. ReduceMax(dst, windowBuf, workBuf, windowSize)
//   3. Store result
```

### Average Pooling
```cpp
// For each output position:
//   1. Gather window elements into a contiguous buffer
//   2. ReduceSum(dst, windowBuf, workBuf, windowSize)
//   3. Muls(dst, dst, 1.0f / windowSize, 1)  // divide by window size
//   4. Store result
```

### Dimension Handling
- **1D**: Single spatial dimension, window slides along W
- **2D**: Two spatial dimensions (H, W), nested loops over (oh, ow)
- **3D**: Three spatial dimensions (D, H, W), triple nested loops

### Key Points
- No `MaxPool()`, `AvgPool()`, or `Pool()` API — implement with ReduceMax/ReduceSum
- Process one output element at a time or tile multiple output positions
- Handle padding: check bounds, use 0 for avg pooling pad, -inf for max pooling pad
- DataCopy with stride can help gather non-contiguous window elements
- Work buffer needed for ReduceMax/ReduceSum
- Use Vector unit, standard TQue/TBuf buffer management

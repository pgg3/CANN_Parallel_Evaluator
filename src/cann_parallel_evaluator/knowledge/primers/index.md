## Index Pattern Guide

Index operators select, scatter, or rearrange elements based on index tensors. They use Gather/Scatter APIs or manual offset computation in the Vector unit.

### Key APIs
```cpp
// Gather: collect elements from src at positions specified by offset
Gather(dst, src, offsetLocal, repeatTimes, params);
// offset: LocalTensor<uint32_t>, values are byte offsets into src

// Scatter: distribute elements from src to dst at positions specified by offset
Scatter(dst, src, offsetLocal, repeatTimes, params);

// GatherMask: conditional gather using a mask
GatherMask(dst, src, maskLocal, params, rsvdCnt);
```

### Common Operator Mappings
- **Embedding**: `Gather` from embedding table using token indices
- **Gather**: Direct use of `Gather` API with index-to-offset conversion
- **Scatter / Scatter_Add**: Use `Scatter` API (scatter_add accumulates instead of overwriting)
- **Index_Select**: Gather along a specific dimension
- **Argmax / Argmin**: `ReduceMax(dst, src, workBuf, count, /*calIndex=*/true)` — calIndex=true returns the index
- **Masked_Fill**: Use `CompareScalar` to create mask, then `Select` to fill

### Offset Computation
Gather/Scatter offsets are byte offsets (not element indices):
```cpp
// Convert element index to byte offset:
// offset_bytes = index * sizeof(T)
// For float: offset_bytes = index * 4
```

### Dimension Handling
For multi-dimensional gather along a specific dim:
1. TilingFunc: compute outer_size, inner_size, gather_size from shape and dim
2. Kernel: loop over outer dims, for each slice apply Gather/Scatter

### Key Points
- Offset tensors must be `LocalTensor<uint32_t>` with byte offsets
- For simple 1D gather, directly compute offsets and call Gather
- Argmax uses ReduceMax with `calIndex=true` — result contains both max value and index
- Use Vector unit, standard TQue/TBuf buffer management

### Scatter-Add Accumulation Pattern
For scatter_add / index_add, accumulate in UB then write back:
```cpp
// 1. Load destination slice into UB (or zero-initialize)
Duplicate(outLocal, 0.0f, outLen);
// 2. Accumulate using scalar access (index-dependent, cannot vectorize)
for (uint32_t i = 0; i < numIndices; i++) {
    uint32_t idx = indexLocal.GetValue(i);
    float val = outLocal.GetValue(idx) + srcLocal.GetValue(i);
    outLocal.SetValue(idx, val);
}
// 3. Write accumulated result back to GM
DataCopy(outputGm[offset], outLocal, outLen);
```
> Scalar GetValue/SetValue is acceptable here because access is index-dependent.
> For non-accumulating scatter, just `SetValue(idx, srcVal)` without reading dst first.

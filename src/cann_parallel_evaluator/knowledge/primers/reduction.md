## Reduction Pattern Guide

Reduction operators collapse one or more dimensions (e.g., sum, max, mean along an axis).

### Buffer Layout
- Input queue (`VECIN`) + output queue (`VECOUT`)
- **workLocal buffer** required by Reduce APIs — allocate via `TBuf<QuePosition::VECCALC>`
- workLocal size must be >= `count * sizeof(T)`

### Key APIs
- `ReduceSum(dst, src, workLocal, count)` — dst[0] = sum of src[0..count-1]
- `ReduceMax(dst, src, workLocal, count, calIndex=false)` — dst[0] = max
- `ReduceMin(dst, src, workLocal, count, calIndex=false)` — dst[0] = min

### Reduction Along a Dimension
For reducing along a specific axis of a multi-dimensional tensor:
1. In TilingFunc, compute the stride and reduction length for the target axis
2. In the kernel, loop over the non-reduced dimensions
3. For each slice, call ReduceSum/ReduceMax on a contiguous chunk

### Mean Reduction
There is no `ReduceMean`. Compute it as:
```cpp
ReduceSum(dst, src, workLocal, count);
Muls(dst, dst, 1.0f / static_cast<float>(count), 1);
```

### Common Mistakes
- Forgetting the workLocal buffer (causes silent corruption or crash)
- workLocal too small — must be >= count * sizeof(T)
- Writing reduced scalar back to GM: use DataCopy with length=1 (or the aligned minimum)

### Cumulative Scan (cumsum / cumprod)
Cumsum/cumprod have sequential dependencies (each output depends on the previous) — they cannot use ReduceSum. Use a scalar accumulation loop:
```cpp
// cumsum: output[i] = sum(input[0..i])
float acc = 0.0f;
for (uint32_t i = 0; i < count; i++) {
    acc += xLocal.GetValue(i);
    yLocal.SetValue(i, acc);
}
// cumprod: same pattern with acc *= xLocal.GetValue(i)
// cumsum_exclusive: write acc before adding (output[0] = 0)
// cumsum_reverse: iterate from count-1 to 0
```
> Scalar GetValue/SetValue loop is necessary here — vector APIs cannot express sequential prefix operations.

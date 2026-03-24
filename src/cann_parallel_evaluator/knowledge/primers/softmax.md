## Softmax Pattern Guide

Softmax = max → sub → exp → sum → div. Requires multiple passes over the data.

### Algorithm
```
1. max_val = ReduceMax(input)           // find max for numerical stability
2. shifted = input - max_val            // subtract max (broadcast scalar)
3. exp_vals = Exp(shifted)              // exponentiate
4. sum_val = ReduceSum(exp_vals)        // sum of exponentials
5. output = exp_vals / sum_val          // normalize (broadcast scalar)
```

### Buffer Layout
- Input queue (`VECIN`) + output queue (`VECOUT`)
- At least 1 `VECCALC` queue for intermediate results
- workLocal buffer for ReduceMax and ReduceSum
- Total: ~4-5 buffers → use smaller tileLength (see UB budget table)

### Key APIs Used
- `ReduceMax(dst, src, workLocal, count)` — find max
- `Adds(dst, src, -max_val, count)` — subtract max (broadcast scalar subtraction)
- `Exp(dst, src, count)` — exponentiate
- `ReduceSum(dst, src, workLocal, count)` — sum
- `Muls(dst, src, 1.0f/sum_val, count)` — divide by sum (as multiply by reciprocal)

### Accessing Scalar Results
After ReduceMax/ReduceSum, the result is in dst[0]. To use it as a scalar for Adds/Muls, you need to read it from the LocalTensor. Use `dst.GetValue(0)` to extract the scalar value.

### Common Mistakes
- Forgetting numerical stability (subtract max before exp)
- Not allocating enough VECCALC buffers for intermediates
- UB overflow from too many buffers — reduce tileLength accordingly

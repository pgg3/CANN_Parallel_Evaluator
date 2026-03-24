## Elementwise Pattern Guide

Elementwise operators apply the same operation to every element independently. This is the simplest and most common pattern.

### Buffer Layout
- 1 input queue (`VECIN`) per input tensor + 1 output queue (`VECOUT`)
- For unary ops: 2 queues total (in + out)
- For binary ops: 3 queues total (inX + inY + out)
- If you need intermediate results, add `VECCALC` queues

### Key APIs
- **Unary math**: `Exp(d,s,n)` `Ln(d,s,n)` `Sqrt(d,s,n)` `Rsqrt(d,s,n)` `Abs(d,s,n)` `Reciprocal(d,s,n)`
- **Binary**: `Add(d,a,b,n)` `Sub(d,a,b,n)` `Mul(d,a,b,n)` `Div(d,a,b,n)` `Max(d,a,b,n)` `Min(d,a,b,n)`
- **Scalar-vector**: `Adds(d,s,scalar,n)` `Muls(d,s,scalar,n)` `Maxs(d,s,scalar,n)` `Mins(d,s,scalar,n)`
- **Activation**: `Relu(d,s,n)` `LeakyRelu(d,s,alpha,n)`
- **High-level** (need `#include`): `Tanh` `Sigmoid` `Gelu` `Swish`

### Conditional Operations (e.g., SELU, ELU)
Use the Compare + Select pattern:
1. Compute both branches into separate buffers
2. `CompareScalar(mask, src, 0.0f, CMPMODE::GT, n)` → `LocalTensor<uint8_t>` mask
3. `Select(dst, mask, branch_pos, branch_neg, SELMODE::VSEL_TENSOR_TENSOR_MODE, n)`

### Common Mistakes
- No `Neg()` function → use `Muls(d, s, -1.0f, n)`
- No `Subs()` / `Divs()` → use `Adds(d, s, -scalar, n)` / `Muls(d, s, 1.0f/scalar, n)`
- `Sub(d, s, scalar)` does NOT exist → `Sub` requires two tensors
- Mask type is `LocalTensor<uint8_t>`, not `bool` or `SelectMask`

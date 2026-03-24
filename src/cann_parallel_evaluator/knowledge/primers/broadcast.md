## Broadcast Pattern Guide

Broadcast operators apply a scalar or small tensor to a larger tensor element-wise.

### Scalar Broadcast
Use scalar-vector APIs directly:
- `Adds(dst, src, scalar, n)` — add scalar to every element
- `Muls(dst, src, scalar, n)` — multiply every element by scalar
- `Maxs(dst, src, scalar, n)` — element-wise max with scalar
- `Mins(dst, src, scalar, n)` — element-wise min with scalar

### Tensor Broadcast (different shapes)
When broadcasting tensors of different shapes:
1. In TilingFunc, determine the broadcast pattern (which dims need broadcasting)
2. In the kernel, use loops over the broadcast dimensions
3. Use `Duplicate(dst, scalar, n)` to fill a LocalTensor with a repeated value

### Common Mistakes
- Using `Sub(d, s, scalar)` instead of `Adds(d, s, -scalar, n)`
- Trying to index individual elements — use vector ops instead

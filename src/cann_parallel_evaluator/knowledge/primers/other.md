## General Pattern Guide

For operators that don't fit neatly into elementwise/reduction/softmax/broadcast:

### Key Principles
1. All computation must use vector APIs — no scalar C math functions
2. Data flows through the CopyIn → Compute → CopyOut pipeline
3. All buffers must fit in UB (~176KB usable)
4. Use `VECCALC` queues for any intermediate buffers
5. Align tileLength to 32 bytes (8 floats / 16 halfs)

### Complex Operators
For operators requiring matrix multiplication, use the Cube unit:
```cpp
#include "lib/matmul_intf.h"
Matmul<half, half, float> mm;
```

For normalization layers:
```cpp
#include "lib/normalization/layernorm.h"
#include "lib/normalization/rmsnorm.h"
```

### Multi-pass Algorithms
Some operators need multiple passes over the data (like softmax). Strategy: process each pass as a separate CopyIn→Compute→CopyOut loop, or keep intermediate results in VECCALC buffers between passes.

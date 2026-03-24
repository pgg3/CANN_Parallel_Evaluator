## Advanced APIs (for complex operators)

### Matmul API (Cube Unit)
```cpp
#include "lib/matmul_intf.h"

// Template: Matmul<AType, BType, CType, BiasType=CType>
// AType/BType/CType use MatmulType<Position, Format, DataType>
// A/B use half (Cube native), C uses float (fp32 accumulation)
// Cast Python float32 inputs to half in OUTPUT_ALLOC_CODE
Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
       MatmulType<TPosition::GM, CubeFormat::ND, half>,
       MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

// Lifecycle:
mm.Init(&tilingData.cubeTiling, &pipe);  // takes TCubeTiling* pointer
mm.SetTensorA(aGm);                      // GlobalTensor<half>
mm.SetTensorB(bGm);                      // GlobalTensor<half>
mm.IterateAll(cGm);                      // compute all tiles → GlobalTensor<float>
mm.End();                                 // finalize

// Formats:
//   CubeFormat::ND — normal row-major
//   CubeFormat::NT — B is transposed (compute A × B^T)
//   CubeFormat::TN — A is transposed (compute A^T × B)
```

### Matmul Host Tiling
```cpp
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling matmulTiling(platform);
// SetAType/SetBType/SetCType require 3 args: (TPosition, CubeFormat, DataType)
matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
matmulTiling.SetShape(M, N, K);            // parameter order: (M, N, K)
matmulTiling.SetFixSplit(-1, -1, -1);      // auto split (-1) or fixed split
matmulTiling.SetBatchNum(batchNum);        // for batched matmul
TCubeTiling cubeTiling;
int64_t workspaceSize = matmulTiling.GetTiling(cubeTiling);  // returns workspace size
```

### Normalization APIs
```cpp
// LayerNorm: normalizes over last dim(s)
#include "lib/normalization/layernorm.h"
LayerNorm(output,       // GlobalTensor<T> — normalized output
          mean,         // GlobalTensor<float> — per-sample mean
          variance,     // GlobalTensor<float> — per-sample variance
          inputX,       // GlobalTensor<T> — input
          gamma,        // GlobalTensor<T> — scale (weight)
          beta,         // GlobalTensor<T> — shift (bias)
          epsilon,      // float — stability constant
          tiling);      // LayerNormTiling — computed on host

// RMSNorm: root mean square normalization (no mean subtraction)
#include "lib/normalization/rmsnorm.h"
RMSNorm(output, inputX, gamma, epsilon, tiling);

// BatchNorm: normalizes over batch dimension
#include "lib/normalization/batchnorm.h"
BatchNorm(output, inputX, runningMean, runningVar, gamma, beta, epsilon, tiling);
```

### Index APIs (Gather/Scatter)
```cpp
// Gather: collect elements from src at byte-offset positions
Gather(dst,             // LocalTensor<T> — destination
       src,             // GlobalTensor<T> or LocalTensor<T> — source
       offsetLocal,     // LocalTensor<uint32_t> — byte offsets into src
       repeatTimes,     // uint32_t — number of gather repeats
       params);         // GatherParams — stride/offset config

// Scatter: distribute elements from src to dst at byte-offset positions
Scatter(dst,            // GlobalTensor<T> or LocalTensor<T> — destination
        src,            // LocalTensor<T> — source
        offsetLocal,    // LocalTensor<uint32_t> — byte offsets into dst
        repeatTimes,    // uint32_t — number of scatter repeats
        params);        // ScatterParams — stride/offset config

// GatherMask: conditional gather using mask
GatherMask(dst, src, maskLocal, params, rsvdCnt);

// Offset format: byte offsets = element_index × sizeof(T)
// For float: offset_bytes = element_index × 4
```

### Transpose API
```cpp
#include "lib/transpose/confusion_transpose.h"
ConfusionTranspose(dst,    // GlobalTensor<T>
                   src,    // GlobalTensor<T>
                   tiling); // ConfusionTransposeTiling — perm + shape
```

### Reduction APIs (with index)
```cpp
// ReduceMax with index: calIndex=true returns both max value and its index
ReduceMax(dst, src, workLocal, count, /*calIndex=*/true);
// dst[0] = max value, index available in work buffer

// ReduceMin with index: same pattern
ReduceMin(dst, src, workLocal, count, /*calIndex=*/true);
```

### Data Movement
```cpp
// Duplicate: fill LocalTensor with a scalar value
Duplicate(dst, scalar, count);  // dst[0..count-1] = scalar

// DataCopy with stride (for non-contiguous access):
DataCopyParams params;
params.blockCount = numBlocks;
params.blockLen = blockLen;       // bytes per block
params.srcStride = srcStride;     // bytes between blocks in src
params.dstStride = dstStride;     // bytes between blocks in dst
DataCopy(dst, src, params);
```

## Matmul Pattern Guide

Matrix multiplication operators use the Cube unit via the `Matmul<>` template class. This handles the L1/L0 memory pipeline automatically.

### Matmul<> Template Lifecycle

**CRITICAL**: On 910B, you MUST use `REGIST_MATMUL_OBJ` macro, NOT manual `mm.Init()`. Manual Init causes `free(): double free` crash.

```cpp
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

// 1. Declare type aliases (half for all on 910B)
using aType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using bType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using cType = MatmulType<TPosition::GM, CubeFormat::ND, half>;

// In kernel entry body (FLAT structure, NOT in a class):
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // Required for AIC+AIV scheduling

TPipe pipe;
GlobalTensor<half> aGM, bGM, cGM;
aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(A), M * K);
bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(B), K * N);
cGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), M * N);

// 2. Register matmul object (replaces mm.Init + mm.SetWorkspace)
Matmul<aType, bType, cType> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTiling);

// 3. Set operands (GlobalTensor)
mm.SetTensorA(aGM);
mm.SetTensorB(bGM);

// 4. Iterate: compute all tiles
mm.IterateAll(cGM);
// Or manual loop: while (!mm.Iterate(cGM)) {}

// 5. End
mm.End();
```

### Tiling (Host-Side)
```cpp
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling matmulTiling(platform);
// SetAType/SetBType/SetCType require 3 args: (TPosition, CubeFormat, DataType)
// DT_FLOAT16 for all (Cube native)
matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetShape(M, N, K);  // parameter order: M, N, K
matmulTiling.GetTiling(tiling.cubeTiling);  // fills cubeTiling struct directly
```

### Workspace
Cube operators need system workspace (32 MB fixed) for internal KFC server/client coordination:
```cpp
size_t* ws = context->GetWorkspaceSizes(1);
ws[0] = 32 * 1024 * 1024;  // 32 MB fixed
```
The kernel accesses it via `GetSysWorkSpacePtr()` through `REGIST_MATMUL_OBJ`.
Do NOT use `mm.SetWorkspace()` or pass user workspace manually.

### Key Points
- BLOCK_DIM = 1 (Matmul<> handles multi-core internally)
- TILING_FIELDS must include `struct TCubeTiling cubeTiling`
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` is mandatory
- Flat kernel structure only — do NOT wrap in a class
- Transposed inputs: use `CubeFormat::NT` (transposed B) or `CubeFormat::TN` (transposed A)
- Batched matmul: set batch dim via `matmulTiling.SetBatchNum(batch)`, loop over batches with offset
- Always `#include "lib/matmul_intf.h"` in KERNEL_INCLUDES
- Always `#include "tiling/platform/platform_ascendc.h"` in TILING_FUNC_INCLUDES

### Structured Matrix Variants
For triangular, symmetric, or diagonal matmul, **run a full Matmul then post-process**:
```cpp
// After mm.IterateAll(outputGm): zero out unwanted region with Vector ops
// Lower triangular: zero upper triangle
for (uint32_t i = 0; i < M; i++)
    for (uint32_t j = i + 1; j < N; j++)
        outputGm.SetValue(i * N + j, 0.0f);  // scalar GM write
// Diagonal: zero all off-diagonal elements
// Symmetric: copy lower to upper (or average)
```
> Matmul<> has no built-in masking — always compute full result, then mask.

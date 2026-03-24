## Cube Tiling Fundamentals

Cube (matrix multiplication) operators use a different tiling strategy than Vector operators. The `Matmul<>` template handles L1/L0 memory management internally, but you must configure the tiling on the host side.

### Memory Hierarchy
```
GM → L1 (512KB) → L0A/L0B (64KB) → Cube → L0C (256KB) → UB → GM
```
The Matmul<> template manages L1→L0 data movement automatically.

### M/K/N Tiling
Matrix C[M,N] = A[M,K] × B[K,N]. The tiling splits M, K, N into blocks:
- **M tiles**: rows of A and C
- **K tiles**: shared dimension (accumulation), split for L1 capacity
- **N tiles**: columns of B and C

### TCubeTiling Structure
Host-side tiling uses `TCubeTiling` (from `lib/matmul/matmul_tiling.h`):
```cpp
TCubeTiling cubeTiling;
```
This struct is populated by `MatmulApiTiling` and passed to the kernel via TilingData.

### MatmulApiTiling Host API
```cpp
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling matmulTiling(platform);

// Set data types - require 3 args: (TPosition, CubeFormat, DataType)
matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);

// Set dimensions - parameter order: (M, N, K)
matmulTiling.SetShape(M, N, K);

// Optional: fix split factors (-1 = auto)
matmulTiling.SetFixSplit(splitM, splitN, splitK);

// Compute tiling - GetTiling returns workspace size (int64_t)
TCubeTiling cubeTiling;
int64_t workspaceSize = matmulTiling.GetTiling(cubeTiling);
```

### Workspace
Cube operators require a fixed 32 MB system workspace for KFC server/client coordination:
```cpp
size_t* ws = context->GetWorkspaceSizes(1);
ws[0] = 32 * 1024 * 1024;  // 32 MB fixed
```
The kernel accesses it via `GetSysWorkSpacePtr()` through `REGIST_MATMUL_OBJ`.
Do NOT use manual `mm.Init()` + `mm.SetWorkspace()` — use `REGIST_MATMUL_OBJ` instead.

### TILING_FIELDS for Cube Operators
```
struct TCubeTiling cubeTiling
uint32_t M
uint32_t K
uint32_t N
```

### Key Rules
- **Alignment**: M, K, N should be multiples of 16 for Cube unit efficiency
- **BLOCK_DIM = 1**: Matmul<> handles multi-core distribution internally
- **L0/UB separation**: Do not use UB for Cube compute; UB is for pre/post-processing (e.g., bias add)
- **Data types**: A and B are typically float16; C (accumulation) is float32

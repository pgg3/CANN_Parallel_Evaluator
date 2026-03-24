## Curated Example: Standard Matrix Multiplication (Cube/Matmul Pattern)

This example demonstrates all 3 file components for a standard matrix multiplication C = A × B using the Matmul<> template.
Python inputs are float32; kernel uses half (fp16) for Cube unit; output is half.
The dtype conversion happens in the pybinding function.

**CRITICAL**: On 910B, Cube operators MUST use:
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` — declares mixed AIC+AIV core scheduling
- `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &cubeTiling)` — NOT manual `mm.Init()` + `mm.SetWorkspace()`
- Flat kernel structure (local variables), NOT class-based — because `REGIST_MATMUL_OBJ` may inject `return` on AIC path

### OP_KERNEL

`op_kernel/standard_matrix_multiplication_custom.cpp`:
```cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

using aType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using bType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using cType = MatmulType<TPosition::GM, CubeFormat::ND, half>;

extern "C" __global__ __aicore__ void standard_matrix_multiplication_custom(GM_ADDR A, GM_ADDR B, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    GlobalTensor<half> aGM, bGM, cGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(A), tilingData.M * tilingData.K);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(B), tilingData.K * tilingData.N);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), tilingData.M * tilingData.N);

    Matmul<aType, bType, cType> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTiling);
    mm.SetTensorA(aGM);
    mm.SetTensorB(bGM);
    mm.IterateAll(cGM);
    mm.End();
}
```

### OP_HOST

`op_host/standard_matrix_multiplication_custom_tiling.h`:
```cpp
#ifndef STANDARD_MATRIX_MULTIPLICATION_CUSTOM_TILING_H
#define STANDARD_MATRIX_MULTIPLICATION_CUSTOM_TILING_H

#include "register/tilingdata_base.h"
#include "lib/matmul/matmul_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StandardMatrixMultiplicationCustomTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, K);
    TILING_DATA_FIELD_DEF(uint32_t, N);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StandardMatrixMultiplicationCustom, StandardMatrixMultiplicationCustomTilingData)
}

#endif  // STANDARD_MATRIX_MULTIPLICATION_CUSTOM_TILING_H
```

`op_host/standard_matrix_multiplication_custom.cpp`:
```cpp
#include "standard_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto aShape = context->GetInputShape(0)->GetStorageShape();
    auto bShape = context->GetInputShape(1)->GetStorageShape();

    uint32_t M = aShape.GetDim(0);
    uint32_t K = aShape.GetDim(1);
    uint32_t N = bShape.GetDim(1);

    StandardMatrixMultiplicationCustomTilingData tiling;
    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);

    // Cube tiling - use matmul_tiling::MatmulApiTiling (NOT optiling::MatmulTiling)
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling matmulTiling(platform);
    // SetAType/SetBType/SetCType require 3 args: (TPosition, CubeFormat, DataType)
    // DT_FLOAT16 for all (Cube unit native)
    matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    // SetShape parameter order: (M, N, K)
    matmulTiling.SetShape(M, N, K);

    // GetTiling fills cubeTiling struct (direct assign, no set_ method)
    matmulTiling.GetTiling(tiling.cubeTiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(1);

    // System workspace for Matmul API — 32 MB fixed
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 32 * 1024 * 1024;

    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    gert::Shape* cShape = context->GetOutputShape(0);
    cShape->SetDimNum(2);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, bShape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class StandardMatrixMultiplicationCustom : public OpDef {
public:
    explicit StandardMatrixMultiplicationCustom(const char* name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddWorkspace(32 * 1024 * 1024);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(StandardMatrixMultiplicationCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor standard_matrix_multiplication_custom_impl_npu(const at::Tensor& A_in, const at::Tensor& B_in) {
    // Cast float32 inputs to half for Cube unit (must match kernel's half GlobalTensor)
    at::Tensor A = A_in.to(at::kHalf);
    at::Tensor B = B_in.to(at::kHalf);
    int64_t M = A.size(0);
    int64_t N = B.size(1);
    at::Tensor result = at::empty({M, N}, A.options());
    EXEC_NPU_CMD(aclnnStandardMatrixMultiplicationCustom, A, B, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("standard_matrix_multiplication_custom", &standard_matrix_multiplication_custom_impl_npu, "standard_matrix_multiplication_custom operator");
}
```

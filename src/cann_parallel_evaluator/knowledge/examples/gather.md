## Curated Example: Gather (Mixed/Index Pattern)

This example demonstrates all 3 file components for a gather operator that collects elements from a source tensor along a given dimension using an index tensor.

### OP_KERNEL

`op_kernel/gather_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelGather {
public:
    __aicore__ inline KernelGather() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR index, GM_ADDR dst,
                                 uint32_t outerSize, uint32_t srcDimSize,
                                 uint32_t idxDimSize, uint32_t innerSize) {
        this->outerSize = outerSize;
        this->srcDimSize = srcDimSize;
        this->idxDimSize = idxDimSize;
        this->innerSize = innerSize;

        uint32_t srcTotal = outerSize * srcDimSize * innerSize;
        uint32_t idxTotal = outerSize * idxDimSize * innerSize;
        srcGm.SetGlobalBuffer((__gm__ float*)src, srcTotal);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)index, idxTotal);
        dstGm.SetGlobalBuffer((__gm__ float*)dst, idxTotal);

        uint32_t sliceSize = innerSize;
        sliceSize = (sliceSize + 7) / 8 * 8;  // Align to 32 bytes

        pipe.InitBuffer(srcQueue, BUFFER_NUM, sliceSize * sizeof(float));
        pipe.InitBuffer(dstQueue, BUFFER_NUM, sliceSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t o = 0; o < outerSize; o++) {
            for (uint32_t i = 0; i < idxDimSize; i++) {
                // Read index value
                int32_t idx = 0;
                if (innerSize == 1) {
                    // Scalar index case
                    // Read index from GM (simplified)
                    uint32_t idxOffset = o * idxDimSize + i;
                    LocalTensor<int32_t> idxLocal = srcQueue.AllocTensor<int32_t>();
                    DataCopy(idxLocal, idxGm[idxOffset], 8);  // min aligned copy
                    srcQueue.EnQue(idxLocal);
                    idxLocal = srcQueue.DeQue<int32_t>();
                    idx = idxLocal.GetValue(0);
                    srcQueue.FreeTensor(idxLocal);
                }

                // Gather: copy src[o, idx, :] → dst[o, i, :]
                uint32_t srcOffset = (o * srcDimSize + idx) * innerSize;
                uint32_t dstOffset = (o * idxDimSize + i) * innerSize;

                LocalTensor<float> inLocal = srcQueue.AllocTensor<float>();
                DataCopy(inLocal, srcGm[srcOffset], innerSize);
                srcQueue.EnQue(inLocal);
                inLocal = srcQueue.DeQue<float>();

                LocalTensor<float> outLocal = dstQueue.AllocTensor<float>();
                DataCopy(outLocal, inLocal, innerSize);
                dstQueue.EnQue(outLocal);
                outLocal = dstQueue.DeQue<float>();
                DataCopy(dstGm[dstOffset], outLocal, innerSize);
                dstQueue.FreeTensor(outLocal);
                srcQueue.FreeTensor(inLocal);
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> srcQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dstQueue;
    GlobalTensor<float> srcGm, dstGm;
    GlobalTensor<int32_t> idxGm;
    uint32_t outerSize, srcDimSize, idxDimSize, innerSize;
};

extern "C" __global__ __aicore__ void gather_custom(GM_ADDR src, GM_ADDR index, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelGather op;
    op.Init(src, index, dst, tilingData.outerSize, tilingData.srcDimSize,
            tilingData.idxDimSize, tilingData.innerSize);
    op.Process();
}
```

### OP_HOST

`op_host/gather_custom_tiling.h`:
```cpp
#ifndef GATHER_CUSTOM_TILING_H
#define GATHER_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, outerSize);
    TILING_DATA_FIELD_DEF(uint32_t, srcDimSize);
    TILING_DATA_FIELD_DEF(uint32_t, idxDimSize);
    TILING_DATA_FIELD_DEF(uint32_t, innerSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherCustom, GatherCustomTilingData)
}

#endif  // GATHER_CUSTOM_TILING_H
```

`op_host/gather_custom.cpp`:
```cpp
#include "gather_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GatherCustomTilingData tiling;

    auto srcShape = context->GetInputShape(0)->GetStorageShape();
    auto idxShape = context->GetInputShape(1)->GetStorageShape();

    // Get dim from attrs
    const auto* attrs = context->GetAttrs();
    int dim = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(0)));
    if (dim < 0) dim += srcShape.GetDimNum();

    // Compute outerSize, srcDimSize, idxDimSize, innerSize
    uint32_t outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= srcShape.GetDim(i);
    }
    uint32_t srcDimSize = srcShape.GetDim(dim);
    uint32_t idxDimSize = idxShape.GetDim(dim);
    uint32_t innerSize = 1;
    for (size_t i = dim + 1; i < srcShape.GetDimNum(); i++) {
        innerSize *= srcShape.GetDim(i);
    }

    tiling.set_outerSize(outerSize);
    tiling.set_srcDimSize(srcDimSize);
    tiling.set_idxDimSize(idxDimSize);
    tiling.set_innerSize(innerSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(1);

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* idxShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *idxShape;
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class GatherCustom : public OpDef {
public:
    explicit GatherCustom(const char* name) : OpDef(name) {
        this->Input("src").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("index").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("dst").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GatherCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gather_custom_impl_npu(
        const at::Tensor& src_in, const at::Tensor& index_in) {
    at::Tensor src = src_in;
    at::Tensor index = index_in;
    at::Tensor result = at::empty_like(index, src.options());
    EXEC_NPU_CMD(aclnnGatherCustom, src, index, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_custom", &gather_custom_impl_npu, "gather_custom operator");
}
```

## Curated Example: Nearest Neighbor Upsample 2D (resize pattern)

This example demonstrates all 3 file components for nearest neighbor upsampling.
For each output pixel, compute the source coordinate via `src = dst / scale_factor`,
then copy the nearest source value.

### OP_KERNEL

`op_kernel/nearest_neighbor_upsample_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelNearestUpsample {
public:
    __aicore__ inline KernelNearestUpsample() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                 uint32_t N, uint32_t C,
                                 uint32_t H_in, uint32_t W_in,
                                 uint32_t H_out, uint32_t W_out) {
        this->N = N; this->C = C;
        this->H_in = H_in; this->W_in = W_in;
        this->H_out = H_out; this->W_out = W_out;

        uint32_t inTotal = N * C * H_in * W_in;
        uint32_t outTotal = N * C * H_out * W_out;
        inputGm.SetGlobalBuffer((__gm__ float*)input, inTotal);
        outputGm.SetGlobalBuffer((__gm__ float*)output, outTotal);

        // Process one output row at a time
        uint32_t alignedW = (W_out + 7) / 8 * 8;
        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedW * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedW * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t c = 0; c < C; c++) {
                for (uint32_t oh = 0; oh < H_out; oh++) {
                    // Compute source row: nearest neighbor
                    uint32_t ih = oh * H_in / H_out;

                    // Build one output row
                    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                    for (uint32_t ow = 0; ow < W_out; ow++) {
                        uint32_t iw = ow * W_in / W_out;
                        uint32_t srcIdx = ((n * C + c) * H_in + ih) * W_in + iw;
                        // Scalar read from GM
                        float val = *((__gm__ float*)inputGm.GetPhyAddr() + srcIdx);
                        outLocal.SetValue(ow, val);
                    }

                    // Write output row
                    uint32_t dstOffset = ((n * C + c) * H_out + oh) * W_out;
                    outQueue.EnQue(outLocal);
                    outLocal = outQueue.DeQue<float>();
                    DataCopy(outputGm[dstOffset], outLocal, W_out);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> inputGm, outputGm;
    uint32_t N, C, H_in, W_in, H_out, W_out;
};

extern "C" __global__ __aicore__ void nearest_neighbor_upsample_custom(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelNearestUpsample op;
    op.Init(input, output, tilingData.N, tilingData.C,
            tilingData.H_in, tilingData.W_in, tilingData.H_out, tilingData.W_out);
    op.Process();
}
```

### OP_HOST

`op_host/nearest_neighbor_upsample_custom_tiling.h`:
```cpp
#ifndef NEAREST_NEIGHBOR_UPSAMPLE_CUSTOM_TILING_H
#define NEAREST_NEIGHBOR_UPSAMPLE_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NearestNeighborUpsampleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C);
    TILING_DATA_FIELD_DEF(uint32_t, H_in);
    TILING_DATA_FIELD_DEF(uint32_t, W_in);
    TILING_DATA_FIELD_DEF(uint32_t, H_out);
    TILING_DATA_FIELD_DEF(uint32_t, W_out);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NearestNeighborUpsampleCustom, NearestNeighborUpsampleCustomTilingData)
}

#endif  // NEAREST_NEIGHBOR_UPSAMPLE_CUSTOM_TILING_H
```

`op_host/nearest_neighbor_upsample_custom.cpp`:
```cpp
#include "nearest_neighbor_upsample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    NearestNeighborUpsampleCustomTilingData tiling;

    auto inShape = context->GetInputShape(0)->GetStorageShape();  // [N, C, H_in, W_in]
    uint32_t N = inShape.GetDim(0);
    uint32_t C = inShape.GetDim(1);
    uint32_t H_in = inShape.GetDim(2);
    uint32_t W_in = inShape.GetDim(3);

    // Get scale_factor from attrs
    const auto* attrs = context->GetAttrs();
    int scale_factor = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(0)));

    uint32_t H_out = H_in * scale_factor;
    uint32_t W_out = W_in * scale_factor;

    tiling.set_N(N); tiling.set_C(C);
    tiling.set_H_in(H_in); tiling.set_W_in(W_in);
    tiling.set_H_out(H_out); tiling.set_W_out(W_out);

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
    const gert::Shape* inShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    // Output shape computed from scale_factor; use input shape as conservative estimate
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class NearestNeighborUpsampleCustom : public OpDef {
public:
    explicit NearestNeighborUpsampleCustom(const char* name) : OpDef(name) {
        this->Input("input").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("scale_factor").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(NearestNeighborUpsampleCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor nearest_neighbor_upsample_custom_impl_npu(
        const at::Tensor& input_in, int64_t scale_factor) {
    at::Tensor input = input_in;
    int64_t N = input.size(0), C = input.size(1);
    int64_t H_in = input.size(2), W_in = input.size(3);
    int64_t H_out = H_in * scale_factor, W_out = W_in * scale_factor;
    at::Tensor result = at::empty({N, C, H_out, W_out}, input.options());
    EXEC_NPU_CMD(aclnnNearestNeighborUpsampleCustom, input, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nearest_neighbor_upsample_custom",
          &nearest_neighbor_upsample_custom_impl_npu,
          "nearest_neighbor_upsample_custom operator");
}
```

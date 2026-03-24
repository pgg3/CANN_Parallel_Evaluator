## Curated Example: Max Pooling 2D (Vector/Pooling Pattern)

This example demonstrates all 3 file components for a 2D max pooling operator using manual sliding window + ReduceMax.

### OP_KERNEL

`op_kernel/max_pooling2d_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelMaxPool2d {
public:
    __aicore__ inline KernelMaxPool2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batch, uint32_t channels,
                                 uint32_t inH, uint32_t inW,
                                 uint32_t outH, uint32_t outW,
                                 uint32_t kH, uint32_t kW,
                                 uint32_t strideH, uint32_t strideW,
                                 uint32_t padH, uint32_t padW) {
        this->batch = batch;
        this->channels = channels;
        this->inH = inH;  this->inW = inW;
        this->outH = outH; this->outW = outW;
        this->kH = kH;  this->kW = kW;
        this->strideH = strideH;  this->strideW = strideW;
        this->padH = padH;  this->padW = padW;

        uint32_t inTotal = batch * channels * inH * inW;
        uint32_t outTotal = batch * channels * outH * outW;
        xGm.SetGlobalBuffer((__gm__ float*)x, inTotal);
        yGm.SetGlobalBuffer((__gm__ float*)y, outTotal);

        // Allocate buffers for one input channel slice and window
        uint32_t inSliceSize = inH * inW;
        uint32_t windowSize = kH * kW;
        // Align to 8 floats (32 bytes)
        windowSize = (windowSize + 7) / 8 * 8;

        pipe.InitBuffer(inQueue, BUFFER_NUM, inSliceSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, outW * sizeof(float));
        pipe.InitBuffer(workBuf, windowSize * sizeof(float));
        pipe.InitBuffer(windowBuf, windowSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t n = 0; n < batch; n++) {
            for (uint32_t c = 0; c < channels; c++) {
                ProcessSlice(n, c);
            }
        }
    }

private:
    __aicore__ inline void ProcessSlice(uint32_t n, uint32_t c) {
        uint32_t inOffset = (n * channels + c) * inH * inW;
        uint32_t outOffset = (n * channels + c) * outH * outW;

        // Load entire input slice to UB
        LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        DataCopy(inLocal, xGm[inOffset], inH * inW);
        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<float>();

        // Process each output row
        for (uint32_t oh = 0; oh < outH; oh++) {
            LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            LocalTensor<float> window = windowBuf.Get<float>();
            LocalTensor<float> work = workBuf.Get<float>();

            for (uint32_t ow = 0; ow < outW; ow++) {
                // Gather window elements
                uint32_t cnt = 0;
                for (uint32_t wh = 0; wh < kH; wh++) {
                    for (uint32_t ww = 0; ww < kW; ww++) {
                        int32_t ih = (int32_t)(oh * strideH + wh) - (int32_t)padH;
                        int32_t iw = (int32_t)(ow * strideW + ww) - (int32_t)padW;
                        if (ih >= 0 && ih < (int32_t)inH && iw >= 0 && iw < (int32_t)inW) {
                            window.SetValue(cnt, inLocal.GetValue(ih * inW + iw));
                        } else {
                            window.SetValue(cnt, -3.4028235e+38f);  // -FLT_MAX for padding
                        }
                        cnt++;
                    }
                }
                // Pad remaining to aligned size
                uint32_t alignedCnt = (cnt + 7) / 8 * 8;
                for (uint32_t p = cnt; p < alignedCnt; p++) {
                    window.SetValue(p, -3.4028235e+38f);
                }
                ReduceMax(work, window, work, alignedCnt);
                outLocal.SetValue(ow, work.GetValue(0));
            }

            // Write output row
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<float>();
            DataCopy(yGm[outOffset + oh * outW], outLocal, outW);
            outQueue.FreeTensor(outLocal);
        }

        inQueue.FreeTensor(inLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> workBuf, windowBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t batch, channels, inH, inW, outH, outW;
    uint32_t kH, kW, strideH, strideW, padH, padW;
};

extern "C" __global__ __aicore__ void max_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelMaxPool2d op;
    op.Init(x, y, tilingData.batch, tilingData.channels,
            tilingData.inH, tilingData.inW, tilingData.outH, tilingData.outW,
            tilingData.kH, tilingData.kW, tilingData.strideH, tilingData.strideW,
            tilingData.padH, tilingData.padW);
    op.Process();
}
```

### OP_HOST

`op_host/max_pooling2d_custom_tiling.h`:
```cpp
#ifndef MAX_POOLING2D_CUSTOM_TILING_H
#define MAX_POOLING2D_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPooling2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inH);
    TILING_DATA_FIELD_DEF(uint32_t, inW);
    TILING_DATA_FIELD_DEF(uint32_t, outH);
    TILING_DATA_FIELD_DEF(uint32_t, outW);
    TILING_DATA_FIELD_DEF(uint32_t, kH);
    TILING_DATA_FIELD_DEF(uint32_t, kW);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, padH);
    TILING_DATA_FIELD_DEF(uint32_t, padW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling2dCustom, MaxPooling2dCustomTilingData)
}

#endif  // MAX_POOLING2D_CUSTOM_TILING_H
```

`op_host/max_pooling2d_custom.cpp`:
```cpp
#include "max_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MaxPooling2dCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t batch = xShape.GetDim(0);
    uint32_t channels = xShape.GetDim(1);
    uint32_t inH = xShape.GetDim(2);
    uint32_t inW = xShape.GetDim(3);

    // Get pooling params from attrs
    const auto* attrs = context->GetAttrs();
    uint32_t kH = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(0)));
    uint32_t kW = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(1)));
    uint32_t strideH = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(2)));
    uint32_t strideW = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(3)));
    uint32_t padH = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(4)));
    uint32_t padW = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(5)));

    uint32_t outH = (inH + 2 * padH - kH) / strideH + 1;
    uint32_t outW = (inW + 2 * padW - kW) / strideW + 1;

    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inH(inH);  tiling.set_inW(inW);
    tiling.set_outH(outH); tiling.set_outW(outW);
    tiling.set_kH(kH);  tiling.set_kW(kW);
    tiling.set_strideH(strideH); tiling.set_strideW(strideW);
    tiling.set_padH(padH); tiling.set_padW(padW);

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
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    // Output shape: [N, C, outH, outW] — computed from input + pooling params
    // Placeholder: same batch and channels, spatial dims set by TilingFunc
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class MaxPooling2dCustom : public OpDef {
public:
    explicit MaxPooling2dCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("kernel_size").Int();
        this->Attr("kernel_w").Int();
        this->Attr("stride").Int();
        this->Attr("stride_w").Int();
        this->Attr("padding").Int();
        this->Attr("padding_w").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxPooling2dCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor max_pooling2d_custom_impl_npu(
        const at::Tensor& x_in,
        int64_t kernel_size, int64_t stride, int64_t padding) {
    at::Tensor x = x_in;
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    // Pooling params from init
    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;
    at::Tensor result = at::empty({N, C, outH, outW}, x.options());
    EXEC_NPU_CMD(aclnnMaxPooling2dCustom, x, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pooling2d_custom", &max_pooling2d_custom_impl_npu, "max_pooling2d_custom operator");
}
```

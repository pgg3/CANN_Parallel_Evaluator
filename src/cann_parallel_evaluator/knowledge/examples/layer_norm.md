## Curated Example: Layer Normalization (Mixed/Normalization Pattern)

This example demonstrates all 3 file components for Layer Normalization using manual Vector operations: mean → variance → normalize → scale → shift.

### OP_KERNEL

`op_kernel/layer_norm_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNorm {
public:
    __aicore__ inline KernelLayerNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                 uint32_t batchSize, uint32_t normSize,
                                 float epsilon) {
        this->batchSize = batchSize;
        this->normSize = normSize;
        this->epsilon = epsilon;

        uint32_t totalLen = batchSize * normSize;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLen);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, normSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, normSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLen);

        pipe.InitBuffer(inQueue, BUFFER_NUM, normSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, normSize * sizeof(float));
        pipe.InitBuffer(workBuf, normSize * sizeof(float));
        pipe.InitBuffer(gammaBuf, normSize * sizeof(float));
        pipe.InitBuffer(betaBuf, normSize * sizeof(float));
    }

    __aicore__ inline void Process() {
        // Pre-load gamma and beta
        LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        LocalTensor<float> betaLocal = betaBuf.Get<float>();
        DataCopy(gammaLocal, gammaGm, normSize);
        DataCopy(betaLocal, betaGm, normSize);
        event_t evtGamma = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtGamma);
        WaitFlag<HardEvent::MTE2_V>(evtGamma);

        for (uint32_t b = 0; b < batchSize; b++) {
            CopyIn(b);
            Compute(b);
            CopyOut(b);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[batch * normSize], normSize);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t batch) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        LocalTensor<float> betaLocal = betaBuf.Get<float>();

        // 1. mean = ReduceSum(x) / normSize
        ReduceSum(work, xLocal, work, normSize);
        float mean = work.GetValue(0) / static_cast<float>(normSize);

        // 2. diff = x - mean
        Adds(yLocal, xLocal, -mean, normSize);

        // 3. var = ReduceSum(diff^2) / normSize
        Mul(work, yLocal, yLocal, normSize);
        ReduceSum(work, work, work, normSize);
        float var = work.GetValue(0) / static_cast<float>(normSize);

        // 4. normalize = diff / sqrt(var + eps)
        //    Use Duplicate + Adds + Rsqrt to compute 1/sqrt(var+eps) vectorized,
        //    then Muls to scale. (sqrtf / math.h do NOT exist in AscendC kernels)
        Duplicate(work, var + epsilon, normSize);
        Rsqrt(work, work, normSize);
        float invStd = work.GetValue(0);
        Muls(yLocal, yLocal, invStd, normSize);

        // 5. output = normalize * gamma + beta
        Mul(yLocal, yLocal, gammaLocal, normSize);
        Add(yLocal, yLocal, betaLocal, normSize);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batch) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        DataCopy(yGm[batch * normSize], yLocal, normSize);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> workBuf, gammaBuf, betaBuf;
    GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    uint32_t batchSize, normSize;
    float epsilon;
};

extern "C" __global__ __aicore__ void layer_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelLayerNorm op;
    op.Init(x, gamma, beta, y, tilingData.batchSize, tilingData.normSize, tilingData.epsilon);
    op.Process();
}
```

### OP_HOST

`op_host/layer_norm_custom_tiling.h`:
```cpp
#ifndef LAYER_NORM_CUSTOM_TILING_H
#define LAYER_NORM_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayerNormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, normSize);
    TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormCustom, LayerNormCustomTilingData)
}

#endif  // LAYER_NORM_CUSTOM_TILING_H
```

`op_host/layer_norm_custom.cpp`:
```cpp
#include "layer_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LayerNormCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetStorageShape();

    // Get normalized_shape (list_int) from attrs
    const auto* attrs = context->GetAttrs();
    auto normalizedShapePtr = attrs->GetAttrPointer<std::vector<int64_t>>(0);

    // Compute normSize = product of normalized_shape elements
    // e.g. normalized_shape=(64, 256, 256) → normSize = 64*256*256 = 4194304
    uint32_t normDims = normalizedShapePtr->size();
    uint32_t normSize = 1;
    for (uint32_t i = 0; i < normDims; i++) {
        normSize *= (*normalizedShapePtr)[i];
    }

    // batchSize = product of remaining (non-normalized) dimensions
    uint32_t batchSize = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - normDims; i++) {
        batchSize *= xShape.GetDim(i);
    }

    float epsilon = 1e-5f;

    tiling.set_batchSize(batchSize);
    tiling.set_normSize(normSize);
    tiling.set_epsilon(epsilon);

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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class LayerNormCustom : public OpDef {
public:
    explicit LayerNormCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("normalized_shape").ListInt();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LayerNormCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor layer_norm_custom_impl_npu(
        const at::Tensor& x_in, const at::Tensor& gamma_in, const at::Tensor& beta_in) {
    at::Tensor x = x_in;
    at::Tensor gamma = gamma_in;
    at::Tensor beta = beta_in;
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLayerNormCustom, x, gamma, beta, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_custom", &layer_norm_custom_impl_npu, "layer_norm_custom operator");
}
```

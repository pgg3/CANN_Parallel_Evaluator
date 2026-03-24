## Curated Example: Softmax (numerically stable, per-row)

This example demonstrates all 3 file components for softmax along the last dimension.
Algorithm: for each row, compute max → subtract max → exp → sum → divide.

### OP_KERNEL

`op_kernel/softmax_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t numRows, uint32_t rowLength,
                                 uint32_t alignedRowLen) {
        this->numRows = numRows;
        this->rowLength = rowLength;
        this->alignedRowLen = alignedRowLen;

        xGm.SetGlobalBuffer((__gm__ float*)x, numRows * rowLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, numRows * rowLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedRowLen * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedRowLen * sizeof(float));
        pipe.InitBuffer(workBuf, alignedRowLen * sizeof(float));
        pipe.InitBuffer(maxBuf, 8 * sizeof(float));  // scalar result buffer
    }

    __aicore__ inline void Process() {
        for (uint32_t row = 0; row < numRows; row++) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        // Zero-pad the aligned region first
        if (alignedRowLen > rowLength) {
            Duplicate(xLocal, 0.0f, alignedRowLen);
        }
        DataCopy(xLocal, xGm[row * rowLength], rowLength);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> work = workBuf.Get<float>();
        LocalTensor<float> maxLocal = maxBuf.Get<float>();

        // Step 1: Find row max for numerical stability
        ReduceMax(maxLocal, xLocal, work, rowLength);
        float rowMax = maxLocal.GetValue(0);

        // Step 2: Subtract max → exp
        Adds(yLocal, xLocal, -rowMax, rowLength);
        Exp(yLocal, yLocal, rowLength);

        // Step 3: Sum of exp values
        ReduceSum(maxLocal, yLocal, work, rowLength);
        float expSum = maxLocal.GetValue(0);

        // Step 4: Divide by sum
        float invSum = 1.0f / expSum;
        Muls(yLocal, yLocal, invSum, rowLength);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        LocalTensor<float> yLocal = outQueue.DeQue<float>();
        DataCopy(yGm[row * rowLength], yLocal, rowLength);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> maxBuf;
    GlobalTensor<float> xGm, yGm;
    uint32_t numRows, rowLength, alignedRowLen;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelSoftmax op;
    op.Init(x, y, tilingData.numRows, tilingData.rowLength, tilingData.alignedRowLen);
    op.Process();
}
```

### OP_HOST

`op_host/softmax_custom_tiling.h`:
```cpp
#ifndef SOFTMAX_CUSTOM_TILING_H
#define SOFTMAX_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, rowLength);
    TILING_DATA_FIELD_DEF(uint32_t, alignedRowLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}

#endif  // SOFTMAX_CUSTOM_TILING_H
```

`op_host/softmax_custom.cpp`:
```cpp
#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    SoftmaxCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t ndim = shape.GetDimNum();

    // Softmax along last dimension
    uint32_t rowLength = shape.GetDim(ndim - 1);
    uint32_t numRows = 1;
    for (uint32_t i = 0; i < ndim - 1; i++) {
        numRows *= shape.GetDim(i);
    }

    // Align row to 8 elements (32 bytes for float)
    uint32_t alignedRowLen = (rowLength + 7) / 8 * 8;

    tiling.set_numRows(numRows);
    tiling.set_rowLength(rowLength);
    tiling.set_alignedRowLen(alignedRowLen);

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
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SoftmaxCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor softmax_custom_impl_npu(const at::Tensor& x_in) {
    at::Tensor x = x_in;
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnSoftmaxCustom, x, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_custom", &softmax_custom_impl_npu, "softmax_custom operator");
}
```

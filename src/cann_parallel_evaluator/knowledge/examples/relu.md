## Complete Example: ReLU Operator (elementwise unary: max(0, x) → y)

### OP_KERNEL

`op_kernel/relu_custom.cpp`:
```cpp
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t tileLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->tailLength = this->blockLength - tileNum * BUFFER_NUM * tileLength;
        this->hasTail = (this->tailLength > 0);

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        if (this->hasTail) {
            CopyIn(loopCount);
            Compute(loopCount);
            CopyOut(loopCount);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t len = (progress < this->tileNum * BUFFER_NUM) ? this->tileLength : this->tailLength;
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        uint32_t len = (progress < this->tileNum * BUFFER_NUM) ? this->tileLength : this->tailLength;
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        Relu(yLocal, xLocal, len);
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t len = (progress < this->tileNum * BUFFER_NUM) ? this->tileLength : this->tailLength;
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum, tileLength, tailLength;
    bool hasTail;
};

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelRelu op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
    op.Process();
}
```

### OP_HOST

`op_host/relu_custom_tiling.h`:
```cpp
#ifndef RELU_CUSTOM_TILING_H
#define RELU_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReluCustom, ReluCustomTilingData)
}

#endif  // RELU_CUSTOM_TILING_H
```

`op_host/relu_custom.cpp`:
```cpp
#include "relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ReluCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        totalLength *= shape.GetDim(i);
    }

    constexpr uint32_t BLOCK_DIM = 8;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t UB_SIZE = 176 * 1024;
    constexpr uint32_t NUM_QUEUES = 2;  // inQueueX + outQueueY

    uint32_t maxTileLength = UB_SIZE / (NUM_QUEUES * BUFFER_NUM * sizeof(float));
    maxTileLength = maxTileLength / 8 * 8;

    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t tileNum = blockLength / (maxTileLength * BUFFER_NUM);
    if (tileNum == 0) tileNum = 1;
    uint32_t tileLength = blockLength / (tileNum * BUFFER_NUM);
    tileLength = tileLength / 8 * 8;

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileLength(tileLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

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

class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReluCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor relu_custom_impl_npu(const at::Tensor& x_in) {
    at::Tensor x = x_in;
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnReluCustom, x, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_custom", &relu_custom_impl_npu, "relu_custom operator");
}
```

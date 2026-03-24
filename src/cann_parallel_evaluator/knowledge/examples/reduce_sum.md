## Complete Example: ReduceSum Operator (reduction: sum over all elements)

### KERNEL_IMPL
```cpp
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelReduceSum {
public:
    __aicore__ inline KernelReduceSum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t tileLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->tailLength = this->blockLength - tileNum * BUFFER_NUM * tileLength;
        this->hasTail = (this->tailLength > 0);
        this->accumulator = 0.0f;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + GetBlockIdx(), 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, sizeof(float) * 8);  // small output buffer
        pipe.InitBuffer(workBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
        }
        if (this->hasTail) {
            CopyIn(loopCount, this->tailLength);
            Compute(this->tailLength);
        }
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> workLocal = workBuf.Get<float>();
        LocalTensor<float> resultLocal = outQueueY.AllocTensor<float>();

        ReduceSum(resultLocal, xLocal, workLocal, len);
        this->accumulator += resultLocal.GetValue(0);

        outQueueY.FreeTensor(resultLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut() {
        LocalTensor<float> resultLocal = outQueueY.AllocTensor<float>();
        resultLocal.SetValue(0, this->accumulator);
        DataCopy(yGm[0], resultLocal, 1);
        outQueueY.FreeTensor(resultLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> workBuf;
    GlobalTensor<float> xGm, yGm;
    float accumulator;
    uint32_t blockLength, tileNum, tileLength, tailLength;
    bool hasTail;
};
```

### KERNEL_ENTRY_BODY
```cpp
KernelReduceSum op;
op.Init(x, y, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
op.Process();
```

### TILING_FIELDS
```
uint32_t totalLength
uint32_t tileNum
uint32_t tileLength
```

### TILING_FUNC_BODY
```cpp
ReduceSumCustomTilingData tiling;

auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {
    totalLength *= shape.GetDim(i);
}

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 176 * 1024;
// inQueueX(2 bufs) + outQueueY(1 small) + workBuf(1)
// Approximate: 2*BUFFER_NUM + 1 + 1 = 6 buffers of tileLength
constexpr uint32_t NUM_BUFFERS = 6;

uint32_t maxTileLength = UB_SIZE / (NUM_BUFFERS * sizeof(float));
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
```

### INFER_SHAPE_BODY
```cpp
gert::Shape* y_shape = context->GetOutputShape(0);
y_shape->SetDimNum(1);
y_shape->SetDim(0, 1);
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::zeros({1}, x.options());
```

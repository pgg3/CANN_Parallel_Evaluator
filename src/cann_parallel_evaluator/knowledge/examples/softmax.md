## Curated Example: Softmax (numerically stable, per-row)

This example demonstrates all 6 components for softmax along the last dimension.
Algorithm: for each row, compute max → subtract max → exp → sum → divide.

### KERNEL_IMPL
```cpp
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
```

### KERNEL_ENTRY_BODY
```cpp
KernelSoftmax op;
op.Init(x, y, tilingData.numRows, tilingData.rowLength, tilingData.alignedRowLen);
op.Process();
```

### TILING_FIELDS
```
uint32_t numRows
uint32_t rowLength
uint32_t alignedRowLen
```

### TILING_FUNC_BODY
```cpp
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
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* inShape = context->GetInputShape(0);
gert::Shape* outShape = context->GetOutputShape(0);
*outShape = *inShape;
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::empty_like(x);
```

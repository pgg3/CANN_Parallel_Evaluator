## Curated Example: Layer Normalization (Mixed/Normalization Pattern)

This example demonstrates all 6 components for Layer Normalization using manual Vector operations: mean → variance → normalize → scale → shift.

### KERNEL_IMPL
```cpp
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
```

### KERNEL_ENTRY_BODY
```cpp
KernelLayerNorm op;
op.Init(x, gamma, beta, y, tilingData.batchSize, tilingData.normSize, tilingData.epsilon);
op.Process();
```

### TILING_FIELDS
```
uint32_t batchSize
uint32_t normSize
float epsilon
```

### TILING_FUNC_BODY
```cpp
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
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::empty_like(x);
```

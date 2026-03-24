## Curated Example: Scaled Dot-Product Attention (Cube+Vector Mixed Pattern)

This example demonstrates all 6 components for `Attention(Q,K,V) = softmax(Q×K^T / sqrt(d_k)) × V`.
It uses the Cube unit (Matmul<>) for Q×K^T and scores×V, and the Vector unit for scale + softmax.

### KERNEL_IMPL
```cpp
#include "lib/matmul_intf.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAttention {
public:
    __aicore__ inline KernelAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                 GM_ADDR workspace,
                                 TCubeTiling& qkTiling,
                                 TCubeTiling& svTiling,
                                 uint32_t seqLen, uint32_t headDim,
                                 float scale) {
        this->seqLen = seqLen;
        this->headDim = headDim;
        this->scale = scale;

        qGm.SetGlobalBuffer((__gm__ half*)q, seqLen * headDim);
        kGm.SetGlobalBuffer((__gm__ half*)k, seqLen * headDim);
        vGm.SetGlobalBuffer((__gm__ half*)v, seqLen * headDim);
        // scores and output are float32 (Cube accumulation type)
        scoresGm.SetGlobalBuffer((__gm__ float*)workspace, seqLen * seqLen);
        outGm.SetGlobalBuffer((__gm__ float*)out, seqLen * headDim);

        mmQK.Init(&qkTiling, &pipe);  // Init takes TCubeTiling* pointer
        mmSV.Init(&svTiling, &pipe);

        // Vector buffers for softmax (one row at a time)
        uint32_t rowLen = (seqLen + 7) / 8 * 8;  // align to 32 bytes
        pipe.InitBuffer(vecQueue, BUFFER_NUM, rowLen * sizeof(float));
        pipe.InitBuffer(workBuf, rowLen * sizeof(float));
    }

    __aicore__ inline void Process() {
        // Step 1: scores = Q × K^T  (Cube, float32 output)
        mmQK.SetTensorA(qGm);
        mmQK.SetTensorB(kGm);
        mmQK.IterateAll(scoresGm);
        mmQK.End();

        // Step 2-4: scale + softmax (Vector, row by row)
        for (uint32_t row = 0; row < seqLen; row++) {
            SoftmaxRow(row);
        }

        // Step 5: output = softmax_scores × V  (Cube)
        // scores is [seqLen, seqLen] float32 → need to cast to half for Matmul
        // For simplicity, use float32 Matmul (cType=float)
        mmSV.SetTensorA(scoresGm);
        mmSV.SetTensorB(vGm);
        mmSV.IterateAll(outGm);
        mmSV.End();
    }

private:
    __aicore__ inline void SoftmaxRow(uint32_t row) {
        uint32_t offset = row * seqLen;
        LocalTensor<float> rowLocal = vecQueue.AllocTensor<float>();
        LocalTensor<float> work = workBuf.Get<float>();

        // Load one row of scores
        DataCopy(rowLocal, scoresGm[offset], seqLen);
        event_t evt1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evt1);
        WaitFlag<HardEvent::MTE2_V>(evt1);

        // Scale: scores / sqrt(d_k)
        Muls(rowLocal, rowLocal, scale, seqLen);

        // Softmax: max → sub → exp → sum → div
        ReduceMax(work, rowLocal, work, seqLen);
        float maxVal = work.GetValue(0);
        Adds(rowLocal, rowLocal, -maxVal, seqLen);
        Exp(rowLocal, rowLocal, seqLen);
        ReduceSum(work, rowLocal, work, seqLen);
        float sumVal = work.GetValue(0);
        Muls(rowLocal, rowLocal, 1.0f / sumVal, seqLen);

        // Write back
        event_t evt2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evt2);
        WaitFlag<HardEvent::V_MTE3>(evt2);
        DataCopy(scoresGm[offset], rowLocal, seqLen);
        vecQueue.FreeTensor(rowLocal);
    }

private:
    TPipe pipe;
    // Two Matmul instances: Q×K^T and scores×V
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::NT, half>,   // K^T: transposed B
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mmQK;
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,  // scores are float
           MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mmSV;
    GlobalTensor<half> qGm, kGm, vGm;
    GlobalTensor<float> scoresGm, outGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> vecQueue;
    TBuf<QuePosition::VECCALC> workBuf;
    uint32_t seqLen, headDim;
    float scale;
};
```

### KERNEL_ENTRY_BODY
```cpp
KernelAttention op;
op.Init(q, k, v, out, workspace,
        tilingData.qkTiling, tilingData.svTiling,
        tilingData.seqLen, tilingData.headDim, tilingData.scale);
op.Process();
```

### TILING_FIELDS
```
struct TCubeTiling qkTiling
struct TCubeTiling svTiling
uint32_t seqLen
uint32_t headDim
float scale
```

### TILING_FUNC_BODY
```cpp
ScaledDotProductAttentionCustomTilingData tiling;

auto qShape = context->GetInputShape(0)->GetStorageShape();
uint32_t seqLen = qShape.GetDim(0);
uint32_t headDim = qShape.GetDim(1);
float scale = 1.0f / sqrtf(static_cast<float>(headDim));

tiling.set_seqLen(seqLen);
tiling.set_headDim(headDim);
tiling.set_scale(scale);

auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

// QK tiling: [seqLen, headDim] × [headDim, seqLen] → [seqLen, seqLen]
matmul_tiling::MatmulApiTiling qkMm(platform);
qkMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
qkMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
qkMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
qkMm.SetShape(seqLen, seqLen, headDim);  // M, N, K
qkMm.SetFixSplit(-1, -1, -1);
TCubeTiling qkTiling;
int64_t qkWsSize = qkMm.GetTiling(qkTiling);
tiling.qkTiling = qkTiling;  // struct fields: direct assign (no set_ method)

// SV tiling: [seqLen, seqLen] × [seqLen, headDim] → [seqLen, headDim]
matmul_tiling::MatmulApiTiling svMm(platform);
svMm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   // scores are float32 after softmax
svMm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
svMm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
svMm.SetShape(seqLen, headDim, seqLen);  // M, N, K
svMm.SetFixSplit(-1, -1, -1);
TCubeTiling svTiling;
int64_t svWsSize = svMm.GetTiling(svTiling);
tiling.svTiling = svTiling;  // struct fields: direct assign (no set_ method)

tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(1);

// Workspace: matmul workspaces + scores buffer
size_t* ws = context->GetWorkspaceSizes(1);
ws[0] = static_cast<size_t>(qkWsSize) + static_cast<size_t>(svWsSize)
      + seqLen * seqLen * sizeof(float);  // scores intermediate buffer

return ge::GRAPH_SUCCESS;
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* qShape = context->GetInputShape(0);
gert::Shape* outShape = context->GetOutputShape(0);
*outShape = *qShape;  // output has same shape as Q: [seqLen, headDim]
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
int64_t seqLen = q.size(0);
int64_t headDim = q.size(1);
at::Tensor result = at::empty({seqLen, headDim}, q.options().dtype(at::kFloat));
```

### KERNEL_INCLUDES
```cpp
#include "lib/matmul_intf.h"
```

### TILING_INCLUDES
```cpp
#include "lib/matmul/matmul_tiling.h"
```

### TILING_FUNC_INCLUDES
```cpp
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"
#include <cmath>
```

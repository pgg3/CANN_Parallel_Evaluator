## Curated Example: Gather (Mixed/Index Pattern)

This example demonstrates all 6 components for a gather operator that collects elements from a source tensor along a given dimension using an index tensor.

### KERNEL_IMPL
```cpp
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
```

### KERNEL_ENTRY_BODY
```cpp
KernelGather op;
op.Init(src, index, dst, tilingData.outerSize, tilingData.srcDimSize,
        tilingData.idxDimSize, tilingData.innerSize);
op.Process();
```

### TILING_FIELDS
```
uint32_t outerSize
uint32_t srcDimSize
uint32_t idxDimSize
uint32_t innerSize
```

### TILING_FUNC_BODY
```cpp
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
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* idxShape = context->GetInputShape(1);
gert::Shape* outShape = context->GetOutputShape(0);
*outShape = *idxShape;
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::empty_like(index, src.options());
```

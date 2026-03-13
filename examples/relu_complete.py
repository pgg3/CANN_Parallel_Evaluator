"""完整示例：ReLU 算子评估

使用真实的 LLM 生成组件，展示从构建 task 到获取评估结果的完整流程。
运行前需要昇腾 NPU 环境 + CANN Toolkit + torch-npu。
"""

from pathlib import Path

from evotoolkit.core import Solution
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig

# ──────────────────────────────────────────────
# 1. Python Reference（org 格式）
# ──────────────────────────────────────────────
PYTHON_REFERENCE = Path(__file__).with_name("relu_org.py").read_text()

# ──────────────────────────────────────────────
# 2. LLM 生成的 6 个组件
# ──────────────────────────────────────────────

KERNEL_IMPL = r"""
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR output,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t tileLength) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->tailLength = this->blockLength - tileNum * BUFFER_NUM * tileLength;
        this->hasTail = (this->tailLength > 0);

        uint32_t offset = this->blockLength * GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ float*)x + offset, this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float*)output + offset, this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        if (this->hasTail) {
            CopyInTail();
            ComputeTail();
            CopyOutTail();
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        Relu(outLocal, xLocal, this->tileLength);
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyInTail() {
        uint32_t offset = this->tileNum * BUFFER_NUM * this->tileLength;
        LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], this->tailLength);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void ComputeTail() {
        LocalTensor<float> xLocal = inQueue.DeQue<float>();
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        Relu(outLocal, xLocal, this->tailLength);
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutTail() {
        uint32_t offset = this->tileNum * BUFFER_NUM * this->tileLength;
        LocalTensor<float> outLocal = outQueue.DeQue<float>();
        DataCopy(outGm[offset], outLocal, this->tailLength);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm, outGm;
    uint32_t blockLength, tileNum, tileLength, tailLength;
    bool hasTail;
};
"""

KERNEL_ENTRY_BODY = r"""
KernelRelu op;
op.Init(x, output, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
op.Process();
"""

TILING_FIELDS = [
    {"name": "totalLength", "type": "uint32_t"},
    {"name": "tileNum", "type": "uint32_t"},
    {"name": "tileLength", "type": "uint32_t"},
]

TILING_FUNC_BODY = r"""
ReluCustomTilingData tiling;

auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {
    totalLength *= shape.GetDim(i);
}

constexpr uint32_t BLOCK_DIM = 24;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 128 * 1024;   // 128KB usable UB
constexpr uint32_t NUM_QUEUES = 2;          // 1 input + 1 output

uint32_t maxTileLength = UB_SIZE / (NUM_QUEUES * BUFFER_NUM * sizeof(float));
maxTileLength = maxTileLength / 8 * 8;     // 32-byte alignment

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
"""

INFER_SHAPE_BODY = r"""
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
"""

OUTPUT_ALLOC_CODE = "at::Tensor result = at::empty_like(x);"

# ──────────────────────────────────────────────
# 3. 构建任务并评估
# ──────────────────────────────────────────────

task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": PYTHON_REFERENCE,
})

config = CANNSolutionConfig(
    kernel_impl=KERNEL_IMPL,
    kernel_entry_body=KERNEL_ENTRY_BODY,
    tiling_fields=TILING_FIELDS,
    tiling_func_body=TILING_FUNC_BODY,
    infer_shape_body=INFER_SHAPE_BODY,
    output_alloc_code=OUTPUT_ALLOC_CODE,
)

result = task.evaluate_solution(Solution("", config.to_dict()))

# ──────────────────────────────────────────────
# 4. 输出结果
# ──────────────────────────────────────────────

if result.valid:
    info = result.additional_info
    print(f"[PASS] Runtime: {info['runtime']:.4f} ms "
          f"(baseline: {info['baseline_runtime']:.4f} ms, "
          f"speedup: {info['speedup']:.3f}x)")
else:
    info = result.additional_info
    print(f"[FAIL] Stage: {info['stage']}, Error: {info['error']}")

#include "kernel_operator.h"

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

extern "C" __global__ __aicore__ void relu_custom(GM_ADDR x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
KernelRelu op;
op.Init(x, output, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
op.Process();
}

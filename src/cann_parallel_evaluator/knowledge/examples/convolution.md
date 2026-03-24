## Curated Example: Standard 2D Convolution (im2col + Matmul Pattern)

This example demonstrates all 3 file components for conv2d with square input and square kernel,
decomposed as im2col (rearrange input into column matrix) + Matmul on the Cube unit.

```
Input [N,C_in,H,W] → im2col → ColMatrix [N*H_out*W_out, C_in*kH*kW]
Weight [C_out, C_in, kH, kW] → reshape → [C_out, C_in*kH*kW]
Output = ColMatrix × Weight^T → reshape → [N, C_out, H_out, W_out]
```

### OP_KERNEL

`op_kernel/conv_standard2d_square_input_square_kernel_custom.cpp`:
```cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelConv2d {
public:
    __aicore__ inline KernelConv2d() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                 GM_ADDR workspace,
                                 TCubeTiling& cubeTiling,
                                 uint32_t N, uint32_t C_in, uint32_t H, uint32_t W,
                                 uint32_t C_out, uint32_t kH, uint32_t kW,
                                 uint32_t stride, uint32_t pad,
                                 uint32_t H_out, uint32_t W_out) {
        this->N = N; this->C_in = C_in; this->H = H; this->W = W;
        this->C_out = C_out; this->kH = kH; this->kW = kW;
        this->stride = stride; this->pad = pad;
        this->H_out = H_out; this->W_out = W_out;

        uint32_t M = N * H_out * W_out;
        uint32_t K = C_in * kH * kW;

        inputGm.SetGlobalBuffer((__gm__ half*)input, N * C_in * H * W);
        weightGm.SetGlobalBuffer((__gm__ half*)weight, C_out * K);
        outputGm.SetGlobalBuffer((__gm__ float*)output, M * C_out);
        // im2col buffer in workspace
        colGm.SetGlobalBuffer((__gm__ half*)workspace, M * K);

        mm.Init(&cubeTiling, &pipe);  // Init takes TCubeTiling* pointer

        // UB buffer for im2col: one output row at a time = K elements
        uint32_t rowLen = (K + 15) / 16 * 16;  // align to 32 bytes for half
        pipe.InitBuffer(colQueue, BUFFER_NUM, rowLen * sizeof(half));
    }

    __aicore__ inline void Process() {
        // Step 1: im2col — rearrange input into column matrix in GM
        Im2Col();

        // Step 2: Matmul — ColMatrix × Weight^T
        mm.SetTensorA(colGm);    // [M, K]
        mm.SetTensorB(weightGm); // [C_out, K], transposed via CubeFormat::NT
        mm.IterateAll(outputGm); // [M, C_out]
        mm.End();
    }

private:
    __aicore__ inline void Im2Col() {
        uint32_t K = C_in * kH * kW;
        uint32_t rowLen = (K + 15) / 16 * 16;

        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t oh = 0; oh < H_out; oh++) {
                for (uint32_t ow = 0; ow < W_out; ow++) {
                    uint32_t colRow = (n * H_out + oh) * W_out + ow;

                    LocalTensor<half> colLocal = colQueue.AllocTensor<half>();

                    // Fill one im2col row: iterate over C_in × kH × kW
                    uint32_t idx = 0;
                    for (uint32_t c = 0; c < C_in; c++) {
                        for (uint32_t fh = 0; fh < kH; fh++) {
                            for (uint32_t fw = 0; fw < kW; fw++) {
                                int32_t ih = (int32_t)(oh * stride + fh) - (int32_t)pad;
                                int32_t iw = (int32_t)(ow * stride + fw) - (int32_t)pad;
                                half val = (half)0.0f;
                                if (ih >= 0 && ih < (int32_t)H && iw >= 0 && iw < (int32_t)W) {
                                    uint32_t srcIdx = ((n * C_in + c) * H + ih) * W + iw;
                                    // Read single element from GM via scalar path
                                    val = *((__gm__ half*)inputGm.GetPhyAddr() + srcIdx);
                                }
                                colLocal.SetValue(idx, val);
                                idx++;
                            }
                        }
                    }
                    // Pad to aligned length
                    for (uint32_t p = idx; p < rowLen; p++) {
                        colLocal.SetValue(p, (half)0.0f);
                    }

                    // Write to colGm
                    colQueue.EnQue(colLocal);
                    colLocal = colQueue.DeQue<half>();
                    DataCopy(colGm[colRow * K], colLocal, K);
                    colQueue.FreeTensor(colLocal);
                }
            }
        }
    }

private:
    TPipe pipe;
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>,
           MatmulType<TPosition::GM, CubeFormat::NT, half>,   // Weight^T
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;
    GlobalTensor<half> inputGm, weightGm, colGm;
    GlobalTensor<float> outputGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> colQueue;
    uint32_t N, C_in, H, W, C_out, kH, kW, stride, pad, H_out, W_out;
};

extern "C" __global__ __aicore__ void conv_standard2d_square_input_square_kernel_custom(GM_ADDR input, GM_ADDR weight, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelConv2d op;
    op.Init(input, weight, output, workspace,
            tilingData.cubeTiling,
            tilingData.N, tilingData.C_in, tilingData.H, tilingData.W,
            tilingData.C_out, tilingData.kH, tilingData.kW,
            tilingData.stride, tilingData.pad,
            tilingData.H_out, tilingData.W_out);
    op.Process();
}
```

### OP_HOST

`op_host/conv_standard2d_square_input_square_kernel_custom_tiling.h`:
```cpp
#ifndef CONV_STANDARD2D_SQUARE_INPUT_SQUARE_KERNEL_CUSTOM_TILING_H
#define CONV_STANDARD2D_SQUARE_INPUT_SQUARE_KERNEL_CUSTOM_TILING_H

#include "register/tilingdata_base.h"
#include "lib/matmul/matmul_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard2dSquareInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, C_in);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, W);
    TILING_DATA_FIELD_DEF(uint32_t, C_out);
    TILING_DATA_FIELD_DEF(uint32_t, kH);
    TILING_DATA_FIELD_DEF(uint32_t, kW);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, pad);
    TILING_DATA_FIELD_DEF(uint32_t, H_out);
    TILING_DATA_FIELD_DEF(uint32_t, W_out);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard2dSquareInputSquareKernelCustom, ConvStandard2dSquareInputSquareKernelCustomTilingData)
}

#endif  // CONV_STANDARD2D_SQUARE_INPUT_SQUARE_KERNEL_CUSTOM_TILING_H
```

`op_host/conv_standard2d_square_input_square_kernel_custom.cpp`:
```cpp
#include "conv_standard2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lib/matmul/matmul_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ConvStandard2dSquareInputSquareKernelCustomTilingData tiling;

    auto inShape = context->GetInputShape(0)->GetStorageShape();  // [N, C_in, H, W]
    auto wShape = context->GetInputShape(1)->GetStorageShape();    // [C_out, C_in, kH, kW]

    uint32_t N = inShape.GetDim(0);
    uint32_t C_in = inShape.GetDim(1);
    uint32_t H = inShape.GetDim(2);
    uint32_t W = inShape.GetDim(3);
    uint32_t C_out = wShape.GetDim(0);
    uint32_t kH = wShape.GetDim(2);
    uint32_t kW = wShape.GetDim(3);

    // Get stride and padding from attrs
    const auto* attrs = context->GetAttrs();
    uint32_t stride = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(0)));
    uint32_t pad = *(reinterpret_cast<const int*>(attrs->GetAttrPointer(1)));

    uint32_t H_out = (H + 2 * pad - kH) / stride + 1;
    uint32_t W_out = (W + 2 * pad - kW) / stride + 1;

    uint32_t M = N * H_out * W_out;
    uint32_t K = C_in * kH * kW;

    tiling.set_N(N); tiling.set_C_in(C_in);
    tiling.set_H(H); tiling.set_W(W);
    tiling.set_C_out(C_out);
    tiling.set_kH(kH); tiling.set_kW(kW);
    tiling.set_stride(stride); tiling.set_pad(pad);
    tiling.set_H_out(H_out); tiling.set_W_out(W_out);

    // Cube tiling for ColMatrix[M, K] × Weight^T[K, C_out] → Output[M, C_out]
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling matmulTiling(platform);
    matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    matmulTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    matmulTiling.SetShape(M, C_out, K);  // parameter order: M, N, K
    matmulTiling.SetFixSplit(-1, -1, -1);

    TCubeTiling cubeTiling;
    int64_t matmulWsSize = matmulTiling.GetTiling(cubeTiling);  // returns workspace size
    tiling.cubeTiling = cubeTiling;  // struct fields: direct assign (no set_ method)

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(1);

    // Workspace: im2col buffer + matmul workspace
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = M * K * sizeof(uint16_t) + static_cast<size_t>(matmulWsSize);

    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* inShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(4);
    outShape->SetDim(0, inShape->GetDim(0));   // N
    outShape->SetDim(1, wShape->GetDim(0));    // C_out
    // H_out, W_out: conservative upper bound (actual values computed in TilingFunc)
    int64_t H = inShape->GetDim(2), W = inShape->GetDim(3);
    int64_t kH = wShape->GetDim(2), kW = wShape->GetDim(3);
    outShape->SetDim(2, H - kH + 1);  // stride=1, pad=0 default
    outShape->SetDim(3, W - kW + 1);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class ConvStandard2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputSquareKernelCustom(const char* name) : OpDef(name) {
        this->Input("input").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("stride").Int();
        this->Attr("padding").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvStandard2dSquareInputSquareKernelCustom);

}
```

### PYBINDING

`CppExtension/csrc/op.cpp`:
```cpp
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_standard2d_square_input_square_kernel_custom_impl_npu(
        const at::Tensor& input_in, const at::Tensor& weight_in,
        int64_t stride, int64_t padding) {
    at::Tensor input = input_in.to(at::kHalf);
    at::Tensor weight = weight_in.to(at::kHalf);
    int64_t N = input.size(0);
    int64_t C_out = weight.size(0);
    int64_t H = input.size(2), W = input.size(3);
    int64_t kH = weight.size(2), kW = weight.size(3);
    // stride and padding from init params
    int64_t H_out = (H + 2 * padding - kH) / stride + 1;
    int64_t W_out = (W + 2 * padding - kW) / stride + 1;
    at::Tensor result = at::empty({N, C_out, H_out, W_out}, input.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnConvStandard2dSquareInputSquareKernelCustom, input, weight, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_standard2d_square_input_square_kernel_custom",
          &conv_standard2d_square_input_square_kernel_custom_impl_npu,
          "conv_standard2d_square_input_square_kernel_custom operator");
}
```

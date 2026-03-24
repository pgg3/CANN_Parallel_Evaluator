#include "relu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {

/**
 * TilingFunc - Compute tiling parameters for the operator.
 *
 * Steps:
 *   1. Get input shape from context
 *   2. Compute tiling parameters (totalLength, tileNum, etc.)
 *   3. Set blockDim via context->SetBlockDim()
 *   4. Serialize TilingData to buffer
 *   5. (Optional) Set workspace size
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
ReluCustomTilingData tiling;

auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {
    totalLength *= shape.GetDim(i);
}

constexpr uint32_t BLOCK_DIM = 24;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 128 * 1024;
constexpr uint32_t NUM_QUEUES = 2;

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

/**
 * InferShape - Infer output shape from input shapes.
 */
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class ReluCustom : public OpDef {
public:
    explicit ReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReluCustom);

}

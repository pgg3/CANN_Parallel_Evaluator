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

KernelRelu op;
op.Init(x, output, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
op.Process();

## Multi-Dimensional Tiling Fundamentals

Operators that work on multi-dimensional tensors (normalization, pooling, index, resize) need tiling strategies that account for batch, channel, and spatial dimensions.

### General Strategy
For a tensor with shape [N, C, H, W] (or [N, C, D, H, W] for 3D):
1. **Outer loop**: iterate over batch (N) and channel (C) dimensions
2. **Inner tiling**: tile the spatial dimensions (H, W) to fit in UB

### Shape Passing
Pass all relevant dimensions from TilingFunc to the kernel:
```cpp
// TILING_FIELDS:
uint32_t batchSize       // N
uint32_t channels        // C
uint32_t height          // H
uint32_t width           // W
uint32_t tileHeight      // how many rows per tile
uint32_t tileWidth       // how many cols per tile (or full width)
```

### Nested Loop Pattern
```cpp
// Kernel Process():
for (uint32_t n = 0; n < batchSize; n++) {
    for (uint32_t c = 0; c < channels; c++) {
        uint32_t baseOffset = (n * channels + c) * height * width;
        // Process one (N,C) slice, tiled along spatial dims
        for (uint32_t hTile = 0; hTile < height; hTile += tileHeight) {
            uint32_t curH = min(tileHeight, height - hTile);
            // CopyIn: DataCopy(localBuf, gmPtr[baseOffset + hTile*width], curH * width)
            // Compute: apply operation on the tile
            // CopyOut: DataCopy(gmOut[...], localBuf, curH * width)
        }
    }
}
```

### UB Budget for Multi-Dim
```
tileElements = tileHeight × width (or tileHeight × tileWidth)
Total UB = num_buffers × tileElements × sizeof(dtype) ≤ 176KB
```
Solve for tileHeight: `tileHeight = 176KB / (num_buffers × width × sizeof(dtype))`

### Per-Operator Strategies
- **Normalization**: Reduce along the normalization axis; other dims are the "batch" outer loop
- **Pooling**: Output spatial dims differ from input; tile output space, gather input window
- **Index ops**: Tile along the indexed dimension; other dims form the outer loop
- **Resize**: Tile output spatial dims; map back to input coordinates

### Key Rules
- Always pass shape dimensions through TilingData (never hardcode)
- Handle the case where a full row/column doesn't fit in UB
- For non-contiguous data (e.g., gather along dim 1), use strided DataCopy or manual offset
- Multi-core: divide the outermost dimension (batch or channel) across cores via `GetBlockIdx()`

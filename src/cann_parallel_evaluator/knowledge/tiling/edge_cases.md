## Tiling Edge Case Handling

When `blockLength` is not perfectly divisible by `tileNum * BUFFER_NUM * tileLength`, there are leftover elements (tail).
The tileLength is determined by UB capacity (see Tiling Fundamentals), NOT by dividing blockLength.

```cpp
// In Init(): tileLength is passed from TilingFunc (already UB-safe)
this->tileLength = tileLength;
uint32_t totalTiles = tileNum * BUFFER_NUM;
this->tailLength = this->blockLength - totalTiles * this->tileLength;
this->hasTail = (this->tailLength > 0);

// In Process(): handle main loop + tail
int32_t loopCount = this->tileNum * BUFFER_NUM;
for (int32_t i = 0; i < loopCount; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
if (this->hasTail) {
    // Process remaining elements with tailLength
    // Use the SAME buffers (they are large enough since tailLength < tileLength)
}
```

**Key rules**:
1. `tileLength` must fit in UB — derive it from UB capacity, not from data size.
2. `tailLength` is always < `tileLength`, so the same buffers can be reused.
3. DataCopy length must match actual data size to avoid out-of-bounds access.
4. `tileLength` should be aligned to 32 bytes (8 floats / 16 halfs) for optimal DMA performance.

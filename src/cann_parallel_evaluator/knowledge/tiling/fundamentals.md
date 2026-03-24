## Tiling Fundamentals

### Why Tiling is Needed
Each AI Core has a small **Unified Buffer (UB)** — a fast on-chip SRAM used for all local computation.
All buffers allocated via `pipe.InitBuffer()` must fit within the **usable UB budget** (see Hardware Constraints above).

For large tensors (millions of elements), you CANNOT load everything into UB at once.
**Tiling** splits the data into small chunks ("tiles") that fit in UB, and processes them one at a time in a loop.

### Memory Layout
```
Global Memory (HBM, GBs)  →  DataCopy  →  UB (on-chip)  →  Vector Compute  →  DataCopy  →  Global Memory
```

### UB Budget Calculation
**CRITICAL**: You must ensure all buffers fit in usable UB. The formula:

```
Total UB used = num_queues × BUFFER_NUM × tileLength × sizeof(dtype)
```

For example, if your kernel has 1 input queue + 1 output queue (2 queues), BUFFER_NUM=2, float32:
```
UB used = 2 queues × 2 buffers × tileLength × 4 bytes = 16 × tileLength bytes
```

Use the **usable UB budget** from the Hardware Constraints section above to compute `UB_SIZE`.

### How to Calculate tileLength in TILING_FUNC_BODY

**Step 1**: Determine how many buffers your kernel needs (count all `pipe.InitBuffer` calls).
**Step 2**: Calculate the maximum tileLength that fits in UB:
```cpp
// UB_SIZE: use the usable UB budget from Hardware Constraints
constexpr uint32_t NUM_QUEUES = 2;         // inQueueX + outQueueY
constexpr uint32_t BUFFER_NUM = 2;
uint32_t maxTileLength = UB_SIZE / (NUM_QUEUES * BUFFER_NUM * sizeof(float));
```

**Step 3**: Align tileLength to 32B boundary (8 floats) for optimal DMA:
```cpp
maxTileLength = maxTileLength / 8 * 8;  // Align down to multiple of 8
```
**Step 4**: Calculate tileNum based on blockLength:
```cpp
// BLOCK_DIM: must be one of the valid values from Hardware Constraints
uint32_t blockLength = totalLength / BLOCK_DIM;
uint32_t tileNum = blockLength / (maxTileLength * BUFFER_NUM);
if (tileNum == 0) tileNum = 1;
// Actual tileLength may be smaller for small tensors
uint32_t tileLength = blockLength / (tileNum * BUFFER_NUM);
tileLength = tileLength / 8 * 8;  // Align to 32 bytes
```

### Common Mistake — UB Overflow
```cpp
// ❌ WRONG: tileLength derived from data size, NOT from UB capacity!
this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
// For relu (4096×393216), this gives tileLength = 12.5M floats >> UB!

// ✅ CORRECT: tileLength derived from UB capacity
uint32_t maxTileLength = UB_SIZE / (NUM_QUEUES * BUFFER_NUM * sizeof(float));
// Then calculate tileNum = blockLength / (maxTileLength * BUFFER_NUM)
```

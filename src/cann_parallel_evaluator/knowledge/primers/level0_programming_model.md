## AscendC Programming Model

### Vector vs Cube Operators
- **Vector operators** use the Vector unit for element-wise, reduction, and data-movement operations. Data flows through UB (Unified Buffer) and is processed with vectorized APIs like Add, Mul, Exp, ReduceSum.
- **Cube operators** use the Cube unit for matrix multiplication (GEMM). Data flows through L1/L0 buffers with special tiling requirements.

Most custom operators are Vector operators.

### Pipe Model: CopyIn → Compute → CopyOut
AscendC uses a 3-stage pipeline:
1. **CopyIn**: DMA transfer from Global Memory (GM) → UB (local memory)
2. **Compute**: Vector/Cube computation on UB data
3. **CopyOut**: DMA transfer from UB → GM

These stages run concurrently via double-buffering (BUFFER_NUM=2), so while one tile is being computed, the next tile is being loaded.

### Memory Hierarchy
```
Global Memory (HBM, GBs)  →  DataCopy  →  UB (256KB per core)  →  Vector Compute  →  DataCopy  →  GM
```

- **GM (Global Memory)**: Large but slow. Accessed via `GlobalTensor<T>`.
- **UB (Unified Buffer)**: ~256KB per AI Core, fast on-chip SRAM. Accessed via `LocalTensor<T>`.
- Data moves between GM and UB using `DataCopy()`.

### Buffer Management: TQue and TBuf
- `TQue<QuePosition, BUFFER_NUM>`: Queue-based buffer for pipelined data flow. Supports `AllocTensor`, `EnQue`, `DeQue`, `FreeTensor`.
- `TBuf<QuePosition>`: Simple scratch buffer (no queue semantics). Supports `Get<T>()` for a single allocation.
- Queue positions: `VECIN` (input), `VECOUT` (output), `VECCALC` (temporary/intermediate).

### Operator Lifecycle
```cpp
class KernelOp {
    void Init(GM_ADDR ..., tiling params) {
        // 1. Set up GlobalTensor pointers
        // 2. Allocate UB buffers via pipe.InitBuffer()
    }
    void Process() {
        // 3. Loop over tiles: CopyIn → Compute → CopyOut
        // 4. Handle tail (remaining elements)
    }
};
```

### Tiling: Why and How
UB is only ~256KB. For large tensors (millions of elements), you must split data into small "tiles" that fit in UB and process them in a loop.

**Key formula**:
```
Total UB used = num_queues × BUFFER_NUM × tileLength × sizeof(dtype)
```

Use ~176KB budget (256KB minus ~80KB for TPipe/stack/system). The host-side TilingFunc calculates tileLength and tileNum, then passes them to the kernel via TilingData.

### Cube Pipeline (Matrix Multiplication)
For matrix operations, AscendC provides a separate Cube unit with its own memory hierarchy:
```
GM → L1 (512KB) → L0A/L0B (64KB each) → Cube Compute → L0C (256KB) → UB → GM
```

- **L1 Buffer**: Staging area for matrix tiles loaded from GM. Managed automatically by the Matmul<> template.
- **L0A / L0B**: Left and right matrix operand buffers feeding the Cube unit.
- **L0C**: Cube output accumulation buffer, stores partial sums in float32.
- **Cube Unit**: Performs 16×16 matrix multiply-accumulate per cycle.

Cube operators use the `Matmul<>` template class which manages the L1/L0 pipeline automatically. You only need to provide M/K/N dimensions and tiling parameters via `TCubeTiling`.

**Key difference**: Vector operators manage UB directly; Cube operators use the `Matmul<>` abstraction which manages L1/L0 internally while sharing UB for pre/post-processing.

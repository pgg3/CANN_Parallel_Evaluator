## CRITICAL CONSTRAINTS (Common LLM Errors)

### ❌ DO NOT cast between float and unsigned integer in aicore functions:
```cpp
// WRONG - cast between floating and unsigned integer variable is not allowed in aicore function!
float invN = 1.0f / (float)totalLength;    // ❌ totalLength is uint32_t
float ratio = (float)unsignedVar;           // ❌ unsigned int to float cast
uint32_t n = (uint32_t)floatVar;            // ❌ float to unsigned int cast
```
✅ Use signed integer as intermediate: `float invN = 1.0f / (float)(int32_t)totalLength;`
✅ Or declare variables as `int32_t` instead of `uint32_t` when float conversion is needed.

### ❌ DO NOT use standard C/C++ math functions:
```cpp
// WRONG - these functions DO NOT EXIST in AscendC kernel code!
float result = exp(x);      // ❌ Compilation error: 'exp' undeclared
float result = expf(x);     // ❌ Compilation error: 'expf' undeclared
float result = sin(x);      // ❌ Compilation error: 'sin' undeclared
```
✅ Use AscendC vector functions: `Exp(dst, src, calCount)`, `Ln(dst, src, calCount)`

### ❌ DO NOT access LocalTensor elements as scalars:
```cpp
// WRONG - LocalTensor does NOT support element-wise indexing!
float val = xLocal[i];           // ❌ no conversion to float
zLocal[i] = val * 2;             // ❌ no viable overloaded '='
if (xLocal[i] > 0) { ... }       // ❌ invalid operands
```
✅ Use vector operations on entire tensors.

### ❌ DO NOT invent QuePosition values:
```cpp
// WRONG - these QuePosition values DO NOT EXIST!
TQue<QuePosition::TEMP, BUFFER_NUM> tmpQueue;     // ❌ no member named 'TEMP'
TQue<QuePosition::VECTMP, BUFFER_NUM> tmpQueue;   // ❌ no member named 'VECTMP'
TQue<QuePosition::VECBUF, BUFFER_NUM> tmpQueue;   // ❌ no member named 'VECBUF'
TQue<QuePosition::TMP, BUFFER_NUM> tmpQueue;      // ❌ no member named 'TMP'
```
✅ Use: `QuePosition::VECIN`, `QuePosition::VECOUT`, `QuePosition::VECCALC`

### ❌ DO NOT use Sub for tensor-scalar subtraction:
```cpp
// WRONG - Sub requires TWO tensors, not tensor and scalar!
Sub(dst, src, scalar, len);   // ❌ no matching function
Sub(dst, src, alpha, len);    // ❌ compilation error
```
✅ Use `Adds` with negative scalar: `Adds(dst, src, -scalar, len)`

### ❌ DO NOT use Compare with scalar directly:
```cpp
// WRONG - Compare requires TWO tensors!
Compare(mask, xLocal, 0.0f, CMPMODE::GT, len);  // ❌ no matching function
```
✅ Use `CompareScalar`: `CompareScalar(mask, xLocal, 0.0f, CMPMODE::GT, len)`
### ❌ DO NOT invent mask types:
```cpp
// WRONG - these types DO NOT EXIST!
SelectMask selMask;           // ❌ unknown type name 'SelectMask'
Tensor<bool> mask;            // ❌ no template named 'Tensor'
LocalTensor<bool> mask;       // ❌ bool not supported
```
✅ Use: `LocalTensor<uint8_t>` for comparison masks

### ❌ DO NOT use for loops for element-wise computation:
```cpp
// WRONG - this is scalar code pattern, not vectorized!
for (int i = 0; i < len; i++) {
    output[i] = input[i] > 0 ? input[i] : alpha * (exp(input[i]) - 1);
}
```
✅ Use vectorized operations: CompareScalar + Select pattern

### ❌ DO NOT use non-existent APIs:
```cpp
// WRONG - these APIs DO NOT EXIST!
Pow(dst, src, exp, len);       // ❌ No Pow function
MaxPool(dst, src, ...);        // ❌ No direct pooling API
AvgPool(dst, src, ...);        // ❌ No direct pooling API
Neg(dst, src, len);            // ❌ No Neg function
Subs(dst, src, scalar, len);   // ❌ No Subs function
Divs(dst, src, scalar, len);   // ❌ No Divs function
```
✅ Alternatives:
- Negation: `Muls(dst, src, -1.0f, len)`
- Subtraction: `Adds(dst, src, -scalar, len)`
- Division by scalar: `Muls(dst, src, 1.0f/scalar, len)`

### ✅ Cube operators MUST use REGIST_MATMUL_OBJ (NOT manual Init):
```cpp
// WRONG - manual Init causes free(): double free crash on 910B!
mm.Init(&tiling, &pipe);           // ❌ bypasses KFC server/client setup
mm.SetWorkspace(workspace, size);  // ❌ wrong workspace type

// CORRECT - use REGIST_MATMUL_OBJ macro
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // ✅ declares AIC+AIV scheduling
Matmul<aType, bType, cType> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.cubeTiling);  // ✅
```
- `KERNEL_TASK_TYPE_DEFAULT` must be FIRST line of kernel entry body
- Kernel must use FLAT structure (local variables), NOT class-based
- Host tiling: fixed 32 MB workspace: `ws[0] = 32 * 1024 * 1024;`

### ❌ DO NOT use set_xxx() for struct tiling fields:
```cpp
// WRONG - TILING_DATA_FIELD_DEF_STRUCT does NOT generate set_ methods!
tiling.set_cubeTiling(cubeTiling);   // ❌ 'has no member named set_cubeTiling'
tiling.set_qkTiling(qkTiling);      // ❌ same error for any struct field
```
✅ Struct fields are public members — assign directly:
```cpp
tiling.cubeTiling = cubeTiling;   // ✅ direct assignment
tiling.qkTiling = qkTiling;      // ✅ direct assignment
```
> Note: Only struct fields (`TILING_DATA_FIELD_DEF_STRUCT`) lack setters. Scalar fields (`set_M()`, `set_N()`) work normally.

### ❌ DO NOT mix Cube and Vector memory incorrectly:
```cpp
// WRONG - L0/L1 buffers are managed by Matmul<> internally!
TQue<QuePosition::VECIN, 2> queue;
mm.SetTensorA(queue.DeQue<half>());   // ❌ Matmul takes GlobalTensor, not LocalTensor
```
✅ Pass `GlobalTensor` to `Matmul<>::SetTensorA/B`, let it manage L1/L0 internally.

### ❌ DO NOT pass TCubeTiling by value/reference to Matmul::Init:
```cpp
// WRONG - Init expects a non-const POINTER!
mm.Init(tiling, &pipe);                    // ❌ no conversion from 'TCubeTiling' to 'TCubeTiling *'
mm.Init(tilingData.cubeTiling, &pipe);     // ❌ same error
// Also WRONG - const pointer doesn't work:
void Init(GM_ADDR a, const TCubeTiling& t, ...) {
    mm.Init(&t, &pipe);  // ❌ 'const TCubeTiling *' → 'TCubeTiling *' fails
}
```
✅ Accept as non-const reference, then take address:
```cpp
void Init(GM_ADDR a, TCubeTiling& tiling, ...) {  // non-const!
    mm.Init(&tiling, &pipe);  // ✅ TCubeTiling* from non-const ref
}
```

### ❌ DO NOT forget Cube alignment:
```cpp
// WRONG - Cube unit requires 16-element alignment for matrix dimensions!
matmulTiling.SetShape(M, K, N);  // ❌ if M, K, or N not multiple of 16
```
✅ Pad M, K, N to multiples of 16: `M_pad = (M + 15) / 16 * 16`

### ❌ DO NOT set BLOCK_DIM > 1 for Cube operators:
```cpp
// WRONG - Matmul<> handles multi-core internally!
context->SetBlockDim(8);  // ❌ causes incorrect results with Matmul<>
```
✅ Use `context->SetBlockDim(1)` for all Cube/Matmul operators.

### ❌ DO NOT forget workspace for Cube operators:
```cpp
// WRONG - Matmul<> needs workspace for L1/L0 buffers!
size_t* ws = context->GetWorkspaceSizes(1);
ws[0] = 0;  // ❌ Matmul will crash without workspace
```
✅ `GetTiling()` returns workspace size: `int64_t wsSize = matmulTiling.GetTiling(cubeTiling); ws[0] = wsSize;`

### ❌ DO NOT use wrong MatmulTiling API:
```cpp
// WRONG - these classes and signatures DO NOT EXIST!
optiling::MatmulTiling matmulTiling(platform);  // ❌ wrong namespace/class
matmulTiling.SetAType(ge::DT_FLOAT16);          // ❌ wrong signature (needs 3 args)
matmulTiling.SetShape(M, K, N);                  // ❌ wrong parameter order
matmulTiling.GetWorkspaceSize();                 // ❌ method does not exist
```
✅ Correct API:
```cpp
matmul_tiling::MatmulApiTiling matmulTiling(platform);
matmulTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
matmulTiling.SetShape(M, N, K);  // order: M, N, K
int64_t wsSize = matmulTiling.GetTiling(cubeTiling);  // returns workspace size
```

### ❌ DO NOT use shape {1} for scalar output:
```cpp
// WRONG - torch.mean() returns torch.Size([]), not torch.Size([1])!
at::Tensor result = at::empty({1}, x.options());  // ❌ shape [1] ≠ shape []
y_shape->SetDimNum(1);
y_shape->SetDim(0, 1);                            // ❌ produces shape [1]
```
✅ For scalar outputs (mean, sum, dot product, loss functions returning scalar):
```cpp
// OUTPUT_ALLOC_CODE:
at::Tensor result = at::empty({}, predictions.options());  // ✅ 0-dim scalar

// INFER_SHAPE_BODY:
gert::Shape* y_shape = context->GetOutputShape(0);
y_shape->SetDimNum(0);  // ✅ 0 dimensions = scalar tensor
return ge::GRAPH_SUCCESS;
```

### ❌ DO NOT use uint32_t for large tensor offsets:
```cpp
// WRONG - overflows when totalLength > ~1B elements (4GB in float32)!
uint32_t totalLength = dim0 * dim1;        // ❌ multiplication can overflow
uint32_t offset = blockLength * coreIdx;   // ❌ byte address may overflow
```
✅ Use `int64_t` for lengths and offsets when tensor size can exceed 500M elements:
```cpp
int64_t totalLength = (int64_t)dim0 * dim1;           // ✅ safe multiplication
int64_t blockLength = totalLength / blockDim;          // ✅ int64_t division
int64_t offset = blockLength * (int64_t)GetBlockIdx(); // ✅ int64_t offset
xGm.SetGlobalBuffer((__gm__ float*)x + offset, blockLength);  // ✅
```
> Use `int64_t` tiling fields (`int64_t totalLength` not `uint32_t totalLength`) for operators with potentially large inputs.

### ❌ DO NOT DataCopy past buffer boundary:
```cpp
// WRONG - when loading individual elements, aligned read may overshoot buffer end!
int32_t alignedIdx = rowIdx / 4 * 4;
DataCopy(tgtLocal, targetGm[alignedIdx], 4);  // ❌ if alignedIdx + 4 > bufferSize → DDR out-of-range
```
✅ DataCopy minimum is 32 bytes (`count * sizeof(T) >= 32`): 8 for float32, 4 for int64, 16 for int16.
Near the buffer end, clamp the start index to leave room for the minimum copy count:
```cpp
constexpr int32_t MIN_COPY = 4;  // 4 int64 = 32 bytes
int32_t alignedIdx = rowIdx / MIN_COPY * MIN_COPY;
if (alignedIdx + MIN_COPY > bufferSize) {
    alignedIdx = bufferSize - MIN_COPY;  // ✅ boundary guard
}
int32_t localOffset = rowIdx - alignedIdx;
DataCopy(tgtLocal, targetGm[alignedIdx], MIN_COPY);
pipe_barrier(PIPE_MTE2);
int32_t value = (int32_t)tgtLocal.GetValue(localOffset);  // ✅ extract target element
```

### ❌ DO NOT use SyncAll or workspace polling for multi-core scalar reduction:
```cpp
// WRONG - SyncAll + core-0 workspace read is fragile and error-prone!
DataCopy(workspaceGm[coreIdx], partialBuf, 8);
SyncAll();
if (coreIdx == 0) {  // ❌ timing issues, workspace sizing errors
    DataCopy(localBuf, workspaceGm[0], totalCores * 8);
    for (int c = 0; c < totalCores; c++) total += localBuf.GetValue(c);
}
```
✅ Use `SetAtomicAdd` — each core atomically accumulates its partial result to GM output:
```cpp
LocalTensor<float> outLocal = outBuf.Get<float>();
Duplicate(outLocal, 0.0f, 8);
pipe_barrier(PIPE_V);
outLocal.SetValue(0, partialResult);
pipe_barrier(PIPE_V);
SetAtomicAdd<float>();
DataCopy(outputGm, outLocal, 8);
pipe_barrier(PIPE_MTE3);
SetAtomicNone();
```
Output must be zero-initialized in OUTPUT_ALLOC_CODE: `at::Tensor result = at::zeros({}, x.options());`

### ❌ DO NOT use VECOUT as MTE2 (GM→UB) target:
```cpp
// WRONG - MTE2 can only write to VECIN or VECCALC, never VECOUT!
TQue<QuePosition::VECOUT, 2> rowQueue;
LocalTensor<float> rowLocal = rowQueue.AllocTensor<float>();
DataCopy(rowLocal, weightGm[offset], count);  // ❌ MTE2 into VECOUT → silent corruption
```
✅ Use VECIN for DMA-in, VECOUT for DMA-out:
```cpp
TQue<QuePosition::VECIN, 2> inQueue;    // MTE2 target (GM → UB)
TQue<QuePosition::VECOUT, 2> outQueue;  // MTE3 source (UB → GM)

// CopyIn: GM → VECIN
LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
DataCopy(inLocal, gmSrc[offset], count);  // ✅ MTE2 into VECIN
inQueue.EnQue(inLocal);

// Compute: VECIN → VECOUT
inLocal = inQueue.DeQue<float>();
LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
DataCopy(outLocal, inLocal, count);       // ✅ UB-to-UB copy
outQueue.EnQue(outLocal);
inQueue.FreeTensor(inLocal);

// CopyOut: VECOUT → GM
outLocal = outQueue.DeQue<float>();
DataCopy(gmDst[offset], outLocal, count); // ✅ MTE3 from VECOUT
outQueue.FreeTensor(outLocal);
```
Or use `TBuf<QuePosition::VECCALC>` as scratch for DMA-in when no vector compute is needed.

### ❌ DO NOT allocate two TBuf on the same QuePosition:
```cpp
// WRONG - two TBuf on VECCALC share/corrupt UB address space!
TBuf<QuePosition::VECCALC> rowBuf;
TBuf<QuePosition::VECCALC> idxBuf;
pipe.InitBuffer(rowBuf, rowSize);  // ✅ first allocation
pipe.InitBuffer(idxBuf, idxSize);  // ❌ may overwrite rowBuf's address range
```
✅ Use different QuePositions, or partition one TBuf:
```cpp
// Option A: different positions
TQue<QuePosition::VECIN, 2> rowQueue;     // VECIN for row data
TBuf<QuePosition::VECCALC> idxBuf;        // VECCALC for indices

// Option B: one TBuf, partition by offset
TBuf<QuePosition::VECCALC> scratchBuf;
pipe.InitBuffer(scratchBuf, rowSize + idxSize);
LocalTensor<float> rowLocal = scratchBuf.Get<float>();
LocalTensor<int64_t> idxLocal = scratchBuf.Get<int64_t>(rowSize);  // offset by rowSize bytes
```

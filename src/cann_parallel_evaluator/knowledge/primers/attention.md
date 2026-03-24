## Attention Pattern Guide

Attention operators decompose into multiple Matmul + Vector operations following the standard attention formula:

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

### Step-by-Step Decomposition
1. **Q × K^T** → scores [seq_len, seq_len] — use `Matmul<>` (Cube unit)
2. **Scale** → scores / sqrt(d_k) — use `Muls()` (Vector unit)
3. **Mask** (optional) → add causal/padding mask — use `Adds()` with -inf for masked positions
4. **Softmax** → row-wise softmax — use Vector ops (ReduceMax, Sub, Exp, ReduceSum, Div)
5. **scores × V** → output [seq_len, d_v] — use `Matmul<>` (Cube unit)

### Multi-Head Attention
- Split Q, K, V into `num_heads` slices along the head dimension
- Process each head independently (loop or batch Matmul)
- Concatenate heads and project with output weight matrix

### Multi-Head Attention — Offset Pattern
Split Q/K/V by heads, process each independently:
```cpp
uint32_t headDim = D / numHeads;
for (uint32_t h = 0; h < numHeads; h++) {
    // Q_h row i starts at: Q_gm[i * D + h * headDim]  (stride between rows = D)
    // Same layout for K_h, V_h
    // Run single-head attention on (seqLen, headDim) slices
    // Write output_h to: O_gm[i * D + h * headDim]
}
```
> For MQA/GQA: K/V use fewer heads. Map Q head index to KV head: `kv_h = h / (numHeads / numKVHeads)`.

### Causal Mask — Lower-Triangle
After computing QK^T scores, set future positions to -infinity:
```cpp
for (uint32_t i = 0; i < seqLen; i++)
    for (uint32_t j = i + 1; j < seqLen; j++)
        scores.SetValue(i * seqLen + j, -1e9f);  // before softmax
```
> Apply mask **after scale, before softmax**. Use scalar SetValue since mask is position-dependent.

### Variant Handling
- **Causal/Windowed**: Apply lower-triangular or banded mask before softmax
- **Multi-Query (MQA)**: K, V have 1 head; broadcast to match Q's num_heads
- **Group-Query (GQA)**: K, V have fewer heads; each KV head serves a group of Q heads
- **KV-Cache (Inference)**: Concatenate new K, V to cached KV; only compute attention for new tokens
- **Linear Attention**: Replace softmax(QK^T)V with φ(Q)(φ(K)^T V), compute K^T V first (linear cost)
- **Sparse Attention**: Only compute attention for selected (sparse) positions

### Implementation Notes
- Uses both Cube (matmul) and Vector (scale, mask, softmax) units
- Softmax must be computed in Vector unit (no high-level softmax Cube API)
- For long sequences, tile along the sequence dimension to fit in memory
- Workspace needed for Matmul<> operations
- Each Matmul step needs its own TCubeTiling, or reuse with updated shapes
- BLOCK_DIM = 1 for Cube operations

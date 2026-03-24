## Convolution Pattern Guide

Convolution operators are implemented as im2col + Matmul on the Cube unit. AscendC does NOT have a direct Conv2D kernel API — you decompose convolution into matrix multiplication.

### im2col + Matmul Approach
```
Input [N,C,H,W] → im2col → Column Matrix [N*H_out*W_out, C*kH*kW]
Weight [C_out, C, kH, kW] → reshape → [C_out, C*kH*kW]
Output = Column × Weight^T → reshape → [N, C_out, H_out, W_out]
```

### Implementation Strategy
1. **TilingFunc** computes output spatial dimensions:
   ```
   H_out = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1
   W_out = (W + 2*pad - dilation*(kW-1) - 1) / stride + 1
   M = N * H_out * W_out   (rows of column matrix)
   K = C_in * kH * kW      (cols of column matrix)
   N_mat = C_out            (output channels)
   ```

2. **Kernel** performs im2col in UB, then uses `Matmul<>` for the GEMM:
   - Load input tile from GM to UB
   - Rearrange into column format (handle padding/stride/dilation)
   - Feed column matrix and weight matrix to Matmul<>
   - Write output to GM

### Variant Handling
- **Standard Conv**: Regular im2col + Matmul
- **Pointwise Conv (1x1)**: No im2col needed — direct Matmul on reshaped input
- **Dilated Conv**: Modify im2col indexing to skip elements: `src_idx = base + d * dilation`
- **Grouped Conv**: Split channels into groups, run independent Matmul per group

### Depthwise Conv — Vector Only (No Matmul)
Depthwise conv processes each channel independently. Use **Vector sliding window** (like pooling but with weighted sum instead of max/avg):
```cpp
// For each sample n, channel c, output position (oh, ow):
for (uint32_t c = 0; c < C; c++) {
    for (uint32_t oh = 0; oh < H_out; oh++) {
        for (uint32_t ow = 0; ow < W_out; ow++) {
            float acc = 0.0f;
            for (uint32_t fh = 0; fh < kH; fh++) {
                for (uint32_t fw = 0; fw < kW; fw++) {
                    int32_t ih = oh * stride + fh - pad;
                    int32_t iw = ow * stride + fw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        acc += input[c][ih][iw] * weight[c][fh][fw];
                }
            }
            output[c][oh][ow] = acc;
        }
    }
}
```
> **No Matmul<> needed** — depthwise is pure scalar/Vector computation.

### Transposed Conv — Scatter-Add Pattern
Transposed conv reverses convolution: each input pixel contributes to multiple output pixels.
```cpp
// Initialize output to zero
// For each input pixel (ih, iw):
//   For each filter position (fh, fw):
//     oh = ih * stride + fh - pad
//     ow = iw * stride + fw - pad
//     output[oh][ow] += input[ih][iw] * weight[fh][fw]
```
> This is the im2col approach in reverse (col2im). Use Matmul<> with swapped M/N dimensions, or implement with Vector scalar accumulation.

### Key Points
- No `Conv2D()` or `Conv3D()` API exists — always decompose
- Use `Matmul<>` template for the GEMM portion
- 1D conv: reshape to 2D (W→H×1), apply 2D logic
- 3D conv: extend im2col to 3 spatial dims (D×H×W)
- Workspace required for Matmul<> L1/L0 buffers
- BLOCK_DIM = 1 for Cube operations

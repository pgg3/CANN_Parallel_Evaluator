## Resize Pattern Guide

Resize (interpolation/upsample) operators change spatial dimensions by computing output values from input values at mapped coordinates.

### Coordinate Mapping
For each output position (oy, ox), compute the corresponding input position:
```
src_y = oy * (H_in / H_out)    // or (oy + 0.5) * (H_in / H_out) - 0.5 for align_corners=False
src_x = ox * (W_in / W_out)
```

### Interpolation Methods
**Nearest Neighbor** — simplest:
```
output[oy][ox] = input[round(src_y)][round(src_x)]
```
Implementation: compute index, use Gather or direct DataCopy with computed offsets.

**Bilinear** — weighted average of 4 neighbors:
```
(y0, x0) = floor(src_y), floor(src_x)
(y1, x1) = y0 + 1, x0 + 1
wy = src_y - y0,  wx = src_x - x0
output = (1-wy)*(1-wx)*input[y0][x0] + (1-wy)*wx*input[y0][x1]
       + wy*(1-wx)*input[y1][x0]     + wy*wx*input[y1][x1]
```
Implementation: Gather 4 corner values, compute weights, weighted sum with Muls+Add.

**Bicubic** — weighted average of 16 neighbors (4×4 grid).

**Trilinear** — 3D extension of bilinear (8 neighbors).

### Implementation Strategy
1. **TilingFunc**: Pass input/output spatial dims, scale factors, mode
2. **Kernel**: Loop over output spatial positions
   - Compute source coordinate
   - Gather required input values
   - Apply interpolation formula using Vector ops
   - Write result

### Bilinear Kernel Pattern
For each output pixel, read 4 corners via scalar GM access and blend:
```cpp
float src_y = oh * (float)H_in / H_out;
float src_x = ow * (float)W_in / W_out;
uint32_t y0 = (uint32_t)src_y, y1 = (y0 + 1 < H_in) ? y0 + 1 : y0;
uint32_t x0 = (uint32_t)src_x, x1 = (x0 + 1 < W_in) ? x0 + 1 : x0;
float wy = src_y - (float)y0, wx = src_x - (float)x0;
// Read 4 corners from GM via scalar path
float v00 = *((__gm__ float*)inputGm.GetPhyAddr() + y0 * W_in + x0);
float v01 = *((__gm__ float*)inputGm.GetPhyAddr() + y0 * W_in + x1);
float v10 = *((__gm__ float*)inputGm.GetPhyAddr() + y1 * W_in + x0);
float v11 = *((__gm__ float*)inputGm.GetPhyAddr() + y1 * W_in + x1);
float val = (1-wy)*(1-wx)*v00 + (1-wy)*wx*v01 + wy*(1-wx)*v10 + wy*wx*v11;
outLocal.SetValue(ow, val);
```
> Integer cast `(uint32_t)` acts as floor for positive values. Clamp y1/x1 to avoid out-of-bounds.

### Grid Sample Variants
Grid sample takes an explicit coordinate grid tensor (instead of uniform scaling):
- `grid[N, H_out, W_out, 2]` specifies (x, y) source coordinates for each output position
- Same interpolation logic, but coordinates come from the grid tensor

### Key Points
- Process output positions in tiles that fit in UB
- Pre-compute coordinate mappings in TilingFunc when possible
- Handle boundary conditions: clamp, reflect, or zero-pad
- Use Vector unit with standard TQue/TBuf buffer management
- For antialias resize: apply low-pass filter before downsampling

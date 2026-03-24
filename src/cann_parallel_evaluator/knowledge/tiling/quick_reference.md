## Tiling Quick Reference

Use the **usable UB budget** and **valid BLOCK_DIM values** from the Hardware Constraints section above.

```
maxTileLength = UB_SIZE / (NUM_QUEUES × BUFFER_NUM × sizeof(dtype))
tileNum = blockLength / (maxTileLength × BUFFER_NUM)
tailLength = blockLength - tileNum × BUFFER_NUM × tileLength
```

Align tileLength to 32 bytes (8 floats / 16 halfs).

### Reading Attributes in TilingFunc

```cpp
const auto* attrs = context->GetAttrs();

// Scalar types — use convenience methods:
int64_t  val = *attrs->GetInt(0);     // returns const int64_t*
float    val = *attrs->GetFloat(1);   // returns const float*
bool     val = *attrs->GetBool(2);    // returns const bool*

// List types — use GetAttrPointer<T> with std::vector:
auto listIntPtr   = attrs->GetAttrPointer<std::vector<int64_t>>(0);  // list_int
auto listFloatPtr = attrs->GetAttrPointer<std::vector<float>>(1);    // list_float
// Then iterate: for (auto v : *listIntPtr) { ... }
```

**IMPORTANT**: Do NOT invent APIs like `gert::ArrayAttr`, `GetListInt()`, or `GetInitParam()` — they do not exist.

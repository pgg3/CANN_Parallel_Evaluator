# 知识内容参考

本文档详细说明各知识片段的具体内容、设计意图和来源。

所有知识内容以 `.md` 文件存储在 `knowledge/` 子目录中，由 `CANNKnowledgeProvider` 的细粒度方法加载。

---

## Level 0: AscendC 编程模型

> 文件：`knowledge/primers/level0_programming_model.md` (3,283 chars)
> 方法：`provider.get_programming_model()`

### 覆盖内容

| 主题 | 要点 |
|------|------|
| **Vector vs Cube** | Vector 用于 element-wise/reduction，Cube 用于矩阵乘法 |
| **Pipe 模型** | CopyIn → Compute → CopyOut 三段式流水线 |
| **内存层次** | GM（HBM, GBs）↔ UB（SRAM, 256KB/core）；Cube: L1/L0A/L0B/L0C |
| **Buffer 管理** | TQue（VECIN/VECOUT/VECCALC）、TBuf、double buffering |
| **算子生命周期** | Init → Process → CopyIn/Compute/CopyOut 循环 |
| **Tiling 基础** | 为什么要分块、UB 容量约束、blockLength/tileLength |
| **Cube Pipeline** | L1→L0A/L0B→Cube Core→L0C→GM 数据流 |

### 设计意图

Level 0 是所有 AscendC 算子开发的基础知识，不区分计算模式。
目标是让 LLM 理解 AscendC 的核心编程范式，避免写出"普通 C++ 风格"的代码。
包含 Cube Pipeline 段落，确保 Cube/Mixed 范式算子也有正确的内存层次认知。

### 使用阶段

- **init**: 通过 `assemble_for_init()` 注入（所有范式）
- compile-fix / correctness-fix / evolve: 不注入（已在 init 阶段学过）

---

## Level 1: 计算模式指南

> 文件：`knowledge/primers/{pattern}.md`
> 方法：`provider.get_pattern_guide(pattern)`

### 三范式体系

150 个算子分为 11 个计算模式，归入 3 个范式：

| 范式 | Pattern | 文件 | 大小 | 算子数 |
|------|---------|------|------|--------|
| **Vector** | elementwise | `elementwise.md` | 1,441 chars | 31 |
| **Vector** | reduction | `reduction.md` | 1,267 chars | 5 |
| **Vector** | softmax | `softmax.md` | 1,475 chars | 2 |
| **Vector** | broadcast | `broadcast.md` | 878 chars | 10 |
| **Vector** | pooling | `pooling.md` | 1,793 chars | 6 |
| **Cube** | matmul | `matmul.md` | 1,899 chars | 17 |
| **Cube** | convolution | `convolution.md` | 1,960 chars | 34 |
| **Cube** | attention | `attention.md` | 1,795 chars | 15 |
| **Mixed** | normalization | `normalization.md` | 2,086 chars | 8 |
| **Mixed** | index | `index.md` | 1,930 chars | 12 |
| **Mixed** | resize | `resize.md` | 1,949 chars | 10 |
| *(fallback)* | other | `other.md` | 927 chars | — |

### 各范式要点

#### Vector 范式（54 算子）

```
执行单元: Vector Unit + UB
内存路径: GM → UB → Vector → UB → GM
Tiling: 1D tiling（UB 容量驱动）

Elementwise:  Add, Sub, Mul, Div, Exp, Ln, Sqrt, Abs, Relu — 最简单
Reduction:    ReduceMax, ReduceSum, ReduceMin — 需 workLocal buffer
Softmax:      ReduceMax → Sub → Exp → ReduceSum → Div 多步
Broadcast:    Adds, Muls 标量-向量操作
Pooling:      滑窗 + ReduceMax/ReduceSum（无 MaxPool/AvgPool 直接 API）
```

#### Cube 范式（66 算子）

```
执行单元: Cube Unit + L1/L0A/L0B/L0C
内存路径: GM → L1 → L0A/L0B → Cube → L0C → GM
Tiling: M/K/N tiling（TCubeTiling 自动）

Matmul:       Matmul<> 模板, Init/SetTensorA/B/IterateAll, 自动 pipeline
Convolution:  im2col + Matmul 分解, stride/padding 索引计算
Attention:    Q*K^T → scale → softmax → *V, Cube+Vector 混合
```

#### Mixed 范式（30 算子）

```
执行单元: Vector Unit + 多维循环
内存路径: GM → UB → Vector → UB → GM（但需处理多维张量）
Tiling: 多维 tiling（batch/channel/spatial 嵌套循环）

Normalization: mean/var/normalize/scale/shift 手动 vector 实现
Index:         Gather/Scatter API, offset 格式, embedding=Gather
Resize:        坐标映射 + 加权插值
```

### 使用阶段

- **init**: 通过 `assemble_for_init()` 注入（按 pattern 选择）
- **correctness-fix**: 通过 `assemble_for_correctness_fix(pattern)` 注入
- compile-fix / evolve: 不注入

---

## 约束知识

### 完整约束

> 文件：`knowledge/constraints/critical_full.md` (4,250 chars)
> 方法：`provider.get_critical_constraints()`

包含 ❌ 错误代码 + ✅ 正确做法的完整对比：

| 约束 | 错误示例 | 正确做法 |
|------|---------|---------|
| 不能用 C math | `exp(x)`, `sqrtf(x)` | `Exp(dst, src, n)`, `Rsqrt()` |
| 不能标量索引 | `xLocal[i]` | 向量操作 |
| QuePosition 只有 3 个 | `TEMP`, `VECTMP` | `VECIN`, `VECOUT`, `VECCALC` |
| Sub 不接受标量 | `Sub(d,s,scalar,n)` | `Adds(d,s,-scalar,n)` |
| Compare 不接受标量 | `Compare(m,s,scalar,...)` | `CompareScalar(m,s,scalar,...)` |
| Mask 类型 | `SelectMask`, `bool` | `LocalTensor<uint8_t>` |
| 不能用 for 循环 | `for(i) output[i]=...` | 向量化操作 |
| 不存在的 API | `Pow`, `Neg`, `Subs`, `Divs` | `Muls(d,s,-1,n)` 等替代 |
| Cube 对齐 | 非 16 对齐维度 | M/K/N 需 16 对齐 |
| 无 MaxPool/AvgPool API | `MaxPool()` | 手动滑窗 + ReduceMax |

### 精简约束

> 文件：`knowledge/constraints/critical_compact.md` (1,040 chars)
> 方法：`provider.get_critical_constraints_compact()`

仅包含规则列表，无代码示例。用于 compile-fix 和 evolve 阶段。

### 使用阶段

- **init**: 完整版，通过 `assemble_for_init()` 注入
- **compile-fix**: 精简版，通过 `assemble_for_compile_fix()` 注入
- **evolve**: 精简版，通过 `assemble_for_evolve()` 注入
- correctness-fix: 不注入（正确性问题不是 API 用法问题）

---

## Tiling 知识（按范式分化）

### Vector Tiling

#### 完整教学

> 文件：`knowledge/tiling/fundamentals.md` (3,018 chars)
> 方法：`provider.get_tiling_fundamentals()`

核心公式：
```
Total UB used = num_queues × BUFFER_NUM × tileLength × sizeof(dtype)
UB budget ≈ 176 KB (256KB - 80KB system reserve)
maxTileLength = UB_SIZE / (NUM_QUEUES × BUFFER_NUM × sizeof(dtype))
```

包含详细的步骤和反例，纠正 LLM 从数据大小推导 tileLength 的错误倾向。

#### 边界处理

> 文件：`knowledge/tiling/edge_cases.md` (1,205 chars)
> 方法：`provider.get_tiling_edge_cases()`

Tail handling 指南：最后一个 tile 可能不满，需要特殊处理。

#### 速查

> 文件：`knowledge/tiling/quick_reference.md` (604 chars)
> 方法：`provider.get_tiling_quick_reference()`

仅公式和表格，无详细解释。

### Cube Tiling

> 文件：`knowledge/tiling/cube_fundamentals.md` (2,254 chars)
> 方法：`provider.get_tiling_cube_fundamentals()`

```
核心内容:
- L1/L0A/L0B/L0C 内存层次与容量
- M/K/N tiling 策略
- TCubeTiling 结构体
- MatmulTiling host API (SetAType/SetBType/SetCType/SetShape/GetTiling)
- workspace 大小计算与分配
```

### Mixed Tiling

> 文件：`knowledge/tiling/multidim_fundamentals.md` (2,371 chars)
> 方法：`provider.get_tiling_multidim_fundamentals()`

```
核心内容:
- 多维张量的 batch/channel/spatial 嵌套循环
- shape 信息从 TilingFunc 传递到 kernel 的方法
- 与 Vector 1D tiling 的区别
```

### 范式路由

> 方法：`provider.get_tiling_for_paradigm(paradigm)`

```python
if paradigm == "cube":     → cube_fundamentals.md
elif paradigm == "mixed":  → multidim_fundamentals.md
else (vector):             → fundamentals.md + edge_cases.md
```

### 使用阶段

- **init**: 按范式选择完整 tiling 指南，通过 `assemble_for_init()` 注入
  - Vector: fundamentals + edge_cases
  - Cube: cube_fundamentals
  - Mixed: multidim_fundamentals
- **correctness-fix**: 按范式选择，通过 `assemble_for_correctness_fix()` 注入
  - Vector: quick_reference
  - Cube: cube_fundamentals
  - Mixed: multidim_fundamentals
- **evolve**: 速查版，通过 `assemble_for_evolve()` 注入
- compile-fix: 不注入（编译错误不是 tiling 问题）

---

## API 参考

### API 签名速查

> 文件：`knowledge/api/quick_reference.md` (977 chars)
> 方法：`provider.get_api_quick_reference()`

所有常用 AscendC Vector API 的函数签名列表。

### 高级 API 参考

> 文件：`knowledge/api/advanced_reference.md` (4,663 chars)
> 方法：`provider.get_advanced_api_reference()`

Cube 和高级 API 的完整签名与参数说明：
- Matmul<> 模板与 lifecycle
- Conv2D API
- LayerNorm / RMSNorm
- Gather / Scatter
- Transpose

### 使用阶段

- **init**: 速查版始终注入；高级版在 cube/mixed 范式或 `needs_advanced=True` 时注入
- **compile-fix**: 速查版，通过 `assemble_for_compile_fix()` 注入
- **evolve**: 速查版，通过 `assemble_for_evolve()` 注入
- correctness-fix: 不注入

---

## Level 2: API 实时检索

> 文件：`knowledge/api_scanner.py`
> 方法：`provider.search_api(name)`, `provider.list_apis()`

### 数据源

从 CANN SDK 安装目录的头文件中实时扫描：

```
{ASCEND_HOME_PATH}/{arch}-linux/ascendc/include/basic_api/interface/
```

### 头文件到分类映射

| 头文件 | 分类 | 典型 API |
|--------|------|---------|
| `kernel_operator_vec_binary_intf.h` | vec_binary | Add, Sub, Mul, Div |
| `kernel_operator_vec_unary_intf.h` | vec_unary | Abs, Exp, Ln, Sqrt |
| `kernel_operator_vec_reduce_intf.h` | vec_reduce | ReduceMax, ReduceSum |
| `kernel_operator_vec_compare_intf.h` | vec_compare | Compare, CompareScalar |
| `kernel_operator_vec_select_intf.h` | vec_select | Select |
| `kernel_operator_vec_scalar_intf.h` | vec_scalar | Adds, Muls, Maxs, Mins |
| `kernel_operator_vec_dup_intf.h` | vec_dup | Duplicate |
| `kernel_operator_mm_intf.h` | cube_matmul | Mmad |
| `kernel_operator_data_copy_intf.h` | data_copy | DataCopy |

### 搜索结果格式

```python
{
    "status": "found",          # found / not_found / ambiguous
    "name": "Add",
    "category": "vec_binary",
    "header": "kernel_operator_vec_binary_intf.h",
    "brief": "dst = src0 + src1",
    "candidates": []            # ambiguous 时的候选列表
}
```

### 当前使用方式

Level 2 主要供 cann-claude-tools MCP 工具使用（按需检索）。
在 prompt 中，API 知识通过 quick_reference.md 和 advanced_reference.md 提供。

---

## Level 3: 精选示例

> 文件：`knowledge/examples/{name}.md`
> 方法：`provider.get_example(pattern)`

### 示例列表

| 示例 | 文件 | 大小 | 范式 | 展示要点 |
|------|------|------|------|---------|
| **Add** | `add.md` | 5,733 chars | Vector | 最简单的双输入 elementwise，完整骨架 |
| **ReLU** | `relu.md` | 4,402 chars | Vector | 单输入模式，CompareScalar + Select |
| **ReduceSum** | `reduce_sum.md` | 4,512 chars | Vector | 归约模式，workLocal buffer |
| **Pooling** | `pooling.md` | 7,203 chars | Vector | 滑窗 + ReduceMax，无直接 API |
| **Matmul** | `matmul.md` | 3,279 chars | Cube | Matmul<> 模板 lifecycle, TCubeTiling |
| **Convolution** | `convolution.md` | 8,603 chars | Cube | im2col 索引计算 + Matmul, workspace |
| **Attention** | `attention.md` | 6,989 chars | Cube | 双 Matmul 实例 + Vector softmax 混合 |
| **LayerNorm** | `layer_norm.md` | 5,448 chars | Mixed | 手动 Vector 实现 mean/var/normalize |
| **Gather** | `gather.md` | 4,973 chars | Mixed | offset 格式, 按索引访问 |

### 每个示例包含完整 6 组件

```
KERNEL_IMPL          — 完整的 Kernel 类（Init, Process, CopyIn, Compute, CopyOut）
KERNEL_ENTRY_BODY    — extern "C" 入口代码
TILING_FIELDS        — TilingData 字段定义
TILING_FUNC_BODY     — TilingFunc 完整实现
INFER_SHAPE_BODY     — InferShape 实现
OUTPUT_ALLOC_CODE    — Python binding 输出分配
```

### 模式到示例的映射

```python
_EXAMPLE_FILES = {
    "elementwise_binary": "add.md",
    "elementwise_unary": "relu.md",
    "elementwise": "add.md",       # default → binary (Add)
    "reduction": "reduce_sum.md",
    "softmax": "reduce_sum.md",    # softmax 复用 reduction 示例
    "broadcast": "add.md",         # broadcast 复用 add 示例
    "pooling": "pooling.md",
    "matmul": "matmul.md",
    "convolution": "convolution.md",
    "attention": "attention.md",
    "normalization": "layer_norm.md",
    "index": "gather.md",
    "resize": "gather.md",         # resize 复用 gather 示例
}
```

### 使用阶段

- **init**: 通过 `assemble_for_init()` 注入（按 pattern 选择）
- compile-fix / correctness-fix / evolve: 不注入（代码结构已有）

---

## 非 Provider 知识（保留在 cann_init_task.py 中）

以下知识与具体算子的签名相关，由 task 直接生成，不通过 provider：

### Component Specification

6 组件的输出格式定义和文件模板，包含 Add 算子的完整工作示例。
这是 "Assemble 模式" 的体现 — 固定结构由程序控制，LLM 只填充变量部分。

### Attribute Access Guide

教 LLM 如何在 TilingFunc 中获取算子属性（init_params）。
仅当算子有 `init_params` 时才注入。包含：
- `context->GetAttrs()->GetAttrPointer(i)` 的正确用法
- `reinterpret_cast` 类型转换
- 明确警告不存在的 API（`GetInitParam()` 等）

### Signature Summary

从 Python reference 解析出的输入输出参数摘要。

### 150 算子预分类字典

`_OPERATOR_PATTERN_MAP` 内置于 `cann_init_task.py`，覆盖全部 150 个 benchmark 算子。
用于 `_infer_compute_pattern()` 的第 2 优先级查表，确保每个算子精确映射到正确的 pattern。

---

## 各阶段知识组装汇总（按范式分化）

| 知识片段 | Vector init | Cube init | Mixed init | compile-fix | correctness-fix | evolve |
|---------|------------|----------|-----------|-------------|-----------------|--------|
| Level 0 编程模型 | ✅ | ✅ | ✅ | | | |
| Level 1 模式指南 | ✅ | ✅ | ✅ | | ✅ | |
| 约束 (完整) | ✅ | ✅ | ✅ | | | |
| 约束 (精简) | | | | ✅ | | ✅ |
| API (速查) | ✅ | ✅ | ✅ | ✅ | | ✅ |
| API (高级) | ✅* | ✅ | ✅ | | | |
| 精选示例 | ✅ | ✅ | ✅ | | | |
| tiling (Vector) | ✅ | | | | | |
| tiling (边界) | ✅ | | | | | |
| tiling (Cube) | | ✅ | | | Cube ✅ | |
| tiling (多维) | | | ✅ | | Mixed ✅ | |
| tiling (速查) | | | | | Vector ✅ | ✅ |
| Attribute Guide (条件) | ✅ | ✅ | ✅ | | | |
| Component Spec | ✅ (完整) | ✅ (完整) | ✅ (完整) | ✅ (输出格式) | ✅ (输出格式) | ✅ (精简) |

> `*` Vector 仅在 `needs_advanced=True` 时

### 估算大小

| 阶段 | Vector | Cube | Mixed |
|------|--------|------|-------|
| **init** | ~18K | ~25K | ~22K |
| **compile-fix** | ~2K | ~2K | ~2K |
| **correctness-fix** | ~2K (速查) | ~4K (cube tiling) | ~4K (多维 tiling) |
| **evolve** | ~2.6K | ~2.6K | ~2.6K |

# 知识系统架构

## 设计原则

### 细粒度知识 + 灵活组装

知识系统的核心设计是**细粒度存储、按需组装**：

- 每个知识片段独立存储为 `.md` 文件
- `CANNKnowledgeProvider` 为每个片段提供独立的 getter 方法
- 不同阶段通过 assembly helper 组装不同的知识子集
- 调用者（task、agentic interface）可以直接调用细粒度方法，也可以使用预组装 helper

**为什么不是一次性注入所有知识？**

旧方案将完整 27K task description 注入每个 fix prompt，导致：
- compile-fix 时 LLM 被大量无关的 tiling 教学和示例干扰
- correctness-fix 时 LLM 被大量无关的 API 语法信息干扰
- token 浪费严重

新方案按阶段精准注入：
- compile-fix 只需 API 语法 + 约束 → ~2K chars
- correctness-fix 只需计算模式 + tiling → 2-4K chars

### 分层注入

不同阶段需要不同深度的知识：

| 阶段 | 使用者 | 知识深度 | 原因 |
|------|--------|---------|------|
| **init** | CANNInitTask | 完整教学 | LLM 从零开始，需要编程模型、模式指南、完整示例 |
| **compile-fix** | AgenticFixLoopMixin | API + 约束 | 代码结构已有，只需修正 API 用法 |
| **correctness-fix** | AgenticFixLoopMixin | 模式 + tiling | API 语法正确，需修正计算逻辑和数据流 |
| **evolve** | CANNInitTask | 精简参考 | 已有父代代码，只需速查表 |

### 按范式分化

150 算子归入 3 个范式（vector/cube/mixed），不同范式需要不同的知识组合：

```python
_PARADIGM: dict[str, str] = {
    "elementwise": "vector",   "reduction": "vector",
    "softmax": "vector",       "broadcast": "vector",
    "pooling": "vector",
    "matmul": "cube",          "convolution": "cube",
    "attention": "cube",
    "normalization": "mixed",  "index": "mixed",
    "resize": "mixed",
    "other": "vector",  # fallback
}
```

范式决定：
- **init 阶段**的 tiling 指南和高级 API 注入
- **correctness-fix 阶段**的 tiling 指南选择
- **evolve 阶段**的 tiling 速查选择

---

## CANNKnowledgeProvider

```python
class CANNKnowledgeProvider:
    """Fine-grained knowledge provider for AscendC operator development.

    不继承任何抽象基类 — 只服务于 CANNInitTask，无需泛化。
    所有知识以 .md 文件存储，通过 _load_md() 懒加载并缓存。
    """

    def __init__(self, cann_path=None):
        self._cann_path = cann_path or api_scanner.default_cann_path()
        self._api_index = None  # 懒加载

    # === Level 0/1: 编程模型 + 计算模式 ===
    def get_programming_model(self) -> str              # 编程模型概览（含 Cube Pipeline）
    def get_pattern_guide(self, pattern) -> str          # 计算模式指南 (12 个: 11 patterns + other)
    def get_primer(self, compute_pattern) -> str         # 便捷: Level 0 + Level 1 组合

    # === 约束 ===
    def get_critical_constraints(self) -> str            # 约束 (完整, ~4250 chars)
    def get_critical_constraints_compact(self) -> str    # 约束 (精简, ~1040 chars)

    # === Tiling/内存管理 ===
    def get_tiling_fundamentals(self) -> str             # Vector tiling: UB 预算、计算
    def get_tiling_edge_cases(self) -> str               # Vector tiling: tail handling
    def get_tiling_quick_reference(self) -> str          # Vector tiling: 公式速查
    def get_tiling_cube_fundamentals(self) -> str        # Cube tiling: TCubeTiling、MatmulTiling、workspace
    def get_tiling_multidim_fundamentals(self) -> str    # Mixed tiling: 多维嵌套循环、shape 传递
    def get_tiling_for_paradigm(self, paradigm) -> str   # 按范式自动选择 tiling 指南

    # === API 参考 ===
    def get_api_quick_reference(self) -> str             # 速查: Vector API 签名
    def get_advanced_api_reference(self) -> str          # 高级: Matmul/Normalization/Index/Transpose

    # === Level 2: 实时 SDK 搜索 ===
    def search_api(self, name) -> dict                   # 按名称搜索 CANN SDK 头文件
    def list_apis(self) -> dict                          # 按分类列出所有 API

    # === Level 3: 精选示例 ===
    def get_example(self, pattern) -> Optional[str]      # 完整可编译的示例 (9 个)

    # === 范式判断 ===
    @staticmethod
    def get_paradigm(pattern) -> str                     # pattern → vector/cube/mixed

    # === Assembly helpers — 按阶段预组装 ===
    def assemble_for_init(self, pattern, needs_advanced=False) -> str
    def assemble_for_compile_fix(self) -> str
    def assemble_for_correctness_fix(self, pattern) -> str
    def assemble_for_evolve(self, pattern="other") -> str
```

### 设计决策

**为什么不放在 `core/` 中？**

KnowledgeProvider 只服务于 CANNInitTask 一个 task。放在 `core/` 会引入不必要的抽象层级。
直接作为 `task/cann_init/knowledge/` 下的具体类，简单直接。

**为什么不用抽象基类？**

目前只有一个实现（CANN），没有多态需求。如果未来有其他 task 需要类似能力，
再提取抽象接口也不迟。

**为什么用 assembly helper 而不是让调用者自己组装？**

两者都支持。Assembly helper 是常用组合的便捷方法，调用者也可以直接调用细粒度 getter
自行组装。AgenticFixLoopMixin 目前使用 helper，但如果需要更精细的控制可以随时切换。

---

## 计算模式推断

`CANNInitTask._infer_compute_pattern()` 采用 3 级优先：

```python
def _infer_compute_pattern(self) -> str:
    # 1. 显式覆盖：调用者传入 compute_pattern
    if self.compute_pattern:
        return self.compute_pattern

    # 2. 内置查表：150 算子预分类字典
    mapped = _OPERATOR_PATTERN_MAP.get(self.op_name)
    if mapped:
        return mapped

    # 3. 关键词回退：适用于未收录的新算子
    name = self.op_name.lower()
    if any(kw in name for kw in ["reduce", "sum_reduction", ...]):
        return "reduction"
    if "softmax" in name:
        return "softmax"
    # ... 其他关键词匹配 ...
    return "other"
```

**`_OPERATOR_PATTERN_MAP`** 内置于 `cann_init_task.py`，覆盖全部 150 个 benchmark 算子：

| pattern | 算子数 | 来源 category |
|---------|--------|--------------|
| elementwise | 31 | activation(13) + loss(7) + math(1) + optimizer(5) |
| reduction | 10 | reduce(5) + math(5, scan ops) |
| softmax | 2 | activation 中的 softmax/log_softmax |
| broadcast | 10 | broadcast(10) |
| pooling | 6 | pooling(6) |
| matmul | 17 | matmul(17) |
| convolution | 34 | convolution(34) |
| attention | 15 | attention(15) |
| normalization | 8 | normalization(8) |
| index | 12 | index(12) |
| resize | 10 | resize(10) |

---

## API Scanner

从 CANN SDK 头文件实时扫描 API 签名。

### 数据源

```
{ASCEND_HOME_PATH}/{arch}-linux/ascendc/include/basic_api/interface/
├── kernel_operator_vec_binary_intf.h    → vec_binary (Add, Sub, Mul, Div)
├── kernel_operator_vec_unary_intf.h     → vec_unary (Exp, Ln, Sqrt, Abs)
├── kernel_operator_vec_reduce_intf.h    → vec_reduce (ReduceMax, ReduceSum)
├── kernel_operator_mm_intf.h            → cube_matmul (Mmad)
├── kernel_operator_data_copy_intf.h     → data_copy (DataCopy)
└── ...
```

### 搜索策略

三级匹配：
1. **精确匹配**：`name == api_name`
2. **大小写不敏感**：`name.lower() == api_name.lower()`
3. **子串匹配**：`name.lower() in api_name.lower()`

### Fallback

当 CANN SDK 不可用时，使用内置的 `FALLBACK_APIS` 字典（覆盖常用 API）。

---

## Prompt 组装流程

### Init 阶段 (CANNInitTask)

```python
def _get_init_task_description(self) -> str:
    provider = self.get_knowledge_provider()
    pattern = self._infer_compute_pattern()

    parts = [
        self._get_base_description(),       # 角色 + 设备 + Python Reference
        self._get_signature_summary(),       # 输入输出参数
        provider.assemble_for_init(          # 知识从 provider 组装
            pattern=pattern,
            needs_advanced=self._needs_advanced_api(),
        ),
    ]

    # 条件性添加属性获取指南
    attr_guide = self._get_attribute_access_guide()
    if attr_guide:
        parts.append(attr_guide)

    parts.append(self._get_component_specification())  # 6 组件模板 + Add 示例
    return "\n\n".join(parts)
```

`assemble_for_init()` 内部组装（按范式分化）：
```python
def assemble_for_init(self, pattern: str, needs_advanced: bool = False) -> str:
    paradigm = self.get_paradigm(pattern)

    parts = [
        self.get_primer(pattern),            # Level 0 + Level 1
        self.get_critical_constraints(),     # 完整约束
        self.get_api_quick_reference(),      # API 速查（所有范式都需要）
    ]

    example = self.get_example(pattern)
    if example:
        parts.append(example)                # 精选示例

    parts.append(self.get_tiling_for_paradigm(paradigm))  # 按范式选择 tiling

    if paradigm in ("cube", "mixed") or needs_advanced:
        parts.append(self.get_advanced_api_reference())   # 高级 API

    return "\n\n".join(parts)
```

> API 速查始终注入的原因：即使 Attention 算子属于 Cube 范式，其 softmax 步骤仍需
> ReduceMax/Exp/ReduceSum 等 Vector API。

### Compile-Fix 阶段 (AgenticFixLoopMixin)

```python
def _build_compile_fix_prompt(self, interface, solution, error):
    knowledge = self._provider.assemble_for_compile_fix()  # ~2K chars
    # prompt = 之前的代码 + 编译错误 + knowledge + 输出格式
```

`assemble_for_compile_fix()` 内部组装：
```
API 签名速查 + 约束精简版
```

设计意图：编译错误通常是 API 用法错误（不存在的函数、参数类型错误），
不需要 tiling 教学和完整示例。

### Correctness-Fix 阶段 (AgenticFixLoopMixin)

```python
def _build_correctness_fix_prompt(self, interface, solution, eval_result):
    knowledge = self._provider.assemble_for_correctness_fix(self._pattern)  # 2-4K chars
    # prompt = 之前的代码 + 正确性错误 + diff 信息 + knowledge + 输出格式
```

`assemble_for_correctness_fix()` 内部组装（按范式分化）：
```python
def assemble_for_correctness_fix(self, pattern: str) -> str:
    paradigm = self.get_paradigm(pattern)
    parts = [self.get_pattern_guide(pattern)]

    if paradigm == "cube":
        parts.append(self.get_tiling_cube_fundamentals())    # M/K/N, TCubeTiling
    elif paradigm == "mixed":
        parts.append(self.get_tiling_multidim_fundamentals()) # 多维嵌套循环
    else:
        parts.append(self.get_tiling_quick_reference())       # UB 公式速查

    return "\n\n".join(parts)
```

设计意图：正确性错误通常是计算逻辑或 tiling 计算错误，
不需要 API 语法参考。Cube 算子的 tiling 错误需要完整的 TCubeTiling 指南，
而非 Vector 的 UB 公式。

### Evolve 阶段 (CANNInitTask)

```python
def _get_evolve_task_description(self) -> str:
    provider = self.get_knowledge_provider()
    pattern = self._infer_compute_pattern()
    parts = [
        self._get_base_description(),
        self._get_signature_summary(),
        provider.assemble_for_evolve(pattern),           # 按范式精简知识
        self._get_component_specification_minimal(),     # 只有输出格式
    ]
    return "\n\n".join(parts)
```

`assemble_for_evolve(pattern)` 内部组装（按范式分化）：
```python
def assemble_for_evolve(self, pattern: str = "other") -> str:
    paradigm = self.get_paradigm(pattern)

    parts = [
        self.get_api_quick_reference(),
        self.get_critical_constraints_compact(),
    ]

    if paradigm == "cube":
        parts.append(self.get_tiling_cube_fundamentals())      # Cube: TCubeTiling
    elif paradigm == "mixed":
        parts.append(self.get_tiling_multidim_fundamentals())  # Mixed: 多维嵌套
    else:
        parts.append(self.get_tiling_quick_reference())        # Vector: UB 公式

    return "\n\n".join(parts)
```

设计意图：evolve 阶段已有父代代码，不需要完整教学，
但 tiling 知识仍需按范式区分 — Cube 算子的 crossover/mutation
需要 TCubeTiling 指南而非 Vector 的 UB 公式。

---

## Agentic Fix Loop 集成

AgenticFixLoopMixin 是多轮 LLM + 编译反馈循环的接口 mixin，使用 provider 进行细粒度知识注入。

### 流程

```
1. 初始生成 (init prompt，由 task 组装)
   ↓
2. 编译检查
   ├── 通过 → 进入正确性检查
   └── 失败 → compile-fix 循环 (最多 3 轮)
              provider.assemble_for_compile_fix() → ~2K chars
   ↓
3. 正确性检查
   ├── 通过 → 返回结果
   └── 失败 → correctness-fix 循环 (最多 2 轮)
              provider.assemble_for_correctness_fix(pattern)
              → Vector ~2K / Cube ~4K / Mixed ~4K chars
```

### 关键代码

```python
class AgenticFixLoopMixin:
    def _run_fix_loop(self, llm, solution, total_usage):
        # provider 在每次 fix 时按需获取
        provider = self.task.get_knowledge_provider()
        pattern = self.task.get_compute_pattern()
```

### 知识注入对比

| 阶段 | 旧方案 (v1) | 新方案 (v2) | 缩减 |
|------|------------|------------|------|
| init | 完整 task description ~27K | 完整 task description ~18-25K | 范式分化 |
| compile-fix | 完整 task description 27K | API + 约束 ~2K | **13x** |
| correctness-fix | 完整 task description 27K | 模式 + tiling 2-4K | **7-13x** |

---

## 知识内容概览

| 知识片段 | 文件 | 大小 |
|---------|------|------|
| Level 0 编程模型 | `primers/level0_programming_model.md` | 3,283 chars |
| Level 1 elementwise | `primers/elementwise.md` | 1,441 chars |
| Level 1 reduction | `primers/reduction.md` | 1,267 chars |
| Level 1 softmax | `primers/softmax.md` | 1,475 chars |
| Level 1 broadcast | `primers/broadcast.md` | 878 chars |
| Level 1 pooling | `primers/pooling.md` | 1,793 chars |
| Level 1 matmul | `primers/matmul.md` | 1,899 chars |
| Level 1 convolution | `primers/convolution.md` | 1,960 chars |
| Level 1 attention | `primers/attention.md` | 1,795 chars |
| Level 1 normalization | `primers/normalization.md` | 2,086 chars |
| Level 1 index | `primers/index.md` | 1,930 chars |
| Level 1 resize | `primers/resize.md` | 1,949 chars |
| Level 1 other | `primers/other.md` | 927 chars |
| 约束 (完整) | `constraints/critical_full.md` | 4,250 chars |
| 约束 (精简) | `constraints/critical_compact.md` | 1,040 chars |
| Tiling (Vector) | `tiling/fundamentals.md` | 3,018 chars |
| Tiling (边界) | `tiling/edge_cases.md` | 1,205 chars |
| Tiling (速查) | `tiling/quick_reference.md` | 604 chars |
| Tiling (Cube) | `tiling/cube_fundamentals.md` | 2,254 chars |
| Tiling (多维) | `tiling/multidim_fundamentals.md` | 2,371 chars |
| API 速查 | `api/quick_reference.md` | 977 chars |
| API 高级 | `api/advanced_reference.md` | 4,663 chars |
| 示例 Add | `examples/add.md` | 5,733 chars |
| 示例 ReLU | `examples/relu.md` | 4,402 chars |
| 示例 ReduceSum | `examples/reduce_sum.md` | 4,512 chars |
| 示例 Matmul | `examples/matmul.md` | 3,279 chars |
| 示例 Pooling | `examples/pooling.md` | 7,203 chars |
| 示例 Gather | `examples/gather.md` | 4,973 chars |
| 示例 LayerNorm | `examples/layer_norm.md` | 5,448 chars |
| 示例 Attention | `examples/attention.md` | 6,989 chars |
| 示例 Convolution | `examples/convolution.md` | 8,603 chars |
| **总计** | | **~95,000 chars** |

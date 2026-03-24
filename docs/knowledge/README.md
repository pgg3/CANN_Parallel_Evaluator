# 知识系统设计

## 概述

知识系统负责为 LLM 提供特定硬件平台的算子开发领域知识。

核心设计：**细粒度知识 + 灵活组装**。

- 每个知识片段独立存储为 `.md` 文件
- Provider 为每个片段提供独立 getter
- 不同阶段通过 assembly helper 组装不同的知识子集

当前实现：`CANNKnowledgeProvider`（AscendC / 昇腾 NPU）。
同样的模式可以复用到其他平台（如 CUDA）。

### 设计目标

| 阶段 | 目标 | 知识策略 |
|------|------|---------|
| **init** | LLM 从零生成可编译的算子 | 完整教学（编程模型 + 模式 + 约束 + API 速查 + 示例 + 范式 tiling + 高级API*） |
| **compile-fix** | LLM 修复编译错误 | 精准注入（API 速查 + 约束精简） |
| **correctness-fix** | LLM 修复正确性错误 | 精准注入（计算模式 + 范式 tiling） |
| **evolve** | LLM 优化已有实现 | 速查参考（API + 约束 + 范式 tiling） |

关键原则：**fix prompt 不应重复完整教学**，只注入与错误类型相关的知识。

---

## 三范式体系

150 个算子分为 11 个计算模式（pattern），归入 3 个范式（paradigm）：

| 范式 | Pattern | 算子数 | 执行单元 | Tiling 策略 |
|------|---------|-------|---------|------------|
| **Vector** | elementwise, reduction, softmax, broadcast, pooling | 54 | Vector Unit + UB | 1D tiling（UB 容量驱动） |
| **Cube** | matmul, convolution, attention | 66 | Cube Unit + L1/L0 | M/K/N tiling（TCubeTiling 自动） |
| **Mixed** | normalization, index, resize | 30 | Vector Unit + 多维循环 | 多维 tiling（batch/channel/spatial） |

Init 阶段根据范式自动选择 tiling 指南和 API 深度；correctness-fix 阶段同样按范式注入对应的 tiling 知识。

### 计算模式推断

`CANNInitTask._infer_compute_pattern()` 采用 3 级优先：

1. **显式覆盖**：`data["compute_pattern"]`（调用者传入）
2. **内置查表**：`_OPERATOR_PATTERN_MAP`（150 算子预分类字典）
3. **关键词回退**：算子名匹配（适用于未收录的新算子）

---

## 知识分类（平台无关）

任何硬件平台的算子开发知识都可以归为以下类别：

| 类别 | 说明 | 粒度 |
|------|------|------|
| **编程模型** | 平台的核心编程范式 | 完整版 |
| **计算模式** | 按算子类型分的实现指南 | 按 pattern 选择 |
| **约束/反幻觉** | 防止 LLM 常见错误的规则 | 完整版 / 精简版 |
| **Tiling/内存管理** | 分片策略、内存层次 | 按范式选择 |
| **API 参考** | 平台 API 签名 | 速查版 / 高级版 / 实时搜索 |
| **精选示例** | 完整可编译的示例代码 | 按 pattern 选择 |

### CANN 实例

| 类别 | CANN 对应 | 文件 |
|------|----------|------|
| 编程模型 | AscendC Pipe 模型、Vector/Cube Pipeline、UB/L1/L0 内存 | `primers/level0_programming_model.md` |
| 计算模式 | 11 个 pattern | `primers/{pattern}.md` |
| 约束 | Vector + Cube 约束（C math、标量索引、不存在的 API、Cube 对齐等） | `constraints/critical_full.md`, `critical_compact.md` |
| Tiling (Vector) | UB tiling 计算、tail handling | `tiling/fundamentals.md`, `edge_cases.md`, `quick_reference.md` |
| Tiling (Cube) | TCubeTiling、MatmulTiling、workspace | `tiling/cube_fundamentals.md` |
| Tiling (Mixed) | 多维嵌套循环、shape 传递 | `tiling/multidim_fundamentals.md` |
| API 参考 | AscendC vector/cube API 签名 | `api/quick_reference.md`, `advanced_reference.md`, `api_scanner.py` |
| 精选示例 | 11 个完整 6 组件示例 | `examples/*.md` |

---

## Provider 接口模式

```python
class CANNKnowledgeProvider:
    # === Level 0/1: 编程模型 + 计算模式 ===
    def get_programming_model(self) -> str              # 编程模型概览（含 Cube Pipeline）
    def get_pattern_guide(self, pattern) -> str          # 计算模式指南 (12 个: 11 patterns + other)
    def get_primer(self, compute_pattern) -> str         # 便捷: Level 0 + Level 1 组合

    # === 约束 ===
    def get_critical_constraints(self) -> str            # 约束 (完整, ~4250 chars)
    def get_critical_constraints_compact(self) -> str    # 约束 (精简, ~1040 chars)

    # === 硬件约束 ===
    def get_hardware_constraints(self) -> str             # 动态生成目标 NPU 的硬件约束（UB 容量、核心数等）

    # === Tiling/内存管理 ===
    def get_tiling_fundamentals(self) -> str             # Vector tiling: UB 预算、计算
    def get_tiling_edge_cases(self) -> str               # Vector tiling: tail handling
    def get_tiling_quick_reference(self) -> str          # Vector tiling: 公式速查
    def get_tiling_cube_fundamentals(self) -> str        # Cube tiling: TCubeTiling、MatmulTiling、workspace
    def get_tiling_multidim_fundamentals(self) -> str    # Mixed tiling: 多维嵌套循环、shape 传递
    def get_tiling_for_paradigm(self, paradigm) -> str   # 按范式自动选择 tiling 指南

    # === API 参考 ===
    def get_api_quick_reference(self) -> str             # 速查: 所有 Vector API 签名
    def get_advanced_api_reference(self) -> str          # 高级 API: Matmul/Normalization/Index/Transpose

    # === Level 2: 实时 SDK 搜索 ===
    def search_api(self, name) -> dict                   # 按名称搜索 CANN SDK 头文件
    def list_apis(self) -> dict                          # 按分类列出所有 API

    # === Level 3: 精选示例 ===
    def get_example(self, pattern) -> Optional[str]      # 完整可编译的示例

    # === 范式判断 ===
    @staticmethod
    def get_paradigm(pattern) -> str                     # pattern → vector/cube/mixed

    # === Assembly helpers — 按阶段预组装 ===
    def assemble_for_init(self, pattern, needs_advanced=False) -> str
    def assemble_for_compile_fix(self) -> str
    def assemble_for_correctness_fix(self, pattern) -> str
    def assemble_for_evolve(self, pattern="other") -> str
```

### 各阶段组装矩阵（按范式分化）

| 知识片段 | Vector init | Cube init | Mixed init | compile-fix | correctness-fix | evolve |
|---------|------------|----------|-----------|-------------|-----------------|--------|
| 编程模型 | ✅ | ✅ | ✅ | | | |
| 计算模式 | ✅ | ✅ | ✅ | | ✅ | |
| 约束 (完整) | ✅ | ✅ | ✅ | | | |
| 约束 (精简) | | | | ✅ | | ✅ |
| API (速查) | ✅ | ✅ | ✅ | ✅ | | ✅ |
| 精选示例 | ✅ | ✅ | ✅ | | | |
| tiling (Vector) | ✅ | | | | | |
| tiling (边界) | ✅ | | | | | |
| tiling (速查) | | | | | Vector ✅ | Vector ✅ |
| tiling (Cube) | | ✅ | | | Cube ✅ | Cube ✅ |
| tiling (多维) | | | ✅ | | Mixed ✅ | Mixed ✅ |
| API (高级) | ✅* | ✅ | ✅ | | | |

> `*` Vector 仅在 `needs_advanced=True` 时

---

## 与 Agentic Fix Loop 的集成

Task 通过两个方法暴露知识能力给 agentic 接口：

```python
task.get_knowledge_provider()  → Provider 实例
task.get_compute_pattern()     → "elementwise" / "matmul" / "normalization" / ...
```

AgenticFixLoopMixin 在 fix 循环中调用 assembly helper，而非重复注入完整 task description：

```
init:            task 组装完整 prompt (Vector ~18K, Cube ~25K)
compile-fix:     provider.assemble_for_compile_fix()           → ~2K (vs 旧方案 27K)
correctness-fix: provider.assemble_for_correctness_fix(pattern)→ ~2-4K  (按范式选择 tiling)
```

---

## 文档结构

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | CANN Provider 完整方法列表、组装流程代码、Agentic Fix Loop 集成细节 |
| [content_reference.md](content_reference.md) | CANN 各知识片段的具体内容、文件大小和设计意图 |

## 代码位置 (CANN)

```
CANN_Parallel_Evaluator/src/cann_parallel_evaluator/
├── cann_init_task.py              # Task 主类 + _OPERATOR_PATTERN_MAP (150 算子预分类)
└── knowledge/
    ├── provider.py                # CANNKnowledgeProvider (范式判断 + 组装)
    ├── api_scanner.py             # CANN SDK 头文件扫描
    ├── api/                       # API 参考 (.md)
    ├── constraints/               # 约束 (.md, 完整版 + 精简版)
    ├── primers/                   # 编程模型 + 11 个模式指南 (.md)
    ├── examples/                  # 11 个精选示例 (.md)
    └── tiling/                    # Vector/Cube/Mixed tiling (.md)
```

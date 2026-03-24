# Evaluator

Ascend C 算子代码的评估系统，支持编译、正确性验证和性能测量。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         构造阶段 (CANNInitTask)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  python_reference                                                        │
│  支持两种格式：                                                            │
│  • org: Model 类 + get_inputs/get_init_inputs  (py_reference/)           │
│  • fn:  module_fn() + Model(fn=module_fn)      (fn_py_reference/)        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌──────────────────┐                                                   │
│  │OperatorSignature│ ──► signature (inputs, outputs, init_params)      │
│  │  Parser (auto)   │     自动检测 fn/org 格式                            │
│  └──────────────────┘                                                   │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                     评估阶段 (evaluate_solution)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                           LLM 生成                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌───────────────────┐  ┌────────────────────────┐ │
│  │   kernel_impl   │  │ kernel_entry_body │  │    tiling_fields       │ │
│  │  (Kernel 类)     │  │  (入口函数体)      │  │    (结构体字段)         │ │
│  └─────────────────┘  └───────────────────┘  └────────────────────────┘ │
│                                                                          │
│  ┌─────────────────┐  ┌───────────────────┐  ┌────────────────────────┐ │
│  │tiling_func_body │  │  infer_shape_body │  │  output_alloc_code     │ │
│  │  (分片计算逻辑)   │  │  (Shape 推断逻辑)  │  │  (输出 tensor 分配)     │ │
│  └─────────────────┘  └───────────────────┘  └────────────────────────┘ │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                         模板生成器                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                          signature                                       │
│                              +                                           │
│                        LLM 生成组件                                      │
│                              │                                           │
│                              ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  AscendCTemplateGenerator                         │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐│   │
│  │  │project_json_src│ │ host_tiling_src│ │   host_operator_src    ││   │
│  │  │    (自动)       │ │ (模板+字段)     │ │  (模板+函数体)          ││   │
│  │  └────────────────┘ └────────────────┘ └────────────────────────┘│   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐│   │
│  │  │   kernel_src   │ │ python_bind_src│ │      model_src         ││   │
│  │  │ (模板+impl)     │ │ (模板+分配代码) │ │       (自动)           ││   │
│  │  └────────────────┘ └────────────────┘ └────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                          评估流水线                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  compile ──► verify_and_measure (单次沙箱) ──► EvaluationResult         │
│               ├─ 正确性验证                                               │
│               └─ 性能测量 (correctness 通过后)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Python Reference 格式

OperatorSignatureParser 自动检测并支持两种 python_reference 格式：

### org 格式 (`py_reference/`)

```python
class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)

def get_inputs():
    return [torch.rand(4096, 393216)]

def get_init_inputs():
    return [1.0]
```

解析方式：`get_inputs()` → inputs，`get_init_inputs()` → init_params，`Model.forward()` → 参数名。

### fn 格式 (`fn_py_reference/`)

```python
def module_fn(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return F.elu(x, alpha=alpha)

class Model(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.alpha)
```

解析方式：`module_fn()` 的参数签名直接区分 tensor/scalar。用 spy 函数拦截 `module_fn` 调用来推断运行时类型。

### 自动检测

```python
mode = "fn" if "def module_fn(" in python_code else "org"
```

两种格式解析后产出相同的 signature 结构：

```python
{
    "op_name": "elu",
    "inputs": [{"name": "x", "dtype": "float", "is_tensor": True}],
    "outputs": [{"name": "output", "dtype": "float", "is_tensor": True}],
    "init_params": [{"name": "alpha", "dtype": "float", "is_tensor": False, "default": 1.0}],
}
```

## LLM 生成组件

### 必填组件（6 个）

| 组件 | 目标文件 | 说明 |
|------|----------|------|
| `kernel_impl` | kernel_src | Kernel 类和辅助代码（Ascend C），可在顶部包含 `#include` |
| `kernel_entry_body` | kernel_src | 入口函数体（GET_TILING_DATA 之后的代码） |
| `tiling_fields` | host_tiling_src | TilingData 结构体字段，支持 list 或 dict 格式（见下文） |
| `tiling_func_body` | host_operator_src | TilingFunc 函数体，可在顶部包含 `#include`（自动提升到文件头） |
| `infer_shape_body` | host_operator_src | InferShape 函数体（推断输出 shape） |
| `output_alloc_code` | python_bind_src | 输出 tensor 分配代码（定义 `result` 变量） |

额外头文件直接写在对应组件中，不需要单独指定：

- `kernel_impl` 顶部写 `#include "lib/matmul_intf.h"` 等
- `tiling_func_body` 顶部写 `#include`，模板自动提取并提升到文件头部
- `tiling_fields` 使用 dict 格式 `{"includes": [...], "fields": [...]}` 嵌入头文件

### tiling_fields 格式

简单算子（无额外头文件）使用 list 格式：

```json
[{"name": "totalLength", "type": "uint32_t"}]
```

需要额外头文件时（如 Cube 算子的 `TCubeTiling`）使用 dict 格式：

```json
{
    "includes": ["lib/tiling_api.h"],
    "fields": [
        {"name": "M", "type": "uint32_t"},
        {"name": "cubeTiling", "type": "TCubeTiling", "is_struct": true}
    ]
}
```

### 模板自动生成的组件

| 组件 | 输入来源 | 说明 |
|------|----------|------|
| `project_json_src` | signature | msopgen 配置 JSON |
| `host_tiling_src` | signature + tiling_fields | Tiling 数据结构头文件 |
| `host_operator_src` | signature + func_bodies | Host 端算子注册 |
| `kernel_src` | signature + kernel_impl + kernel_entry_body | Device Kernel |
| `python_bind_src` | signature + output_alloc | Python/PyTorch 绑定 |
| `model_src` | signature | 测试模型 ModelNew |

> **为什么用 func body 而不是完整文件**：模板中有固定部分（如函数名、命令名）不能被 LLM 随意修改，
> 用 func body 方式确保固定部分由模板控制，LLM 只负责逻辑实现。

## 快速开始

### 代码示例

```python
from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

# 1. 创建任务（构造时自动解析 signature）
task = CANNInitTask(data={
    "op_name": "add",
    "python_reference": PYTHON_REFERENCE,
})

# 2. 配置并评估
config = CANNSolutionConfig(
    kernel_impl="class KernelAdd { ... };",
    kernel_entry_body="    KernelAdd op; ...",
    tiling_fields=[{"name": "totalLength", "type": "uint32_t"}],
    tiling_func_body="...",
    infer_shape_body="...",
    output_alloc_code="at::Tensor result = at::empty_like(x);",
    # 可选执行控制：
    # compile_only=True,           # 仅编译，跳过验证和性能测量
    # load_from="/path/to/saved",  # 加载已保存的编译结果
    # save_compile_to="/path/to/save",  # 保存编译结果供复用
    # skip_correctness=True,       # 跳过正确性验证
    # skip_performance=True,       # 跳过性能测量
)
solution = Solution("", config.to_dict())
result = task.evaluate_solution(solution)

# 3. 检查结果
if result.valid:
    print(f"成功! 运行时间: {result.additional_info['runtime']:.4f} ms")
else:
    print(f"失败 [{result.additional_info['stage']}]: {result.additional_info['error']}")
```

### 前置条件

- 真实 Ascend NPU 环境（如华为云服务器）
- CANN Toolkit 8.0+
- torch-npu 已安装

> WSL/x86 模拟环境不支持完整编译，可用 `fake_mode=True` 仅写入文件。

## 返回结果

```python
# 成功
EvaluationResult(
    valid=True,
    score=-runtime,
    additional_info={
        "stage": "success",
        "runtime": 0.1234,
        "runtime_std": 0.0012,
        "baseline_runtime": 0.1500,    # Python reference 运行时间
        "baseline_std": 0.0015,
        "speedup": 1.21,               # 相对加速比
        "kernel_src": "...",            # 生成的 kernel 代码
        "project_path": "/tmp/xxx",    # 编译项目路径
    }
)

# 失败
EvaluationResult(
    valid=False,
    score=None,
    additional_info={
        "stage": "compile",            # 或 "validation", "correctness", "sandbox", "exception"
        "error": "...",
        "kernel_src": "...",
        "project_path": "/tmp/xxx",
    }
)

# correctness 失败时额外包含
# "python_output": ..., "ascend_output": ..., "max_diff": ...
```

## 内部实现

### 计算范式推断

CANNInitTask 根据算子名推断计算范式（compute pattern），用于组装差异化知识：

```
优先级：data["compute_pattern"]（显式） > _OPERATOR_PATTERN_MAP（150 算子内建表） > 关键词回退

范式 → 编程模型映射：
  vector:  elementwise, reduction, softmax, broadcast, pooling
  cube:    matmul, convolution, attention
  mixed:   normalization, index, resize
```

该推断影响：
- 知识组装内容（primer、约束、API 参考的粒度）
- 是否注入高级 API（Cube/Mixed 自动注入，Vector 仅复杂算子注入）
- 精选示例的选择

### 任务描述分阶段

`get_task_description(phase)` 支持两种阶段：

| 阶段 | 内容 | 场景 |
|------|------|------|
| `init` | 完整教学 prompt：角色 + 签名 + Domain Primer + 约束 + 属性指南 + 精选示例 + Tiling 教程 + 高级 API + 组件规范（含 Add 完整示例） | 从零生成 |
| `evolve` | 精简 prompt：角色 + 签名 + 模式约束 + 属性指南 + 组件格式（无示例） | 优化已有实现 |

### 评估流程

```
CANNInitTask(data=..., parallel=False)
    │
    └─► _process_data()
        ├─ OperatorSignatureParser.parse(mode="auto") ──► self.signature
        └─ AscendCTemplateGenerator(signature)

evaluate_solution(solution)
    │
    ├─► 0. CANNSolutionConfig.from_dict(solution.other_info)
    │
    ├─► 0a. [可选] load_from 路径 → 加载已编译结果，跳过编译
    │
    ├─► 1. 验证 6 个必填字段完整性
    │       └─ 缺失 → stage="validation"
    │
    ├─► 2. TemplateGenerator.generate() ──► full_code (6 组件)
    │
    ├─► 2a. [可选] fake_mode → 仅写入文件，stage="files_written"
    │
    ├─► 3. Evaluator.compile() ──► CompileResult
    │       ├─ ascend_setup() — Step 1-3 (msopgen + 写源文件)
    │       └─ ascend_build(skip_model_exec=True) — Step 4-7 (build + deploy + pybind)
    │       └─ 失败 → stage="compile"
    │
    ├─► 3a. [可选] save_compile_to → 保存编译结果
    │
    ├─► 3b. [可选] compile_only → stage="compile_only"，跳过验证
    │
    └─► 4. _run_verify_and_perf()
            ├─ parallel=False（默认）: verify_and_measure() 单次沙箱调用
            └─ parallel=True: verify_correctness() + measure_performance() 拆分调用
               └─ 支持多设备并行（device pool 保证每设备同时只有一个评估）
```

> **注意**：串行模式（默认）下正确性和性能合并在一次沙箱调用中执行，避免两次 ~10-20s 的 NPU 初始化开销。

### 多设备并行评估

```python
# 初始化 8 卡并行
CANNInitTask.init_device_pool([0, 1, 2, 3, 4, 5, 6, 7])

# 创建任务时启用 parallel 模式
task = CANNInitTask(data=..., parallel=True)
```

device pool 机制：
- `_acquire_device()` 从池中获取设备（阻塞等待）
- `_release_device()` 归还设备
- 未显式初始化时，自动创建单设备池 `[0]`

### 超时配置

| 阶段 | 默认超时 | 代码位置 |
|------|----------|----------|
| msopgen | 60s | `ascend_setup()` |
| build.sh | 180s | `ascend_build()` Step 4 |
| deploy (.run) | 60s | `ascend_build()` Step 5 |
| pybind (build_and_run.sh) | 120s | `ascend_build()` Step 6 |
| 沙箱总体 | 600s | `CANNSandboxExecutor` |

### 关键类

| 类 | 位置 | 职责 |
|---|------|------|
| `CANNInitTask` | cann_init_task.py | 对外入口（构造时解析 signature） |
| `AscendCEvaluator` | evaluator.py | 编译/验证/测量 |
| `CANNSandboxExecutor` | utils/backend/sandbox.py | 沙箱子进程管理 |
| `AscendCTemplateGenerator` | utils/templates/generator.py | 6 组件生成协调器 |
| `KernelSrcGenerator` | utils/templates/kernel_src.py | kernel_src 模板生成 |
| `OperatorSignatureParser` | signature_parser.py | 签名解析（fn/org 双格式） |
| `CompileResult` | data_structures.py | 编译结果（可序列化） |
| `CANNSolutionConfig` | data_structures.py | 配置封装（6 组件 + 执行控制） |
| `CANNKnowledgeProvider` | knowledge/provider.py | 领域知识组装：assemble_for_init/evolve()，按范式差异化 |

### 后端模块

| 模块 | 位置 | 职责 |
|------|------|------|
| ascend_compile | utils/backend/ascend_compile.py | 8 步编译流程（ascend_setup + ascend_build） |
| correctness | utils/backend/correctness.py | Model vs ModelNew 对比 |
| performance | utils/backend/performance.py | 多轮计时测量 |
| sandbox | utils/backend/sandbox.py | spawn 子进程管理 |

### 沙箱隔离

所有评估在 `multiprocessing.spawn` 子进程执行：

| 目标 | 说明 |
|------|------|
| 防止环境污染 | `exec()` 不影响主进程 `sys.modules` |
| 防止崩溃扩散 | segfault 不影响主进程 |
| 超时控制 | 可强制终止超时任务 |
| 资源泄漏隔离 | 子进程结束自动释放资源 |

详见：[sandbox_design.md](sandbox_design.md)

### 并行编译

批量编译时需注意 CMake 竞争条件，使用延时启动：

```python
delay = index * 2.0  # 0s, 2s, 4s...
time.sleep(delay)
task.evaluate_solution(solution)
```

详见：[compile_pipeline.md](compile_pipeline.md#并行编译)

## 文档导航

| 文档 | 说明 |
|------|------|
| [api.md](api.md) | API 参考 + 模板详解 |
| [compile_pipeline.md](compile_pipeline.md) | 编译流水线 8 步详解 + 并行编译 |
| [sandbox_design.md](sandbox_design.md) | 沙箱隔离机制 |

## 代码结构

```
CANN_Parallel_Evaluator/src/cann_parallel_evaluator/
├── __init__.py
├── cann_init_task.py        # CANNInitTask 入口
├── evaluator.py             # AscendCEvaluator
├── signature_parser.py      # OperatorSignatureParser（支持 fn/org 格式）
├── data_structures.py       # CompileResult, CANNSolutionConfig
├── knowledge/               # 领域知识系统
│   ├── provider.py          # CANNKnowledgeProvider（知识组装入口）
│   ├── api_scanner.py       # CANN SDK 头文件扫描
│   ├── api/                 # API 参考文档
│   │   ├── quick_reference.md
│   │   └── advanced_reference.md
│   ├── constraints/         # 关键约束（避免常见错误）
│   │   ├── critical_full.md
│   │   └── critical_compact.md
│   ├── tiling/              # Tiling 知识
│   │   ├── fundamentals.md
│   │   ├── multidim_fundamentals.md
│   │   ├── cube_fundamentals.md
│   │   ├── edge_cases.md
│   │   └── quick_reference.md
│   ├── examples/            # 精选完整算子示例（.py 代码 + .md 讲解）
│   │   ├── curated_examples.py
│   │   ├── relu.md, add.md, softmax.md, reduce_sum.md,
│   │   ├── gather.md, layer_norm.md, pooling.md, resize.md,
│   │   └── matmul.md, convolution.md, attention.md
│   └── primers/             # 编程模型 + 按范式的模式指南
│       ├── level0_programming_model.py  # 通用编程模型（加载同名 .md）
│       ├── level0_programming_model.md  # 编程模型内容
│       ├── level1_patterns.py           # 模式注册
│       └── elementwise.md, softmax.md, reduction.md, broadcast.md,
│           pooling.md, normalization.md, matmul.md, convolution.md,
│           attention.md, index.md, resize.md, other.md
├── utils/
│   ├── backend/             # 编译/评估后端
│   │   ├── ascend_compile.py  # ascend_setup + ascend_build
│   │   ├── correctness.py
│   │   ├── performance.py
│   │   └── sandbox.py       # CANNSandboxExecutor
│   └── templates/           # 模板生成
│       ├── base.py          # 模板基类
│       ├── generator.py     # AscendCTemplateGenerator
│       ├── kernel_src.py
│       ├── host_tiling.py
│       ├── host_operator.py
│       ├── python_bind.py
│       ├── project_json.py
│       ├── model_src.py
│       └── pybind_templates/
│           └── pytorch_npu_helper.hpp  # PyTorch NPU 辅助头文件

# 方法级接口（位于 cann-benchmark）
cann-benchmark/src/cann_benchmark/cann_init/method_interface/
├── agentic_mixin.py     # AgenticFixLoopMixin（编译/正确性反馈循环）
├── agentic_funsearch_interface.py
├── funsearch_interface.py
├── eoh_interface.py
└── evoengineer/
    ├── interface.py
    ├── agentic_interface.py
    ├── prompts.py
    └── parser.py
```

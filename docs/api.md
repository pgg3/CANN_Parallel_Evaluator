# API 参考

## 核心类

| 类 | 说明 |
|----|------|
| `CANNInitTask` | 主入口。构造时自动解析算子签名，编排完整的评估流水线。 |
| `CANNSolutionConfig` | 封装 LLM 生成的 6 个组件和执行控制参数。 |
| `AscendCEvaluator` | 底层评估器：`compile()`、`verify_correctness()`、`measure_performance()`。 |
| `AscendCTemplateGenerator` | 将 6 个逻辑组件组装为完整源文件（基于算子签名的模板）。 |
| `OperatorSignatureParser` | 解析 Python Reference 代码，提取算子签名（inputs、outputs、init_params）。支持 `fn` 和 `org` 两种格式。 |
| `CompileResult` | 可序列化的编译结果。支持 `save()` / `load()` 缓存编译产物。 |
| `CANNKnowledgeProvider` | 按计算范式组装领域知识（编程模型、约束、API 参考、精选示例）。 |

## CANNInitTask

### 构造函数

```python
CANNInitTask(
    data: dict,                    # 必须包含 "op_name" 和 "python_reference"
    project_path: str = None,      # 自定义构建目录（默认：临时目录）
    fake_mode: bool = False,       # 仅写入文件，跳过编译
    verbose: bool = True,          # 打印编译输出
    parallel: bool = False,        # 拆分验证和性能测量，用于并行执行
)
```

### 设备池（类方法）

```python
# 用可用的 NPU 设备 ID 初始化
CANNInitTask.init_device_pool([0, 1, 2, 3, 4, 5, 6, 7])
```

- `init_device_pool(device_ids)` — 填充设备池
- `_acquire_device()` — 从池中获取设备，无空闲时阻塞等待（基于线程安全的 `queue.Queue`）
- `_release_device(device)` — 将设备归还到池中
- 未显式初始化时，首次访问自动创建单设备池 `[0]`

## CANNSolutionConfig

### LLM 生成组件（必填）

| 字段 | 目标文件 | 说明 |
|------|----------|------|
| `kernel_impl` | kernel_src | Kernel 类和辅助代码（Ascend C）。顶部可包含 `#include` 指令。 |
| `kernel_entry_body` | kernel_src | 入口函数体（`GET_TILING_DATA` 之后的代码）。 |
| `tiling_fields` | host_tiling_src | TilingData 结构体字段。支持字段 dict 列表，或带 `"fields"` + `"includes"` 键的 dict。 |
| `tiling_func_body` | host_operator_src | Host 端分片计算逻辑。顶部的 `#include` 会自动提取到文件头。 |
| `infer_shape_body` | host_operator_src | 输出 Shape 推断逻辑。 |
| `output_alloc_code` | python_bind_src | 输出 tensor 分配代码（必须定义 `result` 变量）。 |

### 执行控制（可选）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `compile_only` | `False` | 仅编译，跳过正确性验证和性能测量 |
| `skip_correctness` | `False` | 跳过正确性验证 |
| `skip_performance` | `False` | 跳过性能测量 |
| `load_from` | `None` | 从指定路径加载已编译的结果 |
| `save_compile_to` | `None` | 将编译结果保存到指定路径，供后续复用 |

### 序列化

```python
config.to_dict()                    # 转为 dict（用于 Solution.other_info）
CANNSolutionConfig.from_dict(d)     # 从 dict 重建
```

## 评估结果

### 成功

```python
EvaluationResult(
    valid=True,
    score=-runtime,                    # 负的运行时间（越小越好）
    additional_info={
        "stage": "success",
        "runtime": 0.1234,             # Ascend C kernel 运行时间 (ms)
        "runtime_std": 0.0012,
        "baseline_runtime": 0.1500,    # Python Reference 运行时间 (ms)
        "baseline_std": 0.0015,
        "speedup": 1.21,               # 相对加速比
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

### 失败

```python
EvaluationResult(
    valid=False,
    score=None,
    additional_info={
        "stage": "compile",            # 或 "validation", "correctness",
                                       #    "sandbox", "performance", "exception"
        "error": "...",
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

当 `stage="correctness"` 时，包含额外的诊断字段：

| 字段 | 说明 |
|------|------|
| `python_output` | Python Reference 模型的输出 |
| `ascend_output` | Ascend C kernel 的输出 |
| `max_diff` | 逐元素最大差异 |

### 特殊阶段

| 阶段 | 含义 |
|------|------|
| `compile_only` | 编译成功；跳过了验证（`compile_only=True`） |
| `correctness_only` | 正确性通过；跳过了性能测量（`skip_performance=True`） |
| `files_written` | 文件已写入磁盘；跳过了编译（`fake_mode=True`） |

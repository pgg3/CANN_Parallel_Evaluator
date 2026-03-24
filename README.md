# CANN Parallel Evaluator

[English](README_en.md) | **中文**

面向昇腾 C 算子的编译、正确性验证、性能测量框架，内置多 NPU 并行评估支持。

基于 [evotoolkit](https://github.com/pgg3/evotoolkit) 构建，现已作为独立包发布。

## 特性

- **一体化流水线** — 编译、正确性校验、性能测量一次调用完成
- **多设备并行** — 设备池机制将评估分发到多张 NPU，保证每卡独占访问
- **沙箱隔离** — `multiprocessing.spawn` 子进程运行，OOM / segfault 不影响主进程
- **模板驱动** — LLM 只需提供 6 个逻辑组件，模板自动处理所有样板代码

## 环境要求

- 昇腾 NPU + **CANN Toolkit 8.0+**
- **torch-npu**
- Python 3.10+

## 安装

```bash
git clone https://github.com/pgg3/CANN_Parallel_Evaluator.git
pip install -e ./CANN_Parallel_Evaluator
```

> CANN 任务无额外 Python 依赖，CANN Toolkit 和 torch-npu 需在系统级安装。

## 快速开始

### 单设备评估

框架支持两种 Python Reference 格式（自动检测）：
- **org 格式** — Model 类 + `get_inputs` / `get_init_inputs`（[relu_org.py](examples/relu_org.py)）
- **fn 格式** — `module_fn` + Model 代理调用（[elu_fn.py](examples/elu_fn.py)）

完整的可运行示例（含真实的 6 组件代码）见 [relu_complete.py](examples/relu_complete.py)，以下为精简版：

```python
from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": open("examples/relu_org.py").read(),
})

config = CANNSolutionConfig(
    kernel_impl=KERNEL_IMPL,             # Ascend C Kernel 类
    kernel_entry_body=KERNEL_ENTRY_BODY, # 入口函数体
    tiling_fields=TILING_FIELDS,         # TilingData 结构体字段
    tiling_func_body=TILING_FUNC_BODY,   # Host 端分片计算逻辑
    infer_shape_body=INFER_SHAPE_BODY,   # 输出 Shape 推断
    output_alloc_code=OUTPUT_ALLOC_CODE, # 输出 tensor 分配
)

result = task.evaluate_solution(Solution("", config.to_dict()))

if result.valid:
    info = result.additional_info
    print(f"Runtime: {info['runtime']:.4f} ms, Speedup: {info['speedup']:.3f}x")
```

### 多设备并行评估

```python
from concurrent.futures import ThreadPoolExecutor

# 初始化设备池（类级别，只需调用一次）
CANNInitTask.init_device_pool([0, 1, 2, 3, 4, 5, 6, 7])

def evaluate_one(op_data, components):
    task = CANNInitTask(data=op_data, parallel=True)
    return task.evaluate_solution(
        Solution("", CANNSolutionConfig(**components).to_dict())
    )

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(evaluate_one, op, comp)
               for op, comp in zip(operators, component_list)]
    results = [f.result() for f in futures]
```

**工作原理：**

```
线程 1 ──► 获取(npu:0) ──► 评估 ──► 释放(npu:0)
线程 2 ──► 获取(npu:1) ──► 评估 ──► 释放(npu:1)
线程 3 ──► 获取(阻塞等待空闲设备) ──► ...
```

设备池基于线程安全的 `queue.Queue`，保证每张卡同一时刻最多运行一个评估任务。

## 评估结果

`evaluate_solution()` 返回 `EvaluationResult`：

**成功：**

```python
EvaluationResult(
    valid=True,
    score=-10.4015,                    # 负的运行时间（越小越好，用于排序）
    additional_info={
        "stage": "success",
        "runtime": 10.4015,            # Ascend C kernel 运行时间 (ms)
        "runtime_std": 0.0243,
        "baseline_runtime": 10.0402,   # Python Reference 运行时间 (ms)
        "baseline_std": 0.0152,
        "speedup": 0.965,              # 相对加速比
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

**失败：**

```python
EvaluationResult(
    valid=False,
    score=None,
    additional_info={
        "stage": "compile",            # 失败阶段：compile / validation / correctness / sandbox / exception
        "error": "...",                # 错误信息
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
        # correctness 失败时额外包含：
        # "python_output": ..., "ascend_output": ..., "max_diff": 0.0023,
    }
)
```

详细的 stage 定义和字段说明见 [API 参考](docs/api.md#评估结果)。

## 文档

| 文档 | 内容 |
|------|------|
| [API 参考](docs/api.md) | 构造函数参数、Solution 配置字段、设备池方法、返回结果结构 |
| [架构设计](docs/architecture.md) | 评估流水线、沙箱设计、串行 vs 并行模式 |
| [编译流水线](docs/compile_pipeline.md) | 8 步编译流水线 + 并行编译 |
| [沙箱设计](docs/sandbox_design.md) | 沙箱隔离机制 |
| [工程化改进](docs/engineering.md) | 与 MultiKernelBench 的对比 |
| [知识系统](docs/knowledge/README.md) | CANNKnowledgeProvider 设计 |

## 许可证

[MIT](LICENSE)

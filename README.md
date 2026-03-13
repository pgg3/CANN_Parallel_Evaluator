# CANN Parallel Evaluator

[English](README_en.md) | **中文**

面向昇腾 C 算子的编译、正确性验证、性能测量框架，内置多 NPU 并行评估支持。

基于 [evotoolkit](https://github.com/pgg3/evotoolkit) 构建。

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
git clone https://github.com/pgg3/evotoolkit.git
cd evotoolkit
git checkout feature/cann-init
pip install -e .
```

> CANN 任务无额外 Python 依赖，CANN Toolkit 和 torch-npu 需在系统级安装。

## 快速开始

### 单设备评估

框架支持两种 Python Reference 格式（自动检测），完整示例见 [`examples/`](examples/)：
- **org 格式** — Model 类 + `get_inputs` / `get_init_inputs`（[relu_org.py](examples/relu_org.py)）
- **fn 格式** — `module_fn` + Model 代理调用（[elu_fn.py](examples/elu_fn.py)）

```python
from pathlib import Path
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution

# 读取 Python Reference（org 或 fn 格式均可）
python_ref = Path("examples/relu_org.py").read_text()

task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": python_ref,
})

config = CANNSolutionConfig(
    kernel_impl="...", kernel_entry_body="...",
    tiling_fields=[{"name": "totalLength", "type": "uint32_t"}],
    tiling_func_body="...", infer_shape_body="...",
    output_alloc_code='at::Tensor result = at::empty_like(x);',
)

result = task.evaluate_solution(Solution("", config.to_dict()))

if result.valid:
    print(f"运行时间: {result.additional_info['runtime']:.4f} ms")
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

## 文档

| 文档 | 内容 |
|------|------|
| [API 参考](docs/api.md) | 构造函数参数、Solution 配置字段、设备池方法、返回结果结构 |
| [架构设计](docs/architecture.md) | 评估流水线、沙箱设计、串行 vs 并行模式 |

## 许可证

[MIT](LICENSE)

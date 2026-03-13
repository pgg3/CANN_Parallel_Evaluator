# CANN Parallel Evaluator

**English** | [中文](README.md)

A compile, correctness verification, and performance measurement framework for Ascend C operators, with built-in multi-NPU parallel evaluation support.

Built on [evotoolkit](https://github.com/pgg3/evotoolkit).

## Features

- **Unified pipeline** — compile, correctness check, and performance measurement in one call
- **Multi-device parallel** — device pool distributes evaluations across NPUs with exclusive access
- **Sandbox isolation** — `multiprocessing.spawn` subprocesses prevent OOM / segfault propagation
- **Template-driven** — LLM provides 6 logic components; templates handle all boilerplate

## Prerequisites

- Ascend NPU + **CANN Toolkit 8.0+**
- **torch-npu**
- Python 3.10+

## Installation

```bash
git clone https://github.com/pgg3/evotoolkit.git
cd evotoolkit
git checkout feature/cann-init
pip install -e .
```

> The CANN task has no extra Python dependencies. CANN Toolkit and torch-npu must be installed at the system level.

## Quick Start

### Single Device

```python
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution

task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": PYTHON_REFERENCE_CODE,
})

config = CANNSolutionConfig(
    kernel_impl="...", kernel_entry_body="...",
    tiling_fields=[{"name": "totalLength", "type": "uint32_t"}],
    tiling_func_body="...", infer_shape_body="...",
    output_alloc_code='at::Tensor result = at::empty_like(x);',
)

result = task.evaluate_solution(Solution("", config.to_dict()))

if result.valid:
    print(f"Runtime: {result.additional_info['runtime']:.4f} ms")
```

### Multi-Device Parallel

```python
from concurrent.futures import ThreadPoolExecutor

# Initialize device pool once (class-level)
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

**How it works:**

```
Thread 1 ──► acquire(npu:0) ──► evaluate ──► release(npu:0)
Thread 2 ──► acquire(npu:1) ──► evaluate ──► release(npu:1)
Thread 3 ──► acquire(blocks until free) ──► ...
```

The pool is a thread-safe `queue.Queue`. Each device runs at most one evaluation at a time.

## Documentation

| Document | Content |
|----------|---------|
| [API Reference](docs/api.md) | Constructor parameters, solution config fields, device pool methods, result structure |
| [Architecture](docs/architecture.md) | Evaluation pipeline, sandbox design, serial vs parallel modes |

## License

[MIT](LICENSE)

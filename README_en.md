# CANN Parallel Evaluator

**English** | [中文](README.md)

A compile, correctness verification, and performance measurement framework for Ascend C operators, with built-in multi-NPU parallel evaluation support.

Built on [evotoolkit](https://github.com/pgg3/evotoolkit), now published as a standalone package.

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
git clone https://github.com/pgg3/CANN_Parallel_Evaluator.git
pip install -e ./CANN_Parallel_Evaluator
```

> The CANN task has no extra Python dependencies. CANN Toolkit and torch-npu must be installed at the system level.

## Quick Start

### Single Device

Two Python reference formats are supported (auto-detected):
- **org format** — Model class + `get_inputs` / `get_init_inputs` ([relu_org.py](examples/relu_org.py))
- **fn format** — `module_fn` + Model proxy call ([elu_fn.py](examples/elu_fn.py))

For a complete runnable example with all 6 real components, see [relu_complete.py](examples/relu_complete.py). Minimal version:

```python
from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": open("examples/relu_org.py").read(),
})

config = CANNSolutionConfig(
    kernel_impl=KERNEL_IMPL,             # Ascend C Kernel class
    kernel_entry_body=KERNEL_ENTRY_BODY, # Entry function body
    tiling_fields=TILING_FIELDS,         # TilingData struct fields
    tiling_func_body=TILING_FUNC_BODY,   # Host-side tiling logic
    infer_shape_body=INFER_SHAPE_BODY,   # Output shape inference
    output_alloc_code=OUTPUT_ALLOC_CODE, # Output tensor allocation
)

result = task.evaluate_solution(Solution("", config.to_dict()))

if result.valid:
    info = result.additional_info
    print(f"Runtime: {info['runtime']:.4f} ms, Speedup: {info['speedup']:.3f}x")
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

## Evaluation Result

`evaluate_solution()` returns an `EvaluationResult`:

**Success:**

```python
EvaluationResult(
    valid=True,
    score=-10.4015,                    # Negative runtime (lower is better, used for ranking)
    additional_info={
        "stage": "success",
        "runtime": 10.4015,            # Ascend C kernel runtime (ms)
        "runtime_std": 0.0243,
        "baseline_runtime": 10.0402,   # Python reference runtime (ms)
        "baseline_std": 0.0152,
        "speedup": 0.965,              # Relative speedup
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

**Failure:**

```python
EvaluationResult(
    valid=False,
    score=None,
    additional_info={
        "stage": "compile",            # Failure stage: compile / validation / correctness / sandbox / exception
        "error": "...",                # Error message
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
        # Additional fields when stage="correctness":
        # "python_output": ..., "ascend_output": ..., "max_diff": 0.0023,
    }
)
```

See [API Reference](docs/api.md) for detailed stage definitions and field descriptions.

## Documentation

| Document | Content |
|----------|---------|
| [API Reference](docs/api.md) | Constructor parameters, solution config fields, device pool methods, result structure |
| [Architecture](docs/architecture.md) | Evaluation pipeline, sandbox design, serial vs parallel modes |
| [Compile Pipeline](docs/compile_pipeline.md) | 8-step compile pipeline + parallel compilation |
| [Sandbox Design](docs/sandbox_design.md) | Sandbox isolation mechanism |
| [Engineering](docs/engineering.md) | Comparison with MultiKernelBench |
| [Knowledge System](docs/knowledge/README.md) | CANNKnowledgeProvider design |

## License

[MIT](LICENSE)

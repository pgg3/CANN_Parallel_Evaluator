# CANN Parallel Evaluator

A compilation, correctness verification, and performance measurement framework for Ascend C operators, with built-in multi-NPU parallel evaluation support.

Part of [evotoolkit](https://github.com/pgg3/evotoolkit) — an LLM-driven code evolution toolkit.

## Features

- **Unified pipeline** — compile, verify correctness, and measure performance in a single `evaluate_solution()` call
- **Multi-device parallel evaluation** — device pool mechanism distributes evaluations across NPUs, guaranteeing exclusive access per device
- **Sandbox isolation** — each evaluation runs in a `multiprocessing.spawn` subprocess, preventing OOM, segfaults, and resource leaks from affecting the main process
- **Template-driven code generation** — LLM provides 6 logic components; templates handle boilerplate (function signatures, build config, PyTorch bindings)
- **Flexible execution control** — compile-only mode, skip correctness/performance, save/load compiled artifacts

## Prerequisites

- **Ascend NPU** (e.g., Ascend 910B) with **CANN Toolkit 8.0+** installed
- **torch-npu** (PyTorch with Ascend NPU backend)
- **Python 3.10+**

> WSL / x86 environments do not support full compilation. Use `fake_mode=True` to write files without compiling.

## Installation

```bash
pip install evotoolkit
```

Or install from source:

```bash
git clone https://github.com/pgg3/evotoolkit.git
cd evotoolkit
pip install -e .
```

No extra dependencies are required for the CANN task — the `cann_init` optional group is empty. The CANN Toolkit and torch-npu must be installed at the system level.

## Quick Start

### Single-Device Evaluation

```python
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution

# 1. Create a task (automatically parses operator signature from Python reference)
task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": PYTHON_REFERENCE_CODE,
})

# 2. Configure the 6 LLM-generated components
config = CANNSolutionConfig(
    kernel_impl="class KernelRelu { ... };",
    kernel_entry_body="    KernelRelu op; op.Init(...); op.Process();",
    tiling_fields=[{"name": "totalLength", "type": "uint32_t"}],
    tiling_func_body="...",
    infer_shape_body="...",
    output_alloc_code="at::Tensor result = at::empty_like(x);",
)

# 3. Evaluate: compile → verify correctness → measure performance
solution = Solution("", config.to_dict())
result = task.evaluate_solution(solution)

# 4. Check result
if result.valid:
    info = result.additional_info
    print(f"Passed! Runtime: {info['runtime']:.4f} ms, Speedup: {info['speedup']:.2f}x")
else:
    info = result.additional_info
    print(f"Failed at [{info['stage']}]: {info['error']}")
```

### Multi-Device Parallel Evaluation

This is the key feature for scaling evaluations across multiple NPU cards. The device pool ensures at most one evaluation runs on each device at a time, preventing OOM from concurrent workloads.

```python
from concurrent.futures import ThreadPoolExecutor
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution

# 1. Initialize the device pool (class-level, call once)
CANNInitTask.init_device_pool([0, 1, 2, 3, 4, 5, 6, 7])  # 8 NPUs

# 2. Create tasks with parallel=True
#    This splits correctness and performance into separate sandbox calls,
#    enabling concurrent evaluation across devices.
def evaluate_operator(op_data, components):
    task = CANNInitTask(data=op_data, parallel=True)
    config = CANNSolutionConfig(**components)
    solution = Solution("", config.to_dict())
    return task.evaluate_solution(solution)

# 3. Run evaluations concurrently
operators = [
    {"op_name": "relu", "python_reference": "..."},
    {"op_name": "gelu", "python_reference": "..."},
    {"op_name": "sigmoid", "python_reference": "..."},
    # ... more operators
]
component_list = [...]  # Corresponding LLM-generated components

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [
        pool.submit(evaluate_operator, op, comp)
        for op, comp in zip(operators, component_list)
    ]
    results = [f.result() for f in futures]
```

#### How Device Pool Works

```
Thread 1 ──► _acquire_device() ──► npu:0 ──► evaluate ──► _release_device(npu:0)
Thread 2 ──► _acquire_device() ──► npu:1 ──► evaluate ──► _release_device(npu:1)
Thread 3 ──► _acquire_device() ──► (blocks until a device is free)
...
```

- `init_device_pool([0, 1, ...])` — populate the pool with available NPU IDs
- `_acquire_device()` — blocks until a device is free (thread-safe `queue.Queue`)
- `_release_device()` — returns the device to the pool
- If no pool is explicitly initialized, a single-device pool `[0]` is created on first use

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `CANNInitTask` | Main entry point. Parses operator signature on construction, orchestrates the full evaluate pipeline. |
| `CANNSolutionConfig` | Wraps the 6 LLM-generated components + execution control flags (`compile_only`, `skip_correctness`, etc.). |
| `AscendCEvaluator` | Low-level evaluator that handles compilation (`compile()`), correctness (`verify_correctness()`), and performance (`measure_performance()`). |
| `AscendCTemplateGenerator` | Assembles 6 logic components into complete source files using operator-specific templates. |
| `OperatorSignatureParser` | Parses Python reference code to extract operator signature (inputs, outputs, init_params). Supports `fn` and `org` formats. |
| `CompileResult` | Serializable compilation result. Supports `save()` / `load()` for caching compiled artifacts. |
| `CANNKnowledgeProvider` | Assembles domain knowledge (primers, constraints, API references, examples) by compute pattern. |

### CANNInitTask Constructor

```python
CANNInitTask(
    data: dict,                    # Must contain "op_name" and "python_reference"
    project_path: str = None,      # Custom build directory (default: temp dir)
    fake_mode: bool = False,       # Write files only, skip compilation
    verbose: bool = True,          # Print compilation output
    parallel: bool = False,        # Split verify + perf for parallel execution
)
```

### CANNSolutionConfig Fields

**LLM-generated components (required):**

| Field | Target File | Description |
|-------|-------------|-------------|
| `kernel_impl` | kernel_src | Kernel class and helper code (Ascend C) |
| `kernel_entry_body` | kernel_src | Entry function body (after `GET_TILING_DATA`) |
| `tiling_fields` | host_tiling_src | TilingData struct fields (list or dict with includes) |
| `tiling_func_body` | host_operator_src | Host-side tiling computation logic |
| `infer_shape_body` | host_operator_src | Output shape inference logic |
| `output_alloc_code` | python_bind_src | Output tensor allocation (defines `result`) |

**Execution control (optional):**

| Field | Default | Description |
|-------|---------|-------------|
| `compile_only` | `False` | Only compile, skip verification and performance |
| `skip_correctness` | `False` | Skip correctness check |
| `skip_performance` | `False` | Skip performance measurement |
| `load_from` | `None` | Load pre-compiled result from path |
| `save_compile_to` | `None` | Save compiled result for reuse |

### Device Pool (Class Methods)

```python
CANNInitTask.init_device_pool([0, 1, 2, 3])  # Initialize with NPU device IDs
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      CANNInitTask                                    │
│                                                                      │
│  python_reference ──► OperatorSignatureParser ──► signature          │
│                                                                      │
│  evaluate_solution(solution)                                         │
│    │                                                                 │
│    ├─ 1. Validate 6 required components                              │
│    ├─ 2. AscendCTemplateGenerator.generate() ──► full source code    │
│    ├─ 3. AscendCEvaluator.compile() ──► CompileResult                │
│    │     ├─ msopgen (project scaffold)                               │
│    │     ├─ write source files                                       │
│    │     ├─ build.sh (cmake + make)                                  │
│    │     ├─ deploy (install .run package)                             │
│    │     └─ pybind (build Python extension)                          │
│    └─ 4. Verify & Measure (sandbox subprocess)                       │
│          ├─ correctness: Model vs ModelNew output comparison          │
│          └─ performance: multi-round timing with warmup               │
│                                                                      │
│  Device Pool (multi-NPU)                                             │
│    _acquire_device() ──► npu:X ──► evaluate ──► _release_device()    │
└──────────────────────────────────────────────────────────────────────┘
```

### Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Serial (`parallel=False`) | Correctness + performance in a single sandbox call | Default, avoids double NPU init overhead |
| Parallel (`parallel=True`) | Separate sandbox calls for correctness and performance | Multi-device concurrent evaluation |

### Sandbox Isolation

All evaluations run in `multiprocessing.spawn` subprocesses:
- Prevents `exec()`-based environment pollution
- Contains segfaults within the subprocess
- Enforces timeouts (configurable per stage)
- Automatically reclaims resources on subprocess exit

## Evaluation Result

### Success

```python
EvaluationResult(
    valid=True,
    score=-runtime,              # Negative runtime (lower = better)
    additional_info={
        "stage": "success",
        "runtime": 0.1234,       # Ascend C kernel runtime (ms)
        "runtime_std": 0.0012,
        "baseline_runtime": 0.15,  # Python reference runtime (ms)
        "baseline_std": 0.0015,
        "speedup": 1.21,          # Relative speedup
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

### Failure

```python
EvaluationResult(
    valid=False,
    score=None,
    additional_info={
        "stage": "compile",      # or "validation", "correctness", "sandbox", "exception"
        "error": "...",
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

When `stage="correctness"`, additional fields are included:

```python
{
    "python_output": ...,
    "ascend_output": ...,
    "max_diff": 0.0023,
}
```

## License

[MIT](LICENSE)

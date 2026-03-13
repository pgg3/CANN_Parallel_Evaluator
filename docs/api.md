# API Reference

## Core Classes

| Class | Description |
|-------|-------------|
| `CANNInitTask` | Main entry point. Parses operator signature on construction, orchestrates the full evaluation pipeline. |
| `CANNSolutionConfig` | Wraps the 6 LLM-generated components and execution control flags. |
| `AscendCEvaluator` | Low-level evaluator: `compile()`, `verify_correctness()`, `measure_performance()`. |
| `AscendCTemplateGenerator` | Assembles 6 logic components into complete source files using operator-specific templates. |
| `OperatorSignatureParser` | Parses Python reference code to extract operator signature (inputs, outputs, init_params). Supports `fn` and `org` formats. |
| `CompileResult` | Serializable compilation result. Supports `save()` / `load()` for caching compiled artifacts. |
| `CANNKnowledgeProvider` | Assembles domain knowledge (primers, constraints, API references, curated examples) by compute pattern. |

## CANNInitTask

### Constructor

```python
CANNInitTask(
    data: dict,                    # Must contain "op_name" and "python_reference"
    project_path: str = None,      # Custom build directory (default: temp dir)
    fake_mode: bool = False,       # Write files only, skip compilation
    verbose: bool = True,          # Print compilation output
    parallel: bool = False,        # Split verify + perf for parallel execution
)
```

### Device Pool (Class Methods)

```python
# Initialize with available NPU device IDs
CANNInitTask.init_device_pool([0, 1, 2, 3, 4, 5, 6, 7])
```

- `init_device_pool(device_ids)` — populate the pool with NPU IDs
- `_acquire_device()` — blocks until a device is free (thread-safe `queue.Queue`)
- `_release_device(device)` — returns the device to the pool
- If no pool is explicitly initialized, a single-device pool `[0]` is created on first use

## CANNSolutionConfig

### LLM-Generated Components (required)

| Field | Target File | Description |
|-------|-------------|-------------|
| `kernel_impl` | kernel_src | Kernel class and helper code (Ascend C). May include `#include` directives at the top. |
| `kernel_entry_body` | kernel_src | Entry function body (code after `GET_TILING_DATA`). |
| `tiling_fields` | host_tiling_src | TilingData struct fields. Accepts a list of field dicts, or a dict with `"fields"` + `"includes"` keys. |
| `tiling_func_body` | host_operator_src | Host-side tiling computation logic. `#include` at the top is auto-extracted to file header. |
| `infer_shape_body` | host_operator_src | Output shape inference logic. |
| `output_alloc_code` | python_bind_src | Output tensor allocation code (must define a `result` variable). |

### Execution Control (optional)

| Field | Default | Description |
|-------|---------|-------------|
| `compile_only` | `False` | Only compile, skip correctness and performance |
| `skip_correctness` | `False` | Skip correctness verification |
| `skip_performance` | `False` | Skip performance measurement |
| `load_from` | `None` | Load pre-compiled result from path |
| `save_compile_to` | `None` | Save compiled result for reuse |

### Serialization

```python
config.to_dict()                    # Convert to dict (for Solution.other_info)
CANNSolutionConfig.from_dict(d)     # Reconstruct from dict
```

## Evaluation Result

### Success

```python
EvaluationResult(
    valid=True,
    score=-runtime,                    # Negative runtime (lower is better)
    additional_info={
        "stage": "success",
        "runtime": 0.1234,             # Ascend C kernel runtime (ms)
        "runtime_std": 0.0012,
        "baseline_runtime": 0.1500,    # Python reference runtime (ms)
        "baseline_std": 0.0015,
        "speedup": 1.21,               # Relative speedup vs Python reference
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
        "stage": "compile",            # or "validation", "correctness",
                                       #    "sandbox", "performance", "exception"
        "error": "...",
        "kernel_src": "...",
        "project_path": "/tmp/cann_relu_xxx",
    }
)
```

When `stage="correctness"`, additional diagnostic fields are included:

| Field | Description |
|-------|-------------|
| `python_output` | Output from the Python reference model |
| `ascend_output` | Output from the Ascend C kernel |
| `max_diff` | Maximum element-wise difference |

### Special Stages

| Stage | Meaning |
|-------|---------|
| `compile_only` | Compilation succeeded; verification was skipped (`compile_only=True`) |
| `correctness_only` | Correctness passed; performance was skipped (`skip_performance=True`) |
| `files_written` | Files written to disk; compilation was skipped (`fake_mode=True`) |

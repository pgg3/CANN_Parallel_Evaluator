# Architecture

## Evaluation Pipeline

```
CANNInitTask(data, parallel=False)
    │
    └─► _process_data()
        ├─ OperatorSignatureParser.parse(mode="auto") ──► signature
        └─ AscendCTemplateGenerator(signature)

evaluate_solution(solution)
    │
    ├─ 1. CANNSolutionConfig.from_dict(solution.other_info)
    │
    ├─ 2. Validate 6 required components
    │     └─ Missing → stage="validation"
    │
    ├─ 3. TemplateGenerator.generate() ──► full source code (6 files)
    │
    ├─ 4. [fake_mode] Write files only → stage="files_written"
    │
    ├─ 5. AscendCEvaluator.compile()
    │     ├─ msopgen (project scaffold)
    │     ├─ Write source files
    │     ├─ build.sh (cmake + make)
    │     ├─ Deploy (.run package)
    │     └─ Pybind (build Python extension)
    │     └─ Failure → stage="compile"
    │
    ├─ 6. [compile_only] → stage="compile_only"
    │
    └─ 7. Verify & Measure (sandbox subprocess)
          ├─ Correctness: Model vs ModelNew output comparison
          └─ Performance: multi-round timing with warmup
```

## Serial vs Parallel Mode

| | Serial (`parallel=False`) | Parallel (`parallel=True`) |
|---|---|---|
| **Sandbox calls** | 1 (correctness + performance combined) | 2 (separate calls) |
| **NPU init overhead** | 1x (~10-20s) | 2x |
| **Concurrency** | N/A | Safe with device pool |
| **Use case** | Single evaluation | Batch evaluation across multiple NPUs |

**Serial mode** (default) combines correctness and performance into a single sandbox subprocess, avoiding the ~10-20s NPU initialization overhead of a second call.

**Parallel mode** splits them into separate calls so that multiple evaluations can run concurrently across devices. The device pool guarantees exclusive access — each NPU runs at most one evaluation at a time.

## Device Pool

```
┌──────────────────────────────────────────────────┐
│              queue.Queue (thread-safe)            │
│                                                   │
│   ┌─────┐ ┌─────┐ ┌─────┐         ┌─────┐       │
│   │npu:0│ │npu:1│ │npu:2│  . . .   │npu:7│       │
│   └─────┘ └─────┘ └─────┘         └─────┘       │
└───────────────────────┬──────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Thread 1│    │ Thread 2│    │ Thread 3│
   │ acquire │    │ acquire │    │ acquire │
   │ eval    │    │ eval    │    │ (block) │
   │ release │    │ release │    │         │
   └─────────┘    └─────────┘    └─────────┘
```

- `init_device_pool([0,1,...])` — called once, populates the queue
- `_acquire_device()` — `queue.get()`, blocks until a device is available
- `_release_device()` — `queue.put()`, returns device to the pool
- Auto-initialized with `[0]` on first access if not explicitly configured

## Sandbox Isolation

All correctness and performance evaluations run in `multiprocessing.spawn` subprocesses:

| Goal | Mechanism |
|------|-----------|
| Prevent environment pollution | `exec()` does not affect main process `sys.modules` |
| Contain crashes | Segfaults are isolated to the subprocess |
| Enforce timeouts | Subprocess can be forcefully terminated |
| Isolate resource leaks | Resources are automatically reclaimed on subprocess exit |

### Timeout Configuration

| Stage | Default | Description |
|-------|---------|-------------|
| msopgen | 60s | Project scaffold generation |
| build.sh | 180s | CMake + make |
| deploy | 60s | Install .run package |
| pybind | 120s | Build Python extension |
| sandbox (total) | 600s | Overall sandbox timeout |

## Code Generation: 6-Component Model

LLM generates only the logic; templates handle boilerplate:

```
┌─────────────────────────────────────────────────────────┐
│                  LLM generates 6 components              │
│                                                          │
│  kernel_impl ─────────────┐                              │
│  kernel_entry_body ────────┼──► kernel_src               │
│                            │                              │
│  tiling_fields ────────────┼──► host_tiling_src          │
│                            │                              │
│  tiling_func_body ─────────┼──► host_operator_src        │
│  infer_shape_body ─────────┘                              │
│                                                          │
│  output_alloc_code ────────────► python_bind_src         │
│                                                          │
│  (auto-generated)                                        │
│  signature ────────────────────► project_json_src        │
│  signature ────────────────────► model_src               │
└─────────────────────────────────────────────────────────┘
```

This separation ensures that fixed parts (function names, build config, PyTorch bindings) are controlled by templates, while the LLM focuses solely on implementation logic.

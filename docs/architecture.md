# 架构设计

## 评估流水线

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
    ├─ 2. 验证 6 个必填组件完整性
    │     └─ 缺失 → stage="validation"
    │
    ├─ 3. TemplateGenerator.generate() ──► 完整源代码（6 个文件）
    │
    ├─ 4. [fake_mode] 仅写入文件 → stage="files_written"
    │
    ├─ 5. AscendCEvaluator.compile()
    │     ├─ msopgen（项目脚手架）
    │     ├─ 写入源文件
    │     ├─ build.sh（cmake + make）
    │     ├─ deploy（安装 .run 包）
    │     └─ pybind（构建 Python 扩展）
    │     └─ 失败 → stage="compile"
    │
    ├─ 6. [compile_only] → stage="compile_only"
    │
    └─ 7. 验证 & 测量（沙箱子进程）
          ├─ 正确性：Model vs ModelNew 输出对比
          └─ 性能：多轮计时（含预热）
```

## 串行 vs 并行模式

| | 串行（`parallel=False`） | 并行（`parallel=True`） |
|---|---|---|
| **沙箱调用次数** | 1（正确性 + 性能合并） | 2（分开调用） |
| **NPU 初始化开销** | 1 次（~10-20s） | 2 次 |
| **并发支持** | 不适用 | 通过设备池安全并发 |
| **适用场景** | 单次评估 | 多 NPU 批量评估 |

**串行模式**（默认）将正确性和性能合并在一次沙箱子进程中执行，避免第二次 ~10-20s 的 NPU 初始化开销。

**并行模式**将两者拆分为独立调用，使多个评估可以跨设备并发运行。设备池保证独占访问——每张 NPU 同一时刻最多运行一个评估。

## 设备池

```
┌──────────────────────────────────────────────────┐
│              queue.Queue（线程安全）                │
│                                                   │
│   ┌─────┐ ┌─────┐ ┌─────┐         ┌─────┐       │
│   │npu:0│ │npu:1│ │npu:2│  . . .   │npu:7│       │
│   └─────┘ └─────┘ └─────┘         └─────┘       │
└───────────────────────┬──────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  线程 1  │    │  线程 2  │    │  线程 3  │
   │  获取    │    │  获取    │    │  获取    │
   │  评估    │    │  评估    │    │ (阻塞)  │
   │  释放    │    │  释放    │    │         │
   └─────────┘    └─────────┘    └─────────┘
```

- `init_device_pool([0,1,...])` — 调用一次，填充设备队列
- `_acquire_device()` — `queue.get()`，阻塞直到有设备可用
- `_release_device()` — `queue.put()`，将设备归还到池中
- 未显式初始化时，首次访问自动创建单设备池 `[0]`

## 沙箱隔离

所有正确性和性能评估在 `multiprocessing.spawn` 子进程中运行：

| 目标 | 机制 |
|------|------|
| 防止环境污染 | `exec()` 不影响主进程的 `sys.modules` |
| 隔离崩溃 | segfault 限制在子进程内 |
| 超时控制 | 可强制终止超时的子进程 |
| 资源泄漏隔离 | 子进程退出时自动回收资源 |

### 超时配置

| 阶段 | 默认超时 | 说明 |
|------|----------|------|
| msopgen | 60s | 项目脚手架生成 |
| build.sh | 180s | CMake + make |
| deploy | 60s | 安装 .run 包 |
| pybind | 120s | 构建 Python 扩展 |
| sandbox（总体） | 600s | 沙箱整体超时 |

## 代码生成：6 组件模型

LLM 只生成逻辑，模板处理所有样板代码：

```
┌─────────────────────────────────────────────────────────┐
│                  LLM 生成 6 个组件                        │
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
│  （自动生成）                                              │
│  signature ────────────────────► project_json_src        │
│  signature ────────────────────► model_src               │
└─────────────────────────────────────────────────────────┘
```

这种分离确保固定部分（函数名、构建配置、PyTorch 绑定）由模板控制，LLM 只需专注于实现逻辑。

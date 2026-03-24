# CANN 沙箱隔离机制

评估操作（正确性验证、性能测量）在独立的沙箱子进程中执行，确保主进程的稳定性和隔离性。编译在主进程中运行（全部是 `subprocess.run` 调用，无 segfault 风险）。

## 设计目标

| 目标 | 说明 |
|------|------|
| 防止环境污染 | `exec()` 加载的模块不会污染主进程的 `sys.modules` |
| 防止崩溃扩散 | NPU 内存问题或 segfault 不会影响主进程 |
| 超时控制 | 可以强制终止超时的评估任务 |
| 资源泄漏隔离 | 子进程结束后自动释放所有资源 |

## 架构概览

```
evaluator.compile()              ← 主进程（subprocess.run 调 shell，安全）
    ├── ascend_setup()             Step 1-3: msopgen + 写源文件
    └── ascend_build()             Step 4-7: build.sh + deploy + pybind
        └── skip_model_exec=True   跳过 Step 8 的 exec（沙箱会做）

evaluator.verify_and_measure()   ← 单次沙箱子进程
    └── _verify_and_measure_worker()
        ├── _setup_npu_environment()    设置 ASCEND_CUSTOM_OPP_PATH
        ├── _init_npu_context()         import torch_npu + exec model
        ├── execute_correctness_check() 正确性验证
        └── measure_performance()       性能测量（correctness 通过后才执行）
```

**为什么编译不需要沙箱**：编译阶段全部通过 `subprocess.run()` 调用外部命令（msopgen、build.sh、.run 安装包），即使外部命令崩溃也不影响主进程，`CalledProcessError` / `TimeoutExpired` 都有 catch。

**为什么合并 correctness + performance**：两者的前置步骤完全相同（设环境变量 → import torch_npu → exec model），分两次沙箱调用会浪费 ~10-20s 的 NPU 初始化开销。

## 实现方式

使用 `multiprocessing.spawn` 创建全新进程，通过 `Manager().dict()` 共享结果：

```python
import multiprocessing as mp

ctx = mp.get_context("spawn")  # spawn = 全新进程，不继承父进程状态
manager = ctx.Manager()
return_dict = manager.dict()   # 子进程写入结果
timing_dict = manager.dict()   # 子进程标记完成状态

full_args = worker_args + (return_dict, timing_dict)
process = ctx.Process(target=worker_func, args=full_args)
process.start()

# 轮询监控：每 0.5s 检查完成状态和超时
while process.is_alive():
    if timing_dict.get("completed", False):
        process.join()
        break
    if time.time() - start_time > timeout:
        process.terminate()  # 超时处理（见下文）
        break
    time.sleep(0.5)

result = dict(return_dict.get("result", default_error))
```

### spawn vs fork

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| `fork` | 复制父进程内存，继承已加载模块 | 简单任务，无模块冲突 |
| `spawn` | 全新 Python 解释器，完全隔离 | CANN 评估（选用） |

**选择 spawn 的原因**：
- 每次测试都是全新进程，Python 模块缓存不会跨测试污染
- 即使所有 solution 都绑定到同一个 `custom_ops_lib` 包名，也不会相互影响
- 每个子进程独立设置 `ASCEND_CUSTOM_OPP_PATH` 环境变量

## 环境变量传递机制

### 问题

`spawn` 方式创建的子进程**不会继承**父进程后来设置的环境变量：

```python
# ❌ 错误：主进程设置的环境变量，spawn 子进程看不到
os.environ["ASCEND_CUSTOM_OPP_PATH"] = "/path/to/opp"
ctx = mp.get_context("spawn")
process = ctx.Process(target=worker)  # 子进程继承的是 spawn 时刻的环境
```

### 解决方案

环境变量设置逻辑提取为 `_setup_npu_environment()` 函数，在子进程内部、`import torch_npu` 之前调用：

```python
def _setup_npu_environment(project_path: Optional[str]) -> None:
    """在子进程内设置环境变量。必须在 import torch_npu 之前调用！"""
    if not project_path:
        return

    import os
    from pathlib import Path

    custom_opp_path = Path(project_path) / "opp" / "vendors" / "customize"
    if custom_opp_path.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
        lib_path = custom_opp_path / "op_api" / "lib"
        if lib_path.exists():
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}"

    extension_build = Path(project_path) / "CppExtension" / "build"
    if extension_build.exists():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if str(extension_build) not in existing:
            os.environ["LD_LIBRARY_PATH"] = f"{extension_build}:{existing}"


def _init_npu_context(context_data, python_reference, device_str):
    """公共初始化：import torch_npu + exec 模型代码"""
    import torch
    import torch_npu
    device = torch.device(device_str)
    context = {}
    if "model_src" in context_data:
        exec(context_data["model_src"], context)
    exec(python_reference, context)
    return context, device, torch_npu
```

> **关键**：环境变量必须在 `import torch_npu` **之前**设置！
> `torch_npu` 导入时会初始化 CANN 运行时并读取环境变量，之后再设置无效。

### 调用链

```
evaluator.verify_and_measure()
    ↓ 传递 project_path
sandbox.verify_and_measure_sandbox(project_path=self.project_path)
    ↓ spawn 子进程
_verify_and_measure_worker(..., project_path, ...)
    ├── _setup_npu_environment(project_path)     # 设置环境变量
    ├── _init_npu_context(...)                   # import torch_npu + exec model
    ├── execute_correctness_check(...)           # 正确性验证
    └── measure_performance(...)                 # 性能测量（通过后才执行）
```

## 沙箱开销

每次沙箱调用需要：

| 阶段 | 耗时 |
|------|------|
| 启动新 Python 进程 | ~1-2s |
| 导入依赖 (torch, torch_npu) | ~5-10s |
| 初始化 NPU 设备 | ~2-5s |
| 分配 NPU 内存 | ~1-2s |
| **总计** | **~10-20s** |

合并 worker 将正确性 + 性能合并到一次沙箱调用中，避免了两次 ~10-20s 的开销。

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sandbox_timeout` | 600 秒 | 沙箱超时时间 |
| `num_correctness_trials` | 5 | 正确性验证轮数 |
| `num_perf_trials` | 100 | 性能测量轮数 |
| `num_warmup` | 3 | 性能测量预热次数 |

### 环境变量

| 环境变量 | 设置位置 | 说明 |
|----------|----------|------|
| `ASCEND_CUSTOM_OPP_PATH` | sandbox worker 内 | 指向 `{project_path}/opp/vendors/customize` |
| `LD_LIBRARY_PATH` | sandbox worker 内 | 包含 `op_api/lib` 和 `CppExtension/build` |
| `ASCEND_AICPU_EXCEPTION_DUMP` | evaluator 初始化 | 默认 `"0"`，禁用异常 dump 以避免生成大量 `exception_info` 文件 |

## 错误处理

### 超时处理

```python
if process.is_alive():
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()  # 强制终止
        process.join()  # 等待进程完全退出，防止僵尸进程
    return {"error": "Sandbox timeout"}
```

### 子进程崩溃

如果子进程异常退出（segfault 等），`return_dict` 中不会有 `result`，
`_execute_in_sandbox` 返回调用方传入的 `default_error`：

```python
# _execute_in_sandbox 中的兜底逻辑
return dict(return_dict.get("result", default_error))

# default_error 示例：
# 合并验证:   {"correctness": None, "performance": None, "error": "Unknown error"}
# 单独正确性: {"pass": False, "error": "Unknown error"}
# 单独性能:   {"runtime": None, "error": "Unknown error"}
```

## 临时文件：fusion_result.json

CANN 运行时在执行算子时会在**当前工作目录**生成 `fusion_result.json` 临时文件。

**解决方案**：在沙箱中切换到独立的工作目录，或添加到 `.gitignore`。

---

## WSL/x86 模拟环境限制

在 WSL x86_64 环境下编译 CANN 算子存在限制。

### 问题现象

```
[ERROR] TBE: An error occurred during compile phases of CompileStage.INFERCHANNEL
error: use of undeclared identifier 'tiling_data'
```

### 根本原因

| 特性 | 真实硬件 | WSL x86_64 |
|------|----------|------------|
| TBE 编译预处理 | 完整 | 不完整 |
| GET_TILING_DATA 宏 | 自动注入 | 未注入 |
| 算子编译 | 成功 | 失败 |

### WSL 可用工作流

| 工作流 | 可用性 |
|--------|--------|
| 代码编辑 | ✅ |
| 模板生成 (fake_mode) | ✅ |
| 语法检查 | ⚠️ 部分 |
| 完整编译 | ❌ |
| 算子运行 | ❌ |

### fake_mode 用法

```python
task = CANNInitTask(
    data={"op_name": "add", "python_reference": PYTHON_REF},
    fake_mode=True  # 跳过编译，只写入文件
)
```

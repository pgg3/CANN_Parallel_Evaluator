# Evaluator 工程化改动：与 MultiKernelBench 对比

本文档对比 [MultiKernelBench](https://github.com/wzzll123/MultiKernelBench) 原始 AscendC 编译流水线与 cann-parallel-evaluator 的改动，记录所有工程化改进。

## 改动总览

| 类别 | MultiKernelBench 原始实现 | cann-parallel-evaluator 改动 |
|------|--------------------------|------------------------------|
| **代码架构** | 单文件 `ascend_compile()` + `exec()` 6 个字符串变量 | 模块化模板系统，6 组件分离，签名自动解析 |
| **算子部署** | 全局 OPP 路径，无 `--install-path` | 项目本地 `opp/` 目录，`--install-path` 隔离 |
| **Python 绑定** | `pip install --force-reinstall` 到全局 site-packages | 只 `setup.py build`，从本地 `build/lib.*/` 加载 |
| **沙箱隔离** | 无沙箱，`exec()` 在主进程 | `multiprocessing.spawn` 子进程隔离 |
| **并行编译** | 不支持（`os.chdir` + 全局状态） | 支持（目录隔离 + 延时启动 + 环境变量隔离） |
| **CANN Bug 修复** | 无 | 4 项修复（重复定义、缺失 include、异常 dump、多卡 OPP 绑定） |
| **环境变量** | 主进程直接 `os.environ` 修改 | sandbox worker 内部设置，先于 `import torch_npu` |
| **工作目录** | `os.chdir()` 切换（非线程安全） | 绝对路径 + `cwd` 参数（线程安全） |
| **错误处理** | 基础 try/except | 分阶段超时 + 错误过滤 + 磁盘保护 |

---

## 1. 代码架构重构

### MultiKernelBench

LLM 生成一段 Python 代码，`exec()` 后产出 6 个字符串变量（`project_json_src`, `host_tiling_src`, `host_operator_src`, `kernel_src`, `python_bind_src`, `model_src`），直接写入文件：

```python
# ascend_compile_pipeline.py
exec(generated_code, context)  # 一次 exec 产出所有源码
```

LLM 需要生成完整的 C++ 文件内容，包括所有固定结构（函数签名、宏定义、注册代码等）。

### cann-parallel-evaluator拆分为 6 个 LLM 组件 + 模板系统：

```
LLM 只生成逻辑部分          模板负责固定结构
─────────────────          ──────────────
kernel_impl                → kernel_src.py 拼接 #include + extern "C" 入口
kernel_entry_body          → kernel_src.py 填入入口函数
tiling_fields              → host_tiling.py 生成 BEGIN/END_TILING_DATA_DEF
tiling_func_body           → host_operator.py 填入 TilingFunc
infer_shape_body           → host_operator.py 填入 InferShape
output_alloc_code          → python_bind.py 填入分配代码
```

新增 `SignatureParser` 自动从 Python reference 解析算子签名，`project_json_src` 和 `model_src` 完全自动生成。

**改动意义**：LLM 只负责逻辑实现，固定结构由模板控制，减少生成错误。

---

## 2. 编译隔离

### 2.1 算子 Deploy 隔离

**MultiKernelBench 问题**：

```python
# 原始代码：无 --install-path，安装到全局 OPP
subprocess.run(["./custom_opp_ubuntu_aarch64.run"])
# → 安装到 /usr/local/Ascend/.../opp/vendors/customize/
```

并行编译时后一个算子覆盖前一个的全局部署。

**cann-parallel-evaluator 修复**：

```python
local_opp_path = os.path.join(project_path, "opp")
subprocess.run(["./custom_opp_ubuntu_aarch64.run", f"--install-path={local_opp_path}"])
# → 安装到 project_path/opp/vendors/customize/
```

每个项目有独立的 `opp/` 目录，互不干扰。

### 2.2 Python Binding 隔离

**MultiKernelBench 问题**：

```bash
# build_and_run.sh：打包 wheel 并全局安装
python setup.py bdist_wheel
pip install --force-reinstall ./*.whl
# → 安装到全局 site-packages/custom_ops_lib.*.so
```

```python
# model_src 从全局导入
import custom_ops_lib  # 可能加载到错误版本
```

**cann-parallel-evaluator 修复**：

```bash
# build_and_run.sh：只编译，不安装
python setup.py build
# → .so 留在 build/lib.*/ 目录
```

```python
# model_src 从本地 build 目录加载
_build_dirs = glob.glob(os.path.join(_project_path, "CppExtension", "build", "lib.*"))
sys.path.insert(0, _build_dirs[0])
import custom_ops_lib  # 从项目本地加载
```

### 2.3 环境变量隔离

**MultiKernelBench 问题**：

```python
# 主进程直接修改全局环境变量
os.environ["ASCEND_CUSTOM_OPP_PATH"] = "..."
os.environ["LD_LIBRARY_PATH"] = "..."
```

**cann-parallel-evaluator 修复**：

环境变量在 sandbox worker 子进程内部设置，先于 `import torch_npu`：

```python
def _setup_npu_environment(project_path, device_str="npu:0"):
    """在 spawn 子进程内设置，不影响主进程"""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_id  # 见 5.4
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_path)
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}"
    # 之后才 import torch_npu
```

`spawn` 子进程不继承父进程后来设置的环境变量，所以必须在子进程内部设置。

---

## 3. 沙箱隔离

### MultiKernelBench

无沙箱。`exec()` 在主进程运行 LLM 生成的代码。唯一的隔离是 `evaluation.py` 用 `subprocess.run()` 调用 `eval_single_runner.py`（180s 超时），提供崩溃隔离但非进程内隔离。

```python
# eval_single_runner.py - 整个评估在一个子进程
exec(generated_code, context)  # 直接 exec 不可信代码
# ... 编译 + 正确性 + 性能 全在同一进程
```

### cann-parallel-evaluator分层隔离：

```
编译阶段（主进程）：
  全部通过 subprocess.run() 调用外部命令
  无 exec()，无 segfault 风险

评估阶段（sandbox 子进程）：
  multiprocessing.spawn 创建全新 Python 进程
  exec(model_src) 在子进程内执行
  正确性 + 性能合并在一次沙箱调用（省 ~10-20s NPU 初始化）
```

**关键区别**：

| 特性 | MultiKernelBench | cann-parallel-evaluator |
|------|-----------------|-------------------------|
| exec() 位置 | 主进程 | sandbox 子进程 |
| 进程隔离方式 | subprocess（整个评估） | spawn（仅 exec + 测试） |
| 模块缓存污染 | 有（同一进程多次 exec） | 无（每次全新进程） |
| segfault 影响 | 杀死 eval_single_runner | 仅杀死 sandbox worker |
| 超时控制 | 180s 整体 | 分阶段（编译 60-180s，沙箱 600s） |

---

## 4. 并行编译

### MultiKernelBench

不支持并行编译：

- `os.chdir()` 切换工作目录（非线程安全）
- 共享 `CppExtension/csrc/op.cpp`（每次覆盖）
- 全局 OPP 部署路径
- 全局 `pip install`
- `evaluation.py` 串行 for 循环逐个评估

### cann-parallel-evaluator支持并行编译，解决了 4 个挑战：

| 挑战 | 解决方案 |
|------|----------|
| 工作目录竞争 | `subprocess.run(cwd=...)` 替代 `os.chdir()`，线程安全 |
| 算子 Deploy 冲突 | `--install-path` 项目目录隔离 |
| 环境变量污染 | sandbox worker 内部设置，`ascend_build` 不修改 `os.environ` |
| Python 包名冲突 | spawn 子进程隔离 + 本地 .so 加载 |
| 多卡 OPP 绑定 | `ASCEND_RT_VISIBLE_DEVICES` 设备隔离，每卡独立 sandbox（见 5.4） |

流水线式工作流（generate → compile → evaluate 同一个 ThreadPoolExecutor）：

```
Thread 0: LLM generate → ascend_setup(cwd=project_0/) → ascend_build(cwd=project_0/) → sandbox verify+measure(device=npu:0)
Thread 1: LLM generate → ascend_setup(cwd=project_1/) → ascend_build(cwd=project_1/) → sandbox verify+measure(device=npu:1)
Thread 2: LLM generate → ascend_setup(cwd=project_2/) → ascend_build(cwd=project_2/) → sandbox verify+measure(device=npu:2)
```

支持 `CompileResult.save()` / `CompileResult.load()` 实现编译-测试分离。

---

## 5. CANN 工具链 Bug 修复

MultiKernelBench 没有处理这些问题。在大规模并行实验中暴露。

### 5.1 重复定义 Bug

**问题**：CANN `build.sh` 生成的 `.ini` 文件包含重复 section（`DuplicateSectionError`），`.h` 文件包含重复 `REG_OP` 定义（C++ 编译错误）。

**修复**（`ascend_compile.py`）：

1. Patch `ascendc_get_op_name.py`：注入 `DuplicateTolerantConfigParser`，跳过重复 section
2. 注入 `fix_duplicates.py`：扫描 `build_out/autogen/` 清理重复定义
3. Patch `build.sh`：在 `cmake --build` 前自动运行修复脚本

运行时 Patch 策略，无需修改 CANN 工具链源码。

### 5.2 缺失 Include 路径

**问题**：Cube 算子使用的高级 API 头文件（`lib/matmul_intf.h`, `tiling/platform/platform_ascendc.h`）不在 CMake 默认搜索路径中。

**修复**（`ascend_compile.py`）：

Patch `intf.cmake` 和 `func.cmake`，添加：
```
${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api
```

### 5.3 异常 Dump 清理 ⚠️ 待改进

**问题**：CANN 编译器在编译失败时生成 4GB+ 的 `exception_info` dump 文件（`extra-info/data-dump/`），迭代实验中快速填满磁盘。

**当前处理**：
- `evaluator.py` 设置 `ASCEND_AICPU_EXCEPTION_DUMP=0` 尝试禁用
- `ascend_compile.py` 在每次 build 后（成功/失败/超时）调用 `_cleanup_exception_dumps()` 删除 `extra-info/` 目录

**不足**：
- 事后清理策略，并行编译时 dump 可能在清理前已占满磁盘
- `ASCEND_AICPU_EXCEPTION_DUMP=0` 不一定能阻止所有类型的 dump 生成
- ~~只清理 `target_directory/extra-info/`~~ 已修复：现在同时清理 `target_directory` 和 Python 启动目录（`original_cwd`）的 `extra-info/`
- CANN 可能在其他未知位置也生成 dump

### 5.4 多卡自定义 OPP 加载失败

**问题**：CANN 运行时的自定义算子（Custom OPP）加载机制只将 kernel binary 绑定到 device 0。当进程能看到多张 NPU 卡时，使用 `npu:X`（X≠0）调用自定义算子会导致 kernel 不执行，输出全零。

**现象**：

```
多卡并行评估：
  npu:0 → max_diff = 0.0    ✓ 正确
  npu:1 → max_diff ≈ 1.0    ✗ 输出全零
  npu:2 → max_diff ≈ 1.0    ✗ 输出全零
  npu:3 → max_diff ≈ 1.0    ✗ 输出全零
```

对于 ReLU + `torch.rand()` 输入（范围 [0,1)），`max_diff ≈ 1.0` 意味着自定义 kernel 输出全为零（`nonzero_frac=0.0`），kernel 根本没有执行。

**根因分析**：CANN Custom OPP 的 kernel binary 在运行时只加载到 device 0 对应的 AI Core 上。当 `torch_npu` 初始化时看到多张卡，使用 `npu:2` 执行自定义算子时，CANN 在 device 2 上找不到 custom kernel binary，静默返回全零。

**修复**（`sandbox.py`）：

在 sandbox 子进程中，`import torch_npu` **之前**设置 `ASCEND_RT_VISIBLE_DEVICES`，将物理设备映射为子进程内唯一的 `npu:0`：

```python
def _setup_npu_environment(project_path, device_str="npu:0"):
    """在 spawn 子进程内设置，必须先于 import torch_npu"""
    # 从 device_str 提取物理设备 ID（如 "npu:2" → "2"）
    if ":" in device_str:
        physical_id = device_str.split(":")[1]
    else:
        physical_id = "0"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_id
    # ... 之后才 import torch_npu
```

```python
def _init_npu_context(context_data, python_reference, device_str):
    """device_str 参数被忽略 — 始终使用 npu:0"""
    import torch
    import torch_npu
    # ASCEND_RT_VISIBLE_DEVICES 已将物理设备隔离为 npu:0
    device = torch.device("npu:0")
```

**原理**：`ASCEND_RT_VISIBLE_DEVICES=2` 让 CANN 运行时只看到物理设备 2，并将其映射为 `npu:0`。此时 Custom OPP 加载到 device 0（即物理设备 2），kernel 正常执行。

**验证结果**：

```
修复前（4 卡并行）：Gen 0 valid rate = 25%（仅 npu:0 通过）
修复后（4 卡并行）：Gen 0 valid rate = 100%（全部通过）
```

**关键约束**：`ASCEND_RT_VISIBLE_DEVICES` 必须在 `import torch_npu` 之前设置，因为 `torch_npu` 在 import 时就会初始化 CANN 运行时。这也是为什么必须使用 `multiprocessing.spawn`（全新进程）而非 `fork`（继承已初始化的 torch_npu）。

---

## 6. 其他改进

### 6.1 工作目录安全

**MultiKernelBench**：使用 `os.chdir()` 切换到项目目录执行命令。

**cann-parallel-evaluator**：使用绝对路径 + `subprocess.run(cwd=...)` 参数，不修改进程工作目录。

### 6.2 分阶段 API

**MultiKernelBench**：单一 `ascend_compile()` 函数，不可拆分。

**cann-parallel-evaluator**：拆分为 `ascend_setup()` + `ascend_build()`：

| 函数 | 阶段 | 并行性 |
|------|------|--------|
| `ascend_setup()` | msopgen + 写源文件 | 可并行（`cwd=` 隔离，每个 project 独立目录） |
| `ascend_build()` | build + deploy + pybind | 可并行（`cwd=` 隔离 + 不修改 `os.environ`） |

### 6.3 错误处理

**MultiKernelBench**：基础 try/except，超时 180s 整体。

**cann-parallel-evaluator**：

| 步骤 | 超时 | 错误处理 |
|------|------|----------|
| msopgen | 60s | stdout/stderr 捕获 |
| build.sh | 180s | 过滤 `[ERROR]` 行，提取关键错误 |
| deploy | 60s | stdout 捕获 |
| pybind | 120s | stdout 捕获 |
| sandbox | 600s | terminate → kill → 僵尸进程清理 |

### 6.4 Fake Mode

MultiKernelBench 无此功能。evotoolkit 支持 `fake_mode=True` 仅写入文件不编译，用于 WSL/x86 开发环境。

---

## 关键文件对照

| 功能 | MultiKernelBench | cann-parallel-evaluator |
|------|-----------------|-------------------------|
| 编译流水线 | `utils/ascend_compile_pipeline.py` | `utils/backend/ascend_compile.py` |
| 正确性验证 | `utils/correctness.py` | `utils/backend/correctness.py` |
| 性能测量 | `utils/performance.py` | `utils/backend/performance.py` |
| 沙箱 | 无 | `utils/backend/sandbox.py` |
| 模板生成 | 无（LLM 生成完整文件） | `utils/templates/*.py` |
| 签名解析 | 无（硬编码 6 个算子） | `signature_parser.py` |
| Pybind 模板 | `ascend_op_projects/CppExtension/` | `utils/templates/pybind_templates/` |
| 评估入口 | `eval_single_runner.py` | `cann_init_task.py` |
| 后端注册 | `backends/ascendc_backend.py` | `evaluator.py` |

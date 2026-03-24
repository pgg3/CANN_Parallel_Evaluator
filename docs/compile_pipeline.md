# CANN 编译流水线

Ascend C 算子完整编译部署流水线，包括项目创建、代码编译、算子部署和 Python 绑定构建。

## 编译流程 (8 步)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ascend_compile()                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: 创建项目目录                                                │
│      └─► 删除旧目录，写入 {op}_custom.json                           │
│                                                                      │
│  Step 2: msopgen 生成项目骨架                                        │
│      └─► msopgen gen -i {op}.json -c ai_core-Ascend910B2 ...        │
│                                                                      │
│  Step 3: 写入源文件                                                  │
│      ├─► op_host/{op}_custom_tiling.h                               │
│      ├─► op_host/{op}_custom.cpp                                    │
│      └─► op_kernel/{op}_custom.cpp                                  │
│                                                                      │
│  Step 4: 构建算子                                                    │
│      └─► ./build.sh                                                 │
│                                                                      │
│  Step 5: 部署算子包                                                  │
│      └─► ./custom_opp_ubuntu_aarch64.run --install-path=...         │
│                                                                      │
│  Step 6: 构建 Python 绑定                                            │
│      └─► bash build_and_run.sh                                      │
│                                                                      │
│  Step 7: 设置环境变量                                                │
│      ├─► ASCEND_CUSTOM_OPP_PATH (指向 project_path/opp/vendors/customize) │
│      └─► LD_LIBRARY_PATH (包含 opp/vendors/customize/op_api/lib)     │
│                                                                      │
│  Step 8: 加载模型代码                                                │
│      └─► exec(model_src, context)                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

> **注意**：在沙箱模式下，环境变量由各 worker 在子进程内部设置（先于 `import torch_npu`），
> 因为 `spawn` 方式创建的子进程不会继承父进程后来设置的环境变量。
> 沙箱 worker 还会额外将 `CppExtension/build` 加入 `LD_LIBRARY_PATH`。
> 详见 [sandbox_design.md](sandbox_design.md#环境变量传递机制)。

## 超时设置

| 步骤 | 超时时间 | 说明 |
|------|----------|------|
| msopgen | 60 秒 | 生成项目骨架 |
| build.sh | 180 秒 | 编译算子 |
| deploy | 60 秒 | 部署 .run 包 |
| pybind | 120 秒 | 构建 Python 绑定 |

## 目录结构

编译完成后的完整目录结构：

```
project_path/
├── add_custom.json              # 算子配置 (Step 1)
├── model_src.py                 # ascend_setup() 写入的模型代码
├── AddCustom/                   # msopgen 生成的项目 (Step 2-4)
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── cmake/                   # CMake 构建脚本
│   ├── framework/               # TF 插件 (自动生成)
│   ├── scripts/                 # 安装/升级脚本
│   ├── op_host/
│   │   ├── add_custom_tiling.h  # Tiling 数据结构 (Step 3)
│   │   └── add_custom.cpp       # Host 端实现 (Step 3)
│   ├── op_kernel/
│   │   └── add_custom.cpp       # Device Kernel (Step 3)
│   └── build_out/               # 构建产物 (Step 4)
│       ├── custom_opp_ubuntu_aarch64.run  # 算子安装包
│       ├── op_host/
│       │   ├── libcust_opapi.so           # 算子 API 库
│       │   ├── libcust_opmaster_rt2.0.so  # Tiling 运行时
│       │   └── libcust_opsproto_rt2.0.so  # 算子原型
│       └── op_kernel/binary/ascend910b/   # 编译后的 kernel 二进制
├── CppExtension/                # Python 绑定 (Step 6)
│   ├── setup.py
│   ├── build_and_run.sh
│   ├── csrc/
│   │   ├── op.cpp
│   │   └── pytorch_npu_helper.hpp
│   └── build/                   # setuptools 构建产物
│       └── lib.linux-aarch64-cpython-310/
│           └── custom_ops_lib.cpython-310-aarch64-linux-gnu.so
└── opp/                         # 部署后的算子包 (Step 5)
    └── vendors/
        └── customize/           # ← ASCEND_CUSTOM_OPP_PATH 指向此处
            ├── op_api/
            │   ├── include/
            │   │   └── aclnn_add_custom.h
            │   └── lib/
            │       └── libcust_opapi.so   # ← LD_LIBRARY_PATH 包含此目录
            ├── op_impl/ai_core/tbe/
            │   ├── config/ascend910b/     # 算子信息配置
            │   ├── kernel/ascend910b/     # kernel 二进制
            │   └── op_tiling/lib/         # tiling 运行时库
            ├── op_proto/
            │   ├── inc/op_proto.h
            │   └── lib/linux/aarch64/libcust_opsproto_rt2.0.so
            └── version.info
```

## msopgen 命令

```bash
msopgen gen \
    -i add_custom.json \
    -c ai_core-Ascend910B2 \
    -lan cpp \
    -out AddCustom
```

**compute_unit 格式**: `ai_core-{Device}{Version}`
- `Ascend910B` → `ai_core-Ascend910B2`
- 大小写敏感，需要版本后缀

## 错误处理

| 步骤 | 错误类型 | 返回信息 |
|------|----------|----------|
| msopgen | CalledProcessError | stdout/stderr |
| msopgen | TimeoutExpired | "msopgen timed out" |
| build.sh | CalledProcessError | 过滤 [ERROR] 行 |
| build.sh | TimeoutExpired | "Build timed out" |
| deploy | CalledProcessError | stdout |
| pybind | CalledProcessError | stdout |
| model load | Exception | 异常信息 |

## CANN 构建补丁

CANN 工具链有时会在 `.ini` 文件中生成重复 section，在 `.h` 文件中生成重复 `REG_OP` 定义。`ascend_build()` 会自动：

1. **修补 `ascendc_get_op_name.py`**：替换 `configparser.ConfigParser` 为 `DuplicateTolerantConfigParser`，容忍重复 section
2. **注入 `fix_duplicates.py`**：在每次 `cmake --build` 前运行，清理 `build_out/autogen/` 下的重复定义
3. **清理 `extra-info/` 目录**：CANN 编译器在编译失败时会生成 4GB+ 的 `exception_info` dump 文件，自动清理防止磁盘爆满

## 分阶段执行 API

支持分阶段执行以实现并行编译：

| 函数 | 阶段 | 并行性 |
|------|------|--------|
| `ascend_setup()` | Step 1-3 + pybind 目录准备 + 写入 `model_src.py` | **必须串行**（msopgen 全局锁） |
| `ascend_build()` | Step 4-8（`skip_model_exec=True` 时跳过 Step 8） | 可并行（独立目录） |
| `ascend_compile()` | Step 1-8 | 完整流程 |

```python
# 串行 setup
for sol in solutions:
    ascend_setup(full_code, op_name, project_path)

# 并行 build
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(ascend_build, ...) for sol in solutions]
```

---

## 并行编译

批量评估多个 kernel 变体时的并行编译注意事项和解决方案。

### 并行编译的挑战

| 挑战 | 阶段 | 解决方案 |
|------|------|----------|
| CMake 编译器检测竞争 | 编译 | 延时启动 |
| 算子 Deploy 冲突 | 部署 | 项目目录隔离 |
| 环境变量污染 | 测试 | 沙箱内设置环境变量 |
| Python 包名冲突 | 测试 | spawn 子进程隔离 |

### CMake 编译器检测竞争条件

**问题**：多个 `build.sh` 同时启动时，CMake 配置阶段会发生竞争条件：

```
CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineCCompiler.cmake:115
CMake Error at /usr/share/cmake-3.22/Modules/CMakeTestCCompiler.cmake:69
```

**原因**：CMake 配置阶段在 `CMakeFiles/CMakeTmp/` 创建临时测试文件，多个实例同时运行时产生冲突。

**解决方案：延时启动 (Staggered Start)**

```python
import time
from concurrent.futures import ThreadPoolExecutor

def build_with_delay(task, solution, delay_seconds):
    time.sleep(delay_seconds)
    return task.evaluate_solution(solution)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i, sol in enumerate(solutions):
        delay = i * 2.0  # 0s, 2s, 4s, 6s...
        future = executor.submit(build_with_delay, task, sol, delay)
        futures.append(future)
```

**推荐延时间隔**：

| 并行数量 | 推荐间隔 |
|----------|----------|
| 2-4 | 2.0 秒 |
| 5-8 | 3.0 秒 |
| >8 | 5.0 秒 |

### 项目目录隔离

为每个编译任务分配独立的项目目录：

```python
config = CANNSolutionConfig(project_path=f"/tmp/cann_build_{index}")
```

部署后的目录结构：

```
/tmp/cann_build_0/
├── add_custom.json             # 算子配置
├── model_src.py                # 模型代码
├── AddCustom/                  # msopgen 项目 + 构建产物
├── CppExtension/build/         # Python 绑定库 (加入 LD_LIBRARY_PATH)
└── opp/vendors/customize/      # 部署后的算子包 (ASCEND_CUSTOM_OPP_PATH)
    └── op_api/lib/             # LD_LIBRARY_PATH 包含此目录
/tmp/cann_build_1/
├── ...
```

### 环境变量隔离

**问题**：并行编译后，每个解决方案的算子部署到不同路径。测试时需要设置 `ASCEND_CUSTOM_OPP_PATH` 指向正确的路径。

**错误做法**（不工作）：

```python
# ❌ 主进程设置环境变量
os.environ["ASCEND_CUSTOM_OPP_PATH"] = project_path + "/opp/vendors/customize"

# spawn 子进程不会继承后来设置的环境变量！
ctx = mp.get_context("spawn")
process = ctx.Process(target=worker)  # 子进程看不到新设置的变量
```

**正确做法**：环境变量必须在**沙箱子进程内部**、**`import torch_npu` 之前**设置。详见 [sandbox_design.md](sandbox_design.md#环境变量传递机制)。

### Python 包名冲突

**问题**：所有解决方案的 Python 绑定都叫 `custom_ops_lib`，如果在同一进程中多次 import 会冲突。

**解决方案**：沙箱机制已解决——每次测试都在全新的 `spawn` 子进程中执行，`sys.modules` 不会跨进程共享。

### 两阶段并行编译工作流

推荐的高效工作流：

```
Phase 1: 并行 compile (staggered start，避免 CMake 竞争)
         ↓
Phase 2: 串行 test (需要 NPU，沙箱隔离)
```

编译阶段（`ascend_setup` + `ascend_build`）在各自隔离的 `project_path` 中执行，通过延时启动避免 CMake 竞争。测试阶段从保存的编译结果加载，在沙箱中串行执行。

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

def build_with_delay(task, sol, delay_seconds):
    time.sleep(delay_seconds)
    return task.evaluate_solution(sol)

def parallel_compile_workflow(task, kernel_variants, output_dir, delay=2.0):
    """两阶段并行编译工作流"""

    # Phase 1: Parallel compile with staggered start
    print("Phase 1: Parallel compile...")
    compile_results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, variant in enumerate(kernel_variants):
            config = CANNSolutionConfig(
                project_path=f"{output_dir}/sol_{i:03d}",
                kernel_impl=variant["kernel_impl"],
                kernel_entry_body=variant["kernel_entry_body"],
                tiling_fields=variant["tiling_fields"],
                tiling_func_body=variant["tiling_func_body"],
                infer_shape_body=variant["infer_shape_body"],
                output_alloc_code=variant["output_alloc_code"],
                compile_only=True,
                save_compile_to=f"{output_dir}/sol_{i:03d}",
            )
            sol = Solution("", config.to_dict())
            future = executor.submit(build_with_delay, task, sol, i * delay)
            futures[future] = i

        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            status = "compiled" if result.valid else "failed"
            print(f"  sol_{i}: {status}")
            compile_results.append((i, result))

    # Phase 2: Sequential testing
    print("Phase 2: Sequential testing...")
    results = []
    for i, compile_result in sorted(compile_results):
        if not compile_result.valid:
            continue
        config = CANNSolutionConfig(load_from=f"{output_dir}/sol_{i:03d}")
        result = task.evaluate_solution(Solution("", config.to_dict()))
        if result.valid:
            runtime = result.additional_info.get("runtime")
            print(f"  sol_{i}: {runtime:.4f} ms")
        results.append(result)

    return results
```

### 参考资料

- [CMake Discourse: parallel cmake --build race](https://discourse.cmake.org/t/how-to-avoid-race-for-two-cmake-build-in-parallel/9431)
- [vcpkg Issue #7952](https://github.com/microsoft/vcpkg/issues/7952)

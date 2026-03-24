# API 参考

## CANNInitTask

主入口类，管理 Ascend C 算子的生成与评估。

```python
class CANNInitTask(BaseTask):
    def __init__(
        self,
        data: Dict[str, Any],        # op_name, python_reference, npu_type, cann_version
        project_path: Optional[str] = None,
        fake_mode: bool = False,     # 跳过编译，仅写入文件
        verbose: bool = True,        # 是否打印编译进度
    )

    def evaluate_solution(self, solution: Solution) -> EvaluationResult
    def get_task_description(self, phase: str = "init") -> str  # "init" 或 "evolve"
    def get_task_type(self) -> str  # 返回 "CANNInit"
    def format_solution_components(self, solution: Solution) -> str  # 格式化 6 组件为可读文本
```

### data 字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `op_name` | `str` | 是 | — | 算子名称 |
| `python_reference` | `str` | 是 | — | Python 参考实现代码 |
| `npu_type` | `str` | 否 | `"Ascend910B2"` | 目标 NPU 设备类型 |
| `cann_version` | `str` | 否 | `"8.1.0rc1"` | CANN 版本号 |

### Solution 输入

Solution 通过 `other_info` 字典传递配置，必须包含 6 个必填组件：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `kernel_impl` | `str` | 是 | Kernel 类和辅助代码（可在顶部包含 `#include`） |
| `kernel_entry_body` | `str` | 是 | 入口函数体（GET_TILING_DATA 之后） |
| `tiling_fields` | `List[Dict]` 或 `Dict` | 是 | TilingData 字段定义（dict 格式可嵌入 includes） |
| `tiling_func_body` | `str` | 是 | TilingFunc 函数体（可在顶部包含 `#include`） |
| `infer_shape_body` | `str` | 是 | InferShape 函数体 |
| `output_alloc_code` | `str` | 是 | 输出 tensor 分配代码 |

### CANNSolutionConfig

```python
@dataclass
class CANNSolutionConfig:
    # === Project Config ===
    project_path: Optional[str] = None

    # === Kernel (Device) - generates kernel_src ===
    kernel_impl: Optional[str] = None           # Kernel class (may contain #include)
    kernel_entry_body: Optional[str] = None     # Entry function body

    # === Host Tiling - generates host_tiling_src ===
    tiling_fields: Optional[Any] = None         # List[Dict] or Dict with "fields"+"includes"

    # === Host Operator - generates host_operator_src ===
    tiling_func_body: Optional[str] = None      # TilingFunc body (may contain #include)
    infer_shape_body: Optional[str] = None      # InferShape function body

    # === Python Bind - generates python_bind_src ===
    output_alloc_code: Optional[str] = None     # Output tensor allocation

    # === Execution control ===
    compile_only: bool = False
    load_from: Optional[str] = None
    save_compile_to: Optional[str] = None
    skip_correctness: bool = False
    skip_performance: bool = False

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "CANNSolutionConfig"
    def to_dict(self) -> Dict[str, Any]
```

### 返回结果

```python
# 成功
EvaluationResult(valid=True, score=-runtime,
    additional_info={
        "stage": "success",
        "runtime": 0.1234,
        "runtime_std": 0.0012,
        "baseline_runtime": 0.1500,
        "baseline_std": 0.0015,
        "speedup": 1.21,
    })

# 失败
EvaluationResult(valid=False, score=None,
    additional_info={"stage": "compile", "error": "..."})
```

### Stage 返回值

| stage | 说明 |
|-------|------|
| `success` | 完整流程成功 |
| `compile` | 编译失败 |
| `compile_only` | 仅编译成功 |
| `correctness` | 正确性验证失败 |
| `correctness_only` | 跳过性能测试成功 |
| `performance` | 性能测量失败 |
| `validation` | 输入参数缺失 |
| `write_files` | fake_mode 写文件失败 |
| `files_written` | fake_mode 写文件成功 |
| `load` | 加载已保存结果失败 |
| `load_exception` | 加载异常 |
| `sandbox` | 沙箱子进程异常 |
| `exception` | 其他异常 |

---

## OperatorSignatureParser

从 Python Reference 提取算子签名。支持 fn 和 org 两种格式。

```python
class OperatorSignatureParser:
    def parse(
        self,
        python_code: str,
        op_name: str,
        mode: str = "auto",  # "auto" | "fn" | "org"
    ) -> Dict[str, Any]
```

**mode 参数**：
- `"auto"`（默认）：通过检测 `def module_fn(` 自动判断格式
- `"fn"`：强制使用 fn_py_reference 解析（从 `module_fn()` 签名提取）
- `"org"`：强制使用 py_reference 解析（从 `Model` 类 + `get_inputs`/`get_init_inputs` 提取）

输出格式（两种模式统一）：

```python
signature = {
    "op_name": "elu",
    "inputs": [{"name": "x", "dtype": "float", "is_tensor": True}],
    "outputs": [{"name": "output", "dtype": "float", "is_tensor": True}],
    "init_params": [{"name": "alpha", "dtype": "float", "is_tensor": False, "default": 1.0}],
}
```

### 类型推断策略

| 优先级 | 方法 | 准确度 |
|--------|------|--------|
| 1 | 执行代码并检查运行时类型 | 100% |
| 2 | AST 类型注解 | 高 |
| 3 | 全局/局部变量赋值推断 | 中 |
| 4 | 参数名启发式 (`x` → tensor, `alpha` → scalar) | 低 |

**fn 模式特殊机制**：用 spy 函数替换 `module_fn`，拦截 `Model.forward()` 调用来捕获每个参数的运行时类型。

---

## AscendCTemplateGenerator

根据算子签名和 LLM 组件生成 6 个源文件。

```python
class AscendCTemplateGenerator:
    def __init__(self, signature: Dict[str, Any])

    def generate(
        self,
        kernel_impl: str,
        kernel_entry_body: str,
        tiling_fields: List[Dict],       # 或 Dict（含 "fields" + "includes"）
        tiling_func_body: str,
        infer_shape_body: str,
        project_path: str,
        output_alloc_code: str,
        soc_versions: Optional[List[str]] = None,
    ) -> Dict[str, str]
```

### 6 个组件

| # | 组件 | 生成方式 | 说明 |
|---|------|----------|------|
| 1 | project_json_src | 自动 (from signature) | msopgen 配置 JSON |
| 2 | host_tiling_src | 模板 (from tiling_fields) | Tiling 数据结构头文件 |
| 3 | host_operator_src | 模板 (from func bodies) | Host 端实现 |
| 4 | kernel_src | 模板 (from kernel_impl + entry_body) | Device Kernel |
| 5 | python_bind_src | 模板 (from signature + alloc_code) | Python 绑定 |
| 6 | model_src | 自动 (from signature) | 测试模型 ModelNew |

---

## 模板详解

### 1. project_json_src

msopgen 项目配置 JSON：

```json
[{
    "op": "AddCustom",
    "language": "cpp",
    "input_desc": [{"name": "x", "param_type": "required", "format": ["ND"], "type": ["float"]}],
    "output_desc": [{"name": "z", "param_type": "required", "format": ["ND"], "type": ["float"]}],
    "attr": [{"name": "negative_slope", "type": "float", "param_type": "optional", "default_value": "0.1"}]
}]
```

生成规则：`is_tensor=True` → `input_desc`，`is_tensor=False` → `attr`

### 2. host_tiling_src

tiling_fields 支持两种格式：

简单算子（list 格式）：

```python
tiling_fields = [
    {"name": "totalLength", "type": "uint32_t"},                         # 标量
    {"name": "inputShapes", "type": "int64_t", "size": 4},               # 数组
]
```

Cube 算子需要额外头文件时（dict 格式）：

```python
tiling_fields = {
    "includes": ["lib/tiling_api.h"],
    "fields": [
        {"name": "M", "type": "uint32_t"},
        {"name": "cubeTiling", "type": "TCubeTiling", "is_struct": True},
    ],
}
```

生成代码：

```cpp
#include "register/tilingdata_base.h"
#include "lib/tiling_api.h"  // 来自 tiling_fields["includes"]

BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 4, inputShapes);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;
```

支持类型：`int8_t`, `int16_t`, `int32_t`, `int64_t`, `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`, `float`

**结构体字段**：使用 `is_struct: True`，所需头文件通过 dict 格式的 `includes` 指定

### 3. host_operator_src

`tiling_func_body` 可在顶部包含 `#include` 行，模板会自动提取并提升到文件头部：

```cpp
// tiling_func_body 内容（#include 会被自动提升）
#include "lib/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

    AddCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    // ...
```

生成的 host_operator.cpp：

```cpp
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "lib/tiling_api.h"                    // 从 tiling_func_body 提取
#include "tiling/platform/platform_ascendc.h"  // 从 tiling_func_body 提取

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // tiling_func_body（去掉 #include 后的部分）
    AddCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetStorageShape();
    ...
}
```

TilingFunc 函数体示例：

```cpp
AddCustomTilingData tiling;
auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {
    totalLength *= shape.GetDim(i);
}
constexpr uint32_t BLOCK_DIM = 8;
tiling.set_totalLength(totalLength);
tiling.set_tileNum(BLOCK_DIM);
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(BLOCK_DIM);
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = 0;
return ge::GRAPH_SUCCESS;
```

InferShape 函数体示例：

```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return GRAPH_SUCCESS;
```

### 4. kernel_src（模板生成）

kernel_src 由模板自动生成，LLM 提供两个组件：

**kernel_impl**：Kernel 类和辅助代码（可在顶部包含 `#include`）

```cpp
#include "lib/matmul_intf.h"  // 额外头文件直接写在 kernel_impl 中

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum) { ... }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i); Compute(i); CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress) { ... }
    __aicore__ inline void Compute(int32_t progress) { ... }
    __aicore__ inline void CopyOut(int32_t progress) { ... }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
};
```

**kernel_entry_body**：入口函数体（GET_TILING_DATA 之后的代码）

```cpp
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
```

模板自动生成完整的 kernel_src：

```cpp
#include "kernel_operator.h"

#include "lib/matmul_intf.h"  // 来自 kernel_impl 顶部

{kernel_impl 的其余部分}

extern "C" __global__ __aicore__ void add_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {  // 签名自动生成
    GET_TILING_DATA(tilingData, tiling);
{kernel_entry_body}
}
```

### 5. python_bind_src

生成代码：

```cpp
at::Tensor add_custom_impl_npu(const at::Tensor& x, const at::Tensor& y) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnAddCustom, x, y, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &add_custom_impl_npu, "add operator");
}
```

自定义输出形状（如 matmul）：

```python
output_alloc_code = """int64_t M = A.size(-2);
    int64_t N = B.size(-1);
    at::Tensor result = at::empty({M, N}, A.options());"""
```

### 6. model_src

测试模型，用于正确性验证：

```python
class ModelNew(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return custom_ops_lib.leaky_relu_custom(x, self.negative_slope)
```

---

## AscendCEvaluator

封装编译、正确性验证、性能测量。所有评估在沙箱子进程执行。

```python
class AscendCEvaluator:
    def __init__(
        self,
        project_path: Optional[str] = None,
        device: str = "Ascend910B",
        num_correctness_trials: int = 5,
        num_perf_trials: int = 100,
        num_warmup: int = 3,
        seed: int = 1024,
        sandbox_timeout: int = 600,
        verbose: bool = True,
    )

    def compile(self, full_code: Dict, op_name: str,
                project_path: str = None, kernel_src: str = None) -> CompileResult
    def verify_and_measure(self, python_reference: str,
                           skip_correctness: bool = False,
                           skip_performance: bool = False) -> Dict
    def rebuild_context(self, compile_result: CompileResult) -> bool
    def verify_correctness(self, python_reference: str, op_name: str) -> Dict
    def measure_performance(self, op_name: str, python_reference: str = None) -> Dict
    def deploy(self, op_name: str) -> Dict  # 空操作，compile() 已包含部署
    def cleanup(self) -> None  # 清理 context
```

> **推荐**：使用 `verify_and_measure()` 而非分别调用 `verify_correctness()` + `measure_performance()`，
> 可避免两次 NPU 初始化开销（~10-20s）。

---

## CompileResult

编译结果，支持保存/加载。

```python
@dataclass
class CompileResult:
    success: bool
    error: Optional[str]
    project_path: str
    op_name: str
    context: Dict
    kernel_src: Optional[str]
    full_code: Optional[Dict]

    def save(self, path: str)
    @classmethod
    def load(cls, path: str) -> "CompileResult"
    def is_loadable(self) -> bool
```

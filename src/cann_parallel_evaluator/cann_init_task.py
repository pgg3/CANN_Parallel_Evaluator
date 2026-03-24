# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import tempfile
import queue
from typing import Any, Dict, List, Optional

from .core_types import BaseTask, EvaluationResult, Solution, TaskSpec

from .evaluator import AscendCEvaluator
from .knowledge import CANNKnowledgeProvider
from .utils.templates import AscendCTemplateGenerator
from .signature_parser import OperatorSignatureParser
from .data_structures import CompileResult, CANNSolutionConfig

# Pre-classified operator → compute pattern mapping.
# Lookup table replaces keyword-based inference for known operators.
# Patterns: vector (elementwise, reduction, softmax, broadcast, pooling),
#           cube (matmul, convolution, attention),
#           mixed (normalization, index, resize)
_OPERATOR_PATTERN_MAP: dict[str, str] = {
    # activation → elementwise (13) + softmax (2)
    "elu": "elementwise", "gelu": "elementwise", "hardsigmoid": "elementwise",
    "hardtanh": "elementwise", "leaky_relu": "elementwise",
    "min_gpt_new_gelu": "elementwise", "relu": "elementwise",
    "selu": "elementwise", "sigmoid": "elementwise", "softplus": "elementwise",
    "softsign": "elementwise", "swish": "elementwise", "tanh": "elementwise",
    "softmax": "softmax", "log_softmax": "softmax",
    # broadcast (10)
    "add_bias_broadcast": "broadcast", "add_bias_four_dim_broadcast": "broadcast",
    "clamp_broadcast": "broadcast", "division_broadcast": "broadcast",
    "elmentwise_mul_broadcast": "broadcast", "logic_and_broadcast": "broadcast",
    "max_broadcast": "broadcast", "power_broadcast": "broadcast",
    "subtract_with_bias_broadcast": "broadcast", "where_broadcast": "broadcast",
    # loss → elementwise (7)
    "cosine_similarity_loss": "elementwise", "cross_entropy_loss": "elementwise",
    "hinge_loss": "elementwise", "huber_loss": "elementwise",
    "kl_div_loss": "elementwise", "mse_loss": "elementwise",
    "triplet_margin_loss": "elementwise",
    # math — scan ops need sequential dependency (reduction pattern)
    "cumprod": "reduction", "cumsum": "reduction",
    "cumsum_exclusive": "reduction", "cumsum_reverse": "reduction",
    "masked_cumsum": "reduction", "matrix_scalar_multiplication": "elementwise",
    # reduce → reduction (5)
    "max_reduction_over_a_dimension": "reduction",
    "mean_reduction_over_a_dimension": "reduction",
    "min_reduction_over_a_dimension": "reduction",
    "product_reduction_over_a_dimension": "reduction",
    "sum_reduction_over_a_dimension": "reduction",
    # optimizer → elementwise (5)
    "adagrad": "elementwise", "adam": "elementwise", "lamb": "elementwise",
    "rmsprop": "elementwise", "sgd": "elementwise",
    # pooling (6)
    "average_pooling_1d": "pooling", "average_pooling_2d": "pooling",
    "average_pooling_3d": "pooling", "max_pooling_1d": "pooling",
    "max_pooling_2d": "pooling", "max_pooling_3d": "pooling",
    # matmul (17)
    "batched_matrix_multiplication": "matmul",
    "four_dim_tensor_matrix_multiplication": "matmul",
    "matmul_for_lower_triangular_matrices": "matmul",
    "matmul_for_symmetric_matrices": "matmul",
    "matmul_for_upper_triangular_matrices": "matmul",
    "matmul_with_diagonal_matrices": "matmul",
    "matmul_with_irregular_shapes": "matmul",
    "matmul_with_large_k_dimension": "matmul",
    "matmul_with_small_k_dimension": "matmul",
    "matmul_with_transposed_a": "matmul",
    "matmul_with_transposed_b": "matmul",
    "matmul_with_transposed_both": "matmul",
    "matrix_vector_multiplication": "matmul",
    "square_matrix_multiplication": "matmul",
    "standard_matrix_multiplication": "matmul",
    "tall_skinny_matrix_multiplication": "matmul",
    "three_dim_tensor_matrix_multiplication": "matmul",
    # convolution (34)
    "conv_depthwise_2d_asymmetric_input_asymmetric_kernel": "convolution",
    "conv_depthwise_2d_asymmetric_input_square_kernel": "convolution",
    "conv_depthwise_2d_square_input_asymmetric_kernel": "convolution",
    "conv_depthwise_2d_square_input_square_kernel": "convolution",
    "conv_depthwise_separable_2d": "convolution",
    "conv_pointwise_2d": "convolution",
    "conv_standard_1d": "convolution",
    "conv_standard_1d_dilated_strided": "convolution",
    "conv_standard_2d_asymmetric_input_asymmetric_kernel": "convolution",
    "conv_standard_2d_asymmetric_input_square_kernel": "convolution",
    "conv_standard_2d_square_input_asymmetric_kernel": "convolution",
    "conv_standard_2d_square_input_asymmetric_kernel_dilated_padded": "convolution",
    "conv_standard_2d_square_input_square_kernel": "convolution",
    "conv_standard_3d_asymmetric_input_asymmetric_kernel": "convolution",
    "conv_standard_3d_asymmetric_input_square_kernel": "convolution",
    "conv_standard_3d_square_input_asymmetric_kernel": "convolution",
    "conv_standard_3d_square_input_square_kernel": "convolution",
    "conv_transposed_1d": "convolution",
    "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated": "convolution",
    "conv_transposed_1d_dilated": "convolution",
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel": "convolution",
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded": "convolution",
    "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated": "convolution",
    "conv_transposed_2d_asymmetric_input_square_kernel": "convolution",
    "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided": "convolution",
    "conv_transposed_2d_square_input_asymmetric_kernel": "convolution",
    "conv_transposed_2d_square_input_square_kernel": "convolution",
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel": "convolution",
    "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped": "convolution",
    "conv_transposed_3d_asymmetric_input_square_kernel": "convolution",
    "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped": "convolution",
    "conv_transposed_3d_square_input_asymmetric_kernel": "convolution",
    "conv_transposed_3d_square_input_square_kernel": "convolution",
    "conv_transposed_3d_square_input_square_kernel_padded_dilated_strided": "convolution",
    # attention (15)
    "causal_attention": "attention", "cross_attention": "attention",
    "cross_modal_attention": "attention", "group_query_attention": "attention",
    "kv_cached_attention_inference": "attention",
    "kv_cached_chat_batch_attention": "attention",
    "kv_cached_speculative_attention": "attention",
    "linear_attention": "attention", "multi_head_attention": "attention",
    "multi_query_attention": "attention",
    "scaled_dot_product_attention": "attention",
    "scaled_dot_product_attention_inference": "attention",
    "scaled_dot_product_attention_long_context": "attention",
    "sparse_attention": "attention", "windowed_causal_attention": "attention",
    # normalization (8)
    "batch_norm": "normalization", "frobenius_norm": "normalization",
    "group_norm": "normalization", "instance_norm": "normalization",
    "l1_norm": "normalization", "l2_norm": "normalization",
    "layer_norm": "normalization", "rms_norm": "normalization",
    # index (12)
    "argmax_over_a_dimension": "index", "argmin_over_a_dimension": "index",
    "embedding": "index", "gather": "index", "index_add": "index",
    "index_copy": "index", "index_select": "index", "inplace_update": "index",
    "masked_fill": "index", "scatter": "index", "scatter_add": "index",
    "take_along_dim": "index",
    # resize (10)
    "bicubic_upsample": "resize", "bilinear_upsample": "resize",
    "downsample_bilinear": "resize", "grid_sample_affine": "resize",
    "grid_sample_random_warp": "resize", "interpolate_dynamic": "resize",
    "nearest_neighbor_upsample": "resize", "resize_with_antialias": "resize",
    "trilinear_upsample": "resize", "upsample_grid_sample": "resize",
}


class CANNInitTask(BaseTask):
    # Device pool for NPU evaluation. Each evaluation thread acquires a device
    # exclusively from the pool, ensuring at most one eval per device at a time.
    # Auto-initialized with [0] on first access if not explicitly configured.
    _device_pool: Optional[queue.Queue] = None

    @classmethod
    def init_device_pool(cls, device_ids: List[int]):
        """Initialize NPU device pool for multi-device parallel evaluation.

        Args:
            device_ids: List of NPU device IDs (e.g., [0, 1, 2, 3, 4, 5, 6, 7]).
                         Each evaluation thread will acquire one device exclusively.
        """
        cls._device_pool = queue.Queue()
        for d in device_ids:
            cls._device_pool.put(d)

    @classmethod
    def _acquire_device(cls) -> str:
        """Acquire an NPU device from the pool. Blocks until one is available.

        If no device pool has been explicitly initialized, a single-device pool
        containing only device 0 is created on first access. This ensures that
        even in single-NPU mode, at most one evaluation runs on the device at a
        time, preventing OOM from concurrent correctness/performance tests.
        """
        if cls._device_pool is None:
            cls.init_device_pool([0])
        device_id = cls._device_pool.get()
        return f"npu:{device_id}"

    @classmethod
    def _release_device(cls, device: str):
        """Release an NPU device back to the pool."""
        if cls._device_pool is not None:
            # Extract device ID from "npu:X"
            device_id = int(device.split(":")[1])
            cls._device_pool.put(device_id)

    def __init__(
        self,
        data: Dict[str, Any],
        project_path: Optional[str] = None,
        fake_mode: bool = False,
        verbose: bool = True,
        parallel: bool = False,
    ):
        self.default_project_path = project_path
        self.fake_mode = fake_mode
        self.verbose = verbose
        self.parallel = parallel
        self._parser = None
        self._template_gen = None
        super().__init__(data)

    def _process_data(self, data: Dict[str, Any]):
        self.op_name = data["op_name"]
        self.python_reference = data["python_reference"]
        self.npu_type = data.get("npu_type", "Ascend910B2")
        self.cann_version = data.get("cann_version", "8.1.0rc1")
        self.compute_pattern = data.get("compute_pattern")  # explicit pattern from map

        self._parser = OperatorSignatureParser()
        self.signature = self._parser.parse(self.python_reference, self.op_name)
        self._template_gen = AscendCTemplateGenerator(self.signature)

        self.task_info = {
            "op_name": self.op_name,
            "python_reference": self.python_reference,
            "npu_type": self.npu_type,
            "cann_version": self.cann_version,
            "signature": self.signature,
        }

        # evotoolkit Method base class requires task.spec (TaskSpec)
        self.spec = TaskSpec(
            name=self.op_name,
            modality="cann",
            extras={"npu_type": self.npu_type, "cann_version": self.cann_version},
        )

    def get_task_type(self) -> str:
        return "CANNInit"

    def _infer_compute_pattern(self) -> str:
        """Infer the compute pattern from the operator name.

        Priority: explicit data["compute_pattern"] > built-in map > keyword fallback.
        """
        # 1. Explicit override from caller
        if self.compute_pattern:
            return self.compute_pattern

        # 2. Built-in lookup table (covers all 150 benchmark operators)
        mapped = _OPERATOR_PATTERN_MAP.get(self.op_name)
        if mapped:
            return mapped

        # 3. Keyword fallback for unknown operators
        name = self.op_name.lower()

        # Reduction patterns
        if any(kw in name for kw in ["reduction", "reduce", "sum_reduction", "mean_reduction",
                                      "max_reduction", "min_reduction", "product_reduction"]):
            return "reduction"

        # Softmax family
        if "softmax" in name or "log_softmax" in name:
            return "softmax"

        # Broadcast patterns
        if "broadcast" in name:
            return "broadcast"

        # Elementwise (default for simple ops)
        simple_keywords = [
            "relu", "gelu", "sigmoid", "tanh", "elu", "swish", "softplus",
            "hardsigmoid", "selu", "mish", "hardtanh", "leaky_relu", "softsign",
            "add_bias", "where", "clamp", "abs", "exp", "log", "sqrt", "pow", "sign",
            "loss", "entropy", "mse", "hinge", "triplet", "cosine_similarity",
            "division", "elmentwise_mul", "logic_and", "max_broadcast", "power",
            "subtract", "cumsum", "cumprod", "matrix_scalar",
        ]
        if any(kw in name for kw in simple_keywords):
            return "elementwise"

        return "other"

    def get_base_task_description(self) -> str:
        """Get the base task description for prompt generation.

        Returns the basic role and device info. This is the abstract method
        required by BaseTask. For full task description including signature
        and component specification, use get_task_description() instead.
        """
        return self._get_base_description()

    def get_task_description(self, phase: str = "init") -> str:
        """根据阶段生成定制化的任务描述。

        Args:
            phase: "init" (从零生成，完整教学) 或 "evolve" (优化已有实现，精简)

        Init 阶段包括：
        - 角色定义 + 设备信息 + Python Reference
        - Operator Signature（输入输出参数）
        - Domain Primer（Level 0 编程模型 + Level 1 模式指南，来自 KnowledgeProvider）
        - Critical Constraints（关键约束，避免常见错误）
        - Attribute Access Guide（属性获取指南，仅当有 init_params 时）
        - Curated Example（来自 KnowledgeProvider 的精选示例）
        - Tiling Fundamentals（Tiling 基础知识 + UB 容量约束）
        - Tiling Edge Case Guide（Tiling 边界处理）
        - Advanced API Reference（高级 API，仅复杂算子）
        - Component Specification（6 组件的定义和模板，含 Add 完整示例）

        Evolve 阶段只包括：
        - 角色定义 + 设备信息 + Python Reference
        - Operator Signature（输入输出参数）
        - Component Specification Minimal（精简版，只有输出格式）
        """
        if phase == "evolve":
            return self._get_evolve_task_description()
        return self._get_init_task_description()

    def get_knowledge_provider(self) -> CANNKnowledgeProvider:
        """Get the knowledge provider for this task.

        Exposed so agentic fix loop can access fine-grained knowledge pieces.
        """
        return CANNKnowledgeProvider(npu_type=self.npu_type)

    def get_compute_pattern(self) -> str:
        """Get the inferred compute pattern for this operator.

        Exposed so agentic fix loop can assemble targeted knowledge.
        """
        return self._infer_compute_pattern()

    def _get_init_task_description(self) -> str:
        """Init 阶段：完整的教学 prompt，知识从 provider 灵活组装"""
        provider = self.get_knowledge_provider()
        pattern = self._infer_compute_pattern()

        parts = [
            self._get_base_description(),
            self._get_signature_summary(),
        ]

        # Knowledge from provider: Level 0 + Level 1 + constraints + example + tiling
        parts.append(provider.assemble_for_init(
            pattern=pattern,
            needs_advanced=self._needs_advanced_api(),
        ))

        # 条件性添加属性获取指南 (templated, stays on task)
        attr_guide = self._get_attribute_access_guide()
        if attr_guide:
            parts.append(attr_guide)

        # Component specification (templated, stays on task)
        parts.append(self._get_component_specification())

        return "\n\n".join(parts)

    def _get_evolve_task_description(self) -> str:
        """Evolve 阶段：精简 prompt，知识从 provider 灵活组装"""
        provider = self.get_knowledge_provider()
        pattern = self._infer_compute_pattern()

        parts = [
            self._get_base_description(),
            self._get_signature_summary(),
            provider.assemble_for_evolve(pattern),
        ]

        # Include attribute access guide if operator has init_params
        attr_guide = self._get_attribute_access_guide()
        if attr_guide:
            parts.append(attr_guide)

        parts.append(self._get_component_specification_minimal())
        return "\n\n".join(parts)

    def _get_base_description(self) -> str:
        """内部方法：基础描述（角色 + 设备 + Reference）"""
        return f"""You are an Ascend C operator development expert.
Your task is to implement an optimized, high-performance kernel for the "{self.op_name}" operator.
Target device: {self.npu_type} NPU with CANN {self.cann_version}.

Python Reference:
```python
{self.python_reference}
```"""

    def _get_signature_summary(self) -> str:
        """内部方法：签名摘要"""
        lines = ["## Operator Signature", ""]

        # Inputs
        lines.append("**Inputs (forward parameters):**")
        for inp in self.signature["inputs"]:
            dtype = inp["dtype"]
            tensor_info = "tensor" if inp.get("is_tensor", True) else "scalar"
            lines.append(f"- `{inp['name']}`: {dtype} {tensor_info}")

        # Outputs
        lines.append("")
        lines.append("**Outputs:**")
        for out in self.signature["outputs"]:
            dtype = out["dtype"]
            lines.append(f"- `{out['name']}`: {dtype} tensor")

        # Init params (if any)
        if self.signature.get("init_params"):
            lines.append("")
            lines.append("**Init Parameters (__init__ arguments):**")
            for param in self.signature["init_params"]:
                dtype = param["dtype"]
                default_str = f" = {param['default']}" if "default" in param else ""
                lines.append(f"- `{param['name']}`: {dtype}{default_str}")

        return "\n".join(lines)

    def _get_critical_constraints(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_critical_constraints()

    def _get_api_quick_reference(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_api_quick_reference()

    def _get_critical_constraints_compact(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_critical_constraints_compact()

    def _get_tiling_quick_reference(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_tiling_quick_reference()

    def _get_attribute_access_guide(self) -> str:
        """内部方法：算子属性获取指南（仅当有 init_params 时使用）"""
        if not self.signature.get("init_params"):
            return ""

        params = self.signature["init_params"]
        param_examples = []
        # Map list types to C++ types for GetAttrPointer<T>
        _list_type_map = {
            "list_int": "std::vector<int64_t>",
            "list_float": "std::vector<float>",
        }
        for i, param in enumerate(params):
            dtype = param['dtype']
            param_examples.append(f"// Get {param['name']} ({dtype})")
            if dtype == 'float':
                param_examples.append(f"float {param['name']} = *attrs->GetFloat({i});")
                param_examples.append(f"tiling.set_{param['name']}({param['name']});")
            elif dtype == 'int':
                param_examples.append(f"int64_t {param['name']} = *attrs->GetInt({i});")
                param_examples.append(f"tiling.set_{param['name']}({param['name']});")
            elif dtype == 'bool':
                param_examples.append(f"bool {param['name']} = *attrs->GetBool({i});")
                param_examples.append(f"tiling.set_{param['name']}({param['name']});")
            elif dtype in _list_type_map:
                cpp_type = _list_type_map[dtype]
                param_examples.append(f"auto {param['name']}Ptr = attrs->GetAttrPointer<{cpp_type}>({i});")
                param_examples.append(f"// {param['name']}Ptr is const {cpp_type}*")
                param_examples.append(f"// Iterate or compute derived scalars, e.g.:")
                param_examples.append(f"// int64_t product = 1;")
                param_examples.append(f"// for (auto v : *{param['name']}Ptr) product *= v;")
                param_examples.append(f"// tiling.set_{param['name']}Size(product);")
            else:
                param_examples.append(f"auto {param['name']} = *attrs->GetAttrPointer<{dtype}>({i});")
                param_examples.append(f"tiling.set_{param['name']}({param['name']});")
            param_examples.append("")

        tiling_fields = "\n".join(
            f"// derive scalar fields from {p['name']} ({p['dtype']})" if p['dtype'] in _list_type_map
            else f"{p['dtype']} {p['name']}"
            for p in params
        )
        attr_code = "\n".join(param_examples)

        return f"""## Accessing Operator Attributes

This operator has init parameters that must be passed from host (TilingFunc) to device (kernel).

### Step 1: Define fields in TILING_FIELDS
```
{tiling_fields}
```

Note: `list_int` / `list_float` attrs cannot be stored directly in TilingData.
Compute derived scalar values (e.g., product, count) and store those instead.

### Step 2: Get attributes in TILING_FUNC_BODY
```cpp
// Get attributes from context
const auto* attrs = context->GetAttrs();
{attr_code}
```

In the kernel, access these via `tilingData.{params[0]['name']}` etc.

IMPORTANT:
- Use the convenience methods: `GetInt(i)` returns `const int64_t*`, `GetFloat(i)` returns `const float*`, `GetBool(i)` returns `const bool*`.
- For list types, use `GetAttrPointer<std::vector<int64_t>>(i)` or `GetAttrPointer<std::vector<float>>(i)` — returns a const pointer to the vector.
- Do NOT call `GetAttrPointer(i)` without a template parameter — it will NOT compile.
- Do NOT invent APIs like `gert::ArrayAttr`, `GetListInt()`, `GetInitParam()` — they do not exist!"""

    def _get_tiling_fundamentals(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_tiling_fundamentals()

    def _get_tiling_edge_case_guide(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_tiling_edge_cases()

    def _needs_advanced_api(self) -> bool:
        """判断当前算子是否需要高级 API（Matmul/Norm/Index/Transpose）

        Cube and mixed paradigm operators always need advanced API.
        Vector operators only need it for specific complex cases.
        """
        pattern = self._infer_compute_pattern()
        paradigm = CANNKnowledgeProvider.get_paradigm(pattern)
        if paradigm in ("cube", "mixed"):
            return True

        # Vector paradigm: check for specific complex cases
        name = self.op_name.lower()
        simple_keywords = [
            "relu", "gelu", "sigmoid", "tanh", "elu", "swish", "softmax",
            "softplus", "hardsigmoid", "selu", "mish", "hardtanh", "leaky_relu",
            "broadcast", "add_bias", "where", "clamp",
            "abs", "exp", "log", "sqrt", "pow", "sign", "softsign",
            "sum_reduction", "mean_reduction", "max_reduction", "min_reduction",
            "product_reduction", "frobenius",
            "loss", "entropy", "mse", "hinge", "triplet", "cosine_similarity",
        ]
        if any(kw in name for kw in simple_keywords):
            return False
        return True

    def _get_component_specification_minimal(self) -> str:
        """精简版组件规范：只有文件结构和输出格式，无 Add 完整示例。

        用于 evolve 阶段，此时父代代码已经包含了所有必要的模式知识。
        """
        op_name_snake = self.op_name.replace("-", "_").lower()
        op_custom = f"{op_name_snake}_custom"
        op_custom_capital = "".join(
            word.capitalize() for word in op_custom.split("_")
        )

        tensor_inputs = [
            inp for inp in self.signature["inputs"] if inp.get("is_tensor", True)
        ]
        gm_params = [inp["name"] for inp in tensor_inputs]
        gm_params.extend([out["name"] for out in self.signature["outputs"]])
        gm_signature = ", ".join(f"GM_ADDR {p}" for p in gm_params)
        first_input = tensor_inputs[0]["name"] if tensor_inputs else "x"

        # Python binding 签名和 EXEC_NPU_CMD 参数（包含 init_params）
        init_params = self.signature.get("init_params", [])
        pybind_param_parts = [f"const at::Tensor& {inp['name']}" for inp in tensor_inputs]
        exec_args = [inp["name"] for inp in tensor_inputs]
        for param in init_params:
            cpp_type = {"int": "int64_t", "float": "float", "bool": "bool"}.get(
                param.get("dtype", "float"), param.get("dtype", "float")
            )
            if param.get("is_tensor", False):
                pybind_param_parts.append(f"const at::Tensor& {param['name']}")
            else:
                pybind_param_parts.append(f"{cpp_type} {param['name']}")
            exec_args.append(param["name"])
        pybind_params = ", ".join(pybind_param_parts)
        exec_args_str = ", ".join(exec_args)

        return f"""## Code Architecture (Compact)

Provide **6 components**: KERNEL_IMPL, KERNEL_ENTRY_BODY, TILING_FIELDS, TILING_FUNC_BODY, INFER_SHAPE_BODY, OUTPUT_ALLOC_CODE.

kernel entry: `{op_custom}({gm_signature}, GM_ADDR workspace, GM_ADDR tiling)`
tiling class: `{op_custom_capital}TilingData`
python bind: `{op_custom}_impl_npu({pybind_params})`
EXEC_NPU_CMD: `aclnn{op_custom_capital}, {exec_args_str}, result`

Optional: KERNEL_INCLUDES, TILING_INCLUDES, TILING_FUNC_INCLUDES"""

    def _get_advanced_api_reference(self) -> str:
        """Delegate to provider."""
        return self.get_knowledge_provider().get_advanced_api_reference()

    def _get_component_specification(self) -> str:
        """内部方法：完整代码架构说明和组件标注"""
        # 准备变量
        op_name = self.op_name
        op_name_snake = op_name.replace("-", "_").lower()
        op_name_pascal = "".join(word.capitalize() for word in op_name_snake.split("_"))
        # 实际模板使用 _custom 后缀
        op_custom = f"{op_name_snake}_custom"
        op_custom_capital = "".join(word.capitalize() for word in op_custom.split("_"))

        # 输入输出参数
        tensor_inputs = [inp for inp in self.signature["inputs"] if inp.get("is_tensor", True)]
        gm_params = [inp["name"] for inp in tensor_inputs]
        gm_params.extend([out["name"] for out in self.signature["outputs"]])
        gm_signature = ", ".join(f"GM_ADDR {p}" for p in gm_params)
        gm_args = ", ".join(gm_params)

        # 第一个输入名
        first_input = tensor_inputs[0]["name"] if tensor_inputs else "x"

        # Python binding 函数签名（包含 init_params）
        init_params = self.signature.get("init_params", [])
        pybind_param_parts = [f"const at::Tensor& {inp['name']}" for inp in tensor_inputs]
        for param in init_params:
            cpp_type = {"int": "int64_t", "float": "float", "bool": "bool"}.get(
                param.get("dtype", "float"), param.get("dtype", "float")
            )
            if param.get("is_tensor", False):
                pybind_param_parts.append(f"const at::Tensor& {param['name']}")
            else:
                pybind_param_parts.append(f"{cpp_type} {param['name']}")
        pybind_params = ", ".join(pybind_param_parts)

        # EXEC_NPU_CMD 参数（包含 init_params）
        exec_cmd_args = [inp["name"] for inp in tensor_inputs]
        exec_cmd_args.extend([p["name"] for p in init_params])
        exec_cmd_args_str = ", ".join(exec_cmd_args)

        spec = f"""## Code Architecture

Ascend C operator requires 4 source files. You need to provide **6 components** that will be assembled into these files:

| File | Description | Your Components |
|------|-------------|-----------------|
| **kernel_src** | Device kernel running on NPU | KERNEL_IMPL, KERNEL_ENTRY_BODY |
| **host_tiling_src** | TilingData structure definition | TILING_FIELDS |
| **host_operator_src** | Host-side TilingFunc and InferShape | TILING_FUNC_BODY, INFER_SHAPE_BODY |
| **python_bind_src** | PyTorch Python binding | OUTPUT_ALLOC_CODE |

---

## File Templates (Structure)

### 1. kernel_src
```cpp
#include "kernel_operator.h"
// [KERNEL_INCLUDES - optional]

[KERNEL_IMPL]  // Your Kernel class goes here

extern "C" __global__ __aicore__ void {op_custom}(
    {gm_signature}, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
    [KERNEL_ENTRY_BODY]  // Your entry code goes here
}}
```

### 2. host_tiling_src
```cpp
#include "register/tilingdata_base.h"
// [TILING_INCLUDES - optional]

BEGIN_TILING_DATA_DEF({op_custom_capital}TilingData)
    [TILING_FIELDS]  // Your fields go here
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({op_custom_capital}, {op_custom_capital}TilingData)
```

### 3. host_operator_src
```cpp
#include "{op_custom}_tiling.h"
#include "register/op_def_registry.h"
// [TILING_FUNC_INCLUDES - optional]

namespace optiling {{
static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
    [TILING_FUNC_BODY]  // Your tiling logic goes here
}}

static ge::graphStatus InferShape(gert::InferShapeContext* context) {{
    [INFER_SHAPE_BODY]  // Your shape inference goes here
}}
}}
```

### 4. python_bind_src
```cpp
#include <torch/extension.h>
#include "aclnn_{op_custom}.h"

at::Tensor {op_custom}_impl_npu({pybind_params}) {{
    [OUTPUT_ALLOC_CODE]  // Must define 'result'
    EXEC_NPU_CMD(aclnn{op_custom_capital}, {exec_cmd_args_str}, result);
    return result;
}}
```

---

## Complete Example: Add Operator (x + y → z)

Below is a complete working example showing all 6 components for element-wise addition.
This example demonstrates UB-aware tiling for large tensors.

### KERNEL_IMPL
```cpp
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {{
public:
    __aicore__ inline KernelAdd() {{}}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t tileLength) {{
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = tileLength;  // UB-safe, calculated by TilingFunc
        this->tailLength = this->blockLength - tileNum * BUFFER_NUM * tileLength;
        this->hasTail = (this->tailLength > 0);

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }}

    __aicore__ inline void Process() {{
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
        if (this->hasTail) {{
            CopyInTail();
            ComputeTail();
            CopyOutTail();
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }}

    __aicore__ inline void Compute(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress) {{
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }}

    __aicore__ inline void CopyInTail() {{
        uint32_t offset = this->tileNum * BUFFER_NUM * this->tileLength;
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[offset], this->tailLength);
        DataCopy(yLocal, yGm[offset], this->tailLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }}

    __aicore__ inline void ComputeTail() {{
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Add(zLocal, xLocal, yLocal, this->tailLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }}

    __aicore__ inline void CopyOutTail() {{
        uint32_t offset = this->tileNum * BUFFER_NUM * this->tileLength;
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[offset], zLocal, this->tailLength);
        outQueueZ.FreeTensor(zLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm, yGm, zGm;
    uint32_t blockLength, tileNum, tileLength, tailLength;
    bool hasTail;
}};
```

### KERNEL_ENTRY_BODY
```cpp
KernelAdd op;
op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum, tilingData.tileLength);
op.Process();
```

### TILING_FIELDS
```
uint32_t totalLength
uint32_t tileNum
uint32_t tileLength
```

Format: `TYPE NAME` or `TYPE NAME[SIZE]` or `struct TYPE NAME`

### TILING_FUNC_BODY
```cpp
AddCustomTilingData tiling;

auto shape = context->GetInputShape(0)->GetStorageShape();
uint32_t totalLength = 1;
for (size_t i = 0; i < shape.GetDimNum(); i++) {{
    totalLength *= shape.GetDim(i);
}}

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 176 * 1024;  // 176KB usable UB (256KB - 80KB system/TPipe reserve)
constexpr uint32_t NUM_QUEUES = 3;         // inQueueX + inQueueY + outQueueZ

// Calculate max tileLength that fits in UB
uint32_t maxTileLength = UB_SIZE / (NUM_QUEUES * BUFFER_NUM * sizeof(float));
maxTileLength = maxTileLength / 8 * 8;  // Align to 32 bytes (8 floats)

uint32_t blockLength = totalLength / BLOCK_DIM;
uint32_t tileNum = blockLength / (maxTileLength * BUFFER_NUM);
if (tileNum == 0) tileNum = 1;
uint32_t tileLength = blockLength / (tileNum * BUFFER_NUM);
tileLength = tileLength / 8 * 8;  // Align to 32 bytes

tiling.set_totalLength(totalLength);
tiling.set_tileNum(tileNum);
tiling.set_tileLength(tileLength);

tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(BLOCK_DIM);

size_t* currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = 0;

return ge::GRAPH_SUCCESS;
```

### INFER_SHAPE_BODY
```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
```

### OUTPUT_ALLOC_CODE
```cpp
at::Tensor result = at::empty_like({first_input});
```

---

## Optional Components

- **KERNEL_INCLUDES**: Extra kernel headers (e.g., `lib/matmul_intf.h`)
- **TILING_INCLUDES**: Extra tiling headers (e.g., `tiling/platform/platform_ascendc.h` for `TCubeTiling`)
- **TILING_FUNC_INCLUDES**: Extra TilingFunc headers
"""
        return spec

    def format_solution_components(self, solution: "Solution") -> str:
        """格式化 solution 的 6 个组件为可读的字符串"""
        if not solution.other_info:
            return "(empty solution)"

        info = solution.other_info
        parts = []

        if info.get("kernel_impl"):
            parts.append(f"### KERNEL_IMPL\n```cpp\n{info['kernel_impl']}\n```")

        if info.get("kernel_entry_body"):
            parts.append(f"### KERNEL_ENTRY_BODY\n```cpp\n{info['kernel_entry_body']}\n```")

        if info.get("tiling_fields"):
            fields_str = self._format_tiling_fields(info["tiling_fields"])
            parts.append(f"### TILING_FIELDS\n```\n{fields_str}\n```")

        if info.get("tiling_func_body"):
            parts.append(f"### TILING_FUNC_BODY\n```cpp\n{info['tiling_func_body']}\n```")

        if info.get("infer_shape_body"):
            parts.append(f"### INFER_SHAPE_BODY\n```cpp\n{info['infer_shape_body']}\n```")

        if info.get("output_alloc_code"):
            parts.append(f"### OUTPUT_ALLOC_CODE\n```cpp\n{info['output_alloc_code']}\n```")

        return "\n\n".join(parts) if parts else "(no components)"

    def _format_tiling_fields(self, fields) -> str:
        """将 tiling_fields 列表格式化为文本格式

        Args:
            fields: Either a list of field dicts, or a dict with "fields" and "includes" keys.
        """
        # Handle dict format with embedded includes
        if isinstance(fields, dict):
            fields = fields.get("fields", [])
        lines = []
        for field in fields:
            if field.get("is_struct"):
                # struct TYPE NAME
                lines.append(f"struct {field['type']} {field['name']}")
            elif field.get("size"):
                # TYPE NAME[SIZE]
                lines.append(f"{field['type']} {field['name']}[{field['size']}]")
            else:
                # TYPE NAME
                lines.append(f"{field['type']} {field['name']}")
        return "\n".join(lines)

    def _make_result(
        self,
        valid: bool,
        stage: str,
        score: Optional[float] = None,
        error: Optional[str] = None,
        **extra,
    ) -> EvaluationResult:
        """辅助方法：构造 EvaluationResult"""
        info = {"stage": stage}
        if error:
            info["error"] = error
        info.update(extra)
        return EvaluationResult(valid=valid, score=score, additional_info=info)

    def _run_verify_and_perf(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """执行正确性验证和性能测量。

        parallel=False (默认): 单次沙箱调用 verify_and_measure()，效率更高。
        parallel=True: 拆分为 verify_correctness() + measure_performance()，
            正确性验证可并行，性能测量通过类级锁串行，避免 NPU 资源竞争。
        """
        if self.parallel:
            return self._run_verify_and_perf_parallel(
                evaluator, config, kernel_src, project_path, extra_info
            )
        return self._run_verify_and_perf_serial(
            evaluator, config, kernel_src, project_path, extra_info
        )

    def _run_verify_and_perf_serial(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """单次沙箱调用，适合串行评估。支持多设备池。"""
        base_info = {"kernel_src": kernel_src, "project_path": project_path}
        if extra_info:
            base_info.update(extra_info)

        device = self._acquire_device()
        try:
            result = evaluator.verify_and_measure(
                python_reference=self.python_reference,
                skip_correctness=config.skip_correctness,
                skip_performance=config.skip_performance,
                device=device,
            )
        finally:
            self._release_device(device)

        # Handle unexpected sandbox error
        if "error" in result and result.get("correctness") is None and result.get("performance") is None:
            return self._make_result(
                valid=False,
                stage="sandbox",
                error=result["error"],
                **base_info,
            )

        # Handle correctness failure
        corr = result.get("correctness")
        if corr is not None and not corr.get("pass", False):
            return self._make_result(
                valid=False,
                stage="correctness",
                error=corr.get("error"),
                python_output=corr.get("python_output"),
                ascend_output=corr.get("ascend_output"),
                max_diff=corr.get("max_diff"),
                **base_info,
            )

        # Handle performance result
        perf = result.get("performance")
        if perf is not None:
            runtime = perf.get("runtime")
            if runtime is None:
                return self._make_result(
                    valid=False,
                    stage="performance",
                    error=perf.get("error", "Performance measurement failed"),
                    **base_info,
                )
            return self._make_result(
                valid=True,
                stage="success",
                score=-runtime,
                runtime=runtime,
                runtime_std=perf.get("std"),
                baseline_runtime=perf.get("baseline_runtime"),
                baseline_std=perf.get("baseline_std"),
                speedup=perf.get("speedup"),
                **base_info,
            )

        # Correctness only (skip_performance=True)
        return self._make_result(valid=True, stage="correctness_only", **base_info)

    def _run_verify_and_perf_parallel(
        self,
        evaluator: AscendCEvaluator,
        config: CANNSolutionConfig,
        kernel_src: str,
        project_path: str,
        extra_info: Optional[Dict] = None,
    ) -> EvaluationResult:
        """拆分调用，适合并发评估。

        With device pool: each evaluation acquires a dedicated NPU device,
        enabling fully parallel correctness + performance across multiple cards.
        Device pool guarantees at most one evaluation per device at a time.
        """
        base_info = {"kernel_src": kernel_src, "project_path": project_path}
        if extra_info:
            base_info.update(extra_info)

        device = self._acquire_device()
        try:
            # Correctness check
            if not config.skip_correctness:
                verify_result = evaluator.verify_correctness(
                    self.python_reference, self.op_name, device=device,
                )
                if not verify_result["pass"]:
                    return self._make_result(
                        valid=False,
                        stage="correctness",
                        error=verify_result["error"],
                        python_output=verify_result.get("python_output"),
                        ascend_output=verify_result.get("ascend_output"),
                        max_diff=verify_result.get("max_diff"),
                        **base_info,
                    )

            # Performance measurement — device pool guarantees exclusive access,
            # so no additional locking is needed.
            if not config.skip_performance:
                perf_result = evaluator.measure_performance(
                    self.op_name, python_reference=self.python_reference,
                    device=device,
                )
                runtime = perf_result.get("runtime")

                if runtime is None:
                    return self._make_result(
                        valid=False,
                        stage="performance",
                        error=perf_result.get("error", "Performance measurement failed"),
                        **base_info,
                    )

                return self._make_result(
                    valid=True,
                    stage="success",
                    score=-runtime,
                    runtime=runtime,
                    runtime_std=perf_result.get("std"),
                    baseline_runtime=perf_result.get("baseline_runtime"),
                    baseline_std=perf_result.get("baseline_std"),
                    speedup=perf_result.get("speedup"),
                    **base_info,
                )
        finally:
            self._release_device(device)

        return self._make_result(valid=True, stage="correctness_only", **base_info)

    def evaluate_code(self, candidate_code: str) -> EvaluationResult:  # noqa: ARG002
        return self._make_result(
            valid=False,
            stage="validation",
            error="CANNInitTask requires evaluate_solution() with other_info containing tiling_fields, tiling_func_body, infer_shape_body",
        )

    def evaluate(self, solution: Solution) -> EvaluationResult:
        """evotoolkit Method base class calls task.evaluate() — delegate to evaluate_solution."""
        return self.evaluate_solution(solution)

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        config = CANNSolutionConfig.from_dict(solution.other_info)
        kernel_src = solution.sol_string

        try:
            project_path = config.project_path or self.default_project_path
            if project_path is None:
                project_path = tempfile.mkdtemp(prefix=f"cann_{self.op_name}_")

            # 从已保存结果加载
            if config.load_from:
                evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type, verbose=self.verbose)
                return self._evaluate_from_loaded(evaluator, config)

            # 验证必要字段 (6 个组件都必须提供)
            required_fields = [
                config.tiling_fields,
                config.tiling_func_body,
                config.infer_shape_body,
                config.output_alloc_code,
                config.kernel_impl,
                config.kernel_entry_body,
            ]
            if not all(required_fields):
                return self._make_result(
                    valid=False,
                    stage="validation",
                    error="Missing required fields: tiling_fields, tiling_func_body, infer_shape_body, output_alloc_code, kernel_impl, kernel_entry_body",
                    kernel_src=kernel_src,
                )

            # 生成完整代码
            full_code = self._template_gen.generate(
                kernel_impl=config.kernel_impl,
                kernel_entry_body=config.kernel_entry_body,
                tiling_fields=config.tiling_fields,
                tiling_func_body=config.tiling_func_body,
                infer_shape_body=config.infer_shape_body,
                project_path=project_path,
                output_alloc_code=config.output_alloc_code,
            )

            # fake_mode: 仅写入文件
            if self.fake_mode:
                return self._handle_fake_mode(full_code, project_path, kernel_src)

            # 完整编译流程
            evaluator = AscendCEvaluator(project_path=project_path, device=self.npu_type, verbose=self.verbose)
            compile_result = evaluator.compile(full_code, self.op_name, project_path=project_path, kernel_src=kernel_src)

            if not compile_result.success:
                return self._make_result(
                    valid=False,
                    stage="compile",
                    error=compile_result.error,
                    kernel_src=kernel_src,
                    project_path=project_path,
                )

            if config.save_compile_to:
                compile_result.save(config.save_compile_to)

            if config.compile_only:
                return self._make_result(
                    valid=True,
                    stage="compile_only",
                    project_path=project_path,
                    kernel_src=kernel_src,
                    compile_result=compile_result,
                )

            return self._run_verify_and_perf(evaluator, config, kernel_src, project_path)

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="exception",
                error=str(e),
                kernel_src=kernel_src,
            )

    def _handle_fake_mode(
        self, full_code: Dict, project_path: str, kernel_src: str
    ) -> EvaluationResult:
        """处理 fake_mode: 仅写入文件不编译"""
        from .utils.backend import write_project_files

        write_result = write_project_files(full_code=full_code, op_name=self.op_name, project_path=project_path)

        if not write_result["success"]:
            return self._make_result(
                valid=False,
                stage="write_files",
                error=write_result["error"],
                fake_mode=True,
                project_path=project_path,
                kernel_src=kernel_src,
            )

        return self._make_result(
            valid=True,
            stage="files_written",
            score=1.0,
            fake_mode=True,
            project_path=project_path,
            kernel_src=kernel_src,
            generated_components=list(full_code.keys()),
            files_written=write_result.get("files_written", []),
        )

    def _evaluate_from_loaded(
        self, evaluator: AscendCEvaluator, config: CANNSolutionConfig
    ) -> EvaluationResult:
        """从已保存的编译结果加载并继续评估"""
        try:
            compile_result = CompileResult.load(config.load_from)

            if not compile_result.is_loadable():
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Loaded compile result is not usable",
                    load_from=config.load_from,
                )

            evaluator.project_path = compile_result.project_path

            if not evaluator.rebuild_context(compile_result):
                return self._make_result(
                    valid=False,
                    stage="load",
                    error="Failed to rebuild context from loaded result",
                    load_from=config.load_from,
                )

            return self._run_verify_and_perf(
                evaluator,
                config,
                compile_result.kernel_src,
                compile_result.project_path,
                extra_info={"load_from": config.load_from},
            )

        except Exception as e:
            return self._make_result(
                valid=False,
                stage="load_exception",
                error=str(e),
                load_from=config.load_from,
            )

    def make_init_sol_wo_other_info(self) -> Solution:
        return Solution("")

    def cleanup(self):
        pass

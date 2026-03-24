"""完整示例：ReLU 算子评估

使用真实的 LLM 生成组件，展示从构建 task 到获取评估结果的完整流程。
6 个组件的源码见 relu_components/ 目录。
运行前需要昇腾 NPU 环境 + CANN Toolkit + torch-npu。
"""

import json
from pathlib import Path

from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

EXAMPLES_DIR = Path(__file__).parent
COMPONENTS_DIR = EXAMPLES_DIR / "relu_components"

# ──────────────────────────────────────────────
# 1. Python Reference（org 格式）
# ──────────────────────────────────────────────
python_reference = (EXAMPLES_DIR / "relu_org.py").read_text()

# ──────────────────────────────────────────────
# 2. 从文件加载 LLM 生成的 6 个组件
# ──────────────────────────────────────────────
kernel_impl = (COMPONENTS_DIR / "kernel_impl.cpp").read_text()
kernel_entry_body = (COMPONENTS_DIR / "kernel_entry_body.cpp").read_text()
tiling_fields = json.loads((COMPONENTS_DIR / "tiling_fields.json").read_text())
tiling_func_body = (COMPONENTS_DIR / "tiling_func_body.cpp").read_text()
infer_shape_body = (COMPONENTS_DIR / "infer_shape_body.cpp").read_text()
output_alloc_code = (COMPONENTS_DIR / "output_alloc_code.cpp").read_text().strip()

# ──────────────────────────────────────────────
# 3. 构建任务并评估
# ──────────────────────────────────────────────
task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": python_reference,
})

config = CANNSolutionConfig(
    kernel_impl=kernel_impl,
    kernel_entry_body=kernel_entry_body,
    tiling_fields=tiling_fields,
    tiling_func_body=tiling_func_body,
    infer_shape_body=infer_shape_body,
    output_alloc_code=output_alloc_code,
)

result = task.evaluate_solution(Solution("", config.to_dict()))

# ──────────────────────────────────────────────
# 4. 输出结果
# ──────────────────────────────────────────────
if result.valid:
    info = result.additional_info
    print(f"[PASS] Runtime: {info['runtime']:.4f} ms "
          f"(baseline: {info['baseline_runtime']:.4f} ms, "
          f"speedup: {info['speedup']:.3f}x)")
else:
    info = result.additional_info
    print(f"[FAIL] Stage: {info['stage']}, Error: {info['error']}")

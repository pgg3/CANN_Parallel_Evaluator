"""完整示例：ReLU 算子评估

使用真实的 LLM 生成组件，展示从构建 task 到获取评估结果的完整流程。
4 个完整源文件见 relu_components/ 目录。
运行前需要昇腾 NPU 环境 + CANN Toolkit + torch-npu。
"""

from pathlib import Path

from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

EXAMPLES_DIR = Path(__file__).parent
COMPONENTS_DIR = EXAMPLES_DIR / "relu_components"

# ──────────────────────────────────────────────
# 1. Python Reference（org 格式）
# ──────────────────────────────────────────────
python_reference = (EXAMPLES_DIR / "relu_org.py").read_text()

# ──────────────────────────────────────────────
# 2. 加载 4 个完整源文件
# ──────────────────────────────────────────────
op_kernel = (COMPONENTS_DIR / "op_kernel.cpp").read_text()
op_host_tiling = (COMPONENTS_DIR / "op_host_tiling.h").read_text()
op_host = (COMPONENTS_DIR / "op_host.cpp").read_text()
pybinding = (COMPONENTS_DIR / "pybinding.cpp").read_text()

# ──────────────────────────────────────────────
# 3. 构建任务并评估
# ──────────────────────────────────────────────
task = CANNInitTask(data={
    "op_name": "relu",
    "python_reference": python_reference,
})

config = CANNSolutionConfig(
    op_kernel=op_kernel,           # op_kernel/*.cpp
    op_host_tiling=op_host_tiling, # op_host/*_tiling.h
    op_host=op_host,               # op_host/*.cpp
    pybinding=pybinding,           # CppExtension/csrc/op.cpp
)

if __name__ == "__main__":
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

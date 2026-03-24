# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
cann_parallel_evaluator — Ascend C operator evaluation framework.

Provides compile, correctness verification, and performance measurement
for Ascend C operators, with multi-NPU parallel evaluation support.

Quick start:
    from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

    task = CANNInitTask({"op_name": "relu", "python_reference": PYTHON_REF})
    config = CANNSolutionConfig(
        kernel_impl=..., kernel_entry_body=..., tiling_fields=...,
        tiling_func_body=..., infer_shape_body=..., output_alloc_code=...,
    )
    solution = Solution("", other_info=config.to_dict())
    result = task.evaluate_solution(solution)
"""

from .core_types import EvaluationResult, Solution, SolutionMetadata, TaskSpec, BaseTask
from .cann_init_task import CANNInitTask
from .data_structures import CompileResult, CANNSolutionConfig
from .evaluator import AscendCEvaluator
from .signature_parser import OperatorSignatureParser
from .knowledge import CANNKnowledgeProvider
from .utils.templates import AscendCTemplateGenerator

__all__ = [
    "EvaluationResult",
    "Solution",
    "SolutionMetadata",
    "TaskSpec",
    "BaseTask",
    "CANNInitTask",
    "CompileResult",
    "CANNSolutionConfig",
    "AscendCEvaluator",
    "OperatorSignatureParser",
    "CANNKnowledgeProvider",
    "AscendCTemplateGenerator",
]

# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CompileResult:
    success: bool
    error: Optional[str] = None
    project_path: Optional[str] = None
    op_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    kernel_src: Optional[str] = None
    full_code: Optional[Dict[str, str]] = None

    def save(self, path: str) -> None:
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "success": self.success,
            "error": self.error,
            "project_path": self.project_path,
            "op_name": self.op_name,
            "kernel_src": self.kernel_src,
        }

        with open(save_path / "compile_result.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if self.full_code:
            with open(save_path / "full_code.json", "w") as f:
                json.dump(self.full_code, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CompileResult":
        load_path = Path(path)

        with open(load_path / "compile_result.json") as f:
            metadata = json.load(f)

        full_code = None
        full_code_path = load_path / "full_code.json"
        if full_code_path.exists():
            with open(full_code_path) as f:
                full_code = json.load(f)

        return cls(
            success=metadata["success"],
            error=metadata.get("error"),
            project_path=metadata.get("project_path"),
            op_name=metadata.get("op_name"),
            kernel_src=metadata.get("kernel_src"),
            full_code=full_code,
            context={},
        )

    def is_loadable(self) -> bool:
        return self.success and self.project_path is not None


@dataclass
class CANNSolutionConfig:
    """CANN Solution 配置，包含 LLM 生成的完整源文件和执行控制参数。

    LLM 生成 3 个逻辑单元（4 个物理文件）：
    - OP_KERNEL:     op_kernel/*.cpp           (完整 kernel 源文件)
    - OP_HOST:       op_host/*_tiling.h        (完整 tiling header)
                     op_host/*.cpp             (完整 host 源文件)
    - PYBINDING:     CppExtension/csrc/op.cpp  (完整 pybinding 源文件)
    """

    # === Project Config ===
    project_path: Optional[str] = None

    # === Complete source files (LLM-generated) ===
    op_kernel: Optional[str] = None       # Complete op_kernel/*.cpp
    op_host_tiling: Optional[str] = None  # Complete op_host/*_tiling.h
    op_host: Optional[str] = None         # Complete op_host/*.cpp
    pybinding: Optional[str] = None       # Complete CppExtension/csrc/op.cpp

    # === Execution control ===
    compile_only: bool = False                  # Only compile, skip correctness/performance
    load_from: Optional[str] = None             # Load compiled result from path
    save_compile_to: Optional[str] = None       # Save compiled result to path
    skip_correctness: bool = False              # Skip correctness verification
    skip_performance: bool = False              # Skip performance measurement

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CANNSolutionConfig":
        if not d:
            return cls()

        return cls(
            # Project config
            project_path=d.get("project_path"),
            # Complete source files
            op_kernel=d.get("op_kernel"),
            op_host_tiling=d.get("op_host_tiling"),
            op_host=d.get("op_host"),
            pybinding=d.get("pybinding"),
            # Execution control
            compile_only=d.get("compile_only", False),
            load_from=d.get("load_from"),
            save_compile_to=d.get("save_compile_to"),
            skip_correctness=d.get("skip_correctness", False),
            skip_performance=d.get("skip_performance", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {}

        # Project config
        if self.project_path is not None:
            result["project_path"] = self.project_path

        # Complete source files
        if self.op_kernel is not None:
            result["op_kernel"] = self.op_kernel
        if self.op_host_tiling is not None:
            result["op_host_tiling"] = self.op_host_tiling
        if self.op_host is not None:
            result["op_host"] = self.op_host
        if self.pybinding is not None:
            result["pybinding"] = self.pybinding

        # Execution control
        if self.compile_only:
            result["compile_only"] = True
        if self.load_from is not None:
            result["load_from"] = self.load_from
        if self.save_compile_to is not None:
            result["save_compile_to"] = self.save_compile_to
        if self.skip_correctness:
            result["skip_correctness"] = True
        if self.skip_performance:
            result["skip_performance"] = True

        return result

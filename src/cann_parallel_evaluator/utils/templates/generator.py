# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from typing import Any, Dict, Optional

from .project_json import ProjectJsonGenerator
from .model_src import ModelSrcGenerator


class AscendCTemplateGenerator:
    def __init__(self, signature: Dict[str, Any]):
        self.signature = signature
        self._project_json_gen = ProjectJsonGenerator(signature)
        self._model_src_gen = ModelSrcGenerator(signature)

    def generate(
        self,
        op_kernel: str,
        op_host_tiling: str,
        op_host: str,
        pybinding: str,
        project_path: str,
        soc_versions: Optional[list] = None,
    ) -> Dict[str, str]:
        """Assemble full_code dict from 4 LLM-provided complete source files.

        The 4 source files are passed through unchanged. Only project_json_src
        and model_src are auto-generated from the operator signature.

        Returns a dict with the same keys consumed by ascend_compile.py:
          kernel_src, host_tiling_src, host_operator_src, python_bind_src,
          project_json_src, model_src
        """
        return {
            "project_json_src": self._project_json_gen.generate(),
            "kernel_src": op_kernel,
            "host_tiling_src": op_host_tiling,
            "host_operator_src": op_host,
            "python_bind_src": pybinding,
            "model_src": self._model_src_gen.generate(project_path),
        }


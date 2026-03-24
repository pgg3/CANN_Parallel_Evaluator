# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

import copy
import re
from typing import Any, Dict, List, Optional

from .project_json import ProjectJsonGenerator
from .host_tiling import HostTilingGenerator
from .host_operator import HostOperatorGenerator
from .python_bind import PythonBindGenerator
from .model_src import ModelSrcGenerator
from .kernel_src import KernelSrcGenerator


class AscendCTemplateGenerator:
    def __init__(self, signature: Dict[str, Any]):
        self.signature = signature
        self._project_json_gen = ProjectJsonGenerator(signature)
        self._host_tiling_gen = HostTilingGenerator(signature)
        self._python_bind_gen = PythonBindGenerator(signature)
        self._model_src_gen = ModelSrcGenerator(signature)
        self._kernel_src_gen = KernelSrcGenerator(signature)

    def generate(
        self,
        kernel_impl: str,
        kernel_entry_body: str,
        tiling_fields: List[Dict[str, str]],
        tiling_func_body: str,
        infer_shape_body: str,
        project_path: str,
        output_alloc_code: str,
        soc_versions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        # Normalize tiling_fields: support dict format with embedded includes
        tiling_includes = None
        if isinstance(tiling_fields, dict):
            tiling_includes = tiling_fields.get("includes")
            tiling_fields = tiling_fields["fields"]

        # Detect dtype casts in output_alloc_code to adjust operator registration.
        # If binding casts inputs to half, operator must be registered for float16.
        op_signature = self._detect_dtype_overrides(output_alloc_code)

        host_tiling_src = self._host_tiling_gen.generate(
            tiling_fields=tiling_fields,
            tiling_includes=tiling_includes,
        )
        host_operator_gen = HostOperatorGenerator(op_signature)
        host_operator_src = host_operator_gen.generate(
            tiling_func_body=tiling_func_body,
            infer_shape_body=infer_shape_body,
            soc_versions=soc_versions,
        )
        python_bind_src = self._python_bind_gen.generate(output_alloc_code)
        kernel_src = self._kernel_src_gen.generate(
            kernel_impl=kernel_impl,
            kernel_entry_body=kernel_entry_body,
        )

        return {
            "project_json_src": self._project_json_gen.generate(),
            "host_tiling_src": host_tiling_src,
            "host_operator_src": host_operator_src,
            "kernel_src": kernel_src,
            "python_bind_src": python_bind_src,
            "model_src": self._model_src_gen.generate(project_path),
        }

    def _detect_dtype_overrides(self, output_alloc_code: str) -> Dict[str, Any]:
        """Check if output_alloc_code casts inputs to a different dtype.

        If e.g. ``A = A.to(at::kHalf);`` is found, override A's registered
        dtype from float to float16 so that the operator registration matches
        what EXEC_NPU_CMD actually receives.
        """
        # Map of C++ cast target → dtype string for registration
        cast_map = {
            "at::kHalf": "float16",
            "torch::kHalf": "float16",
            "at::kFloat": "float",
            "torch::kFloat": "float",
        }

        # Find patterns like: A = A.to(at::kHalf)  or  B = B.to(torch::kHalf)
        cast_pattern = re.compile(r"(\w+)\s*=\s*\w+\.to\((at::kHalf|torch::kHalf|at::kFloat|torch::kFloat)\)")

        overrides = {}
        for m in cast_pattern.finditer(output_alloc_code):
            var_name = m.group(1)
            cast_target = m.group(2)
            overrides[var_name] = cast_map.get(cast_target, "float")

        if not overrides:
            return self.signature

        # Deep copy signature and apply overrides
        sig = copy.deepcopy(self.signature)
        for inp in sig.get("inputs", []):
            if inp["name"] in overrides:
                inp["dtype"] = overrides[inp["name"]]
        for param in sig.get("init_params", []):
            if param["name"] in overrides:
                param["dtype"] = overrides[param["name"]]

        # Check if output uses options from a cast input (e.g., A.options())
        # If output inherits dtype from a cast input, registration must match
        for out in sig.get("outputs", []):
            for var_name, dtype in overrides.items():
                # A.options() without .dtype() → output inherits cast dtype
                if f"{var_name}.options()" in output_alloc_code:
                    # Check it's NOT followed by .dtype() override
                    has_dtype_override = re.search(
                        rf'{var_name}\.options\(\)\.dtype\(', output_alloc_code
                    )
                    if not has_dtype_override:
                        out["dtype"] = dtype
        return sig

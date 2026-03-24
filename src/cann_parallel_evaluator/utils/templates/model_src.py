# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Component 6: model_src generator.

Generates test model code (ModelNew class) for verification.
"""

from .base import TemplateBase


class ModelSrcGenerator(TemplateBase):
    """Generate test model code for Ascend C operator."""

    def generate(self, project_path: str) -> str:
        """
        Generate test model code (ModelNew class).

        ModelNew must have the same interface as Model:
        - Same __init__ parameters
        - Same forward parameters

        Args:
            project_path: Absolute path to project directory (for .so loading)

        Returns:
            Complete Python model file content.

        Example output (simple case):
        ```python
        import torch
        import torch_npu
        import custom_ops_lib

        class ModelNew(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                return custom_ops_lib.add_custom(x, y)
        ```

        Example output (with init params):
        ```python
        import torch
        import torch_npu
        import custom_ops_lib

        class ModelNew(torch.nn.Module):
            def __init__(self, alpha = 1.0) -> None:
                super().__init__()
                self.alpha = alpha

            def forward(self, x):
                return custom_ops_lib.elu_custom(x, self.alpha)
        ```
        """
        inputs = self.signature.get("inputs", [])
        init_params = self.signature.get("init_params", [])

        # Split inputs: forward inputs vs model parameters (nn.Parameter)
        forward_inputs = [inp for inp in inputs if inp.get("source") != "model_param"]
        model_params = [inp for inp in inputs if inp.get("source") == "model_param"]

        # Generate forward parameters (only forward inputs, not model params)
        forward_params = ", ".join([inp["name"] for inp in forward_inputs])

        # Generate __init__ signature and body
        init_body_lines = []

        # Scalar init_params (e.g., alpha, stride)
        if init_params:
            init_param_strs = []
            for param in init_params:
                # Build parameter string with optional default
                param_str = param["name"]
                if "default" in param and param["default"] is not None:
                    default_val = param["default"]
                    if isinstance(default_val, str):
                        param_str += f' = "{default_val}"'
                    else:
                        param_str += f" = {default_val}"
                init_param_strs.append(param_str)
                init_body_lines.append(f"        self.{param['name']} = {param['name']}")
            init_signature = ", ".join(init_param_strs)
        else:
            init_signature = ""

        # Model parameters (nn.Parameter, e.g., weight for embedding)
        for mp in model_params:
            shape = mp.get("shape")
            if shape:
                shape_str = repr(shape)
                init_body_lines.append(
                    f"        self.{mp['name']} = torch.nn.Parameter(torch.randn({shape_str}))"
                )

        if not init_body_lines:
            init_body = "        pass"
        else:
            init_body = "\n".join(init_body_lines)

        # Generate custom op call args:
        #   forward inputs as-is, model params as self.xxx, init_params as self.xxx
        op_args = [inp["name"] for inp in forward_inputs]
        for mp in model_params:
            op_args.append(f"self.{mp['name']}")
        for param in init_params:
            op_args.append(f"self.{param['name']}")
        op_args_str = ", ".join(op_args)

        # Generate model_src with local .so loading to avoid global pip conflicts
        # This ensures each project uses its own compiled custom_ops_lib
        # NOTE: project_path is hardcoded at generation time to work with exec()
        return f'''import sys
import os
import glob

# Priority load project-local custom_ops_lib to avoid global conflicts
# This enables parallel compilation without .so file conflicts
# Path hardcoded at generation time (exec() doesn't have __file__)
_project_path = "{project_path}"
_build_dirs = glob.glob(os.path.join(_project_path, "CppExtension", "build", "lib.*"))
if _build_dirs:
    sys.path.insert(0, _build_dirs[0])

import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, {init_signature}) -> None:
        super().__init__()
{init_body}

    def forward(self, {forward_params}):
        return custom_ops_lib.{self.op_custom}({op_args_str})
'''

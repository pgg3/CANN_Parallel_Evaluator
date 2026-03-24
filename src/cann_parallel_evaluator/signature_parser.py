# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Operator signature parser for Python reference code.

This module extracts operator signature (inputs, outputs) from
Python reference implementations using AST parsing.

Supports MultiKernelBench reference format:
- Model class with __init__ and forward methods
- get_inputs() and get_init_inputs() functions

The actual invocation pattern is:
    model = Model(*get_init_inputs())
    result = model(*get_inputs())

Type inference strategy:
1. For inline expressions in get_inputs()/get_init_inputs(): Execute and check type (100% accurate)
2. For variable references: Use AST-based inference with priority:
   - Global variable assignment (e.g., kernel_size = 3 → int)
   - Type hints from __init__ / forward (e.g., kernel_size: int)
   - torch.* function calls (e.g., torch.randn() → float tensor)
   - Default fallback (float for unknown)
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


# PyTorch dtype to signature dtype mapping
_TORCH_DTYPE_MAP = {
    "torch.float32": "float",
    "torch.float": "float",
    "torch.float16": "float16",
    "torch.half": "float16",
    "torch.float64": "double",
    "torch.double": "double",
    "torch.bfloat16": "bfloat16",
    "torch.int64": "int64",
    "torch.long": "int64",
    "torch.int32": "int32",
    "torch.int": "int32",
    "torch.int16": "int16",
    "torch.short": "int16",
    "torch.int8": "int8",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
}


class OperatorSignatureParser:
    """
    Parse Python reference code to extract operator signature.

    Extracts:
    - Inputs from get_inputs() function
    - Init params from get_init_inputs() function
    - Parameter names from Model.forward() and Model.__init__()
    - dtype inferred from execution (for inline expressions) or AST analysis

    Type inference uses execution for 100% accuracy on complex expressions.
    """

    def __init__(self):
        """Initialize parser with execution context cache."""
        self._exec_globals: Optional[Dict[str, Any]] = None
        self._python_code: Optional[str] = None

    def parse(self, python_code: str, op_name: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Parse Python code and extract operator signature.

        Args:
            python_code: Python reference implementation
            op_name: Operator name (typically from filename, e.g., "elu", "add")
            mode: Parsing mode:
                - "auto": Auto-detect fn vs org format
                - "fn": Parse module_fn() signature directly (fn_py_reference format)
                - "org": Parse Model class + get_inputs/get_init_inputs (original format)

        Returns:
            Signature dict containing:
                - op_name: Operator name (passed in, not parsed)
                - inputs: List of forward() input info [{name, dtype, is_tensor}]
                - outputs: List of output info [{name, dtype, is_tensor}]
                - init_params: List of __init__() param info [{name, dtype, is_tensor, default}]
        """
        # Store code for execution-based inference
        self._python_code = python_code
        self._exec_globals = None  # Reset cache

        # Auto-detect mode
        if mode == "auto":
            mode = "fn" if "def module_fn(" in python_code else "org"

        if mode == "fn":
            return self._parse_fn_mode(python_code, op_name)
        else:
            return self._parse_org_mode(python_code, op_name)

    def _parse_fn_mode(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """Parse fn_py_reference format: extract signature from module_fn().

        In fn format, module_fn(x, weight, bias, stride, ...) takes ALL parameters
        explicitly. Tensors become inputs, scalars become init_params.
        """
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            return self._parse_with_regex(python_code, op_name)

        # Find module_fn definition
        module_fn_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "module_fn":
                module_fn_node = node
                break

        if module_fn_node is None:
            # Fallback to org mode
            return self._parse_org_mode(python_code, op_name)

        # Execute code to get runtime type info
        exec_globals = self._get_exec_globals()

        # Get module_fn params (with type hints and defaults)
        fn_type_hints = self._extract_func_type_hints(module_fn_node)

        # Determine which params are tensors vs scalars via execution
        inputs = []
        init_params = []

        args = module_fn_node.args
        # Count defaults offset
        num_args = len(args.args)
        num_defaults = len(args.defaults)
        first_default_idx = num_args - num_defaults

        # Try to get runtime types by instantiating Model and calling module_fn
        runtime_types = self._infer_module_fn_types(exec_globals)

        for i, arg in enumerate(args.args):
            name = arg.arg
            default_value = None
            has_default = False

            # Get default value if exists
            if i >= first_default_idx:
                default_node = args.defaults[i - first_default_idx]
                default_value = self._extract_literal_value(default_node)
                has_default = True

            # Determine type: runtime > type hints > heuristic
            is_tensor = False
            dtype = "float"
            source = "forward"
            shape = None

            if runtime_types and name in runtime_types:
                info = runtime_types[name]
                dtype, is_tensor = info[0], info[1]
                if len(info) > 2:
                    source = info[2]
                if len(info) > 3:
                    shape = info[3]
            elif name in fn_type_hints:
                dtype, is_tensor = self._type_hint_to_dtype(fn_type_hints[name])
            else:
                # Heuristic based on parameter name
                dtype, is_tensor = self._infer_type_from_name(name)

            if is_tensor:
                inp_info: Dict[str, Any] = {"name": name, "dtype": dtype, "is_tensor": True}
                if source == "model_param":
                    inp_info["source"] = "model_param"
                    if shape is not None:
                        inp_info["shape"] = shape
                inputs.append(inp_info)
            else:
                param_info: Dict[str, Any] = {
                    "name": name,
                    "dtype": dtype,
                    "is_tensor": False,
                }
                if has_default and default_value is not None:
                    param_info["default"] = default_value
                init_params.append(param_info)

        # Infer outputs: prefer runtime capture, fallback to heuristic
        if runtime_types and "__output__" in runtime_types:
            output_dtype = runtime_types["__output__"][0]
            outputs = [{"name": "output", "dtype": output_dtype, "is_tensor": True}]
        else:
            outputs = self._infer_outputs(inputs)

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": outputs,
            "init_params": init_params,
        }

    def _infer_module_fn_types(
        self, exec_globals: Dict[str, Any]
    ) -> Optional[Dict[str, tuple]]:
        """Infer module_fn parameter types by executing Model and inspecting forward args.

        Instantiates Model(*get_init_inputs()), then inspects what types
        Model.forward passes to module_fn.
        """
        if not exec_globals:
            return None

        try:
            import torch

            Model = exec_globals.get("Model")
            get_init_inputs = exec_globals.get("get_init_inputs")
            get_inputs = exec_globals.get("get_inputs")
            module_fn = exec_globals.get("module_fn")

            if not all([Model, get_init_inputs, get_inputs, module_fn]):
                return None

            # Instantiate model
            model = Model(*get_init_inputs())
            model.eval()

            # Capture what module_fn receives by replacing it with a spy
            captured_args = {}

            def _classify_val(val):
                """Classify a value into (dtype, is_tensor, source, shape)."""
                if isinstance(val, torch.nn.Parameter):
                    dtype_str = _TORCH_DTYPE_MAP.get(str(val.dtype), "float")
                    return (dtype_str, True, "model_param", list(val.shape))
                elif isinstance(val, torch.Tensor):
                    dtype_str = _TORCH_DTYPE_MAP.get(str(val.dtype), "float")
                    return (dtype_str, True)
                elif isinstance(val, bool):
                    return ("bool", False)
                elif isinstance(val, int):
                    return ("int", False)
                elif isinstance(val, float):
                    return ("float", False)
                elif isinstance(val, (list, tuple)):
                    return ("list_int", False)
                elif val is None:
                    return ("float", True)
                else:
                    return ("float", False)

            def spy_fn(*args, **kwargs):
                # Map positional args to module_fn parameter names
                import inspect
                sig = inspect.signature(module_fn)
                param_names = list(sig.parameters.keys())
                for idx, val in enumerate(args):
                    if idx < len(param_names):
                        captured_args[param_names[idx]] = _classify_val(val)
                for k, v in kwargs.items():
                    captured_args[k] = _classify_val(v)
                # Call original and capture output dtype
                result = module_fn(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    out_dtype = _TORCH_DTYPE_MAP.get(str(result.dtype), "float")
                    captured_args["__output__"] = (out_dtype, True)
                return result

            # Call forward with spy
            inputs = get_inputs()
            with torch.no_grad():
                model(*inputs, fn=spy_fn)

            return captured_args if captured_args else None

        except Exception:
            return None

    @staticmethod
    def _infer_type_from_name(name: str) -> tuple:
        """Heuristic: infer (dtype, is_tensor) from parameter name."""
        name_lower = name.lower()
        # Common tensor parameter names
        tensor_names = {
            "x", "y", "z", "input", "output", "data", "src", "dst",
            "weight", "bias", "query", "key", "value",
            "q", "k", "v", "q_proj", "k_proj", "v_proj",
            "grad", "param", "m", "v_hat", "m_hat",
            "predictions", "targets", "indices", "mask",
            "running_mean", "running_var",
            "in_proj_weight", "in_proj_bias",
            "out_proj_weight", "out_proj_bias",
        }
        if name_lower in tensor_names:
            return ("float", True)
        # Names containing 'weight' or 'bias' are likely tensors
        if "weight" in name_lower or "bias" in name_lower:
            return ("float", True)
        # Common scalar parameter names
        scalar_names = {
            "stride", "padding", "dilation", "groups", "eps", "epsilon",
            "momentum", "dim", "num_heads", "num_groups", "num_features",
            "kernel_size", "output_padding", "training",
            "normalized_shape", "embed_dim", "d_model",
            "lr", "beta1", "beta2", "step",
            "negative_slope", "alpha", "scale", "threshold",
        }
        if name_lower in scalar_names:
            return ("float", False)
        # Default: assume tensor
        return ("float", True)

    def _parse_org_mode(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """Parse original format (Model class + get_inputs/get_init_inputs)."""
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            return self._parse_with_regex(python_code, op_name)

        # 0. Collect global variables and type hints for type inference
        global_vars = self._collect_global_vars(tree)
        type_hints = self._collect_type_hints(tree)

        # 1. Parse get_inputs() to get forward inputs (uses execution for inline exprs)
        inputs = self._parse_get_inputs(tree, global_vars, type_hints.get("forward", {}))

        # 2. Parse get_init_inputs() to get __init__ params (uses execution for inline exprs)
        init_params = self._parse_get_init_inputs(tree, global_vars, type_hints.get("init", {}))

        # 3. Get parameter names from Model class (for better naming)
        model_info = self._find_model_class(tree)
        if model_info:
            # Use parameter names from forward() if available
            inputs = self._merge_names(inputs, model_info.get("forward_params", []))
            init_params = self._merge_names(init_params, model_info.get("init_params", []))

        # 4. Infer outputs (default: same dtype as first input tensor)
        outputs = self._infer_outputs(inputs)

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": outputs,
            "init_params": init_params,
        }

    def _get_exec_globals(self) -> Dict[str, Any]:
        """
        Get execution globals by running the Python code once.

        This sets up the context (imports, global variables) needed for
        evaluating inline expressions in get_inputs()/get_init_inputs().

        Returns:
            Dict of global variables after executing the code.
        """
        if self._exec_globals is not None:
            return self._exec_globals

        if self._python_code is None:
            return {}

        try:
            import torch
            exec_globals: Dict[str, Any] = {"torch": torch}
            # Execute the code to set up context
            exec(self._python_code, exec_globals)
            self._exec_globals = exec_globals
            return exec_globals
        except Exception:
            # If execution fails, return empty dict (will fall back to AST)
            self._exec_globals = {}
            return {}

    def _infer_dtype_by_execution(self, func_name: str, index: int) -> Optional[Tuple[str, bool]]:
        """
        Infer dtype by executing get_inputs() or get_init_inputs() and checking result type.

        This is the most accurate method as it uses Python's runtime type system.

        Args:
            func_name: "get_inputs" or "get_init_inputs"
            index: Index of the element in the returned list

        Returns:
            (dtype, is_tensor) tuple, or None if execution fails.
        """
        exec_globals = self._get_exec_globals()
        if not exec_globals or func_name not in exec_globals:
            return None

        try:
            import torch
            func = exec_globals[func_name]
            result_list = func()

            if not isinstance(result_list, (list, tuple)) or index >= len(result_list):
                return None

            value = result_list[index]

            # Check if tensor
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype)  # e.g., "torch.float32"
                dtype = _TORCH_DTYPE_MAP.get(dtype_str, "float")
                return (dtype, True)

            # Check scalar types
            if isinstance(value, bool):
                return ("bool", False)
            elif isinstance(value, int):
                return ("int", False)
            elif isinstance(value, float):
                return ("float", False)
            elif isinstance(value, str):
                return ("str", False)
            elif isinstance(value, (list, tuple)):
                # Check first element for list type
                if value and isinstance(value[0], int):
                    return ("list_int", False)
                elif value and isinstance(value[0], float):
                    return ("list_float", False)
                return ("list_int", False)

            return ("float", False)

        except Exception:
            return None

    def _collect_global_vars(self, tree: ast.Module) -> Dict[str, Dict[str, Any]]:
        """
        Collect global variable assignments from module level.

        Scans for:
            kernel_size = 3        → {dtype: "int", value: 3, is_tensor: False}
            x = torch.randn(...)   → {dtype: "float", value: None, is_tensor: True}

        Returns:
            Dict mapping variable name to {dtype, value, is_tensor}
        """
        global_vars: Dict[str, Dict[str, Any]] = {}

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        dtype, is_tensor = self._infer_dtype_from_expr(node.value)
                        value = self._extract_literal_value(node.value)
                        global_vars[target.id] = {
                            "dtype": dtype,
                            "is_tensor": is_tensor,
                            "value": value,
                        }

        return global_vars

    def _collect_type_hints(self, tree: ast.Module) -> Dict[str, Dict[str, str]]:
        """
        Collect type hints from Model.__init__ and Model.forward.

        Extracts:
            def __init__(self, kernel_size: int, stride: int = None):
                → {"init": {"kernel_size": "int", "stride": "int"}}

            def forward(self, x: torch.Tensor, dim: int):
                → {"forward": {"x": "Tensor", "dim": "int"}}

        Returns:
            Dict with "init" and "forward" sub-dicts mapping param name to type string
        """
        type_hints: Dict[str, Dict[str, str]] = {"init": {}, "forward": {}}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            type_hints["init"] = self._extract_func_type_hints(item)
                        elif item.name == "forward":
                            type_hints["forward"] = self._extract_func_type_hints(item)

        return type_hints

    def _extract_func_type_hints(self, func_node: ast.FunctionDef) -> Dict[str, str]:
        """Extract type hints from function arguments."""
        hints: Dict[str, str] = {}

        for arg in func_node.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation:
                hint_str = self._annotation_to_string(arg.annotation)
                hints[arg.arg.lower()] = hint_str

        return hints

    def _annotation_to_string(self, annotation: ast.AST) -> str:
        """Convert AST annotation to type string."""
        if isinstance(annotation, ast.Name):
            return annotation.id.lower()
        elif isinstance(annotation, ast.Attribute):
            # torch.Tensor → "tensor"
            if annotation.attr.lower() == "tensor":
                return "tensor"
            return annotation.attr.lower()
        elif isinstance(annotation, ast.Subscript):
            # Optional[int], List[int], etc.
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id.lower()
        return "unknown"

    def _type_hint_to_dtype(self, hint: str) -> Tuple[str, bool]:
        """
        Convert type hint string to (dtype, is_tensor).

        Args:
            hint: Type hint string like "int", "float", "tensor", "bool"

        Returns:
            (dtype, is_tensor) tuple
        """
        hint_lower = hint.lower()

        # Tensor types
        if hint_lower in ("tensor", "torch.tensor"):
            return ("float", True)

        # Scalar types
        type_map = {
            "int": ("int", False),
            "float": ("float", False),
            "bool": ("bool", False),
            "str": ("str", False),
            "list": ("list_int", False),
            "tuple": ("list_int", False),
            "optional": ("float", False),  # Optional[X] - need deeper parsing
        }

        return type_map.get(hint_lower, ("float", False))

    def _parse_get_inputs(
        self,
        tree: ast.AST,
        global_vars: Dict[str, Dict[str, Any]],
        forward_hints: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Parse get_inputs() function to extract forward inputs.

        Args:
            tree: AST tree
            global_vars: Global variable info from _collect_global_vars
            forward_hints: Type hints from forward() method

        Returns:
            List of {name, dtype, is_tensor} dicts.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
                return self._parse_input_function(node, global_vars, forward_hints)
        return []

    def _parse_get_init_inputs(
        self,
        tree: ast.AST,
        global_vars: Dict[str, Dict[str, Any]],
        init_hints: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Parse get_init_inputs() function to extract __init__ params.

        Args:
            tree: AST tree
            global_vars: Global variable info from _collect_global_vars
            init_hints: Type hints from __init__() method

        Returns:
            List of {name, dtype, is_tensor, default?} dicts.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
                return self._parse_init_input_function(node, global_vars, init_hints)
        return []

    def _parse_input_function(
        self,
        func_node: ast.FunctionDef,
        global_vars: Dict[str, Dict[str, Any]],
        type_hints: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Parse a get_inputs() function body.

        Type inference strategy:
        1. Try execution-based inference first (100% accurate)
        2. Fall back to AST-based inference if execution fails:
           - Local variable assignment in function body
           - Global variable assignment
           - Type hints from forward()
           - Default (float tensor)
        """
        # Build variable -> dtype mapping from local assignments (for AST fallback)
        local_vars: Dict[str, Dict[str, Any]] = {}

        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        dtype, is_tensor = self._infer_dtype_from_expr(stmt.value)
                        local_vars[target.id] = {
                            "dtype": dtype,
                            "is_tensor": is_tensor,
                        }

        # Find return statement and extract variable order
        return_stmt = self._find_return_stmt(func_node)
        if return_stmt is None:
            return []

        inputs = []
        if isinstance(return_stmt.value, ast.List):
            # return [Q, K, V] or return [x, dim]
            for idx, elt in enumerate(return_stmt.value.elts):
                # Get variable name (if it's a Name node)
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    var_name_lower = var_name.lower()
                else:
                    var_name = None
                    var_name_lower = f"x{len(inputs)}"

                # Priority 0: Execution-based inference (100% accurate)
                exec_result = self._infer_dtype_by_execution("get_inputs", idx)
                if exec_result is not None:
                    dtype, is_tensor = exec_result
                    inputs.append({
                        "name": var_name_lower,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    })
                    continue

                # Fallback to AST-based inference
                if isinstance(elt, ast.Name):
                    # Priority 1: Local variable
                    if var_name in local_vars:
                        info = local_vars[var_name]
                        inputs.append({
                            "name": var_name_lower,
                            "dtype": info["dtype"],
                            "is_tensor": info["is_tensor"],
                        })
                    # Priority 2: Global variable
                    elif var_name in global_vars:
                        info = global_vars[var_name]
                        inputs.append({
                            "name": var_name_lower,
                            "dtype": info["dtype"],
                            "is_tensor": info["is_tensor"],
                        })
                    # Priority 3: Type hints (need to match by position later via _merge_names)
                    # For now, use default
                    else:
                        inputs.append({
                            "name": var_name_lower,
                            "dtype": "float",
                            "is_tensor": True,  # Assume tensor by default for forward inputs
                        })
                else:
                    # Inline expression - use AST inference
                    dtype, is_tensor = self._infer_dtype_from_expr(elt)
                    inputs.append({
                        "name": var_name_lower,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    })

        # Apply type hints by position (after _merge_names updates names)
        # This is handled in a second pass if needed

        return inputs

    def _parse_init_input_function(
        self,
        func_node: ast.FunctionDef,
        global_vars: Dict[str, Dict[str, Any]],
        type_hints: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Parse get_init_inputs() function body.

        Type inference strategy:
        1. Try execution-based inference first (100% accurate for dtype/is_tensor)
        2. Fall back to AST-based inference if execution fails:
           - Local variable assignment in function body
           - Global variable assignment
           - Type hints from __init__()
           - Default (float scalar)

        Note: Default values are always extracted from AST (not execution) for accuracy.
        """
        # Build variable -> (dtype, value) mapping from local assignments (for defaults)
        local_vars: Dict[str, Dict[str, Any]] = {}

        for stmt in func_node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        dtype, is_tensor = self._infer_dtype_from_expr(stmt.value)
                        default = self._extract_literal_value(stmt.value)
                        local_vars[target.id] = {
                            "dtype": dtype,
                            "is_tensor": is_tensor,
                            "default": default,
                        }

        return_stmt = self._find_return_stmt(func_node)
        if return_stmt is None:
            return []

        params = []

        if isinstance(return_stmt.value, ast.List):
            # return [alpha, beta] or return []
            for idx, elt in enumerate(return_stmt.value.elts):
                # Get variable name (if it's a Name node)
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    var_name_lower = var_name.lower()
                else:
                    var_name = None
                    var_name_lower = f"param{len(params)}"

                # Priority 0: Execution-based inference (100% accurate for dtype/is_tensor)
                exec_result = self._infer_dtype_by_execution("get_init_inputs", idx)
                if exec_result is not None:
                    dtype, is_tensor = exec_result
                    param = {
                        "name": var_name_lower,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    }
                    # Get default value from AST (execution result may differ due to randomness)
                    if var_name and var_name in local_vars:
                        if local_vars[var_name].get("default") is not None:
                            param["default"] = local_vars[var_name]["default"]
                    elif var_name and var_name in global_vars:
                        if global_vars[var_name].get("value") is not None:
                            param["default"] = global_vars[var_name]["value"]
                    elif not isinstance(elt, ast.Name):
                        default = self._extract_literal_value(elt)
                        if default is not None:
                            param["default"] = default
                    params.append(param)
                    continue

                # Fallback to AST-based inference
                if isinstance(elt, ast.Name):
                    # Priority 1: Local variable
                    if var_name in local_vars:
                        info = local_vars[var_name]
                        param = {
                            "name": var_name_lower,
                            "dtype": info["dtype"],
                            "is_tensor": info["is_tensor"],
                        }
                        if info.get("default") is not None:
                            param["default"] = info["default"]
                        params.append(param)
                    # Priority 2: Global variable
                    elif var_name in global_vars:
                        info = global_vars[var_name]
                        param = {
                            "name": var_name_lower,
                            "dtype": info["dtype"],
                            "is_tensor": info["is_tensor"],
                        }
                        if info.get("value") is not None:
                            param["default"] = info["value"]
                        params.append(param)
                    # Priority 3: Type hints
                    elif var_name_lower in type_hints:
                        hint = type_hints[var_name_lower]
                        dtype, is_tensor = self._type_hint_to_dtype(hint)
                        params.append({
                            "name": var_name_lower,
                            "dtype": dtype,
                            "is_tensor": is_tensor,
                        })
                    # Priority 4: Default (scalar)
                    else:
                        params.append({
                            "name": var_name_lower,
                            "dtype": "float",
                            "is_tensor": False,
                        })
                else:
                    # Inline literal
                    dtype, is_tensor = self._infer_dtype_from_expr(elt)
                    default = self._extract_literal_value(elt)
                    param = {
                        "name": var_name_lower,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    }
                    if default is not None:
                        param["default"] = default
                    params.append(param)

        elif isinstance(return_stmt.value, ast.Dict):
            # return {"alpha": 1.0, "beta": 2.0}
            for key, value in zip(return_stmt.value.keys, return_stmt.value.values):
                if isinstance(key, ast.Constant):
                    name = str(key.value)
                    dtype, is_tensor = self._infer_dtype_from_expr(value)
                    default = self._extract_literal_value(value)
                    param = {
                        "name": name,
                        "dtype": dtype,
                        "is_tensor": is_tensor,
                    }
                    if default is not None:
                        param["default"] = default
                    params.append(param)

        return params

    def _find_return_stmt(self, func_node: ast.FunctionDef) -> Optional[ast.Return]:
        """Find the first return statement in a function."""
        for stmt in func_node.body:
            if isinstance(stmt, ast.Return):
                return stmt
        return None

    def _contains_torch_call(self, expr: ast.AST) -> bool:
        """
        Check if expression tree contains any torch.* call.

        This is a fallback check for complex expressions like:
            torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1

        Uses AST visitor to traverse the entire expression tree.
        """
        class TorchCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.has_torch = False

            def visit_Attribute(self, node: ast.Attribute):
                # Check for torch.xxx
                if isinstance(node.value, ast.Name) and node.value.id == "torch":
                    self.has_torch = True
                self.generic_visit(node)

        visitor = TorchCallVisitor()
        visitor.visit(expr)
        return visitor.has_torch

    def _infer_dtype_from_expr(self, expr: ast.AST) -> Tuple[str, bool]:
        """
        Infer dtype and is_tensor from an expression.

        Supports:
            - torch.randn(...) → (float, True)
            - torch.randn(..., dtype=torch.float16) → (float16, True)
            - torch.zeros(..., dtype=torch.int32) → (int32, True)
            - torch.randint(...) → (int64, True)
            - 1.0 → (float, False)
            - 1 → (int, False)
            - True → (bool, False)

        For complex expressions (method chains, binary ops), uses fallback check:
        if expression contains any torch.* call, result is assumed to be tensor.

        Returns:
            (dtype: str, is_tensor: bool)
        """
        if isinstance(expr, ast.Call):
            # Check for dtype keyword argument
            for kw in expr.keywords:
                if kw.arg == "dtype":
                    dtype = self._extract_torch_dtype(kw.value)
                    return (dtype, True)

            # Infer from function name
            func_name = self._get_call_name(expr)

            # Tensor creation functions
            tensor_funcs_float = {
                "torch.randn", "torch.rand", "torch.zeros", "torch.ones",
                "torch.empty", "torch.full", "torch.randn_like", "torch.zeros_like",
            }
            tensor_funcs_int = {"torch.randint", "torch.arange"}

            if func_name in tensor_funcs_float:
                return ("float", True)
            elif func_name in tensor_funcs_int:
                return ("int64", True)
            elif "torch" in func_name or "tensor" in func_name.lower():
                return ("float", True)

            # Check for method chain: tensor.float(), tensor.view(), etc.
            # If the call's receiver (func.value) contains torch call, it's still a tensor
            if self._contains_torch_call(expr):
                # Method call on tensor preserves tensor type
                # Try to infer dtype from method name
                if isinstance(expr.func, ast.Attribute):
                    method_name = expr.func.attr
                    method_dtype_map = {
                        "float": "float", "float32": "float",
                        "half": "float16", "float16": "float16",
                        "double": "double", "float64": "double",
                        "int": "int64", "long": "int64", "int64": "int64",
                        "int32": "int32", "short": "int16", "int16": "int16",
                        "bool": "bool",
                    }
                    if method_name in method_dtype_map:
                        return (method_dtype_map[method_name], True)
                return ("float", True)

            # Non-tensor function call
            return ("float", False)

        elif isinstance(expr, ast.Constant):
            # Literal values
            value = expr.value
            if isinstance(value, bool):
                return ("bool", False)
            elif isinstance(value, int):
                return ("int", False)
            elif isinstance(value, float):
                return ("float", False)
            elif isinstance(value, str):
                return ("str", False)
            return ("float", False)

        elif isinstance(expr, ast.List):
            # List literal, check first element
            if expr.elts:
                inner_dtype, _ = self._infer_dtype_from_expr(expr.elts[0])
                return (f"list_{inner_dtype}", False)
            return ("list_int", False)

        elif isinstance(expr, ast.Tuple):
            # Tuple literal, treat same as list
            if expr.elts:
                inner_dtype, _ = self._infer_dtype_from_expr(expr.elts[0])
                return (f"list_{inner_dtype}", False)
            return ("list_int", False)

        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            # Negative number: -1.0
            inner_dtype, is_tensor = self._infer_dtype_from_expr(expr.operand)
            return (inner_dtype, is_tensor)

        elif isinstance(expr, ast.BinOp):
            # Binary operation: torch.rand(...) * 5, x + y, etc.
            # If either operand is a tensor, result is a tensor
            left_dtype, left_is_tensor = self._infer_dtype_from_expr(expr.left)
            right_dtype, right_is_tensor = self._infer_dtype_from_expr(expr.right)

            is_tensor = left_is_tensor or right_is_tensor
            # Use dtype from tensor operand, or left operand if both are scalars
            dtype = left_dtype if left_is_tensor else (right_dtype if right_is_tensor else left_dtype)

            return (dtype, is_tensor)

        # Fallback: if expression contains torch.* call, assume tensor
        # This handles complex cases like: torch.randint(...).float() * 2 - 1
        if self._contains_torch_call(expr):
            return ("float", True)

        return ("float", False)

    def _get_call_name(self, call_node: ast.Call) -> str:
        """Get the full name of a function call (e.g., 'torch.randn')."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            node = func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))
        return ""

    def _extract_torch_dtype(self, node: ast.AST) -> str:
        """
        Extract dtype from torch.float16, torch.int32, etc.
        """
        dtype_map = {
            "float16": "float16",
            "float32": "float",
            "float64": "double",
            "half": "float16",
            "float": "float",
            "double": "double",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "int": "int64",
            "long": "int64",
            "bool": "bool",
            "bfloat16": "bfloat16",
        }

        if isinstance(node, ast.Attribute):
            return dtype_map.get(node.attr, "float")
        elif isinstance(node, ast.Name):
            return dtype_map.get(node.id, "float")

        return "float"

    def _extract_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._extract_literal_value(node.operand)
            if isinstance(inner, (int, float)):
                return -inner
        elif isinstance(node, ast.List):
            return [self._extract_literal_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return [self._extract_literal_value(elt) for elt in node.elts]
        return None

    def _find_model_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """
        Find Model class and extract parameter names from __init__ and forward.

        Returns dict with forward_params and init_params (just names).
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Model":
                forward_params = []
                init_params = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "forward":
                            for arg in item.args.args:
                                if arg.arg != "self":
                                    forward_params.append(arg.arg.lower())
                        elif item.name == "__init__":
                            for arg in item.args.args:
                                if arg.arg != "self":
                                    init_params.append(arg.arg.lower())

                return {
                    "forward_params": forward_params,
                    "init_params": init_params,
                }

        return None

    def _merge_names(
        self, parsed_inputs: List[Dict[str, Any]], param_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Merge parameter names from Model class into parsed inputs.

        If param_names are available, use them instead of auto-generated names.
        """
        if not param_names:
            return parsed_inputs

        result = []
        for i, inp in enumerate(parsed_inputs):
            new_inp = inp.copy()
            if i < len(param_names):
                new_inp["name"] = param_names[i]
            result.append(new_inp)

        return result

    def _infer_outputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Infer output info.

        Default: single tensor output with same dtype as first non-bool tensor input.
        Falls back to first tensor input if all are bool.
        """
        dtype = "float"
        # Prefer first non-bool tensor (handles where/masked_fill with bool condition)
        for inp in inputs:
            if inp.get("is_tensor", False) and inp.get("dtype") != "bool":
                dtype = inp.get("dtype", "float")
                break
        else:
            # Fallback: first tensor input regardless of dtype
            for inp in inputs:
                if inp.get("is_tensor", False):
                    dtype = inp.get("dtype", "float")
                    break

        return [{"name": "output", "dtype": dtype, "is_tensor": True}]

    def _parse_with_regex(self, python_code: str, op_name: str) -> Dict[str, Any]:
        """
        Fallback regex-based parsing for malformed Python code.
        """
        inputs = []

        # Try to find get_inputs function
        get_inputs_match = re.search(
            r"def\s+get_inputs\s*\(\s*\):\s*\n(.*?)(?=\ndef|\Z)",
            python_code,
            re.DOTALL,
        )

        if get_inputs_match:
            body = get_inputs_match.group(1)
            # Find torch.randn calls
            randn_matches = re.findall(r"(\w+)\s*=\s*torch\.\w+\(", body)
            for var_name in randn_matches:
                inputs.append({
                    "name": var_name.lower(),
                    "dtype": "float",
                    "is_tensor": True,
                })

        if not inputs:
            # Default inputs
            inputs = [
                {"name": "x", "dtype": "float", "is_tensor": True},
            ]

        return {
            "op_name": op_name,
            "inputs": inputs,
            "outputs": [{"name": "output", "dtype": "float", "is_tensor": True}],
            "init_params": [],
        }

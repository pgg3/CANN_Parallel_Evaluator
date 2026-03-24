# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from .base import TemplateBase


class KernelSrcGenerator(TemplateBase):
    """Generate kernel source code with fixed entry function signature."""

    def generate(
        self,
        kernel_impl: str,
        kernel_entry_body: str,
    ) -> str:
        """Generate kernel source code.

        Args:
            kernel_impl: Kernel class implementation and helper code.
                May contain #include directives at the top.
            kernel_entry_body: Entry function body (after GET_TILING_DATA).

        Returns:
            Complete kernel source code.
        """
        # Generate GM_ADDR parameters from signature
        gm_params = self._generate_gm_params()

        return f'''#include "kernel_operator.h"

{kernel_impl}

extern "C" __global__ __aicore__ void {self.op_custom}({gm_params}, GM_ADDR workspace, GM_ADDR tiling) {{
    GET_TILING_DATA(tilingData, tiling);
{kernel_entry_body}
}}
'''

    def _generate_gm_params(self) -> str:
        """Generate GM_ADDR parameters from signature.

        Order: inputs (tensors only) -> outputs (tensors only)
        Non-tensor inputs are passed via tiling data, not as GM_ADDR.
        """
        params = []

        # Input tensors
        for inp in self.signature.get("inputs", []):
            if inp.get("is_tensor", True):
                params.append(f"GM_ADDR {inp['name']}")

        # Init param tensors (rare, but possible)
        for param in self.signature.get("init_params", []):
            if param.get("is_tensor", False):
                params.append(f"GM_ADDR {param['name']}")

        # Output tensors
        for out in self.signature.get("outputs", []):
            if out.get("is_tensor", True):
                params.append(f"GM_ADDR {out['name']}")

        return ", ".join(params)

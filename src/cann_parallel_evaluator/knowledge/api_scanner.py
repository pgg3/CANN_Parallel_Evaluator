# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANN SDK header scanner for API discovery.

Ported from cann-benchmark's knowledge_base.py — extracts API function
signatures from CANN SDK interface headers and provides search capability.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Map header file names to API categories
HEADER_TO_CATEGORY = {
    # Vector operations
    "kernel_operator_vec_binary_intf.h": "vec_binary",
    "kernel_operator_vec_binary_scalar_intf.h": "vec_binary_scalar",
    "kernel_operator_vec_unary_intf.h": "vec_unary",
    "kernel_operator_vec_reduce_intf.h": "vec_reduce",
    "kernel_operator_vec_cmpsel_intf.h": "vec_compare",
    "kernel_operator_vec_duplicate_intf.h": "vec_duplicate",
    "kernel_operator_vec_gather_intf.h": "vec_gather",
    "kernel_operator_vec_scatter_intf.h": "vec_scatter",
    "kernel_operator_vec_transpose_intf.h": "vec_transpose",
    "kernel_operator_vec_vconv_intf.h": "vec_convert",
    "kernel_operator_vec_vpadding_intf.h": "vec_padding",
    "kernel_operator_vec_brcb_intf.h": "vec_broadcast",
    "kernel_operator_vec_mulcast_intf.h": "vec_mulcast",
    "kernel_operator_vec_ternary_scalar_intf.h": "vec_ternary",
    "kernel_operator_vec_createvecindex_intf.h": "vec_index",
    "kernel_operator_vec_gather_mask_intf.h": "vec_gather",
    "kernel_operator_vec_bilinearinterpalation_intf.h": "vec_interpolation",
    # Cube/Matrix operations
    "kernel_operator_mm_intf.h": "cube_matmul",
    "kernel_operator_gemm_intf.h": "cube_gemm",
    "kernel_operator_conv2d_intf.h": "cube_conv",
    # Data movement
    "kernel_operator_data_copy_intf.h": "data_copy",
    "kernel_operator_fixpipe_intf.h": "data_fixpipe",
    # Scalar operations
    "kernel_operator_scalar_intf.h": "scalar",
    # Synchronization
    "kernel_operator_determine_compute_sync_intf.h": "sync",
    "kernel_operator_set_atomic_intf.h": "atomic",
    # System
    "kernel_operator_sys_var_intf.h": "system",
    "kernel_operator_common_intf.h": "common",
    # Pipe and Buffer types
    "kernel_tpipe.h": "pipe_buffer",
    # Other
    "kernel_operator_dump_tensor_intf.h": "debug",
    "kernel_operator_list_tensor_intf.h": "tensor_list",
    "kernel_operator_proposal_intf.h": "proposal",
}

# Higher-level category grouping for display
CATEGORY_GROUPS = {
    "Vector Compute": [
        "vec_binary", "vec_unary", "vec_reduce", "vec_compare",
        "vec_ternary", "vec_binary_scalar",
    ],
    "Vector Data": [
        "vec_duplicate", "vec_gather", "vec_scatter", "vec_transpose",
        "vec_convert", "vec_padding", "vec_broadcast", "vec_mulcast",
        "vec_index", "vec_interpolation",
    ],
    "Cube/Matrix": ["cube_matmul", "cube_gemm", "cube_conv"],
    "Data Movement": ["data_copy", "data_fixpipe"],
    "Scalar": ["scalar"],
    "Sync & Atomic": ["sync", "atomic"],
    "System & Debug": ["system", "common", "debug", "tensor_list", "proposal"],
    "Pipe & Buffer": ["pipe_buffer"],
}

# Types (classes/templates) that should be indexed from kernel_tpipe.h
INDEXED_TYPES = {
    "TPipe": "Tensor pipeline for managing buffers",
    "TQue": "Queue buffer for pipelined data transfer",
    "TBuf": "Scratch buffer for temporary data",
    "GlobalTensor": "Tensor in global memory",
    "LocalTensor": "Tensor in local memory (UB)",
}

# Fallback APIs when CANN headers are not available
FALLBACK_APIS = {
    "vec_binary": ["Add", "Sub", "Mul", "Div", "Max", "Min", "And", "Or"],
    "vec_binary_scalar": ["Adds", "Muls", "Maxs", "Mins"],
    "vec_unary": ["Abs", "Exp", "Ln", "Sqrt", "Rsqrt", "Relu", "Not", "Reciprocal"],
    "vec_reduce": [
        "ReduceMax", "ReduceMin", "ReduceSum",
        "BlockReduceMax", "BlockReduceMin", "BlockReduceSum",
        "WholeReduceMax", "WholeReduceMin", "WholeReduceSum",
    ],
    "vec_compare": ["Compare", "CompareScalar", "Select"],
    "vec_duplicate": ["Duplicate"],
    "vec_convert": ["Cast"],
    "cube_matmul": ["Mmad", "LoadData"],
    "cube_gemm": ["Gemm"],
    "data_copy": ["DataCopy", "DataCopyPad", "DataCopyExtParams"],
    "scalar": ["ScalarAdd", "ScalarSub", "ScalarMul", "ScalarDiv"],
    "sync": ["SetFlag", "WaitFlag", "PipeBarrier"],
    "system": ["GetBlockIdx", "GetBlockNum", "GetBlockDim"],
}


def default_cann_path() -> str:
    """Get default CANN installation path from env or standard locations."""
    cann_path = os.environ.get("ASCEND_HOME_PATH")
    if cann_path and Path(cann_path).exists():
        return cann_path

    candidates = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/8.1.RC1",
        "/opt/Ascend/ascend-toolkit/latest",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    return "/usr/local/Ascend/ascend-toolkit/latest"


def _find_interface_dir(cann_path: str) -> Optional[Path]:
    """Find the CANN SDK interface headers directory."""
    cann = Path(cann_path)
    candidates = [
        cann / "compiler" / "ascendc" / "include" / "basic_api" / "interface",
        cann / "aarch64-linux" / "ascendc" / "include" / "basic_api" / "interface",
        cann / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "interface",
        cann / "include" / "ascendc" / "basic_api" / "interface",
    ]
    for d in candidates:
        if d.exists():
            return d
    return None


def _extract_apis_from_header(header_path: Path) -> List[Tuple[str, str]]:
    """Extract API function names and descriptions from a header file.

    Returns list of (api_name, description) tuples.
    """
    apis = []
    seen: Set[str] = set()

    try:
        content = header_path.read_text(errors="ignore")
    except Exception:
        return apis

    # Pattern 1: Comment block followed by function declaration
    pattern_with_comment = r"""
        /\*[^*]*\*+(?:[^/*][^*]*\*+)*/           # Comment block /* ... */
        \s*                                      # Whitespace
        (?:template\s*<[^>]+>\s*)?               # Optional template
        __aicore__\s+inline\s+                   # __aicore__ inline
        (?:__inout_pipe__\([^)]+\)\s+)?          # Optional pipe annotation
        (?:__out_pipe__\([^)]+\)\s+)?            # Optional out pipe annotation
        (?:void|[A-Za-z_][A-Za-z0-9_:<>]*)\s+    # Return type
        ([A-Z][A-Za-z0-9]+)                      # Function name (PascalCase)
        \s*\(                                    # Opening paren
    """

    for match in re.finditer(pattern_with_comment, content, re.VERBOSE):
        api_name = match.group(1)
        if api_name not in seen:
            seen.add(api_name)
            comment_block = match.group(0)
            brief_match = re.search(r"@brief\s+(.+?)(?:\n|$)", comment_block)
            desc = brief_match.group(1).strip() if brief_match else ""
            apis.append((api_name, desc))

    # Pattern 2: Simple __aicore__ inline declarations without comment blocks
    pattern_simple = r"""
        ^[ \t]*                                  # Start of line
        (?:template\s*<[^>]+>\s*)?               # Optional template
        __aicore__\s+inline\s+                   # __aicore__ inline
        (?:__inout_pipe__\([^)]+\)\s+)?          # Optional pipe annotation
        (?:__in_pipe__\([^)]+\)\s+)?             # Optional in pipe annotation
        (?:__out_pipe__\([^)]+\)\s+)?            # Optional out pipe annotation
        (?:void|[A-Za-z_][A-Za-z0-9_:<>]*)\s+    # Return type
        ([A-Z][A-Za-z0-9]+)                      # Function name (PascalCase)
        \s*\(                                    # Opening paren
    """

    for match in re.finditer(pattern_simple, content, re.VERBOSE | re.MULTILINE):
        api_name = match.group(1)
        if api_name not in seen:
            seen.add(api_name)
            apis.append((api_name, ""))

    return apis


def scan_headers(cann_path: str) -> Dict[str, Any]:
    """Scan CANN SDK headers and build an API index.

    Returns:
        {
            "apis": {api_name: {"category", "description", "header"}},
            "api_categories": {category: [api_names]},
        }
    """
    index: Dict[str, Any] = {"apis": {}, "api_categories": {}}

    interface_dir = _find_interface_dir(cann_path)
    if not interface_dir:
        # Use fallback APIs
        for category, api_list in FALLBACK_APIS.items():
            index["api_categories"][category] = []
            for api_name in api_list:
                index["apis"][api_name] = {
                    "category": category,
                    "description": "",
                    "header": "fallback",
                }
                index["api_categories"][category].append(api_name)
        return index

    # Scan each header file
    for header_file in interface_dir.glob("kernel_operator_*.h"):
        if header_file.name.startswith("kernel_struct_"):
            continue

        category = HEADER_TO_CATEGORY.get(header_file.name, "other")
        apis = _extract_apis_from_header(header_file)

        for api_name, description in apis:
            if api_name not in index["apis"]:
                index["apis"][api_name] = {
                    "category": category,
                    "description": description,
                    "header": header_file.name,
                }
                if category not in index["api_categories"]:
                    index["api_categories"][category] = []
                index["api_categories"][category].append(api_name)

    # Add pipe/buffer types from kernel_tpipe.h
    category = "pipe_buffer"
    if category not in index["api_categories"]:
        index["api_categories"][category] = []
    for type_name, description in INDEXED_TYPES.items():
        if type_name not in index["apis"]:
            index["apis"][type_name] = {
                "category": category,
                "description": description,
                "header": "kernel_tpipe.h",
                "type": "class",
            }
            index["api_categories"][category].append(type_name)

    # Add InitBuffer
    if "InitBuffer" not in index["apis"]:
        index["apis"]["InitBuffer"] = {
            "category": category,
            "description": "Initialize buffer. TQue: InitBuffer(que, num, len); TBuf: InitBuffer(buf, len)",
            "header": "kernel_tpipe.h",
            "type": "method",
        }
        index["api_categories"][category].append("InitBuffer")

    return index


def search(name: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """Search for an API by name in the index.

    Strategy: exact match → case-insensitive → substring candidates.

    Returns:
        {"status": "found"|"not_found"|"ambiguous", "api_info": dict|None, "candidates": list}
    """
    apis = index.get("apis", {})

    # Exact match
    if name in apis:
        return {"status": "found", "api_info": {"name": name, **apis[name]}, "candidates": []}

    # Case-insensitive match
    name_lower = name.lower()
    for api_name, api_data in apis.items():
        if api_name.lower() == name_lower:
            return {"status": "found", "api_info": {"name": api_name, **api_data}, "candidates": []}

    # Substring candidates
    candidates = [
        api for api in apis
        if name_lower in api.lower() or api.lower() in name_lower
    ]
    if candidates:
        return {"status": "ambiguous", "api_info": None, "candidates": candidates[:5]}

    return {"status": "not_found", "api_info": None, "candidates": []}


def list_apis_grouped(index: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return all APIs grouped by high-level category."""
    api_categories = index.get("api_categories", {})
    result = {}
    for group_name, subcategories in CATEGORY_GROUPS.items():
        apis_in_group = []
        for subcat in subcategories:
            apis_in_group.extend(api_categories.get(subcat, []))
        if apis_in_group:
            result[group_name] = sorted(apis_in_group)
    return result

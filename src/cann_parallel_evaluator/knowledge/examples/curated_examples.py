# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Curated complete examples for AscendC operator patterns.

Content is stored in per-example .md files alongside this file.
Each example includes all 6 components (KERNEL_IMPL, KERNEL_ENTRY_BODY,
TILING_FIELDS, TILING_FUNC_BODY, INFER_SHAPE_BODY, OUTPUT_ALLOC_CODE).
"""

from pathlib import Path
from typing import Optional

_DIR = Path(__file__).parent

# Pattern → .md filename mapping
_EXAMPLE_FILES = {
    "elementwise_binary": "add.md",
    "elementwise_unary": "relu.md",
    "elementwise": "add.md",  # default elementwise → binary (Add)
    "reduction": "reduce_sum.md",
    "softmax": "softmax.md",  # dedicated: per-row max→exp→sum→div
    "broadcast": "add.md",  # broadcast reuses add example
    "pooling": "pooling.md",
    "matmul": "matmul.md",
    "convolution": "convolution.md",  # im2col + Matmul decomposition
    "attention": "attention.md",  # Cube+Vector mixed: QK^T→softmax→scores×V
    "normalization": "layer_norm.md",
    "index": "gather.md",
    "resize": "resize.md",  # dedicated: nearest neighbor coordinate mapping
}

# Cache loaded content
_cache: dict[str, str] = {}


def _load(filename: str) -> str:
    if filename not in _cache:
        _cache[filename] = (_DIR / filename).read_text()
    return _cache[filename]


def get_example(pattern: str) -> Optional[str]:
    """Get a curated example for the given pattern.

    Returns None if no example is available for the pattern.
    """
    filename = _EXAMPLE_FILES.get(pattern)
    if filename is None:
        return None
    return _load(filename)

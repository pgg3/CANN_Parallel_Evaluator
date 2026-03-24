# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Level 1 Primers: Compute-pattern-specific guides.

Content is stored in per-pattern .md files alongside this file.
"""

from pathlib import Path

_DIR = Path(__file__).parent

# Pattern name → .md filename mapping
_PATTERN_FILES = {
    "elementwise": "elementwise.md",
    "reduction": "reduction.md",
    "softmax": "softmax.md",
    "broadcast": "broadcast.md",
    "pooling": "pooling.md",
    "matmul": "matmul.md",
    "convolution": "convolution.md",
    "attention": "attention.md",
    "normalization": "normalization.md",
    "index": "index.md",
    "resize": "resize.md",
    "other": "other.md",
}

# Cache loaded content
_cache: dict[str, str] = {}


def _load(pattern: str) -> str:
    if pattern not in _cache:
        filename = _PATTERN_FILES.get(pattern, "other.md")
        _cache[pattern] = (_DIR / filename).read_text()
    return _cache[pattern]


def get_pattern_primer(pattern: str) -> str:
    """Get the Level 1 primer for a compute pattern.

    Falls back to 'other' for unknown patterns.
    """
    if pattern in _PATTERN_FILES:
        return _load(pattern)
    return _load("other")

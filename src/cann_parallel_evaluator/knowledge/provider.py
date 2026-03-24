# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""CANN Knowledge Provider — fine-grained domain knowledge for AscendC.

All domain knowledge is stored as .md files in subdirectories.
The provider exposes fine-grained methods so callers (task, sampler)
can assemble knowledge flexibly for different stages.

Knowledge pieces:
- Level 0: Programming model overview (primers/level0_programming_model.md)
- Level 1: Pattern-specific guides (primers/{pattern}.md)
- Constraints: Critical error warnings (constraints/critical_*.md)
- Tiling: UB capacity, edge cases (tiling/*.md)
- API: Quick reference, advanced reference (api/*.md)
- Examples: Curated complete operator examples (examples/*.md)
- Level 2: Real-time API search from CANN SDK headers (api_scanner)
"""

import configparser
from pathlib import Path
from typing import Any, Dict, Optional

from . import api_scanner
from .examples import get_example as _get_example
from .primers import LEVEL0_PROGRAMMING_MODEL, get_pattern_primer

_KNOWLEDGE_DIR = Path(__file__).parent

# Cache for .md file content
_file_cache: dict[str, str] = {}

# Cache for parsed platform specs
_platform_cache: dict[str, dict] = {}

# Default CANN toolkit path for platform configs
_DEFAULT_PLATFORM_DIR = Path(
    "/usr/local/Ascend/ascend-toolkit/latest/compiler/data/platform_config"
)


def _read_platform_spec(npu_type: str) -> dict:
    """Read hardware specs from Ascend platform config INI.

    Returns dict with keys: ai_core_cnt, ub_size, vir_type_list, vector_core_cnt.
    Falls back to safe defaults if the INI file is not found.
    """
    if npu_type in _platform_cache:
        return _platform_cache[npu_type]

    ini_path = _DEFAULT_PLATFORM_DIR / f"{npu_type}.ini"
    spec: dict = {}

    if ini_path.exists():
        # The INI has duplicate section names; use RawConfigParser with strict=False
        parser = configparser.RawConfigParser(strict=False)
        parser.read(str(ini_path))

        if parser.has_section("SoCInfo"):
            spec["ai_core_cnt"] = parser.getint("SoCInfo", "ai_core_cnt", fallback=24)
            spec["vector_core_cnt"] = parser.getint(
                "SoCInfo", "vector_core_cnt", fallback=48
            )
            vir_list = parser.get("SoCInfo", "vir_type_list", fallback="2,3,6,12,24")
            spec["vir_type_list"] = [int(v.strip()) for v in vir_list.split(",")]
        if parser.has_section("AICoreSpec"):
            spec["ub_size"] = parser.getint("AICoreSpec", "ub_size", fallback=196608)

    # Fill defaults for any missing values
    spec.setdefault("ai_core_cnt", 24)
    spec.setdefault("vector_core_cnt", 48)
    spec.setdefault("vir_type_list", [2, 3, 6, 12, 24])
    spec.setdefault("ub_size", 196608)

    _platform_cache[npu_type] = spec
    return spec


def _format_hardware_constraints(spec: dict, npu_type: str) -> str:
    """Generate the hardware constraints section for tiling knowledge."""
    ub_kb = spec["ub_size"] // 1024
    usable_kb = ub_kb - 64  # reserve ~64 KB for TPipe/stack/system
    usable_bytes = usable_kb * 1024
    max_block_dim = max(spec["vir_type_list"])
    vir_str = ", ".join(str(v) for v in sorted(spec["vir_type_list"]))

    # Pre-compute max tileLength table for common queue/buffer/dtype combos
    rows = []
    for nq, bn, dtype, dsize in [
        (2, 2, "float", 4),
        (3, 2, "float", 4),
        (4, 2, "float", 4),
        (2, 2, "half", 2),
    ]:
        max_tile = usable_bytes // (nq * bn * dsize)
        max_tile = max_tile // 8 * 8  # align
        rows.append(f"| {nq}      | {bn}         | {dtype:6s} | ~{max_tile:<13,d} |")
    table = "\n".join(rows)

    return f"""### Hardware Constraints ({npu_type})
- **AI Cores**: {spec['ai_core_cnt']} cores. `BLOCK_DIM` must be one of: **{{{vir_str}}}** (max {max_block_dim}).
- **BLOCK_DIM requirement**: `totalLength` MUST be evenly divisible by `BLOCK_DIM`, otherwise data is silently dropped.
- **UB per core**: {ub_kb} KB total, **~{usable_kb} KB usable** budget for `InitBuffer` allocations.
- **DataCopy alignment**: All transfer lengths must be multiples of **8 (float32)** or **16 (float16)** — i.e., 32-byte aligned.

### Quick Reference Table
| Queues | BUFFER_NUM | dtype  | Max tileLength |
|--------|-----------|--------|----------------|
{table}
"""

# Pattern → paradigm mapping
_PARADIGM: dict[str, str] = {
    "elementwise": "vector",
    "reduction": "vector",
    "softmax": "vector",
    "broadcast": "vector",
    "pooling": "vector",
    "matmul": "cube",
    "convolution": "cube",
    "attention": "cube",
    "normalization": "mixed",
    "index": "mixed",
    "resize": "mixed",
    "other": "vector",  # fallback
}


def _load_md(relative_path: str) -> str:
    """Load a .md file from the knowledge directory, with caching."""
    if relative_path not in _file_cache:
        _file_cache[relative_path] = (_KNOWLEDGE_DIR / relative_path).read_text()
    return _file_cache[relative_path]


class CANNKnowledgeProvider:
    """Fine-grained knowledge provider for AscendC operator development.

    Each method returns a single knowledge piece. Callers assemble
    the pieces they need for their specific stage/context.
    """

    def __init__(
        self,
        cann_path: Optional[str] = None,
        npu_type: str = "Ascend910B2",
    ):
        self._cann_path = cann_path or api_scanner.default_cann_path()
        self._api_index: Optional[Dict[str, Any]] = None
        self._npu_type = npu_type
        self._hw_spec = _read_platform_spec(npu_type)

    @property
    def _index(self) -> Dict[str, Any]:
        """Lazy-load the API index from CANN SDK headers."""
        if self._api_index is None:
            self._api_index = api_scanner.scan_headers(self._cann_path)
        return self._api_index

    # ================================================================
    # Level 0: Programming Model
    # ================================================================

    def get_programming_model(self) -> str:
        """Level 0: AscendC programming model overview (~2000 chars)."""
        return LEVEL0_PROGRAMMING_MODEL

    # ================================================================
    # Level 1: Pattern-Specific Guides
    # ================================================================

    def get_pattern_guide(self, pattern: str) -> str:
        """Level 1: Guide for a specific compute pattern.

        Patterns: elementwise, reduction, softmax, broadcast, pooling,
                  matmul, convolution, attention, normalization, index, resize, other
        """
        return get_pattern_primer(pattern)

    def get_primer(self, compute_pattern: str) -> str:
        """Convenience: Level 0 + Level 1 combined."""
        return "\n\n".join([
            self.get_programming_model(),
            self.get_pattern_guide(compute_pattern),
        ])

    # ================================================================
    # Constraints
    # ================================================================

    def get_critical_constraints(self) -> str:
        """Full critical constraints with code examples (~4250 chars)."""
        return _load_md("constraints/critical_full.md")

    def get_critical_constraints_compact(self) -> str:
        """Compact constraints — key rules only (~1040 chars)."""
        return _load_md("constraints/critical_compact.md")

    # ================================================================
    # Tiling
    # ================================================================

    def get_hardware_constraints(self) -> str:
        """Hardware constraints for the target NPU (dynamically generated)."""
        return _format_hardware_constraints(self._hw_spec, self._npu_type)

    def get_tiling_fundamentals(self) -> str:
        """Full tiling guide: UB budget, calculation, common mistakes."""
        hw = self.get_hardware_constraints()
        body = _load_md("tiling/fundamentals.md")
        return f"{hw}\n\n{body}"

    def get_tiling_edge_cases(self) -> str:
        """Tiling tail handling guide."""
        return _load_md("tiling/edge_cases.md")

    def get_tiling_quick_reference(self) -> str:
        """Compact tiling reference: formula + table (with hardware constraints)."""
        hw = self.get_hardware_constraints()
        body = _load_md("tiling/quick_reference.md")
        return f"{hw}\n\n{body}"

    def get_tiling_cube_fundamentals(self) -> str:
        """Cube tiling guide: TCubeTiling, MatmulTiling, workspace."""
        return _load_md("tiling/cube_fundamentals.md")

    def get_tiling_multidim_fundamentals(self) -> str:
        """Multi-dimensional tiling: batch/channel/spatial loops, shape passing."""
        return _load_md("tiling/multidim_fundamentals.md")

    def get_tiling_for_paradigm(self, paradigm: str) -> str:
        """Get the appropriate tiling guide for a paradigm."""
        if paradigm == "cube":
            return self.get_tiling_cube_fundamentals()
        elif paradigm == "mixed":
            return self.get_tiling_multidim_fundamentals()
        else:
            # vector: original fundamentals + edge cases
            return "\n\n".join([
                self.get_tiling_fundamentals(),
                self.get_tiling_edge_cases(),
            ])

    # ================================================================
    # API Reference
    # ================================================================

    def get_api_quick_reference(self) -> str:
        """Compact API reference: all function signatures."""
        return _load_md("api/quick_reference.md")

    def get_advanced_api_reference(self) -> str:
        """Advanced APIs: Matmul, Normalization, Index, Transpose."""
        return _load_md("api/advanced_reference.md")

    # ================================================================
    # Level 2: Real-time API Search (CANN SDK headers)
    # ================================================================

    def search_api(self, name: str) -> dict:
        """Search for an API by name in CANN SDK headers."""
        return api_scanner.search(name, self._index)

    def list_apis(self) -> dict:
        """List all APIs grouped by high-level category."""
        return api_scanner.list_apis_grouped(self._index)

    # ================================================================
    # Level 3: Curated Examples
    # ================================================================

    def get_example(self, pattern: str) -> Optional[str]:
        """Get a curated complete example for the given pattern."""
        return _get_example(pattern)

    # ================================================================
    # Assembly helpers — pre-built knowledge bundles for common stages
    # ================================================================

    @staticmethod
    def get_paradigm(pattern: str) -> str:
        """Get the paradigm (vector/cube/mixed) for a compute pattern."""
        return _PARADIGM.get(pattern, "vector")

    def assemble_for_init(self, pattern: str, needs_advanced: bool = False) -> str:
        """Assemble knowledge for init stage (from-scratch generation).

        Tiling guide is selected based on paradigm:
        - vector: fundamentals + edge_cases (UB-based 1D tiling)
        - cube: cube_fundamentals (TCubeTiling, MatmulTiling, workspace)
        - mixed: multidim_fundamentals (batch/channel/spatial nested loops)

        API quick reference is always included (all paradigms need basic
        Vector APIs — even attention needs ReduceMax/Exp for softmax).
        Advanced API is included for cube/mixed paradigms, or when
        explicitly requested (needs_advanced=True).
        """
        paradigm = self.get_paradigm(pattern)

        parts = [
            self.get_primer(pattern),
            self.get_critical_constraints(),
            self.get_api_quick_reference(),
        ]

        example = self.get_example(pattern)
        if example:
            parts.append(example)

        # Paradigm-specific tiling
        parts.append(self.get_tiling_for_paradigm(paradigm))

        # Advanced API: always for cube/mixed, optional for vector
        if paradigm in ("cube", "mixed") or needs_advanced:
            parts.append(self.get_advanced_api_reference())

        return "\n\n".join(parts)

    def assemble_for_compile_fix(self) -> str:
        """Assemble knowledge for compile-fix stage.

        Targeted: API reference + compact constraints only.
        No tiling guide, no examples — the code structure is already there.
        """
        return "\n\n".join([
            self.get_api_quick_reference(),
            self.get_critical_constraints_compact(),
        ])

    def assemble_for_correctness_fix(self, pattern: str) -> str:
        """Assemble knowledge for correctness-fix stage.

        Targeted: pattern guide + paradigm-appropriate tiling reference.
        Focus on computation logic and data flow, not API syntax.

        - vector: tiling quick reference (UB formula table)
        - cube: cube tiling fundamentals (M/K/N, TCubeTiling, workspace)
        - mixed: multidim tiling fundamentals (batch/channel/spatial loops)
        """
        paradigm = self.get_paradigm(pattern)

        parts = [self.get_pattern_guide(pattern)]

        if paradigm == "cube":
            parts.append(self.get_tiling_cube_fundamentals())
        elif paradigm == "mixed":
            parts.append(self.get_tiling_multidim_fundamentals())
        else:
            parts.append(self.get_tiling_quick_reference())

        return "\n\n".join(parts)

    def assemble_for_evolve(self, pattern: str = "other") -> str:
        """Assemble knowledge for evolve stage (optimize existing code).

        Compact: API ref + constraints + paradigm-appropriate tiling reference.
        """
        paradigm = self.get_paradigm(pattern)

        parts = [
            self.get_api_quick_reference(),
            self.get_critical_constraints_compact(),
        ]

        if paradigm == "cube":
            parts.append(self.get_tiling_cube_fundamentals())
        elif paradigm == "mixed":
            parts.append(self.get_tiling_multidim_fundamentals())
        else:
            parts.append(self.get_tiling_quick_reference())

        return "\n\n".join(parts)

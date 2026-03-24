# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Core types for cann_parallel_evaluator.

Inlines evotoolkit's Solution, EvaluationResult, SolutionMetadata, and TaskSpec
so the package has zero dependency on evotoolkit.

Also provides BaseTask — the abstract base for CANNInitTask.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

class EvaluationResult:
    """Stores the result of evaluating a solution."""

    def __init__(self, valid, score, additional_info):
        self.valid = valid
        self.score = score
        self.additional_info = additional_info


# ---------------------------------------------------------------------------
# SolutionMetadata
# ---------------------------------------------------------------------------

@dataclass
class SolutionMetadata:
    """Typed metadata carried alongside a candidate solution."""

    name: str = ""
    description: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, metadata: "SolutionMetadata | Mapping[str, Any] | None") -> "SolutionMetadata":
        if metadata is None:
            return cls()
        if isinstance(metadata, cls):
            return cls(
                name=metadata.name,
                description=metadata.description,
                extras=dict(metadata.extras),
            )
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Unsupported solution metadata type: {type(metadata)!r}")

        payload = dict(metadata)
        name = payload.pop("name", "")
        description = payload.pop("description", payload.pop("thought", payload.pop("algorithm", "")))
        return cls(
            name="" if name is None else str(name),
            description="" if description is None else str(description),
            extras=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extras)
        if self.name:
            payload["name"] = self.name
        if self.description:
            payload["description"] = self.description
        return payload

    def with_defaults(self, *, name: str = "", description: str = "") -> "SolutionMetadata":
        return SolutionMetadata(
            name=self.name or name,
            description=self.description or description,
            extras=dict(self.extras),
        )


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------

class _BaseSolution:
    """Base solution class."""

    def __init__(
        self,
        sol_string: str,
        metadata: "SolutionMetadata | Mapping[str, Any] | None" = None,
        evaluation_res: Optional[EvaluationResult] = None,
    ):
        self.sol_string = sol_string
        self.metadata = SolutionMetadata.coerce(metadata)
        self.evaluation_res = evaluation_res


class Solution(_BaseSolution):
    """Solution with backward-compatible other_info property.

    Stores the 6 CANN components in metadata.extras, accessible as
    other_info for convenience.
    """

    def __init__(
        self,
        sol_string: str,
        other_info: Optional[dict[str, Any]] = None,
        evaluation_res: Optional[EvaluationResult] = None,
        metadata: Optional[SolutionMetadata] = None,
    ):
        if other_info is not None and metadata is None:
            metadata = SolutionMetadata(extras=other_info)
        super().__init__(sol_string=sol_string, metadata=metadata, evaluation_res=evaluation_res)

    @property
    def other_info(self) -> Optional[dict[str, Any]]:
        """Alias for metadata.extras."""
        return self.metadata.extras if self.metadata.extras else None

    @other_info.setter
    def other_info(self, value: Optional[dict[str, Any]]):
        if value is not None:
            self.metadata = SolutionMetadata(
                name=self.metadata.name,
                description=self.metadata.description,
                extras=value,
            )


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Static task specification."""

    name: str = ""
    prompt: str = ""
    modality: str = "generic"
    extras: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "TaskSpec":
        return TaskSpec(
            name=self.name,
            prompt=self.prompt,
            modality=self.modality,
            extras=dict(self.extras),
        )


# ---------------------------------------------------------------------------
# BaseTask
# ---------------------------------------------------------------------------

class BaseTask(ABC):
    """Abstract base class for CANN operator evaluation tasks."""

    def __init__(self, data):
        self._process_data(data)

    def _process_data(self, data):
        self.data = data
        self.task_info = {}

    @abstractmethod
    def evaluate_code(self, candidate_code: str) -> EvaluationResult:
        pass

    def evaluate_solution(self, solution: Solution) -> EvaluationResult:
        return self.evaluate_code(solution.sol_string)

    @abstractmethod
    def get_base_task_description(self) -> str:
        pass

    @abstractmethod
    def make_init_sol_wo_other_info(self) -> Solution:
        pass

    def get_task_type(self) -> str:
        return "Python"

    def get_task_info(self) -> dict:
        return self.task_info

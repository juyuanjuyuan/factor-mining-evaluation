"""Typed contracts for reusable factor-admission evaluation standards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from evaluators import EvaluationMethod


@dataclass(frozen=True)
class GateCondition:
    """One auditable condition used by an evaluation-standard gate."""

    name: str
    metric: str
    operator: str
    expected: Any
    actual: Any
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StandardGateResult:
    """Gate outcome plus every atomic condition that produced it."""

    standard: str
    passed: bool
    explanation: str
    conditions: tuple[GateCondition, ...]
    logic: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "standard": self.standard,
            "passed": self.passed,
            "explanation": self.explanation,
            "logic": self.logic,
            "conditions": [condition.as_dict() for condition in self.conditions],
        }


@dataclass(frozen=True)
class EvaluationStandard:
    """A frozen ordered evaluator pipeline used as one admission standard."""

    name: str
    label: str
    methods: tuple[EvaluationMethod, ...]
    description: str

    @property
    def method_names(self) -> tuple[str, ...]:
        from evaluators import evaluation_method_names

        return evaluation_method_names(self.methods)


GateMetrics = Mapping[str, Any]

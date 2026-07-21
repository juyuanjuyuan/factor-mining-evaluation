"""Registered model-training primitives for reusable multi-factor models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import pandas as pd

from engine import parse_and_validate_expression


@dataclass(frozen=True)
class ModelTerm:
    """One frozen factor definition and its currently stored linear weight."""

    batch_id: str
    factor_name: str
    expression: str
    weight: float


@dataclass(frozen=True)
class TrainingParameter:
    """One user-configurable scalar exposed by a training method."""

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float
    description: str


@dataclass(frozen=True)
class ModelTrainingContext:
    """Immutable data supplied to one selected training method."""

    terms: tuple[ModelTerm, ...]
    market_data: Mapping[str, pd.DataFrame]
    signal_start: str
    signal_end: str
    horizon: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ModelTrainingResult:
    """A fitted executable expression plus auditable training diagnostics."""

    weights: tuple[float, ...]
    expression: str
    diagnostics: Mapping[str, Any]

    def as_dict(self, context: ModelTrainingContext, method_name: str) -> dict[str, Any]:
        return {
            "method": method_name,
            "parameters": dict(context.parameters),
            "terms": [
                {
                    "batch_id": term.batch_id,
                    "factor_name": term.factor_name,
                    "expression": term.expression,
                    "weight": weight,
                }
                for term, weight in zip(context.terms, self.weights, strict=True)
            ],
            "expression": self.expression,
            "diagnostics": dict(self.diagnostics),
        }


ModelTrainingMethod = Callable[[ModelTrainingContext], ModelTrainingResult]
ScoreExpressionBuilder = Callable[[tuple[ModelTerm, ...], tuple[float, ...], Mapping[str, Any]], str]


@dataclass(frozen=True)
class ModelTrainingMethodMetadata:
    name: str
    label: str
    description: str
    requires_fitting: bool
    term_weight_editable: bool
    parameters: tuple[TrainingParameter, ...]
    score_expression_builder: ScoreExpressionBuilder


REGISTERED_MODEL_TRAINING_METHODS: dict[str, ModelTrainingMethod] = {}


def model_training_method(
    name: str,
    *,
    label: str,
    description: str,
    requires_fitting: bool,
    term_weight_editable: bool,
    score_expression_builder: ScoreExpressionBuilder,
    parameters: tuple[TrainingParameter, ...] = (),
) -> Callable[[ModelTrainingMethod], ModelTrainingMethod]:
    """Register one independently replaceable model-training implementation."""

    if not name.strip():
        raise ValueError("Model training method name cannot be empty")

    def decorator(function: ModelTrainingMethod) -> ModelTrainingMethod:
        existing = REGISTERED_MODEL_TRAINING_METHODS.get(name)
        if existing is not None and existing is not function:
            raise ValueError(f"Duplicate model training method name: {name}")
        setattr(
            function,
            "__model_training_method__",
            ModelTrainingMethodMetadata(
                name=name,
                label=label,
                description=description,
                requires_fitting=requires_fitting,
                term_weight_editable=term_weight_editable,
                parameters=parameters,
                score_expression_builder=score_expression_builder,
            ),
        )
        REGISTERED_MODEL_TRAINING_METHODS[name] = function
        return function

    return decorator


def training_method_metadata(
    method: ModelTrainingMethod,
) -> ModelTrainingMethodMetadata:
    metadata = getattr(method, "__model_training_method__", None)
    if not isinstance(metadata, ModelTrainingMethodMetadata):
        raise TypeError(f"{method!r} is not a registered model training method")
    return metadata


def build_rank_linear_expression(
    terms: tuple[ModelTerm, ...], weights: tuple[float, ...]
) -> str:
    """Build the exact rank-linear score evaluated in both train and test."""

    if len(terms) != len(weights) or not terms:
        raise ValueError("Linear model terms and weights must have equal nonzero length")
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("Linear model weights must all be finite")
    if not any(weight != 0 for weight in weights):
        raise ValueError("At least one fitted linear weight must be nonzero")
    pieces = [
        f"({weight:.17g} * rank_cs(({term.expression})))"
        for term, weight in zip(terms, weights, strict=True)
    ]
    expression = " + ".join(pieces)
    parse_and_validate_expression(expression)
    return expression


def build_winsorized_zscore_linear_expression(
    terms: tuple[ModelTerm, ...],
    weights: tuple[float, ...],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> str:
    """Build a deployable raw-amplitude score with daily robust z-scores."""

    if not (0 <= lower_quantile < upper_quantile <= 1):
        raise ValueError("winsorization quantiles must satisfy 0 <= lower < upper <= 1")
    if len(terms) != len(weights) or not terms:
        raise ValueError("Linear model terms and weights must have equal nonzero length")
    if not all(math.isfinite(weight) for weight in weights):
        raise ValueError("Linear model weights must all be finite")
    if not any(weight != 0 for weight in weights):
        raise ValueError("At least one fitted linear weight must be nonzero")
    pieces = [
        "("
        f"{weight:.17g} * zscore_cs(winsorize_cs(({term.expression}), "
        f"{lower_quantile:.12g}, {upper_quantile:.12g}))"
        ")"
        for term, weight in zip(terms, weights, strict=True)
    ]
    expression = " + ".join(pieces)
    parse_and_validate_expression(expression)
    return expression

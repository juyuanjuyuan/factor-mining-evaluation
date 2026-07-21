"""Registry, validation, and execution for selectable model trainers."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .base import (
    REGISTERED_MODEL_TRAINING_METHODS,
    ModelTrainingContext,
    ModelTrainingMethod,
    ModelTrainingMethodMetadata,
    ModelTrainingResult,
    training_method_metadata,
)
from .manual_weights import train_manual_weights
from .rank_ridge import train_rank_ridge
from .winsorized_zscore_ridge import train_winsorized_zscore_ridge


DEFAULT_MODEL_TRAINING_METHOD = "winsorized_zscore_ridge"


def resolve_model_training_method(name: str | None) -> ModelTrainingMethod:
    normalized = (name or DEFAULT_MODEL_TRAINING_METHOD).strip()
    try:
        return REGISTERED_MODEL_TRAINING_METHODS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(REGISTERED_MODEL_TRAINING_METHODS))
        raise ValueError(
            f"Unknown model training method {normalized!r}; available methods: {available}"
        ) from exc


def normalize_training_parameters(
    name: str | None, supplied: Mapping[str, Any] | None
) -> dict[str, float]:
    metadata = training_method_metadata(resolve_model_training_method(name))
    supplied_values = dict(supplied or {})
    known = {parameter.name for parameter in metadata.parameters}
    unknown = sorted(set(supplied_values) - known)
    if unknown:
        raise ValueError(
            f"Training method {metadata.name!r} does not accept parameters: {unknown}"
        )
    result: dict[str, float] = {}
    for parameter in metadata.parameters:
        raw = supplied_values.get(parameter.name, parameter.default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{parameter.name} must be numeric") from exc
        if not math.isfinite(value) or not parameter.minimum <= value <= parameter.maximum:
            raise ValueError(
                f"{parameter.name} must be between {parameter.minimum} and {parameter.maximum}"
            )
        result[parameter.name] = value
    return result


def run_model_training(
    name: str | None, context: ModelTrainingContext
) -> ModelTrainingResult:
    method = resolve_model_training_method(name)
    result = method(context)
    if len(result.weights) != len(context.terms):
        raise ValueError("Training method returned a weight count that does not match the terms")
    return result


def build_model_expression(
    name: str | None,
    terms,
    weights,
    parameters: Mapping[str, Any],
) -> str:
    """Build the selected method's deployable score without router branches."""

    metadata = training_method_metadata(resolve_model_training_method(name))
    return metadata.score_expression_builder(tuple(terms), tuple(weights), parameters)


def available_model_training_methods() -> tuple[ModelTrainingMethodMetadata, ...]:
    return tuple(
        training_method_metadata(REGISTERED_MODEL_TRAINING_METHODS[name])
        for name in sorted(REGISTERED_MODEL_TRAINING_METHODS)
    )

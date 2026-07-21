"""Public model-training registry surface."""

from .base import (
    REGISTERED_MODEL_TRAINING_METHODS,
    ModelTerm,
    ModelTrainingContext,
    ModelTrainingMethodMetadata,
    ModelTrainingResult,
    TrainingParameter,
    build_rank_linear_expression,
    build_winsorized_zscore_linear_expression,
    training_method_metadata,
)
from .registry import (
    DEFAULT_MODEL_TRAINING_METHOD,
    available_model_training_methods,
    build_model_expression,
    normalize_training_parameters,
    resolve_model_training_method,
    run_model_training,
)

__all__ = [
    "DEFAULT_MODEL_TRAINING_METHOD",
    "REGISTERED_MODEL_TRAINING_METHODS",
    "ModelTerm",
    "ModelTrainingContext",
    "ModelTrainingMethodMetadata",
    "ModelTrainingResult",
    "TrainingParameter",
    "available_model_training_methods",
    "build_model_expression",
    "build_rank_linear_expression",
    "build_winsorized_zscore_linear_expression",
    "normalize_training_parameters",
    "resolve_model_training_method",
    "run_model_training",
    "training_method_metadata",
]

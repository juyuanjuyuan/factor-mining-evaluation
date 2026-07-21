"""Manual rank-linear weights, kept as a registered baseline method."""

from __future__ import annotations

from .base import (
    ModelTrainingContext,
    ModelTrainingResult,
    build_rank_linear_expression,
    model_training_method,
)


def build_manual_expression(terms, weights, parameters) -> str:
    return build_rank_linear_expression(terms, weights)


@model_training_method(
    "manual_weights",
    label="手动线性权重",
    description="每个因子先做每日横截面百分位排名，再按用户填写的权重相加。",
    requires_fitting=False,
    term_weight_editable=True,
    score_expression_builder=build_manual_expression,
)
def train_manual_weights(context: ModelTrainingContext) -> ModelTrainingResult:
    weights = tuple(float(term.weight) for term in context.terms)
    return ModelTrainingResult(
        weights=weights,
        expression=build_rank_linear_expression(context.terms, weights),
        diagnostics={
            "feature_transform": "daily_cross_sectional_percentile_rank",
            "target_transform": None,
            "observation_count": None,
            "trading_day_count": None,
        },
    )

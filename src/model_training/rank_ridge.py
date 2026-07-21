"""Daily cross-sectionally centred rank-feature Ridge regression."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine import (
    evaluate_expression,
    prepare_market_data,
    rank_cs,
    restrict_evaluation_window,
)
from returns import RETURN_DEFINITION, calculate_forward_open_return

from .base import (
    ModelTrainingContext,
    ModelTrainingResult,
    TrainingParameter,
    build_rank_linear_expression,
    model_training_method,
)


RIDGE_ALPHA = TrainingParameter(
    name="ridge_alpha",
    label="Ridge 正则强度",
    default=1e-6,
    minimum=0.0,
    maximum=10.0,
    step=1e-6,
    description="越大越压缩相关因子的系数；0 退化为普通最小二乘。",
)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def build_rank_ridge_expression(terms, weights, parameters) -> str:
    return build_rank_linear_expression(terms, weights)


@model_training_method(
    "rank_ridge",
    label="横截面秩 Ridge 回归",
    description=(
        "在训练集内将各因子转换为每日横截面百分位秩，因子和未来收益按日去均值，"
        "再以每个交易日等权的 Ridge 回归拟合线性权重。"
    ),
    requires_fitting=True,
    term_weight_editable=False,
    parameters=(RIDGE_ALPHA,),
    score_expression_builder=build_rank_ridge_expression,
)
def train_rank_ridge(context: ModelTrainingContext) -> ModelTrainingResult:
    if not context.terms:
        raise ValueError("Rank Ridge training requires at least one factor term")
    alpha = float(context.parameters[RIDGE_ALPHA.name])
    if not math.isfinite(alpha) or alpha < RIDGE_ALPHA.minimum:
        raise ValueError("ridge_alpha must be a finite nonnegative number")

    combined_expression = " + ".join(f"({term.expression})" for term in context.terms)
    data = prepare_market_data(combined_expression, context.market_data)
    ranked_features = []
    for term in context.terms:
        raw = evaluate_expression(term.expression, data).replace(
            [np.inf, -np.inf], np.nan
        )
        raw = raw.reindex(index=data["c"].index, columns=data["c"].columns)
        ranked_features.append(rank_cs(raw))

    forward_return = calculate_forward_open_return(data["o"], context.horizon)
    first_feature, _, target, sample = restrict_evaluation_window(
        ranked_features[0],
        data["c"],
        forward_return,
        signal_start=context.signal_start,
        signal_end=context.signal_end,
        horizon=context.horizon,
    )
    effective_index = first_feature.index
    features = [feature.reindex(index=effective_index) for feature in ranked_features]

    minimum_pairs = max(3, len(features) + 1)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    used_dates: list[pd.Timestamp] = []
    valid_pairs_per_day: list[int] = []
    for day in effective_index:
        frame = pd.concat(
            [
                *[feature.loc[day].rename(f"x{position}") for position, feature in enumerate(features)],
                target.loc[day].rename("target"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < minimum_pairs:
            continue
        x_day = frame.iloc[:, :-1].to_numpy(dtype=float)
        y_day = frame.iloc[:, -1].to_numpy(dtype=float)
        x_day -= x_day.mean(axis=0, keepdims=True)
        y_day -= y_day.mean()
        # Dividing each date by sqrt(N_t) makes the objective the average of
        # daily cross-sectional MSEs instead of overweighting broad universes.
        scale = math.sqrt(len(frame))
        x_blocks.append(x_day / scale)
        y_blocks.append(y_day / scale)
        used_dates.append(pd.Timestamp(day))
        valid_pairs_per_day.append(len(frame))

    if len(used_dates) < 2:
        raise ValueError(
            "Rank Ridge training needs at least two training dates with "
            f"{minimum_pairs} complete factor/return observations"
        )

    design = np.vstack(x_blocks)
    response = np.concatenate(y_blocks)
    day_count = len(used_dates)
    gram = design.T @ design / day_count
    rhs = design.T @ response / day_count
    penalized = gram + alpha * np.eye(len(features), dtype=float)
    try:
        coefficients = np.linalg.solve(penalized, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(penalized, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all() or not np.any(coefficients != 0):
        raise ValueError("Rank Ridge produced invalid or all-zero coefficients")

    prediction = design @ coefficients
    residual = response - prediction
    target_variance = float(response @ response / day_count)
    mse = float(residual @ residual / day_count)
    r_squared = (
        1.0 - mse / target_variance if target_variance > 0 else float("nan")
    )
    weights = tuple(float(value) for value in coefficients)
    diagnostics = {
        "return_definition": RETURN_DEFINITION,
        "feature_transform": "daily_cross_sectional_percentile_rank_then_demean",
        "target_transform": "daily_cross_sectional_demean",
        "objective": "equal_weighted_daily_cross_sectional_mse_plus_l2",
        "intercept": 0.0,
        "ridge_alpha": alpha,
        "observation_count": int(sum(valid_pairs_per_day)),
        "trading_day_count": day_count,
        "minimum_pairs_per_day": minimum_pairs,
        "mean_pairs_per_day": float(np.mean(valid_pairs_per_day)),
        "sample_start_day": used_dates[0].date().isoformat(),
        "sample_end_day": used_dates[-1].date().isoformat(),
        "requested_sample_start_day": context.signal_start,
        "requested_sample_end_day": context.signal_end,
        "training_mse": mse,
        "training_r_squared": _finite_or_none(r_squared),
        "design_condition_number": _finite_or_none(float(np.linalg.cond(gram))),
    }
    return ModelTrainingResult(
        weights=weights,
        expression=build_rank_linear_expression(context.terms, weights),
        diagnostics=diagnostics,
    )

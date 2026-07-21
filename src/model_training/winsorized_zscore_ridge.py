"""Raw-amplitude Ridge regression with daily winsorization and z-scoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine import (
    evaluate_expression,
    prepare_market_data,
    restrict_evaluation_window,
    winsorize_cs,
    zscore_cs,
)
from returns import RETURN_DEFINITION, calculate_forward_open_return

from .base import (
    ModelTrainingContext,
    ModelTrainingResult,
    TrainingParameter,
    build_winsorized_zscore_linear_expression,
    model_training_method,
)
from .rank_ridge import RIDGE_ALPHA, _finite_or_none


WINSOR_LOWER_QUANTILE = TrainingParameter(
    name="winsor_lower_quantile",
    label="去极值下分位点",
    default=0.01,
    minimum=0.0,
    maximum=0.49,
    step=0.01,
    description="低于该日截面分位点的原始因子值会被截到该分位点。",
)
WINSOR_UPPER_QUANTILE = TrainingParameter(
    name="winsor_upper_quantile",
    label="去极值上分位点",
    default=0.99,
    minimum=0.51,
    maximum=1.0,
    step=0.01,
    description="高于该日截面分位点的原始因子值会被截到该分位点。",
)


def build_winsorized_zscore_ridge_expression(terms, weights, parameters) -> str:
    return build_winsorized_zscore_linear_expression(
        terms,
        weights,
        lower_quantile=float(parameters[WINSOR_LOWER_QUANTILE.name]),
        upper_quantile=float(parameters[WINSOR_UPPER_QUANTILE.name]),
    )


@model_training_method(
    "winsorized_zscore_ridge",
    label="原始幅度 Z-score Ridge 回归",
    description=(
        "保留因子原始幅度：每天先按分位数去极值，再做横截面 z-score；"
        "在训练集内以每个交易日等权的 Ridge 回归拟合线性权重。"
    ),
    requires_fitting=True,
    term_weight_editable=False,
    score_expression_builder=build_winsorized_zscore_ridge_expression,
    parameters=(RIDGE_ALPHA, WINSOR_LOWER_QUANTILE, WINSOR_UPPER_QUANTILE),
)
def train_winsorized_zscore_ridge(
    context: ModelTrainingContext,
) -> ModelTrainingResult:
    if not context.terms:
        raise ValueError("Z-score Ridge training requires at least one factor term")
    alpha = float(context.parameters[RIDGE_ALPHA.name])
    lower = float(context.parameters[WINSOR_LOWER_QUANTILE.name])
    upper = float(context.parameters[WINSOR_UPPER_QUANTILE.name])
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge_alpha must be a finite nonnegative number")
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper <= 1):
        raise ValueError("winsorization quantiles must satisfy 0 <= lower < upper <= 1")

    combined_expression = " + ".join(f"({term.expression})" for term in context.terms)
    data = prepare_market_data(combined_expression, context.market_data)
    standardized_features = []
    for term in context.terms:
        raw = evaluate_expression(term.expression, data).replace(
            [np.inf, -np.inf], np.nan
        )
        raw = raw.reindex(index=data["c"].index, columns=data["c"].columns)
        standardized_features.append(zscore_cs(winsorize_cs(raw, lower, upper)))

    forward_return = calculate_forward_open_return(data["o"], context.horizon)
    first_feature, _, target, _ = restrict_evaluation_window(
        standardized_features[0],
        data["c"],
        forward_return,
        signal_start=context.signal_start,
        signal_end=context.signal_end,
        horizon=context.horizon,
    )
    effective_index = first_feature.index
    features = [feature.reindex(index=effective_index) for feature in standardized_features]

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
        # z-score uses each factor's available universe. Recentring after the
        # common finite-value intersection removes only daily score constants.
        x_day -= x_day.mean(axis=0, keepdims=True)
        y_day -= y_day.mean()
        scale = math.sqrt(len(frame))
        x_blocks.append(x_day / scale)
        y_blocks.append(y_day / scale)
        used_dates.append(pd.Timestamp(day))
        valid_pairs_per_day.append(len(frame))

    if len(used_dates) < 2:
        raise ValueError(
            "Z-score Ridge training needs at least two training dates with "
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
        raise ValueError("Z-score Ridge produced invalid or all-zero coefficients")

    prediction = design @ coefficients
    residual = response - prediction
    target_variance = float(response @ response / day_count)
    mse = float(residual @ residual / day_count)
    r_squared = 1.0 - mse / target_variance if target_variance > 0 else float("nan")
    weights = tuple(float(value) for value in coefficients)
    diagnostics = {
        "return_definition": RETURN_DEFINITION,
        "feature_transform": "daily_winsorize_then_cross_sectional_zscore_then_common_universe_demean",
        "target_transform": "daily_cross_sectional_demean",
        "objective": "equal_weighted_daily_cross_sectional_mse_plus_l2",
        "intercept": 0.0,
        "ridge_alpha": alpha,
        "winsor_lower_quantile": lower,
        "winsor_upper_quantile": upper,
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
        expression=build_winsorized_zscore_ridge_expression(
            context.terms, weights, context.parameters
        ),
        diagnostics=diagnostics,
    )

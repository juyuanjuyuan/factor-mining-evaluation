"""Code-owned evaluation standards mirroring the two Webapp templates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np

from evaluators.base import REGISTERED_EVALUATION_METHODS
from evaluators.registry import available_evaluation_methods  # noqa: F401

from .base import EvaluationStandard, GateCondition, StandardGateResult


IC_STANDARD_NAME = "ic_test"
PROFITABILITY_STANDARD_NAME = "profitability_test"

IC_METHOD_NAMES = (
    "prefix_truncation_consistency",
    "market_cap_neutralize",
    "rank_ic",
    "rank_icir",
    "newey_west_ic_significance",
    "ic_trend_filter",
    "ic_peak_decay",
)

PROFITABILITY_METHOD_NAMES = (
    "market_cap_neutralize",
    "tradability_filter",
    "quantile_net_returns",
    "quantile_cumulative",
    "quantile_plot",
    "rolling_sharpe",
    "rolling_drawdown",
    "top_quantile_performance",
    "cycle_context",
)


def _methods(names: tuple[str, ...]):
    missing = [name for name in names if name not in REGISTERED_EVALUATION_METHODS]
    if missing:
        raise RuntimeError(f"Evaluation-standard methods are not registered: {missing}")
    return tuple(REGISTERED_EVALUATION_METHODS[name] for name in names)


REGISTERED_EVALUATION_STANDARDS: dict[str, EvaluationStandard] = {
    IC_STANDARD_NAME: EvaluationStandard(
        name=IC_STANDARD_NAME,
        label="IC检测",
        methods=_methods(IC_METHOD_NAMES),
        description=(
            "测试集上的因果一致性、市值中性 Rank IC/IR、Newey-West 显著性、"
            "IC 趋势与 60 日滚动 Mean IC。"
        ),
    ),
    PROFITABILITY_STANDARD_NAME: EvaluationStandard(
        name=PROFITABILITY_STANDARD_NAME,
        label="盈利能力测试",
        methods=_methods(PROFITABILITY_METHOD_NAMES),
        description=(
            "测试集上的市值中性化、入场可交易性过滤、净分组收益、分段风险与"
            "最高组表现。"
        ),
    ),
}


def evaluation_standard_names() -> tuple[str, ...]:
    return tuple(REGISTERED_EVALUATION_STANDARDS)


def resolve_evaluation_standards(
    names: str | Iterable[str] | None,
) -> tuple[EvaluationStandard, ...]:
    if names is None or names == "all":
        requested = list(evaluation_standard_names())
    elif isinstance(names, str):
        requested = [part.strip() for part in names.split(",") if part.strip()]
    else:
        requested = [str(name).strip() for name in names if str(name).strip()]
    if not requested:
        raise ValueError("At least one evaluation standard must be selected")
    unknown = [name for name in requested if name not in REGISTERED_EVALUATION_STANDARDS]
    if unknown:
        raise ValueError(
            f"Unknown evaluation standards {unknown}; available: "
            f"{list(evaluation_standard_names())}"
        )
    if len(requested) != len(set(requested)):
        raise ValueError("Evaluation standards cannot be repeated")
    return tuple(REGISTERED_EVALUATION_STANDARDS[name] for name in requested)


def _finite_number(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def evaluate_ic_gate(
    metrics: Mapping[str, Any],
    *,
    significance_level: float = 0.05,
    minimum_ic_mean: float = 0.0,
) -> StandardGateResult:
    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be strictly between zero and one")
    ic_mean = _finite_number(metrics.get("ic_mean"))
    p_value = _finite_number(metrics.get("nw_ic_p_value"))
    prefix_passed = metrics.get("prefix_truncation_passed") is True
    conditions = (
        GateCondition(
            name="前缀截断一致性",
            metric="prefix_truncation_passed",
            operator="is",
            expected=True,
            actual=metrics.get("prefix_truncation_passed"),
            passed=prefix_passed,
        ),
        GateCondition(
            name="市值中性 Rank IC 均值为正",
            metric="ic_mean",
            operator=">",
            expected=float(minimum_ic_mean),
            actual=ic_mean,
            passed=bool(np.isfinite(ic_mean) and ic_mean > minimum_ic_mean),
        ),
        GateCondition(
            name="Newey-West 双侧显著性",
            metric="nw_ic_p_value",
            operator="<",
            expected=float(significance_level),
            actual=p_value,
            passed=bool(np.isfinite(p_value) and p_value < significance_level),
        ),
    )
    passed = all(condition.passed for condition in conditions)
    return StandardGateResult(
        standard=IC_STANDARD_NAME,
        passed=passed,
        explanation=(
            "通过：因果一致、IC 为正且 Newey-West 显著"
            if passed
            else "未同时满足因果一致、正 IC 与 Newey-West 显著性"
        ),
        conditions=conditions,
        logic="all",
    )


def evaluate_profitability_gate(
    metrics: Mapping[str, Any],
    *,
    minimum_rolling_sharpe_60_median: float = 1.0,
    minimum_annualized_return: float = 0.30,
) -> StandardGateResult:
    leadership = metrics.get("top_group_above_every_lower_group") is True
    sharpe = _finite_number(metrics.get("gn_rolling_sharpe_60_median"))
    annualized_return = _finite_number(metrics.get("top_group_annualized_return"))
    conditions = (
        GateCondition(
            name="最高组全面占优",
            metric="top_group_above_every_lower_group",
            operator="is",
            expected=True,
            actual=metrics.get("top_group_above_every_lower_group"),
            passed=leadership,
        ),
        GateCondition(
            name="最高组分段夏普(60日)中位数",
            metric="gn_rolling_sharpe_60_median",
            operator=">=",
            expected=float(minimum_rolling_sharpe_60_median),
            actual=sharpe,
            passed=bool(
                np.isfinite(sharpe)
                and sharpe >= minimum_rolling_sharpe_60_median
            ),
        ),
        GateCondition(
            name="最高组年化收益",
            metric="top_group_annualized_return",
            operator=">",
            expected=float(minimum_annualized_return),
            actual=annualized_return,
            passed=bool(
                np.isfinite(annualized_return)
                and annualized_return > minimum_annualized_return
            ),
        ),
    )
    passed = conditions[0].passed and (conditions[1].passed or conditions[2].passed)
    return StandardGateResult(
        standard=PROFITABILITY_STANDARD_NAME,
        passed=passed,
        explanation=(
            "通过：最高组全面占优，且夏普中位数/年化收益至少一项达标"
            if passed
            else "未满足最高组全面占优且两项盈利门槛至少一项达标"
        ),
        conditions=conditions,
        logic="condition[0] and (condition[1] or condition[2])",
    )


def evaluate_standard_gate(
    standard_name: str,
    metrics: Mapping[str, Any],
    *,
    significance_level: float = 0.05,
    minimum_ic_mean: float = 0.0,
    minimum_rolling_sharpe_60_median: float = 1.0,
    minimum_annualized_return: float = 0.30,
) -> StandardGateResult:
    if standard_name == IC_STANDARD_NAME:
        return evaluate_ic_gate(
            metrics,
            significance_level=significance_level,
            minimum_ic_mean=minimum_ic_mean,
        )
    if standard_name == PROFITABILITY_STANDARD_NAME:
        return evaluate_profitability_gate(
            metrics,
            minimum_rolling_sharpe_60_median=minimum_rolling_sharpe_60_median,
            minimum_annualized_return=minimum_annualized_return,
        )
    raise ValueError(f"Unknown evaluation standard: {standard_name}")

"""Causality checks and catalog metadata for expression operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from engine import evaluate_expression
from evaluators.future_perturbation import future_data_perturbation_test
from evaluators.prefix_truncation import prefix_truncation_consistency_test


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    signature: str
    description: str
    group: str
    expression: str


@dataclass(frozen=True)
class OperatorCausalityResult:
    name: str
    signature: str
    description: str
    group: str
    expression: str
    passed: bool
    prefix_passed: bool
    perturbation_passed: bool
    reason: str = ""


OPERATOR_SPECS: tuple[OperatorSpec, ...] = (
    OperatorSpec("ts_mean", "ts_mean(x, w)", "滚动均值", "时间序列", "ts_mean(c, 5)"),
    OperatorSpec("ts_std", "ts_std(x, w)", "滚动标准差", "时间序列", "ts_std(c, 5)"),
    OperatorSpec("ts_sum", "ts_sum(x, w)", "滚动求和", "时间序列", "ts_sum(c, 5)"),
    OperatorSpec("ts_min", "ts_min(x, w)", "滚动最小值", "时间序列", "ts_min(c, 5)"),
    OperatorSpec("ts_max", "ts_max(x, w)", "滚动最大值", "时间序列", "ts_max(c, 5)"),
    OperatorSpec("ts_median", "ts_median(x, w)", "滚动中位数", "时间序列", "ts_median(c, 5)"),
    OperatorSpec("ts_product", "ts_product(x, w)", "滚动乘积", "时间序列", "ts_product(c / 100, 5)"),
    OperatorSpec("ts_corr", "ts_corr(x, y, w)", "滚动相关系数", "时间序列", "ts_corr(c, vol, 5)"),
    OperatorSpec("ts_cov", "ts_cov(x, y, w)", "滚动协方差", "时间序列", "ts_cov(c, vol, 5)"),
    OperatorSpec("ts_rank", "ts_rank(x, w)", "窗口内百分位排名", "时间序列", "ts_rank(c, 5)"),
    OperatorSpec("ts_argmax", "ts_argmax(x, w)", "窗口内最大值位置", "时间序列", "ts_argmax(c, 5)"),
    OperatorSpec("ts_argmin", "ts_argmin(x, w)", "窗口内最小值位置", "时间序列", "ts_argmin(c, 5)"),
    OperatorSpec("decay_linear", "decay_linear(x, w)", "线性加权均值", "时间序列", "decay_linear(c, 5)"),
    OperatorSpec("delay", "delay(x, p)", "滞后 p 天", "滞后与变化", "delay(c, 2)"),
    OperatorSpec("delta", "delta(x, p)", "当前值减 p 日前值", "滞后与变化", "delta(c, 2)"),
    OperatorSpec("pct", "pct(x, p)", "p 日百分比变化", "滞后与变化", "pct(c, 2)"),
    OperatorSpec("rank_cs", "rank_cs(x)", "当日横截面百分位排名", "横截面", "rank_cs(c)"),
    OperatorSpec("winsorize_cs", "winsorize_cs(x, lo, hi)", "当日横截面分位数去极值", "横截面", "winsorize_cs(c, 0.01, 0.99)"),
    OperatorSpec("zscore_cs", "zscore_cs(x)", "当日横截面标准化", "横截面", "zscore_cs(c)"),
    OperatorSpec("scale_cs", "scale_cs(x)", "当日横截面绝对值归一", "横截面", "scale_cs(c - o)"),
    OperatorSpec("where", "where(cond, a, b)", "宽表条件选择", "条件与逐元素", "where(c > o, c, o)"),
    OperatorSpec("signed_power", "signed_power(x, a)", "保留符号的幂函数", "条件与逐元素", "signed_power(c - o, 2)"),
    OperatorSpec("elementwise_min", "elementwise_min(x, y)", "逐元素最小值", "条件与逐元素", "elementwise_min(c, o)"),
    OperatorSpec("elementwise_max", "elementwise_max(x, y)", "逐元素最大值", "条件与逐元素", "elementwise_max(c, o)"),
    OperatorSpec("adv", "adv(amt, w)", "滚动平均成交额", "时间序列", "adv(amt, 5)"),
    OperatorSpec("abs", "abs(x)", "绝对值", "数学函数", "abs(c - o)"),
    OperatorSpec("log", "log(x)", "自然对数", "数学函数", "log(c)"),
    OperatorSpec("log1p", "log1p(x)", "log(1+x)", "数学函数", "log1p(c)"),
    OperatorSpec("sqrt", "sqrt(x)", "平方根", "数学函数", "sqrt(c)"),
    OperatorSpec("exp", "exp(x)", "指数函数", "数学函数", "exp((c - o) / 100)"),
    OperatorSpec("sign", "sign(x)", "符号函数", "数学函数", "sign(c - o)"),
    OperatorSpec("np.abs", "np.abs(x)", "NumPy 绝对值", "NumPy 函数", "np.abs(c - o)"),
    OperatorSpec("np.log", "np.log(x)", "NumPy 自然对数", "NumPy 函数", "np.log(c)"),
    OperatorSpec("np.log1p", "np.log1p(x)", "NumPy log(1+x)", "NumPy 函数", "np.log1p(c)"),
    OperatorSpec("np.sqrt", "np.sqrt(x)", "NumPy 平方根", "NumPy 函数", "np.sqrt(c)"),
    OperatorSpec("np.exp", "np.exp(x)", "NumPy 指数函数", "NumPy 函数", "np.exp((c - o) / 100)"),
    OperatorSpec("np.sign", "np.sign(x)", "NumPy 符号函数", "NumPy 函数", "np.sign(c - o)"),
    OperatorSpec("np.clip", "np.clip(x, lo, hi)", "NumPy 截尾", "NumPy 函数", "np.clip(c, 80, 140)"),
    OperatorSpec("np.maximum", "np.maximum(x, y)", "NumPy 逐元素最大值", "NumPy 函数", "np.maximum(c, o)"),
    OperatorSpec("np.minimum", "np.minimum(x, y)", "NumPy 逐元素最小值", "NumPy 函数", "np.minimum(c, o)"),
    OperatorSpec("np.isfinite", "np.isfinite(x)", "NumPy 有限值判断", "NumPy 函数", "np.isfinite(c)"),
    OperatorSpec("np.isnan", "np.isnan(x)", "NumPy NaN 判断", "NumPy 函数", "np.isnan(c)"),
)


def synthetic_operator_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20260707)
    index = pd.date_range("2024-01-02", periods=36, freq="B")
    columns = [f"{number:06d}" for number in range(1, 13)]
    base = rng.lognormal(4.6, 0.12, size=(len(index), len(columns)))
    trend = np.linspace(0.95, 1.08, len(index))[:, None]
    close = pd.DataFrame(base * trend, index=index, columns=columns)
    open_prices = close * pd.DataFrame(
        1 + rng.normal(0, 0.01, close.shape),
        index=index,
        columns=columns,
    )
    high = pd.DataFrame(
        np.maximum(close, open_prices) * 1.01,
        index=index,
        columns=columns,
    )
    low = pd.DataFrame(
        np.minimum(close, open_prices) * 0.99,
        index=index,
        columns=columns,
    )
    volume = pd.DataFrame(
        rng.uniform(1e5, 1e6, close.shape),
        index=index,
        columns=columns,
    )
    amount = volume * close
    return {
        "c": close,
        "o": open_prices,
        "h": high,
        "l": low,
        "vol": volume,
        "amt": amount,
        "vwap": (high + low) / 2,
        "cap": pd.DataFrame(
            rng.uniform(1e9, 1e10, close.shape),
            index=index,
            columns=columns,
        ),
    }


def evaluate_operator_causality(spec: OperatorSpec) -> OperatorCausalityResult:
    data = synthetic_operator_data()

    def build(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return evaluate_expression(spec.expression, inputs).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    try:
        factor = build(data)
        if not factor.index.equals(data["c"].index) or not factor.columns.equals(
            data["c"].columns
        ):
            raise ValueError("operator expression did not preserve wide axes")
        _, prefix_summary = prefix_truncation_consistency_test(
            factor,
            data,
            build,
            checkpoint_positions=(8, 18, 30),
        )
        _, perturbation_summary = future_data_perturbation_test(
            factor,
            data,
            build,
            checkpoint_positions=(8, 18, 30),
        )
        prefix_passed = bool(prefix_summary["prefix_truncation_passed"])
        perturbation_passed = bool(
            perturbation_summary["future_perturbation_changed_values"] == 0
        )
        return OperatorCausalityResult(
            name=spec.name,
            signature=spec.signature,
            description=spec.description,
            group=spec.group,
            expression=spec.expression,
            passed=prefix_passed and perturbation_passed,
            prefix_passed=prefix_passed,
            perturbation_passed=perturbation_passed,
        )
    except Exception as exc:
        return OperatorCausalityResult(
            name=spec.name,
            signature=spec.signature,
            description=spec.description,
            group=spec.group,
            expression=spec.expression,
            passed=False,
            prefix_passed=False,
            perturbation_passed=False,
            reason=str(exc),
        )


@lru_cache(maxsize=1)
def operator_causality_results() -> tuple[OperatorCausalityResult, ...]:
    return tuple(evaluate_operator_causality(spec) for spec in OPERATOR_SPECS)


def operator_causality_catalog() -> list[dict[str, Any]]:
    return [asdict(result) for result in operator_causality_results()]


def causal_operator_whitelist() -> frozenset[str]:
    return frozenset(
        result.name for result in operator_causality_results() if result.passed
    )

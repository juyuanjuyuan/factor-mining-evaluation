"""Evaluate a user-defined metric gate against one run's produced metrics.

Unlike the four-stage funnel (which has fixed, canonical gate metrics), an
evaluate-kind job may carry an ad-hoc gate: a list of threshold conditions on
any metric the pipeline produces (e.g. "最高组滚动回撤(60日)最差值 ≥ -0.40").
The gate is purely informational — it never skips or reorders work, it only
labels each finished run passed / not_passed so researchers can screen a batch.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

# Comparison operators keyed by the same short codes the frontend sends.
_BINARY_OPS = {
    "gte": lambda actual, threshold: actual >= threshold,
    "lte": lambda actual, threshold: actual <= threshold,
    "gt": lambda actual, threshold: actual > threshold,
    "lt": lambda actual, threshold: actual < threshold,
}


def _as_number(value: Any) -> float | None:
    """Coerce a stored metric/threshold to a finite float, or None."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _evaluate_condition(
    condition: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metric = str(condition.get("metric") or "")
    op = str(condition.get("op") or "gte")
    threshold = _as_number(condition.get("value"))
    threshold2 = _as_number(condition.get("value2"))
    actual = _as_number(metrics.get(metric)) if metric else None

    missing = actual is None or threshold is None
    if missing:
        passed = False
    elif op == "between":
        if threshold2 is None:
            passed, missing = False, True
        else:
            low, high = sorted((threshold, threshold2))
            passed = low <= actual <= high
    else:
        comparator = _BINARY_OPS.get(op)
        passed = bool(comparator(actual, threshold)) if comparator else False

    return {
        "metric": metric,
        "op": op,
        "value": condition.get("value"),
        "value2": condition.get("value2"),
        "actual": actual,
        "passed": passed,
        "missing": missing,
    }


def evaluate_metric_gate(
    gate: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], str] | None:
    """Return (outcome, per-condition details, explanation) or None.

    None means "no gate configured" — the run keeps its blank gate columns.
    Otherwise outcome is "passed" or "not_passed", details carries the actual
    value and pass flag for every condition (for the results view), and the
    explanation is a short Chinese summary.
    """

    if not gate:
        return None
    conditions = gate.get("conditions") or []
    if not conditions:
        return None
    match = str(gate.get("match") or "all")

    details = [_evaluate_condition(condition, metrics) for condition in conditions]
    flags = [item["passed"] for item in details]
    overall = all(flags) if match == "all" else any(flags)
    passed_count = sum(1 for flag in flags if flag)
    total = len(flags)

    if overall:
        prefix = "全部达标" if match == "all" else "满足任一条件"
        explanation = f"{prefix}（{passed_count}/{total} 项通过）"
    else:
        failed = total - passed_count
        explanation = f"未过关：{passed_count}/{total} 项通过，{failed} 项未达标"

    return ("passed" if overall else "not_passed", details, explanation)

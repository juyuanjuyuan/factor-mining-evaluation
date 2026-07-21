"""Trailing-window maximum-drawdown evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method
from .rolling_windows import DEFAULT_WINDOWS, complete_non_overlapping_windows


def _window_max_drawdown(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return np.nan
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    drawdowns = wealth / peaks - 1.0
    return float(np.min(drawdowns))


def rolling_max_drawdown(
    returns: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> dict[int, pd.DataFrame]:
    """Calculate maximum drawdowns for non-overlapping full windows."""

    results: dict[int, pd.DataFrame] = {}
    for window in windows:
        if window < 2:
            raise ValueError("Rolling drawdown windows must be at least two")
        rows: list[pd.Series] = []
        end_days: list[object] = []
        for end_day, window_returns in complete_non_overlapping_windows(
            returns,
            window=window,
        ):
            rows.append(window_returns.apply(_window_max_drawdown, raw=True))
            end_days.append(end_day)
        results[int(window)] = pd.DataFrame(
            rows,
            index=pd.Index(end_days, name=returns.index.name),
            columns=returns.columns,
            dtype=float,
        )
    return results


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def drawdown_distribution_summary(series: pd.Series) -> dict[str, float]:
    """Summarize the worst and median non-overlapping maximum drawdown."""

    finite = _finite(series)
    if not len(finite):
        return {"worst": np.nan, "median": np.nan}
    return {
        "worst": float(finite.min()),
        "median": float(finite.median()),
    }


@evaluation_method("rolling_drawdown", requires=("quantile_returns",))
def evaluate_rolling_drawdown(state: EvaluationState) -> None:
    """Persist non-overlapping drawdown details and top-quantile summaries."""

    detail = state.require_detail("group_returns")
    if not isinstance(detail, pd.DataFrame):
        raise TypeError("Quantile-return detail must be a pandas DataFrame")
    n_quantiles = state.context.n_quantiles
    metrics: dict[str, float] = {}
    for window, values in rolling_max_drawdown(detail).items():
        state.add_detail(f"rolling_drawdown_{window}", values)
        summary = drawdown_distribution_summary(values[f"G{n_quantiles}"])
        for statistic, value in summary.items():
            metrics[f"gn_rolling_drawdown_{window}_{statistic}"] = value
    state.add_metrics(metrics)

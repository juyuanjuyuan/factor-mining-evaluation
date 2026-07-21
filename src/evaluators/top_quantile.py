"""Top-quantile performance comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


TOP_QUANTILE_LEADERSHIP_WINDOW = 60
TRADING_DAYS_PER_YEAR = 252


def _compound_return(values: pd.Series) -> float:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan
    return float((1.0 + finite).prod() - 1.0)


def non_overlapping_top_quantile_leadership(
    group_returns: pd.DataFrame,
    *,
    n_quantiles: int,
    window: int = TOP_QUANTILE_LEADERSHIP_WINDOW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check whether the top quantile has the best cumulative return."""

    if n_quantiles < 3:
        raise ValueError("Top-quantile analysis requires at least three groups")
    if window < 1:
        raise ValueError("Top-quantile leadership window must be positive")

    group_columns = [f"G{i}" for i in range(1, n_quantiles + 1)]
    top_column = f"G{n_quantiles}"
    competitor_columns = [f"G{i}" for i in range(1, n_quantiles)]
    missing = [column for column in group_columns if column not in group_returns]
    if missing:
        raise ValueError(f"Missing quantile-return columns: {missing}")

    rows: list[dict[str, float]] = []
    missed_rows: list[dict[str, float]] = []
    top = group_returns[top_column]
    full_window_count = len(group_returns) // window

    for window_number in range(full_window_count):
        start_position = window_number * window
        end_position = start_position + window - 1
        window_slice = slice(start_position, end_position + 1)
        window_frame = group_returns[group_columns].iloc[window_slice]
        window_top = top.iloc[window_slice]
        start_day = group_returns.index[start_position]
        end_day = group_returns.index[end_position]

        group_cumulative = {
            column: _compound_return(window_frame[column])
            for column in group_columns
        }
        top_cumulative = _compound_return(window_top)
        finite_group_cumulative = {
            column: value
            for column, value in group_cumulative.items()
            if column in competitor_columns and np.isfinite(value)
        }
        if finite_group_cumulative:
            best_group = max(
                finite_group_cumulative,
                key=lambda column: finite_group_cumulative[column],
            )
            best_group_number = int(best_group.removeprefix("G"))
            best_group_cumulative = finite_group_cumulative[best_group]
        else:
            best_group_number = np.nan
            best_group_cumulative = np.nan

        top_group_is_best = bool(
            np.isfinite(top_cumulative)
            and np.isfinite(best_group_cumulative)
            and top_cumulative >= best_group_cumulative
        )
        row = {
            "window_number": float(window_number + 1),
            "window_start_ordinal": float(pd.Timestamp(start_day).toordinal()),
            "window_end_ordinal": float(pd.Timestamp(end_day).toordinal()),
            "top_group_cumulative": top_cumulative,
            "best_group_number": float(best_group_number),
            "best_group_cumulative": best_group_cumulative,
            "top_group_excess_vs_best_group": top_cumulative - best_group_cumulative,
            "top_group_is_best": float(top_group_is_best),
        }
        for column in group_columns:
            row[f"{column.lower()}_cumulative"] = group_cumulative[column]
        rows.append({"day": end_day, **row})
        if not top_group_is_best:
            missed_rows.append({"day": start_day, **row})

    detail = (
        pd.DataFrame(rows).set_index("day")
        if rows
        else pd.DataFrame(index=pd.Index([], name="day"))
    )
    missed_detail = (
        pd.DataFrame(missed_rows).set_index("day")
        if missed_rows
        else pd.DataFrame(index=pd.Index([], name="day"))
    )
    top_group_best = (
        detail["top_group_is_best"].dropna()
        if len(detail)
        else pd.Series(dtype=float)
    )
    metrics = {
        "top_group_60d_window_count": int(len(top_group_best)),
        "top_group_60d_best_window_count": (
            int(top_group_best.sum()) if len(top_group_best) else 0
        ),
        "top_group_60d_best_window_ratio": (
            float(top_group_best.mean()) if len(top_group_best) else np.nan
        ),
        "top_group_60d_mean_cumulative": (
            float(detail["top_group_cumulative"].dropna().mean())
            if len(detail)
            else np.nan
        ),
        "top_group_60d_mean_excess_vs_best_group": (
            float(detail["top_group_excess_vs_best_group"].dropna().mean())
            if len(detail)
            else np.nan
        ),
    }
    return detail, missed_detail, metrics


def top_quantile_performance(
    group_returns: pd.DataFrame,
    *,
    n_quantiles: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare the latest top-quantile return stream with lower groups."""

    if n_quantiles < 3:
        raise ValueError("Top-quantile analysis requires at least three groups")
    group_columns = [f"G{i}" for i in range(1, n_quantiles + 1)]
    missing = [column for column in group_columns if column not in group_returns]
    if missing:
        raise ValueError(f"Missing quantile-return columns: {missing}")

    top_column = f"G{n_quantiles}"
    lower_columns = [f"G{i}" for i in range(1, n_quantiles)]
    top_return = group_returns[top_column]
    other_return = group_returns[lower_columns].mean(axis=1, skipna=True)
    best_lower_return = group_returns[lower_columns].max(axis=1, skipna=True)
    top_group_is_daily_best = top_return >= best_lower_return
    detail = pd.DataFrame(
        {
            "top_group_return": top_return,
            "other_groups_return": other_return,
            "best_lower_group_return": best_lower_return,
            "top_group_minus_other": top_return - other_return,
            "top_group_minus_best_lower": top_return - best_lower_return,
            "top_group_is_daily_best": top_group_is_daily_best.astype(float),
        },
        index=group_returns.index,
    ).dropna(how="all")

    if detail.empty:
        raise ValueError("No dates have enough observations for top-quantile analysis")
    detail["top_group_cumulative_return"] = (
        (1 + detail["top_group_return"].fillna(0)).cumprod() - 1
    )
    top_group_final_cumulative = float(
        detail["top_group_cumulative_return"].iloc[-1]
    )
    top_group_wealth = 1.0 + top_group_final_cumulative
    top_group_annualized_return = (
        float(
            top_group_wealth
            ** (TRADING_DAYS_PER_YEAR / len(detail))
            - 1.0
        )
        if top_group_wealth >= 0
        else np.nan
    )
    summary = {
        "top_group_mean_return": float(detail["top_group_return"].mean()),
        "top_group_final_cumulative": top_group_final_cumulative,
        "top_group_annualized_return": top_group_annualized_return,
        "other_groups_mean_return": float(detail["other_groups_return"].mean()),
        "top_group_outperformance_ratio": float(
            detail["top_group_is_daily_best"].dropna().mean()
        ),
    }
    return detail, summary


@evaluation_method("top_quantile_performance", requires=("quantile_returns",))
def evaluate_top_quantile_performance(state: EvaluationState) -> None:
    group_returns = state.require_detail("group_returns")
    if not isinstance(group_returns, pd.DataFrame):
        raise TypeError("Quantile-return detail must be a pandas DataFrame")
    n_quantiles = state.context.n_quantiles
    detail, metrics = top_quantile_performance(
        group_returns,
        n_quantiles=n_quantiles,
    )
    lower_columns = [f"G{i}" for i in range(1, n_quantiles)]
    top_column = f"G{n_quantiles}"
    top_group_mean = float(group_returns[top_column].mean())
    lower_group_means = group_returns[lower_columns].mean(axis=0)
    metrics.update(
        {
            "best_lower_group_mean_return": float(lower_group_means.max()),
            "top_group_above_every_lower_group": bool(
                top_group_mean > lower_group_means.max()
            ),
        }
    )
    leadership_detail, missed_detail, leadership_metrics = (
        non_overlapping_top_quantile_leadership(
            group_returns,
            n_quantiles=n_quantiles,
        )
    )
    metrics.update(leadership_metrics)
    state.add_detail("top_quantile_performance", detail)
    if len(leadership_detail):
        state.add_detail("top_quantile_leadership_60", leadership_detail)
    if len(missed_detail):
        state.add_detail("top_quantile_missed_windows_60", missed_detail)
    state.add_metrics(metrics)

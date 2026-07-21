"""Quantile grouping, equal-weight returns, and cumulative-return methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


@evaluation_method("quantile_returns")
def evaluate_quantile_returns(state: EvaluationState) -> None:
    """Calculate daily Q1..QN equal-weight returns."""

    n_quantiles = state.context.n_quantiles
    factor_values = state.factor.to_numpy(dtype=float, copy=False)
    return_values = state.context.forward_return.to_numpy(dtype=float, copy=False)
    dates: list[object] = []
    group_rows: list[list[float]] = []

    for row_number, day in enumerate(state.context.close.index):
        factor_row = factor_values[row_number]
        return_row = return_values[row_number]
        valid = np.isfinite(factor_row) & np.isfinite(return_row)
        factor_valid = factor_row[valid]
        return_valid = return_row[valid]
        count = len(factor_valid)
        if count < n_quantiles:
            continue

        ranks = pd.Series(factor_valid).rank(method="first").to_numpy()
        groups = np.floor((ranks - 1) * n_quantiles / count).astype(int)
        sums = np.bincount(groups, weights=return_valid, minlength=n_quantiles)
        counts = np.bincount(groups, minlength=n_quantiles)
        means = np.divide(
            sums,
            counts,
            out=np.full(n_quantiles, np.nan, dtype=float),
            where=counts > 0,
        )
        dates.append(day)
        group_rows.append(means.tolist())

    columns = [f"G{i}" for i in range(1, n_quantiles + 1)]
    group_returns = pd.DataFrame(
        group_rows,
        index=pd.Index(dates, name="day"),
        columns=columns,
    )
    if group_returns.empty:
        raise ValueError(
            "No dates have enough valid observations for quantile evaluation"
        )
    state.add_detail("group_returns", group_returns)


@evaluation_method(
    "quantile_cumulative",
    requires=("quantile_returns",),
)
def evaluate_quantile_cumulative(state: EvaluationState) -> None:
    """Compound daily quantile returns and record final portfolio metrics."""

    detail = state.require_detail("group_returns")
    if not isinstance(detail, pd.DataFrame):
        raise TypeError("Quantile-return detail must be a pandas DataFrame")
    cumulative = (1 + detail.fillna(0)).cumprod() - 1
    state.add_detail("cumulative_returns", cumulative)

    final = cumulative.iloc[-1]
    n_quantiles = state.context.n_quantiles
    state.add_metrics(
        {
            "g1_final_cumulative": float(final["G1"]),
            "gn_final_cumulative": float(final[f"G{n_quantiles}"]),
        }
    )


def assign_quantile(
    frame: pd.DataFrame,
    factor_name: str = "score",
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Compatibility helper assigning ascending 1..N daily groups."""

    if n_quantiles < 1 or int(n_quantiles) != n_quantiles:
        raise ValueError("n_quantiles must be a positive integer")
    n_quantiles = int(n_quantiles)
    result = frame.copy()

    def daily_group(values: pd.Series) -> pd.Series:
        groups = pd.Series(pd.NA, index=values.index, dtype="Int64")
        valid = values.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return groups
        ranks = valid.rank(method="first")
        assigned = np.floor((ranks - 1) * n_quantiles / len(valid)).astype(int) + 1
        groups.loc[valid.index] = assigned
        return groups

    result["group"] = result.groupby("day", sort=False, observed=True)[
        factor_name
    ].transform(daily_group)
    return result


def calc_group_returns(
    frame: pd.DataFrame,
    group_col: str = "group",
    ret_col: str = "f1",
    n_quantiles: int | None = None,
) -> pd.DataFrame:
    """Compatibility helper calculating long-form equal-weight returns."""

    grouped = (
        frame.dropna(subset=[group_col, ret_col])
        .groupby(["day", group_col], observed=True)[ret_col]
        .mean()
        .unstack(group_col)
        .sort_index()
    )
    grouped.columns = [f"G{int(column)}" for column in grouped.columns]
    if n_quantiles is not None:
        expected_columns = {f"G{group}" for group in range(1, n_quantiles + 1)}
        missing = sorted(expected_columns.difference(grouped.columns))
        if missing:
            raise ValueError(f"Missing quantile-return columns: {missing}")
    return grouped

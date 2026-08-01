"""Quantile net returns after conservative A-share transaction costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method
from .quantile_groups import quantile_membership


# Default to the strict A-share GP testing assumption: 12.5 bps to buy and
# 17.5 bps to sell.  A complete one-unit rebalance therefore costs 30 bps.
DEFAULT_BUY_COST_RATE = 0.00125
DEFAULT_SELL_COST_RATE = 0.00175


def default_buy_cost_rate() -> float:
    return DEFAULT_BUY_COST_RATE


def default_sell_cost_rate() -> float:
    return DEFAULT_SELL_COST_RATE


def calculate_quantile_net_returns(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    *,
    n_quantiles: int,
    buy_cost_rate: float = default_buy_cost_rate(),
    sell_cost_rate: float = default_sell_cost_rate(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build quantile weights, turnover, costs, and net equal-weight returns."""

    if n_quantiles < 1 or int(n_quantiles) != n_quantiles:
        raise ValueError("n_quantiles must be a positive integer")
    if buy_cost_rate < 0 or sell_cost_rate < 0:
        raise ValueError("Transaction cost rates must be non-negative")

    n_quantiles = int(n_quantiles)
    factor = factor.reindex(
        index=forward_return.index,
        columns=forward_return.columns,
    )
    factor_values = factor.to_numpy(dtype=float, copy=False)
    return_values = forward_return.to_numpy(dtype=float, copy=False)
    previous_weights: list[dict[int, float]] = [
        {} for _ in range(n_quantiles)
    ]

    dates: list[object] = []
    net_rows: list[list[float]] = []
    cost_rows: list[list[float]] = []
    turnover_rows: list[dict[str, float]] = []

    for row_number, day in enumerate(factor.index):
        factor_row = factor_values[row_number]
        return_row = return_values[row_number]
        membership = quantile_membership(
            factor_row,
            return_row,
            n_quantiles=n_quantiles,
        )
        if membership is None:
            continue
        valid_positions, _ranks, groups = membership
        count = len(valid_positions)
        gross_sums = np.bincount(
            groups,
            weights=return_row[valid_positions],
            minlength=n_quantiles,
        )
        group_counts = np.bincount(groups, minlength=n_quantiles)
        gross_means = np.divide(
            gross_sums,
            group_counts,
            out=np.full(n_quantiles, np.nan, dtype=float),
            where=group_counts > 0,
        )

        net_values: list[float] = []
        cost_values: list[float] = []
        turnover_values: dict[str, float] = {}
        for group_number in range(n_quantiles):
            positions = valid_positions[groups == group_number]
            current_weights = {
                int(position): 1.0 / len(positions)
                for position in positions
            }
            previous = previous_weights[group_number]
            union = set(previous) | set(current_weights)
            buy_turnover = sum(
                max(current_weights.get(position, 0.0) - previous.get(position, 0.0), 0.0)
                for position in union
            )
            sell_turnover = sum(
                max(previous.get(position, 0.0) - current_weights.get(position, 0.0), 0.0)
                for position in union
            )
            one_way_turnover = 0.5 * (buy_turnover + sell_turnover)
            transaction_cost = (
                buy_turnover * buy_cost_rate
                + sell_turnover * sell_cost_rate
            )
            previous_weights[group_number] = current_weights

            column = f"G{group_number + 1}"
            turnover_values[f"{column}_buy_turnover"] = float(buy_turnover)
            turnover_values[f"{column}_sell_turnover"] = float(sell_turnover)
            turnover_values[f"{column}_one_way_turnover"] = float(one_way_turnover)
            cost_values.append(float(transaction_cost))
            net_values.append(float(gross_means[group_number] - transaction_cost))

        dates.append(day)
        net_rows.append(net_values)
        cost_rows.append(cost_values)
        turnover_rows.append(turnover_values)

    columns = [f"G{i}" for i in range(1, n_quantiles + 1)]
    index = pd.Index(dates, name="day")
    net_returns = pd.DataFrame(net_rows, index=index, columns=columns)
    transaction_costs = pd.DataFrame(cost_rows, index=index, columns=columns)
    turnover = pd.DataFrame(turnover_rows, index=index)
    if net_returns.empty:
        raise ValueError(
            "No dates have enough valid observations for net quantile evaluation"
        )

    top_column = f"G{n_quantiles}"
    metrics = {
        "gn_mean_daily_buy_turnover": float(
            turnover[f"{top_column}_buy_turnover"].mean()
        ),
        "gn_mean_daily_sell_turnover": float(
            turnover[f"{top_column}_sell_turnover"].mean()
        ),
        "gn_mean_daily_one_way_turnover": float(
            turnover[f"{top_column}_one_way_turnover"].mean()
        ),
        "gn_mean_daily_transaction_cost": float(
            transaction_costs[top_column].mean()
        ),
        "gn_mean_daily_net_return": float(net_returns[top_column].mean()),
    }
    return net_returns, turnover, transaction_costs, metrics


@evaluation_method("quantile_net_returns", provides=("quantile_returns",))
def evaluate_quantile_net_returns(state: EvaluationState) -> None:
    """Calculate Q1..QN net returns after fixed single-side transaction costs."""

    net_returns, turnover, transaction_costs, metrics = (
        calculate_quantile_net_returns(
            state.factor,
            state.context.forward_return,
            n_quantiles=state.context.n_quantiles,
        )
    )
    state.add_detail("group_returns", net_returns)
    state.add_detail("quantile_turnover", turnover)
    state.add_detail("quantile_transaction_cost", transaction_costs)
    state.add_metrics(metrics)

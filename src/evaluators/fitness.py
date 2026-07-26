"""Natural-year portfolio fitness after explicit transaction costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method


TRADING_DAYS_PER_YEAR = 252
TURNOVER_FLOOR = 0.125


def _finite(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_yearly_fitness(
    group_returns: pd.DataFrame,
    quantile_turnover: pd.DataFrame,
    *,
    n_quantiles: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate the highest group's annual turnover/drawdown fitness.

    A row belongs to the calendar year of its factor signal date. The annual
    return uses all finite net returns inside that natural year and is
    geometrically annualized with 252 trading days, including a partial first
    or last year. Turnover is the mean daily one-way turnover of that same
    portfolio. Maximum drawdown is calculated from the year's net-value path
    and stored as a positive loss magnitude. The radicand follows the selected
    formula exactly: ``sharpe * sqrt(abs(R) / max(turnover, 0.125)) -
    lambda * max_drawdown``, with ``lambda = 1 / (1 - abs(max_drawdown)) ** 2``.
    """

    if n_quantiles < 1 or int(n_quantiles) != n_quantiles:
        raise ValueError("n_quantiles must be a positive integer")
    if not isinstance(group_returns.index, pd.DatetimeIndex):
        raise TypeError("Fitness requires a DatetimeIndex of signal dates")
    if not isinstance(quantile_turnover.index, pd.DatetimeIndex):
        raise TypeError("Fitness turnover requires a DatetimeIndex of signal dates")

    n_quantiles = int(n_quantiles)
    top_group = f"G{n_quantiles}"
    turnover_column = f"{top_group}_one_way_turnover"
    if top_group not in group_returns:
        raise ValueError(f"Fitness is missing net-return column {top_group!r}")
    if turnover_column not in quantile_turnover:
        raise ValueError(f"Fitness is missing turnover column {turnover_column!r}")

    observations = pd.concat(
        (
            group_returns[top_group].rename("net_return"),
            quantile_turnover[turnover_column].rename("one_way_turnover"),
        ),
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if observations.empty:
        raise ValueError("Fitness requires at least one aligned net-return and turnover day")

    rows: list[dict[str, float]] = []
    for year, values in observations.groupby(observations.index.year, sort=True):
        returns = _finite(values["net_return"])
        turnover = _finite(values.loc[returns.index, "one_way_turnover"])
        trading_days = len(returns)
        if not trading_days or len(turnover) != trading_days:
            continue

        cumulative_return = float((1.0 + returns).prod() - 1.0)
        wealth = 1.0 + cumulative_return
        annualized_return = (
            float(wealth ** (TRADING_DAYS_PER_YEAR / trading_days) - 1.0)
            if wealth > 0
            else np.nan
        )
        standard_deviation = float(returns.std(ddof=1)) if trading_days >= 2 else np.nan
        annualized_sharpe = (
            float(returns.mean() / standard_deviation * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else np.nan
        )
        net_value = (1.0 + returns).cumprod()
        running_peak = net_value.cummax().clip(lower=1.0)
        annual_max_drawdown = float((1.0 - net_value / running_peak).max())
        mean_turnover = float(turnover.mean())
        drawdown_penalty_lambda = (
            float(1.0 / (1.0 - abs(annual_max_drawdown)) ** 2)
            if np.isfinite(annual_max_drawdown) and abs(annual_max_drawdown) < 1.0
            else np.nan
        )
        fitness_radicand = (
            float(
                abs(annualized_return) / max(mean_turnover, TURNOVER_FLOOR)
            )
            if np.isfinite(annualized_return)
            and np.isfinite(mean_turnover)
            else np.nan
        )
        fitness = (
            float(
                annualized_sharpe * np.sqrt(fitness_radicand)
                - drawdown_penalty_lambda * annual_max_drawdown
            )
            if np.isfinite(annualized_sharpe)
            and np.isfinite(fitness_radicand)
            and np.isfinite(drawdown_penalty_lambda)
            else np.nan
        )
        rows.append(
            {
                "year": float(year),
                "trading_days": float(trading_days),
                "cumulative_net_return": cumulative_return,
                "annualized_net_return": annualized_return,
                "annualized_sharpe": annualized_sharpe,
                "mean_daily_one_way_turnover": mean_turnover,
                "annual_max_drawdown": annual_max_drawdown,
                "drawdown_penalty_lambda": drawdown_penalty_lambda,
                "fitness_radicand": fitness_radicand,
                "fitness": fitness,
            }
        )

    if not rows:
        raise ValueError("Fitness has no natural-year observations")
    detail = pd.DataFrame(rows).set_index("year")
    detail.index = detail.index.astype(int)
    detail.index.name = "year"
    valid_fitness = _finite(detail["fitness"])
    metrics = {
        "fitness": float(valid_fitness.mean()) if len(valid_fitness) else np.nan,
        "fitness_year_count": int(len(detail)),
        "fitness_valid_year_count": int(len(valid_fitness)),
    }
    return detail, metrics


@evaluation_method("fitness", requires=("quantile_net_returns",))
def evaluate_fitness(state: EvaluationState) -> None:
    """Publish natural-year highest-quantile fitness and its equal-weight mean."""

    group_returns = state.require_detail("group_returns")
    quantile_turnover = state.require_detail("quantile_turnover")
    if not isinstance(group_returns, pd.DataFrame):
        raise TypeError("Fitness requires quantile group returns as a DataFrame")
    if not isinstance(quantile_turnover, pd.DataFrame):
        raise TypeError("Fitness requires quantile turnover as a DataFrame")
    detail, metrics = calculate_yearly_fitness(
        group_returns,
        quantile_turnover,
        n_quantiles=state.context.n_quantiles,
    )
    state.add_detail("yearly_fitness", detail)
    state.add_metrics(metrics)

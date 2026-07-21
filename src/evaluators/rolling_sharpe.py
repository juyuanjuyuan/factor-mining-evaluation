"""Rolling annualized Sharpe-ratio evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .base import EvaluationState, evaluation_method, versioned_key
from .rolling_windows import DEFAULT_WINDOWS, complete_non_overlapping_windows


def rolling_sharpe(
    returns: pd.DataFrame,
    *,
    windows: Iterable[int] = DEFAULT_WINDOWS,
    annualization: int = 252,
) -> dict[int, pd.DataFrame]:
    """Calculate annualized Sharpe ratios for non-overlapping full windows."""

    results: dict[int, pd.DataFrame] = {}
    for window in windows:
        if window < 2:
            raise ValueError("Rolling Sharpe windows must be at least two")
        rows: list[pd.Series] = []
        end_days: list[object] = []
        for end_day, window_returns in complete_non_overlapping_windows(
            returns,
            window=window,
        ):
            standard_deviation = window_returns.std(ddof=1).replace(0, np.nan)
            rows.append(
                window_returns.mean().div(standard_deviation) * np.sqrt(annualization)
            )
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


def sharpe_distribution_summary(series: pd.Series) -> dict[str, float]:
    """Summarize non-overlapping Sharpe observations without a last-value bias."""

    finite = _finite(series)
    if not len(finite):
        return {"pos_share": np.nan, "min": np.nan, "median": np.nan}
    return {
        "pos_share": float((finite > 0).mean()),
        "min": float(finite.min()),
        "median": float(finite.median()),
    }


def _plot_non_overlapping_sharpe(
    sharpe: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Write a line chart of full-window annualized Sharpe observations."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 7))
    plotted_columns = [
        column for column in sharpe.columns if sharpe[column].notna().any()
    ]
    colors = plt.cm.RdYlGn(
        np.linspace(0.15, 0.9, max(len(plotted_columns), 1))
    )[::-1]
    for position, column in enumerate(plotted_columns):
        is_long_short = column == "Long-Short"
        color = "#111827" if is_long_short else colors[position]
        axis.plot(
            sharpe.index,
            sharpe[column],
            label=column,
            color=color,
            linewidth=2.6 if is_long_short else 1.5,
            alpha=0.95 if is_long_short else 0.8,
            marker="o",
            markersize=3.5,
        )
    axis.axhline(0, color="gray", linestyle="--", alpha=0.6)
    axis.set_xlabel("60-trading-day window end date (each point: [t-59, t])")
    axis.set_ylabel("Annualized Sharpe ratio")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if plotted_columns:
        axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
    else:
        axis.text(
            0.5,
            0.5,
            "No complete 60-trading-day windows",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="gray",
        )
    figure.autofmt_xdate()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _next_60d_plot_path(state: EvaluationState) -> tuple[str, Path]:
    """Return a non-overwriting artifact key and filename for this evaluator."""

    artifact_name = "rolling_sharpe_60_plot"
    version = 1
    while versioned_key(artifact_name, version) in state.artifacts:
        version += 1
    suffix = "" if version == 1 else f"__{version}"
    return (
        artifact_name,
        state.context.output_dir
        / "plots"
        / f"{state.context.artifact_name}__rolling_sharpe_60{suffix}.png",
    )


@evaluation_method("rolling_sharpe", requires=("quantile_returns",))
def evaluate_rolling_sharpe(state: EvaluationState) -> None:
    """Persist non-overlapping Sharpe details, summaries, and the 60-day chart."""

    detail = state.require_detail("group_returns")
    if not isinstance(detail, pd.DataFrame):
        raise TypeError("Quantile-return detail must be a pandas DataFrame")
    n_quantiles = state.context.n_quantiles
    metrics: dict[str, float] = {}
    for window, values in rolling_sharpe(detail).items():
        state.add_detail(f"rolling_sharpe_{window}", values)
        if window == 60:
            artifact_name, output_path = _next_60d_plot_path(state)
            _plot_non_overlapping_sharpe(
                values,
                (
                    f"{state.context.factor_name} 60-day non-overlapping Sharpe\n"
                    "each point uses [t-59, t], including the labeled end date"
                ),
                output_path,
            )
            state.add_artifact(artifact_name, output_path)
        summary = sharpe_distribution_summary(values[f"G{n_quantiles}"])
        for statistic, value in summary.items():
            metrics[f"gn_rolling_sharpe_{window}_{statistic}"] = value
    state.add_metrics(metrics)
